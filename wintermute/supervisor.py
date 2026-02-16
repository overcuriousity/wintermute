"""
Lightweight Two-Stage Supervisor Agent for Workflow Validation

Post-inference check that detects when the main model claims to have
started a background workflow/session without actually calling
spawn_sub_session.  When a mismatch is found, a corrective system event is
enqueued so the main model can self-correct.

Two-Stage Design
----------------
  **Stage 1 (LLM analysis):** A supervisor LLM analyses the assistant's
  response to detect claims of having spawned a session/workflow.  This
  always runs — there is no early programmatic exit.

  **Stage 2 (programmatic validation):** Only fires when Stage 1 flags
  ``hallucination_detected=true``.  Cross-checks against ``tool_calls_made``
  and ``active_sessions``.  If ``spawn_sub_session`` IS in
  ``tool_calls_made``, this is a false positive — log and return None.
  Otherwise, inject the correction.

Other details
-------------
  - Runs asynchronously *after* the main reply is broadcast (zero added
    latency on the happy path).
  - Uses a dedicated LLM provider (cheap/fast model recommended).
  - A ``correction_depth`` counter on the queue item allows the supervisor
    to re-check responses to its own corrections up to MAX_CORRECTION_DEPTH
    times, with escalating prompt severity.

Prompt configuration
--------------------
  Supervisor prompts are loaded from ``data/SUPERVISOR_PROMPTS.txt``, which
  must be a JSON array of objects with ``"name"`` and ``"system_prompt"``
  keys.  If the file is absent or malformed the built-in defaults are used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wintermute.llm_thread import BackendPool

from wintermute.tools import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPTS_FILE = Path("data") / "SUPERVISOR_PROMPTS.txt"

# ------------------------------------------------------------------
# Built-in fallback prompt
# ------------------------------------------------------------------

_DEFAULT_WORKFLOW_SPAWN_PROMPT = """\
You are a minimal validation supervisor.  You receive a snapshot of one
exchange between a user and an AI assistant, together with ground-truth
metadata about what actually happened during that exchange.

Your ONLY task: decide whether the assistant's response **claims** it
started / spawned / launched a background task, session, or workflow,
when in reality NO spawn_sub_session tool call was made.

You receive a JSON object with:
  - tool_calls_made:     list of tool function names actually invoked  ← check this first
  - user_message:        the user's message
  - assistant_response:  the assistant's reply
  - active_sessions:     currently running/pending background sessions (with start times)

Respond with EXACTLY one JSON object (no markdown, no explanation).
Do NOT use any internal thinking or reasoning — reply directly.
  { "hallucination_detected": true/false, "reason": "..." }

Rules:
  - "hallucination_detected": true  when the assistant's text claims — in any
    tense or phrasing — that a session, workflow, or background task was
    started/spawned/launched during this response, AND "spawn_sub_session" is
    NOT in tool_calls_made.

  - Declarative claim examples that SHOULD be flagged (not exhaustive):
      "I've started a background session"
      "I've kicked off a workflow"
      "I've spawned a session"
      "I've launched the task"
      "spinning up a session"
      "I'm handling this in the background"
      "I've started working on this in the background"
      "a session is now running"
      "the workflow has been started"

  - Do NOT flag pure intentions or promises that have not yet been acted on:
      "I'll start a session", "let me kick that off", "I'm going to spawn…"

  - Do NOT flag references to previously existing sessions that are visible
    in active_sessions (compare session IDs and start times).

  - "reason" is required only when hallucination_detected is true; it should
    quote the specific phrase that triggered the flag.
"""


# ------------------------------------------------------------------
# Prompt loading
# ------------------------------------------------------------------

def _load_supervisor_prompts() -> dict[str, str]:
    """Load supervisor prompts from SUPERVISOR_PROMPTS.txt.

    Returns a dict mapping prompt name -> system_prompt string.
    Falls back to built-in defaults if the file is missing or malformed.
    """
    defaults = {"workflow_spawn": _DEFAULT_WORKFLOW_SPAWN_PROMPT}
    try:
        raw = SUPERVISOR_PROMPTS_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return defaults
        entries = json.loads(raw)
        if not isinstance(entries, list):
            logger.error(
                "%s must contain a JSON array; using built-in defaults",
                SUPERVISOR_PROMPTS_FILE,
            )
            return defaults
        result: dict[str, str] = {}
        for entry in entries:
            name = entry.get("name")
            prompt = entry.get("system_prompt")
            if name and prompt:
                result[name] = prompt
            else:
                logger.warning(
                    "Skipping malformed entry in %s: %s", SUPERVISOR_PROMPTS_FILE, entry
                )
        # Merge: file overrides defaults; missing entries fall back.
        return {**defaults, **result}
    except FileNotFoundError:
        return defaults
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error(
            "Cannot parse %s (%s); using built-in defaults", SUPERVISOR_PROMPTS_FILE, exc
        )
        return defaults
    except OSError as exc:
        logger.error(
            "Cannot read %s (%s); using built-in defaults", SUPERVISOR_PROMPTS_FILE, exc
        )
        return defaults


# ------------------------------------------------------------------
# Truncation helpers
# ------------------------------------------------------------------

def _truncate_middle(text: str, keep_head: int, keep_tail: int) -> str:
    """Keep the first *keep_head* and last *keep_tail* characters of *text*.

    If *text* is short enough to fit entirely, it is returned unchanged.
    Otherwise a ``[… N chars omitted …]`` marker is inserted in the middle.
    """
    total = keep_head + keep_tail
    if len(text) <= total:
        return text
    omitted = len(text) - total
    return text[:keep_head] + f"\n[… {omitted} chars omitted …]\n" + text[-keep_tail:]


def _get_spawn_tool_schema() -> str:
    """Return a compact JSON string of the spawn_sub_session tool schema."""
    for tool in TOOL_SCHEMAS:
        if tool.get("function", {}).get("name") == "spawn_sub_session":
            return json.dumps(tool, indent=2)
    return "(spawn_sub_session tool schema not found)"


def _ordinal(n: int) -> str:
    """Return the English ordinal string for *n* (e.g. 1 → '1st')."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

async def check_workflow_consistency(
    pool: "BackendPool",
    user_message: str,
    assistant_response: str,
    tool_calls_made: list[str],
    active_sessions: list[dict],
    correction_depth: int = 0,
) -> Optional[str]:
    """Run a one-shot supervisor check with automatic backend failover.

    Returns a corrective prompt string if a hallucination is detected,
    or *None* if the response is consistent with reality.

    Parameters
    ----------
    pool : BackendPool
        Backend pool for the supervisor role (handles failover).
    user_message : str
        The user's most recent message.  Truncated with keep-head/keep-tail
        so that log dumps in the middle don't bury the actual instructions.
    assistant_response : str
        Wintermute's reply to evaluate (passed in full — no truncation).
    tool_calls_made : list[str]
        Tool function names actually invoked during this inference round.
    active_sessions : list[dict]
        Output of SubSessionManager.list_active().
    """
    # Stage 1: Always run the LLM analysis — no early programmatic exit.
    logger.debug("Stage 1: Running LLM analysis (tool_calls_made=%s)", tool_calls_made)

    prompts = _load_supervisor_prompts()
    system_prompt = prompts["workflow_spawn"]

    # tool_calls_made first so it appears prominently in the JSON the
    # supervisor model reads.
    context = {
        "tool_calls_made": tool_calls_made,
        "user_message": _truncate_middle(user_message, keep_head=300, keep_tail=200),
        "assistant_response": assistant_response,
        "active_sessions": [
            {
                "id": s.get("session_id", "?"),
                "objective": (s.get("objective") or "")[:100],
                "status": s.get("status", "?"),
                "started_at": s.get("started_at"),
            }
            for s in active_sessions
        ],
    }

    try:
        response = await pool.call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, indent=2)},
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            logger.warning(
                "Supervisor returned empty content (model may have spent all tokens on reasoning/thinking)"
            )
            return None
        # Strip markdown code fences if the model wraps its JSON.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw)

        if result.get("hallucination_detected"):
            reason = result.get("reason", "workflow claim without actual spawn")
            logger.warning(
                "Stage 1: LLM detected workflow hallucination: %s", reason,
            )

            # Stage 2: Programmatic validation — cross-check against ground truth.
            if "spawn_sub_session" in tool_calls_made:
                logger.info(
                    "Stage 2: False positive — spawn_sub_session was actually called. "
                    "LLM reason: %s", reason,
                )
                return None
            logger.warning("Stage 2: Confirmed hallucination — spawn_sub_session NOT in tool_calls_made")

            tool_schema = _get_spawn_tool_schema()

            if correction_depth == 0:
                return (
                    "[SUPERVISOR CORRECTION] Your previous response indicated that "
                    "a background session or workflow was started, but no "
                    "spawn_sub_session tool call was actually made during that "
                    f"response.  Detected issue: {reason}\n\n"
                    "You MUST now either:\n"
                    "  1. Actually call spawn_sub_session to start the task, OR\n"
                    "  2. Send a corrected message to the user acknowledging that "
                    "the session was not started and offer to start it now.\n\n"
                    "Do NOT repeat the claim that the session is already running.\n\n"
                    "Here is the spawn_sub_session tool schema — use it if you "
                    "intend to start a background task:\n"
                    f"```json\n{tool_schema}\n```"
                )
            else:
                return (
                    "[SUPERVISOR CORRECTION — REPEATED VIOLATION] You have AGAIN "
                    "claimed to start a workflow or session without actually "
                    "calling spawn_sub_session.  This is the "
                    f"{_ordinal(correction_depth + 1)} consecutive violation.\n\n"
                    f"Detected issue: {reason}\n\n"
                    "You MUST send a message to the user that:\n"
                    "  - Apologises for the error\n"
                    "  - States clearly that NO background task is running\n"
                    "  - Asks the user if they would like you to attempt the "
                    "task now\n\n"
                    "Do NOT claim any session is running.  Do NOT fabricate "
                    "session IDs.  Do NOT use phrases like 'starting now' or "
                    "'spinning up' unless you are ACTUALLY making a "
                    "spawn_sub_session tool call in the SAME response.\n\n"
                    "Here is the spawn_sub_session tool schema — you MUST use "
                    "this exact tool call format if you want to start a task:\n"
                    f"```json\n{tool_schema}\n```"
                )

        logger.debug("Stage 1: LLM check passed (no hallucination detected)")
        return None

    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Supervisor returned unparseable response: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supervisor check failed (non-fatal): %s", exc)
        return None
