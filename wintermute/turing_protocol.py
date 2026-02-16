"""
TURING PROTOCOL — Three-Stage Post-Inference Validation Framework

Universal pipeline that detects, validates, and corrects violations in
assistant responses.  Fires after each inference round.

Three-Stage Design
------------------
  **Stage 1 (Detection):** A single LLM call analyses the assistant's
  response against ALL enabled violation detectors at once.  The detectors'
  ``detection_prompt`` fields are assembled into a universal system prompt.

  **Stage 2 (Validation):** Per-violation dispatch.  Each flagged violation
  is validated by its hook's ``validator_type``:
    - ``"programmatic"``: calls a registered Python function (fast, no LLM).
    - ``"llm"``: runs an LLM-based validation (future use).
  False positives are eliminated here.

  **Stage 3 (Correction):** All confirmed violations are aggregated into a
  single correction prompt injected into the conversation thread.

Hook configuration
------------------
  Hook definitions are loaded from ``data/TURING_PROTOCOL_HOOKS.txt``
  (JSON array).  If the file is absent or malformed, built-in defaults
  are used.  File entries override built-ins by hook name.

Hook behavior flags
-------------------
  - ``halt_inference`` (default False): When True, the inference thread
    pauses BEFORE broadcasting the reply.  When False (current
    ``workflow_spawn`` behavior), validation runs async after broadcast.
  - ``kill_on_detect`` (default False): When True, a confirmed violation
    causes the response to be discarded entirely (not broadcast, not
    saved).  The correction prompt is still injected.
"""

from __future__ import annotations

import copy
import json
import logging
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wintermute.llm_thread import BackendPool

from wintermute import database

from wintermute.tools import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

HOOKS_FILE = Path("data") / "TURING_PROTOCOL_HOOKS.txt"


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass
class TuringHook:
    name: str                              # e.g. "workflow_spawn"
    detection_prompt: str                  # Stage 1: bullet added to universal prompt
    validator_type: str                    # "programmatic" or "llm"
    validator_fn_name: Optional[str]       # Stage 2 programmatic: registered function name
    validator_prompt: Optional[str]        # Stage 2 LLM: system prompt (future use)
    correction_template: str              # Stage 3: first violation
    correction_template_repeat: str       # Stage 3: repeated violations
    halt_inference: bool = False           # Stage 2: block inference thread until validation completes
    kill_on_detect: bool = False           # Stage 2: if confirmed, discard the response entirely
    enabled: bool = True


@dataclass
class TuringResult:
    correction: Optional[str]              # Aggregated correction prompt (None = all clear)
    confirmed_violations: list[dict] = field(default_factory=list)  # [{type, reason, halt, kill}, ...]
    has_halt_violations: bool = False      # Any confirmed violation with halt_inference=True
    has_kill_violations: bool = False      # Any confirmed violation with kill_on_detect=True


# ------------------------------------------------------------------
# Built-in hook definitions
# ------------------------------------------------------------------

_BUILTIN_HOOKS: list[TuringHook] = [
    TuringHook(
        name="workflow_spawn",
        detection_prompt=(
            '- **workflow_spawn**: The assistant\'s text claims -- in any tense or '
            'phrasing -- that a session, workflow, or background task was '
            'started/spawned/launched during this response, AND '
            '"spawn_sub_session" is NOT in tool_calls_made. Do NOT flag pure '
            'intentions ("I\'ll start..."). Do NOT flag references to sessions '
            'in active_sessions.'
        ),
        validator_type="programmatic",
        validator_fn_name="validate_workflow_spawn",
        validator_prompt=None,
        correction_template=(
            "[TURING PROTOCOL CORRECTION] Your previous response indicated that "
            "a background session or workflow was started, but no "
            "spawn_sub_session tool call was actually made. "
            "Detected issue: {reason}\n\n"
            "You MUST now either:\n"
            "  1. Actually call spawn_sub_session to start the task, OR\n"
            "  2. Send a corrected message acknowledging the session was not "
            "started.\n\n"
            "Do NOT repeat the claim that the session is already running.\n\n"
            "spawn_sub_session tool schema:\n"
            "```json\n{tool_schema}\n```"
        ),
        correction_template_repeat=(
            "[TURING PROTOCOL CORRECTION -- REPEATED VIOLATION] You have AGAIN "
            "claimed to start a workflow without calling spawn_sub_session. "
            "This is the {ordinal} consecutive violation.\n\n"
            "Detected issue: {reason}\n\n"
            "You MUST apologise, state NO background task is running, and ask "
            "the user if they want you to attempt it now.\n\n"
            "spawn_sub_session tool schema:\n"
            "```json\n{tool_schema}\n```"
        ),
        halt_inference=False,
        kill_on_detect=False,
    ),
    TuringHook(
        name="phantom_tool_result",
        detection_prompt=(
            '- **phantom_tool_result**: The assistant\'s text presents specific data '
            '(file contents, search results, URL/webpage content, command output, '
            'directory listings) as if freshly obtained from a tool during THIS '
            'exchange, AND the corresponding tool (`read_file`, `search_web`, '
            '`fetch_url`, `execute_shell`, `list_reminders`) is NOT in '
            'tool_calls_made. Do NOT flag general knowledge, reasoning from '
            'context, references to information the user provided, or information '
            'from earlier in the conversation history.'
        ),
        validator_type="programmatic",
        validator_fn_name="validate_phantom_tool_result",
        validator_prompt=None,
        correction_template=(
            "[TURING PROTOCOL CORRECTION] Your previous response presented data "
            "as if obtained from a tool call, but no such tool was actually "
            "invoked. Detected issue: {reason}\n\n"
            "You MUST now either:\n"
            "  1. Actually call the appropriate tool to obtain the data, OR\n"
            "  2. Clearly state that you do not have that information and "
            "cannot verify it without calling the tool.\n\n"
            "Do NOT fabricate or assume tool output."
        ),
        correction_template_repeat=(
            "[TURING PROTOCOL CORRECTION -- REPEATED VIOLATION] You have AGAIN "
            "presented fabricated tool output without calling a tool. "
            "This is the {ordinal} consecutive violation.\n\n"
            "Detected issue: {reason}\n\n"
            "You MUST apologise, state clearly what information you actually "
            "have access to, and ask the user if they want you to call the "
            "relevant tool to obtain real data."
        ),
        halt_inference=False,
        kill_on_detect=False,
    ),
]


# ------------------------------------------------------------------
# Hook loading
# ------------------------------------------------------------------

def _load_hooks(enabled_validators: Optional[dict[str, bool]] = None) -> list[TuringHook]:
    """Load hooks from TURING_PROTOCOL_HOOKS.txt, merged with built-in defaults.

    Returns a list of enabled TuringHook objects.
    ``enabled_validators`` overrides each hook's enabled flag by name.
    """
    builtin_map = {h.name: copy.copy(h) for h in _BUILTIN_HOOKS}

    # Try loading from file.
    file_hooks: dict[str, TuringHook] = {}
    try:
        raw = HOOKS_FILE.read_text(encoding="utf-8").strip()
        if raw:
            entries = json.loads(raw)
            if not isinstance(entries, list):
                logger.error(
                    "%s must contain a JSON array; using built-in defaults",
                    HOOKS_FILE,
                )
            else:
                for entry in entries:
                    name = entry.get("name")
                    if not name:
                        logger.warning("Skipping hook entry without 'name' in %s", HOOKS_FILE)
                        continue
                    file_hooks[name] = TuringHook(
                        name=name,
                        detection_prompt=entry.get("detection_prompt", ""),
                        validator_type=entry.get("validator_type", "programmatic"),
                        validator_fn_name=entry.get("validator_fn_name"),
                        validator_prompt=entry.get("validator_prompt"),
                        correction_template=entry.get("correction_template", ""),
                        correction_template_repeat=entry.get("correction_template_repeat", ""),
                        halt_inference=entry.get("halt_inference", False),
                        kill_on_detect=entry.get("kill_on_detect", False),
                    )
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Cannot parse %s (%s); using built-in defaults", HOOKS_FILE, exc)
    except OSError as exc:
        logger.error("Cannot read %s (%s); using built-in defaults", HOOKS_FILE, exc)

    # Merge: file overrides built-ins by name.
    merged = {**builtin_map, **file_hooks}

    # Apply enabled_validators overrides.
    if enabled_validators:
        for name, enabled in enabled_validators.items():
            if name in merged:
                merged[name].enabled = enabled

    return [h for h in merged.values() if h.enabled]


# ------------------------------------------------------------------
# Stage 1: Detection — Universal LLM call
# ------------------------------------------------------------------

_STAGE1_PREAMBLE = """\
You are a post-inference validation agent.  You receive a snapshot of one
exchange between a user and an AI assistant, together with ground-truth
metadata about what actually happened during that exchange.

Your task: check for ALL of the following violation types in the assistant's
response.  For each type, a bullet describes the detection rule.

Violation types to check:
{detection_bullets}

You receive a JSON object with:
  - tool_calls_made:     list of tool function names actually invoked
  - user_message:        the user's message
  - assistant_response:  the assistant's reply
  - active_sessions:     currently running/pending background sessions

Respond with EXACTLY one JSON object (no markdown, no explanation).
Do NOT use any internal thinking or reasoning — reply directly.

{{
  "violations": [
    {{"type": "<hook_name>", "reason": "quoted phrase or brief explanation"}},
    ...
  ]
}}

If no violations are found, return: {{"violations": []}}
"""


def _build_stage1_system_prompt(hooks: list[TuringHook]) -> str:
    """Assemble the Stage 1 system prompt from all enabled hooks' detection_prompt fields."""
    bullets = "\n".join(h.detection_prompt for h in hooks)
    return _STAGE1_PREAMBLE.format(detection_bullets=bullets)


# ------------------------------------------------------------------
# Stage 2: Validation — Per-hook dispatch
# ------------------------------------------------------------------

def validate_workflow_spawn(context: dict, detection_result: dict) -> bool:
    """Programmatic validator for workflow_spawn.

    Returns True if the violation is confirmed (spawn_sub_session NOT in
    tool_calls_made), False if it's a false positive.
    """
    tool_calls_made = context.get("tool_calls_made", [])
    if "spawn_sub_session" in tool_calls_made:
        logger.info(
            "Stage 2: False positive — spawn_sub_session was actually called. "
            "LLM reason: %s", detection_result.get("reason", "?"),
        )
        return False
    logger.warning("Stage 2: Confirmed hallucination — spawn_sub_session NOT in tool_calls_made")
    return True


def validate_phantom_tool_result(context: dict, detection_result: dict) -> bool:
    """Programmatic validator for phantom_tool_result.

    Returns True if the violation is confirmed — the assistant claimed to
    present output from a tool but no tools were actually called this turn.
    If any tool was called, the claim may be grounded and we treat it as a
    false positive (the LLM detection already verified the specific claim).
    """
    tool_calls_made = context.get("tool_calls_made", [])
    if tool_calls_made:
        logger.info(
            "Stage 2: False positive — tools were called (%s). "
            "LLM reason: %s", tool_calls_made, detection_result.get("reason", "?"),
        )
        return False
    logger.warning("Stage 2: Confirmed phantom tool result — no tools called but output was claimed")
    return True


# Registry of programmatic validator functions.
_PROGRAMMATIC_VALIDATORS = {
    "validate_workflow_spawn": validate_workflow_spawn,
    "validate_phantom_tool_result": validate_phantom_tool_result,
}


# ------------------------------------------------------------------
# Stage 3: Correction — Aggregate confirmed violations
# ------------------------------------------------------------------

def _build_correction(confirmed: list[dict], hooks_by_name: dict[str, TuringHook],
                      correction_depth: int) -> str:
    """Build an aggregated correction prompt from all confirmed violations."""
    parts = []
    tool_schema = _get_spawn_tool_schema()

    for violation in confirmed:
        hook = hooks_by_name.get(violation["type"])
        if not hook:
            continue

        reason = violation.get("reason", "unknown violation")
        if correction_depth == 0:
            template = hook.correction_template
        else:
            template = hook.correction_template_repeat

        part = template.format(
            reason=reason,
            ordinal=_ordinal(correction_depth + 1),
            tool_schema=tool_schema,
        )
        parts.append(part)

    return "\n\n".join(parts)


# ------------------------------------------------------------------
# Helpers (preserved from supervisor.py)
# ------------------------------------------------------------------

def _truncate_middle(text: str, keep_head: int, keep_tail: int) -> str:
    """Keep the first *keep_head* and last *keep_tail* characters of *text*.

    If *text* is short enough to fit entirely, it is returned unchanged.
    Otherwise a ``[... N chars omitted ...]`` marker is inserted in the middle.
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
    """Return the English ordinal string for *n* (e.g. 1 -> '1st')."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

async def run_turing_protocol(
    pool: "BackendPool",
    user_message: str,
    assistant_response: str,
    tool_calls_made: list[str],
    active_sessions: list[dict],
    correction_depth: int = 0,
    enabled_validators: Optional[dict[str, bool]] = None,
    thread_id: str = "unknown",
) -> TuringResult:
    """Run the three-stage Turing Protocol validation pipeline.

    Returns a ``TuringResult`` with aggregated correction and metadata.

    Parameters
    ----------
    pool : BackendPool
        Backend pool for the protocol's own LLM calls (handles failover).
    user_message : str
        The user's most recent message.
    assistant_response : str
        Wintermute's reply to evaluate.
    tool_calls_made : list[str]
        Tool function names actually invoked during this inference round.
    active_sessions : list[dict]
        Output of SubSessionManager.list_active().
    correction_depth : int
        How many consecutive corrections have already been issued.
    enabled_validators : dict[str, bool] or None
        Per-hook enable/disable overrides from config.
    """
    hooks = _load_hooks(enabled_validators)
    if not hooks:
        return TuringResult(correction=None)

    hooks_by_name = {h.name: h for h in hooks}

    # -- Stage 1: Detection (universal LLM call) --------------------------
    logger.debug("Stage 1: Running LLM analysis (tool_calls_made=%s, hooks=%s)",
                 tool_calls_made, [h.name for h in hooks])

    system_prompt = _build_stage1_system_prompt(hooks)

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
                "Turing Protocol Stage 1 returned empty content "
                "(model may have spent all tokens on reasoning/thinking)"
            )
            return TuringResult(correction=None)

        # Strip markdown code fences if the model wraps its JSON.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw)
        violations = result.get("violations", [])

        try:
            log_status = "ok" if not violations else "violation_detected"
            database.save_interaction_log(
                _time.time(), "turing_protocol", thread_id,
                pool.last_used,
                json.dumps(context)[:2000], raw[:2000], log_status,
            )
        except Exception:
            pass

        if not violations:
            logger.debug("Stage 1: No violations detected")
            return TuringResult(correction=None)

        logger.warning("Stage 1: Detected %d violation(s): %s",
                        len(violations), [v.get("type") for v in violations])

        # -- Stage 2: Validation (per-hook dispatch) -----------------------
        confirmed: list[dict] = []

        for violation in violations:
            vtype = violation.get("type", "")
            reason = violation.get("reason", "")
            hook = hooks_by_name.get(vtype)

            if not hook:
                logger.warning("Stage 2: Unknown violation type %r — skipping", vtype)
                continue

            if hook.validator_type == "programmatic":
                fn = _PROGRAMMATIC_VALIDATORS.get(hook.validator_fn_name or "")
                if not fn:
                    logger.error(
                        "Stage 2: No programmatic validator %r registered for hook %r",
                        hook.validator_fn_name, hook.name,
                    )
                    continue
                is_confirmed = fn(context, violation)
            elif hook.validator_type == "llm":
                # Future: LLM-based validation via pool
                logger.warning("Stage 2: LLM validator not yet implemented for hook %r — assuming confirmed", hook.name)
                is_confirmed = True
            else:
                logger.error("Stage 2: Unknown validator_type %r for hook %r", hook.validator_type, hook.name)
                continue

            if is_confirmed:
                confirmed.append({
                    "type": vtype,
                    "reason": reason,
                    "halt": hook.halt_inference,
                    "kill": hook.kill_on_detect,
                })

        if not confirmed:
            logger.debug("Stage 2: All violations were false positives")
            return TuringResult(correction=None)

        # -- Stage 3: Correction (aggregate all confirmed) -----------------
        logger.warning("Stage 3: %d confirmed violation(s) — building correction", len(confirmed))

        correction_text = _build_correction(confirmed, hooks_by_name, correction_depth)

        has_halt = any(v["halt"] for v in confirmed)
        has_kill = any(v["kill"] for v in confirmed)

        return TuringResult(
            correction=correction_text,
            confirmed_violations=confirmed,
            has_halt_violations=has_halt,
            has_kill_violations=has_kill,
        )

    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Turing Protocol returned unparseable response: %s", exc)
        return TuringResult(correction=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Turing Protocol check failed (non-fatal): %s", exc)
        return TuringResult(correction=None)


def get_hooks(enabled_validators: Optional[dict[str, bool]] = None) -> list[TuringHook]:
    """Return the list of enabled hooks (for pre-flight checks by callers)."""
    return _load_hooks(enabled_validators)
