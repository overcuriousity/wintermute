"""
Lightweight Supervisor Agent for Workflow Validation

One-shot post-inference check that detects when the main model claims to have
started a background workflow/session without actually calling
spawn_sub_session.  When a mismatch is found, a corrective system event is
enqueued so the main model can self-correct.

Design
------
  - Runs asynchronously *after* the main reply is broadcast (zero added
    latency on the happy path).
  - Uses a dedicated LLM provider (cheap/fast model recommended).
  - Receives hard evidence: the list of tool calls actually made during
    the inference round, plus the programmatic list of active sessions.
  - A ``_supervisor_correction`` flag on the queue item prevents the
    supervisor from re-checking its own correction events (loop guard).
"""

import json
import logging
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Supervisor system prompt
# ------------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """\
You are a minimal validation supervisor.  You receive a snapshot of one
exchange between a user and an AI assistant, together with ground-truth
metadata about what actually happened during that exchange.

Your ONLY task: decide whether the assistant's response **claims** it
started / spawned / launched a background task, session, or workflow,
when in reality NO spawn_sub_session tool call was made.

You receive a JSON object with:
  - user_message:        the user's message
  - assistant_response:  the assistant's reply
  - tool_calls_made:     list of tool function names actually invoked
  - active_sessions:     currently running/pending background sessions

Respond with EXACTLY one JSON object (no markdown, no explanation).
Do NOT use any internal thinking or reasoning — reply directly.
  { "hallucination_detected": true/false, "reason": "..." }

Rules:
  - "hallucination_detected": true  ONLY when the assistant explicitly
    states it has ALREADY started/spawned/launched a session AND
    "spawn_sub_session" is NOT in tool_calls_made.
  - Do NOT flag promises or intentions ("I'll start a session", "let me
    look into that") — only flag past-tense/present-tense claims of
    having started something.
  - Do NOT flag references to previously existing sessions visible in
    active_sessions.
  - Be conservative.  When in doubt, return false.
  - "reason" is required only when hallucination_detected is true.
"""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

async def check_workflow_consistency(
    client: AsyncOpenAI,
    model: str,
    user_message: str,
    assistant_response: str,
    tool_calls_made: list[str],
    active_sessions: list[dict],
    *,
    max_tokens: int = 150,
    reasoning: bool = False,
) -> Optional[str]:
    """Run a one-shot supervisor check.

    Returns a corrective prompt string if a hallucination is detected,
    or *None* if the response is consistent with reality.

    Parameters
    ----------
    client : AsyncOpenAI
        The LLM client to use for the supervisor check.
    model : str
        Model name for the supervisor call.
    user_message : str
        The user's most recent message (truncated for cost).
    assistant_response : str
        Wintermute's reply to evaluate.
    tool_calls_made : list[str]
        Tool function names actually invoked during this inference round.
    active_sessions : list[dict]
        Output of SubSessionManager.list_active().
    max_tokens : int
        Max tokens for the supervisor response.
    reasoning : bool
        Whether the supervisor model uses reasoning tokens.
    """
    # Quick exit: if spawn_sub_session WAS called, nothing to check.
    if "spawn_sub_session" in tool_calls_made:
        return None

    context = {
        "user_message": user_message[:500],
        "assistant_response": assistant_response[:1500],
        "tool_calls_made": tool_calls_made,
        "active_sessions": [
            {
                "id": s.get("session_id", "?"),
                "objective": (s.get("objective") or "")[:100],
                "status": s.get("status", "?"),
            }
            for s in active_sessions
        ],
    }

    try:
        token_kwarg = (
            {"max_completion_tokens": max_tokens}
            if reasoning
            else {"max_tokens": max_tokens}
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context, indent=2)},
            ],
            **token_kwarg,
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
                "Supervisor detected workflow hallucination: %s", reason,
            )
            return (
                "[SUPERVISOR CORRECTION] Your previous response indicated that "
                "a background session or workflow was started, but no "
                "spawn_sub_session tool call was actually made during that "
                f"response.  Detected issue: {reason}\n\n"
                "You MUST now either:\n"
                "  1. Actually call spawn_sub_session to start the task, OR\n"
                "  2. Send a corrected message to the user acknowledging that "
                "the session was not started and offer to start it now.\n\n"
                "Do NOT repeat the claim that the session is already running."
            )

        logger.debug("Supervisor check passed (no hallucination detected)")
        return None

    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Supervisor returned unparseable response: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supervisor check failed (non-fatal): %s", exc)
        return None
