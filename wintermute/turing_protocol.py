"""
TURING PROTOCOL — Phase-Aware Validation Framework

Universal pipeline that detects, validates, and corrects violations in
assistant responses.  Hooks can fire at three phases of the inference
cycle and are scoped to run in the main thread, sub-sessions, or both.

Phases
------
  ``post_inference``  — After the LLM produces a response (text or tool
                        calls), before delivery / next action.
  ``pre_execution``   — After the LLM requests a tool call, before
                        ``execute_tool()`` runs.
  ``post_execution``  — After ``execute_tool()`` returns, before the
                        result is appended to history.

Three-Stage Design (per phase batch)
-------------------------------------
  **Stage 1 (Detection):** A single LLM call analyses the context against
  ALL enabled hooks for the current phase.  Hooks whose ``validator_type``
  is ``"programmatic"`` skip Stage 1 and go directly to Stage 2.

  **Stage 2 (Validation):** Per-violation dispatch.
    - ``"programmatic"``: calls a registered Python function (fast, no LLM).
    - ``"llm"``: runs a dedicated LLM call per hook.
  False positives are eliminated here.

  **Stage 3 (Correction):** All confirmed violations are aggregated into a
  single correction prompt injected into the conversation thread (main) or
  appended to the sub-session message list (sub-sessions).

Hook configuration
------------------
  Hook definitions are loaded from ``data/TURING_PROTOCOL_HOOKS.txt``
  (JSON array).  If the file is absent or malformed, built-in defaults
  are used.  File entries override built-ins by hook name.

Scope
-----
  Each hook declares a ``scope`` list: ``["main"]``, ``["sub_session"]``,
  or ``["main", "sub_session"]``.  Hooks without scope default to
  ``["main"]`` for backward compatibility.

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
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wintermute.llm_thread import BackendPool

from wintermute import database

from wintermute.tools import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

HOOKS_FILE = Path("data") / "TURING_PROTOCOL_HOOKS.txt"


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

VALID_PHASES = frozenset({"post_inference", "pre_execution", "post_execution"})
VALID_SCOPES = frozenset({"main", "sub_session"})


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
    phase: str = "post_inference"          # When to fire: post_inference | pre_execution | post_execution
    scope: list = field(default_factory=lambda: ["main"])  # Where: ["main"], ["sub_session"], or both


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
        phase="post_inference",
        scope=["main"],
    ),
    TuringHook(
        name="phantom_tool_result",
        detection_prompt=(
            '- **phantom_tool_result**: The assistant\'s text presents specific data '
            '(file contents, search results, URL/webpage content, command output, '
            'directory listings) as if **already obtained** from a tool during THIS '
            'exchange — i.e. it states results in past tense or presents concrete '
            'output — AND the corresponding tool (`read_file`, `search_web`, '
            '`fetch_url`, `execute_shell`, `list_reminders`) is NOT in '
            'tool_calls_made. Do NOT flag general knowledge, reasoning from '
            'context, references to information the user provided, or information '
            'from earlier in the conversation history. '
            'Do NOT flag future-tense action commitments ("I\'ll check…") — '
            'those are handled by empty_promise.'
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
        phase="post_inference",
        scope=["main"],
    ),
    TuringHook(
        name="empty_promise",
        detection_prompt=(
            '- **empty_promise**: The assistant\'s response commits to performing '
            'an action (in any language) — e.g. "I\'ll do X", "Let me check", '
            '"Ich werde das prüfen", "Je vais vérifier" — as its **final answer**, '
            'WITHOUT having made any tool call or spawn_sub_session call to '
            'actually execute that action. The response ends with the promise '
            'but no action was taken. tool_calls_made must be empty AND '
            '"spawn_sub_session" must NOT be in tool_calls_made. '
            'Do NOT flag responses where the assistant describes what it *did* '
            '(past tense) — those are handled by phantom_tool_result. '
            'Do NOT flag planning/explanation responses where the assistant '
            'outlines steps and asks for user confirmation before acting. '
            'Do NOT flag responses where the assistant explains it cannot '
            'perform the action.'
        ),
        validator_type="programmatic",
        validator_fn_name="validate_empty_promise",
        validator_prompt=None,
        correction_template=(
            "[TURING PROTOCOL CORRECTION] Your previous response committed to "
            "performing an action but you did not actually execute it — no tool "
            "was called. Detected issue: {reason}\n\n"
            "You MUST now either:\n"
            "  1. Actually call the appropriate tool to perform the action, OR\n"
            "  2. Explain clearly why you cannot perform it and ask the user "
            "how to proceed.\n\n"
            "Do NOT repeat the promise without acting."
        ),
        correction_template_repeat=(
            "[TURING PROTOCOL CORRECTION -- REPEATED VIOLATION] You have AGAIN "
            "promised to perform an action without actually calling any tool. "
            "This is the {ordinal} consecutive violation.\n\n"
            "Detected issue: {reason}\n\n"
            "You MUST apologise, state clearly that you failed to execute the "
            "action, and ask the user if they want you to attempt it now."
        ),
        halt_inference=False,
        kill_on_detect=False,
        phase="post_inference",
        scope=["main"],
    ),
    # -- objective_completion: sub-session exit gatekeeper --
    # Fires on the final text-only response of a sub-session.  Uses an LLM
    # call to evaluate whether the response actually satisfies the stated
    # objective.  If not, a correction is injected and the worker loop
    # continues instead of exiting.
    TuringHook(
        name="objective_completion",
        detection_prompt="",  # Not used — this hook has its own dedicated LLM call
        validator_type="llm",
        validator_fn_name=None,
        validator_prompt=None,  # Uses TURING_OBJECTIVE_COMPLETION.txt template
        correction_template=(
            "[TURING PROTOCOL — OBJECTIVE NOT MET] Your response does not "
            "satisfy the task objective.\n\n"
            "Objective: {objective}\n\n"
            "Issue: {reason}\n\n"
            "You MUST continue working toward the objective. Use the "
            "available tools to make progress. Do NOT produce a final "
            "summary until the objective is genuinely complete."
        ),
        correction_template_repeat=(
            "[TURING PROTOCOL — OBJECTIVE STILL NOT MET] This is the "
            "{ordinal} attempt. Your response still does not satisfy the "
            "objective.\n\n"
            "Objective: {objective}\n\n"
            "Issue: {reason}\n\n"
            "Focus on the specific gap identified above. If the objective "
            "is truly impossible to complete with available tools, explain "
            "exactly what is blocking you and what partial progress was made."
        ),
        halt_inference=False,
        kill_on_detect=False,
        phase="post_inference",
        scope=["sub_session"],
    ),
]


# ------------------------------------------------------------------
# Hook loading
# ------------------------------------------------------------------

def _load_hooks(
    enabled_validators: Optional[dict[str, Any]] = None,
    *,
    phase_filter: Optional[str] = None,
    scope_filter: Optional[str] = None,
) -> list[TuringHook]:
    """Load hooks from TURING_PROTOCOL_HOOKS.txt, merged with built-in defaults.

    Returns a list of enabled TuringHook objects, optionally filtered by
    ``phase_filter`` (e.g. ``"post_inference"``) and ``scope_filter``
    (e.g. ``"main"`` or ``"sub_session"``).

    ``enabled_validators`` overrides each hook's enabled flag by name.
    Values can be ``True``/``False`` (simple toggle) or a dict with keys
    like ``enabled``, ``scope``, ``phase`` for granular overrides.
    """
    builtin_map = {h.name: copy.deepcopy(h) for h in _BUILTIN_HOOKS}

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
                    raw_scope = entry.get("scope", ["main"])
                    if isinstance(raw_scope, str):
                        raw_scope = [raw_scope]
                    raw_phase = entry.get("phase", "post_inference")
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
                        phase=raw_phase,
                        scope=raw_scope,
                    )
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Cannot parse %s (%s); using built-in defaults", HOOKS_FILE, exc)
    except OSError as exc:
        logger.error("Cannot read %s (%s); using built-in defaults", HOOKS_FILE, exc)

    # Merge: file overrides built-ins by name.
    merged = {**builtin_map, **file_hooks}

    # Apply enabled_validators overrides (supports bool or dict values).
    if enabled_validators:
        for name, val in enabled_validators.items():
            if name not in merged:
                continue
            hook = merged[name]
            if isinstance(val, bool):
                hook.enabled = val
            elif isinstance(val, dict):
                if "enabled" in val:
                    hook.enabled = bool(val["enabled"])
                if "scope" in val:
                    s = val["scope"]
                    hook.scope = [s] if isinstance(s, str) else list(s)
                if "phase" in val:
                    hook.phase = str(val["phase"])

    # Filter by enabled, phase, and scope.
    result = []
    for h in merged.values():
        if not h.enabled:
            continue
        if phase_filter and h.phase != phase_filter:
            continue
        if scope_filter and scope_filter not in h.scope:
            continue
        result.append(h)

    return result


# ------------------------------------------------------------------
# Stage 1: Detection — Universal LLM call
# ------------------------------------------------------------------

def _build_stage1_system_prompt(hooks: list[TuringHook]) -> str:
    """Assemble the Stage 1 system prompt from all enabled hooks' detection_prompt fields."""
    from wintermute import prompt_loader
    bullets = "\n".join(h.detection_prompt for h in hooks)
    return prompt_loader.load("TURING_STAGE1.txt", detection_bullets=bullets)


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


def validate_empty_promise(context: dict, detection_result: dict) -> bool:
    """Programmatic validator for empty_promise.

    Returns True if the violation is confirmed — the assistant committed to
    an action but no tools were called at all this turn.  If any tool was
    called, the promise may have been fulfilled and we treat it as a false
    positive.
    """
    tool_calls_made = context.get("tool_calls_made", [])
    if tool_calls_made:
        logger.info(
            "Stage 2: False positive (empty_promise) — tools were called (%s). "
            "LLM reason: %s", tool_calls_made, detection_result.get("reason", "?"),
        )
        return False
    logger.warning("Stage 2: Confirmed empty_promise — action committed but no tools called")
    return True


# Registry of programmatic validator functions.
_PROGRAMMATIC_VALIDATORS = {
    "validate_workflow_spawn": validate_workflow_spawn,
    "validate_phantom_tool_result": validate_phantom_tool_result,
    "validate_empty_promise": validate_empty_promise,
}


# ------------------------------------------------------------------
# Stage 3: Correction — Aggregate confirmed violations
# ------------------------------------------------------------------

def _build_correction(confirmed: list[dict], hooks_by_name: dict[str, TuringHook],
                      correction_depth: int, *, objective: Optional[str] = None) -> str:
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
            objective=objective or "(unknown)",
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
    enabled_validators: Optional[dict[str, Any]] = None,
    thread_id: str = "unknown",
    *,
    phase: str = "post_inference",
    scope: str = "main",
    objective: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    tool_result: Optional[str] = None,
) -> TuringResult:
    """Run the three-stage Turing Protocol validation pipeline.

    Returns a ``TuringResult`` with aggregated correction and metadata.

    Parameters
    ----------
    pool : BackendPool
        Backend pool for the protocol's own LLM calls (handles failover).
    user_message : str
        The user's most recent message (or objective for sub-sessions).
    assistant_response : str
        Wintermute's reply to evaluate.
    tool_calls_made : list[str]
        Tool function names actually invoked during this inference round.
    active_sessions : list[dict]
        Output of SubSessionManager.list_active().
    correction_depth : int
        How many consecutive corrections have already been issued.
    enabled_validators : dict or None
        Per-hook enable/disable overrides from config.
    phase : str
        Which phase to run: ``post_inference``, ``pre_execution``,
        or ``post_execution``.
    scope : str
        Execution context: ``"main"`` or ``"sub_session"``.
    objective : str or None
        Sub-session objective (used by ``objective_completion``).
    tool_name : str or None
        Tool being called (for ``pre_execution`` phase).
    tool_args : dict or None
        Tool arguments (for ``pre_execution`` phase).
    tool_result : str or None
        Tool result (for ``post_execution`` phase).
    """
    hooks = _load_hooks(enabled_validators, phase_filter=phase, scope_filter=scope)
    if not hooks:
        return TuringResult(correction=None)

    hooks_by_name = {h.name: h for h in hooks}

    # Partition hooks: those needing a Stage 1 LLM detection call vs.
    # those that are purely programmatic (skip Stage 1) vs. those that
    # use a dedicated LLM call (e.g. objective_completion).
    stage1_hooks = [h for h in hooks if h.detection_prompt and h.validator_type == "programmatic"]
    dedicated_llm_hooks = [h for h in hooks if h.validator_type == "llm"]
    programmatic_only = [h for h in hooks if not h.detection_prompt and h.validator_type == "programmatic"]

    # Build shared context for all validators.
    context: dict[str, Any] = {
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
    # Phase-specific context enrichment.
    if objective:
        context["objective"] = objective
    if tool_name:
        context["tool_name"] = tool_name
    if tool_args is not None:
        context["tool_args"] = tool_args
    if tool_result is not None:
        context["tool_result"] = _truncate_middle(tool_result, keep_head=500, keep_tail=200)
    context["phase"] = phase
    context["scope"] = scope

    try:
        violations: list[dict] = []

        # -- Stage 1: Detection via universal LLM call (only for hooks with detection_prompt) --
        if stage1_hooks:
            logger.debug("Stage 1: Running LLM analysis (phase=%s, scope=%s, hooks=%s)",
                         phase, scope, [h.name for h in stage1_hooks])

            system_prompt = _build_stage1_system_prompt(stage1_hooks)

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
            else:
                # Strip markdown code fences if the model wraps its JSON.
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

                result = json.loads(raw)
                violations.extend(result.get("violations", []))

            try:
                log_status = "ok" if not violations else "violation_detected"
                database.save_interaction_log(
                    _time.time(), "turing_protocol", thread_id,
                    pool.last_used,
                    json.dumps(context)[:2000], (raw if stage1_hooks else "skipped")[:2000],
                    log_status,
                )
            except Exception:
                pass

        # -- Dedicated LLM hooks (each gets its own call) --
        for hook in dedicated_llm_hooks:
            try:
                llm_result = await _run_dedicated_llm_hook(pool, hook, context, thread_id)
                if llm_result:
                    violations.append(llm_result)
            except Exception:  # noqa: BLE001
                logger.exception("Dedicated LLM hook %r raised (non-fatal)", hook.name)

        # -- Programmatic-only hooks (no detection_prompt, run directly) --
        for hook in programmatic_only:
            fn = _PROGRAMMATIC_VALIDATORS.get(hook.validator_fn_name or "")
            if fn and fn(context, {}):
                violations.append({"type": hook.name, "reason": "programmatic check failed"})

        if not violations:
            logger.debug("Stage 1: No violations detected (phase=%s, scope=%s)", phase, scope)
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
                # Dedicated LLM hooks already validated during detection.
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

        # Log Stage 2 result
        try:
            stage2_status = "ok" if not confirmed else "violation_detected"
            stage2_output = json.dumps({
                "confirmed": confirmed,
                "false_positives": len(violations) - len(confirmed),
                "total_checked": len(violations),
            })
            database.save_interaction_log(
                _time.time(), "turing_stage2", thread_id,
                pool.last_used,
                json.dumps(violations)[:2000], stage2_output[:2000], stage2_status,
            )
        except Exception:
            pass

        if not confirmed:
            logger.debug("Stage 2: All violations were false positives")
            return TuringResult(correction=None)

        # -- Stage 3: Correction (aggregate all confirmed) -----------------
        logger.warning("Stage 3: %d confirmed violation(s) — building correction (phase=%s, scope=%s)",
                        len(confirmed), phase, scope)

        correction_text = _build_correction(
            confirmed, hooks_by_name, correction_depth, objective=objective,
        )

        # Log Stage 3 correction
        try:
            database.save_interaction_log(
                _time.time(), "turing_stage3", thread_id,
                pool.last_used,
                json.dumps(confirmed)[:2000], correction_text[:2000], "violation_detected",
            )
        except Exception:
            pass

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


# ------------------------------------------------------------------
# Dedicated LLM hook execution
# ------------------------------------------------------------------

async def _run_dedicated_llm_hook(
    pool: "BackendPool",
    hook: TuringHook,
    context: dict,
    thread_id: str,
) -> Optional[dict]:
    """Run a hook that uses its own LLM call for detection+validation.

    Returns a violation dict ``{"type": ..., "reason": ...}`` if the hook
    fires, or ``None`` if the check passes.
    """
    if hook.name == "objective_completion":
        return await _check_objective_completion(pool, hook, context, thread_id)

    # Generic fallback for future dedicated LLM hooks.
    logger.warning("No dedicated LLM handler for hook %r — skipping", hook.name)
    return None


async def _check_objective_completion(
    pool: "BackendPool",
    hook: TuringHook,
    context: dict,
    thread_id: str,
) -> Optional[dict]:
    """Evaluate whether a sub-session's response satisfies its objective.

    Makes a single LLM call using the TURING_OBJECTIVE_COMPLETION.txt
    prompt template.  Returns a violation dict if the objective is NOT met.
    """
    from wintermute import prompt_loader

    objective = context.get("objective", "")
    assistant_response = context.get("assistant_response", "")
    if not objective or not assistant_response:
        return None

    # Build tool call history summary for context.
    tool_calls = context.get("tool_calls_made", [])
    tool_summary = ", ".join(tool_calls) if tool_calls else "none"

    system_prompt = prompt_loader.load(
        "TURING_OBJECTIVE_COMPLETION.txt",
        objective=objective,
    )

    eval_context = json.dumps({
        "objective": objective,
        "assistant_response": _truncate_middle(assistant_response, keep_head=800, keep_tail=400),
        "tools_called_this_session": tool_summary,
    }, indent=2)

    response = await pool.call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": eval_context},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        logger.warning("objective_completion: empty LLM response — assuming objective met")
        return None

    # Strip markdown code fences.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("objective_completion: unparseable response %r — assuming met", raw[:200])
        return None

    met = result.get("objective_met", True)
    reason = result.get("reason", "")

    try:
        database.save_interaction_log(
            _time.time(), "turing_objective", thread_id,
            pool.last_used,
            eval_context[:2000], raw[:2000],
            "ok" if met else "violation_detected",
        )
    except Exception:
        pass

    if met:
        logger.debug("objective_completion: objective met for %s", thread_id)
        return None

    logger.info("objective_completion: objective NOT met for %s — %s", thread_id, reason[:200])
    return {"type": "objective_completion", "reason": reason}


def get_hooks(
    enabled_validators: Optional[dict[str, Any]] = None,
    *,
    phase_filter: Optional[str] = None,
    scope_filter: Optional[str] = None,
) -> list[TuringHook]:
    """Return the list of enabled hooks (for pre-flight checks by callers)."""
    return _load_hooks(enabled_validators, phase_filter=phase_filter, scope_filter=scope_filter)
