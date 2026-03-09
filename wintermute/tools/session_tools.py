"""Sub-session spawning and telemetry tool implementations."""

import json
import logging
import time
from typing import Optional

from wintermute.core.tool_deps import ToolDeps
from wintermute.infra import database

logger = logging.getLogger(__name__)


def tool_worker_delegation(inputs: dict, thread_id: Optional[str] = None,
                           nesting_depth: int = 0,
                           tool_deps: Optional[ToolDeps] = None, **_kw) -> str:
    deps = tool_deps or ToolDeps()
    action = inputs.get("action", "spawn")

    if action not in ("spawn", "status", "cancel"):
        return json.dumps({"error": f"Unknown action: {action}. Use spawn, status, or cancel."})

    if action == "status":
        if deps.sub_session_manager is None:
            return json.dumps({"error": "Sub-session manager not ready."})
        sessions = deps.sub_session_manager.list_active_threadsafe(thread_id)
        if not sessions:
            return json.dumps({"status": "No active background workers."})
        return json.dumps({"active_workers": sessions})

    if action == "cancel":
        target = inputs.get("target_id")
        if not target:
            return json.dumps({"error": "target_id required for cancel."})
        if deps.sub_session_manager is None:
            return json.dumps({"error": "Sub-session manager not ready."})
        result = deps.sub_session_manager.cancel(target, thread_id)
        return json.dumps({"result": result})

    # action == "spawn" — existing logic below
    if not inputs.get("objective"):
        return json.dumps({"error": "objective is required for spawn."})
    if nesting_depth >= deps.max_nesting_depth:
        return json.dumps({
            "error": (
                f"Maximum nesting depth ({deps.max_nesting_depth}) reached. "
                "Cannot spawn further sub-sessions."
            )
        })
    if deps.sub_session_manager is None:
        return json.dumps({"error": "Sub-session manager not ready yet."})
    try:
        context_blobs = list(inputs.get("context_blobs") or [])

        # Auto-inject last user message from parent thread
        if thread_id:
            try:
                last_user_msg = database.get_last_user_message(thread_id)
                if last_user_msg:
                    context_blobs.append(
                        f"[Parent conversation — last user message]\n{last_user_msg}"
                    )
            except Exception:
                logger.debug("Failed to fetch parent user message", exc_info=True)

        # Query historical outcomes for similar objectives.
        try:
            similar = database.get_similar_outcomes(inputs["objective"], limit=5)
            if similar:
                lines = []
                total_dur = 0
                dur_count = 0
                success_count = 0
                for o in similar:
                    dur = o.get("duration_seconds")
                    tov = o.get("timeout_value")
                    tc = o.get("tool_call_count", 0)
                    st = o.get("status", "?")
                    obj_short = (o.get("objective") or "")[:60]
                    dur_str = f"{dur:.0f}s" if dur is not None else "?"
                    tov_str = f"{tov}s timeout" if tov is not None else "no timeout"
                    cont = o.get("continuation_count", 0)
                    extra = f", continued {cont}x" if cont else ""
                    lines.append(f"- \"{obj_short}\" ({tov_str}): {st} in {dur_str}, {tc} tool calls{extra}")
                    if dur is not None:
                        total_dur += dur
                        dur_count += 1
                    if st == "completed":
                        success_count += 1
                avg_dur = f"{total_dur / dur_count:.0f}s" if dur_count else "N/A"
                rate = f"{success_count * 100 / len(similar):.0f}%"
                blob = (
                    "[Historical Feedback] Similar past sub-sessions:\n"
                    + "\n".join(lines)
                    + f"\nAverage duration: {avg_dur} | Success rate: {rate}"
                )
                context_blobs.insert(0, blob)
        except Exception as exc:
            logger.debug("Historical feedback lookup failed: %s", exc, exc_info=True)

        kwargs = dict(
            objective=inputs["objective"],
            context_blobs=context_blobs,
            parent_thread_id=thread_id,
            nesting_depth=nesting_depth + 1,
        )
        if "timeout" in inputs:
            kwargs["timeout"] = int(inputs["timeout"])
        if "depends_on" in inputs:
            kwargs["depends_on"] = inputs["depends_on"]
        if inputs.get("depends_on_previous"):
            kwargs["depends_on_previous"] = True
        if "not_before" in inputs:
            kwargs["not_before"] = inputs["not_before"]
        if "profile" in inputs:
            kwargs["profile"] = inputs["profile"]
        session_id = deps.sub_session_manager.spawn(**kwargs)
        has_deps = bool(inputs.get("depends_on")) or bool(inputs.get("depends_on_previous"))
        has_gate = bool(inputs.get("not_before"))
        if has_deps or has_gate:
            reasons = []
            if has_deps:
                reasons.append("dependencies")
            if has_gate:
                reasons.append(f"time gate ({inputs['not_before']})")
            status = f"pending (waiting for {' and '.join(reasons)})"
        else:
            status = "started"
        return json.dumps({
            "status": status,
            "session_id": session_id,
            "IMPORTANT": (
                "The worker is now running in the background. "
                "You do NOT have its results yet. "
                "Tell the user you started the task and STOP. "
                "Do NOT fabricate, guess, or anticipate what the worker will find. "
                "The results will be delivered to you later as a [SYSTEM EVENT]. "
                "Only then should you report them to the user."
            ),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("worker_delegation failed")
        return json.dumps({"error": str(exc)})


def tool_query_telemetry(inputs: dict, tool_deps: Optional[ToolDeps] = None, **_kw) -> str:
    """Query operational telemetry data."""
    query_type = inputs.get("query_type", "")
    since_hours = int(inputs.get("since_hours", 24))
    limit = int(inputs.get("limit", 10))
    status_filter = inputs.get("status_filter")
    since_ts = time.time() - since_hours * 3600

    try:
        if query_type == "outcome_stats":
            stats = database.get_outcome_stats()
            return json.dumps(stats)

        elif query_type == "recent_outcomes":
            kwargs = {"since": since_ts, "limit": limit}
            if status_filter:
                kwargs["status_filter"] = status_filter
            outcomes = database.get_outcomes_since(**kwargs)
            rows = []
            for o in outcomes:
                rows.append({
                    "session_id": o.get("session_id", ""),
                    "objective": (o.get("objective") or "")[:120],
                    "status": o.get("status", ""),
                    "duration_seconds": o.get("duration_seconds"),
                    "tool_call_count": o.get("tool_call_count", 0),
                })
            return json.dumps({"outcomes": rows, "count": len(rows)})

        elif query_type == "skill_stats":
            from wintermute.infra import skill_store
            return json.dumps(skill_store.stats())

        elif query_type == "top_tools":
            tool_stats = database.get_tool_usage_stats(since_ts)
            return json.dumps({"tools": [{"name": t, "count": c} for t, c in tool_stats[:limit]]})

        elif query_type == "interaction_log":
            entries = database.get_interaction_log(limit=limit)
            rows = []
            for e in entries:
                rows.append({
                    "id": e.get("id"),
                    "action": e.get("action", ""),
                    "session": e.get("session", ""),
                    "input": (e.get("input") or "")[:200],
                    "output": (e.get("output") or "")[:200],
                    "status": e.get("status", ""),
                })
            return json.dumps({"entries": rows, "count": len(rows)})

        elif query_type == "self_model":
            deps = tool_deps or ToolDeps()
            if deps.self_model_profiler is None:
                return json.dumps({"error": "Self-model profiler not available"})
            summary = deps.self_model_profiler.get_summary()
            raw_metrics = {}
            try:
                yaml_path = deps.self_model_profiler.yaml_path
                if yaml_path.exists():
                    import yaml
                    raw_metrics = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
            return json.dumps({"summary": summary, "metrics": raw_metrics})

        else:
            return json.dumps({"error": f"Unknown query_type: {query_type}"})

    except Exception as exc:  # noqa: BLE001
        logger.exception("query_telemetry failed")
        return json.dumps({"error": str(exc)})
