"""Memory and skill tool implementations."""

import json
import logging
from typing import Optional

from wintermute.core.tool_deps import ToolDeps
from wintermute.infra import prompt_assembler

logger = logging.getLogger(__name__)


def tool_append_memory(inputs: dict, tool_deps: Optional[ToolDeps] = None, **_kw) -> str:
    try:
        deps = tool_deps or ToolDeps()
        source = inputs.get("source", "user_explicit")
        total_len = prompt_assembler.append_memory(inputs["entry"], source=source)
        if deps.event_bus:
            deps.event_bus.emit("memory.appended", entry=inputs["entry"][:200])
        return json.dumps({"status": "ok", "total_chars": total_len})
    except Exception as exc:  # noqa: BLE001
        logger.exception("append_memory failed")
        return json.dumps({"error": str(exc)})


def tool_add_skill(inputs: dict, tool_deps: Optional[ToolDeps] = None, **_kw) -> str:
    try:
        deps = tool_deps or ToolDeps()
        prompt_assembler.add_skill(
            inputs["skill_name"],
            inputs["documentation"],
            summary=inputs.get("summary"),
        )
        if deps.event_bus:
            deps.event_bus.emit("skill.added", skill_name=inputs["skill_name"])
        try:
            from wintermute.workers import skill_stats
            skill_stats.record_skill_written(inputs["skill_name"])
        except Exception:
            pass
        return json.dumps({"status": "ok", "skill": inputs["skill_name"]})
    except Exception as exc:  # noqa: BLE001
        logger.exception("add_skill failed")
        return json.dumps({"error": str(exc)})
