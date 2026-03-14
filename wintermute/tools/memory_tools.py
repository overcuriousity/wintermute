"""Memory and skill tool implementations."""

import json
import logging
from typing import Optional

from wintermute.core.tool_deps import ToolDeps
from wintermute.infra.memory_io import append_memory
from wintermute.infra import skill_io

logger = logging.getLogger(__name__)


def tool_append_memory(inputs: dict, tool_deps: Optional[ToolDeps] = None, **_kw) -> str:
    try:
        deps = tool_deps or ToolDeps()
        source = inputs.get("source", "user_explicit")
        total, status = append_memory(
            inputs["entry"],
            source=source,
            pool=deps.memory_pool,
            loop=deps.event_loop,
            event_bus=deps.event_bus,
        )
        # Event emission is handled by append_memory() (single source of truth).
        return json.dumps({"status": status, "total_entries": total})
    except Exception as exc:  # noqa: BLE001
        logger.exception("append_memory failed")
        return json.dumps({"error": str(exc)})


def tool_skill(inputs: dict, tool_deps: Optional[ToolDeps] = None, **_kw) -> str:
    """Unified skill tool: add / read / search."""
    try:
        deps = tool_deps or ToolDeps()
        action = inputs.get("action", "add")

        if action == "add":
            name = inputs.get("skill_name")
            if not name:
                return json.dumps({"error": "skill_name is required for action 'add'"})
            documentation = inputs.get("documentation")
            if not documentation:
                return json.dumps({"error": "documentation is required for action 'add'"})
            skill_io.add_skill(
                name,
                documentation,
                summary=inputs.get("summary"),
            )
            if deps.event_bus:
                deps.event_bus.emit("skill.added", skill_name=name)
            return json.dumps({"status": "ok", "skill": name})

        if action == "read":
            name = inputs.get("skill_name")
            if not name:
                return json.dumps({"error": "skill_name is required for action 'read'"})
            record = skill_io.read_skill(name)
            if record is None:
                return json.dumps({"error": f"Skill '{name}' not found"})
            return json.dumps({"status": "ok", "skill": record})

        if action == "search":
            query = inputs.get("query", "")
            try:
                top_k = int(inputs.get("top_k", 5))
            except (TypeError, ValueError):
                top_k = 5
            top_k = max(1, min(top_k, 50))  # clamp to reasonable range
            results = skill_io.search_skills(query, top_k)
            return json.dumps({"status": "ok", "results": results,
                               "count": len(results)})

        return json.dumps({"error": f"Unknown action: {action}"})

    except Exception as exc:  # noqa: BLE001
        logger.exception("tool_skill failed")
        return json.dumps({"error": str(exc)})
