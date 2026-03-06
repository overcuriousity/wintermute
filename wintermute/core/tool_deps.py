"""Dependency container for tool execution.

Replaces the module-level globals and ``register_*`` / ``set_*`` wiring
that previously lived in ``tools.py`` and ``prompt_assembler.py``.

A single ``ToolDeps`` instance is created in ``main.py`` and threaded
through ``ToolCallContext`` → ``execute_tool()`` → individual tool
functions.

Object references (``sub_session_manager``, ``task_scheduler``,
``self_model_profiler``) are set once after the target objects are
constructed.  Tool code accesses callables through these typed
references instead of storing individual callables — extending a
dependency requires zero rewiring in ``main.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from wintermute.core.sub_session import SubSessionManager
    from wintermute.infra.event_bus import EventBus
    from wintermute.workers.scheduler_thread import TaskScheduler
    from wintermute.workers.self_model import SelfModelProfiler


@dataclass
class ToolDeps:
    """Runtime dependencies injected into tool functions.

    Holds typed object references rather than individual callables so that
    adding new functionality from an existing dependency requires no extra
    wiring in ``main.py``.
    """

    # --- Typed object references (set once after construction) ---

    # Sub-session manager (spawn / cancel / status).
    sub_session_manager: Optional["SubSessionManager"] = None

    # Task scheduler (ensure_job / remove_job / list_jobs).
    task_scheduler: Optional["TaskScheduler"] = None

    # Event bus instance.
    event_bus: Optional["EventBus"] = None

    # Self-model profiler instance.
    self_model_profiler: Optional["SelfModelProfiler"] = None

    # --- Simple config values (set at construction) ---

    # SearXNG search URL.
    searxng_url: str = ""

    # Maximum sub-session nesting depth.
    max_nesting_depth: int = 2

    # Tool profiles for sub-session spawning (injected into prompt assembly).
    tool_profiles: dict[str, dict] = field(default_factory=dict)
