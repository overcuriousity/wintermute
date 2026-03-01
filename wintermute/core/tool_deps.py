"""Dependency container for tool execution.

Replaces the module-level globals and ``register_*`` / ``set_*`` wiring
that previously lived in ``tools.py`` and ``prompt_assembler.py``.

A single ``ToolDeps`` instance is created in ``main.py`` and threaded
through ``ToolCallContext`` → ``execute_tool()`` → individual tool
functions.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolDeps:
    """Runtime dependencies injected into tool functions."""

    # Task scheduler callables (from scheduler_thread.py).
    task_scheduler_ensure: Optional[Callable] = None
    task_scheduler_remove: Optional[Callable] = None
    task_scheduler_list: Optional[Callable] = None

    # Sub-session lifecycle callables (from SubSessionManager).
    sub_session_spawn: Optional[Callable] = None
    sub_session_cancel: Optional[Callable] = None
    sub_session_status: Optional[Callable] = None

    # Event bus instance.
    event_bus: Optional[Any] = None

    # Self-model profiler instance.
    self_model_profiler: Optional[Any] = None

    # SearXNG search URL.
    searxng_url: str = ""

    # Maximum sub-session nesting depth.
    max_nesting_depth: int = 2

    # Tool profiles for sub-session spawning (injected into prompt assembly).
    tool_profiles: dict[str, dict] = field(default_factory=dict)
