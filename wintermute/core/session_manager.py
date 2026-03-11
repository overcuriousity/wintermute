"""
Session Manager

Manages session lifecycle: timeouts, resets, per-thread config resolution,
and pool resolution.

Extracted from LLMThread as part of the Phase 4 god-object decomposition (#79).
"""

import logging
import time as _time
from collections.abc import Callable
from typing import Optional, TYPE_CHECKING

from wintermute.infra import database
from wintermute.core.types import BackendPool
if TYPE_CHECKING:
    from wintermute.core.sub_session import SubSessionManager
    from wintermute.infra.thread_config import ThreadConfigManager, ResolvedThreadConfig
    from wintermute.core.conversation_store import ConversationStore

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session lifecycle: timeouts, resets, per-thread config/pool resolution."""

    def __init__(
        self,
        main_pool: BackendPool,
        thread_config_manager: "Optional[ThreadConfigManager]" = None,
        backend_pools_by_name: "Optional[dict[str, BackendPool]]" = None,
        sub_session_getter: "Optional[Callable[[], Optional[SubSessionManager]]]" = None,
        store: "Optional[ConversationStore]" = None,
    ) -> None:
        self._main_pool = main_pool
        self._thread_config_manager = thread_config_manager
        self._backend_pools_by_name = backend_pools_by_name or {}
        self._get_sub_sessions = sub_session_getter
        self._store = store
        # Per-thread last activity timestamp (for session timeout tracking).
        self.last_activity: dict[str, float] = {}
        # Per-thread tool calls from the previous turn.
        self.prior_tool_calls: dict[str, list[str]] = {}

    @property
    def thread_config_manager(self) -> "Optional[ThreadConfigManager]":
        return self._thread_config_manager

    def resolve_pool(self, thread_id: str) -> BackendPool:
        """Return the inference pool for a thread, respecting per-thread overrides."""
        if not self._thread_config_manager:
            return self._main_pool
        resolved = self._thread_config_manager.resolve(thread_id)
        if resolved.backend_name and resolved.backend_name in self._backend_pools_by_name:
            return self._backend_pools_by_name[resolved.backend_name]
        return self._main_pool

    def resolve_config(self, thread_id: str) -> "Optional[ResolvedThreadConfig]":
        """Return resolved per-thread config, or None if no manager is set."""
        if not self._thread_config_manager:
            return None
        return self._thread_config_manager.resolve(thread_id)

    def check_session_timeouts(self) -> list[str]:
        """Return thread_ids that have exceeded their configured session timeout."""
        if not self._thread_config_manager:
            return []
        now = _time.time()
        expired = []
        for tid, last_ts in self.last_activity.items():
            resolved = self._thread_config_manager.resolve(tid)
            timeout = resolved.session_timeout_minutes
            if timeout is not None and (now - last_ts) > timeout * 60:
                expired.append(tid)
        return expired

    def record_activity(self, thread_id: str) -> None:
        """Record user activity for session timeout tracking."""
        self.last_activity[thread_id] = _time.time()

    async def reset_session(self, thread_id: str = "default") -> None:
        """Clear active messages and cancel sub-sessions for a thread."""
        await database.async_call(database.clear_active_messages, thread_id)
        if self._store:
            self._store.compaction_summaries.pop(thread_id, None)
        ssm = self._get_sub_sessions() if self._get_sub_sessions else None
        if ssm:
            n = await ssm.cancel_for_thread(thread_id)
            if n:
                logger.info("Cancelled %d sub-session(s) for reset thread %s", n, thread_id)
        logger.info("Session reset for thread %s", thread_id)
