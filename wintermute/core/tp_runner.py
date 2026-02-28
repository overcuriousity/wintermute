"""
Turing Protocol Runner — unified entry point for all TP phase checks.

Replaces three near-identical methods:
  - ``LLMThread._run_phase_check``
  - ``LLMThread._run_turing_check``  (post_inference for main thread)
  - ``SubSessionManager._run_tp_phase``

A ``TuringProtocolRunner`` binds ``pool``, ``scope``, and
``enabled_validators`` at construction time so callers only need to pass
the per-turn arguments (phase, messages, tool info, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wintermute.core.types import BackendPool

from wintermute.core import turing_protocol as turing_protocol_module

logger = logging.getLogger(__name__)


class TuringProtocolRunner:
    """Bound TP executor for a specific pool + scope.

    Parameters
    ----------
    pool : BackendPool
        The LLM pool used for TP validation calls.
    scope : str
        ``"main"`` or ``"sub_session"``.
    enabled_validators : dict | None
        Per-hook enable/disable overrides from config.
    """

    def __init__(
        self,
        pool: "BackendPool",
        scope: str,
        enabled_validators: "Optional[dict]" = None,
    ) -> None:
        self._pool = pool
        self._scope = scope
        self._validators = enabled_validators

    @property
    def enabled(self) -> bool:
        return self._pool is not None and self._pool.enabled

    async def run_phase(
        self,
        phase: str,
        *,
        thread_id: str = "",
        tool_calls_made: Optional[list[str]] = None,
        assistant_response: str = "",
        user_message: str = "",
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_result: Optional[str] = None,
        nl_tools: "set[str] | None" = None,
        active_sessions: Optional[list] = None,
        objective: Optional[str] = None,
        prior_assistant_message: Optional[str] = None,
        recent_assistant_messages: Optional[list[str]] = None,
    ) -> Optional[turing_protocol_module.TuringResult]:
        """Run TP hooks for *phase* in the bound scope.

        Returns ``TuringResult`` if violations were confirmed, ``None``
        otherwise.  Exceptions are caught and logged as non-fatal.
        """
        if not self.enabled:
            return None

        hooks = turing_protocol_module.get_hooks(
            self._validators,
            phase_filter=phase,
            scope_filter=self._scope,
        )
        if not hooks:
            return None

        try:
            return await turing_protocol_module.run_turing_protocol(
                pool=self._pool,
                user_message=user_message,
                assistant_response=assistant_response,
                tool_calls_made=tool_calls_made or [],
                active_sessions=active_sessions or [],
                enabled_validators=self._validators,
                thread_id=thread_id,
                phase=phase,
                scope=self._scope,
                objective=objective,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                nl_tools=nl_tools,
                prior_assistant_message=prior_assistant_message,
                recent_assistant_messages=recent_assistant_messages,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Turing Protocol %s/%s check raised (non-fatal)",
                phase, self._scope,
            )
            return None
