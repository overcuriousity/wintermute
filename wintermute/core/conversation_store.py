"""
Conversation Store

Manages conversation history persistence, message construction, token counting,
and read-only accessors for compaction summaries and system prompts.

Extracted from LLMThread as part of the Phase 4 god-object decomposition (#79).
"""

import json
import logging
from typing import Optional, TYPE_CHECKING

from wintermute.infra import database
from wintermute.infra import prompt_assembler
from wintermute.core.types import ProviderConfig
if TYPE_CHECKING:
    from wintermute.core.tool_deps import ToolDeps
    from wintermute.infra.event_bus import EventBus

logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str) -> int:
    """Estimate token count using tiktoken.

    Falls back to cl100k_base (GPT-4 / DeepSeek / Qwen BPE) for unknown
    model names, and to len//4 if tiktoken itself is unavailable.
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return len(text) // 4


class ConversationStore:
    """Owns conversation history, message persistence, and token accounting."""

    def __init__(
        self,
        primary_config: ProviderConfig,
        event_bus: "Optional[EventBus]" = None,
        nl_translation_config: "Optional[dict]" = None,
        tool_deps: "Optional[ToolDeps]" = None,
    ) -> None:
        self._cfg = primary_config
        self._event_bus = event_bus
        self._nl_translation_config = nl_translation_config or {}
        self._tool_deps = tool_deps
        # Per-thread compaction summaries: thread_id -> summary text
        self.compaction_summaries: dict[str, Optional[str]] = {}
        # Per-thread sequence counter for user-facing turns.
        self.thread_seq: dict[str, int] = {}
        # Per-thread cache of the last system prompt actually sent to the LLM.
        self.last_system_prompt: dict[str, str] = {}
        # Per-thread cache of the last tool schemas actually sent to the LLM.
        self.last_tool_schemas: dict[str, list] = {}

    async def load_summaries(self) -> None:
        """Load compaction summaries for all known threads from DB."""
        for tid in await database.async_call(database.get_active_thread_ids):
            summary = await database.async_call(database.load_latest_summary, tid)
            if summary:
                self.compaction_summaries[tid] = summary

    def get_compaction_summary(self, thread_id: str = "default") -> Optional[str]:
        """Return the current in-memory compaction summary for a thread, or None."""
        return self.compaction_summaries.get(thread_id)

    def get_last_system_prompt(self, thread_id: str = "default") -> Optional[str]:
        """Return the last system prompt actually sent to the LLM for *thread_id*."""
        return self.last_system_prompt.get(thread_id)

    def get_last_tool_schemas(self, thread_id: str = "default") -> Optional[list]:
        """Return the last tool schemas actually sent to the LLM for *thread_id*."""
        return self.last_tool_schemas.get(thread_id)

    def get_token_budget(self, thread_id: str = "default") -> dict:
        """Return precise token accounting for a thread."""
        from wintermute import tools as tool_module

        total_limit = max(self._cfg.context_size - self._cfg.max_tokens, 1)
        model = self._cfg.model

        nl_enabled = self._nl_translation_config.get("enabled", False)
        nl_tools = self._nl_translation_config.get("tools", set()) if nl_enabled else None

        sp_text = self.last_system_prompt.get(thread_id)
        if sp_text is None:
            summary = self.compaction_summaries.get(thread_id)
            try:
                sp_text = prompt_assembler.assemble(
                    extra_summary=summary,
                    prompt_mode="minimal",
                    tool_profiles=self._tool_deps.tool_profiles if self._tool_deps else None,
                    nl_tools=nl_tools,
                )
            except Exception:  # noqa: BLE001
                sp_text = ""
        sp_tokens = count_tokens(sp_text, model)
        # Prefer the exact schemas last sent to the LLM (respects lite-mode
        # exclusions, tool profiles, etc.).  Fall back to a fresh build only
        # when no cached version exists yet.
        active_schemas = self.last_tool_schemas.get(thread_id)
        if active_schemas is None:
            active_schemas = tool_module.get_tool_schemas(
                nl_tools=nl_tools,
                tool_profiles=self._tool_deps.tool_profiles if self._tool_deps else None,
            )
        tools_tokens = count_tokens(json.dumps(active_schemas), model)

        stats = database.get_thread_stats(thread_id)
        hist_tokens = stats["token_used"]

        total_used = sp_tokens + tools_tokens + hist_tokens
        pct = round(min(total_used / total_limit * 100, 100), 1)

        return {
            "total_limit": total_limit,
            "sp_tokens": sp_tokens,
            "tools_tokens": tools_tokens,
            "hist_tokens": hist_tokens,
            "total_used": total_used,
            "pct": pct,
            "msg_count": stats["msg_count"],
        }

    async def build_messages(
        self, new_text: str, is_system_event: bool,
        thread_id: str = "default",
        content: Optional[list] = None,
        ephemeral: bool = False,
    ) -> list[dict]:
        """Load active messages from DB and append the new user message.

        When *ephemeral* is True, prior history is skipped — only the new
        message is returned (group mode single-turn).
        """
        if ephemeral:
            messages: list[dict] = []
        else:
            rows = await database.async_call(database.load_active_messages, thread_id)
            messages = [
                {"role": r["role"], "content": r["content"] or "..."}
                for r in rows
            ]
        prefix = "[SYSTEM EVENT] " if is_system_event else ""
        if content is not None and not is_system_event:
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": f"{prefix}{new_text}"})
        return messages

    async def save_user_message(
        self, text: str, thread_id: str, is_system_event: bool,
        is_sub_session_result: bool, convergence_depth: int,
        content: Optional[list], model: str,
    ) -> None:
        """Persist the incoming user/system message to DB and emit events."""
        if not is_system_event:
            db_text = text
            if content is not None:
                db_text = text or "[image attached]"
            await database.async_call(
                database.save_message, "user", db_text, thread_id,
                token_count=count_tokens(db_text, model))
            if self._event_bus:
                self._event_bus.emit("message.received", thread_id=thread_id, text=db_text)
        elif is_sub_session_result:
            _se_text = f"[SYSTEM EVENT] {text}"
            await database.async_call(
                database.save_message, "user", _se_text, thread_id,
                token_count=count_tokens(_se_text, model))
        elif convergence_depth > 0:
            await database.async_call(
                database.save_message, "user", text, thread_id,
                token_count=count_tokens(text, model))

    async def save_assistant_message(
        self, text: Optional[str], thread_id: str, model: str,
    ) -> None:
        """Persist the assistant response to DB and emit events."""
        _assistant_text = text or "..."
        await database.async_call(
            database.save_message, "assistant", _assistant_text, thread_id,
            token_count=count_tokens(_assistant_text, model))
        if self._event_bus:
            self._event_bus.emit("message.sent", thread_id=thread_id, text=_assistant_text)
