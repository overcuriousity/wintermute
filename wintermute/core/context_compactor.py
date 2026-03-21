"""
Context Compactor

Handles context window compaction: summarising old messages, archiving them,
trimming tool results, and monitoring component sizes.

Extracted from LLMThread as part of the Phase 4 god-object decomposition (#79).
"""

import logging
import time as _time
from typing import TYPE_CHECKING

from wintermute.infra import database
from wintermute.infra import prompt_assembler
from wintermute.infra import prompt_loader
from wintermute.core.types import BackendPool
from wintermute.core.conversation_store import count_tokens
if TYPE_CHECKING:
    from wintermute.core.conversation_store import ConversationStore

logger = logging.getLogger(__name__)

# Keep the last N messages untouched during compaction (default; overridable via config).
COMPACTION_KEEP_RECENT = 10


class ContextCompactor:
    """Manages context window compaction and tool-result trimming."""

    def __init__(
        self,
        compaction_pool: BackendPool,
        broadcast_fn,
        store: "ConversationStore",
        keep_recent: int = COMPACTION_KEEP_RECENT,
        enqueue_system_event_fn=None,
        event_bus=None,
    ) -> None:
        self._compaction_pool = compaction_pool
        self._broadcast = broadcast_fn
        self._store = store
        self._enqueue_system_event = enqueue_system_event_fn
        self._event_bus = event_bus
        try:
            _keep = int(keep_recent)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid compaction_keep_recent %r; falling back to default %d",
                keep_recent, COMPACTION_KEEP_RECENT,
            )
            _keep = COMPACTION_KEEP_RECENT
        if _keep < 1:
            logger.warning("compaction_keep_recent %r < 1; clamping to 1", _keep)
            _keep = 1
        self._keep_recent = _keep
        self._last_skills_oversized_emit: float = 0.0
        self._SKILLS_EMIT_INTERVAL: float = 300.0  # seconds

    @property
    def pool(self) -> BackendPool:
        """The compaction backend pool."""
        return self._compaction_pool

    async def compact(self, thread_id: str = "default",
                      keep_recent: int | None = None,
                      pool_override: "BackendPool | None" = None) -> None:
        """Summarise and archive old messages for the given thread.

        *keep_recent* overrides the instance default when provided (per-thread config).
        *pool_override* uses the given pool instead of the instance compaction pool.
        """
        effective_keep = keep_recent if keep_recent is not None else self._keep_recent
        if effective_keep < 1:
            logger.warning(
                "Invalid keep_recent value %r in compact(); clamping to 1 for thread %s",
                keep_recent,
                thread_id,
            )
            effective_keep = 1
        rows = await database.async_call(database.load_active_messages, thread_id)
        if len(rows) <= effective_keep:
            return

        to_summarise = rows[:-effective_keep]

        prior_summary = self._store.compaction_summaries.get(thread_id)
        parts = []
        if prior_summary:
            parts.append(f"[PRIOR SUMMARY]\n{prior_summary}\n")
        parts.append("[NEW MESSAGES]\n" + "\n".join(
            f"{r['role'].upper()}: {r['content']}" for r in to_summarise
        ))
        history_text = "\n\n".join(parts)

        summary_prompt = prompt_loader.load("COMPACTION_PROMPT.txt", history=history_text)

        try:
            _pool = pool_override or self._compaction_pool
            summary_response = await _pool.call(
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens_override=2048,
            )
            summary = (summary_response.content or "").strip()
        except Exception:  # noqa: BLE001
            logger.exception("Compaction failed for thread %s — skipping", thread_id)
            return

        try:
            await database.async_call(
                database.save_interaction_log,
                _time.time(), "compaction", thread_id,
                _pool.last_used,
                summary_prompt, summary, "ok",
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to save compaction interaction log", exc_info=True)

        if to_summarise:
            await database.async_call(database.archive_messages, to_summarise[-1]["id"], thread_id)
        await database.async_call(database.save_summary, summary, thread_id)
        self._store.compaction_summaries[thread_id] = summary

        logger.info("Compacted %d messages into summary (%d chars) for thread %s",
                     len(to_summarise), len(summary), thread_id)
        try:
            await self._broadcast(
                "\U0001f4e6 Context compacted: old messages archived and summarised.",
                thread_id,
            )
        except Exception:  # noqa: BLE001
            pass

    async def maybe_summarise_components(
        self, thread_id: str = "default",
        *, _from_system_event: bool = False,
    ) -> None:
        """Check prompt component sizes and request summarisation if oversized."""
        if _from_system_event:
            return
        if not self._enqueue_system_event:
            return
        sizes = prompt_assembler.check_component_sizes()
        for component, oversized in sizes.items():
            if not oversized:
                continue
            # Skills are managed by the nightly dreaming cycle (auto-retire,
            # dedup, condense) — the inline LLM lacks delete capability and
            # cannot reduce the skill count.  Emit an event so DreamingLoop
            # can schedule an early consolidation instead.
            if component == "skills":
                if not self._event_bus:
                    now = _time.monotonic()
                    if now - self._last_skills_oversized_emit >= self._SKILLS_EMIT_INTERVAL:
                        self._last_skills_oversized_emit = now
                        logger.warning("Skills TOC oversized but event bus unavailable")
                    continue
                now = _time.monotonic()
                if now - self._last_skills_oversized_emit >= self._SKILLS_EMIT_INTERVAL:
                    self._last_skills_oversized_emit = now
                    logger.info("Skills TOC oversized – requesting early dreaming consolidation")
                    self._event_bus.emit("skills.oversized")
                continue
            logger.info("Component '%s' oversized – requesting AI summarisation", component)
            prompt = prompt_loader.load("COMPONENT_OVERSIZE.txt", component=component)
            try:
                await self._enqueue_system_event(prompt, thread_id)
                await self._broadcast(
                    f"\u2139\ufe0f Auto-summarising {component} (size limit reached).",
                    thread_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Could not request summarisation for %s", component)

    def trim_tool_results(self, messages: list[dict], token_budget: int, model: str) -> None:
        """Truncate oldest tool-result messages if total tokens exceed budget.

        Modifies *messages* in place.  Only tool-role messages are truncated
        (replaced with a short notice).
        """
        total = sum(
            count_tokens(
                m.get("content", "") if isinstance(m.get("content"), str)
                else " ".join(
                    p.get("text", "") for p in m.get("content", [])
                    if isinstance(p, dict)
                ) if isinstance(m.get("content"), list)
                else str(m.get("content", "") or ""),
                model,
            )
            for m in messages
        )
        if total <= token_budget:
            return

        tool_indices = [
            i for i, m in enumerate(messages)
            if m["role"] == "tool"
        ]
        truncation_notice = "[tool output truncated to fit context window]"
        for idx in tool_indices:
            if total <= token_budget:
                break
            msg = messages[idx]
            old_content = msg["content"]
            if old_content == truncation_notice:
                continue
            old_tokens = count_tokens(
                old_content if isinstance(old_content, str) else str(old_content),
                model,
            )
            new_tokens = count_tokens(truncation_notice, model)
            msg["content"] = truncation_notice
            total -= (old_tokens - new_tokens)
            logger.info("Trimmed tool result at index %d (saved ~%d tokens)", idx, old_tokens - new_tokens)
