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

# Token headroom reserved for the shrink prompt wrapper when computing the per-message
# input limit.  The shrink prompt itself (role label + instruction framing) adds roughly
# 100–200 tokens; the extra margin covers variance across templates and backends.
_SHRINK_PROMPT_HEADROOM = 768


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
                      pool_override: "BackendPool | None" = None,
                      inference_context_size: int | None = None) -> None:
        """Summarise and archive old messages for the given thread.

        *keep_recent* overrides the instance default when provided (per-thread config).
        *pool_override* uses the given pool instead of the instance compaction pool.
        *inference_context_size* is the context window of the inference backend that
        triggered compaction.  When provided it is used as the basis for the per-message
        shrink threshold so messages that are oversized for inference are always condensed,
        even when the compaction backend has a larger context window.
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

        # Pre-pass: atomically shrink individual messages that exceed their fair-share
        # token budget.  This handles the case where a small number of very large
        # messages fills the context window (count-based compaction can't help there).
        _pool = pool_override or self._compaction_pool
        rows = await self._shrink_large_messages(
            rows, effective_keep, _pool, thread_id,
            inference_context_size=inference_context_size,
        )

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

    async def _shrink_large_messages(
        self,
        rows: list[dict],
        keep_recent: int,
        pool: BackendPool,
        thread_id: str,
        *,
        inference_context_size: int | None = None,
    ) -> list[dict]:
        """Atomically summarise individual messages that exceed their per-message budget.

        Threshold = max(300, available_context // keep_recent).  Any message
        whose content exceeds this is condensed via a single LLM call to reduce
        verbosity.  Messages that exceed shrink_input_limit are truncated before
        the LLM call to avoid ContextTooLargeError; very large messages may lose
        information beyond that limit.

        Messages in the keep_recent tail are also persisted to the DB so that
        subsequent build_messages() calls load the smaller content.  Messages
        that will be archived are updated in-memory only — they are discarded
        anyway after the count-based compaction step.

        *inference_context_size* pins the threshold to the inference backend's
        context window when it differs from the compaction pool's.  This ensures
        messages that are oversized for inference are condensed even when the
        compaction backend has a larger context window.  The compaction pool's
        context_size still caps the content sent to the shrink LLM.
        """
        # The shrink call uses max_tokens_override=512, so reserve exactly that
        # (not pool.primary.max_tokens) when computing available input space.
        _SHRINK_MAX_TOKENS = 512
        # Use the inference backend's context window for the per-message threshold
        # so messages oversized for inference are always shrunk, even when the
        # compaction backend has a larger window.  Take the minimum to never
        # exceed what the shrink LLM can handle.
        _threshold_context = min(
            inference_context_size or pool.primary.context_size,
            pool.primary.context_size,
        )
        available = max(1, _threshold_context - _SHRINK_MAX_TOKENS)
        # Use the actual message count as the denominator so a keep_recent value
        # larger than the history doesn't produce an artificially tiny threshold.
        effective_divisor = max(1, min(keep_recent, len(rows)))
        threshold = max(300, available // effective_divisor)
        # Cap the content sent to the shrink LLM: must fit within the backend
        # context window (minus headroom for the prompt wrapper + response)
        # and never exceed the per-message budget.
        shrink_input_limit = min(threshold, max(1, available - _SHRINK_PROMPT_HEADROOM))
        model = pool.primary.model
        keep_start_idx = max(0, len(rows) - keep_recent)
        shrink_template = prompt_loader.load("MESSAGE_SHRINK_PROMPT.txt")
        updated = list(rows)

        # Cap LLM calls to avoid runaway cost on threads with many large messages.
        # Archivable messages (to-be-summarised) and kept messages each get their
        # own counter so that a large keep_recent value can't cause an unbounded
        # number of calls while still ensuring kept messages are always attempted.
        _MAX_SHRINK_OPS = 20
        archive_shrink_attempts = 0
        kept_shrink_attempts = 0
        for i, row in enumerate(updated):
            is_kept = i >= keep_start_idx
            if is_kept:
                if kept_shrink_attempts >= _MAX_SHRINK_OPS:
                    continue
            else:
                if archive_shrink_attempts >= _MAX_SHRINK_OPS:
                    continue

            content = row.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            tokens = count_tokens(content, model)
            original_tokens = tokens  # preserve pre-truncation count for logging
            if tokens <= threshold:
                continue

            # Truncate content that would exceed the backend's context window so
            # the LLM call doesn't fail with ContextTooLargeError.
            # Use iterative halving with token recount to handle CJK/symbol-heavy
            # text where the ≈4 chars-per-token estimate doesn't hold.
            # Include the suffix in the loop condition so the final content
            # (with marker) is guaranteed to fit within shrink_input_limit.
            if tokens > shrink_input_limit:
                suffix = "\n[truncated for compaction]"
                while tokens > shrink_input_limit and len(content) > 1:
                    content = content[: max(1, len(content) // 2)]
                    tokens = count_tokens(content + suffix, model)
                content = content + suffix
                tokens = count_tokens(content, model)
                logger.warning(
                    "Message %d (%d tokens) exceeds shrink input limit (%d) — "
                    "truncating before LLM call for thread %s",
                    row["id"], original_tokens, shrink_input_limit, thread_id,
                )

            if is_kept:
                kept_shrink_attempts += 1
            else:
                archive_shrink_attempts += 1
            try:
                shrink_prompt = shrink_template.format(role=row["role"], content=content)
                response = await pool.call(
                    messages=[{"role": "user", "content": shrink_prompt}],
                    max_tokens_override=512,
                )
                shrunken = (response.content or "").strip()
                if not shrunken:
                    logger.warning(
                        "Empty shrink response for message %d in thread %s — skipping",
                        row["id"], thread_id,
                    )
                    continue
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to shrink message %d for thread %s — skipping",
                    row["id"], thread_id,
                )
                continue

            new_tokens = count_tokens(shrunken, model)
            logger.info(
                "Shrank message %d (%s) from %d to %d tokens for thread %s",
                row["id"], row["role"], original_tokens, new_tokens, thread_id,
            )
            updated[i] = {**row, "content": shrunken, "token_count": new_tokens}

            # Persist only kept messages — archived ones are summarised then discarded.
            if is_kept:
                await database.async_call(
                    database.update_message_content,
                    row["id"], shrunken, new_tokens, thread_id,
                )

        return updated

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
