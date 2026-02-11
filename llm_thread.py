"""
LLM Inference Thread

Owns the conversation history and all interactions with the Anthropic API.
Receives user messages via an asyncio Queue, runs inference (including
multi-step tool use), and delivers responses back through a reply Queue.

Public API used by other modules
---------------------------------
  LLMThread.enqueue_user_message(text)      -> reply future
  LLMThread.enqueue_system_event(text)      -> None (fire-and-forget)
  LLMThread.reset_session()
  LLMThread.force_compact()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

import database
import prompt_assembler
import tools as tool_module

logger = logging.getLogger(__name__)

# Approximate character threshold at which we trigger compaction.
COMPACTION_THRESHOLD_CHARS = 150_000
# Keep the last N messages untouched during compaction.
COMPACTION_KEEP_RECENT = 10


@dataclass
class LLMConfig:
    api_key: str
    model: str = "claude-opus-4-5-20251101"
    max_tokens: int = 4096


@dataclass
class _QueueItem:
    text: str
    is_system_event: bool = False
    future: Optional[asyncio.Future] = field(default=None, compare=False)


class LLMThread:
    """Runs in its own asyncio task within the shared event loop."""

    def __init__(self, config: LLMConfig, matrix_send_fn) -> None:
        self._cfg = config
        self._matrix_send = matrix_send_fn  # async callable(text: str)
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._client = anthropic.AsyncAnthropic(api_key=config.api_key)
        self._running = False
        self._compaction_summary: Optional[str] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def enqueue_user_message(self, text: str) -> str:
        """Submit a user message and await the AI reply."""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_QueueItem(text=text, future=fut))
        return await fut

    async def enqueue_system_event(self, text: str) -> None:
        """Submit an autonomous system event (heartbeat, reminder, etc.)."""
        await self._queue.put(_QueueItem(text=text, is_system_event=True))

    async def reset_session(self) -> None:
        """Archive conversation history and start a fresh session."""
        database.clear_active_messages()
        self._compaction_summary = None
        logger.info("Session reset by /new command")

    async def force_compact(self) -> None:
        """Force immediate context compaction."""
        await self._compact_context()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        # Restore latest compaction summary if one exists.
        self._compaction_summary = database.load_latest_summary()
        logger.info("LLM thread started (model=%s)", self._cfg.model)

        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                reply = await self._process(item)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM processing error")
                reply = f"[Error: {exc}]"

            if item.future and not item.future.done():
                item.future.set_result(reply)

            self._queue.task_done()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    async def _process(self, item: _QueueItem) -> str:
        """Run one full inference cycle, including tool loops."""
        messages = self._build_messages(item.text, item.is_system_event)

        # Check whether we should compact before this inference.
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > COMPACTION_THRESHOLD_CHARS:
            logger.info("Context approaching limit (%d chars) – compacting", total_chars)
            await self._compact_context()
            messages = self._build_messages(item.text, item.is_system_event)

        system_prompt = prompt_assembler.assemble(
            extra_summary=self._compaction_summary
        )

        # Persist the user/system message to DB.
        if not item.is_system_event:
            database.save_message("user", item.text)

        response_text = await self._inference_loop(system_prompt, messages)

        # Persist AI reply.
        database.save_message("assistant", response_text)

        # After responding, check component sizes and trigger summarisation.
        await self._maybe_summarise_components()

        return response_text

    async def _inference_loop(self, system_prompt: str, messages: list[dict]) -> str:
        """
        Repeatedly call the API until there are no more tool_use blocks.
        Returns the final text response.
        """
        while True:
            response = await self._client.messages.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                system=system_prompt,
                tools=tool_module.TOOL_SCHEMAS,
                messages=messages,
            )

            # Collect text and tool_use blocks.
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(block)

            if response.stop_reason == "tool_use" and tool_calls:
                # Execute each tool and feed results back.
                assistant_content = []
                tool_results = []

                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id":    block.id,
                            "name":  block.name,
                            "input": block.input,
                        })
                        result = tool_module.execute_tool(block.name, block.input)
                        logger.debug("Tool %s -> %s", block.name, result[:200])
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     result,
                        })

                messages = messages + [
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user",      "content": tool_results},
                ]
                continue  # next iteration

            # No more tools – return final text.
            return "\n".join(text_parts).strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(self, new_text: str, is_system_event: bool) -> list[dict]:
        """Construct the messages list from DB history + new message."""
        rows = database.load_active_messages()
        messages: list[dict] = []
        for row in rows:
            messages.append({"role": row["role"], "content": row["content"]})

        role = "user"
        prefix = "[SYSTEM EVENT] " if is_system_event else ""
        messages.append({"role": role, "content": f"{prefix}{new_text}"})
        return messages

    async def _compact_context(self) -> None:
        """Summarise old messages, archive them, keep recent N."""
        rows = database.load_active_messages()
        if len(rows) <= COMPACTION_KEEP_RECENT:
            return

        to_summarise = rows[:-COMPACTION_KEEP_RECENT]
        recent = rows[-COMPACTION_KEEP_RECENT:]

        history_text = "\n".join(
            f"{r['role'].upper()}: {r['content']}" for r in to_summarise
        )
        summary_prompt = (
            "Please summarise the following conversation history concisely, "
            "preserving all important facts, decisions, and context:\n\n"
            + history_text
        )

        summary_response = await self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = summary_response.content[0].text.strip()

        # Archive old messages and save summary.
        if to_summarise:
            database.archive_messages(to_summarise[-1]["id"])
        database.save_summary(summary)
        self._compaction_summary = summary

        logger.info(
            "Compacted %d messages into summary (%d chars)",
            len(to_summarise),
            len(summary),
        )
        # Notify user about compaction (best-effort).
        try:
            await self._matrix_send(
                "📦 Context compacted: old messages archived and summarised."
            )
        except Exception:  # noqa: BLE001
            pass

    async def _maybe_summarise_components(self) -> None:
        """Check prompt component sizes and ask AI to summarise if needed."""
        sizes = prompt_assembler.check_component_sizes()
        for component, oversized in sizes.items():
            if not oversized:
                continue
            logger.info("Component '%s' is oversized – requesting AI summarisation", component)
            prompt = (
                f"The {component} section of your memory is getting large. "
                f"Please read its current content via read_file, condense and "
                f"prioritise it, and update it using the appropriate tool."
            )
            try:
                await self.enqueue_system_event(prompt)
                await self._matrix_send(
                    f"ℹ️ Auto-summarising {component} (size limit reached)."
                )
            except Exception:  # noqa: BLE001
                logger.exception("Could not request summarisation for %s", component)
