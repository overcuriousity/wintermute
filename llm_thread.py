"""
LLM Inference Thread

Owns the conversation history and all interactions with any OpenAI-compatible
API endpoint (Ollama, vLLM, LM Studio, OpenAI, etc.).

Receives user messages via an asyncio Queue, runs inference (including
multi-step tool-use loops), and delivers responses back through reply Futures.

Public API used by other modules
---------------------------------
  LLMThread.enqueue_user_message(text)  -> str  (awaitable)
  LLMThread.enqueue_system_event(text)  -> None (fire-and-forget)
  LLMThread.reset_session()
  LLMThread.force_compact()
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

import database
import prompt_assembler
import tools as tool_module

logger = logging.getLogger(__name__)

# Approximate character count that triggers context compaction.
COMPACTION_THRESHOLD_CHARS = 150_000
# Keep the last N messages untouched during compaction.
COMPACTION_KEEP_RECENT = 10


@dataclass
class LLMConfig:
    api_key: str
    base_url: str           # e.g. http://localhost:11434/v1  or  https://api.openai.com/v1
    model: str
    max_tokens: int = 4096
    compaction_model: Optional[str] = None  # cheaper model for summarisation; falls back to model


@dataclass
class _QueueItem:
    text: str
    is_system_event: bool = False
    future: Optional[asyncio.Future] = field(default=None, compare=False)


class LLMThread:
    """Runs as an asyncio task within the shared event loop."""

    def __init__(self, config: LLMConfig, matrix_send_fn) -> None:
        self._cfg = config
        self._matrix_send = matrix_send_fn  # async callable(text: str)
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
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
        database.clear_active_messages()
        self._compaction_summary = None
        logger.info("Session reset by /new command")

    async def force_compact(self) -> None:
        await self._compact_context()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        self._compaction_summary = database.load_latest_summary()
        logger.info("LLM thread started (endpoint=%s model=%s)", self._cfg.base_url, self._cfg.model)

        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                reply = await self._process(item)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM processing error")
                reply = f"[Error during inference: {exc}]"

            if item.future and not item.future.done():
                item.future.set_result(reply)

            self._queue.task_done()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    async def _process(self, item: _QueueItem) -> str:
        messages = self._build_messages(item.text, item.is_system_event)

        total_chars = sum(
            len(m["content"]) if isinstance(m["content"], str) else
            sum(p.get("text", "") and len(p["text"]) for p in m["content"] if isinstance(p, dict))
            for m in messages
        )
        if total_chars > COMPACTION_THRESHOLD_CHARS:
            logger.info("Context at %d chars – compacting before inference", total_chars)
            await self._compact_context()
            messages = self._build_messages(item.text, item.is_system_event)

        if not item.is_system_event:
            database.save_message("user", item.text)

        system_prompt = prompt_assembler.assemble(extra_summary=self._compaction_summary)
        response_text = await self._inference_loop(system_prompt, messages)

        database.save_message("assistant", response_text)
        await self._maybe_summarise_components()
        return response_text

    # ------------------------------------------------------------------
    # OpenAI inference loop (handles tool-use rounds)
    # ------------------------------------------------------------------

    async def _inference_loop(self, system_prompt: str, messages: list[dict]) -> str:
        """
        Repeatedly call the API until finish_reason is not 'tool_calls'.
        The system prompt is prepended as a role=system message each call
        (not stored in the DB so it stays fresh on every inference).
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        while True:
            response = await self._client.chat.completions.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                tools=tool_module.TOOL_SCHEMAS,
                tool_choice="auto",
                messages=full_messages,
            )

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                # Append the assistant's tool-call message.
                full_messages.append(choice.message)

                # Execute each tool and collect results.
                for tc in choice.message.tool_calls:
                    try:
                        inputs = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        inputs = {}
                    result = tool_module.execute_tool(tc.function.name, inputs)
                    logger.debug("Tool %s -> %s", tc.function.name, result[:200])
                    full_messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })
                continue  # next round

            # Terminal response.
            return (choice.message.content or "").strip()

    # ------------------------------------------------------------------
    # Context compaction
    # ------------------------------------------------------------------

    async def _compact_context(self) -> None:
        rows = database.load_active_messages()
        if len(rows) <= COMPACTION_KEEP_RECENT:
            return

        to_summarise = rows[:-COMPACTION_KEEP_RECENT]

        history_text = "\n".join(
            f"{r['role'].upper()}: {r['content']}" for r in to_summarise
        )
        summary_prompt = (
            "Summarise the following conversation history concisely, "
            "preserving all important facts, decisions, and context:\n\n"
            + history_text
        )

        compact_model = self._cfg.compaction_model or self._cfg.model
        summary_response = await self._client.chat.completions.create(
            model=compact_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = (summary_response.choices[0].message.content or "").strip()

        if to_summarise:
            database.archive_messages(to_summarise[-1]["id"])
        database.save_summary(summary)
        self._compaction_summary = summary

        logger.info("Compacted %d messages into summary (%d chars)", len(to_summarise), len(summary))
        try:
            await self._matrix_send("📦 Context compacted: old messages archived and summarised.")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Component size monitoring
    # ------------------------------------------------------------------

    async def _maybe_summarise_components(self) -> None:
        sizes = prompt_assembler.check_component_sizes()
        for component, oversized in sizes.items():
            if not oversized:
                continue
            logger.info("Component '%s' oversized – requesting AI summarisation", component)
            prompt = (
                f"The {component} section of your memory has grown large. "
                f"Read its current content via read_file, condense and prioritise it, "
                f"then update it using the appropriate tool."
            )
            try:
                await self.enqueue_system_event(prompt)
                await self._matrix_send(f"ℹ️ Auto-summarising {component} (size limit reached).")
            except Exception:  # noqa: BLE001
                logger.exception("Could not request summarisation for %s", component)

    # ------------------------------------------------------------------
    # Message list construction
    # ------------------------------------------------------------------

    def _build_messages(self, new_text: str, is_system_event: bool) -> list[dict]:
        rows = database.load_active_messages()
        messages = [{"role": r["role"], "content": r["content"]} for r in rows]
        prefix = "[SYSTEM EVENT] " if is_system_event else ""
        messages.append({"role": "user", "content": f"{prefix}{new_text}"})
        return messages
