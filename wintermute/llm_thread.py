"""
LLM Inference Thread

Owns the conversation history and all interactions with any OpenAI-compatible
API endpoint (Ollama, vLLM, LM Studio, OpenAI, etc.).

Receives user messages via an asyncio Queue, runs inference (including
multi-step tool-use loops), and delivers responses back through reply Futures.

Public API used by other modules
---------------------------------
  LLMThread.enqueue_user_message(text, thread_id)  -> str  (awaitable)
  LLMThread.enqueue_system_event(text, thread_id)  -> None (fire-and-forget)
  LLMThread.reset_session(thread_id)
  LLMThread.force_compact(thread_id)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

from wintermute import database
from wintermute import prompt_assembler
from wintermute import tools as tool_module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wintermute.sub_session import SubSessionManager

logger = logging.getLogger(__name__)

# Keep the last N messages untouched during compaction.
COMPACTION_KEEP_RECENT = 10


def _count_tokens(text: str, model: str) -> int:
    """
    Estimate token count using tiktoken.
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


@dataclass
class LLMConfig:
    api_key: str
    base_url: str           # e.g. http://localhost:11434/v1  or  https://api.openai.com/v1
    model: str
    context_size: int       # total token window the model supports (e.g. 65536)
    max_tokens: int = 4096  # maximum tokens in a single response
    compaction_model: Optional[str] = None  # cheaper model for summarisation; falls back to model


@dataclass
class _QueueItem:
    text: str
    thread_id: str = "default"
    is_system_event: bool = False
    future: Optional[asyncio.Future] = field(default=None, compare=False)


class LLMThread:
    """Runs as an asyncio task within the shared event loop."""

    def __init__(self, config: LLMConfig, broadcast_fn,
                 sub_session_manager: "Optional[SubSessionManager]" = None) -> None:
        self._cfg = config
        self._broadcast = broadcast_fn  # async callable(text: str, thread_id: str | None)
        self._sub_sessions = sub_session_manager  # set post-init via inject_sub_session_manager
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self._running = False
        # Per-thread compaction summaries: thread_id -> summary text
        self._compaction_summaries: dict[str, Optional[str]] = {}

    def inject_sub_session_manager(self, manager: "SubSessionManager") -> None:
        """Called after construction once SubSessionManager is built."""
        self._sub_sessions = manager

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def enqueue_user_message(self, text: str, thread_id: str = "default") -> str:
        """Submit a user message and await the AI reply."""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_QueueItem(text=text, thread_id=thread_id, future=fut))
        return await fut

    async def enqueue_system_event(self, text: str, thread_id: str = "default") -> None:
        """Submit an autonomous system event (heartbeat, reminder, etc.)."""
        await self._queue.put(_QueueItem(text=text, thread_id=thread_id, is_system_event=True))

    async def enqueue_system_event_with_reply(self, text: str,
                                              thread_id: str = "default") -> str:
        """Submit a system event and await the AI reply.

        Like enqueue_system_event but returns the reply text. The prompt is
        NOT saved to the DB as a user message; only the assistant response is.
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_QueueItem(text=text, thread_id=thread_id,
                                         is_system_event=True, future=fut))
        return await fut

    async def reset_session(self, thread_id: str = "default") -> None:
        database.clear_active_messages(thread_id)
        self._compaction_summaries.pop(thread_id, None)
        if self._sub_sessions:
            n = self._sub_sessions.cancel_for_thread(thread_id)
            if n:
                logger.info("Cancelled %d sub-session(s) for reset thread %s", n, thread_id)
        logger.info("Session reset for thread %s", thread_id)

    async def force_compact(self, thread_id: str = "default") -> None:
        await self._compact_context(thread_id)

    def get_compaction_summary(self, thread_id: str = "default") -> Optional[str]:
        """Return the current in-memory compaction summary for a thread, or None."""
        return self._compaction_summaries.get(thread_id)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        # Load summaries for all known threads
        for tid in database.get_active_thread_ids():
            summary = database.load_latest_summary(tid)
            if summary:
                self._compaction_summaries[tid] = summary
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
            elif item.is_system_event and not item.future:
                # System events without a future (sub-session results,
                # reminders, /pulse commands) have no caller waiting for
                # the reply.  Broadcast the LLM's response directly so
                # it reaches the user.
                text_to_send = reply or item.text  # fallback to raw event
                try:
                    await self._broadcast(text_to_send, item.thread_id)
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to broadcast system-event reply for thread %s",
                                     item.thread_id)

            self._queue.task_done()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    async def _process(self, item: _QueueItem) -> str:
        thread_id = item.thread_id
        messages = self._build_messages(item.text, item.is_system_event, thread_id)

        # Assemble system prompt first so we can measure its real token cost.
        summary = self._compaction_summaries.get(thread_id)
        system_prompt = prompt_assembler.assemble(extra_summary=summary)
        overhead_tokens = (
            _count_tokens(system_prompt, self._cfg.model)
            + _count_tokens(json.dumps(tool_module.TOOL_SCHEMAS), self._cfg.model)
        )

        history_tokens = sum(
            _count_tokens(m["content"] if isinstance(m["content"], str) else
                          " ".join(p.get("text", "") for p in m["content"] if isinstance(p, dict)),
                          self._cfg.model)
            for m in messages
        )
        compaction_threshold = self._cfg.context_size - self._cfg.max_tokens - overhead_tokens
        if history_tokens > compaction_threshold:
            logger.info(
                "History at %d tokens (overhead %d, threshold %d) – compacting before inference (thread=%s)",
                history_tokens, overhead_tokens, compaction_threshold, thread_id,
            )
            await self._compact_context(thread_id)
            messages = self._build_messages(item.text, item.is_system_event, thread_id)
            # Reassemble with the updated compaction summary.
            summary = self._compaction_summaries.get(thread_id)
            system_prompt = prompt_assembler.assemble(extra_summary=summary)

        is_sub_session_result = item.is_system_event and "[SUB-SESSION " in item.text
        if not item.is_system_event:
            database.save_message("user", item.text, thread_id)
        elif is_sub_session_result:
            database.save_message("user", f"[SYSTEM EVENT] {item.text}", thread_id)

        response_text = await self._inference_loop(
            system_prompt, messages, thread_id,
            disable_tools=is_sub_session_result,
        )

        database.save_message("assistant", response_text, thread_id)
        await self._maybe_summarise_components(thread_id)
        return response_text

    # ------------------------------------------------------------------
    # OpenAI inference loop (handles tool-use rounds)
    # ------------------------------------------------------------------

    async def _inference_loop(self, system_prompt: str, messages: list[dict],
                              thread_id: str = "default",
                              disable_tools: bool = False) -> str:
        """
        Repeatedly call the API until finish_reason is not 'tool_calls'.
        The system prompt is prepended as a role=system message each call
        (not stored in the DB so it stays fresh on every inference).
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        while True:
            if disable_tools:
                response = await self._client.chat.completions.create(
                    model=self._cfg.model,
                    max_tokens=self._cfg.max_tokens,
                    messages=full_messages,
                )
            else:
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
                    result = tool_module.execute_tool(tc.function.name, inputs,
                                                     thread_id=thread_id,
                                                     nesting_depth=0)
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

    async def _compact_context(self, thread_id: str = "default") -> None:
        rows = database.load_active_messages(thread_id)
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
            database.archive_messages(to_summarise[-1]["id"], thread_id)
        database.save_summary(summary, thread_id)
        self._compaction_summaries[thread_id] = summary

        logger.info("Compacted %d messages into summary (%d chars) for thread %s",
                     len(to_summarise), len(summary), thread_id)
        try:
            await self._broadcast(
                "\U0001f4e6 Context compacted: old messages archived and summarised.",
                thread_id,
            )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Component size monitoring
    # ------------------------------------------------------------------

    async def _maybe_summarise_components(self, thread_id: str = "default") -> None:
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
                await self.enqueue_system_event(prompt, thread_id)
                await self._broadcast(
                    f"\u2139\ufe0f Auto-summarising {component} (size limit reached).",
                    thread_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Could not request summarisation for %s", component)

    # ------------------------------------------------------------------
    # Message list construction
    # ------------------------------------------------------------------

    def _build_messages(self, new_text: str, is_system_event: bool,
                        thread_id: str = "default") -> list[dict]:
        rows = database.load_active_messages(thread_id)
        messages = [{"role": r["role"], "content": r["content"]} for r in rows]
        prefix = "[SYSTEM EVENT] " if is_system_event else ""
        messages.append({"role": "user", "content": f"{prefix}{new_text}"})
        return messages
