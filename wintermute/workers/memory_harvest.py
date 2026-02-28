"""
Memory Harvest – Periodic Conversation Mining

Periodically scans user-facing threads for unharvested messages and spawns
sub-session workers to extract personal facts, preferences, and interaction
patterns into MEMORIES.txt.

Trigger logic: a thread is eligible for harvest when EITHER:
  - It has accumulated >= message_threshold new user messages since last harvest
  - It has been inactive for >= inactivity_timeout_minutes after at least
    _INACTIVITY_MIN_MESSAGES new user messages

Only user-facing threads are harvested (Matrix room_ids, web_* IDs, "default").
Sub-session threads (sub_*) are excluded.  Results are fire-and-forget — no
messages are delivered to chat.  Visibility is through logs and interaction_log
(debug panel).
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from wintermute.infra import database
from wintermute.infra import prompt_loader

if TYPE_CHECKING:
    from wintermute.core.types import BackendPool
    from wintermute.core.sub_session import SubSessionManager
    from wintermute.infra.event_bus import EventBus

logger = logging.getLogger(__name__)

# Minimum new user messages required before the inactivity timer can trigger.
_INACTIVITY_MIN_MESSAGES = 5

# Maximum total characters in the conversation blob sent to the worker (default; overridable via config).
# ~15k tokens — leaves headroom for the prompt and tool schemas.
_MAX_BLOB_CHARS = 60_000


@dataclass
class MemoryHarvestConfig:
    enabled: bool = True
    message_threshold: int = 20
    inactivity_timeout_minutes: int = 15
    max_message_chars: int = 2000
    max_blob_chars: int = _MAX_BLOB_CHARS
    poll_interval_seconds: int = 60


class MemoryHarvestLoop:
    """Asyncio task that periodically mines conversations for memories."""

    def __init__(
        self,
        config: MemoryHarvestConfig,
        sub_session_manager: Optional[SubSessionManager] = None,
        pool: Optional[BackendPool] = None,
        event_bus: "Optional[EventBus]" = None,
    ) -> None:
        self._cfg = config
        self._sub_sessions = sub_session_manager
        self._pool = pool
        self._event_bus = event_bus
        self._running = False
        # Per-thread: last harvested message id — restored from DB on startup
        self._last_harvested_id: dict[str, int] = database.load_harvest_state()
        # Threads currently being harvested (prevent overlapping runs)
        self._in_flight: set[str] = set()
        # Per-thread message counter (incremented by event bus, reset on harvest)
        self._msg_counts: dict[str, int] = {}
        self._event_bus_subs: list[str] = []
        # Pending harvest check triggered by event bus
        self._check_event = asyncio.Event()
        # Strong references to background tasks to prevent GC
        self._background_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Public tuning API (used by self-model auto-tuning)
    # ------------------------------------------------------------------

    @property
    def message_threshold(self) -> int:
        """Current message count threshold before harvest triggers."""
        return self._cfg.message_threshold

    @message_threshold.setter
    def message_threshold(self, value: int) -> None:
        self._cfg.message_threshold = max(1, value)

    def has_backlog(self) -> bool:
        """Return True if any thread has messages at/above threshold while a harvest is in flight."""
        return (
            any(count >= self._cfg.message_threshold for count in self._msg_counts.values())
            and len(self._in_flight) > 0
        )

    async def _on_message_received(self, event) -> None:
        """Increment per-thread counter and trigger immediate check."""
        thread_id = event.data.get("thread_id", "")
        if thread_id.startswith("sub_"):
            return
        self._msg_counts[thread_id] = self._msg_counts.get(thread_id, 0) + 1
        if self._msg_counts.get(thread_id, 0) >= self._cfg.message_threshold:
            self._check_event.set()

    async def _on_harvest_session_completed(self, event) -> None:
        """Direct callback when a harvest sub-session finishes."""
        # The _await_harvest coroutine handles the actual logic;
        # this just logs for visibility.
        session_id = event.data.get("session_id", "")
        if session_id:
            logger.debug("Memory harvest: notified of sub_session.completed %s", session_id)

    async def run(self) -> None:
        self._running = True
        # Subscribe to events for near-immediate harvest triggering.
        if self._event_bus:
            sub_id = self._event_bus.subscribe("message.received", self._on_message_received)
            self._event_bus_subs.append(sub_id)
            sub_id = self._event_bus.subscribe("sub_session.completed", self._on_harvest_session_completed)
            self._event_bus_subs.append(sub_id)
        # With event bus, increase fallback poll to 300s (events trigger checks sooner).
        fallback_poll = 300 if self._event_bus else self._cfg.poll_interval_seconds
        logger.info(
            "Memory harvest loop started (threshold=%d msgs, inactivity=%d min, poll=%ds)",
            self._cfg.message_threshold,
            self._cfg.inactivity_timeout_minutes,
            fallback_poll,
        )
        while self._running:
            # Wait for either the fallback poll or an event-driven trigger.
            try:
                await asyncio.wait_for(self._check_event.wait(), timeout=fallback_poll)
            except asyncio.TimeoutError:
                pass
            self._check_event.clear()
            if not self._running:
                break
            try:
                await self._check_threads()
            except Exception:  # noqa: BLE001
                logger.exception("Memory harvest: error during thread check")

    def stop(self) -> None:
        self._running = False
        self._check_event.set()  # unblock the wait
        if self._event_bus:
            for sub_id in self._event_bus_subs:
                self._event_bus.unsubscribe(sub_id)
            self._event_bus_subs.clear()
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _check_threads(self) -> None:
        """Enumerate active threads and spawn harvests where eligible."""
        if self._sub_sessions is None:
            return

        thread_ids = await database.async_call(database.get_active_thread_ids)
        for thread_id in thread_ids:
            # Skip sub-session threads (in-memory only, defensive check).
            if thread_id.startswith("sub_"):
                continue
            # Skip threads already being harvested.
            if thread_id in self._in_flight:
                continue

            messages = await database.async_call(database.load_active_messages, thread_id)
            if self._should_harvest(thread_id, messages):
                await self._spawn_harvest(thread_id, messages)

    def _should_harvest(self, thread_id: str, messages: list[dict]) -> bool:
        """Decide whether a thread is ready for memory extraction."""
        last_id = self._last_harvested_id.get(thread_id, 0)
        new_user_msgs = [
            m for m in messages
            if m["id"] > last_id and m["role"] == "user"
        ]

        if not new_user_msgs:
            return False

        # Condition 1: message count threshold reached.
        if len(new_user_msgs) >= self._cfg.message_threshold:
            return True

        # Condition 2: inactivity timeout (with minimum message floor).
        if len(new_user_msgs) >= _INACTIVITY_MIN_MESSAGES:
            # Use latest message of any role (not just user) as activity marker.
            new_msgs = [m for m in messages if m["id"] > last_id]
            latest_ts = max(m["timestamp"] for m in new_msgs)
            idle_seconds = _time.time() - latest_ts
            if idle_seconds >= self._cfg.inactivity_timeout_minutes * 60:
                return True

        return False

    # ------------------------------------------------------------------
    # Conversation preparation
    # ------------------------------------------------------------------

    def _prepare_conversation_blob(self, messages: list[dict],
                                    since_id: int) -> str:
        """Build a formatted transcript from messages newer than since_id."""
        lines: list[str] = []
        for msg in messages:
            if msg["id"] <= since_id:
                continue
            role = msg["role"]
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            # Only user and assistant messages.
            if role not in ("user", "assistant"):
                continue
            # Exclude system events and sub-session results.
            if content.startswith("[SYSTEM EVENT]") or content.startswith("[SUB-SESSION"):
                continue
            # Mid-truncate very long messages (e.g. pasted log files).
            content = self._mid_truncate(content, self._cfg.max_message_chars)
            label = "USER" if role == "user" else "ASSISTANT"
            lines.append(f"{label}: {content}")

        blob = "\n".join(lines)
        max_chars = self._cfg.max_blob_chars
        if len(blob) > max_chars:
            blob = blob[:max_chars] + "\n[... transcript truncated ...]"
        return blob

    @staticmethod
    def _mid_truncate(text: str, max_chars: int) -> str:
        """Truncate from the middle, preserving beginning and end."""
        if len(text) <= max_chars:
            return text
        keep = max_chars // 2
        return text[:keep] + "\n[... truncated ...]\n" + text[-keep:]

    # ------------------------------------------------------------------
    # Harvest spawning
    # ------------------------------------------------------------------

    async def _spawn_harvest(self, thread_id: str,
                              messages: list[dict]) -> None:
        """Spawn a fire-and-forget sub-session to extract memories."""
        last_id = self._last_harvested_id.get(thread_id, 0)
        blob = self._prepare_conversation_blob(messages, last_id)
        if not blob.strip():
            return

        prompt = prompt_loader.load("MEMORY_HARVEST_PROMPT.txt",
                                    transcript=blob)

        max_id = max(m["id"] for m in messages)
        new_count = sum(
            1 for m in messages
            if m["id"] > last_id and m["role"] == "user"
        )

        self._in_flight.add(thread_id)
        # Reset event-bus message counter for this thread.
        self._msg_counts.pop(thread_id, None)

        if self._event_bus:
            self._event_bus.emit("harvest.started", thread_id=thread_id)

        session_id = self._sub_sessions.spawn(
            objective=prompt,
            parent_thread_id=None,      # fire-and-forget: no chat delivery
            system_prompt_mode="none",   # all instructions in the objective
            tool_names=["append_memory", "read_file"],
            timeout=600, # generous timeout for slow workers and large conversations
            pool=self._pool,
            max_rounds=5,               # hard cap: prevent runaway tool-call loops
            skip_tp_on_exit=True,       # don't let TP override "nothing to extract"
        )

        logger.info(
            "Memory harvest spawned %s for thread %s (%d new user msgs, up to id=%d)",
            session_id, thread_id, new_count, max_id,
        )

        # Monitor completion in background — only commit the harvested ID range
        # and write the interaction_log entry after the sub-session succeeds.
        task = asyncio.create_task(
            self._await_harvest(session_id, thread_id, max_id, new_count)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _await_harvest(self, session_id: str, thread_id: str,
                              max_id: int, new_count: int) -> None:
        """Poll sub-session status; commit harvest state only on success."""
        try:
            # Poll until terminal (completed/failed/timeout).
            while True:
                await asyncio.sleep(5)
                state = self._sub_sessions._states.get(session_id)
                if state is None:
                    logger.warning("Memory harvest %s: state vanished", session_id)
                    break
                if state.status in ("completed", "failed", "timeout"):
                    break

            status = state.status if state else "unknown"

            if status == "completed":
                self._last_harvested_id[thread_id] = max_id
                if self._event_bus:
                    self._event_bus.emit("harvest.completed", thread_id=thread_id,
                                         session_id=session_id)
                logger.info(
                    "Memory harvest %s completed — committed max_id=%d for thread %s",
                    session_id, max_id, thread_id,
                )
            else:
                logger.warning(
                    "Memory harvest %s ended with status=%s for thread %s — "
                    "ID range NOT committed (will retry next cycle)",
                    session_id, status, thread_id,
                )

            try:
                await database.async_call(
                    database.save_interaction_log,
                    _time.time(),
                    "memory_harvest",
                    f"harvest:{thread_id}",
                    "sub_sessions",
                    f"thread={thread_id} msgs={new_count} max_id={max_id}",
                    f"{session_id} {status}",
                    "ok" if status == "completed" else status,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                pass
        except asyncio.CancelledError:
            logger.debug("Memory harvest monitor for %s cancelled during shutdown", session_id)
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Memory harvest monitor for %s failed", session_id)
        finally:
            self._in_flight.discard(thread_id)
