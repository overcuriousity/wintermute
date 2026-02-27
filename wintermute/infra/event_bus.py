"""
Event Bus — Async-native publish/subscribe infrastructure.

Fire-and-forget event emission with subscriber error isolation.
Includes a ring-buffer history for Phase 4 reflection queries and
optional per-subscriber debounce to coalesce rapid-fire events.
"""

import asyncio
import logging
import threading
import time as _time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


@dataclass
class Event:
    event_type: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=_time.time)


class EventBus:
    """Async event bus with history and debounce support."""

    def __init__(self, history_size: int = 1000) -> None:
        # event_type -> {sub_id: (callback, debounce_ms)}
        self._subs: dict[str, dict[str, tuple[Callable, int]]] = {}
        self._history: deque[Event] = deque(maxlen=history_size)
        # Debounce timers: sub_id -> asyncio.TimerHandle
        self._debounce_timers: dict[str, asyncio.TimerHandle] = {}
        # Debounce pending events: sub_id -> Event (latest)
        self._debounce_pending: dict[str, Event] = {}
        # Guard _subs mutations from concurrent emit/subscribe/unsubscribe.
        self._lock = threading.Lock()
        # Capture the event loop at construction time so that emit() works
        # from thread-pool workers (run_in_executor) where there is no
        # running loop.  Falls back to None if constructed outside a loop.
        try:
            self._loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def emit(self, event_type: str, **data: Any) -> None:
        """Fire-and-forget event emission. Never raises."""
        event = Event(event_type=event_type, data=data)
        self._history.append(event)
        with self._lock:
            subs = dict(self._subs.get(event_type, {}))
        if not subs:
            return
        for sub_id, (callback, debounce_ms) in subs.items():
            if debounce_ms > 0:
                self._debounce_emit(sub_id, callback, event, debounce_ms)
            else:
                self._fire(sub_id, callback, event)

    def _fire(self, sub_id: str, callback: Callable, event: Event) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._safe_call(sub_id, callback, event))
        except RuntimeError:
            # Called from a thread-pool worker (run_in_executor) — schedule
            # the coroutine on the main event loop captured at init time.
            loop = self._loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.create_task, self._safe_call(sub_id, callback, event))
            else:
                logger.warning("EventBus: event dropped for %s — no usable event loop", sub_id)

    async def _safe_call(self, sub_id: str, callback: Callable, event: Event) -> None:
        try:
            result = callback(event)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        except Exception:
            logger.exception("EventBus: subscriber %s raised on %s", sub_id, event.event_type)

    def _debounce_emit(self, sub_id: str, callback: Callable,
                       event: Event, debounce_ms: int) -> None:
        # All debounce state mutations (pending dict, timer cancel+create) must
        # happen on the event-loop thread to avoid races when emit() is called
        # from run_in_executor workers.  Wrapping the entire operation in a
        # single call_soon_threadsafe also prevents the double-schedule race
        # that would occur if two worker threads both see no existing timer and
        # enqueue two independent _schedule_later callbacks.
        def _do_debounce():
            self._debounce_pending[sub_id] = event
            existing = self._debounce_timers.pop(sub_id, None)
            if existing:
                existing.cancel()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = self._loop
            if loop is not None and loop.is_running():
                handle = loop.call_later(
                    debounce_ms / 1000.0,
                    lambda: self._flush_debounce(sub_id, callback),
                )
                self._debounce_timers[sub_id] = handle

        try:
            asyncio.get_running_loop()
            # We're already on the event-loop thread — execute directly.
            _do_debounce()
        except RuntimeError:
            # Thread-pool context — serialize the entire operation onto the
            # loop thread so state mutations and timer scheduling are atomic.
            loop = self._loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(_do_debounce)

    def _flush_debounce(self, sub_id: str, callback: Callable) -> None:
        self._debounce_timers.pop(sub_id, None)
        event = self._debounce_pending.pop(sub_id, None)
        if event:
            self._fire(sub_id, callback, event)

    def subscribe(self, event_type: str, callback: Callable,
                  debounce_ms: int = 0) -> str:
        """Subscribe to an event type. Returns subscription ID."""
        sub_id = f"sub_{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._subs.setdefault(event_type, {})[sub_id] = (callback, debounce_ms)
        logger.debug("EventBus: %s subscribed to %s (debounce=%dms)", sub_id, event_type, debounce_ms)
        return sub_id

    def unsubscribe(self, sub_id: str) -> None:
        """Remove a subscription by ID."""
        with self._lock:
            for event_type, subs in self._subs.items():
                if sub_id in subs:
                    del subs[sub_id]
                    # Clean up any pending debounce
                    handle = self._debounce_timers.pop(sub_id, None)
                    if handle:
                        handle.cancel()
                    self._debounce_pending.pop(sub_id, None)
                    logger.debug("EventBus: %s unsubscribed from %s", sub_id, event_type)
                    return

    def history(self, event_type: Optional[str] = None,
                since: Optional[float] = None,
                limit: int = 100) -> list[Event]:
        """Query event history with optional filters."""
        results = []
        for event in reversed(self._history):
            if event_type and event.event_type != event_type:
                continue
            if since and event.timestamp < since:
                continue
            results.append(event)
            if len(results) >= limit:
                break
        return results
