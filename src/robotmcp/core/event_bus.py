"""Simple asynchronous event bus used to synchronize frontend updates."""

from __future__ import annotations

import asyncio
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Deque, Dict


@dataclass(slots=True)
class FrontendEvent:
    """Structured event emitted for frontend consumption."""

    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    step_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    """Fan-out event bus with replay buffer for late subscribers."""

    def __init__(self, replay_buffer: int | None = None):
        if replay_buffer is None:
            try:
                replay_buffer = int(
                    os.environ.get("ROBOTMCP_FRONTEND_EVENT_BUFFER", "2048")
                )
            except ValueError:
                replay_buffer = 2048

        # Each subscriber queue is mapped to the event loop that owns it, so a
        # publisher on a *different* loop (e.g. the MCP loop) can deliver to it
        # safely via call_soon_threadsafe rather than a cross-loop put_nowait
        # (which is a silent no-op — the getter's future never wakes).
        self._subscribers: Dict[asyncio.Queue[FrontendEvent], asyncio.AbstractEventLoop] = {}
        self._replay: Deque[FrontendEvent] = deque(maxlen=replay_buffer)
        # threading.Lock (not asyncio.Lock) because publish/subscribe/recent may
        # run on different loops/threads; the critical sections are tiny and sync.
        self._lock = threading.Lock()

    @staticmethod
    def _put(queue: "asyncio.Queue[FrontendEvent]", event: FrontendEvent) -> None:
        """Enqueue on the queue's OWNER loop (drop oldest when full)."""
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def _fanout(self, event: FrontendEvent) -> None:
        """Deliver to every subscriber on its own loop (cross-loop safe)."""
        with self._lock:
            targets = list(self._subscribers.items())
        for queue, loop in targets:
            try:
                loop.call_soon_threadsafe(self._put, queue, event)
            except RuntimeError:
                # Owner loop is closed -> subscriber is dead; drop it.
                with self._lock:
                    self._subscribers.pop(queue, None)

    async def publish(self, event: FrontendEvent) -> None:
        """Publish an event to all subscribers."""

        with self._lock:
            self._replay.append(event)
        self._fanout(event)

    def publish_sync(self, event: FrontendEvent) -> None:
        """Publish an event from any thread/loop (or none). Delivery is scheduled
        on each subscriber's own loop, so it works regardless of the caller's loop."""

        with self._lock:
            self._replay.append(event)
        self._fanout(event)

    async def subscribe(self) -> AsyncIterator[FrontendEvent]:
        """Yield events for a subscriber; includes replay buffer at subscription time."""

        queue: asyncio.Queue[FrontendEvent] = asyncio.Queue(maxsize=512)
        loop = asyncio.get_running_loop()
        with self._lock:
            for event in self._replay:
                queue.put_nowait(event)
            self._subscribers[queue] = loop
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            with self._lock:
                self._subscribers.pop(queue, None)

    async def recent_events(self, limit: int = 50) -> list[FrontendEvent]:
        with self._lock:
            if limit <= 0:
                return list(self._replay)
            return list(self._replay)[-limit:]


# Global event bus instance shared across the server.
event_bus = EventBus()
