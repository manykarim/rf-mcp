"""Simple asynchronous event bus used to synchronize frontend updates."""

from __future__ import annotations

import asyncio
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Deque, Dict, Set


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

        self._subscribers: Set[asyncio.Queue[FrontendEvent]] = set()
        self._replay: Deque[FrontendEvent] = deque(maxlen=replay_buffer)
        self._lock = asyncio.Lock()

    async def _deliver(self, event: FrontendEvent) -> None:
        for queue in list(self._subscribers):
            if queue.full():
                # Drop oldest item to make room
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

    async def publish(self, event: FrontendEvent) -> None:
        """Publish an event to all subscribers."""

        self._replay.append(event)
        await self._deliver(event)

    def publish_sync(self, event: FrontendEvent) -> None:
        """Synchronously publish an event, scheduling delivery on the running loop."""

        self._replay.append(event)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; events will be replayed for new subscribers later.
            return

        loop.create_task(self._deliver(event))

    async def subscribe(self) -> AsyncIterator[FrontendEvent]:
        """Yield events for a subscriber; includes replay buffer at subscription time."""

        queue: asyncio.Queue[FrontendEvent] = asyncio.Queue(maxsize=512)
        async with self._lock:
            for event in self._replay:
                queue.put_nowait(event)
            self._subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            async with self._lock:
                self._subscribers.discard(queue)

    async def recent_events(self, limit: int = 50) -> list[FrontendEvent]:
        async with self._lock:
            if limit <= 0:
                return list(self._replay)
            return list(self._replay)[-limit:]


# Global event bus instance shared across the server.
event_bus = EventBus()
