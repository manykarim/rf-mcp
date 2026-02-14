"""Response Optimization Config Repository."""
from __future__ import annotations

from typing import Dict, Optional, Protocol

from .aggregates import ResponseOptimizationConfig


class ResponseOptimizationConfigRepository(Protocol):
    """Protocol for config persistence."""

    def get(self, session_id: str) -> Optional[ResponseOptimizationConfig]: ...
    def save(self, config: ResponseOptimizationConfig) -> None: ...


class InMemoryConfigRepository:
    """In-memory config store keyed by session_id."""

    def __init__(self) -> None:
        self._configs: Dict[str, ResponseOptimizationConfig] = {}

    def get(self, session_id: str) -> Optional[ResponseOptimizationConfig]:
        return self._configs.get(session_id)

    def save(self, config: ResponseOptimizationConfig) -> None:
        self._configs[config.session_id] = config

    def remove(self, session_id: str) -> None:
        self._configs.pop(session_id, None)
