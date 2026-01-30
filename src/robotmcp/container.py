"""Dependency Injection Container for rf-mcp DDD domains.

This container wires together the various DDD bounded contexts:
- Snapshot Context: ARIA tree capture and token optimization
- Element Registry Context: Ref-to-locator mapping
- Action Context: Pre-validation and response filtering
- Timeout Context: Dual timeout management

Usage:
    from robotmcp.container import get_container

    container = get_container()
    page_service = container.get_page_source_service()
    timeout_service = container.get_timeout_service()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from robotmcp.domains.timeout import TimeoutService, TimeoutPolicy
    from robotmcp.domains.action import PreValidator
    from robotmcp.domains.element_registry import ElementRegistry
    from robotmcp.domains.snapshot import PageSnapshot
    from robotmcp.optimization import (
        PatternStore,
        PerformanceMetricsCollector,
        PageAnalyzer,
    )

logger = logging.getLogger(__name__)

# Singleton container instance
_container: Optional["ServiceContainer"] = None


@dataclass
class ServiceContainer:
    """Simple dependency injection container for DDD services.

    Manages the lifecycle and dependencies of domain services,
    providing a centralized way to access them.

    Attributes:
        pattern_store: Shared pattern storage for learned optimizations
        performance_collector: Centralized performance metrics
        timeout_service: Timeout management service
        pre_validator: Pre-validation service for actions
        page_analyzer: Page complexity analyzer

    Session-specific services (keyed by session_id):
        - Element registries
        - Timeout policies
        - Snapshot caches
    """

    # Shared services (singletons)
    _pattern_store: Optional["PatternStore"] = field(default=None, repr=False)
    _performance_collector: Optional["PerformanceMetricsCollector"] = field(
        default=None, repr=False
    )
    _timeout_service: Optional["TimeoutService"] = field(default=None, repr=False)
    _pre_validator: Optional["PreValidator"] = field(default=None, repr=False)
    _page_analyzer: Optional["PageAnalyzer"] = field(default=None, repr=False)

    # Session-scoped registries
    _element_registries: Dict[str, "ElementRegistry"] = field(
        default_factory=dict, repr=False
    )
    _timeout_policies: Dict[str, "TimeoutPolicy"] = field(
        default_factory=dict, repr=False
    )
    _snapshot_cache: Dict[str, "PageSnapshot"] = field(
        default_factory=dict, repr=False
    )

    @property
    def pattern_store(self) -> "PatternStore":
        """Get the shared pattern store."""
        if self._pattern_store is None:
            from robotmcp.optimization import PatternStore
            self._pattern_store = PatternStore()
        return self._pattern_store

    @property
    def performance_collector(self) -> "PerformanceMetricsCollector":
        """Get the performance metrics collector."""
        if self._performance_collector is None:
            from robotmcp.optimization import PerformanceMetricsCollector
            self._performance_collector = PerformanceMetricsCollector(
                pattern_store=self.pattern_store
            )
        return self._performance_collector

    @property
    def timeout_service(self) -> "TimeoutService":
        """Get the timeout management service."""
        if self._timeout_service is None:
            from robotmcp.domains.timeout import TimeoutService
            self._timeout_service = TimeoutService()
        return self._timeout_service

    @property
    def pre_validator(self) -> "PreValidator":
        """Get the pre-validation service."""
        if self._pre_validator is None:
            from robotmcp.domains.action import PreValidator
            self._pre_validator = PreValidator()
        return self._pre_validator

    @property
    def page_analyzer(self) -> "PageAnalyzer":
        """Get the page complexity analyzer."""
        if self._page_analyzer is None:
            from robotmcp.optimization import PageAnalyzer
            self._page_analyzer = PageAnalyzer()
        return self._page_analyzer

    def get_element_registry(self, session_id: str) -> "ElementRegistry":
        """Get or create an element registry for a session.

        Args:
            session_id: The session identifier

        Returns:
            ElementRegistry for the session
        """
        if session_id not in self._element_registries:
            from robotmcp.domains.element_registry import ElementRegistry
            self._element_registries[session_id] = ElementRegistry.create_for_session(session_id)
        return self._element_registries[session_id]

    def get_timeout_policy(self, session_id: str) -> "TimeoutPolicy":
        """Get or create a timeout policy for a session.

        Args:
            session_id: The session identifier

        Returns:
            TimeoutPolicy for the session
        """
        if session_id not in self._timeout_policies:
            self._timeout_policies[session_id] = (
                self.timeout_service.create_default_policy(session_id)
            )
        return self._timeout_policies[session_id]

    def get_cached_snapshot(self, session_id: str) -> Optional["PageSnapshot"]:
        """Get a cached snapshot for a session.

        Args:
            session_id: The session identifier

        Returns:
            Cached PageSnapshot or None
        """
        return self._snapshot_cache.get(session_id)

    def cache_snapshot(self, session_id: str, snapshot: "PageSnapshot") -> None:
        """Cache a snapshot for a session.

        Args:
            session_id: The session identifier
            snapshot: The snapshot to cache
        """
        self._snapshot_cache[session_id] = snapshot

    def clear_session(self, session_id: str) -> None:
        """Clear all session-specific data.

        Args:
            session_id: The session to clear
        """
        self._element_registries.pop(session_id, None)
        self._timeout_policies.pop(session_id, None)
        self._snapshot_cache.pop(session_id, None)
        logger.debug(f"Cleared container data for session {session_id}")

    def get_compression_settings(self, page_type: str) -> Dict:
        """Get learned compression settings for a page type.

        Args:
            page_type: The type of page (e.g., "search_results", "form")

        Returns:
            Dict with compression settings
        """
        try:
            from robotmcp.optimization import CompressionPatternLearner
            learner = CompressionPatternLearner(self.pattern_store)
            return learner.get_optimal_settings(page_type)
        except Exception as e:
            logger.warning(f"Failed to get compression settings: {e}")
            return {
                "fold_threshold": 0.85,
                "max_depth": None,
                "include_hidden": False,
            }

    def get_folding_settings(self, page_type: str) -> Dict:
        """Get learned list folding settings for a page type.

        Args:
            page_type: The type of page

        Returns:
            Dict with folding settings
        """
        try:
            from robotmcp.optimization import FoldingPatternLearner
            learner = FoldingPatternLearner(self.pattern_store)
            return learner.get_optimal_settings(page_type)
        except Exception as e:
            logger.warning(f"Failed to get folding settings: {e}")
            return {
                "similarity_threshold": 0.85,
                "min_items_to_fold": 3,
            }

    def get_timeout_settings(self, action_type: str) -> Dict:
        """Get learned timeout settings for an action type.

        Args:
            action_type: The type of action

        Returns:
            Dict with timeout settings
        """
        try:
            from robotmcp.optimization import TimeoutPatternLearner
            learner = TimeoutPatternLearner(self.pattern_store)
            return learner.get_optimal_settings(action_type)
        except Exception as e:
            logger.warning(f"Failed to get timeout settings: {e}")
            # Default timeouts from ADR-001
            if action_type.lower() in ("navigate", "go_to", "reload"):
                return {"timeout_ms": 60000}
            return {"timeout_ms": 5000}

    def record_snapshot_metrics(
        self,
        raw_chars: int,
        aria_chars: int,
        latency_ms: float,
        page_type: str,
        fold_threshold: float = 0.85,
    ) -> None:
        """Record snapshot generation metrics.

        Args:
            raw_chars: Characters in raw page source
            aria_chars: Characters in ARIA snapshot
            latency_ms: Time to generate snapshot
            page_type: Type of page
            fold_threshold: Folding threshold used
        """
        try:
            self.performance_collector.record_snapshot_generation(
                raw_chars=raw_chars,
                aria_chars=aria_chars,
                latency_ms=latency_ms,
                page_type=page_type,
                fold_threshold=fold_threshold,
            )
        except Exception as e:
            logger.warning(f"Failed to record snapshot metrics: {e}")

    def record_action_metrics(
        self,
        action_type: str,
        timeout_used_ms: int,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record action execution metrics.

        Args:
            action_type: Type of action executed
            timeout_used_ms: Timeout configuration used
            latency_ms: Actual execution time
            success: Whether action succeeded
        """
        try:
            self.performance_collector.record_action_execution(
                action_type=action_type,
                timeout_used_ms=timeout_used_ms,
                latency_ms=latency_ms,
                success=success,
            )
        except Exception as e:
            logger.warning(f"Failed to record action metrics: {e}")


def get_container() -> ServiceContainer:
    """Get the singleton service container.

    Returns:
        The shared ServiceContainer instance
    """
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def reset_container() -> None:
    """Reset the container (for testing).

    Clears the singleton instance so a fresh container is created
    on next get_container() call.
    """
    global _container
    _container = None
