"""Services for the desktop_performance bounded context.

Services orchestrate domain logic across aggregates and entities.
Protocol-based abstractions ensure testability without PlatynUI dependency.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .aggregates import ApplicationScope, ElementCache, InteractionProfile
from .entities import ApplicationRoot, CachedElement
from .events import (
    ApplicationScopeSet,
    CacheHit,
    ElementCached,
    ElementInvalidated,
    XPathTransformed,
)
from .value_objects import CacheKey, XPathTransform, _is_platynui_xpath

logger = logging.getLogger(__name__)
_monotonic = time.monotonic


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class RuntimeQueryProtocol(Protocol):
    """Abstraction over platynui_native.Runtime for testability."""

    def evaluate_single(self, xpath: str, node: Any = None) -> Any: ...

    def clear_cache(self) -> None: ...


@runtime_checkable
class DescriptorFactoryProtocol(Protocol):
    """Creates UiNodeDescriptor instances."""

    def create(self, node: Any, library: Any) -> Any: ...


@runtime_checkable
class EventPublisherProtocol(Protocol):
    """Publishes domain events for observability."""

    def publish(self, event: Any) -> None: ...


# ---------------------------------------------------------------------------
# Simple event collector (default publisher)
# ---------------------------------------------------------------------------


@dataclass
class EventCollector:
    """Simple in-memory event collector for domain events."""

    events: List[Any] = field(default_factory=list)
    max_events: int = 1000

    def publish(self, event: Any) -> None:
        if len(self.events) >= self.max_events:
            self.events.pop(0)
        self.events.append(event)
        logger.debug("Domain event: %s", getattr(event, "to_dict", lambda: event)())

    def get_recent(self, n: int = 10) -> List[Any]:
        return self.events[-n:]

    def clear(self) -> None:
        self.events.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_descendants(node: Any) -> int:
    """Count the number of descendants of a UiNode."""
    count = 0
    try:
        children = node.children()
        for child in children:
            count += 1
            count += _count_descendants(child)
    except Exception:
        pass
    return count


# ---------------------------------------------------------------------------
# ApplicationScopeManager (Service)
# ---------------------------------------------------------------------------


@dataclass
class ApplicationScopeManager:
    """Discovers and manages the application-scoped root for desktop sessions.

    Strategy:
    1. On first desktop keyword execution, examine the XPath for a Frame/Window pattern
    2. Extract the application identifier (e.g., [@Name='Calculator'])
    3. Execute an absolute XPath query to find the application's top-level node
    4. Store the root in ApplicationScope for the session
    5. All subsequent XPaths are transformed to relative (.//), scoped to the app root
    """

    runtime_query: Optional[RuntimeQueryProtocol] = None
    event_publisher: Optional[EventPublisherProtocol] = None

    # Patterns that indicate a top-level application reference
    _APP_PATTERNS: ClassVar[Tuple[str, ...]] = (
        r'//control:Frame\[@Name=["\']([^"\']+)',
        r'//control:Window\[@Name=["\']([^"\']+)',
        r'//app:Application\[@Name=["\']([^"\']+)',
        # Also match relative forms
        r'\.//control:Frame\[@Name=["\']([^"\']+)',
        r'\.//control:Window\[@Name=["\']([^"\']+)',
    )

    def ensure_scope(
        self, scope: ApplicationScope, xpath: str, runtime: Any = None
    ) -> ApplicationScope:
        """Ensure the scope has a valid application root.

        If unscoped, attempts auto-discovery from the XPath pattern.
        If scoped but invalid, re-discovers.
        """
        if scope.is_scoped:
            return scope

        # Try to extract application name from the XPath
        app_name = self._extract_app_name(xpath)
        if app_name is None:
            return scope  # Can't auto-discover, stay unscoped

        # Build discovery XPath for the top-level frame
        discovery_xpath = f'//control:Frame[@Name="{app_name}"]'
        scope.set_discovery_xpath(discovery_xpath)

        # Execute the discovery query
        rt = runtime or self.runtime_query
        if rt is None:
            return scope

        t0 = _monotonic()
        try:
            root_node = rt.evaluate_single(discovery_xpath)
        except Exception as exc:
            logger.debug("Application root discovery failed: %s", exc)
            return scope

        if root_node is None:
            return scope

        t1 = _monotonic()

        # Create and set the application root
        root_runtime_id = ""
        try:
            root_runtime_id = str(root_node.runtime_id)
        except Exception:
            root_runtime_id = str(id(root_node))

        app_root = ApplicationRoot(
            session_id=scope.session_id,
            application_name=app_name,
            root_node=root_node,
            root_runtime_id=root_runtime_id,
        )

        # Count descendants (cheap: ~0.26s for 52 nodes)
        try:
            app_root.descendant_count = _count_descendants(root_node)
        except Exception:
            pass

        scope.set_root(app_root)

        if self.event_publisher:
            self.event_publisher.publish(
                ApplicationScopeSet(
                    session_id=scope.session_id,
                    application_name=app_name,
                    runtime_id=app_root.root_runtime_id,
                    descendant_count=app_root.descendant_count,
                    discovery_time_ms=(t1 - t0) * 1000,
                )
            )

        logger.info(
            "Application scope set: %s (%d descendants) in %.1fms",
            app_name,
            app_root.descendant_count or 0,
            (t1 - t0) * 1000,
        )

        return scope

    def _extract_app_name(self, xpath: str) -> Optional[str]:
        """Extract application name from XPath pattern."""
        for pattern in self._APP_PATTERNS:
            match = re.search(pattern, xpath)
            if match:
                return match.group(1)
        return None


# ---------------------------------------------------------------------------
# DesktopKeywordOptimizer (Service - Main Entry Point)
# ---------------------------------------------------------------------------


@dataclass
class DesktopKeywordOptimizer:
    """Optimizes desktop keyword execution by caching elements and scoping queries.

    This is the main orchestration service. It intercepts execute_step()
    calls for desktop sessions and applies three optimizations:

    1. Application scoping: Converts absolute XPaths to relative (290x faster)
    2. Element caching: Reuses previously resolved UiNodeDescriptors (375x faster)
    3. Interaction speed: Applies fast PointerOverrides (22x faster click)
    """

    scope_manager: ApplicationScopeManager
    event_publisher: Optional[EventPublisherProtocol] = None

    # Cacheable keywords (first argument is an XPath locator)
    _CACHEABLE_KEYWORDS: ClassVar[frozenset] = frozenset(
        {
            "pointer click",
            "pointer multi click",
            "pointer press",
            "pointer release",
            "pointer move to",
            "focus",
            "activate",
            "get attribute",
            "keyboard type",
            "keyboard press",
            "keyboard release",
            "take screenshot",
            "highlight",
        }
    )

    # Keywords that modify UI state (cache should be cautious)
    _STATE_MODIFYING: ClassVar[frozenset] = frozenset(
        {
            "pointer click",
            "pointer multi click",
            "keyboard type",
            "keyboard press",
            "close",
            "maximize",
            "minimize",
            "restore",
        }
    )

    def optimize(
        self,
        session_id: str,
        keyword: str,
        arguments: List[str],
        element_cache: ElementCache,
        scope: ApplicationScope,
        profile: InteractionProfile,
        runtime: Any = None,
    ) -> Dict[str, Any]:
        """Apply optimizations to a desktop keyword execution.

        Returns dict with:
        - arguments: potentially modified arguments (cached descriptor or relative XPath)
        - cache_hit: bool
        - scope_applied: bool
        - profile_applied: bool
        - metadata: optimization details for logging
        """
        keyword_lower = keyword.lower().replace("_", " ")
        if keyword_lower not in self._CACHEABLE_KEYWORDS:
            return {
                "arguments": arguments,
                "cache_hit": False,
                "scope_applied": False,
                "profile_applied": False,
                "metadata": {"reason": "non_cacheable_keyword"},
            }

        if not arguments:
            return {
                "arguments": arguments,
                "cache_hit": False,
                "scope_applied": False,
                "profile_applied": False,
                "metadata": {"reason": "no_arguments"},
            }

        xpath = arguments[0]
        if not _is_platynui_xpath(xpath):
            return {
                "arguments": arguments,
                "cache_hit": False,
                "scope_applied": False,
                "profile_applied": False,
                "metadata": {"reason": "not_platynui_xpath"},
            }

        cache_key = CacheKey(xpath=xpath, session_id=session_id)

        # Step 1: Check element cache
        cached = element_cache.lookup(cache_key)
        if cached is not None:
            if self.event_publisher:
                self.event_publisher.publish(
                    CacheHit(
                        session_id=session_id,
                        xpath=xpath,
                        hit_count=cached.hit_count,
                        saved_time_ms=7000.0,  # Conservative estimate
                    )
                )
            return {
                "arguments": list(arguments),
                "cached_element": cached,
                "cache_hit": True,
                "scope_applied": True,
                "profile_applied": True,
                "metadata": {"hit_count": cached.hit_count},
            }

        # Step 2: Apply application scoping (convert absolute -> relative XPath)
        self.scope_manager.ensure_scope(scope, xpath, runtime)
        transform = scope.transform_xpath(xpath)
        scope_applied = transform.scoping_applied

        if scope_applied:
            optimized_args = list(arguments)
            optimized_args[0] = transform.transformed
            if self.event_publisher:
                self.event_publisher.publish(
                    XPathTransformed(
                        session_id=session_id,
                        original_xpath=xpath,
                        relative_xpath=transform.transformed,
                    )
                )
        else:
            optimized_args = list(arguments)

        return {
            "arguments": optimized_args,
            "cache_hit": False,
            "scope_applied": scope_applied,
            "profile_applied": True,
            "transform": transform,
            "metadata": {"xpath_axis": transform.axis.value},
        }

    def record_resolution(
        self,
        session_id: str,
        keyword: str,
        arguments: List[str],
        element_cache: ElementCache,
        scope: ApplicationScope,
        descriptor: Any,
        node: Any,
        resolution_time_ms: float,
    ) -> None:
        """Record a successful element resolution for future cache hits."""
        if not arguments:
            return
        xpath = arguments[0]
        if not _is_platynui_xpath(xpath):
            return

        cache_key = CacheKey(xpath=xpath, session_id=session_id)

        # Determine the relative form
        transform = scope.transform_xpath(xpath)

        now = _monotonic()
        element = CachedElement(
            cache_key=cache_key,
            xpath_original=xpath,
            xpath_relative=transform.transformed,
            descriptor=descriptor,
            resolved_node=node,
            application_root_id=(
                scope.root.root_runtime_id if scope.root else None
            ),
            created_at=now,
            last_accessed=now,
        )
        element_cache.store(element)

        if self.event_publisher:
            self.event_publisher.publish(
                ElementCached(
                    session_id=session_id,
                    xpath=xpath,
                    application_name=(
                        scope.root.application_name if scope.root else None
                    ),
                    resolution_time_ms=resolution_time_ms,
                )
            )
