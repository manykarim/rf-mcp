"""Tests for desktop_performance services."""

from unittest.mock import MagicMock, patch

import pytest

from robotmcp.domains.desktop_performance.aggregates import (
    ApplicationScope,
    ElementCache,
    InteractionProfile,
)
from robotmcp.domains.desktop_performance.entities import ApplicationRoot
from robotmcp.domains.desktop_performance.events import (
    ApplicationScopeSet,
    CacheHit,
    ElementCached,
    XPathTransformed,
)
from robotmcp.domains.desktop_performance.services import (
    ApplicationScopeManager,
    DesktopKeywordOptimizer,
    EventCollector,
    _count_descendants,
)
from robotmcp.domains.desktop_performance.value_objects import CacheKey


# ---------------------------------------------------------------------------
# EventCollector
# ---------------------------------------------------------------------------


class TestEventCollector:
    def test_publish_stores_event(self):
        ec = EventCollector()
        event = MagicMock()
        ec.publish(event)
        assert len(ec.events) == 1

    def test_max_events_evicts_oldest(self):
        ec = EventCollector(max_events=3)
        for i in range(5):
            ec.publish(f"event_{i}")
        assert len(ec.events) == 3
        assert ec.events[0] == "event_2"

    def test_get_recent(self):
        ec = EventCollector()
        for i in range(10):
            ec.publish(f"event_{i}")
        recent = ec.get_recent(3)
        assert len(recent) == 3
        assert recent[0] == "event_7"

    def test_clear(self):
        ec = EventCollector()
        ec.publish("test")
        ec.clear()
        assert len(ec.events) == 0


# ---------------------------------------------------------------------------
# _count_descendants
# ---------------------------------------------------------------------------


class TestCountDescendants:
    def test_simple_tree(self):
        # root -> [child1, child2]
        child1 = MagicMock()
        child1.children.return_value = []
        child2 = MagicMock()
        child2.children.return_value = []
        root = MagicMock()
        root.children.return_value = [child1, child2]
        assert _count_descendants(root) == 2

    def test_nested_tree(self):
        # root -> [child1 -> [grandchild1, grandchild2]]
        gc1 = MagicMock()
        gc1.children.return_value = []
        gc2 = MagicMock()
        gc2.children.return_value = []
        child1 = MagicMock()
        child1.children.return_value = [gc1, gc2]
        root = MagicMock()
        root.children.return_value = [child1]
        assert _count_descendants(root) == 3  # child1 + gc1 + gc2

    def test_empty_tree(self):
        root = MagicMock()
        root.children.return_value = []
        assert _count_descendants(root) == 0

    def test_exception_returns_zero(self):
        root = MagicMock()
        root.children.side_effect = RuntimeError("AT-SPI error")
        assert _count_descendants(root) == 0


# ---------------------------------------------------------------------------
# ApplicationScopeManager
# ---------------------------------------------------------------------------


class TestApplicationScopeManager:
    def test_extract_app_name_from_frame(self):
        mgr = ApplicationScopeManager()
        name = mgr._extract_app_name('//control:Frame[@Name="Calculator"]')
        assert name == "Calculator"

    def test_extract_app_name_from_window(self):
        mgr = ApplicationScopeManager()
        name = mgr._extract_app_name("//control:Window[@Name='Firefox']")
        assert name == "Firefox"

    def test_extract_app_name_from_application(self):
        mgr = ApplicationScopeManager()
        name = mgr._extract_app_name("//app:Application[@Name='MyApp']")
        assert name == "MyApp"

    def test_extract_app_name_from_relative(self):
        mgr = ApplicationScopeManager()
        name = mgr._extract_app_name('.//control:Frame[@Name="Calc"]')
        assert name == "Calc"

    def test_extract_app_name_none_for_button(self):
        mgr = ApplicationScopeManager()
        name = mgr._extract_app_name('//control:Button[@Name="OK"]')
        assert name is None

    def test_ensure_scope_already_scoped(self):
        mgr = ApplicationScopeManager()
        scope = ApplicationScope.create(session_id="s1")
        mock_node = MagicMock()
        mock_node.is_valid.return_value = True
        root = ApplicationRoot(
            session_id="s1",
            application_name="Calc",
            root_node=mock_node,
            root_runtime_id="rt1",
        )
        scope.set_root(root)
        # Should return immediately without doing anything
        result = mgr.ensure_scope(scope, "//control:Button[@Name='2']")
        assert result.is_scoped is True

    def test_ensure_scope_discovers_root(self):
        mock_runtime = MagicMock()
        mock_node = MagicMock()
        mock_node.runtime_id = "atspi://calc"
        mock_node.is_valid.return_value = True
        mock_node.children.return_value = []
        mock_runtime.evaluate_single.return_value = mock_node

        ec = EventCollector()
        mgr = ApplicationScopeManager(
            runtime_query=mock_runtime,
            event_publisher=ec,
        )

        scope = ApplicationScope.create(session_id="s1")
        xpath = '//control:Frame[@Name="Calculator"]//control:Button[@Name="2"]'
        result = mgr.ensure_scope(scope, xpath)

        assert result.is_scoped is True
        assert result.root.application_name == "Calculator"
        mock_runtime.evaluate_single.assert_called_once_with(
            '//control:Frame[@Name="Calculator"]'
        )
        # Event should be published
        assert len(ec.events) == 1
        assert isinstance(ec.events[0], ApplicationScopeSet)

    def test_ensure_scope_no_runtime(self):
        mgr = ApplicationScopeManager()
        scope = ApplicationScope.create(session_id="s1")
        result = mgr.ensure_scope(
            scope, '//control:Frame[@Name="Calc"]//Button'
        )
        assert result.is_scoped is False  # No runtime to query

    def test_ensure_scope_runtime_returns_none(self):
        mock_runtime = MagicMock()
        mock_runtime.evaluate_single.return_value = None
        mgr = ApplicationScopeManager(runtime_query=mock_runtime)
        scope = ApplicationScope.create(session_id="s1")
        result = mgr.ensure_scope(
            scope, '//control:Frame[@Name="Calc"]//Button'
        )
        assert result.is_scoped is False

    def test_ensure_scope_runtime_exception(self):
        mock_runtime = MagicMock()
        mock_runtime.evaluate_single.side_effect = RuntimeError("AT-SPI down")
        mgr = ApplicationScopeManager(runtime_query=mock_runtime)
        scope = ApplicationScope.create(session_id="s1")
        result = mgr.ensure_scope(
            scope, '//control:Frame[@Name="Calc"]//Button'
        )
        assert result.is_scoped is False

    def test_ensure_scope_no_app_name_detected(self):
        mgr = ApplicationScopeManager()
        scope = ApplicationScope.create(session_id="s1")
        result = mgr.ensure_scope(scope, '//control:Button[@Name="OK"]')
        assert result.is_scoped is False  # Can't determine app name


# ---------------------------------------------------------------------------
# DesktopKeywordOptimizer
# ---------------------------------------------------------------------------


class TestDesktopKeywordOptimizer:
    def _make_optimizer(self, event_collector=None):
        ec = event_collector or EventCollector()
        return DesktopKeywordOptimizer(
            scope_manager=ApplicationScopeManager(event_publisher=ec),
            event_publisher=ec,
        ), ec

    def _make_scoped_scope(self, session_id="s1"):
        scope = ApplicationScope.create(session_id=session_id)
        mock_node = MagicMock()
        mock_node.is_valid.return_value = True
        root = ApplicationRoot(
            session_id=session_id,
            application_name="Calculator",
            root_node=mock_node,
            root_runtime_id="rt1",
        )
        scope.set_root(root)
        return scope

    def test_non_cacheable_keyword_passthrough(self):
        opt, _ = self._make_optimizer()
        result = opt.optimize(
            session_id="s1",
            keyword="query",
            arguments=["//Window"],
            element_cache=ElementCache(session_id="s1"),
            scope=ApplicationScope.create(session_id="s1"),
            profile=InteractionProfile.create(session_id="s1"),
        )
        assert result["cache_hit"] is False
        assert result["scope_applied"] is False
        assert result["metadata"]["reason"] == "non_cacheable_keyword"

    def test_no_arguments_passthrough(self):
        opt, _ = self._make_optimizer()
        result = opt.optimize(
            session_id="s1",
            keyword="pointer click",
            arguments=[],
            element_cache=ElementCache(session_id="s1"),
            scope=ApplicationScope.create(session_id="s1"),
            profile=InteractionProfile.create(session_id="s1"),
        )
        assert result["cache_hit"] is False
        assert result["metadata"]["reason"] == "no_arguments"

    def test_not_platynui_xpath_passthrough(self):
        opt, _ = self._make_optimizer()
        result = opt.optimize(
            session_id="s1",
            keyword="pointer click",
            arguments=["id=submit"],
            element_cache=ElementCache(session_id="s1"),
            scope=ApplicationScope.create(session_id="s1"),
            profile=InteractionProfile.create(session_id="s1"),
        )
        assert result["cache_hit"] is False
        assert result["metadata"]["reason"] == "not_platynui_xpath"

    def test_cache_miss_applies_scoping(self):
        opt, ec = self._make_optimizer()
        scope = self._make_scoped_scope()
        result = opt.optimize(
            session_id="s1",
            keyword="pointer click",
            arguments=['//control:Button[@Name="2"]'],
            element_cache=ElementCache(session_id="s1"),
            scope=scope,
            profile=InteractionProfile.create(session_id="s1"),
        )
        assert result["cache_hit"] is False
        assert result["scope_applied"] is True
        assert result["arguments"][0] == './/control:Button[@Name="2"]'
        # XPathTransformed event emitted
        xpath_events = [e for e in ec.events if isinstance(e, XPathTransformed)]
        assert len(xpath_events) == 1

    def test_cache_hit_returns_cached_element(self):
        opt, ec = self._make_optimizer()
        cache = ElementCache(session_id="s1")
        # Pre-populate cache
        mock_node = MagicMock()
        mock_node.is_valid.return_value = True
        from robotmcp.domains.desktop_performance.entities import CachedElement
        import time
        now = time.monotonic()
        elem = CachedElement(
            cache_key=CacheKey(xpath='//control:Button[@Name="2"]', session_id="s1"),
            xpath_original='//control:Button[@Name="2"]',
            xpath_relative='.//control:Button[@Name="2"]',
            descriptor=MagicMock(),
            resolved_node=mock_node,
            application_root_id="rt1",
            created_at=now,
            last_accessed=now,
        )
        cache.store(elem)

        result = opt.optimize(
            session_id="s1",
            keyword="pointer click",
            arguments=['//control:Button[@Name="2"]'],
            element_cache=cache,
            scope=ApplicationScope.create(session_id="s1"),
            profile=InteractionProfile.create(session_id="s1"),
        )
        assert result["cache_hit"] is True
        assert result["cached_element"] is not None
        # CacheHit event emitted
        hit_events = [e for e in ec.events if isinstance(e, CacheHit)]
        assert len(hit_events) == 1

    def test_record_resolution_stores_element(self):
        opt, ec = self._make_optimizer()
        cache = ElementCache(session_id="s1")
        scope = self._make_scoped_scope()

        opt.record_resolution(
            session_id="s1",
            keyword="pointer click",
            arguments=['//control:Button[@Name="2"]'],
            element_cache=cache,
            scope=scope,
            descriptor=MagicMock(),
            node=MagicMock(),
            resolution_time_ms=270.0,
        )
        assert cache.size == 1
        # ElementCached event emitted
        cached_events = [e for e in ec.events if isinstance(e, ElementCached)]
        assert len(cached_events) == 1
        assert cached_events[0].resolution_time_ms == 270.0

    def test_record_resolution_skips_non_xpath(self):
        opt, _ = self._make_optimizer()
        cache = ElementCache(session_id="s1")
        scope = ApplicationScope.create(session_id="s1")
        opt.record_resolution(
            session_id="s1",
            keyword="pointer click",
            arguments=["not-xpath"],
            element_cache=cache,
            scope=scope,
            descriptor=MagicMock(),
            node=MagicMock(),
            resolution_time_ms=100.0,
        )
        assert cache.size == 0

    def test_record_resolution_skips_empty_args(self):
        opt, _ = self._make_optimizer()
        cache = ElementCache(session_id="s1")
        scope = ApplicationScope.create(session_id="s1")
        opt.record_resolution(
            session_id="s1",
            keyword="pointer click",
            arguments=[],
            element_cache=cache,
            scope=scope,
            descriptor=MagicMock(),
            node=MagicMock(),
            resolution_time_ms=100.0,
        )
        assert cache.size == 0

    def test_keyword_normalization(self):
        """Underscores in keyword names should be treated as spaces."""
        opt, _ = self._make_optimizer()
        scope = self._make_scoped_scope()
        result = opt.optimize(
            session_id="s1",
            keyword="Pointer_Click",
            arguments=['//control:Button[@Name="2"]'],
            element_cache=ElementCache(session_id="s1"),
            scope=scope,
            profile=InteractionProfile.create(session_id="s1"),
        )
        assert result["scope_applied"] is True  # Should be recognized as cacheable

    def test_frame_prefix_stripped_from_xpath(self):
        """When XPath contains Frame prefix matching the scope, strip it."""
        opt, ec = self._make_optimizer()
        scope = self._make_scoped_scope()
        # Set discovery xpath so stripping logic kicks in
        scope.set_discovery_xpath('//control:Frame[@Name="Calculator"]')
        # Re-set the root (set_discovery_xpath changes state to discovering)
        mock_node = MagicMock()
        mock_node.is_valid.return_value = True
        from robotmcp.domains.desktop_performance.entities import ApplicationRoot
        root = ApplicationRoot(
            session_id="s1",
            application_name="Calculator",
            root_node=mock_node,
            root_runtime_id="rt1",
        )
        scope.set_root(root)

        result = opt.optimize(
            session_id="s1",
            keyword="pointer click",
            arguments=['//control:Frame[@Name="Calculator"]//control:Button[@Name="2"]'],
            element_cache=ElementCache(session_id="s1"),
            scope=scope,
            profile=InteractionProfile.create(session_id="s1"),
        )
        assert result["scope_applied"] is True
        # Frame prefix should be stripped, leaving only the Button part
        assert result["arguments"][0] == './/control:Button[@Name="2"]'

    def test_unscoped_session_no_transform(self):
        opt, _ = self._make_optimizer()
        result = opt.optimize(
            session_id="s1",
            keyword="pointer click",
            arguments=['//control:Button[@Name="2"]'],
            element_cache=ElementCache(session_id="s1"),
            scope=ApplicationScope.create(session_id="s1"),
            profile=InteractionProfile.create(session_id="s1"),
        )
        assert result["cache_hit"] is False
        assert result["scope_applied"] is False
        # Arguments unchanged
        assert result["arguments"][0] == '//control:Button[@Name="2"]'
