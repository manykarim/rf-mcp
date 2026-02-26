"""Tests for desktop_performance aggregates."""

import time
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.domains.desktop_performance.aggregates import (
    ApplicationScope,
    ElementCache,
    InteractionProfile,
)
from robotmcp.domains.desktop_performance.entities import (
    ApplicationRoot,
    CachedElement,
)
from robotmcp.domains.desktop_performance.value_objects import (
    CacheCapacity,
    CacheKey,
    CacheTTL,
    InteractionSpeed,
    POINTER_SPEED_FAST,
    POINTER_SPEED_INSTANT,
    XPathAxis,
)


# ---------------------------------------------------------------------------
# ElementCache
# ---------------------------------------------------------------------------


class TestElementCache:
    def _make_cache(self, capacity=200, ttl=60.0):
        return ElementCache(
            session_id="s1",
            capacity=CacheCapacity(max_entries=capacity),
            ttl=CacheTTL(value_seconds=ttl),
        )

    def _make_element(self, xpath="//Button", valid=True, created_at=None):
        mock_node = MagicMock()
        mock_node.is_valid.return_value = valid
        now = created_at or time.monotonic()
        return CachedElement(
            cache_key=CacheKey(xpath=xpath, session_id="s1"),
            xpath_original=xpath,
            xpath_relative="./" + xpath.lstrip("/"),
            descriptor=MagicMock(),
            resolved_node=mock_node,
            application_root_id="root1",
            created_at=now,
            last_accessed=now,
        )

    def test_store_and_lookup(self):
        cache = self._make_cache()
        elem = self._make_element()
        cache.store(elem)
        result = cache.lookup(elem.cache_key)
        assert result is not None
        assert result.xpath_original == "//Button"

    def test_lookup_miss_returns_none(self):
        cache = self._make_cache()
        key = CacheKey(xpath="//Missing", session_id="s1")
        assert cache.lookup(key) is None

    def test_lookup_miss_increments_stats(self):
        cache = self._make_cache()
        key = CacheKey(xpath="//Missing", session_id="s1")
        cache.lookup(key)
        assert cache.get_stats()["misses"] == 1

    def test_lookup_expired_ttl_returns_none(self):
        cache = self._make_cache(ttl=1.0)
        elem = self._make_element(created_at=0.0)  # Created at monotonic 0
        cache.store(elem)
        # Manually set created_at to the past to simulate expiry
        elem.created_at = time.monotonic() - 10.0
        result = cache.lookup(elem.cache_key)
        assert result is None

    def test_lookup_invalid_node_returns_none(self):
        cache = self._make_cache()
        elem = self._make_element(valid=False)
        cache.store(elem)
        result = cache.lookup(elem.cache_key)
        assert result is None
        assert cache.get_stats()["invalidations"] == 1

    def test_lookup_hit_increments_stats(self):
        cache = self._make_cache()
        elem = self._make_element()
        cache.store(elem)
        cache.lookup(elem.cache_key)
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["hit_rate"] == 1.0

    def test_lru_eviction_at_capacity(self):
        cache = self._make_cache(capacity=10)
        # Fill to capacity
        for i in range(10):
            cache.store(self._make_element(xpath=f"//Button{i}"))
        assert cache.size == 10
        # One more triggers eviction
        cache.store(self._make_element(xpath="//ButtonNew"))
        assert cache.size == 10  # Still 10
        # First inserted (//Button0) should be evicted
        key0 = CacheKey(xpath="//Button0", session_id="s1")
        assert cache.lookup(key0) is None

    def test_promote_on_access(self):
        cache = self._make_cache(capacity=10)
        # Fill to capacity
        for i in range(10):
            cache.store(self._make_element(xpath=f"//Button{i}"))
        # Access Button0 to promote it
        key0 = CacheKey(xpath="//Button0", session_id="s1")
        cache.lookup(key0)
        # Add new element — should evict Button1 (LRU), not Button0
        cache.store(self._make_element(xpath="//ButtonNew"))
        assert cache.lookup(key0) is not None  # Button0 survived
        key1 = CacheKey(xpath="//Button1", session_id="s1")
        assert cache.lookup(key1) is None  # Button1 was evicted

    def test_invalidate_all(self):
        cache = self._make_cache()
        for i in range(5):
            cache.store(self._make_element(xpath=f"//Button{i}"))
        count = cache.invalidate_all()
        assert count == 5
        assert cache.size == 0

    def test_invalidate_by_prefix(self):
        cache = self._make_cache()
        cache.store(self._make_element(xpath="//control:Button[@Name='1']"))
        cache.store(self._make_element(xpath="//control:Button[@Name='2']"))
        cache.store(self._make_element(xpath="//item:ListItem[@Name='X']"))
        count = cache.invalidate_by_prefix("//control:Button")
        assert count == 2
        assert cache.size == 1

    def test_stats_tracking(self):
        cache = self._make_cache()
        elem = self._make_element()
        cache.store(elem)
        cache.lookup(elem.cache_key)  # hit
        cache.lookup(CacheKey(xpath="//Missing", session_id="s1"))  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_hit_rate_calculation(self):
        cache = self._make_cache()
        elem = self._make_element()
        cache.store(elem)
        cache.lookup(elem.cache_key)  # hit
        cache.lookup(elem.cache_key)  # hit
        cache.lookup(CacheKey(xpath="//Miss", session_id="s1"))  # miss
        stats = cache.get_stats()
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_hit_rate_zero_total(self):
        cache = self._make_cache()
        assert cache.get_stats()["hit_rate"] == 0.0

    def test_to_response_dict(self):
        cache = self._make_cache()
        cache.store(self._make_element())
        d = cache.to_response_dict()
        assert d["session_id"] == "s1"
        assert d["size"] == 1
        assert d["capacity"] == 200
        assert "stats" in d
        assert "entries" in d
        assert len(d["entries"]) == 1

    def test_store_replaces_existing(self):
        cache = self._make_cache()
        elem1 = self._make_element(xpath="//Button")
        cache.store(elem1)
        elem2 = self._make_element(xpath="//Button")
        cache.store(elem2)
        assert cache.size == 1

    def test_eviction_increments_stats(self):
        cache = self._make_cache(capacity=10)
        for i in range(11):
            cache.store(self._make_element(xpath=f"//Button{i}"))
        assert cache.get_stats()["evictions"] == 1

    def test_test_suppression(self):
        assert ElementCache.__test__ is False


# ---------------------------------------------------------------------------
# ApplicationScope
# ---------------------------------------------------------------------------


class TestApplicationScope:
    def test_create_unscoped(self):
        scope = ApplicationScope.create(session_id="s1")
        assert scope.state == "unscoped"
        assert scope.is_scoped is False
        assert scope.root is None

    def test_set_discovery_xpath(self):
        scope = ApplicationScope.create(session_id="s1")
        scope.set_discovery_xpath('//control:Frame[@Name="Calculator"]')
        assert scope.state == "discovering"
        assert scope.discovery_xpath == '//control:Frame[@Name="Calculator"]'

    def test_set_root_transitions_to_scoped(self):
        scope = ApplicationScope.create(session_id="s1")
        mock_node = MagicMock()
        mock_node.is_valid.return_value = True
        root = ApplicationRoot(
            session_id="s1",
            application_name="Calculator",
            root_node=mock_node,
            root_runtime_id="rt1",
        )
        scope.set_root(root)
        assert scope.state == "scoped"
        assert scope.is_scoped is True

    def test_is_scoped_checks_validity(self):
        scope = ApplicationScope.create(session_id="s1")
        mock_node = MagicMock()
        mock_node.is_valid.return_value = True
        root = ApplicationRoot(
            session_id="s1",
            application_name="Calculator",
            root_node=mock_node,
            root_runtime_id="rt1",
        )
        scope.set_root(root)
        assert scope.is_scoped is True
        # Node becomes stale
        mock_node.is_valid.return_value = False
        assert scope.is_scoped is False

    def test_invalidate_clears_root(self):
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
        scope.invalidate()
        assert scope.state == "invalidated"
        assert scope.root is None
        assert scope.is_scoped is False

    def test_transform_xpath_when_scoped(self):
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
        t = scope.transform_xpath('//control:Button[@Name="2"]')
        assert t.scoping_applied is True
        assert t.transformed == './/control:Button[@Name="2"]'

    def test_transform_xpath_strips_frame_prefix(self):
        """When xpath contains the scope's Frame prefix, strip it."""
        scope = ApplicationScope.create(session_id="s1")
        scope.set_discovery_xpath('//control:Frame[@Name="Calculator"]')
        mock_node = MagicMock()
        mock_node.is_valid.return_value = True
        root = ApplicationRoot(
            session_id="s1",
            application_name="Calculator",
            root_node=mock_node,
            root_runtime_id="rt1",
        )
        scope.set_root(root)
        t = scope.transform_xpath(
            '//control:Frame[@Name="Calculator"]//control:Button[@Name="2"]'
        )
        assert t.scoping_applied is True
        # Frame prefix stripped, only Button part remains as relative
        assert t.transformed == './/control:Button[@Name="2"]'

    def test_transform_xpath_no_strip_when_no_prefix_match(self):
        """When xpath doesn't start with the scope's prefix, just prepend '.'."""
        scope = ApplicationScope.create(session_id="s1")
        scope.set_discovery_xpath('//control:Frame[@Name="Calculator"]')
        mock_node = MagicMock()
        mock_node.is_valid.return_value = True
        root = ApplicationRoot(
            session_id="s1",
            application_name="Calculator",
            root_node=mock_node,
            root_runtime_id="rt1",
        )
        scope.set_root(root)
        t = scope.transform_xpath('//control:Button[@Name="2"]')
        assert t.scoping_applied is True
        assert t.transformed == './/control:Button[@Name="2"]'

    def test_transform_xpath_when_unscoped(self):
        scope = ApplicationScope.create(session_id="s1")
        t = scope.transform_xpath('//control:Button[@Name="2"]')
        assert t.scoping_applied is False
        assert t.transformed == '//control:Button[@Name="2"]'

    def test_to_dict(self):
        scope = ApplicationScope.create(session_id="s1")
        d = scope.to_dict()
        assert d["session_id"] == "s1"
        assert d["state"] == "unscoped"
        assert d["root"] is None

    def test_to_dict_with_root(self):
        scope = ApplicationScope.create(session_id="s1")
        mock_node = MagicMock()
        mock_node.is_valid.return_value = True
        root = ApplicationRoot(
            session_id="s1",
            application_name="Calculator",
            root_node=mock_node,
            root_runtime_id="rt1",
            descendant_count=52,
        )
        scope.set_root(root)
        d = scope.to_dict()
        assert d["state"] == "scoped"
        assert d["root"]["application_name"] == "Calculator"

    def test_test_suppression(self):
        assert ApplicationScope.__test__ is False


# ---------------------------------------------------------------------------
# InteractionProfile
# ---------------------------------------------------------------------------


class TestInteractionProfile:
    def test_create_default_fast(self):
        p = InteractionProfile.create(session_id="s1")
        assert p.pointer_profile.speed == InteractionSpeed.FAST

    def test_create_instant(self):
        p = InteractionProfile.create(
            session_id="s1", speed=InteractionSpeed.INSTANT
        )
        assert p.pointer_profile.speed == InteractionSpeed.INSTANT
        assert p.pointer_profile.after_click_delay_ms == 0

    def test_create_realistic(self):
        p = InteractionProfile.create(
            session_id="s1", speed=InteractionSpeed.REALISTIC
        )
        assert p.pointer_profile.speed == InteractionSpeed.REALISTIC

    def test_to_dict(self):
        p = InteractionProfile.create(session_id="s1")
        d = p.to_dict()
        assert d["session_id"] == "s1"
        assert d["speed"] == "fast"
        assert "pointer_overrides" in d

    def test_test_suppression(self):
        assert InteractionProfile.__test__ is False
