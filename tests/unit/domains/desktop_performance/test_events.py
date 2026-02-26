"""Tests for desktop_performance events."""

from datetime import datetime

import pytest

from robotmcp.domains.desktop_performance.events import (
    ApplicationScopeSet,
    CacheHit,
    ElementCached,
    ElementInvalidated,
    XPathTransformed,
)


class TestElementCached:
    def test_to_dict(self):
        e = ElementCached(
            session_id="s1",
            xpath="//Button",
            application_name="Calculator",
            resolution_time_ms=270.5,
        )
        d = e.to_dict()
        assert d["event_type"] == "element_cached"
        assert d["session_id"] == "s1"
        assert d["xpath"] == "//Button"
        assert d["application_name"] == "Calculator"
        assert d["resolution_time_ms"] == 270.5
        assert "timestamp" in d

    def test_frozen(self):
        e = ElementCached(
            session_id="s1",
            xpath="//Button",
            application_name=None,
            resolution_time_ms=100.0,
        )
        with pytest.raises(AttributeError):
            e.session_id = "changed"


class TestCacheHit:
    def test_to_dict(self):
        e = CacheHit(
            session_id="s1",
            xpath="//Button",
            hit_count=5,
            saved_time_ms=7000.0,
        )
        d = e.to_dict()
        assert d["event_type"] == "cache_hit"
        assert d["hit_count"] == 5
        assert d["saved_time_ms"] == 7000.0

    def test_frozen(self):
        e = CacheHit(session_id="s1", xpath="//B", hit_count=1, saved_time_ms=100.0)
        with pytest.raises(AttributeError):
            e.xpath = "changed"


class TestElementInvalidated:
    def test_to_dict(self):
        e = ElementInvalidated(
            session_id="s1",
            xpath="//Button",
            reason="stale_node",
        )
        d = e.to_dict()
        assert d["event_type"] == "element_invalidated"
        assert d["reason"] == "stale_node"

    def test_reason_values(self):
        for reason in ["stale_node", "ttl_expired", "explicit_clear", "capacity_eviction"]:
            e = ElementInvalidated(session_id="s1", xpath="//B", reason=reason)
            assert e.reason == reason


class TestApplicationScopeSet:
    def test_to_dict(self):
        e = ApplicationScopeSet(
            session_id="s1",
            application_name="Calculator",
            runtime_id="atspi://calc",
            descendant_count=52,
            discovery_time_ms=8500.0,
        )
        d = e.to_dict()
        assert d["event_type"] == "application_scope_set"
        assert d["application_name"] == "Calculator"
        assert d["descendant_count"] == 52
        assert d["discovery_time_ms"] == 8500.0

    def test_none_descendant_count(self):
        e = ApplicationScopeSet(
            session_id="s1",
            application_name="Unknown",
            runtime_id="rt1",
            descendant_count=None,
            discovery_time_ms=100.0,
        )
        assert e.to_dict()["descendant_count"] is None


class TestXPathTransformed:
    def test_to_dict(self):
        e = XPathTransformed(
            session_id="s1",
            original_xpath="//control:Button[@Name='2']",
            relative_xpath=".//control:Button[@Name='2']",
        )
        d = e.to_dict()
        assert d["event_type"] == "xpath_transformed"
        assert d["original_xpath"] == "//control:Button[@Name='2']"
        assert d["relative_xpath"] == ".//control:Button[@Name='2']"
