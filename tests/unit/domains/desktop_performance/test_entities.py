"""Tests for desktop_performance entities."""

from unittest.mock import MagicMock

from robotmcp.domains.desktop_performance.entities import (
    ApplicationRoot,
    CachedElement,
)
from robotmcp.domains.desktop_performance.value_objects import CacheKey


class TestCachedElement:
    def _make_element(self, valid=True, xpath="//Button", session_id="s1"):
        mock_node = MagicMock()
        mock_node.is_valid.return_value = valid
        return CachedElement(
            cache_key=CacheKey(xpath=xpath, session_id=session_id),
            xpath_original=xpath,
            xpath_relative="./" + xpath.lstrip("/"),
            descriptor=MagicMock(),
            resolved_node=mock_node,
            application_root_id="root1",
            created_at=100.0,
            last_accessed=100.0,
        )

    def test_is_valid_with_valid_node(self):
        elem = self._make_element(valid=True)
        assert elem.is_valid() is True

    def test_is_valid_with_stale_node(self):
        elem = self._make_element(valid=False)
        assert elem.is_valid() is False

    def test_is_valid_with_none_node(self):
        elem = self._make_element()
        elem.resolved_node = None
        assert elem.is_valid() is False

    def test_is_valid_exception_returns_false(self):
        elem = self._make_element()
        elem.resolved_node.is_valid.side_effect = RuntimeError("AT-SPI error")
        assert elem.is_valid() is False

    def test_record_hit_increments(self):
        elem = self._make_element()
        assert elem.hit_count == 0
        elem.record_hit()
        assert elem.hit_count == 1
        elem.record_hit()
        assert elem.hit_count == 2

    def test_record_hit_updates_last_accessed(self):
        elem = self._make_element()
        old_access = elem.last_accessed
        elem.record_hit()
        assert elem.last_accessed >= old_access

    def test_to_dict_structure(self):
        elem = self._make_element()
        d = elem.to_dict()
        assert "xpath" in d
        assert "xpath_relative" in d
        assert "is_valid" in d
        assert "hit_count" in d
        assert "age_seconds" in d
        assert d["application_root_id"] == "root1"

    def test_to_dict_is_valid_reflects_node_state(self):
        elem = self._make_element(valid=True)
        assert elem.to_dict()["is_valid"] is True
        elem.resolved_node.is_valid.return_value = False
        assert elem.to_dict()["is_valid"] is False

    def test_test_suppression(self):
        assert CachedElement.__test__ is False


class TestApplicationRoot:
    def _make_root(self, valid=True):
        mock_node = MagicMock()
        mock_node.is_valid.return_value = valid
        mock_node.runtime_id = "atspi://1.234/a11y/root"
        return ApplicationRoot(
            session_id="s1",
            application_name="Calculator",
            root_node=mock_node,
            root_runtime_id="atspi://1.234/a11y/root",
            descendant_count=52,
        )

    def test_is_valid_delegates_to_node(self):
        root = self._make_root(valid=True)
        assert root.is_valid() is True

    def test_is_valid_false_when_stale(self):
        root = self._make_root(valid=False)
        assert root.is_valid() is False

    def test_is_valid_with_none_node(self):
        root = self._make_root()
        root.root_node = None
        assert root.is_valid() is False

    def test_is_valid_exception(self):
        root = self._make_root()
        root.root_node.is_valid.side_effect = RuntimeError("gone")
        assert root.is_valid() is False

    def test_to_dict_structure(self):
        root = self._make_root()
        d = root.to_dict()
        assert d["application_name"] == "Calculator"
        assert d["runtime_id"] == "atspi://1.234/a11y/root"
        assert d["descendant_count"] == 52
        assert "is_valid" in d
        assert "age_seconds" in d

    def test_test_suppression(self):
        assert ApplicationRoot.__test__ is False
