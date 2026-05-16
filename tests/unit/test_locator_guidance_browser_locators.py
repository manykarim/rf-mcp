"""G3: Tests verifying the browser_locators cookbook content.

Each of the 8 required entries (a-h) is checked for required fields
and canonical pattern presence.
"""
from __future__ import annotations

import pytest

from robotmcp.domains.locator_guidance.services import LocatorTopicService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def svc() -> LocatorTopicService:
    return LocatorTopicService()


@pytest.fixture(scope="module")
def cookbook(svc: LocatorTopicService) -> dict:
    return svc.get_browser_locators()


@pytest.fixture(scope="module")
def entries(cookbook: dict) -> list[dict]:
    return cookbook.get("entries", [])


# ---------------------------------------------------------------------------
# Top-level structure
# ---------------------------------------------------------------------------

def test_success_flag(cookbook):
    assert cookbook.get("success") is True


def test_topic_field(cookbook):
    assert cookbook.get("topic") == "browser_locators"


def test_at_least_8_entries(entries):
    assert len(entries) >= 8, f"Expected at least 8 entries, got {len(entries)}"


def test_see_also_contains_spa_wizards(cookbook):
    see_also = cookbook.get("see_also", [])
    assert "spa_wizards" in see_also


# ---------------------------------------------------------------------------
# Per-entry field requirements
# ---------------------------------------------------------------------------

class TestRequiredFields:
    """Every entry must have title, locator_template, example, use_when."""

    REQUIRED = {"title", "locator_template", "example", "use_when"}

    def test_all_entries_have_required_fields(self, entries):
        for i, entry in enumerate(entries):
            missing = self.REQUIRED - set(entry.keys())
            assert not missing, f"Entry {i} ('{entry.get('title')}') missing fields: {missing}"

    def test_all_fields_non_empty(self, entries):
        for i, entry in enumerate(entries):
            for key in self.REQUIRED:
                value = entry.get(key, "")
                assert value, f"Entry {i} ('{entry.get('title')}') field '{key}' is empty"


# ---------------------------------------------------------------------------
# Entry (a): Wrapper-label click
# ---------------------------------------------------------------------------

class TestEntryWrapperLabelClick:
    def _find(self, entries):
        return next((e for e in entries if "*css=label >> id=" in e.get("locator_template", "")), None)

    def test_entry_exists(self, entries):
        assert self._find(entries) is not None, "Entry (a) wrapper-label click not found"

    def test_locator_template_pattern(self, entries):
        entry = self._find(entries)
        assert entry is not None
        assert "*css=label >> id=" in entry["locator_template"]

    def test_example_contains_pattern(self, entries):
        entry = self._find(entries)
        assert entry is not None
        assert "*css=label >> id=" in entry["example"]

    def test_example_uses_click(self, entries):
        entry = self._find(entries)
        assert entry is not None
        assert "Click" in entry["example"]


# ---------------------------------------------------------------------------
# Entry (b): Wrapper-label check
# ---------------------------------------------------------------------------

class TestEntryWrapperLabelCheck:
    def _find(self, entries):
        # Both (a) and (b) share the template; differentiate by example keyword
        return next(
            (
                e for e in entries
                if "*css=label >> id=" in e.get("locator_template", "")
                and "Check Checkbox" in e.get("example", "")
            ),
            None,
        )

    def test_entry_exists(self, entries):
        assert self._find(entries) is not None, "Entry (b) wrapper-label check not found"


# ---------------------------------------------------------------------------
# Entry (b2): Sibling label by for= (Bootstrap form-check)
# ---------------------------------------------------------------------------

class TestEntrySiblingForLabel:
    def _find(self, entries):
        return next(
            (
                e for e in entries
                if "label[for=" in e.get("locator_template", "")
            ),
            None,
        )

    def test_entry_exists(self, entries):
        assert self._find(entries) is not None, (
            "Sibling-for label entry not found — Bootstrap form-check pattern is missing"
        )

    def test_pattern_is_attribute_selector_not_descendant(self, entries):
        """The sibling pattern must NOT use '>> id=' (which is for wrapping)."""
        entry = self._find(entries)
        assert entry is not None
        assert "label[for=" in entry["locator_template"]
        assert ">> id=" not in entry["locator_template"], (
            "Sibling-label entry must use attribute-selector form, not descendant chain"
        )


# ---------------------------------------------------------------------------
# Entry (c): Click visible label by text
# ---------------------------------------------------------------------------

class TestEntryClickLabelByText:
    def _find(self, entries):
        return next(
            (
                e for e in entries
                if e.get("locator_template", "").startswith("text=")
            ),
            None,
        )

    def test_entry_exists(self, entries):
        assert self._find(entries) is not None, "Entry (c) click visible label by text not found"

    def test_example_uses_text_locator(self, entries):
        entry = self._find(entries)
        assert entry is not None
        assert "text=" in entry["example"]


# ---------------------------------------------------------------------------
# Entry (d): Visibility-scoped CSS chain
# ---------------------------------------------------------------------------

class TestEntryVisibilityScopedCSS:
    PATTERN = 'section[style="display: block;"]'

    def _find(self, entries):
        return next(
            (e for e in entries if self.PATTERN in e.get("locator_template", "")),
            None,
        )

    def test_entry_exists(self, entries):
        assert self._find(entries) is not None, (
            f"Entry (d) visibility-scoped CSS ({self.PATTERN}) not found"
        )

    def test_pattern_in_locator_template(self, entries):
        entry = self._find(entries)
        assert entry is not None
        assert self.PATTERN in entry["locator_template"]

    def test_pattern_in_example(self, entries):
        entry = self._find(entries)
        assert entry is not None
        assert self.PATTERN in entry["example"]


# ---------------------------------------------------------------------------
# Entry (e): Sibling input by label text
# ---------------------------------------------------------------------------

class TestEntrySiblingInput:
    def _find(self, entries):
        return next(
            (
                e for e in entries
                if ">> .. >> input" in e.get("locator_template", "")
            ),
            None,
        )

    def test_entry_exists(self, entries):
        assert self._find(entries) is not None, "Entry (e) sibling input not found"

    def test_fill_text_keyword_in_example(self, entries):
        entry = self._find(entries)
        assert entry is not None
        assert "Fill Text" in entry["example"]


# ---------------------------------------------------------------------------
# Entry (f): Value-attribute on grouped radio/checkbox
# ---------------------------------------------------------------------------

class TestEntryValueAttribute:
    def _find(self, entries):
        return next(
            (
                e for e in entries
                if "css=[value=" in e.get("locator_template", "")
            ),
            None,
        )

    def test_entry_exists(self, entries):
        assert self._find(entries) is not None, "Entry (f) value-attribute not found"

    def test_uses_label_wrapper(self, entries):
        entry = self._find(entries)
        assert entry is not None
        assert "*css=label" in entry["locator_template"]


# ---------------------------------------------------------------------------
# Entry (g): Strict-mode disambiguation via nth
# ---------------------------------------------------------------------------

class TestEntryNthDisambiguation:
    def _find(self, entries):
        return next(
            (
                e for e in entries
                if ">> nth=" in e.get("locator_template", "")
            ),
            None,
        )

    def test_entry_exists(self, entries):
        assert self._find(entries) is not None, "Entry (g) nth disambiguation not found"

    def test_example_contains_nth(self, entries):
        entry = self._find(entries)
        assert entry is not None
        assert "nth=" in entry["example"]


# ---------------------------------------------------------------------------
# Entry (h): Network-await pattern
# ---------------------------------------------------------------------------

class TestEntryNetworkAwait:
    def _find(self, entries):
        return next(
            (
                e for e in entries
                if "Promise To Wait For Response" in e.get("locator_template", "")
            ),
            None,
        )

    def test_entry_exists(self, entries):
        assert self._find(entries) is not None, "Entry (h) network-await not found"

    def test_example_contains_wait_for(self, entries):
        entry = self._find(entries)
        assert entry is not None
        assert "Wait For" in entry["example"]


# ---------------------------------------------------------------------------
# Service SUPPORTED_TOPICS
# ---------------------------------------------------------------------------

def test_browser_locators_in_supported_topics(svc):
    assert "browser_locators" in svc.SUPPORTED_TOPICS


def test_get_topic_dispatch(svc):
    result = svc.get_topic("browser_locators")
    assert result is not None
    assert result.get("success") is True
    assert result.get("topic") == "browser_locators"


def test_get_topic_unknown_returns_none(svc):
    result = svc.get_topic("nonexistent_topic")
    assert result is None
