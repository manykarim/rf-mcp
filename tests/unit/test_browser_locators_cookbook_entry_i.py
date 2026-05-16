"""Tests for cookbook entry (i): Force-click hidden element.

Verifies that the 9th cookbook entry is present and has the required content.
"""
from __future__ import annotations

__test__ = True

import pytest

from robotmcp.domains.locator_guidance.services import LocatorTopicService


@pytest.fixture(scope="module")
def svc() -> LocatorTopicService:
    return LocatorTopicService()


@pytest.fixture(scope="module")
def cookbook(svc: LocatorTopicService) -> dict:
    return svc.get_browser_locators()


@pytest.fixture(scope="module")
def entries(cookbook: dict) -> list[dict]:
    return cookbook.get("entries", [])


@pytest.fixture(scope="module")
def entry_i(entries: list[dict]) -> dict:
    """The force-click entry — located by title rather than fixed index so
    the cookbook can grow without breaking these tests.
    """
    force_entries = [
        e for e in entries
        if "force-click" in (e.get("title") or "").lower()
        or "Force-click" in (e.get("title") or "")
    ]
    assert force_entries, (
        f"No force-click entry found among {len(entries)} entries"
    )
    return force_entries[0]


def test_nine_entries_present(entries):
    assert len(entries) >= 9, f"Expected at least 9 cookbook entries, got {len(entries)}"


def test_entry_i_title_contains_force_click(entry_i):
    title = entry_i.get("title", "")
    assert "Force-click" in title or "force-click" in title.lower(), (
        f"Entry i title should contain 'Force-click', got: {title!r}"
    )


def test_entry_i_locator_template_starts_with_click_with_options(entry_i):
    template = entry_i.get("locator_template", "")
    assert template.startswith("Click With Options"), (
        f"Entry i locator_template should start with 'Click With Options', got: {template!r}"
    )


def test_entry_i_example_contains_force_true(entry_i):
    example = entry_i.get("example", "")
    assert "force=True" in example, (
        f"Entry i example should contain 'force=True', got: {example!r}"
    )


def test_entry_i_example_contains_click_with_options(entry_i):
    example = entry_i.get("example", "")
    assert "Click With Options" in example, (
        f"Entry i example should contain 'Click With Options', got: {example!r}"
    )


def test_entry_i_use_when_mentions_fallback(entry_i):
    use_when = entry_i.get("use_when", "")
    assert "FALLBACK" in use_when or "fallback" in use_when.lower(), (
        f"Entry i use_when should mention fallback, got: {use_when!r}"
    )


def test_entry_i_use_when_mentions_not_visible(entry_i):
    use_when = entry_i.get("use_when", "")
    assert "not visible" in use_when or "pre-validation" in use_when, (
        f"Entry i use_when should mention visibility failure, got: {use_when!r}"
    )


def test_entry_i_has_all_required_fields(entry_i):
    for field in ("title", "locator_template", "example", "use_when"):
        assert field in entry_i and entry_i[field], (
            f"Entry i missing required field: {field!r}"
        )


def test_get_topic_browser_locators_returns_at_least_9_entries(svc):
    result = svc.get_topic("browser_locators")
    entries = result.get("entries", [])
    assert len(entries) >= 9, (
        f"get_topic('browser_locators') returned {len(entries)} entries, expected >= 9"
    )
