"""Unit tests for PlatynUI focus guidance + intent mapping
(change: platynui-focused-execution).

Covers:
* ``RobotFrameworkNativeConverter.get_platynui_locator_guidance()`` exposing
  a ``focus_and_visibility`` chapter.
* ``IntentVerb.ENSURE_FOCUSED`` and its registry mapping to "Activate Window"
  for PlatynUI.BareMetal (and absence for Browser/Selenium).
"""

from __future__ import annotations

from robotmcp.domains.intent.aggregates import IntentRegistry
from robotmcp.domains.intent.value_objects import IntentVerb
from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter


# --------------------------------------------------------------------------
# Guidance: focus_and_visibility chapter
# --------------------------------------------------------------------------


def _guidance():
    return RobotFrameworkNativeConverter().get_platynui_locator_guidance()


def test_guidance_has_focus_and_visibility_chapter():
    g = _guidance()
    assert "focus_and_visibility" in g
    chapter = g["focus_and_visibility"]
    assert "description" in chapter
    assert isinstance(chapter["description"], str)
    assert chapter["description"]


def test_guidance_rules_is_list_of_strings():
    chapter = _guidance()["focus_and_visibility"]
    assert isinstance(chapter["rules"], list)
    assert chapter["rules"]
    assert all(isinstance(r, str) for r in chapter["rules"])


def test_guidance_rules_mention_key_concepts():
    rules_text = " ".join(_guidance()["focus_and_visibility"]["rules"]).lower()
    # app-scoped descriptors
    assert "app-scoped" in rules_text or "/app:*" in rules_text
    # launch visibly
    assert "visibly" in rules_text or "launch" in rules_text
    # no-keyword-before-launch
    assert "before launch" in rules_text or "before launching" in rules_text


def test_guidance_escape_hatch_mentions_env_var():
    chapter = _guidance()["focus_and_visibility"]
    assert "escape_hatch" in chapter
    assert "ROBOTMCP_PLATYNUI_NO_FOCUS" in chapter["escape_hatch"]


# --------------------------------------------------------------------------
# Intent: ENSURE_FOCUSED
# --------------------------------------------------------------------------


def test_ensure_focused_verb_exists():
    assert IntentVerb.ENSURE_FOCUSED.value == "ensure_focused"


def test_platynui_ensure_focused_maps_to_activate_window():
    registry = IntentRegistry.with_builtins()
    mapping = registry.resolve(IntentVerb.ENSURE_FOCUSED, "PlatynUI.BareMetal")
    assert mapping is not None
    assert mapping.keyword == "Activate Window"


def test_browser_does_not_support_ensure_focused():
    registry = IntentRegistry.with_builtins()
    assert registry.resolve(IntentVerb.ENSURE_FOCUSED, "Browser") is None


def test_selenium_does_not_support_ensure_focused():
    registry = IntentRegistry.with_builtins()
    assert registry.resolve(IntentVerb.ENSURE_FOCUSED, "SeleniumLibrary") is None


def test_ensure_focused_only_supported_by_platynui():
    registry = IntentRegistry.with_builtins()
    libs = {
        lib
        for lib in registry.get_supported_libraries()
        if registry.has_mapping(IntentVerb.ENSURE_FOCUSED, lib)
    }
    assert libs == {"PlatynUI.BareMetal"}
