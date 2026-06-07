"""Unit tests for PlatynUI explicit library detection (ADR-024 + ADR-025).

Covers:
- _LIBRARY_TOKENS['PlatynUI.BareMetal'].
- LIBRARY_RULES_DEFAULT['PlatynUI.BareMetal'] PatternRule annotations.
- Explicit detection: 'use PlatynUI' -> explicit; 'desktop automation' alone
  -> NOT explicit (mention-only weight).

Run with: uv run pytest tests/unit/test_platynui_newcore_detection.py -q
"""

__test__ = True

import pytest

from robotmcp.utils.library_detection import (
    LIBRARY_RULES_DEFAULT,
    LibraryDetector,
    PatternRule,
    _LIBRARY_TOKENS,
    detect_explicit_library_preference,
    get_library_detector,
)

PLATYNUI = "PlatynUI.BareMetal"


# =============================================================================
# Token table + rules
# =============================================================================


class TestTokensAndRules:
    def test_platynui_tokens_present(self):
        tokens = _LIBRARY_TOKENS[PLATYNUI]
        assert "platynui" in tokens

    def test_library_rules_present(self):
        assert PLATYNUI in LIBRARY_RULES_DEFAULT
        rules = LIBRARY_RULES_DEFAULT[PLATYNUI]
        assert all(isinstance(r, PatternRule) for r in rules)

    def test_has_explicit_and_mention_rules(self):
        rules = LIBRARY_RULES_DEFAULT[PLATYNUI]
        assert any(r.explicit for r in rules)
        assert any(not r.explicit for r in rules)

    def test_desktop_automation_pattern_is_mention_only(self):
        rules = LIBRARY_RULES_DEFAULT[PLATYNUI]
        desktop_rules = [r for r in rules if "desktop" in r.pattern]
        assert desktop_rules
        # domain markers must NOT be explicit
        assert all(not r.explicit for r in desktop_rules)


# =============================================================================
# Explicit detection
# =============================================================================


class TestExplicitDetection:
    def test_use_platynui_is_explicit(self):
        r = detect_explicit_library_preference("Use PlatynUI to automate the app")
        assert r.library == PLATYNUI
        assert r.is_decisive

    def test_platynui_baremetal_verbatim_is_explicit(self):
        r = detect_explicit_library_preference(
            "Drive the desktop with PlatynUI.BareMetal keywords"
        )
        assert r.library == PLATYNUI

    def test_standalone_platynui_brand_is_explicit(self):
        # standalone 'platynui' brand token has weight 8 (>= default_min_score)
        r = get_library_detector().detect_explicit_preference(
            "The test relies on platynui for inspection"
        )
        assert r.library == PLATYNUI

    def test_desktop_automation_alone_not_explicit(self):
        r = detect_explicit_library_preference(
            "Perform desktop automation testing of the calculator UI"
        )
        assert r.library is None, (
            f"'desktop automation' is a domain marker (mention-only) and must "
            f"NOT trigger an explicit PlatynUI preference; got {r.evidence}"
        )

    def test_at_spi_alone_not_explicit(self):
        r = detect_explicit_library_preference(
            "Inspect the accessibility tree via at-spi2"
        )
        assert r.library is None

    def test_evidence_present_when_explicit(self):
        r = detect_explicit_library_preference("Use PlatynUI for this test")
        assert r.evidence
        assert all(ev["library"] == PLATYNUI for ev in r.evidence)

    def test_source_is_rule(self):
        r = detect_explicit_library_preference("Use PlatynUI for this test")
        assert r.source == "rule"


# =============================================================================
# Mention layer still scores domain markers (sanity vs explicit layer)
# =============================================================================


class TestMentionLayer:
    def test_desktop_automation_scores_in_mention_layer(self):
        detector = LibraryDetector()
        scores = detector.get_scores("desktop automation testing of an app")
        assert scores.get(PLATYNUI, 0) > 0


# =============================================================================
# Library registry — DESKTOP category + no PlatynUI validation warnings
# =============================================================================


class TestLibraryRegistry:
    def test_desktop_category_exists(self):
        from robotmcp.config.library_registry import LibraryCategory

        assert LibraryCategory.DESKTOP.value == "desktop"

    def test_desktop_category_roundtrips_from_string(self):
        from robotmcp.config.library_registry import LibraryCategory

        assert LibraryCategory("desktop") is LibraryCategory.DESKTOP

    def test_no_platynui_validation_errors(self):
        from robotmcp.config import library_registry

        errors = list(getattr(library_registry, "_VALIDATION_ERRORS", []))
        platynui_errors = [
            e for e in errors if "platynui" in str(e).lower()
        ]
        assert platynui_errors == [], (
            f"PlatynUI must not introduce registry validation errors "
            f"(e.g. duplicate load_priority): {platynui_errors}"
        )
