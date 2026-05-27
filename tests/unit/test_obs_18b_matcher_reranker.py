"""OBS-18B — Action-class reranker for the semantic matcher.

Implements the OBS-18A v2 design: classify query + each keyword into
9 action classes; down-weight mismatches; cap top-match confidence
under Triggers A (top-3 divergent) or B (unknown query + opinionated
top match above cap).

These tests pin:
1. ``classify_keyword_action`` deterministic + v2 precedence-correct
   for real Browser tags (verified against ``LibraryDocumentation('Browser')``
   sample outputs from OBS-18A v2 worked-examples table).
2. ``_classify_query_action_class`` recognises all 9 query intents.
3. ``apply_action_class_reranker`` down-weights mismatches; abstains
   on ``unknown`` query.
4. ``apply_confidence_cap`` fires under Trigger A (divergent top-3)
   and Trigger B (unknown query + opinionated high-confidence top).
5. End-to-end: S02 outcome — top match becomes ``Select From List By
   Label`` (the AC #4a fix).
6. End-to-end: S10 outcome — ``low_confidence_top_match: true``
   surfaces in the response (the AC #4b fix via Trigger B).
7. Feature flag ``ROBOTMCP_MATCHER_RERANK=0`` disables the reranker
   (rollback path).
8. ``KeywordMatch.tags`` field carries tags through.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.components.keyword_matcher import (
    KeywordMatch,
    _classify_query_action_class,
    apply_action_class_reranker,
    apply_confidence_cap,
    classify_keyword_action,
)


def _mk_match(name, library, confidence, tags=None):
    return KeywordMatch(
        keyword_name=name, library=library, confidence=confidence,
        arguments=[], argument_types=[], documentation="",
        usage_example=None, tags=list(tags or []),
    )


# ---------------------------------------------------------------------------
# Classifier — verified against actual Browser library tag distribution
# ---------------------------------------------------------------------------


class TestKeywordClassifier:
    """v2 precedence-correct classifier for real Browser tags."""

    @pytest.mark.parametrize("name,tags,expected", [
        # Browser keywords — verified actual tags
        ("Click",                  ["PageContent", "Setter"], "click"),
        ("Fill Text",              ["PageContent", "Setter"], "fill"),
        ("Type Text",              ["PageContent", "Setter"], "fill"),
        ("Press Keys",             ["PageContent", "Setter"], "fill"),
        ("Go To",                  ["BrowserControl", "Setter"], "navigate"),
        ("New Page",               ["BrowserControl", "Setter"], "navigate"),
        ("Select Options By",      ["PageContent", "Setter"], "select"),
        ("Wait For Elements State",["PageContent", "Wait"], "wait"),
        # Browser getter-asserts — getter wins over assertion (v1 had bug here)
        ("Get Text",               ["Assertion", "Getter", "PageContent"], "query"),
        ("Get Element Count",      ["Assertion", "Getter", "PageContent"], "query"),
        # SL keywords — no tags, name-pattern fallback
        ("Click Element",          [], "click"),
        ("Click Button",           [], "click"),
        ("Click Link",             [], "click"),
        ("Input Text",             [], "fill"),
        ("Input Password",         [], "fill"),
        ("Open Browser",           [], "navigate"),
        ("Go To",                  [], "navigate"),
        ("Select From List By Label",   [], "select"),
        ("Select From List By Value",   [], "select"),
        ("Deselect All From List",      [], "select"),
        ("Get Text",               [], "query"),
        ("Get Element Attribute",  [], "query"),
        ("Wait Until Element Is Visible", [], "wait"),
        ("Page Should Contain",    [], "assert"),
        ("Element Should Be Visible", [], "assert"),
        # BuiltIn keywords
        ("Should Be Equal",        [], "assert"),
        ("Should Contain",         [], "assert"),
        ("Run Keyword If",         [], "control"),
        ("Repeat Keyword",         [], "control"),
        ("Sleep",                  [], "wait"),
        ("Get Length",             [], "query"),
        # Unknown / unclassifiable
        ("Set Window Position",    [], "unknown"),  # SL — not in patterns
        ("My Custom Keyword",      [], "unknown"),
        ("",                       [], "unknown"),
    ])
    def test_classifier(self, name, tags, expected):
        assert classify_keyword_action(name, tags) == expected, (
            f"{name} {tags!r} → expected {expected}, "
            f"got {classify_keyword_action(name, tags)}"
        )

    def test_none_tags_handled(self):
        # Robustness: None entry in tags must not crash
        assert classify_keyword_action("Click", [None, "Setter"]) == "click"

    def test_case_insensitive_tag_matching(self):
        # Lower or upper case tags both match (e.g., "wait" vs "Wait")
        assert classify_keyword_action("X", ["WAIT"]) == "wait"
        assert classify_keyword_action("X", ["wait"]) == "wait"

    def test_space_normalised_tag_matching(self):
        # Browser has one outlier "Page Content" with space; treat as same tag
        assert classify_keyword_action(
            "Get Element States", ["Page Content", "Getter", "Assertion"],
        ) == "query"


class TestQueryClassifier:
    @pytest.mark.parametrize("query,expected", [
        ("click button", "click"),
        ("press the submit button", "click"),
        ("tap the link", "click"),
        ("fill the form field", "fill"),
        ("type into the input", "fill"),
        ("enter username", "fill"),
        ("input text value", "fill"),
        ("set value to 10", "fill"),
        ("go to url", "navigate"),
        ("navigate to page", "navigate"),
        ("visit website", "navigate"),
        ("open page", "navigate"),
        ("select dropdown option by label", "select"),
        ("choose dropdown value", "select"),
        ("verify text is shown", "assert"),
        ("should contain message", "assert"),
        ("expect element visible", "assert"),
        ("wait until loaded", "wait"),
        ("wait for element", "wait"),
        ("sleep 2 seconds", "wait"),
        ("get element text", "query"),
        ("read the value", "query"),
        ("fetch attribute", "query"),
        ("iterate over rows", "control"),
        ("for each item", "control"),
        # Unknown — API queries, ambiguous prose
        ("send http post request with json body", "unknown"),
        ("banana telephone", "unknown"),
        ("", "unknown"),
    ])
    def test_query_classification(self, query, expected):
        assert _classify_query_action_class(query) == expected


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class TestReranker:
    """apply_action_class_reranker down-weights mismatches; abstains
    on unknown."""

    def test_match_class_keeps_confidence(self):
        matches = [
            _mk_match("Click", "Browser", 0.85,
                      tags=["PageContent", "Setter"]),
        ]
        out = apply_action_class_reranker(matches, "click")
        assert out[0].confidence == 0.85

    def test_mismatch_class_downweighted(self):
        matches = [
            _mk_match("Element Should Be Visible", "SeleniumLibrary", 0.87),
        ]
        # query is "select" — keyword class is "assert" → mismatch
        out = apply_action_class_reranker(matches, "select")
        # Default downweight 0.6
        assert abs(out[0].confidence - 0.87 * 0.6) < 1e-6

    def test_unknown_query_abstains(self):
        matches = [
            _mk_match("Click", "Browser", 0.85,
                      tags=["PageContent", "Setter"]),
            _mk_match("Get Text", "Browser", 0.7,
                      tags=["Assertion", "Getter", "PageContent"]),
        ]
        out = apply_action_class_reranker(matches, "unknown")
        # No change — reranker abstains
        confs = [m.confidence for m in out]
        assert confs == [0.85, 0.7]

    def test_s02_outcome(self):
        """S02 — query: 'select dropdown option by visible label'
        (class: select). The pre-rerank winner ``Element Should Be
        Visible`` (assert class) should be down-weighted below the
        ``Select From List By Label`` (select class) at 0.82.
        """
        matches = [
            _mk_match("Element Should Be Visible", "SeleniumLibrary", 0.87),
            _mk_match("Select From List By Label", "SeleniumLibrary", 0.82),
            _mk_match("Set Window Position", "SeleniumLibrary", 0.82),
            _mk_match("Unselect From List By Label", "SeleniumLibrary", 0.80),
        ]
        # Force the reranker (env var defaults ON in newer state but
        # tests run with whatever the environment provides; pass class
        # directly to bypass).
        out = apply_action_class_reranker(matches, "select")
        # Top match becomes Select From List By Label
        assert out[0].keyword_name == "Select From List By Label", (
            f"S02 outcome — expected Select From List By Label as top, "
            f"got {out[0].keyword_name}"
        )

    def test_downweight_env_var_tunable(self, monkeypatch):
        monkeypatch.setenv("ROBOTMCP_RERANK_DOWNWEIGHT", "0.3")
        matches = [
            _mk_match("Element Should Be Visible", "SeleniumLibrary", 1.0),
        ]
        out = apply_action_class_reranker(matches, "select")
        # 1.0 * 0.3 = 0.3
        assert abs(out[0].confidence - 0.3) < 1e-6


class TestConfidenceCap:
    """apply_confidence_cap fires under Trigger A or Trigger B."""

    def test_trigger_a_top3_divergent(self):
        """Top-3 spans ≥ 3 distinct classes → cap fires."""
        matches = [
            _mk_match("Click", "Browser", 0.8,
                      tags=["PageContent", "Setter"]),  # click
            _mk_match("Get Text", "Browser", 0.75,
                      tags=["Assertion", "Getter", "PageContent"]),  # query
            _mk_match("Wait For Elements State", "Browser", 0.7,
                      tags=["PageContent", "Wait"]),  # wait
        ]
        out, flag = apply_confidence_cap(matches, "click")
        assert flag is True
        assert out[0].confidence == 0.5  # capped

    def test_trigger_b_unknown_query_opinionated_top(self):
        """Query class unknown + top has opinionated class + conf > cap
        → cap fires (the S10 fix)."""
        matches = [
            _mk_match("New Persistent Context", "Browser", 0.72,
                      tags=["BrowserControl", "Setter"]),  # navigate
        ]
        out, flag = apply_confidence_cap(matches, "unknown")
        assert flag is True
        assert out[0].confidence == 0.5

    def test_no_trigger_no_cap(self):
        """Top-3 all same class + query matches → no cap."""
        matches = [
            _mk_match("Click", "Browser", 0.85,
                      tags=["PageContent", "Setter"]),
            _mk_match("Click Element", "SeleniumLibrary", 0.80),
            _mk_match("Click Button", "SeleniumLibrary", 0.75),
        ]
        out, flag = apply_confidence_cap(matches, "click")
        assert flag is False
        assert out[0].confidence == 0.85  # unchanged

    def test_top_already_at_cap_still_flags(self):
        """Trigger B fires even if top is at cap — agent learns
        about the uncertainty either way."""
        matches = [
            _mk_match("X", "Browser", 0.5, tags=["PageContent", "Setter"]),
        ]
        out, flag = apply_confidence_cap(matches, "unknown")
        assert flag is True  # flag still set
        assert out[0].confidence == 0.5  # but no actual change

    def test_cap_env_var_tunable(self, monkeypatch):
        monkeypatch.setenv("ROBOTMCP_RERANK_CAP", "0.3")
        matches = [
            _mk_match("X", "Browser", 0.5, tags=["PageContent", "Setter"]),
        ]
        out, flag = apply_confidence_cap(matches, "unknown")
        assert flag is True
        assert out[0].confidence == 0.3


# ---------------------------------------------------------------------------
# End-to-end through discover_keywords
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDiscoverKeywordsIntegration:
    """The reranker is wired into the matcher pipeline. End-to-end
    behaviour pinned via mocked _pattern_based_matching etc."""

    async def _matcher_with_synthetic_matches(self, fake_ranked):
        from robotmcp.components.keyword_matcher import KeywordMatcher
        m = KeywordMatcher()
        m._initialized = True

        async def _ensure():
            return None
        m._ensure_initialized = _ensure
        m._pattern_based_matching = AsyncMock(return_value=[])
        m._context_aware_matching = AsyncMock(return_value=[])
        m._deduplicate_matches = MagicMock(return_value=fake_ranked)
        m._rank_matches = MagicMock(return_value=fake_ranked)
        m._normalize_action = MagicMock(side_effect=lambda x: x.lower())
        m._classify_action = MagicMock(return_value="click")
        m._generate_usage_recommendations = MagicMock(return_value=[])
        return m

    async def test_s02_top_match_changes_to_select(self, monkeypatch):
        monkeypatch.setenv("ROBOTMCP_MATCHER_RERANK", "1")
        fake = [
            _mk_match("Element Should Be Visible", "SeleniumLibrary", 0.87),
            _mk_match("Select From List By Label", "SeleniumLibrary", 0.82),
            _mk_match("Set Window Position", "SeleniumLibrary", 0.82),
        ]
        m = await self._matcher_with_synthetic_matches(fake)
        result = await m.discover_keywords(
            "select dropdown option by visible label", limit=5,
        )
        assert result["matches"][0]["keyword_name"] == "Select From List By Label"

    async def test_s10_low_confidence_top_match_flag(self, monkeypatch):
        monkeypatch.setenv("ROBOTMCP_MATCHER_RERANK", "1")
        fake = [
            _mk_match("New Persistent Context", "Browser", 0.72,
                      tags=["BrowserControl", "Setter"]),
        ]
        m = await self._matcher_with_synthetic_matches(fake)
        result = await m.discover_keywords(
            "send http post request with json body", limit=5,
        )
        # Trigger B fires (unknown query + opinionated top + conf > cap)
        assert result.get("low_confidence_top_match") is True
        assert result["matches"][0]["confidence"] == 0.5

    async def test_feature_flag_off_no_reranker(self, monkeypatch):
        """ROBOTMCP_MATCHER_RERANK=0 disables the reranker entirely
        (rollback path). Top match stays as pre-reranker ranking."""
        monkeypatch.setenv("ROBOTMCP_MATCHER_RERANK", "0")
        fake = [
            _mk_match("Element Should Be Visible", "SeleniumLibrary", 0.87),
            _mk_match("Select From List By Label", "SeleniumLibrary", 0.82),
        ]
        m = await self._matcher_with_synthetic_matches(fake)
        result = await m.discover_keywords(
            "select dropdown option by visible label",
        )
        # Reranker disabled — pre-rerank top remains
        assert result["matches"][0]["keyword_name"] == "Element Should Be Visible"
        # Cap also disabled
        assert "low_confidence_top_match" not in result

    async def test_feature_flag_default_on(self, monkeypatch):
        """Default (env unset) → reranker active."""
        monkeypatch.delenv("ROBOTMCP_MATCHER_RERANK", raising=False)
        fake = [
            _mk_match("Element Should Be Visible", "SeleniumLibrary", 0.87),
            _mk_match("Select From List By Label", "SeleniumLibrary", 0.82),
        ]
        m = await self._matcher_with_synthetic_matches(fake)
        result = await m.discover_keywords(
            "select dropdown option by visible label",
        )
        assert result["matches"][0]["keyword_name"] == "Select From List By Label"


class TestKeywordMatchTagsField:
    """KeywordMatch.tags is the OBS-18B plumbing — verify it's
    populated and propagates."""

    def test_tags_field_default_empty(self):
        m = _mk_match("X", "Browser", 0.5)
        assert m.tags == []

    def test_tags_field_carries_through(self):
        m = _mk_match("Click", "Browser", 0.85,
                      tags=["PageContent", "Setter"])
        assert m.tags == ["PageContent", "Setter"]

    def test_dataclasses_replace_preserves_tags(self):
        import dataclasses
        m = _mk_match("Click", "Browser", 0.85,
                      tags=["PageContent", "Setter"])
        m2 = dataclasses.replace(m, confidence=0.5)
        assert m2.tags == ["PageContent", "Setter"]
        assert m2.confidence == 0.5


# ---------------------------------------------------------------------------
# Regression — pin existing-correct scenarios stay correct
# ---------------------------------------------------------------------------


class TestRegressionBaselines:
    """Per OBS-18A v2 "Pinned regression baselines": the reranker
    must NOT regress existing-correct scenarios."""

    @pytest.mark.parametrize("query,expected_top_class", [
        # Each query's class corresponds to a correct top-match's class
        ("click button", "click"),
        ("fill form input field", "fill"),
        ("navigate to url page", "navigate"),
        ("wait for element visible", "wait"),
        ("get element text", "query"),
    ])
    def test_query_class_matches_expected(self, query, expected_top_class):
        assert _classify_query_action_class(query) == expected_top_class


# ---------------------------------------------------------------------------
# Real end-to-end regression baseline tests (post-Wave-3 review fix)
#
# Codex + Claude subagent flagged the parametrised classifier tests
# above as tautological — they only verify the query classifier, not
# that the actual ``discover_keywords`` pipeline produces the right
# top match. This class pins top-match name+library against the REAL
# matcher with the REAL Browser/SeleniumLibrary keyword data. If
# anyone changes the reranker, classifier, or matcher pipeline in a
# way that breaks the OBS-18A v2 pinned baselines, these tests fail.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPinnedTopMatchEndToEnd:
    """Real end-to-end pinning of top-match name+library for the OBS-18A
    v2 pinned baselines. Uses the real matcher (no mocks of internals)
    so the test exercises the actual production code path."""

    async def _run(self, query, library_name=None, monkeypatch=None):
        if monkeypatch is not None:
            monkeypatch.setenv("ROBOTMCP_MATCHER_RERANK", "1")
        from robotmcp.server import find_keywords as fk
        fn = getattr(fk, "fn", fk)
        kwargs = {"query": query, "strategy": "semantic", "limit": 5}
        if library_name:
            kwargs["library_name"] = library_name
        return await fn(**kwargs)

    async def test_S01_click_button_browser_top_is_click(self, monkeypatch):
        """S01-style: 'click button' + Browser → top match in click class.
        Note benchmark S01 actually uses 'select dropdown' query; this
        test uses the design's pinned 'click button' query directly."""
        r = await self._run("click button", "Browser", monkeypatch)
        matches = r["result"]["matches"]
        assert matches, "expected at least one match"
        top = matches[0]
        # Top match must be from Browser, click-class
        assert top["library"] == "Browser"
        assert classify_keyword_action(top["keyword_name"], top.get("tags", [])) == "click", (
            f"Expected click-class top match for 'click button' query; "
            f"got {top['keyword_name']} ({classify_keyword_action(top['keyword_name'], top.get('tags', []))})"
        )

    async def test_S02_select_dropdown_sl_top_is_select_from_list(self, monkeypatch):
        """S02 (the OBS-18B fix target): query
        'select dropdown option by visible label' + SeleniumLibrary →
        top match MUST be ``Select From List By Label``."""
        r = await self._run(
            "select dropdown option by visible label",
            "SeleniumLibrary", monkeypatch,
        )
        matches = r["result"]["matches"]
        assert matches
        assert matches[0]["keyword_name"] == "Select From List By Label", (
            f"S02 regression: top match must be Select From List By Label; "
            f"got {matches[0]['keyword_name']}"
        )

    async def test_S06_click_browser_top_is_click(self, monkeypatch):
        """S06: single-word 'click' + Browser → top match Browser.Click."""
        r = await self._run("click", "Browser", monkeypatch)
        matches = r["result"]["matches"]
        assert matches
        assert matches[0]["library"] == "Browser"
        assert matches[0]["keyword_name"] == "Click", (
            f"S06 regression: 'click' + Browser top must be Click; "
            f"got {matches[0]['keyword_name']}"
        )

    async def test_S07_navigate_browser_top_is_go_to(self, monkeypatch):
        """S07: 'navigate' + Browser → top match Browser.Go To."""
        r = await self._run("navigate", "Browser", monkeypatch)
        matches = r["result"]["matches"]
        assert matches
        assert matches[0]["library"] == "Browser"
        assert matches[0]["keyword_name"] == "Go To", (
            f"S07 regression: 'navigate' + Browser top must be Go To; "
            f"got {matches[0]['keyword_name']}"
        )

    async def test_S10_api_query_browser_caps_with_flag(self, monkeypatch):
        """S10 (OBS-18B AC #4b): cross-domain 'send http post request' +
        Browser must produce ``low_confidence_top_match: true`` AND a
        matches[0].confidence ≤ 0.5 (capped by Trigger B)."""
        r = await self._run(
            "send http post request with json body",
            "Browser", monkeypatch,
        )
        res = r["result"]
        assert res.get("low_confidence_top_match") is True, (
            "S10 must set low_confidence_top_match=True"
        )
        assert res["matches"][0]["confidence"] <= 0.5, (
            f"S10 top-match confidence must be ≤ 0.5; "
            f"got {res['matches'][0]['confidence']}"
        )

    async def test_S12_compound_wait_click_top_stays_click_class(self, monkeypatch):
        """S12 (Wave-3 round-1 review FIX): compound query with both
        'wait' and 'click' triggers must NOT regress click-class top
        match. Pre-fix this query returned ``Wait For Condition``
        because of trigger ordering. Post-fix it returns Click (the
        reranker abstains on ambiguous compound queries)."""
        r = await self._run(
            "wait for the modal dialog to appear and then click the "
            "confirm button at the bottom of the form",
            "Browser", monkeypatch,
        )
        matches = r["result"]["matches"]
        assert matches
        top_class = classify_keyword_action(
            matches[0]["keyword_name"], matches[0].get("tags", []),
        )
        # Top match must be click-class (NOT wait-class) — compound
        # queries are ambiguous; the reranker abstains; the pre-rerank
        # matcher's top is click-class.
        assert top_class == "click", (
            f"S12 regression: compound 'wait... click...' top must stay "
            f"click-class; got {matches[0]['keyword_name']} ({top_class})"
        )

    async def test_S13_bdd_click_top_is_click(self, monkeypatch):
        """S13: BDD-prefixed 'When I click submit button' + Browser →
        top match in click class (BDD prefix stripped before
        classification)."""
        r = await self._run("When I click submit button", "Browser", monkeypatch)
        matches = r["result"]["matches"]
        assert matches
        top_class = classify_keyword_action(
            matches[0]["keyword_name"], matches[0].get("tags", []),
        )
        assert top_class == "click", (
            f"S13 regression: BDD 'When I click...' must produce click-class top; "
            f"got {matches[0]['keyword_name']} ({top_class})"
        )
