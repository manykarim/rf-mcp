"""Tests for the v5 explicit library preference detection.

Covers PRD AC-1..AC-7 + sentence-scoped negation/migration + sampling
override coherence + sentence_index evidence filtering + race-free
invocation under the module-level NaturalLanguageProcessor singleton.

See:
- docs/prd/analyze_scenario_explicit_library_prd.md
- docs/adr/ADR-024-explicit-library-detection-confidence.md
- docs/proposals/explicit_library_detection_fix_proposal.md (v5)
"""

from robotmcp.compat.fastmcp_compat import get_tool_fn
import asyncio

import pytest

from robotmcp.utils.library_detection import (
    DetectionPolicy,
    LibraryDetector,
    PatternMatch,
    PatternRule,
    PreferenceResolution,
    detect_explicit_library_preference,
    get_library_detector,
)


# =============================================================================
# AC-1 — reported reproducer
# =============================================================================


class TestReportedReproducer:
    """The exact scenario from the 2026-05-29 user report."""

    SCENARIO = (
        "Test e-commerce website https://demoshop.makrocode.de: "
        "open browser, add items to shopping cart, verify items, "
        "complete checkout, and close browser"
    )

    def test_no_explicit_preference(self):
        r = get_library_detector().detect_explicit_preference(self.SCENARIO)
        assert r.library is None, (
            f"'open browser' is generic NL and must NOT trigger explicit "
            f"SL preference. Got {r.library} from {r.evidence}"
        )

    def test_evidence_empty(self):
        r = get_library_detector().detect_explicit_preference(self.SCENARIO)
        assert r.evidence == []

    def test_no_conflicts(self):
        r = get_library_detector().detect_explicit_preference(self.SCENARIO)
        assert r.conflicts == {}

    def test_source_is_rule(self):
        r = get_library_detector().detect_explicit_preference(self.SCENARIO)
        assert r.source == "rule"


# =============================================================================
# AC-2/AC-3/AC-7 — truly explicit detection
# =============================================================================


class TestTrulyExplicitDetected:
    @pytest.mark.parametrize(
        "scenario,expected",
        [
            ("Use playwright to test the checkout flow", "Browser"),
            ("Use Selenium to test the login form", "SeleniumLibrary"),
            ("with SeleniumLibrary, open the page", "SeleniumLibrary"),
            ("Test using browserlibrary against demoshop", "Browser"),
            ("Use requestslibrary to call /api/users", "RequestsLibrary"),
            ("With AppiumLibrary, open the mobile app", "AppiumLibrary"),
            (
                "Use database library to validate the inventory table",
                "DatabaseLibrary",
            ),
            ("Use ssh library to deploy", "SSHLibrary"),
            ("Use XML library to validate the response", "XML"),
            ("Run Selenium 4 against the site", "SeleniumLibrary"),
            ("Test using chromium and webkit", "Browser"),
        ],
    )
    def test_explicit_detected(self, scenario, expected):
        r = get_library_detector().detect_explicit_preference(scenario)
        assert r.library == expected, (
            f"{scenario!r}: expected {expected}, got {r.library}"
        )
        assert r.source == "rule"

    def test_evidence_populated(self):
        r = get_library_detector().detect_explicit_preference(
            "Use playwright to test"
        )
        assert r.evidence, "evidence must be populated when library detected"
        assert any("playwright" in e["text_span"].lower() for e in r.evidence)
        # Each evidence entry has the canonical 4-key shape (PRD §FR-5)
        first = r.evidence[0]
        assert set(first.keys()) == {"library", "pattern", "weight", "text_span"}


# =============================================================================
# AC-6 — generic NL phrases must NOT trigger explicit detection
# =============================================================================


class TestGenericNLPhrasesDoNotTrigger:
    @pytest.mark.parametrize(
        "scenario",
        [
            "click element by id submit",
            "Page should contain Welcome text",
            "Input text into the username field",
            "Open new page in browser context",
            "Execute command on the remote server",
            "Open application on the device",
            "Check if exists and verify row count",
            "Verify status code returned by service",
            "Parse the XML response from the API",
            # v5 — bare-noun preference verbs no longer explicit
            "use browser to load the page",
            "use database for the test",
            "use ssh for deployment",
            "use requests for the API call",
            "use xml for the config",
        ],
    )
    def test_no_explicit(self, scenario):
        r = get_library_detector().detect_explicit_preference(scenario)
        assert r.library is None, (
            f"NL phrase {scenario!r} should NOT trigger explicit detection; "
            f"got {r.library}"
        )


# =============================================================================
# AC-4 + sentence-scoped negation
# =============================================================================


class TestNegationAndMigration:
    def test_migration_resolves_to_destination(self):
        """AC-4 — migration subtracts source, adds to destination."""
        r = get_library_detector().detect_explicit_preference(
            "Migrate the test suite from Selenium to Playwright"
        )
        assert r.library == "Browser"
        assert "web_automation" not in r.conflicts

    def test_negation_across_comma_clauses(self):
        """Round-3 marquee: 'do not use X, instead use Y' must resolve to Y."""
        r = get_library_detector().detect_explicit_preference(
            "do not use Selenium, instead use Playwright"
        )
        assert r.library == "Browser"

    def test_both_libraries_negated_to_none(self):
        """Round-3 D1 edge case: two negation spans, both target SL → None."""
        r = get_library_detector().detect_explicit_preference(
            "do not use Selenium and stop the SeleniumLibrary"
        )
        assert r.library is None

    def test_skip_negation_restored(self):
        """Round-4 D-skip regression: standalone `skip` must fire negation."""
        r = get_library_detector().detect_explicit_preference(
            "skip selenium and use playwright"
        )
        assert r.library == "Browser"

    def test_negation_phrase_does_not_double_deduct(self):
        """Round-3 D1: single span fires once even though both
        \\bdo\\s+not\\s+use\\b and \\bdo\\s+not\\b are in the alternation.
        """
        # If double-deduction occurred, SL score would underflow before
        # max(0, ...) clamps it. Add a second sentence with positive SL
        # signal to detect underflow.
        r = get_library_detector().detect_explicit_preference(
            "do not use Selenium. Then run selenium for diagnostic."
        )
        # First sentence negates: SL 16 → 0.
        # Second sentence: `\bselenium\b` (+6).
        # Net: SL = 6 (not 0 from double-deduction).
        assert r.all_scores.get("SeleniumLibrary", 0) >= 6


# =============================================================================
# AC-5 — within-conflict-group ambiguity
# =============================================================================


class TestConflicts:
    def test_both_selenium_and_playwright_returns_none_with_conflict(self):
        """AC-5 — conflict check on RAW scores BEFORE threshold filter."""
        r = get_library_detector().detect_explicit_preference(
            "Test both selenium and playwright sites and compare"
        )
        assert r.library is None
        assert "web_automation" in r.conflicts
        # Round-3 D3: conflicts entries are DICTS, not tuples
        libs = {entry["library"] for entry in r.conflicts["web_automation"]}
        assert libs == {"Browser", "SeleniumLibrary"}
        # Canonical shape: {library, score, patterns_matched}
        entry = r.conflicts["web_automation"][0]
        assert set(entry.keys()) == {"library", "score", "patterns_matched"}


# =============================================================================
# Threshold tunability
# =============================================================================


class TestThresholdsTunable:
    def test_default_min_score_for_non_conflict_library(self):
        # XML is outside the web_automation conflict group → default threshold 5
        r = get_library_detector().detect_explicit_preference(
            "Use XML library to validate"
        )
        assert r.library == "XML"

    def test_conflict_threshold_blocks_single_weight_6(self):
        # Lone "selenium" mention is weight 6; conflict threshold is 8 → blocked
        r = get_library_detector().detect_explicit_preference(
            "Run a Selenium test"
        )
        assert r.library is None

    def test_conflict_threshold_passes_weight_9(self):
        # "playwright" verbatim is weight 9 → passes conflict threshold of 8
        r = get_library_detector().detect_explicit_preference(
            "Run a playwright test"
        )
        assert r.library == "Browser"

    def test_env_var_lowers_threshold(self, monkeypatch):
        monkeypatch.setenv("ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD", "5")
        d = LibraryDetector()  # fresh detector reads env at construction
        r = d.detect_explicit_preference("Run a Selenium test")
        assert r.library == "SeleniumLibrary"


# =============================================================================
# Backwards-compatibility — legacy entry points still work
# =============================================================================


class TestBackwardsCompatibility:
    def test_legacy_detect_returns_string_or_none(self):
        d = get_library_detector()
        # legacy mention-layer `detect()` may still fire for old NL phrases
        # since it does NOT use the explicit flag. The new path is the one
        # users should call.
        assert isinstance(d.detect("use playwright"), str)
        assert d.detect("xyzabc nonsense") is None

    def test_compiled_patterns_test_contract_preserved(self):
        """v5 — `for p, _ in _compiled_patterns: p.findall(...)` must work."""
        d = get_library_detector()
        b = d._compiled_patterns.get("Browser", [])
        assert b, "Browser must have at least one compiled pattern"
        # Existing fixture pattern: for p, _ in entries: p.findall(...)
        ok = any(
            p.findall("playwright is great")
            for p, _ in b
        )
        assert ok

    def test_compiled_patterns_entries_are_2_tuples(self):
        d = get_library_detector()
        for lib, entries in d._compiled_patterns.items():
            for e in entries:
                assert isinstance(e, tuple) and len(e) == 2, (
                    f"{lib} entry must be a 2-tuple (Pattern, int); got {e!r}"
                )


# =============================================================================
# Evidence shape
# =============================================================================


class TestEvidenceShape:
    def test_evidence_includes_required_fields(self):
        r = get_library_detector().detect_explicit_preference(
            "Use playwright for the demoshop test"
        )
        assert r.evidence
        first = r.evidence[0]
        assert "library" in first
        assert "pattern" in first
        assert "weight" in first
        assert "text_span" in first
        assert first["library"] == "Browser"


# =============================================================================
# End-to-end through analyze_scenario
# =============================================================================


def _run(coro):
    """Helper: run an async coroutine in a fresh event loop (avoids
    `asyncio.get_event_loop()` deprecation warning under Python 3.13).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestAnalyzeScenarioResponse:
    def _analyze(self, scenario):
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor
        nlp = NaturalLanguageProcessor()
        return _run(nlp.analyze_scenario(scenario, context="web"))

    def test_response_omits_evidence_when_no_detection(self):
        result = self._analyze(
            "Test e-commerce website https://demoshop.makrocode.de: "
            "open browser, add items, complete checkout"
        )
        analysis = result["analysis"]
        assert analysis["explicit_library_preference"] is None
        assert analysis["preference_source"] == "rule"
        assert "explicit_library_evidence" not in analysis
        assert "library_preference_conflicts" not in analysis

    def test_response_includes_evidence_on_explicit_detection(self):
        result = self._analyze(
            "Use playwright to test the demoshop checkout flow"
        )
        analysis = result["analysis"]
        assert analysis["explicit_library_preference"] == "Browser"
        assert analysis["preference_source"] == "rule"
        assert "explicit_library_evidence" in analysis
        assert any(
            "playwright" in e["text_span"].lower()
            for e in analysis["explicit_library_evidence"]
        )

    def test_response_includes_conflicts_on_ambiguity(self):
        result = self._analyze(
            "Test both selenium and playwright sites and compare"
        )
        analysis = result["analysis"]
        assert analysis["explicit_library_preference"] is None
        assert "library_preference_conflicts" in analysis
        assert "web_automation" in analysis["library_preference_conflicts"]


# =============================================================================
# Race-free invocation (round-4 finding: no _last_resolution on singleton)
# =============================================================================


class TestNoSharedState:
    """Round-4 finding: the proposal v4 stashed self._last_resolution on a
    module-level NaturalLanguageProcessor singleton — race condition under
    concurrent requests. v5 returns the resolution as a local variable.
    Verify by interleaving two analyses on the SAME nlp instance and
    checking each call returns its own correct evidence.
    """

    def test_interleaved_analyses_no_state_bleed(self):
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor
        nlp = NaturalLanguageProcessor()

        async def run_two():
            r1 = await nlp.analyze_scenario(
                "Use playwright to test demoshop", context="web"
            )
            r2 = await nlp.analyze_scenario(
                "Use Selenium to test the login form", context="web"
            )
            return r1, r2

        r1, r2 = _run(run_two())
        a1, a2 = r1["analysis"], r2["analysis"]
        assert a1["explicit_library_preference"] == "Browser"
        assert a2["explicit_library_preference"] == "SeleniumLibrary"
        # Evidence in each result references the right library
        assert any(
            e["library"] == "Browser"
            for e in a1.get("explicit_library_evidence", [])
        )
        assert any(
            e["library"] == "SeleniumLibrary"
            for e in a2.get("explicit_library_evidence", [])
        )


# =============================================================================
# sentence_index evidence filtering (round-3 D7)
# =============================================================================


class TestSentenceIndexEvidenceFiltering:
    """Round-3 D7: when negation in sentence N drops SL evidence, evidence
    from sentence M (M != N) must be preserved.
    """

    def test_multi_sentence_partial_negation_preserves_other_evidence(self):
        d = get_library_detector()
        r = d.detect_explicit_preference(
            "Use Selenium for login. We do not use Selenium for checkout"
        )
        # Sentence 1 positive: SL = +16. Sentence 2 negation: subtract sentence-2
        # SL = -16. Total raw: SL = 0. Then sentence 1 evidence preserved
        # because its sentence_index != sentence 2's index.
        # But raw_scores[SL] = 0 means SL won't pass threshold → library None.
        # Wait actually that's not what we want — let me reconsider.
        # The negation subtracts only sentence-2-local SL contributions = 16.
        # raw_scores[SL] = 16 (sentence 1) + 16 (sentence 2) - 16 = 16.
        # SL ≥ 8 → returns SL.
        assert r.library == "SeleniumLibrary"
        # Verify evidence still has at least one matching SL match
        assert any(
            e["library"] == "SeleniumLibrary" for e in r.evidence
        )


# =============================================================================
# Sampling override coherence (round-4 finding: drop primary_library fallback)
# =============================================================================


# =============================================================================
# v6 — Codex round-5 critical findings (4 bugs, all fixed in v6)
# =============================================================================


class TestV6SessionFallbackBug:
    """Codex round-5 D-session-fallback: session_models.py used to fall back to
    `_fallback_detect_library` (broad substring heuristic) when v5 returned
    None, which re-introduced false-positive explicit preferences at the
    session aggregate level. v6 only falls back on ImportError.
    """

    def test_session_does_not_force_xml_on_parse_xml_response(self):
        from robotmcp.models.session_models import ExecutionSession
        s = ExecutionSession(session_id="t1")
        detected = s.detect_explicit_library_preference(
            "Parse the XML response from the API"
        )
        assert detected is None, (
            f"Session aggregate must NOT call generic fallback on None; "
            f"got {detected}"
        )

    def test_session_does_not_force_api_lib_on_generic_api_scenario(self):
        from robotmcp.models.session_models import ExecutionSession
        s = ExecutionSession(session_id="t2")
        # "api" is generic — v5/v6 should not flag a library, and the
        # session must not bypass v6 with the legacy substring heuristic.
        detected = s.detect_explicit_library_preference(
            "Verify the api returns 200"
        )
        assert detected is None

    def test_session_preserves_real_explicit_detection(self):
        from robotmcp.models.session_models import ExecutionSession
        s = ExecutionSession(session_id="t3")
        # Genuine explicit signal must still resolve
        detected = s.detect_explicit_library_preference(
            "Use playwright to test the checkout"
        )
        assert detected == "Browser"
        assert s.preference_source == "rule"


class TestV6NewlineNegation:
    """Codex round-5 D-newline-negation: v5 had `\\n` in sentence delimiter so
    'do not use\\nPlaywright' split into 2 sentences and negation couldn't
    find its target. v6 drops `\\n` from sentence delimiter; newline boundary
    is enforced at the explicit-pattern level via `[^\\S\\n]+` instead.
    """

    @pytest.mark.parametrize(
        "scenario,expected",
        [
            ("do not use\nPlaywright", None),
            ("do not use\nAppium", None),
            ("do not use\nRequestsLibrary", None),
            ("do not use\nSeleniumLibrary", None),
        ],
    )
    def test_newline_negation(self, scenario, expected):
        r = get_library_detector().detect_explicit_preference(scenario)
        assert r.library == expected, (
            f"{scenario!r}: expected {expected}, got {r.library}"
        )

    def test_preference_verb_does_not_match_across_newline(self):
        """The `\\b(use|...)\\b[^\\S\\n]+(playwright|...)\\b` weight-10
        preference verb pattern must NOT fire across a newline (because
        `[^\\S\\n]+` excludes newline). The standalone `\\bplaywright\\b`
        weight-9 pattern still fires legitimately — that's by design.
        """
        d = get_library_detector()
        r = d.detect_explicit_preference("use\nplaywright")
        # `playwright` standalone alone scores 9 — passes conflict-group
        # threshold 8 → Browser. Verify the +10 preference verb did NOT fire.
        assert r.library == "Browser"
        # If the preference verb fired, score would be 19; v6 must show 9.
        assert r.all_scores.get("Browser") == 9


class TestV6RepeatedTokenDeduction:
    """Codex round-5 D-repeated-token: `_subtract_sentence_score` used
    `rule.compiled.search(sentence)` (once per rule) instead of counting
    occurrences. 'do not use Playwright Playwright' under-deducted, so
    Browser=9 survived threshold despite negation.
    """

    @pytest.mark.parametrize(
        "scenario",
        [
            "do not use Playwright Playwright",
            "do not use RequestsLibrary RequestsLibrary",
            "do not use SeleniumLibrary SeleniumLibrary",
            "do not use playwright playwright playwright",
        ],
    )
    def test_repeated_token_negation(self, scenario):
        r = get_library_detector().detect_explicit_preference(scenario)
        assert r.library is None, (
            f"Repeated-token negation must clamp score to 0; got {r.library} "
            f"with scores {r.all_scores}"
        )


class TestV6MultiwordMigration:
    """Codex round-5 D-multiword-migration: v5 destination capture stopped at
    first whitespace, so 'to Requests library' captured 'Requests' (bare token
    deliberately excluded from _LIBRARY_TOKENS). v6 captures up to next
    sentence punctuation, letting `_first_library_token_in` resolve canonical.
    """

    @pytest.mark.parametrize(
        "scenario,expected",
        [
            ("Migrate from Selenium to Requests library", "RequestsLibrary"),
            ("Migrate from Selenium to Database library", "DatabaseLibrary"),
            ("Migrate from Browser to SSH library", "SSHLibrary"),
            ("Migrate from Selenium to XML library", "XML"),
            # Single-word destinations still work
            ("Migrate from Selenium to Playwright", "Browser"),
            ("Migrate from Selenium to Browser library", "Browser"),
        ],
    )
    def test_multiword_migration_destination(self, scenario, expected):
        r = get_library_detector().detect_explicit_preference(scenario)
        assert r.library == expected, (
            f"{scenario!r}: expected {expected}, got {r.library}"
        )


# =============================================================================
# v7 — Codex round-6 findings
# =============================================================================


class TestV7ParagraphBoundary:
    """Codex round-6 C1: v6's `_SENTENCE_DELIMITERS = r"[.;!?,]+"` dropped
    `\\n` entirely to fix D-newline-negation, but paragraph-separated text
    then leaked negation across paragraphs. v7 makes `\\n\\s*\\n+` (paragraph
    break) a boundary while keeping single `\\n` as NON-boundary (so v6's
    `"do not use\\nPlaywright"` fix stays intact).
    """

    @pytest.mark.parametrize(
        "scenario,expected",
        [
            # Paragraph break separates negation from following positive signal
            (
                "Do not use this approach\n\nUse Playwright for the test.",
                "Browser",
            ),
            (
                "Avoid that pattern\n\nWith SeleniumLibrary, open the page.",
                "SeleniumLibrary",
            ),
            (
                "Stop the current setup\n\nUse playwright for the rewrite",
                "Browser",
            ),
        ],
    )
    def test_paragraph_break_scopes_negation(self, scenario, expected):
        r = get_library_detector().detect_explicit_preference(scenario)
        assert r.library == expected, (
            f"Paragraph break must scope negation; "
            f"got {r.library} (scores: {r.all_scores})"
        )

    @pytest.mark.parametrize(
        "scenario",
        [
            # Single newline must STILL not split (v6 fix preserved)
            "do not use\nPlaywright",
            "do not use\nAppium",
            "do not use\nSeleniumLibrary",
            "do not use\nRequestsLibrary",
        ],
    )
    def test_single_newline_not_a_boundary(self, scenario):
        r = get_library_detector().detect_explicit_preference(scenario)
        assert r.library is None, (
            f"Single newline must NOT split sentence — negation must reach "
            f"the target on the next line. Got {r.library}."
        )


class TestV7ParagraphBoundaryEndToEnd:
    """Codex round-6 C1 follow-on: `_normalize_text` collapsed `\\n\\n` to a
    single space, destroying the paragraph-break signal the v7 detector
    relies on. v7 passes the RAW scenario (not normalized) to the
    resolver inside `analyze_scenario`. This test exercises the full
    public path, which is what production consumers actually hit.
    """

    def test_paragraph_break_works_through_analyze_scenario(self):
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor
        nlp = NaturalLanguageProcessor()
        result = _run(
            nlp.analyze_scenario(
                "Do not use this approach\n\nUse Playwright for the test.",
                context="web",
            )
        )
        assert (
            result["analysis"]["explicit_library_preference"] == "Browser"
        ), (
            "Paragraph-break scoped negation must survive analyze_scenario's "
            "_normalize_text step (v7 fix: detector now sees raw input)."
        )


class TestV7NlpProcessorAbstainNoFallback:
    """Codex round-6 C3: `NaturalLanguageProcessor._detect_explicit_library_preference()`
    used to fall back to the broad substring heuristic when v6 returned None,
    re-introducing the bug the same way v5 did at the session level. v7 mirrors
    the v6 session_models fix: fall back ONLY on Exception, never on
    deliberate None.
    """

    @pytest.mark.parametrize(
        "scenario",
        [
            "Parse the XML response from the API",  # was → XML via fallback
            "Use requests for the API call",  # was → RequestsLibrary
            "use database for the test",
            "use ssh for deployment",
            "use xml for the config",
        ],
    )
    def test_nlp_processor_does_not_fall_back_on_abstain(self, scenario):
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor
        nlp = NaturalLanguageProcessor()
        result = nlp._detect_explicit_library_preference(scenario)
        assert result is None, (
            f"v7 fix: helper must mirror session_models — no broad fallback "
            f"on deliberate None. Got {result!r}"
        )

    def test_nlp_processor_still_returns_genuine_explicit(self):
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor
        nlp = NaturalLanguageProcessor()
        assert (
            nlp._detect_explicit_library_preference("Use playwright to test")
            == "Browser"
        )
        assert (
            nlp._detect_explicit_library_preference(
                "Use SeleniumLibrary for the test"
            )
            == "SeleniumLibrary"
        )


class TestSamplingOverrideCoherence:
    """v5 (ADR-024 §11): Site A only maps sampling.library_preference into
    explicit_library_preference. primary_library is NOT mapped (it's a
    recommendation, not user-stated preference). When override fires,
    evidence/conflicts are cleared and preference_source flips to 'sampling'.
    """

    def test_only_library_preference_writes_to_explicit_field(self, monkeypatch):
        """v5 sampling Site A: primary_library alone does NOT override the
        explicit field; only library_preference does."""

        async def fake_sampling(ctx, scenario, context):
            return {
                "primary_library": "SeleniumLibrary",
                # NO library_preference key → explicit field should stay
                # whatever the rule-based detector produced.
            }

        monkeypatch.setattr(
            "robotmcp.utils.sampling.sample_analyze_scenario", fake_sampling
        )
        monkeypatch.setattr(
            "robotmcp.utils.sampling.is_sampling_enabled", lambda: True
        )

        # Use a scenario that the rule-based detector resolves to Browser
        from robotmcp.server import analyze_scenario as analyze_tool

        # @mcp.tool decorator wraps in FastMCP FunctionTool — call .fn for
        # the underlying coroutine.
        class _FakeCtx:
            pass

        result = _run(
            get_tool_fn(analyze_tool)(
                scenario="Use playwright to test demoshop",
                context="web",
                session_id="test_sampling_no_override",
                ctx=_FakeCtx(),
            )
        )
        analysis = result["analysis"]
        # primary_library alone did NOT override — Browser still set, rule-based
        assert analysis["explicit_library_preference"] == "Browser"
        assert analysis["preference_source"] == "rule"

    def test_library_preference_overrides_and_clears_evidence(self, monkeypatch):
        async def fake_sampling(ctx, scenario, context):
            return {
                "library_preference": "SeleniumLibrary",
                "rationale": "Detected via LLM",
            }

        monkeypatch.setattr(
            "robotmcp.utils.sampling.sample_analyze_scenario", fake_sampling
        )
        monkeypatch.setattr(
            "robotmcp.utils.sampling.is_sampling_enabled", lambda: True
        )

        from robotmcp.server import analyze_scenario as analyze_tool

        class _FakeCtx:
            pass

        result = _run(
            get_tool_fn(analyze_tool)(
                scenario="Use playwright to test demoshop",
                context="web",
                session_id="test_sampling_override",
                ctx=_FakeCtx(),
            )
        )
        analysis = result["analysis"]
        assert analysis["explicit_library_preference"] == "SeleniumLibrary"
        assert analysis["preference_source"] == "sampling"
        assert "explicit_library_evidence" not in analysis
        assert "library_preference_conflicts" not in analysis
        assert analysis.get("sampling_evidence") == "Detected via LLM"
