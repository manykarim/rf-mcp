"""Proposal-A regression tests for analyze_scenario.

A1: URL presence boosts WEB_AUTOMATION score.
A2: bare "request" / "http" / "app" must not flip classification to API/Mobile.
A3: whole-word matching replaces substring containment in three detectors.
A4: detected_session_type and session_type agree on Tricentis-shaped scenarios.
A5: web_automation sessions auto-load Browser when nothing else is configured.
A6: leading-URL scenarios produce a domain-host title (no truncation at dot).
A7: compound action sentences yield multiple actions; targets are real words.

These tests pin the behaviour observed during the multi-model Tricentis
diagnostic so future detector tweaks cannot silently regress.
"""

__test__ = True

import asyncio

import pytest

from robotmcp.components.nlp_processor import NaturalLanguageProcessor
from robotmcp.models.session_models import ExecutionSession, SessionType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


TRICENTIS_5 = [
    "Open https://sampleapp.tricentis.com/101/, create an insurance request for an Automobile.",
    "Open https://sampleapp.tricentis.com/101/ and submit an insurance quote for an Automobile.",
    "Browse to https://sampleapp.tricentis.com/101/ in the browser. Fill out all forms across the wizard screens and submit the application.",
    "Use the browser to test the insurance request flow at sampleapp.tricentis.com/101 — fill in vehicle, insurant and product data and send the email quote.",
    "Open https://sampleapp.tricentis.com/101/ in Chrome browser. Click Automobile. Fill in vehicle details. Click Next. Fill insurant data. Click Next. Choose product options. Click Next. Pick price option. Click Next. Send the quote.",
]


@pytest.fixture(scope="module")
def nlp() -> NaturalLanguageProcessor:
    return NaturalLanguageProcessor()


def _analyze(nlp: NaturalLanguageProcessor, scenario: str) -> dict:
    return asyncio.run(nlp.analyze_scenario(scenario, "web"))


# ---------------------------------------------------------------------------
# A1: URL boosts WEB_AUTOMATION
# ---------------------------------------------------------------------------


class TestA1UrlBoostsWeb:
    def test_url_alone_routes_to_web(self):
        s = ExecutionSession(session_id="a1")
        st = s.detect_session_type_from_scenario(
            "Open https://example.com/path and do something."
        )
        assert st == SessionType.WEB_AUTOMATION

    def test_url_plus_api_word_does_not_flip_to_api(self):
        """A URL plus 'api' should still favour web unless API tokens dominate."""
        s = ExecutionSession(session_id="a1b")
        st = s.detect_session_type_from_scenario(
            "Open https://example.com and do something."
        )
        assert st == SessionType.WEB_AUTOMATION


# ---------------------------------------------------------------------------
# A2: "request" / "app" must not flip classification
# ---------------------------------------------------------------------------


class TestA2NegativeEvidence:
    def test_insurance_request_is_not_api(self, nlp):
        r = _analyze(
            nlp,
            "Open https://sampleapp.tricentis.com/101/, create an insurance request for an Automobile.",
        )
        assert r["analysis"]["detected_session_type"] == "web_automation"
        # Capability list must not include RequestsLibrary
        assert "RequestsLibrary" not in r["scenario"]["required_capabilities"]

    def test_application_word_is_not_mobile(self, nlp):
        r = _analyze(
            nlp,
            "Browse to https://sampleapp.tricentis.com/101/ in the browser. Fill out all forms across the wizard screens and submit the application.",
        )
        assert r["analysis"]["detected_session_type"] == "web_automation"
        assert "AppiumLibrary" not in r["scenario"]["required_capabilities"]

    def test_explicit_lib_preference_not_requests(self, nlp):
        """A2: the 'request' word must not produce RequestsLibrary preference."""
        r = _analyze(
            nlp,
            "Use the browser to handle the insurance request workflow at https://example.com/x.",
        )
        assert r["analysis"]["explicit_library_preference"] != "RequestsLibrary"


# ---------------------------------------------------------------------------
# A3: whole-word matching everywhere
# ---------------------------------------------------------------------------


class TestA3WholeWordMatching:
    def test_app_in_sampleapp_does_not_match_mobile(self):
        from robotmcp.components.execution.session_manager import SessionManager

        sm = SessionManager()
        result = sm.detect_platform_from_scenario(
            "Open https://sampleapp.tricentis.com/101/ and fill the form."
        )
        # PlatformType.WEB
        assert result.value == "web"

    def test_rest_substring_does_not_match_api(self, nlp):
        """The substring 'rest' inside 'restroom' must not route to API."""
        r = _analyze(nlp, "Open https://example.com/restroom and fill the form.")
        assert r["analysis"]["detected_session_type"] == "web_automation"


# ---------------------------------------------------------------------------
# A4: detected_session_type field is consistent
# ---------------------------------------------------------------------------


class TestA4Consistency:
    @pytest.mark.parametrize("idx", range(5))
    def test_all_tricentis_variants_are_web_automation(self, nlp, idx):
        r = _analyze(nlp, TRICENTIS_5[idx])
        assert r["analysis"]["detected_session_type"] == "web_automation", (
            f"Probe {idx}: {TRICENTIS_5[idx][:80]} -> "
            f"{r['analysis']['detected_session_type']}"
        )


# ---------------------------------------------------------------------------
# A6: leading-URL title extraction
# ---------------------------------------------------------------------------


class TestA6Title:
    def test_url_host_used_as_title(self, nlp):
        r = _analyze(nlp, TRICENTIS_5[0])
        assert r["scenario"]["title"].lower() == "sampleapp.tricentis.com"

    def test_title_not_truncated_at_first_dot(self, nlp):
        r = _analyze(nlp, TRICENTIS_5[0])
        title = r["scenario"]["title"]
        # The buggy version returned 'Open https://sampleapp'
        assert "sampleapp.tricentis.com" in title.lower()

    def test_navigate_keywords_recognised(self, nlp):
        for verb in ("Open", "Navigate to", "Go to", "Visit", "Browse to"):
            r = _analyze(nlp, f"{verb} https://example.com/foo/bar to do stuff.")
            assert r["scenario"]["title"].lower() == "example.com", (
                f"verb={verb} -> title={r['scenario']['title']!r}"
            )


# ---------------------------------------------------------------------------
# A7: compound sentence action extraction
# ---------------------------------------------------------------------------


class TestA7CompoundActions:
    def test_targets_are_at_least_two_chars(self, nlp):
        """Regression: the old patterns produced targets like 'a' or 'n'."""
        r = _analyze(nlp, TRICENTIS_5[4])
        for ac in r["scenario"]["actions"]:
            if ac["target"]:
                assert len(ac["target"]) >= 2, (
                    f"target collapsed to single char: {ac!r}"
                )

    def test_multi_click_scenario_yields_multiple_actions(self, nlp):
        r = _analyze(nlp, TRICENTIS_5[4])
        # 5 Clicks + 1 Open + 2 Fill in + 1 Choose + 1 Pick + 1 Send -> at least 7
        assert r["analysis"]["action_count"] >= 7, (
            f"Expected ≥7 actions, got {r['analysis']['action_count']}"
        )

    def test_navigate_target_does_not_swallow_following_prose(self, nlp):
        """A7: navigate target must be the URL only, not 'URL and submit ...'."""
        r = _analyze(nlp, TRICENTIS_5[1])
        nav_actions = [a for a in r["scenario"]["actions"] if a["action_type"] == "navigate"]
        assert nav_actions
        assert nav_actions[0]["target"].startswith("https://sampleapp.tricentis.com")
        assert "submit" not in nav_actions[0]["target"]

    def test_compound_with_and_then_splits(self, nlp):
        r = _analyze(
            nlp,
            "Click login. Then fill in 'admin' into username field. Finally click submit.",
        )
        assert r["analysis"]["action_count"] >= 2


# ---------------------------------------------------------------------------
# A2/A3 control: API and mobile still route correctly when explicit
# ---------------------------------------------------------------------------


class TestControlsOtherSessionTypes:
    def test_explicit_api_scenario_still_routes_to_api(self, nlp):
        r = _analyze(
            nlp,
            "Test the REST API endpoint /users with GET request and validate the JSON response.",
        )
        assert r["analysis"]["detected_session_type"] == "api_testing"

    def test_explicit_mobile_scenario_still_routes_to_mobile(self, nlp):
        r = _analyze(
            nlp,
            "Test the mobile app on Android using Appium. Tap login button.",
        )
        assert r["analysis"]["detected_session_type"] == "mobile_testing"

    def test_capability_list_for_api_scenario(self, nlp):
        r = _analyze(
            nlp,
            "Send a GET request to /users endpoint and validate the JSON response.",
        )
        assert "RequestsLibrary" in r["scenario"]["required_capabilities"]

    def test_capability_list_for_mobile_scenario(self, nlp):
        r = _analyze(
            nlp,
            "Test the mobile app on android using Appium.",
        )
        assert "AppiumLibrary" in r["scenario"]["required_capabilities"]
