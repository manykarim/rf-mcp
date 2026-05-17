"""nlp_processor URL handling: title host extraction + URL-safe sentence split.

Two regressions the previous logic produced:

1. ``_extract_title`` truncated at the first ``.`` in the input, so a
   scenario opening with ``Open https://sampleapp.tricentis.com/101/``
   produced the meaningless title ``"Open https://sampleapp"``.

2. ``_split_sentences`` split on ``[.!?]+``, so the same URL fragmented
   into three pieces and every action after the first sentence was
   dropped from extraction.

Both fixes preserve the existing "Go to <target>" coverage that the
action-patterns regex matches.
"""

from __future__ import annotations

import pytest

from robotmcp.components.nlp_processor import NaturalLanguageProcessor


@pytest.fixture
def nlp():
    return NaturalLanguageProcessor()


class TestTitleLeadingUrl:
    """A leading URL produces a host-based title, not a dot-truncated one."""

    @pytest.mark.parametrize("scenario,expected", [
        ("Open https://sampleapp.tricentis.com/101/", "Sampleapp.tricentis.com"),
        ("open https://example.com/login and sign in",
         "Example.com"),
        ("navigate to https://www.foo.bar/baz", "Www.foo.bar"),
        ("Go to https://app.local:8080/path", "App.local:8080"),
        ("visit http://192.168.1.1/admin", "192.168.1.1"),
    ])
    def test_leading_url_uses_host(self, nlp, scenario, expected):
        assert nlp._extract_title(scenario) == expected

    def test_no_truncation_at_url_dot(self, nlp):
        # Most important regression: the dot inside the host MUST NOT
        # truncate the title.
        title = nlp._extract_title("Open https://sampleapp.tricentis.com/101/")
        assert "sampleapp.tricentis.com" in title.lower()
        assert title.lower() != "open https://sampleapp"


class TestTitleNonUrlScenarios:
    """Existing non-URL title behaviour is preserved."""

    def test_verb_prefix_pattern(self, nlp):
        assert nlp._extract_title("test login flow") == "Login flow"
        assert nlp._extract_title("verify checkout succeeds") == "Checkout succeeds"

    def test_dot_followed_by_space_still_terminates(self, nlp):
        # A real sentence terminator (period + whitespace) still works.
        assert nlp._extract_title(
            "verify login. then check dashboard"
        ) == "Login"

    def test_default_first_words_fallback(self, nlp):
        # Sentence with no recognised opener falls back to first words.
        title = nlp._extract_title("Quickly tap each option then confirm")
        assert title.startswith("Quickly tap each option")


class TestSplitSentencesUrlSafe:
    """Dots inside URLs do not fragment sentences."""

    def test_url_dot_does_not_split(self, nlp):
        sentences = nlp._split_sentences(
            "Open https://sampleapp.tricentis.com/101/ and click Next"
        )
        assert len(sentences) == 1
        assert "sampleapp.tricentis.com" in sentences[0]

    def test_real_sentence_terminator_still_splits(self, nlp):
        # Trailing dot at end of input has no following whitespace so the
        # splitter does not consume it. That's fine — what matters is the
        # internal split, not stripping the final period.
        sentences = nlp._split_sentences("Open the form. Then click submit.")
        assert len(sentences) == 2
        assert sentences[0] == "Open the form"
        assert sentences[1].rstrip(".") == "Then click submit"

    def test_question_mark_still_splits(self, nlp):
        sentences = nlp._split_sentences("Did it load? Click next")
        assert sentences == ["Did it load", "Click next"]


class TestGoToActionPreserved:
    """The action-extraction navigate pattern still matches "go to <target>"
    for both bare-word and URL forms. This is the CI F2 regression the old
    branch introduced by over-restricting the navigate pattern."""

    @pytest.mark.parametrize("sentence,expected_target_contains", [
        ("go to login", "login"),
        ("Go to dashboard", "dashboard"),
        ("navigate to https://example.com/login", "example.com"),
        ("open the home page", "home"),
        ("visit the about us page", "about us"),
    ])
    def test_navigate_action_extracted(self, nlp, sentence, expected_target_contains):
        action = nlp._extract_action(sentence)
        assert action is not None, f"navigate not extracted from {sentence!r}"
        assert action.action_type == "navigate"
        assert expected_target_contains in (action.target or "").lower()
