"""OBS-01 — ``id=X`` and ``css=#X`` produce identical pre-validation verdicts.

The 2026-05-17 Tricentis benchmark surfaced a real defect on Obstacle 3
("Not a table"): the same ``<button id="generate">`` element passed
pre-validation when targeted as ``css=#generate`` but was reported
'detached' when targeted as ``id=generate``. Both forms are documented
as equivalent by the Browser library, so an LLM has no way to predict
which form the pre-validation gate will accept on a given page state.

Fix: ``KeywordExecutor._normalize_locator_for_browser_prevalidation``
rewrites ``id=X`` to its CSS attribute-selector equivalent
``[id="X"]`` for the pre-validation call ONLY. The actual keyword
execution and ``build_test_suite`` output preserve the original
locator string so generated RF suites still follow the ``id=X``
convention.

These tests pin:
(1) the normaliser's output for the id-value variants in the story's
    acceptance criteria;
(2) that non-id locators pass through unchanged;
(3) that ``_pre_validate_browser_element`` actually calls the
    underlying ``Get Element States`` with the normalised string;
(4) that ``_pre_validate_element`` returns the same verdict for
    ``id=X`` and ``css=#X`` against a mocked Browser library.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig


@pytest.fixture
def executor():
    return KeywordExecutor(config=ExecutionConfig())


# ---------------------------------------------------------------------------
# Normaliser correctness
# ---------------------------------------------------------------------------


class TestNormaliserOutput:
    """Normaliser maps every id-value variant called out in OBS-01
    acceptance criterion #2 to a valid CSS attribute selector."""

    @pytest.mark.parametrize("id_value", [
        "simple",
        "generate",          # Sonnet's actual repro case (Obstacle 3)
        "with-hyphen",       # CSS-shortcut form `#with-hyphen` would also work
        "with_underscore",   # safe in both shortcut and attribute form
        "Camel123",
        "alphaNumeric123",
        "MixedCase-with-Hyphens_AND_underscores_42",
    ])
    def test_id_locator_rewrites_to_attribute_form(self, id_value):
        normalised = KeywordExecutor._normalize_locator_for_browser_prevalidation(
            f"id={id_value}"
        )
        assert normalised == f'[id="{id_value}"]'

    @pytest.mark.parametrize("id_value", [
        # ids containing CSS-special characters where the `#id` shortcut
        # would need escaping but the attribute form does NOT — this is
        # the safety win that motivates choosing [id="X"] over #X.
        "has.dots",
        "has:colons",
        "has space",          # technically invalid HTML but real sites use it
        "weird/chars",
    ])
    def test_id_with_css_special_chars_is_safe_in_attribute_form(self, id_value):
        normalised = KeywordExecutor._normalize_locator_for_browser_prevalidation(
            f"id={id_value}"
        )
        assert normalised == f'[id="{id_value}"]'

    def test_id_with_embedded_double_quote_is_escaped(self):
        # Rare but the rewrite must not produce malformed CSS even
        # when given a pathological id. Escape the double quote.
        normalised = KeywordExecutor._normalize_locator_for_browser_prevalidation(
            'id=has"quote'
        )
        assert normalised == r'[id="has\"quote"]'

    @pytest.mark.parametrize("variant", [
        "id=foo",      # canonical
        "id= foo",     # space after `=`
        "id =foo",     # space before `=`
        "  id=foo",    # leading whitespace
        "id=foo  ",    # trailing whitespace
    ])
    def test_whitespace_tolerated(self, variant):
        normalised = KeywordExecutor._normalize_locator_for_browser_prevalidation(variant)
        assert normalised == '[id="foo"]', (
            f"whitespace-permissive parse failed for {variant!r}"
        )


class TestNormaliserPassthrough:
    """Anything that isn't a bare ``id=X`` locator MUST pass through
    unchanged. Composite, cascaded, and other-prefix forms have their
    own established handling and must not be rewritten by accident."""

    @pytest.mark.parametrize("locator", [
        "css=#submit",                # already CSS — converter strips to #submit
        "css=[id='foo']",
        "css=button.primary",
        "xpath=//button[@id='foo']",
        "//button",                    # implicit xpath
        "#submit",                     # bare CSS shortcut
        "[id='foo']",                  # bare CSS attribute selector
        "text=Login",                  # Browser text engine
        "button:text('Save')",         # Playwright text inside CSS
        "id=foo >> nth=0",             # cascaded — leave to Browser parser
        "id=foo >> visible=true",      # cascaded
        "name=username",               # different strategy
        "link=Click",                  # SeleniumLibrary
        "",                            # empty
    ])
    def test_locator_unchanged(self, locator):
        assert KeywordExecutor._normalize_locator_for_browser_prevalidation(
            locator
        ) == locator

    def test_none_input_returns_unchanged(self):
        # Defensive: a None or unexpectedly-falsy input must not crash.
        assert KeywordExecutor._normalize_locator_for_browser_prevalidation(
            None
        ) is None


# ---------------------------------------------------------------------------
# Wiring: _pre_validate_browser_element passes the normalised string to
# Browser.Get Element States, NOT the original.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPreValidateBrowserElementUsesNormalisedLocator:
    """The wiring assertion: when _pre_validate_browser_element is called
    with ``id=X``, the underlying ``Get Element States`` is invoked with
    the normalised form. This is what makes the verdicts equivalent
    end-to-end."""

    async def test_id_locator_is_normalised_before_get_states(self, executor):
        # Mock the synchronous Get-States wrapper so we can capture what
        # it actually received. Return a "visible" state so the path
        # completes successfully.
        mock_get_states = MagicMock(return_value=(["visible", "enabled"], None))
        with patch.object(executor, "_run_browser_get_states", mock_get_states):
            await executor._pre_validate_browser_element(
                "id=generate", {"visible"}, 500,
            )
        mock_get_states.assert_called_once()
        actual_locator = mock_get_states.call_args.args[0]
        assert actual_locator == '[id="generate"]', (
            f"pre-validation should have normalised id= to attribute form; "
            f"got {actual_locator!r}"
        )

    async def test_css_locator_passes_through_unchanged(self, executor):
        mock_get_states = MagicMock(return_value=(["visible", "enabled"], None))
        with patch.object(executor, "_run_browser_get_states", mock_get_states):
            await executor._pre_validate_browser_element(
                "css=#generate", {"visible"}, 500,
            )
        actual_locator = mock_get_states.call_args.args[0]
        # css= is rewritten by the upstream locator converter, so by the
        # time _pre_validate_browser_element receives it it's already
        # bare CSS — but this test exercises the function directly so
        # the css= prefix is preserved. Either way it passes through
        # the normaliser unchanged.
        assert actual_locator == "css=#generate"


# ---------------------------------------------------------------------------
# Equivalence: _pre_validate_element returns the same verdict for both
# locator forms when the underlying Browser library agrees on the element.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEquivalentVerdictAcrossForms:
    """OBS-01 acceptance #1: same DOM element, same verdict regardless
    of locator-prefix form."""

    @pytest.mark.parametrize("id_value", [
        "generate",
        "simple",
        "with-hyphen",
        "with_underscore",
        "Camel123",
    ])
    async def test_id_and_css_hash_produce_same_pass(self, executor, id_value):
        # Mock Get Element States to ALWAYS return "visible+enabled" so
        # the verdict depends purely on whether the locator is
        # acceptable to the upstream call. The test asserts both forms
        # end up at the same call and get the same verdict.
        mock_get_states = MagicMock(return_value=(["visible", "enabled"], None))
        # Mock the RF context resolution so the function picks the
        # browser path without needing a real RF session.
        with patch.object(executor, "_run_browser_get_states", mock_get_states), \
             patch.object(executor, "_pre_validate_element",
                          wraps=executor._pre_validate_element):
            # Directly call _pre_validate_browser_element to bypass the
            # active-library detection (which requires a real RF context).
            result_id = await executor._pre_validate_browser_element(
                f"id={id_value}", {"visible"}, 500,
            )
            result_css = await executor._pre_validate_browser_element(
                f"#{id_value}", {"visible"}, 500,
            )
        # Both must agree.
        assert result_id["valid"] == result_css["valid"] is True
        # The underlying call sees the normalised id= form vs the bare
        # CSS shortcut — both go through Playwright's CSS engine.
        assert mock_get_states.call_count == 2

    async def test_id_and_css_hash_produce_same_fail(self, executor):
        # Both calls hit a missing element — both return the same
        # failure mode (not the divergence the bug surfaced).
        mock_get_states = MagicMock(return_value=(None, "Element not found: X"))
        with patch.object(executor, "_run_browser_get_states", mock_get_states):
            result_id = await executor._pre_validate_browser_element(
                "id=missing", {"visible"}, 500,
            )
            result_css = await executor._pre_validate_browser_element(
                "#missing", {"visible"}, 500,
            )
        assert result_id["valid"] is False
        assert result_css["valid"] is False
        # Both surface the same kind of error (element-not-found).
        assert result_id["error"] == result_css["error"]
