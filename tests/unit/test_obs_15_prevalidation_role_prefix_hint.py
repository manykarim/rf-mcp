"""OBS-15 — OBS-12 hint must also reach the pre-validation failure path.

The 2026-05-17 v3 validation re-run of the Tricentis Obstacle Course (after
OBS-10/11/12 shipped) found that the OBS-12 ``browser_role_prefix_misuse``
hint never actually fires in production. Sonnet attempted
``intent_action(intent="click", target="button=Calculate")`` on Obstacle 8
and the failure response carried only generic visibility / enabled / timeout
hints — no role-prefix locator hint.

Root cause: ``_check_browser_role_prefix_misuse`` is invoked via
``generate_hints`` in the *keyword-execution failure path* at
``keyword_executor.py:1684+``. But ``button=X`` against Browser library
typically fails earlier — at *pre-validation* (the 500ms gate that checks
element actionability) — which builds its own hint list **inline** at
``keyword_executor.py:1393+`` and does NOT route through ``generate_hints``.

The OBS-12 unit tests in ``test_browser_role_prefix_hint.py`` pin the hint
firing through ``generate_hints``. They pass; the production code path
just never calls into that pipeline for the typical failure mode. This
file pins the pre-validation path integration as well.

These tests pin:
(1) The inline pre-validation hint builder emits a
    ``browser_role_prefix_misuse`` entry when the locator matches the
    role-prefix pattern AND the session uses Browser library.
(2) The new hint preserves the title / message / examples produced by
    the existing ``_check_browser_role_prefix_misuse`` checker.
(3) The hint does NOT fire for SeleniumLibrary sessions (where the
    locator syntax is valid).
(4) The hint does NOT fire when the locator does NOT match the
    role-prefix pattern (id=, css=, xpath=, text=, name=).
(5) Hints are deduplicated by ``type`` so the same hint never appears
    twice if both paths could reach it.

The integration is tested via the real ``_execute_keyword_serialized``
method (not just source-text matching) so any regression that breaks the
wiring is caught.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig


def _failing_pre_validate(
    msg: str = "Element missing required states: enabled, visible",
    missing: List[str] | None = None,
) -> AsyncMock:
    """Build a stub for ``_pre_validate_element_with_retry`` that reports
    the failure shape Browser library actually produces."""
    details = {
        "locator": "<unused>",
        "current_states": [],
        "missing_states": missing if missing is not None else ["enabled", "visible"],
    }
    return AsyncMock(return_value=(False, msg, details))


def _session(*, search_order: List[str] | None = None) -> MagicMock:
    """Build a minimal session stub. Pre-validation reads session_id +
    search_order. We don't go past pre-validation so library managers,
    variables, etc. don't matter."""
    sess = MagicMock()
    sess.session_id = "obs15-test-session"
    sess.variables = {}
    sess.search_order = list(search_order or ["Browser"])
    sess.update_activity = MagicMock()
    return sess


@pytest.fixture
def executor():
    return KeywordExecutor(config=ExecutionConfig())


# ---------------------------------------------------------------------------
# Positive cases — the hint fires from the pre-validation failure path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOBS15HintFiresFromPreValidationPath:
    """When pre-validation rejects a role-prefix locator against Browser
    library, the failure response's ``hints`` list MUST include a
    ``browser_role_prefix_misuse`` entry."""

    # NB: ``link=X`` is NOT in this parametrize. The pre-validation
    # block has a sister table ``_SKIP_PRE_VALIDATION_LOCATOR_PREFIXES``
    # — designed for SeleniumLibrary's link-text strategy — that skips
    # pre-validation entirely whenever a locator starts with ``link=``.
    # For Browser library + ``link=X``, the keyword-execution-failure
    # path still emits the OBS-12 hint through ``generate_hints`` (the
    # existing OBS-12 unit tests pin that). OBS-15 only widens
    # coverage for the four prefixes that *do* enter pre-validation:
    # button, input, select, textarea.
    @pytest.mark.parametrize("locator,role,value", [
        ("button=Calculate", "button", "Calculate"),
        ("input=username", "input", "username"),
        ("select=Country", "select", "Country"),
        ("textarea=Comment", "textarea", "Comment"),
    ])
    async def test_hint_fires_for_each_role_prefix(
        self, executor, locator, role, value,
    ):
        sess = _session(search_order=["Browser"])
        with patch.object(
            executor,
            "_pre_validate_element_with_retry",
            _failing_pre_validate(),
        ):
            result = await executor._execute_keyword_serialized(
                session=sess,
                keyword="Click",
                arguments=[locator],
                browser_library_manager=MagicMock(),
            )

        assert result["success"] is False
        assert result.get("pre_validation_failed") is True

        types = [h.get("type") for h in result.get("hints", [])]
        assert "browser_role_prefix_misuse" in types, (
            f"OBS-15: role-prefix hint missing for {locator!r}; got "
            f"types={types!r}"
        )

        role_hint = next(
            h for h in result["hints"]
            if h.get("type") == "browser_role_prefix_misuse"
        )
        # Title names the rejected role prefix.
        assert role in role_hint["title"].lower()
        # Message names both the literal failing locator and points the
        # user at the SL→Browser correction.
        assert locator in role_hint["message"]
        assert "SeleniumLibrary" in role_hint["message"]
        # Examples include BOTH working alternatives: text= and
        # css=<tag>:text-is(...).
        examples = role_hint["examples"]
        assert any(f"text={value}" in str(e) for e in examples), examples
        assert any(":text-is(" in str(e) for e in examples), examples


@pytest.mark.asyncio
class TestOBS15HintShape:
    """The OBS-15 hint preserves the structured fields produced by the
    existing OBS-12 checker — title, message, examples — and adds a
    ``type`` field for parity with the other pre-validation hints."""

    async def test_hint_has_type_title_message_examples(self, executor):
        sess = _session(search_order=["Browser"])
        with patch.object(
            executor,
            "_pre_validate_element_with_retry",
            _failing_pre_validate(),
        ):
            result = await executor._execute_keyword_serialized(
                session=sess,
                keyword="Click",
                arguments=["button=Calculate"],
                browser_library_manager=MagicMock(),
            )

        role_hints = [
            h for h in result["hints"]
            if h.get("type") == "browser_role_prefix_misuse"
        ]
        assert len(role_hints) == 1
        h = role_hints[0]
        assert h["type"] == "browser_role_prefix_misuse"
        assert isinstance(h["title"], str) and h["title"]
        assert isinstance(h["message"], str) and h["message"]
        assert isinstance(h["examples"], list) and h["examples"]

    async def test_pre_validation_failure_hint_still_present(self, executor):
        """The OBS-15 hint is additive — the existing
        ``pre_validation_failure`` / ``visibility_hint`` /
        ``pre_validate_timeout_hint`` entries must still be there.
        Regression guard against an over-eager dedup."""
        sess = _session(search_order=["Browser"])
        with patch.object(
            executor,
            "_pre_validate_element_with_retry",
            _failing_pre_validate(missing=["visible"]),
        ):
            result = await executor._execute_keyword_serialized(
                session=sess,
                keyword="Click",
                arguments=["button=Calculate"],
                browser_library_manager=MagicMock(),
            )
        types = [h.get("type") for h in result["hints"]]
        assert "pre_validation_failure" in types
        assert "visibility_hint" in types
        assert "pre_validate_timeout_hint" in types
        assert "browser_role_prefix_misuse" in types


# ---------------------------------------------------------------------------
# Negative cases — the hint MUST NOT fire when it shouldn't
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOBS15HintDoesNotFireWhenInapplicable:
    """The OBS-15 hint must stay quiet when the locator is valid for the
    active library."""

    @pytest.mark.parametrize("locator", [
        "id=submit",
        "css=button.primary",
        "xpath=//button[text()='Save']",
        "text=Calculate",
        "name=username",
        "#submit",
        "//button",
    ])
    async def test_silent_for_valid_browser_locators(self, executor, locator):
        sess = _session(search_order=["Browser"])
        with patch.object(
            executor,
            "_pre_validate_element_with_retry",
            _failing_pre_validate(),
        ):
            result = await executor._execute_keyword_serialized(
                session=sess,
                keyword="Click",
                arguments=[locator],
                browser_library_manager=MagicMock(),
            )
        types = [h.get("type") for h in result.get("hints", [])]
        assert "browser_role_prefix_misuse" not in types, (
            f"OBS-15: hint must not fire for valid Browser locator "
            f"{locator!r}; got types={types!r}"
        )

    async def test_silent_for_selenium_library_session(self, executor):
        # ``button=X`` is valid SeleniumLibrary syntax — hint must NOT fire.
        sess = _session(search_order=["SeleniumLibrary"])
        with patch.object(
            executor,
            "_pre_validate_element_with_retry",
            _failing_pre_validate(),
        ):
            result = await executor._execute_keyword_serialized(
                session=sess,
                keyword="Click Element",
                arguments=["button=Calculate"],
                browser_library_manager=MagicMock(),
            )
        types = [h.get("type") for h in result.get("hints", [])]
        assert "browser_role_prefix_misuse" not in types, (
            f"OBS-15: hint must not fire for SeleniumLibrary; got "
            f"types={types!r}"
        )

    async def test_silent_when_pre_validation_passes(self, executor):
        """If pre-validation passes, the failure block isn't even
        reached. The role-prefix hint only fires via the failure path
        OR (separately) via the keyword-execution-failure path. Not via
        the success path."""
        sess = _session(search_order=["Browser"])
        # _pre_validate_element_with_retry returns success → block skipped
        passing = AsyncMock(return_value=(True, None, {}))
        # The keyword-execution path is mocked too so we don't actually
        # try to run anything against a real RF context.
        with patch.object(
            executor, "_pre_validate_element_with_retry", passing,
        ), patch.object(
            executor,
            "_execute_keyword_with_context",
            AsyncMock(return_value={"success": True, "result": None}),
        ):
            result = await executor._execute_keyword_serialized(
                session=sess,
                keyword="Click",
                arguments=["button=Calculate"],
                browser_library_manager=MagicMock(),
            )
        # Successful run — no hints from the pre-validation failure
        # block. The keyword-execution path runs through but doesn't
        # fail (we mocked it), so its own hints aren't surfaced either.
        assert result.get("pre_validation_failed") is not True


# ---------------------------------------------------------------------------
# Dedup safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOBS15HintDedup:
    """The pre-validation failure block dedups hints by ``type``. This
    safeguards against future callers that might (incorrectly) append a
    duplicate hint via a different path."""

    async def test_each_type_appears_at_most_once(self, executor):
        sess = _session(search_order=["Browser"])
        with patch.object(
            executor,
            "_pre_validate_element_with_retry",
            _failing_pre_validate(missing=["enabled", "visible"]),
        ):
            result = await executor._execute_keyword_serialized(
                session=sess,
                keyword="Click",
                arguments=["button=Calculate"],
                browser_library_manager=MagicMock(),
            )

        types = [h.get("type") for h in result["hints"] if h.get("type")]
        # No duplicate types.
        assert len(types) == len(set(types)), (
            f"duplicate hint types in OBS-15 response: {types!r}"
        )


# ---------------------------------------------------------------------------
# Source-shape pin — guards against future refactors moving the call
# site away from the pre-validation failure block.
# ---------------------------------------------------------------------------


class TestOBS15SourceShape:
    """Belt-and-braces: the literal call site for the OBS-15 enrichment
    sits inside the pre-validation failure block. If a future refactor
    moves it out, the tests above keep working through different paths
    but this pin documents the intended location and surfaces the
    regression early."""

    def test_executor_imports_role_prefix_checker_inline(self):
        import pathlib
        executor_src = pathlib.Path(
            "src/robotmcp/components/execution/keyword_executor.py"
        ).read_text(encoding="utf-8")
        # Inline import inside the pre-validation failure block.
        assert "_check_browser_role_prefix_misuse" in executor_src
        # Type tag emitted by the inline builder.
        assert '"type": "browser_role_prefix_misuse"' in executor_src
