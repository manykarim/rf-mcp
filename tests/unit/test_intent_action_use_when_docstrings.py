"""OBS-07 — ``force=True`` / ``commit=True`` docstrings reframed from
"escape hatch" to "Use when:" so LLMs see the trigger condition FIRST
and reach for the flag when it's actually the right tool.

The 2026-05-17 Tricentis benchmark surfaced the problem: Obstacle 8
("Wait a moment") had a textbook ``force=True`` scenario but Haiku
abandoned the obstacle on the first pre-validation rejection. Sonnet
recovered cleanly via a different (correct) path, but had ``force=True``
been more discoverable Haiku could likely have solved it.

Neither knob was reached for in either run. Reframing the docstrings
to lead with the trigger condition + a concrete real-world example
(overlay-blocked button / Vue form validator) instead of the
discouraging "ESCAPE HATCH" / "opt-in" framing closes the gap without
changing any code behaviour.

These tests pin:
(1) Both docstrings start with ``Use when:`` (or contain it before any
    cautionary language).
(2) Both docstrings contain a concrete example phrase the LLM can
    pattern-match against a real symptom.
(3) Both flags still document their caveats / scope (Browser-only for
    force; FILL-only for commit). The reframing did not strip the
    safety information — it just moved it AFTER the trigger condition.
(4) The cookbook in the default instructions template cross-references
    commit=True alongside the existing force=True reference.
"""

from __future__ import annotations

import inspect

import pytest

from robotmcp.domains.instruction.value_objects import InstructionTemplate


def _intent_action_doc() -> str:
    """Return the rendered docstring of the ``intent_action`` MCP tool."""
    from robotmcp.server import intent_action
    fn = getattr(intent_action, "fn", intent_action)
    return inspect.getdoc(fn) or ""


# ---------------------------------------------------------------------------
# force=True
# ---------------------------------------------------------------------------


class TestForceDocstringUseWhen:
    """``force=True`` docstring must lead with a concrete trigger."""

    def test_use_when_phrase_present(self):
        doc = _intent_action_doc()
        # The literal "Use when:" header is what makes the trigger
        # condition scan-able in the docstring.
        assert "force: Use when:" in doc, (
            "force= docstring must lead with 'Use when:' so the LLM sees "
            "the trigger condition before any caveats. Found docstring "
            "extract: " + doc[doc.find("force:"): doc.find("force:") + 200]
        )

    @pytest.mark.parametrize("trigger_keyword", [
        "blocked by another element",   # the literal Playwright phrase
        "overlay",                       # the most common trigger
        "sticky",                        # the second most common
        # Concrete real-world example (acceptance criterion 1).
        "consent banner",
    ])
    def test_concrete_trigger_examples_present(self, trigger_keyword):
        doc = _intent_action_doc()
        assert trigger_keyword in doc, (
            f"force= docstring should mention {trigger_keyword!r} as a "
            f"concrete trigger condition the LLM can match against a real "
            f"symptom. Helps the LLM reach for the flag when it's right."
        )

    def test_caveat_preserved_but_after_trigger(self):
        """The 'do NOT use for genuinely hidden elements' caveat must
        still appear — we're reframing, not removing safety guidance."""
        doc = _intent_action_doc()
        # Caveat must be present somewhere in the force section.
        assert "anti-pattern" in doc.lower() or "do not use" in doc.lower(), (
            "force= caveat about hidden-element misuse must survive the reframe"
        )

    def test_browser_only_constraint_documented(self):
        doc = _intent_action_doc()
        assert "Browser" in doc[doc.find("force:"):], (
            "force= must still document that it's Browser-library-specific "
            "(other libraries' force_keyword is ignored)"
        )


# ---------------------------------------------------------------------------
# commit=True
# ---------------------------------------------------------------------------


class TestCommitDocstringUseWhen:
    """``commit=True`` docstring must lead with a concrete trigger."""

    def test_use_when_phrase_present(self):
        doc = _intent_action_doc()
        assert "commit: Use when:" in doc, (
            "commit= docstring must lead with 'Use when:' (OBS-07). "
            "Found extract: " + doc[doc.find("commit:"): doc.find("commit:") + 200]
        )

    @pytest.mark.parametrize("framework_name", [
        "Vue", "React", "Angular", "jQuery", "idealForms",
    ])
    def test_lists_known_frameworks_that_need_commit(self, framework_name):
        # Naming the frameworks helps the LLM pattern-match against page
        # context (e.g., "this page uses Vue → commit=True is the right
        # tool here"). The story explicitly calls these out.
        doc = _intent_action_doc()
        assert framework_name in doc, (
            f"commit= docstring should name {framework_name!r} as a known "
            f"framework whose form validators need a real `change` event"
        )

    @pytest.mark.parametrize("symptom_phrase", [
        # Concrete user-observable symptom — what the LLM sees in the
        # failed page state that should prompt reaching for commit=True.
        "rejected",
        "validation error",
    ])
    def test_symptom_described_concretely(self, symptom_phrase):
        doc = _intent_action_doc()
        assert symptom_phrase in doc, (
            f"commit= docstring should describe the user-observable "
            f"symptom {symptom_phrase!r} so the LLM can pattern-match"
        )

    def test_best_effort_caveat_preserved(self):
        """The 'follow-up is best-effort; failure logged not escalated'
        guarantee must survive — it's load-bearing for callers who don't
        want commit=True turning a successful fill into a failed step."""
        doc = _intent_action_doc()
        commit_section = doc[doc.find("commit:"):]
        # Some marker word that conveys "won't escalate to failure".
        assert (
            "best-effort" in commit_section
            or "logged and ignored" in commit_section
            or "never escalates" in commit_section
        ), (
            "commit= must still document that follow-up failure does NOT "
            "escalate the original FILL into a failed step"
        )


# ---------------------------------------------------------------------------
# Cookbook cross-reference (OBS-05 / OBS-07)
# ---------------------------------------------------------------------------


@pytest.fixture
def rendered_default():
    tmpl = InstructionTemplate.discovery_first()
    return tmpl.render({"available_tools": "X"}).value


class TestCookbookCrossReferencesBothFlags:
    """The pre-validation recovery cookbook (OBS-05 / OBS-02 / OBS-07)
    must name both ``force=True`` and ``commit=True`` so the LLM reading
    the recovery recipe knows where the related escape hatches live.

    force=True is already named in step 4 (recovery step). commit=True
    is added by OBS-07 as a 'See also:' cross-reference, because it's
    not a pre-validation-recovery tool — it's a separate concern
    (SPA form-state commit after FILL) that the LLM might need to
    discover when reading section 8."""

    def test_force_named_in_recovery_section(self, rendered_default):
        # OBS-05 already put force=True into step 4 of section 8.
        # OBS-07 must not silently strip it.
        section_start = rendered_default.index("WHEN PRE-VALIDATION REJECTS")
        section_end = rendered_default.index("Available discovery tools")
        section = rendered_default[section_start:section_end]
        assert "force=True" in section

    def test_commit_named_in_cookbook_cross_reference(self, rendered_default):
        # OBS-07 acceptance #4: commit=True must be named in the cookbook
        # section.
        section_start = rendered_default.index("WHEN PRE-VALIDATION REJECTS")
        section_end = rendered_default.index("Available discovery tools")
        section = rendered_default[section_start:section_end]
        assert "commit=True" in section, (
            "OBS-07: commit=True must be cross-referenced in the cookbook "
            "section so an LLM reading the pre-validation recovery recipe "
            "also discovers the related FILL-on-SPA-form escape hatch."
        )

    def test_see_also_points_at_intent_action_docstring(self, rendered_default):
        # The cross-reference should direct the LLM to the docstring
        # for the full "Use when:" rationale, not duplicate the trigger
        # condition in the cookbook (which has its own size budget).
        section_start = rendered_default.index("WHEN PRE-VALIDATION REJECTS")
        section_end = rendered_default.index("Available discovery tools")
        section = rendered_default[section_start:section_end]
        assert "See also" in section
        # The cross-reference should name at least one of the SPA
        # frameworks so the LLM can match against page-source clues.
        assert any(fw in section for fw in ("Vue", "React", "Angular", "jQuery")), (
            "Cross-reference should at least name a recognisable SPA "
            "framework so the LLM can pattern-match against the page"
        )
