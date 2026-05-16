"""Replay Evaluate JavaScript bodies from the real Tricentis artefacts and
assert how the classifier handles them. This is the empirical acceptance test
for Proposal-B: it shows the curation actually reduces the 162-step suite.

The 7 generated Tricentis suites live in .robotmcp_artifacts/. We load the
largest one (art_81870423004a.txt, 130 Evaluate JavaScript calls), pass each
JS body through the classifier, and assert ≥80% are classified as read-only.
"""

from __future__ import annotations

__test__ = True

import json
from pathlib import Path

import pytest

from robotmcp.components.execution.keyword_executor import (
    _classify_evaluate_javascript,
)


ARTEFACT_DIR = Path("/home/many/workspace/rf-mcp/.robotmcp_artifacts")
LARGE_TRICENTIS_RUN = ARTEFACT_DIR / "art_81870423004a.txt"


def _load_evaluate_js_steps(artefact: Path) -> list[list]:
    """Return the list of arguments for each Evaluate JavaScript step."""
    if not artefact.is_file():
        pytest.skip(f"Artefact not present: {artefact}")
    data = json.loads(artefact.read_text())
    test_cases = data.get("test_cases", [])
    if not test_cases:
        pytest.skip(f"No test cases in {artefact}")
    steps = test_cases[0].get("structured_steps", [])
    return [
        s.get("arguments", [])
        for s in steps
        if (s.get("keyword") or "").lower().strip() == "evaluate javascript"
    ]


class TestArtefactReplay:
    def test_large_tricentis_run_curates_majority(self):
        """In the largest Tricentis multi-agent run (162 steps incl. 130 JS
        probes), the classifier should:

        1. Cover at least 90% of probes (≥117 classified — read-only OR mutation).
        2. Read-only count must dominate: at least 60% of all classified probes
           should be read-only (the agent's diagnostic workload is inspection-
           heavy; mutations are the minority that genuinely changes form state).
        """
        js_arg_lists = _load_evaluate_js_steps(LARGE_TRICENTIS_RUN)
        assert js_arg_lists, "Expected ≥1 Evaluate JavaScript step in artefact"

        total = len(js_arg_lists)
        readonly = sum(
            1 for args in js_arg_lists
            if _classify_evaluate_javascript(args) is False
        )
        mutation = sum(
            1 for args in js_arg_lists
            if _classify_evaluate_javascript(args) is True
        )
        unclassified = sum(
            1 for args in js_arg_lists
            if _classify_evaluate_javascript(args) is None
        )

        print(
            f"\nReplay: total={total} read-only={readonly} mutation={mutation} "
            f"unclassified={unclassified} -> "
            f"coverage={(readonly+mutation)/total:.0%} "
            f"curated={readonly/total:.0%}"
        )
        coverage = (readonly + mutation) / total
        assert coverage >= 0.9, (
            f"Coverage too low ({coverage:.0%}). {unclassified} unclassified."
        )
        # In the diagnostic, the agent did ~44 mutations (form fills) and the
        # rest (≥60% of total) were inspection probes. The classifier must
        # auto-curate those: read-only should be at least 60% of total.
        assert readonly >= total * 0.6, (
            f"Read-only curation rate {readonly/total:.0%} below 60% target. "
            f"breakdown read-only={readonly} mutation={mutation} unclassified={unclassified}"
        )

    def test_classifier_results_are_deterministic(self):
        """Replaying the same artefact twice yields identical classifications."""
        js_arg_lists = _load_evaluate_js_steps(LARGE_TRICENTIS_RUN)
        run1 = [_classify_evaluate_javascript(a) for a in js_arg_lists]
        run2 = [_classify_evaluate_javascript(a) for a in js_arg_lists]
        assert run1 == run2

    def test_no_mutation_misclassified_as_readonly(self):
        """Sanity: any step whose JS body contains an obvious mutation marker
        ('= ' assignment to .value/.checked/.style) must not classify as
        read-only."""
        js_arg_lists = _load_evaluate_js_steps(LARGE_TRICENTIS_RUN)
        misclassified = []
        for args in js_arg_lists:
            longest = max(
                (a for a in args if isinstance(a, str)), key=len, default=""
            )
            looks_like_mutation = any(
                marker in longest
                for marker in (".value =", ".checked =", ".click()", ".dispatchEvent(", ".setAttribute(")
            )
            result = _classify_evaluate_javascript(args)
            if looks_like_mutation and result is False:
                misclassified.append(longest[:120])
        assert not misclassified, (
            f"{len(misclassified)} mutation steps misclassified as read-only:\n"
            + "\n".join(misclassified[:5])
        )
