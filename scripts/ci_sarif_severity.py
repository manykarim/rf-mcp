"""Severity handling for the CI quality analysis (change: ci-signal-fidelity).

THE PROBLEM THIS EXISTS TO SOLVE
--------------------------------
Ruff's SARIF output carries no severity information. Measured on src/ (2026-09-24):

    295 results, every one level="error"
     18 rules,   every one properties["problem.severity"]="error"
     no rule carries defaultConfiguration

GitHub code scanning reads that constant, raises every finding as an error-level alert,
and fails a check named `ruff` when a pull request touches a line carrying one. That
directly violates the `ci-quality-scanning` requirement "Analysis reports findings
without failing the build", whose scenario says the pull request "shows no failed check
for it". PR #93 hit exactly this: it RELOCATED a `try/except: pass` and dropped an
unused `except ... as e` binding, introducing nothing, and still got a red check.

So the severity has to be assigned deliberately rather than inherited.

WHY THIS SET
------------
`HIGH` mirrors the rules bandit rates HIGH. Do NOT substitute "the S prefix means
security means high": of 222 `S` findings on this source only 6 are what bandit rates
HIGH, and 160 are `S110 try-except-pass`. Prefix is category, not severity.

Validated: this set selects 6 findings, exactly bandit's 6 HIGH for the same source
(all S324/B324). Re-run that comparison whenever the set changes - the spec requires it
("the declared set is justified, not asserted").

This module is the SINGLE source of the set. The digest step and the SARIF re-levelling
step both read it; a second inline copy would drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Rules published at error severity. Mirrors bandit's HIGH ratings.
HIGH: frozenset[str] = frozenset(
    {
        "S102",  # exec-builtin
        "S301",  # suspicious-pickle-usage
        "S302",  # suspicious-marshal-usage
        "S306",  # suspicious-mktemp-usage
        "S307",  # suspicious-eval-usage
        "S324",  # hashlib-insecure-hash-function
        "S602",  # subprocess-popen-with-shell-equals-true
        "S604",  # call-with-shell-equals-true
        "S605",  # start-process-with-a-shell
        "S606",  # start-process-with-no-shell
    }
)

# Severity for everything else. `warning` keeps the finding visible in the code-scanning
# UI and in pull-request annotations without failing the check under GitHub's default
# threshold. If the repository's threshold turns out to count warnings too, pass
# `--non-high note` rather than reaching for the repository setting: the setting is
# invisible state that applies to every tool, not just this one.
DEFAULT_NON_HIGH = "warning"


def _runs(doc: Dict) -> list:
    return doc.get("runs") or []


def count(sarif_path: Path) -> Tuple[int, int]:
    """Return (total findings, high-severity findings).

    Reads the ORIGINAL finding set: the digest reports every finding, not just the high
    ones. Severity mapping must never change what is counted.
    """
    doc = json.loads(sarif_path.read_text(encoding="utf-8"))
    results = [r for run in _runs(doc) for r in run.get("results", [])]
    high = sum(1 for r in results if (r.get("ruleId") or "") in HIGH)
    return len(results), high


def relevel(doc: Dict, non_high: str = DEFAULT_NON_HIGH) -> Tuple[int, int]:
    """Assign severity from HIGH, in place. Returns (n_high, n_non_high).

    Rewrites BOTH severity carriers - the per-result `level` and the rule's
    `properties["problem.severity"]` - because it is not documented which one code
    scanning prefers, and ruff currently sets them to the same constant.

    The result COUNT is never changed. Nothing is filtered, dropped or suppressed; only
    the severity each finding is published at changes.
    """
    n_high = n_non = 0
    for run in _runs(doc):
        for result in run.get("results", []):
            is_high = (result.get("ruleId") or "") in HIGH
            result["level"] = "error" if is_high else non_high
            if is_high:
                n_high += 1
            else:
                n_non += 1

        driver = (run.get("tool") or {}).get("driver") or {}
        for rule in driver.get("rules", []) or []:
            is_high = (rule.get("id") or "") in HIGH
            level = "error" if is_high else non_high
            props = rule.setdefault("properties", {})
            props["problem.severity"] = level
            rule.setdefault("defaultConfiguration", {})["level"] = level

    return n_high, n_non


def _cmd_count(args: argparse.Namespace) -> int:
    total, high = count(Path(args.sarif))
    print(f"count={total}")
    print(f"high={high}")
    return 0


def _cmd_relevel(args: argparse.Namespace) -> int:
    path = Path(args.sarif)
    doc = json.loads(path.read_text(encoding="utf-8"))
    before = sum(len(run.get("results", [])) for run in _runs(doc))
    n_high, n_non = relevel(doc, non_high=args.non_high)
    after = sum(len(run.get("results", [])) for run in _runs(doc))
    if before != after:  # pragma: no cover - guards against a future editing mistake
        print(
            f"::error::re-levelling changed the finding count {before} -> {after}; "
            "severity mapping must never drop findings",
            file=sys.stderr,
        )
        return 1
    path.write_text(json.dumps(doc), encoding="utf-8")
    print(f"re-levelled {after} findings: {n_high} error, {n_non} {args.non_high}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_count = sub.add_parser("count", help="print count=/high= for the digest")
    p_count.add_argument("sarif")
    p_count.set_defaults(func=_cmd_count)

    p_rel = sub.add_parser("relevel", help="assign severity from HIGH, in place")
    p_rel.add_argument("sarif")
    p_rel.add_argument(
        "--non-high",
        default=DEFAULT_NON_HIGH,
        choices=("warning", "note"),
        help="severity for findings outside the HIGH set (default: warning)",
    )
    p_rel.set_defaults(func=_cmd_relevel)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
