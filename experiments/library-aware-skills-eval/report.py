"""Summarise harness results as Markdown tables and the D8 gate verdict."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in rows]
    pointed = sum(len(r["pointed"]) for r in rows)
    return {
        "n": n,
        "pass": sum(r["passed"] for r in rows) / n,
        "strict": sum(r["strict_pass"] for r in rows) / n,
        "turns": statistics.median(r["turns"] for r in rows),
        "tokens": statistics.median(tokens),
        "cost": sum(r["cost"] for r in rows),
        "pointed": pointed,
        "read": sum(len(r["read"]) for r in rows),
        "followed": sum(r["followed"] for r in rows),
        "errors": sum(1 for r in rows if r["stop"] in ("error", "timeout") or r["error"]),
    }


HEADER = "| {key} | n | pass | pass + bug caught | median turns | median tokens | cost $ | pointers | skill reads | reads of a pointed skill | errors |"


def line(label: str, s: dict) -> str:
    return (f"| {label} | {s['n']} | {s['pass']:.2f} | {s['strict']:.2f} | {s['turns']:.0f} | {s['tokens']:,.0f} | "
            f"{s['cost']:.3f} | {s['pointed']} | {s['read']} | {s['followed']} | {s['errors']} |")


def table(rows: list[dict], key: str, keyfn) -> list[str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[keyfn(r)].append(r)
    out = [HEADER.format(key=key), "|" + "---|" * 11]
    out += [line(k, summary(v)) for k, v in sorted(groups.items())]
    return out


def gate(rows: list[dict]) -> list[str]:
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ("A", "B", "C")}
    if not by_arm["A"] or not by_arm["B"]:
        return ["Gate: needs arms A and B."]
    a, b = summary(by_arm["A"]), summary(by_arm["B"])
    pass_ok = b["pass"] >= a["pass"]
    tokens_ok = b["tokens"] <= 1.1 * a["tokens"]
    verdict = "PASSES" if pass_ok and tokens_ok else "FAILS"
    return [
        f"Gate (D8): B pass {b['pass']:.2f} vs A {a['pass']:.2f} ({'ok' if pass_ok else 'not met'}); "
        f"B median tokens {b['tokens']:,.0f} vs 1.1 x A {1.1 * a['tokens']:,.0f} ({'ok' if tokens_ok else 'not met'}). "
        f"**The gate {verdict}.**"
    ]


def main() -> None:
    rows = load(Path(sys.argv[1]))
    print(f"{len(rows)} runs\n")
    print("\n".join(table(rows, "arm", lambda r: r["arm"])), "\n")
    print("\n".join(table(rows, "task / arm", lambda r: f"{r['task']} {r['arm']}")), "\n")
    print("\n".join(table(rows, "model / arm", lambda r: f"{r['model']} {r['arm']}")), "\n")
    print("\n".join(gate(rows)))


if __name__ == "__main__":
    main()
