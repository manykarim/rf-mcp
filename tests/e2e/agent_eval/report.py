"""Markdown results tables from `*.result.json` files."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

FLAG_COLUMNS = [("output_xml", "oXML"), ("file_selection", "file-sel"), ("run_before_analyze", "no-analyze"),
                ("open_session", "open-sess"), ("results_late", "results-late"), ("unverified", "unverified")]


def load_results(directory: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(directory).glob("*.result.json"))]


def _passed(r: dict) -> bool:
    return bool((r.get("grade") or {}).get("pass"))


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) else 0.0


def _row(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: list[str], rows: list[list]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)]


def render_markdown(results: list[dict], title: str = "Agent evaluation") -> str:
    """Per scenario x model x condition: pass/total, tests, turns, cost, failed sends, process flags, isolation."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        cells[(r.get("suite", ""), r.get("scenario", ""), r.get("model", ""), r.get("condition", ""))].append(r)
    lines = [f"# {title}", "", f"{len(results)} runs.", "", "## Runs by scenario", ""]
    rows = []
    for (suite, scen, model, cond), rs in sorted(cells.items()):
        n = len(rs)
        grades = [r.get("grade") or {} for r in rs]
        procs = [r.get("process") or {} for r in rs]
        tests = [g for g in grades if isinstance(g.get("total"), int)]
        test_cell = f"{sum(g.get('passed') or 0 for g in tests)}/{sum(g['total'] for g in tests)}" if tests else "-"
        iso = [g for g in grades if isinstance(g.get("isolated_pass"), int)]
        iso_cell = f"{sum(g['isolated_pass'] for g in iso)}/{sum(g.get('total') or 0 for g in iso)}" if iso else "-"
        bugs = [g for g in grades if isinstance(g.get("bugs_caught"), dict)]
        bug_cell = f"{sum(g.get('bugs_caught_n') or 0 for g in bugs)}/{sum(len(g['bugs_caught']) for g in bugs)}" if bugs else "-"
        flags = [sum(name in (p.get("flags") or []) for p in procs) for name, _ in FLAG_COLUMNS]
        rows.append([suite, scen, model, cond, f"{sum(map(_passed, rs))}/{n}", test_cell,
                     f"{sum(_num(r.get('num_turns')) for r in rs) / n:.1f}", f"{sum(_num(r.get('cost')) for r in rs):.2f}",
                     f"{sum(_num(p.get('failed_sends')) for p in procs) / n:.1f}", bug_cell, iso_cell, *flags])
    lines += _table(["suite", "scenario", "model", "cond", "pass", "tests", "turns (mean)", "cost $", "failed sends (mean)",
                     "bugs caught", "isolated", *(label for _, label in FLAG_COLUMNS)], rows)

    totals: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        totals[(r.get("model", ""), r.get("condition", ""))].append(r)
    rows = []
    for (model, cond), rs in sorted(totals.items()):
        flagged = sum(bool((r.get("process") or {}).get("flags")) for r in rs)
        oxml = sum("output_xml" in ((r.get("process") or {}).get("flags") or []) for r in rs)
        turns = sum(_num(r.get("num_turns")) for r in rs)
        rows.append([model, cond, f"{sum(map(_passed, rs))}/{len(rs)}", f"{turns:.0f}", f"{turns / len(rs):.1f}",
                     f"{sum(_num(r.get('cost')) for r in rs):.2f}", f"{sum(_num(r.get('in_tokens')) for r in rs) / len(rs) / 1000:.0f}",
                     f"{sum(_num(r.get('wall_s')) for r in rs):.0f}", oxml, flagged])
    lines += ["", "## Totals by model and condition", ""]
    lines += _table(["model", "cond", "pass", "turns", "turns/run", "cost $", "in k tok/run", "wall s", "output.xml reads", "runs flagged"], rows)

    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for (model, cond), rs in totals.items():
        by_model[model][cond] = sum(_num(r.get("num_turns")) for r in rs)
    ratios = [f"{m}: mcp/cli turns {t['mcp'] / t['cli']:.2f}x" for m, t in sorted(by_model.items())
              if t.get("cli") and "mcp" in t]
    if ratios:
        lines += ["", "Acceptance bar: MCP turns <= 1.3x CLI, 0 output.xml reads. " + "; ".join(ratios) + "."]
    return "\n".join(lines) + "\n"


def write_report(directory: Path, title: str | None = None) -> Path:
    results = load_results(directory)
    path = Path(directory) / "report.md"
    path.write_text(render_markdown(results, title or f"Agent evaluation: {Path(directory).name}"), encoding="utf-8")
    return path
