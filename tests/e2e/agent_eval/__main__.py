"""CLI: `uv run python -m tests.e2e.agent_eval <suite> [options]`, `... report <dir>`, `... trim <in> <out>`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from .graders import trim_transcript
from .report import write_report
from .runner import Harness, make_jobs, parse_conditions
from .scenarios import SUITES


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tests.e2e.agent_eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, spec in SUITES.items():
        p = sub.add_parser(name, help=f"run the {name} suite ({', '.join(spec.scenarios)})")
        p.add_argument("--scenarios", type=_csv, default=list(spec.scenarios), help="comma-separated subset")
        p.add_argument("--models", type=_csv, default=spec.models, help="haiku, sonnet or model ids")
        p.add_argument("--conditions", default=",".join(spec.conditions),
                       help="mcp, cli, mcp:<skills label|dir> (comma-separated)")
        p.add_argument("--skills", action="append", default=[], metavar="LABEL=DIR",
                       help="skills dir for the condition mcp:LABEL (passed to rf-mcp --skills-dir)")
        p.add_argument("--cli-skill", type=Path, default=os.environ.get("RF_EVAL_CLI_SKILL_DIR") or None,
                       help="upstream robotcode skill folder for the cli condition (RF_EVAL_CLI_SKILL_DIR)")
        p.add_argument("--reps", type=int, default=1)
        p.add_argument("--parallel", type=int, default=4)
        p.add_argument("--max-turns", type=int, default=None, help=f"default {spec.max_turns}")
        p.add_argument("--timeout", type=int, default=None, help=f"seconds per run, default {spec.timeout}")
        p.add_argument("--out", type=Path, default=None, help="results folder (default: temp dir)")
        p.add_argument("--dry-run", action="store_true",
                       help="seed the projects and print the claude commands and MCP configs without running claude")
    r = sub.add_parser("report", help="write report.md for a results folder")
    r.add_argument("dir", type=Path)
    t = sub.add_parser("trim", help="trim a stream-json transcript to the events the graders need")
    t.add_argument("source", type=Path)
    t.add_argument("dest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report":
        print(write_report(args.dir).read_text(encoding="utf-8"))
        return 0
    if args.command == "trim":
        events = trim_transcript(args.source)
        args.dest.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        return 0
    spec = SUITES[args.command]
    unknown = [s for s in args.scenarios if s not in spec.scenarios]
    if unknown:
        print(f"unknown scenarios for {spec.name}: {unknown}", file=sys.stderr)
        return 2
    skills = dict(s.split("=", 1) for s in args.skills)
    try:
        conditions = parse_conditions(args.conditions, skills)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    out = args.out or Path(tempfile.gettempdir()) / "rf-mcp-agent-eval" / f"{spec.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = out.resolve()
    harness = Harness(spec, out, dry_run=args.dry_run, cli_skill=args.cli_skill, max_turns=args.max_turns,
                      timeout=args.timeout)
    jobs = make_jobs(spec, args.scenarios, args.models, conditions, args.reps)
    result = harness.execute(jobs, args.parallel)
    if args.dry_run:
        print(f"\n{'all' if result else 'NOT all'} {len(jobs)} projects seeded under {harness.runs_dir}")
        return 0 if result else 1
    print(f"\nreport: {write_report(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
