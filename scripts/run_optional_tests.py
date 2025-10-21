#!/usr/bin/env python3
"""Run optional dependency smoke tests for selected extras.

This helper installs the requested extras into the current uv environment and
executes the matching pytest marker selection. It assumes ``uv`` is available
on PATH and that ``uv sync`` has been run at least once.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict

COMBO_CONFIG: Dict[str, Dict[str, str | list[str]]] = {
    "slim": {
        "extras": None,
        "command": ["uv", "run", "pytest", "tests/test_mcp_simple.py", "-q"],
    },
    "web": {
        "extras": "web",
        "command": ["uv", "run", "pytest", "-m", "optional_web", "-q"],
    },
    "api": {
        "extras": "api",
        "command": ["uv", "run", "pytest", "-m", "optional_api", "-q"],
    },
    "mobile": {
        "extras": "mobile",
        "command": ["uv", "run", "pytest", "-m", "optional_mobile", "-q"],
    },
    "database": {
        "extras": "database",
        "command": ["uv", "run", "pytest", "-m", "optional_database", "-q"],
    },
    "web+api": {
        "extras": "web,api",
        "command": ["uv", "run", "pytest", "-m", "optional_web_api", "-q"],
    },
    "web+mobile": {
        "extras": "web,mobile",
        "command": ["uv", "run", "pytest", "-m", "optional_web_mobile", "-q"],
    },
    "api+database": {
        "extras": "api,database",
        "command": ["uv", "run", "pytest", "-m", "optional_api_database", "-q"],
    },
    "all": {
        "extras": "all",
        "command": ["uv", "run", "pytest", "-m", "optional_all", "-q"],
    },
}


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "combos",
        nargs="*",
        default=list(COMBO_CONFIG),
        help="Optional dependency combos to test (default: all)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing extras (assumes environment already prepared)",
    )
    args = parser.parse_args()

    for combo in args.combos:
        if combo not in COMBO_CONFIG:
            raise SystemExit(f"Unknown combo '{combo}'")
        config = COMBO_CONFIG[combo]
        extras = config["extras"]
        print(f"\n=== Running optional dependency combo: {combo} ===")
        if extras and not args.skip_install:
            run_command(["uv", "pip", "install", "-e", f".[{extras}]" ])
        elif extras is None and not args.skip_install:
            run_command(["uv", "sync", "--frozen"])
        run_command(config["command"])  # type: ignore[arg-type]

    return 0


if __name__ == "__main__":
    sys.exit(main())

