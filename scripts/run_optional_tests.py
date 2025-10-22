#!/usr/bin/env python3
"""Run optional dependency smoke tests for selected extras.

This helper installs the requested extras into the current uv environment and
executes the matching pytest marker selection. It assumes ``uv`` is available
on PATH and that ``uv sync`` has been run at least once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
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


@lru_cache(maxsize=1)
def current_python() -> str:
    """Return the interpreter path for the active uv project environment."""

    env_path = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if env_path:
        path = Path(env_path)
        bin_dir = "Scripts" if os.name == "nt" else "bin"
        candidate = path / bin_dir / ("python.exe" if os.name == "nt" else "python")
        if candidate.exists():
            return str(candidate)

    result = subprocess.check_output(
        ["uv", "run", "python", "-c", "import sys; print(sys.executable)"]
    )
    return result.decode().strip()


def install_extras(extras: str | None) -> None:
    """Install optional extras into the current uv project environment."""

    run_command(['uv', 'sync', '--frozen'])

    if extras:
        python_executable = current_python()
        run_command(['uv', 'pip', 'install', '--python', python_executable, '-e', f'.[{extras}]'])


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
        if not args.skip_install:
            install_extras(extras if isinstance(extras, str) else None)
        run_command(list(config["command"]))  # type: ignore[arg-type]

    return 0


if __name__ == "__main__":
    sys.exit(main())

