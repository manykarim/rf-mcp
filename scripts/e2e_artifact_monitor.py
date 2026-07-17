#!/usr/bin/env python3
"""E2E Artifact Read Monitor.

Runs opencode e2e tests and monitors whether agents read externalized artifact files.
Tracks: artifact creation, agent file reads (via JSON output parsing), token savings.

Usage:
    uv run python scripts/e2e_artifact_monitor.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    "openrouter/qwen/qwen3-coder",
    "openrouter/z-ai/glm-4.5-air",
    "openrouter/minimax/minimax-m2.5",
]

SCENARIOS: dict[str, dict[str, Any]] = {
    "restful_booker": {
        "prompt": (
            "Use RobotMCP to test the Restful Booker API at https://restful-booker.herokuapp.com.\n"
            "1. Initialize a session for API testing with RequestsLibrary\n"
            "2. Read an existing booking (GET /booking/1) and assert the response\n"
            "3. Create a new booking (POST /booking) with firstname, lastname, totalprice, "
            "depositpaid, bookingdates, additionalneeds and assert the response\n"
            "4. Authenticate as admin (POST /auth with username=admin, password=password123)\n"
            "5. Build the test suite\n"
            "Execute each step and inspect the results. "
            "When you see 'Content saved to <filepath>', READ that file to get the full output. "
            "Use the file path from the response to read artifact files."
        ),
        "type": "api",
    },
    "demoshop_browse": {
        "prompt": (
            "Use RobotMCP to test the Demoshop at https://demoshop.makrocode.de/.\n"
            "1. Initialize a session for web testing with Browser library (headless=True)\n"
            "2. Open the URL https://demoshop.makrocode.de/\n"
            "3. Get the page state (DOM/ARIA snapshot) to understand the page structure\n"
            "4. Click on a product category to browse products\n"
            "5. Get the page state again to see the products\n"
            "6. Build the test suite\n"
            "Execute each step and inspect the results. "
            "When you see 'Content saved to <filepath>', READ that file to get the full output. "
            "Use the file path from the response to read artifact files."
        ),
        "type": "web",
    },
    "carconfig_browse": {
        "prompt": (
            "Use RobotMCP to test the Car Configurator at https://carconfig.makrocode.de/.\n"
            "1. Initialize a session for web testing with Browser library (headless=True)\n"
            "2. Open the URL https://carconfig.makrocode.de/\n"
            "3. Get the page state (DOM/ARIA snapshot) to understand the page structure\n"
            "4. Select a car model by clicking on it\n"
            "5. Get the page state again to see configuration options\n"
            "6. Build the test suite\n"
            "Execute each step and inspect the results. "
            "When you see 'Content saved to <filepath>', READ that file to get the full output. "
            "Use the file path from the response to read artifact files."
        ),
        "type": "web",
    },
}

TIMEOUT_SECONDS = 300  # 5 min per run
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / ".e2e_artifact_results"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ArtifactEvent:
    artifact_id: str
    file_path: str
    byte_size: int
    token_estimate: int
    tool_name: str
    field_name: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class FileReadEvent:
    file_path: str
    tool_name: str  # the tool that triggered the read (e.g. Read, cat)
    timestamp: float = field(default_factory=time.time)


@dataclass
class RunResult:
    scenario: str
    model: str
    start_time: float
    end_time: float = 0.0
    artifacts_created: list[ArtifactEvent] = field(default_factory=list)
    artifact_reads: list[FileReadEvent] = field(default_factory=list)
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    success: bool = False
    error: str = ""
    raw_events: int = 0

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def artifacts_read_count(self) -> int:
        created_paths = {a.file_path for a in self.artifacts_created}
        return sum(1 for r in self.artifact_reads if r.file_path in created_paths)

    @property
    def artifacts_read_ratio(self) -> float:
        if not self.artifacts_created:
            return 0.0
        return self.artifacts_read_count / len(self.artifacts_created)

    @property
    def tokens_saved(self) -> int:
        return sum(a.token_estimate for a in self.artifacts_created)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "model": self.model,
            "duration_s": round(self.duration_s, 1),
            "artifacts_created": len(self.artifacts_created),
            "artifacts_read": self.artifacts_read_count,
            "artifact_read_ratio": round(self.artifacts_read_ratio, 2),
            "tokens_saved_by_externalization": self.tokens_saved,
            "total_tool_calls": self.total_tool_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "success": self.success,
            "error": self.error,
            "raw_events": self.raw_events,
            "artifact_details": [
                {
                    "id": a.artifact_id,
                    "file": a.file_path,
                    "bytes": a.byte_size,
                    "tokens": a.token_estimate,
                    "tool": a.tool_name,
                    "field": a.field_name,
                    "was_read": a.file_path in {r.file_path for r in self.artifact_reads},
                }
                for a in self.artifacts_created
            ],
            "read_events": [
                {"file": r.file_path, "tool": r.tool_name}
                for r in self.artifact_reads
            ],
        }


# ---------------------------------------------------------------------------
# JSON event parser
# ---------------------------------------------------------------------------

# Patterns for artifact file paths in summaries
ARTIFACT_SUMMARY_RE = re.compile(
    r"Content saved to ([^\s]+\.(?:txt|json|html|robot))\s+\((\d+) bytes,\s+~(\d+) tokens\)"
)

# Old format: "Content externalized to artifact art_XXX (N bytes, ~M tokens)..."
# With optional "File: /path/to/file.ext"
OLD_ARTIFACT_RE = re.compile(
    r"Content externalized to artifact (art_[a-f0-9]{12})\s+\((\d+) bytes,\s+~(\d+) tokens\)"
)
FETCH_ARTIFACT_RE = re.compile(
    r"artifact (art_[a-f0-9]{12}).*?File:\s*([^\s.]+\.\w+)"
)


def parse_opencode_events(jsonl_path: Path, result: RunResult) -> None:
    """Parse opencode JSON output to extract artifact and read events.

    Opencode JSON format:
      {"type": "step_finish", "part": {"tokens": {"input": N, "output": M, ...}}}
      {"type": "tool_use", "part": {"type": "tool", "tool": "robotmcp_XXX",
          "state": {"input": {...}, "output": "..."}}}
      {"type": "text", "part": {"text": "..."}}
    """
    if not jsonl_path.exists():
        return

    seen_artifact_ids: set[str] = set()

    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # opencode sometimes emits multiple JSON objects on one line
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Try to find valid JSON object
            brace_start = line.find("{")
            if brace_start < 0:
                continue
            try:
                event = json.loads(line[brace_start:])
            except json.JSONDecodeError:
                continue

        if not isinstance(event, dict):
            continue

        result.raw_events += 1
        etype = event.get("type", "")
        part = event.get("part", {})
        if not isinstance(part, dict):
            part = {}

        # Track token usage from step_finish events
        if etype == "step_finish":
            tokens = part.get("tokens", {})
            if isinstance(tokens, dict):
                result.total_input_tokens += tokens.get("input", 0)
                result.total_output_tokens += tokens.get("output", 0)

        # Track tool calls and inspect their inputs/outputs
        if part.get("type") == "tool" or etype == "tool_use":
            result.total_tool_calls += 1
            tool_name = part.get("tool", "")
            state = part.get("state", {})
            if not isinstance(state, dict):
                state = {}

            tool_input = state.get("input", {})
            tool_output = state.get("output", "")

            # Convert output to string for scanning
            output_str = ""
            if isinstance(tool_output, str):
                output_str = tool_output
            elif isinstance(tool_output, dict):
                output_str = json.dumps(tool_output, default=str)

            # Detect artifact creation summaries in tool output
            for m in ARTIFACT_SUMMARY_RE.finditer(output_str):
                fpath, bsize, toks = m.group(1), int(m.group(2)), int(m.group(3))
                aid_match = re.search(r"(art_[a-f0-9]{12})", fpath)
                aid = aid_match.group(1) if aid_match else fpath
                if aid not in seen_artifact_ids:
                    seen_artifact_ids.add(aid)
                    result.artifacts_created.append(ArtifactEvent(
                        artifact_id=aid,
                        file_path=fpath,
                        byte_size=bsize,
                        token_estimate=toks,
                        tool_name=tool_name,
                        field_name="",
                    ))

            # Old-format: "Content externalized to artifact art_XXX (N bytes, ~M tokens)"
            for m in OLD_ARTIFACT_RE.finditer(output_str):
                aid, bsize, toks = m.group(1), int(m.group(2)), int(m.group(3))
                # Try to find file path from "File: /path" suffix
                fpath = ""
                fm = re.search(rf"{re.escape(aid)}.*?File:\s*(\S+)", output_str)
                if fm:
                    fpath = fm.group(1).rstrip(".")
                if aid not in seen_artifact_ids:
                    seen_artifact_ids.add(aid)
                    result.artifacts_created.append(ArtifactEvent(
                        artifact_id=aid,
                        file_path=fpath,
                        byte_size=bsize,
                        token_estimate=toks,
                        tool_name=tool_name,
                        field_name="",
                    ))

            # Detect file reads of artifact paths in tool input
            if isinstance(tool_input, dict):
                for key in ("file_path", "filePath", "path", "filename", "command"):
                    val = tool_input.get(key, "")
                    if isinstance(val, str) and ("robotmcp_artifacts" in val or "rfmcp_arts" in val):
                        actual_path = val if key != "command" else _extract_path_from_cmd(val)
                        result.artifact_reads.append(FileReadEvent(
                            file_path=actual_path,
                            tool_name=tool_name,
                        ))

            # Also detect artifact paths in bash/shell commands
            if isinstance(tool_input, dict) and "command" in tool_input:
                cmd = tool_input["command"]
                if isinstance(cmd, str) and ("robotmcp_artifacts" in cmd or "rfmcp_arts" in cmd):
                    result.artifact_reads.append(FileReadEvent(
                        file_path=_extract_path_from_cmd(cmd),
                        tool_name=tool_name,
                    ))

            # Detect if a Read/cat tool output contains artifact content (confirms read)
            if tool_name in ("Read", "read_file", "read", "cat", "head", "tail"):
                if isinstance(tool_input, dict):
                    fp = tool_input.get("file_path", tool_input.get("filePath", tool_input.get("path", "")))
                    if isinstance(fp, str) and ("robotmcp_artifacts" in fp or "rfmcp_arts" in fp):
                        result.artifact_reads.append(FileReadEvent(
                            file_path=fp,
                            tool_name=tool_name,
                        ))

        # Also check text events for assistant mentioning artifact files
        if part.get("type") == "text":
            text = part.get("text", "")
            if isinstance(text, str) and "robotmcp_artifacts" in text:
                # Agent is discussing artifact files — check if it mentions reading
                for art in result.artifacts_created:
                    if art.file_path in text:
                        # Note: this is just a mention, not a confirmed read
                        pass  # We only count actual tool reads


def _extract_path_from_cmd(cmd: str) -> str:
    """Extract artifact path from a shell command like 'cat /path/artifacts/art_xxx.txt'."""
    m = re.search(r"([^\s]*robotmcp_artifacts[^\s]+)", cmd)
    if not m:
        m = re.search(r"([^\s]*rfmcp_arts[^\s]+)", cmd)
    return m.group(1) if m else cmd


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(scenario_name: str, model: str, artifact_dir: str) -> RunResult:
    """Run a single opencode scenario and collect metrics."""
    config = SCENARIOS[scenario_name]
    prompt = config["prompt"]

    result = RunResult(
        scenario=scenario_name,
        model=model.split("/")[-1],
        start_time=time.time(),
    )

    output_file = OUTPUT_DIR / f"{scenario_name}_{model.split('/')[-1]}_{int(time.time())}.jsonl"

    env = os.environ.copy()
    # Load API key from .env if not already set
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    env["ROBOTMCP_OUTPUT_MODE"] = "auto"
    env["ROBOTMCP_MAX_INLINE_TOKENS"] = "300"
    env["ROBOTMCP_ARTIFACT_DIR"] = artifact_dir
    # Ensure fetch_artifact is NOT enabled — we want file-path mode
    env.pop("ROBOTMCP_FETCH_ARTIFACT", None)

    cmd = [
        "opencode", "run",
        "--model", model,
        "--format", "json",
        prompt,
    ]

    print(f"\n{'='*70}")
    print(f"Running: {scenario_name} with {model.split('/')[-1]}")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Output: {output_file}")
    print(f"{'='*70}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=env,
            cwd="/home/many/workspace/rf-mcp",
        )
        output_file.write_text(proc.stdout or "")
        result.end_time = time.time()
        result.success = proc.returncode == 0

        if proc.stderr:
            # Check for error indicators
            if "error" in proc.stderr.lower()[:200]:
                result.error = proc.stderr[:500]

    except subprocess.TimeoutExpired:
        result.end_time = time.time()
        result.error = f"Timeout after {TIMEOUT_SECONDS}s"
    except Exception as e:
        result.end_time = time.time()
        result.error = str(e)

    # Parse the output
    parse_opencode_events(output_file, result)

    # Also check artifact dir for files created
    art_dir = Path(artifact_dir)
    if art_dir.exists():
        for f in art_dir.iterdir():
            if f.name.startswith("art_"):
                # Check access time vs modify time to detect reads
                stat = f.stat()
                if stat.st_atime > stat.st_mtime + 1:
                    # File was accessed after creation — likely read by agent
                    result.artifact_reads.append(FileReadEvent(
                        file_path=str(f),
                        tool_name="filesystem_access",
                    ))

    return result


def print_result(r: RunResult) -> None:
    """Print a summary of a run result."""
    print(f"\n--- {r.scenario} / {r.model} ---")
    print(f"  Duration: {r.duration_s:.1f}s")
    print(f"  Tool calls: {r.total_tool_calls}")
    print(f"  Tokens: {r.total_input_tokens:,} in / {r.total_output_tokens:,} out")
    print(f"  Artifacts created: {len(r.artifacts_created)}")
    print(f"  Artifacts READ by agent: {r.artifacts_read_count}")
    print(f"  Artifact read ratio: {r.artifacts_read_ratio:.0%}")
    print(f"  Tokens saved by externalization: ~{r.tokens_saved:,}")
    if r.error:
        print(f"  ERROR: {r.error[:200]}")
    for a in r.artifacts_created:
        read_marker = "READ" if a.file_path in {rd.file_path for rd in r.artifact_reads} else "NOT READ"
        print(f"    [{read_marker}] {a.artifact_id} ({a.byte_size}B, ~{a.token_estimate}tok) from {a.tool_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter scenarios/models from CLI args: --model MODEL --scenario SCENARIO
    selected_scenarios = list(SCENARIOS.keys())
    selected_models = MODELS

    args = sys.argv[1:]
    i = 0
    cli_models: list[str] = []
    cli_scenarios: list[str] = []
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            cli_models.append(args[i + 1])
            i += 2
        elif args[i] in SCENARIOS:
            cli_scenarios.append(args[i])
            i += 1
        else:
            i += 1
    if cli_models:
        selected_models = [m for m in MODELS if any(cm in m for cm in cli_models)]
        if not selected_models:
            selected_models = cli_models  # use as-is
    if cli_scenarios:
        selected_scenarios = cli_scenarios

    results: list[RunResult] = []

    for model in selected_models:
        for scenario_name in selected_scenarios:
            # Each run gets its own in-workspace artifact dir (opencode sandbox blocks /tmp/)
            model_short = model.split("/")[-1]
            artifact_dir = str(PROJECT_ROOT / f".robotmcp_artifacts_e2e/{scenario_name}_{model_short}_{int(time.time())}")
            Path(artifact_dir).mkdir(parents=True, exist_ok=True)
            r = run_scenario(scenario_name, model, artifact_dir)
            results.append(r)
            print_result(r)

    # Summary report
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")
    print(f"{'Scenario':<20} {'Model':<20} {'Arts':<6} {'Read':<6} {'Ratio':<8} {'TokSaved':<10} {'Calls':<6}")
    print("-" * 76)
    for r in results:
        print(
            f"{r.scenario:<20} {r.model:<20} "
            f"{len(r.artifacts_created):<6} {r.artifacts_read_count:<6} "
            f"{r.artifacts_read_ratio:<8.0%} {r.tokens_saved:<10,} {r.total_tool_calls:<6}"
        )

    # Save full report
    report_path = OUTPUT_DIR / f"artifact_read_report_{int(time.time())}.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "output_mode": "auto",
            "max_inline_tokens": 300,
            "fetch_artifact_enabled": False,
        },
        "summary": {
            "total_runs": len(results),
            "total_artifacts_created": sum(len(r.artifacts_created) for r in results),
            "total_artifacts_read": sum(r.artifacts_read_count for r in results),
            "overall_read_ratio": (
                sum(r.artifacts_read_count for r in results) /
                max(1, sum(len(r.artifacts_created) for r in results))
            ),
            "total_tokens_saved": sum(r.tokens_saved for r in results),
        },
        "runs": [r.to_dict() for r in results],
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
