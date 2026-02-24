"""E2E tests for PlatynUI desktop automation with multiple small LLM models.

Runs opencode with each model and verifies that the AI correctly discovers
and uses RobotMCP tools for desktop automation tasks with PlatynUI.

Tests:
1. Session setup: analyze_scenario + manage_session with PlatynUI.BareMetal
2. Desktop keyword execution: execute_step with PlatynUI keywords
3. Calculator interaction: Full workflow with pointer/keyboard/query
4. Session type detection: context="desktop" maps to DESKTOP_TESTING

Usage:
    # Run with default models
    RUN_PLATYNUI_E2E=true uv run pytest tests/e2e/test_platynui_desktop_models.py -v -s

    # Run with specific models
    OPENCODE_MODELS="openrouter/qwen/qwen3-coder,openrouter/z-ai/glm-4.7" \
        RUN_PLATYNUI_E2E=true uv run pytest tests/e2e/test_platynui_desktop_models.py -v -s

    # Run standalone script
    uv run python tests/e2e/test_platynui_desktop_models.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------

_DEFAULT_MODELS = [
    "opencode/glm-5-free",
    "opencode/gpt-5-nano",
    "opencode/minimax-m2.5-free",
]

# Alternative paid models (require OPENROUTER_API_KEY):
# "openrouter/qwen/qwen3-coder",
# "openrouter/z-ai/glm-4.7",
# "openrouter/z-ai/glm-4.7-flash",

_env_models = os.getenv("OPENCODE_MODELS", "").strip()
MODELS = (
    [m.strip() for m in _env_models.split(",") if m.strip()]
    if _env_models
    else _DEFAULT_MODELS
)

# ---------------------------------------------------------------------------
# Prompts — desktop / PlatynUI
# ---------------------------------------------------------------------------

# Prompt 1: Basic PlatynUI session setup + keyword discovery
PLATYNUI_SETUP_PROMPT = """\
You have access to Robot Framework MCP tools (robotmcp).
Your task is to set up a desktop automation session using PlatynUI.

Steps:
1. Call analyze_scenario with scenario="Desktop automation: test GNOME Calculator using PlatynUI" and context="desktop"
2. Call manage_session with action="init", using the session_id from step 1, with libraries=["PlatynUI.BareMetal", "BuiltIn"]
3. Call find_keywords with query="pointer click" and session_id from step 1
4. Call get_keyword_info with keyword="Get Pointer Position" and session_id from step 1

Report what you found. It's OK if some steps produce warnings — I need to verify you called the correct tools.
"""

# Prompt 2: Execute PlatynUI desktop steps
PLATYNUI_EXECUTE_PROMPT = """\
You have access to Robot Framework MCP tools (robotmcp).
Set up a PlatynUI desktop session and execute some steps.

Steps:
1. Call analyze_scenario with scenario="Test GNOME Calculator desktop app with PlatynUI" and context="desktop"
2. Call manage_session with action="init", session_id from step 1, libraries=["PlatynUI.BareMetal", "BuiltIn"]
3. Call execute_step with keyword="Get Pointer Position" and session_id from step 1
4. Call execute_step with keyword="Sleep" and args=["1s"] and session_id from step 1
5. Call get_session_state with session_id from step 1

Report results. Failures are expected (no real desktop) — I need to verify tool calls.
"""

# Prompt 3: Full calculator workflow (realistic)
PLATYNUI_CALCULATOR_PROMPT = """\
You have access to Robot Framework MCP tools.
Create and execute a test for GNOME Calculator using PlatynUI desktop automation.

Steps:
1. analyze_scenario: scenario="Automate GNOME Calculator with PlatynUI to verify 2+3=5", context="desktop"
2. manage_session: action="init", use session_id from step 1, libraries=["PlatynUI.BareMetal", "BuiltIn"]
3. manage_session: action="start_test", test_name="Calculator Addition", session_id from step 1
4. execute_step: keyword="Get Pointer Position", session_id from step 1
5. execute_step: keyword="Query", args=["//control:Window[@Name='Calculator']"], session_id from step 1
6. execute_step: keyword="Pointer Click", args=["None"], kwargs={"x": "400", "y": "350"}, session_id from step 1
7. manage_session: action="end_test", session_id from step 1
8. build_test_suite: session_id from step 1

Report all results. Failures are expected. I need to verify the tool call sequence.
"""

# Prompt 4: Short prompt for weaker models
PLATYNUI_SHORT_PROMPT = """\
Call these MCP tools in order:
1. analyze_scenario(scenario="Desktop test with PlatynUI", context="desktop")
2. manage_session(action="init", session_id=<from step 1>, libraries=["PlatynUI.BareMetal","BuiltIn"])
3. execute_step(keyword="Get Pointer Position", session_id=<from step 1>)
4. execute_step(keyword="Sleep", args=["1s"], session_id=<from step 1>)

Failures are OK — I just need to see you call the tools correctly.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Parsed tool call from opencode JSON output."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    success: bool = False


@dataclass
class ModelResult:
    """Result of testing a single model."""

    model: str
    prompt_id: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    text_output: str = ""
    error: str | None = None
    duration_seconds: float = 0.0
    total_tokens: int = 0
    cost: float = 0.0

    @property
    def used_analyze_scenario(self) -> bool:
        return any(tc.name == "robotmcp_analyze_scenario" for tc in self.tool_calls)

    @property
    def used_manage_session(self) -> bool:
        return any(tc.name == "robotmcp_manage_session" for tc in self.tool_calls)

    @property
    def used_execute_step(self) -> bool:
        return any(tc.name == "robotmcp_execute_step" for tc in self.tool_calls)

    @property
    def used_build_test_suite(self) -> bool:
        return any(tc.name == "robotmcp_build_test_suite" for tc in self.tool_calls)

    @property
    def used_find_keywords(self) -> bool:
        return any(tc.name == "robotmcp_find_keywords" for tc in self.tool_calls)

    @property
    def used_get_keyword_info(self) -> bool:
        return any(tc.name == "robotmcp_get_keyword_info" for tc in self.tool_calls)

    @property
    def used_get_session_state(self) -> bool:
        return any(tc.name == "robotmcp_get_session_state" for tc in self.tool_calls)

    @property
    def used_intent_action(self) -> bool:
        return any(tc.name == "robotmcp_intent_action" for tc in self.tool_calls)

    @property
    def passed_context_desktop(self) -> bool:
        """Check if context='desktop' was passed to analyze_scenario."""
        for tc in self.tool_calls:
            if tc.name == "robotmcp_analyze_scenario":
                ctx = tc.args.get("context", "")
                if ctx == "desktop":
                    return True
        return False

    @property
    def passed_platynui_library(self) -> bool:
        """Check if PlatynUI.BareMetal was in libraries list for manage_session."""
        for tc in self.tool_calls:
            if tc.name == "robotmcp_manage_session":
                libs = tc.args.get("libraries", [])
                if isinstance(libs, list):
                    for lib in libs:
                        if "platynui" in str(lib).lower():
                            return True
        return False

    @property
    def session_type_from_analyze(self) -> str | None:
        """Extract session_type from analyze_scenario result."""
        for tc in self.tool_calls:
            if tc.name == "robotmcp_analyze_scenario" and tc.result:
                try:
                    # Try to parse the result JSON
                    data = json.loads(tc.result) if isinstance(tc.result, str) else tc.result
                    if isinstance(data, dict):
                        return data.get("session_type") or data.get("result", {}).get("session_type")
                except (json.JSONDecodeError, TypeError):
                    # Search for session_type in string
                    if "desktop_testing" in str(tc.result).lower():
                        return "desktop_testing"
                    elif "mobile_testing" in str(tc.result).lower():
                        return "mobile_testing"
                    elif "web_automation" in str(tc.result).lower():
                        return "web_automation"
        return None

    @property
    def execute_step_keywords(self) -> list[str]:
        """Get keywords used in execute_step calls."""
        keywords = []
        for tc in self.tool_calls:
            if tc.name == "robotmcp_execute_step":
                kw = tc.args.get("keyword", "")
                if kw:
                    keywords.append(kw)
        return keywords

    @property
    def all_tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]

    @property
    def tool_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tc in self.tool_calls:
            counts[tc.name] = counts.get(tc.name, 0) + 1
        return counts

    @property
    def workflow_score(self) -> int:
        """Score the workflow completeness (0-5)."""
        score = 0
        if self.used_analyze_scenario:
            score += 1
        if self.used_manage_session:
            score += 1
        if self.used_execute_step:
            score += 1
        if self.passed_context_desktop:
            score += 1
        if self.passed_platynui_library:
            score += 1
        return score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_opencode_json(output: str) -> ModelResult:
    """Parse opencode JSON output into structured result."""
    result = ModelResult(model="")
    text_parts = []

    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "error":
            err = event.get("error", {})
            result.error = err.get("data", {}).get("message", str(err))

        elif event_type == "text":
            part = event.get("part", {})
            text_parts.append(part.get("text", ""))

        elif event_type == "tool_use":
            part = event.get("part", {})
            state = part.get("state", {})
            tc = ToolCall(
                name=part.get("tool", ""),
                args=state.get("input", {}),
                success=state.get("status") == "completed",
            )
            output_data = state.get("output")
            if output_data is not None:
                tc.result = str(output_data)[:1000] if output_data else ""
            result.tool_calls.append(tc)

        elif event_type == "step_finish":
            part = event.get("part", {})
            tokens = part.get("tokens", {})
            result.total_tokens += tokens.get("total", 0)
            result.cost += part.get("cost", 0.0)

    result.text_output = "\n".join(text_parts)
    return result


def _run_model(model: str, prompt: str, timeout_seconds: int = 300) -> ModelResult:
    """Run opencode with a specific model and prompt."""
    start = time.time()
    project_dir = str(Path(__file__).parent.parent.parent)

    cmd = [
        "opencode", "run",
        "--format", "json",
        "-m", model,
        "--dir", project_dir,
        prompt,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=project_dir,
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return ModelResult(
            model=model,
            error=f"Timeout after {timeout_seconds}s",
            duration_seconds=time.time() - start,
        )
    except Exception as e:
        return ModelResult(
            model=model,
            error=str(e),
            duration_seconds=time.time() - start,
        )

    result = _parse_opencode_json(output)
    result.model = model
    result.duration_seconds = time.time() - start
    return result


def _print_result(result: ModelResult) -> None:
    """Print a formatted summary of a model's test result."""
    model_short = result.model.split("/")[-1]
    score = result.workflow_score
    status = "PASS" if score >= 3 else "PARTIAL" if score >= 1 else "FAIL"

    print(f"\n{'=' * 70}")
    print(f"Model: {model_short} [{status}] score={score}/5 prompt={result.prompt_id}")
    print(f"{'=' * 70}")

    if result.error:
        print(f"  ERROR: {result.error}")
        return

    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Tokens: {result.total_tokens}")
    print(f"  Cost: ${result.cost:.4f}")
    print(f"  Total tool calls: {len(result.tool_calls)}")
    print(f"  Tools used: {result.tool_counts}")

    # Desktop-specific checks
    print(f"  analyze_scenario: {'YES' if result.used_analyze_scenario else 'NO'}")
    print(f"  context='desktop': {'YES' if result.passed_context_desktop else 'NO'}")
    print(f"  manage_session: {'YES' if result.used_manage_session else 'NO'}")
    print(f"  PlatynUI in libs: {'YES' if result.passed_platynui_library else 'NO'}")
    print(f"  execute_step: {'YES' if result.used_execute_step else 'NO'}")
    print(f"  session_type: {result.session_type_from_analyze or 'N/A'}")
    if result.execute_step_keywords:
        print(f"  Keywords executed: {result.execute_step_keywords}")


def _should_run() -> bool:
    """Check if PlatynUI E2E tests should run."""
    return os.getenv("RUN_PLATYNUI_E2E", "false").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Test prompts map
# ---------------------------------------------------------------------------

PLATYNUI_PROMPTS = {
    "setup": {
        "name": "PlatynUI Session Setup",
        "prompt": PLATYNUI_SETUP_PROMPT,
        "expected_tools": [
            "robotmcp_analyze_scenario",
            "robotmcp_manage_session",
            "robotmcp_find_keywords",
            "robotmcp_get_keyword_info",
        ],
    },
    "execute": {
        "name": "PlatynUI Execute Steps",
        "prompt": PLATYNUI_EXECUTE_PROMPT,
        "expected_tools": [
            "robotmcp_analyze_scenario",
            "robotmcp_manage_session",
            "robotmcp_execute_step",
            "robotmcp_get_session_state",
        ],
    },
    "calculator": {
        "name": "PlatynUI Calculator Full",
        "prompt": PLATYNUI_CALCULATOR_PROMPT,
        "expected_tools": [
            "robotmcp_analyze_scenario",
            "robotmcp_manage_session",
            "robotmcp_execute_step",
            "robotmcp_build_test_suite",
        ],
    },
}


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def platynui_results() -> dict[str, list[ModelResult]]:
    """Run all models against all PlatynUI prompts. Returns {prompt_id: [results]}."""
    all_results: dict[str, list[ModelResult]] = {}

    for prompt_id, prompt_info in PLATYNUI_PROMPTS.items():
        results = []
        print(f"\n{'#' * 80}")
        print(f"# Prompt: {prompt_info['name']} ({prompt_id})")
        print(f"{'#' * 80}")

        for model in MODELS:
            print(f"\n>>> [{prompt_id}] Running model: {model}")
            result = _run_model(model, prompt_info["prompt"], timeout_seconds=240)
            result.prompt_id = prompt_id
            _print_result(result)
            results.append(result)

            # If model scored poorly, retry with short prompt
            if result.workflow_score < 2 and not result.error and prompt_id != "calculator":
                print(f"  Retrying with short prompt...")
                result2 = _run_model(model, PLATYNUI_SHORT_PROMPT, timeout_seconds=120)
                result2.prompt_id = f"{prompt_id}_short"
                _print_result(result2)
                if result2.workflow_score > result.workflow_score:
                    results[-1] = result2

        all_results[prompt_id] = results

    return all_results


@pytest.mark.skipif(
    not _should_run(),
    reason="Requires RUN_PLATYNUI_E2E=true",
)
class TestPlatynUIDesktopSetup:
    """Tests that verify AI models can set up PlatynUI desktop sessions."""

    def test_at_least_half_models_call_analyze_scenario(
        self, platynui_results: dict[str, list[ModelResult]]
    ):
        results = platynui_results.get("setup", [])
        working = [r for r in results if not r.error]
        using = sum(1 for r in working if r.used_analyze_scenario)
        assert len(working) > 0, "No models completed successfully"
        assert using / len(working) >= 0.5, (
            f"Only {using}/{len(working)} models called analyze_scenario"
        )

    def test_models_pass_desktop_context(
        self, platynui_results: dict[str, list[ModelResult]]
    ):
        """Models should pass context='desktop' to analyze_scenario."""
        results = platynui_results.get("setup", [])
        working = [r for r in results if not r.error and r.used_analyze_scenario]
        passing_desktop = sum(1 for r in working if r.passed_context_desktop)
        if working:
            print(f"\n  Desktop context pass rate: {passing_desktop}/{len(working)}")
        # At least some models should pass context="desktop"
        assert passing_desktop > 0, (
            f"No models passed context='desktop' to analyze_scenario"
        )

    def test_models_include_platynui_library(
        self, platynui_results: dict[str, list[ModelResult]]
    ):
        """Models should include PlatynUI.BareMetal in libraries."""
        results = platynui_results.get("setup", [])
        working = [r for r in results if not r.error and r.used_manage_session]
        including = sum(1 for r in working if r.passed_platynui_library)
        if working:
            print(f"\n  PlatynUI library inclusion rate: {including}/{len(working)}")
        assert including > 0, "No models included PlatynUI in libraries"

    def test_session_type_detection(
        self, platynui_results: dict[str, list[ModelResult]]
    ):
        """Check what session_type analyze_scenario returns for desktop context."""
        results = platynui_results.get("setup", [])
        session_types: dict[str, int] = {}
        for r in results:
            st = r.session_type_from_analyze
            if st:
                session_types[st] = session_types.get(st, 0) + 1
        print(f"\n  Session types detected: {session_types}")
        # Document the distribution - this is informational


@pytest.mark.skipif(
    not _should_run(),
    reason="Requires RUN_PLATYNUI_E2E=true",
)
class TestPlatynUIDesktopExecution:
    """Tests that verify AI models can execute PlatynUI desktop steps."""

    def test_at_least_half_models_execute_steps(
        self, platynui_results: dict[str, list[ModelResult]]
    ):
        results = platynui_results.get("execute", [])
        working = [r for r in results if not r.error]
        using = sum(1 for r in working if r.used_execute_step)
        assert len(working) > 0, "No models completed successfully"
        assert using / len(working) >= 0.5, (
            f"Only {using}/{len(working)} models used execute_step"
        )

    def test_platynui_keywords_attempted(
        self, platynui_results: dict[str, list[ModelResult]]
    ):
        """Check which PlatynUI keywords models attempted."""
        results = platynui_results.get("execute", [])
        all_keywords: dict[str, int] = {}
        for r in results:
            for kw in r.execute_step_keywords:
                all_keywords[kw] = all_keywords.get(kw, 0) + 1
        print(f"\n  Keywords attempted across all models: {all_keywords}")


@pytest.mark.skipif(
    not _should_run(),
    reason="Requires RUN_PLATYNUI_E2E=true",
)
class TestPlatynUICalculatorWorkflow:
    """Tests for full PlatynUI calculator workflow."""

    def test_at_least_one_model_completes_workflow(
        self, platynui_results: dict[str, list[ModelResult]]
    ):
        results = platynui_results.get("calculator", [])
        working = [r for r in results if not r.error]
        completing = [r for r in working if r.workflow_score >= 4]
        assert len(completing) > 0, (
            f"No models scored >= 4/5 on calculator workflow. "
            f"Scores: {[(r.model.split('/')[-1], r.workflow_score) for r in working]}"
        )


@pytest.mark.skipif(
    not _should_run(),
    reason="Requires RUN_PLATYNUI_E2E=true",
)
class TestModelComparison:
    """Compare model performance on PlatynUI desktop tasks."""

    def test_print_comparison_table(
        self, platynui_results: dict[str, list[ModelResult]]
    ):
        """Print a comparison table of all model results."""
        print("\n\n" + "=" * 110)
        print("PLATYNUI DESKTOP E2E MODEL COMPARISON")
        print("=" * 110)

        for prompt_id, results in platynui_results.items():
            prompt_name = PLATYNUI_PROMPTS.get(prompt_id, {}).get("name", prompt_id)
            print(f"\n{'─' * 110}")
            print(f"Prompt: {prompt_name}")
            print(f"{'─' * 110}")
            print(
                f"  {'Model':<25} {'Score':<7} {'Calls':<7} {'a_s':<5} "
                f"{'ctx':<5} {'m_s':<5} {'lib':<5} {'e_s':<5} {'b_s':<5} "
                f"{'s_type':<18} {'Time':<8} {'Cost':<8}"
            )
            print(f"  {'─' * 108}")

            for r in results:
                model_short = r.model.split("/")[-1][:23]
                if r.error:
                    print(f"  {model_short:<25} {'ERR':<7} {'-':<7} "
                          f"{'-':<5} {'-':<5} {'-':<5} {'-':<5} {'-':<5} {'-':<5} "
                          f"{'-':<18} {'-':<8} {'-':<8}  {r.error[:40]}")
                    continue
                st = r.session_type_from_analyze or "-"
                print(
                    f"  {model_short:<25} {r.workflow_score}/5{'':<3} "
                    f"{len(r.tool_calls):<7} "
                    f"{'Y' if r.used_analyze_scenario else '-':<5} "
                    f"{'Y' if r.passed_context_desktop else '-':<5} "
                    f"{'Y' if r.used_manage_session else '-':<5} "
                    f"{'Y' if r.passed_platynui_library else '-':<5} "
                    f"{'Y' if r.used_execute_step else '-':<5} "
                    f"{'Y' if r.used_build_test_suite else '-':<5} "
                    f"{st:<18} "
                    f"{r.duration_seconds:.1f}s{'':<3} "
                    f"${r.cost:.4f}"
                )

        # Summary
        all_results = [r for results in platynui_results.values() for r in results]
        working = [r for r in all_results if not r.error]
        print(f"\n{'=' * 110}")
        print("OVERALL SUMMARY")
        print(f"{'=' * 110}")
        if working:
            avg_score = sum(r.workflow_score for r in working) / len(working)
            avg_time = sum(r.duration_seconds for r in working) / len(working)
            total_cost = sum(r.cost for r in working)
            desktop_ctx = sum(1 for r in working if r.passed_context_desktop)
            platynui_lib = sum(1 for r in working if r.passed_platynui_library)

            print(f"  Working runs:        {len(working)}/{len(all_results)}")
            print(f"  Avg workflow score:  {avg_score:.1f}/5")
            print(f"  Desktop context:     {desktop_ctx}/{len(working)} ({desktop_ctx/len(working):.0%})")
            print(f"  PlatynUI in libs:    {platynui_lib}/{len(working)} ({platynui_lib/len(working):.0%})")
            print(f"  Avg time:            {avg_time:.1f}s")
            print(f"  Total cost:          ${total_cost:.4f}")

            # Session type distribution
            st_counts: dict[str, int] = {}
            for r in working:
                st = r.session_type_from_analyze or "unknown"
                st_counts[st] = st_counts.get(st, 0) + 1
            print(f"  Session types:       {st_counts}")

            # Keywords attempted
            kw_counts: dict[str, int] = {}
            for r in working:
                for kw in r.execute_step_keywords:
                    kw_counts[kw] = kw_counts.get(kw, 0) + 1
            if kw_counts:
                print(f"  Keywords attempted:  {kw_counts}")

        print(f"{'=' * 110}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    """Run PlatynUI desktop E2E tests standalone."""
    print("=" * 100)
    print("PLATYNUI DESKTOP E2E TESTS — MULTI-MODEL")
    print("=" * 100)
    print(f"Models: {len(MODELS)}")
    for m in MODELS:
        print(f"  - {m}")
    print(f"Prompts: {len(PLATYNUI_PROMPTS)}")
    print()

    all_results: dict[str, list[ModelResult]] = {}
    all_raw: list[dict[str, Any]] = []
    start_time = time.time()

    for prompt_id, prompt_info in PLATYNUI_PROMPTS.items():
        results = []
        print(f"\n{'#' * 100}")
        print(f"# Prompt: {prompt_info['name']} ({prompt_id})")
        print(f"{'#' * 100}")

        for i, model in enumerate(MODELS, 1):
            model_short = model.split("/")[-1]
            print(f"\n  [{i}/{len(MODELS)}] {model_short} — {prompt_info['name']}")
            print(f"  {'─' * 60}")

            result = _run_model(model, prompt_info["prompt"], timeout_seconds=240)
            result.prompt_id = prompt_id
            _print_result(result)
            results.append(result)

            # Retry with short prompt if poor score
            if result.workflow_score < 2 and not result.error and prompt_id != "calculator":
                print(f"  Retrying with short prompt...")
                result2 = _run_model(model, PLATYNUI_SHORT_PROMPT, timeout_seconds=120)
                result2.prompt_id = f"{prompt_id}_short"
                _print_result(result2)
                if result2.workflow_score > result.workflow_score:
                    results[-1] = result2

            all_raw.append({
                "model": model,
                "prompt_id": prompt_id,
                "prompt_name": prompt_info["name"],
                "workflow_score": result.workflow_score,
                "total_calls": len(result.tool_calls),
                "tool_counts": result.tool_counts,
                "used_analyze_scenario": result.used_analyze_scenario,
                "passed_context_desktop": result.passed_context_desktop,
                "used_manage_session": result.used_manage_session,
                "passed_platynui_library": result.passed_platynui_library,
                "used_execute_step": result.used_execute_step,
                "used_build_test_suite": result.used_build_test_suite,
                "session_type": result.session_type_from_analyze,
                "execute_keywords": result.execute_step_keywords,
                "duration_seconds": result.duration_seconds,
                "total_tokens": result.total_tokens,
                "cost": result.cost,
                "error": result.error,
            })

        all_results[prompt_id] = results

    total_time = time.time() - start_time

    # Print comparison
    all_flat = [r for results in all_results.values() for r in results]
    working = [r for r in all_flat if not r.error]

    print("\n\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    if working:
        avg_score = sum(r.workflow_score for r in working) / len(working)
        print(f"  Runs: {len(working)}/{len(all_flat)}")
        print(f"  Avg score: {avg_score:.1f}/5")
        print(f"  Total time: {total_time:.0f}s")
        print(f"  Total cost: ${sum(r.cost for r in working):.4f}")

    # Save results
    metrics_dir = Path(__file__).parent / "metrics" / "platynui_desktop"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = metrics_dir / f"platynui_desktop_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_time_seconds": total_time,
            "models": MODELS,
            "prompts": list(PLATYNUI_PROMPTS.keys()),
            "results": all_raw,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
