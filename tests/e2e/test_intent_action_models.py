"""E2E tests for intent_action tool usage across multiple small LLM models.

Runs opencode with each model and verifies that the AI correctly discovers
and uses the intent_action MCP tool for web automation tasks.
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

# Models to test (openrouter provider format)
# Override via OPENCODE_MODELS env var (comma-separated), e.g.:
#   OPENCODE_MODELS="openrouter/qwen/qwen3-coder,openrouter/z-ai/glm-4.5-air"
_DEFAULT_MODELS = [
    "openrouter/z-ai/glm-4.7",
    "openrouter/z-ai/glm-4.5-air",
    "openrouter/openai/gpt-oss-20b",
    "openrouter/qwen/qwen3-coder",
    "openrouter/meta-llama/llama-4-scout",
    "openrouter/z-ai/glm-4.7-flash",
]

_env_models = os.getenv("OPENCODE_MODELS", "").strip()
MODELS = [m.strip() for m in _env_models.split(",") if m.strip()] if _env_models else _DEFAULT_MODELS

# Prompt designed to elicit intent_action usage
INTENT_ACTION_PROMPT = """\
You have access to Robot Framework MCP tools. Your task:

1. First call analyze_scenario with scenario="Navigate to example.com and click the login button" and context="web"
2. Then call manage_session with action="init" using the session_id from step 1, with libraries=["Browser","BuiltIn"]
3. Now use the intent_action tool (NOT execute_step) to perform these web actions:
   a. intent_action with intent="navigate", target="https://example.com"
   b. intent_action with intent="click", target="text=Login"
   c. intent_action with intent="fill", target="#username", value="testuser"

IMPORTANT: You MUST use the intent_action tool for steps 3a-3c. Do NOT use execute_step.
The intent_action tool accepts: intent (string), target (string), value (string), session_id (string).
Pass the session_id from step 1 to all intent_action calls.

It's OK if the browser actions fail (no real browser) - I just need to verify you called intent_action correctly.
"""

# Shorter prompt for models that struggle with long instructions
SHORT_INTENT_PROMPT = """\
Call these MCP tools in order:
1. analyze_scenario(scenario="Web login test", context="web")
2. manage_session(action="init", session_id=<from step 1>, libraries=["Browser","BuiltIn"])
3. intent_action(intent="navigate", target="https://example.com", session_id=<from step 1>)
4. intent_action(intent="click", target="text=Login", session_id=<from step 1>)

Use intent_action, NOT execute_step for steps 3-4. Failures are expected (no browser).
"""


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
    tool_calls: list[ToolCall] = field(default_factory=list)
    text_output: str = ""
    error: str | None = None
    duration_seconds: float = 0.0
    total_tokens: int = 0
    cost: float = 0.0

    @property
    def used_intent_action(self) -> bool:
        return any(tc.name == "robotmcp_intent_action" for tc in self.tool_calls)

    @property
    def intent_action_calls(self) -> list[ToolCall]:
        return [tc for tc in self.tool_calls if tc.name == "robotmcp_intent_action"]

    @property
    def intent_action_intents(self) -> list[str]:
        return [tc.args.get("intent", "") for tc in self.intent_action_calls]

    @property
    def used_analyze_scenario(self) -> bool:
        return any(tc.name == "robotmcp_analyze_scenario" for tc in self.tool_calls)

    @property
    def used_manage_session(self) -> bool:
        return any(tc.name == "robotmcp_manage_session" for tc in self.tool_calls)

    @property
    def passed_session_id(self) -> bool:
        """Check if session_id was passed to intent_action calls."""
        for tc in self.intent_action_calls:
            if tc.args.get("session_id"):
                return True
        return False


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
            # opencode JSON format: type="tool_use", part.type="tool"
            part = event.get("part", {})
            state = part.get("state", {})
            tc = ToolCall(
                name=part.get("tool", ""),
                args=state.get("input", {}),
                success=state.get("status") == "completed",
            )
            # Extract result from state.output if available
            output_data = state.get("output")
            if output_data is not None:
                tc.result = str(output_data)[:500] if output_data else ""
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

    cmd = [
        "opencode", "run",
        "--format", "json",
        "-m", model,
        "--dir", str(Path(__file__).parent.parent.parent),
        prompt,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(Path(__file__).parent.parent.parent),
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
    status = "PASS" if result.used_intent_action else "FAIL"

    print(f"\n{'=' * 70}")
    print(f"Model: {model_short} [{status}]")
    print(f"{'=' * 70}")

    if result.error:
        print(f"  ERROR: {result.error}")
        return

    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Tokens: {result.total_tokens}")
    print(f"  Cost: ${result.cost:.4f}")
    print(f"  Total tool calls: {len(result.tool_calls)}")

    # Tool call summary
    tool_counts: dict[str, int] = {}
    for tc in result.tool_calls:
        tool_counts[tc.name] = tool_counts.get(tc.name, 0) + 1
    print(f"  Tools used: {dict(tool_counts)}")

    # Intent action details
    if result.used_intent_action:
        print(f"  intent_action calls: {len(result.intent_action_calls)}")
        print(f"  Intents used: {result.intent_action_intents}")
        print(f"  Session ID passed: {result.passed_session_id}")
    else:
        print(f"  intent_action: NOT USED")

    # Workflow steps
    print(f"  analyze_scenario: {'YES' if result.used_analyze_scenario else 'NO'}")
    print(f"  manage_session: {'YES' if result.used_manage_session else 'NO'}")


def _should_run() -> bool:
    """Check if intent_action E2E tests should run."""
    return os.getenv("RUN_INTENT_E2E", "false").lower() in ("true", "1", "yes")


@pytest.fixture(scope="module")
def intent_results() -> list[ModelResult]:
    """Run all models and cache results for the module."""
    results = []
    for model in MODELS:
        print(f"\n>>> Running model: {model}")
        result = _run_model(model, INTENT_ACTION_PROMPT, timeout_seconds=180)
        _print_result(result)
        results.append(result)

        # If model didn't use intent_action, try shorter prompt
        if not result.used_intent_action and not result.error:
            print(f"  Retrying with shorter prompt...")
            result2 = _run_model(model, SHORT_INTENT_PROMPT, timeout_seconds=120)
            _print_result(result2)
            # Use the better result
            if result2.used_intent_action:
                results[-1] = result2

    return results


@pytest.mark.skipif(
    not _should_run(),
    reason="Requires RUN_INTENT_E2E=true",
)
class TestIntentActionDiscovery:
    """Tests that verify AI models discover and use intent_action."""

    def test_at_least_half_models_use_intent_action(self, intent_results: list[ModelResult]):
        """At least half the models should discover and use intent_action."""
        using_intent = sum(1 for r in intent_results if r.used_intent_action)
        total_valid = sum(1 for r in intent_results if not r.error)

        print(f"\n\nSUMMARY: {using_intent}/{total_valid} models used intent_action")

        # At least 50% of working models should use intent_action
        assert total_valid > 0, "No models completed successfully"
        rate = using_intent / total_valid
        assert rate >= 0.5, (
            f"Only {using_intent}/{total_valid} ({rate:.0%}) models used intent_action. "
            f"Expected >= 50%"
        )

    def test_intent_action_has_valid_intents(self, intent_results: list[ModelResult]):
        """Models that use intent_action should pass valid intent verbs."""
        valid_intents = {"navigate", "click", "fill", "hover", "select",
                         "assert_visible", "extract_text", "wait_for"}

        for result in intent_results:
            if not result.used_intent_action:
                continue
            for intent in result.intent_action_intents:
                assert intent in valid_intents, (
                    f"Model {result.model} used invalid intent '{intent}'. "
                    f"Valid: {valid_intents}"
                )

    def test_intent_action_has_target(self, intent_results: list[ModelResult]):
        """Models using intent_action should provide target parameter."""
        for result in intent_results:
            for tc in result.intent_action_calls:
                assert tc.args.get("target"), (
                    f"Model {result.model} called intent_action without target. "
                    f"Args: {tc.args}"
                )

    def test_session_id_propagation(self, intent_results: list[ModelResult]):
        """Models should pass session_id to intent_action calls."""
        for result in intent_results:
            if not result.used_intent_action:
                continue
            assert result.passed_session_id, (
                f"Model {result.model} used intent_action but didn't pass session_id. "
                f"Calls: {[(tc.name, tc.args) for tc in result.intent_action_calls]}"
            )

    def test_navigate_intent_used(self, intent_results: list[ModelResult]):
        """Models using intent_action should use 'navigate' intent."""
        for result in intent_results:
            if not result.used_intent_action:
                continue
            assert "navigate" in result.intent_action_intents, (
                f"Model {result.model} used intent_action but not 'navigate'. "
                f"Intents: {result.intent_action_intents}"
            )

    def test_click_intent_used(self, intent_results: list[ModelResult]):
        """Models using intent_action should use 'click' intent."""
        for result in intent_results:
            if not result.used_intent_action:
                continue
            assert "click" in result.intent_action_intents, (
                f"Model {result.model} used intent_action but not 'click'. "
                f"Intents: {result.intent_action_intents}"
            )


@pytest.mark.skipif(
    not _should_run(),
    reason="Requires RUN_INTENT_E2E=true",
)
class TestModelComparison:
    """Compare model performance on intent_action tasks."""

    def test_print_comparison_table(self, intent_results: list[ModelResult]):
        """Print a comparison table of all model results."""
        print("\n\n" + "=" * 90)
        print("INTENT_ACTION E2E MODEL COMPARISON")
        print("=" * 90)
        print(f"{'Model':<30} {'Status':<8} {'intent_action':<15} "
              f"{'Intents':<25} {'Time':<8} {'Cost':<8}")
        print("-" * 90)

        for r in intent_results:
            model_short = r.model.split("/")[-1]
            status = "ERROR" if r.error else ("PASS" if r.used_intent_action else "FAIL")
            ia_count = str(len(r.intent_action_calls)) if not r.error else "-"
            intents = ",".join(r.intent_action_intents) if not r.error else "-"
            time_s = f"{r.duration_seconds:.1f}s" if not r.error else "-"
            cost = f"${r.cost:.4f}" if not r.error else "-"
            print(f"  {model_short:<28} {status:<8} {ia_count:<15} "
                  f"{intents:<25} {time_s:<8} {cost:<8}")

        # Summary
        working = [r for r in intent_results if not r.error]
        using_ia = [r for r in working if r.used_intent_action]
        print(f"\n  Models tested: {len(intent_results)}")
        print(f"  Models working: {len(working)}")
        print(f"  Models using intent_action: {len(using_ia)}")
        if working:
            print(f"  Discovery rate: {len(using_ia)/len(working):.0%}")
        print("=" * 90)
