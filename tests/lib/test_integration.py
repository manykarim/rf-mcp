"""Integration tests for AILibrary with rf-mcp components.

These tests verify that AILibrary integrates properly with the existing
rf-mcp architecture and components.
"""

import os
import tempfile
from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.lib.agent import RFAgent, parse_keyword_call
from robotmcp.lib.context import RFContextBridge, ROBOT_AVAILABLE
from robotmcp.lib.exporter import TestSuiteExporter
from robotmcp.lib.providers import ProviderConfig
from robotmcp.lib.recorder import ExecutionRecorder, RecordedStep, StepType
from robotmcp.lib.retry import RetryHandler, ExecutionResult, RetryContext


class TestProviderIntegration:
    """Test provider configuration with different AI backends."""

    def test_anthropic_provider_config(self):
        """Test Anthropic provider configuration."""
        config = ProviderConfig.from_kwargs(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            max_tokens=4096,
            temperature=0.1,
        )

        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tokens == 4096

    def test_openai_provider_config(self):
        """Test OpenAI provider configuration."""
        config = ProviderConfig.from_kwargs(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4"

    def test_ollama_provider_config(self):
        """Test Ollama (local) provider configuration."""
        config = ProviderConfig.from_kwargs(
            provider="ollama",
            model="llama2",
            base_url="http://localhost:11434",
        )

        assert config.provider == "ollama"
        assert config.model == "llama2"

    def test_azure_provider_config(self):
        """Test Azure OpenAI provider configuration."""
        config = ProviderConfig.from_kwargs(
            provider="azure",
            model="gpt-4-turbo",
            api_key="test-key",
            base_url="https://myendpoint.openai.azure.com/",
        )

        assert config.provider == "azure"


class TestRecorderExporterIntegration:
    """Test recorder and exporter working together."""

    def test_record_and_export_workflow(self):
        """Test complete record -> export workflow."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        # Simulate a test session with multiple keywords
        recorder.record_step(
            keyword="New Browser",
            args=["chromium"],
            kwargs={"headless": "true"},
            library="Browser",
        )
        recorder.record_step(
            keyword="New Page",
            args=["https://example.com"],
            library="Browser",
        )

        # AI step with sub-steps
        ai_step = recorder.start_ai_step(
            prompt="Login as testuser with password secret123",
            ai_keyword="Do",
        )
        recorder.record_step(
            keyword="Fill Text",
            args=["id=username", "testuser"],
            library="Browser",
        )
        recorder.record_step(
            keyword="Fill Text",
            args=["id=password", "secret123"],
            library="Browser",
        )
        recorder.record_step(
            keyword="Click",
            args=["button#login"],
            library="Browser",
        )
        recorder.end_ai_step(success=True, attempts=1)

        recorder.stop_recording()

        # Export to all formats
        exporter = TestSuiteExporter(recorder)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export robot format
            robot_path = os.path.join(tmpdir, "test.robot")
            result_path = exporter.export(
                path=robot_path,
                suite_name="Login Test Suite",
                test_name="Test Login Flow",
                format="robot",
                include_comments=True,
            )

            with open(result_path) as f:
                content = f.read()

            # Verify robot file structure
            assert "*** Settings ***" in content
            assert "Library    Browser" in content
            assert "*** Test Cases ***" in content
            assert "Test Login Flow" in content
            assert "New Browser" in content
            assert "Fill Text" in content
            assert "Click" in content
            assert "# AI: Do" in content  # AI prompt comment

            # Export JSON format
            json_path = os.path.join(tmpdir, "test.json")
            exporter.export(
                path=json_path,
                format="json",
            )

            import json
            with open(json_path) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "steps" in data
            assert "required_libraries" in data
            assert "Browser" in data["required_libraries"]

    def test_ai_step_with_retries_recorded(self):
        """Test that AI steps with retry attempts are properly recorded."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        ai_step = recorder.start_ai_step(
            prompt="Click the submit button",
            ai_keyword="Do",
        )

        # Simulate failed attempts followed by success
        recorder.record_step(
            keyword="Click",
            args=["button.submit"],
            library="Browser",
            success=False,
        )
        recorder.record_step(
            keyword="Click",
            args=["#submit-btn"],
            library="Browser",
            success=True,
        )

        completed = recorder.end_ai_step(success=True, attempts=2)
        recorder.stop_recording()

        assert completed.attempts == 2
        assert len(completed.executed_keywords) == 2


class TestRetryMechanismIntegration:
    """Test retry mechanism integration."""

    @pytest.mark.asyncio
    async def test_retry_with_correction_prompt(self):
        """Test retry mechanism generates proper correction prompts."""
        handler = RetryHandler(max_retries=3, retry_delay=0.01)

        attempt_count = 0
        generated_prompts = []

        def execute_fn(keyword_call):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                return ExecutionResult(
                    success=False,
                    error="Element not found: button#nonexistent",
                    keyword="Click",
                    args=["button#nonexistent"],
                )
            return ExecutionResult(
                success=True,
                result="OK",
                keyword="Click",
                args=["button#submit"],
            )

        async def generate_fn(ctx):
            generated_prompts.append(ctx)
            if ctx.attempt_number == 1:
                return "Click    button#nonexistent"
            elif ctx.attempt_number == 2:
                return "Click    button#wrong"
            else:
                return "Click    button#submit"

        result = await handler.execute_with_retry(
            prompt="Click the submit button",
            execute_fn=execute_fn,
            generate_correction_fn=generate_fn,
            get_page_state_fn=lambda: {"url": "https://example.com"},
            get_keywords_fn=lambda: ["Click", "Fill Text", "Get Text"],
        )

        assert result.success is True
        assert result.attempt == 3
        # Verify error context was passed to correction function
        assert len(generated_prompts) >= 2
        assert generated_prompts[1].error_message == "Element not found: button#nonexistent"


class TestAgentKeywordParsing:
    """Test AI agent keyword parsing integration."""

    def test_parse_complex_keyword_calls(self):
        """Test parsing various keyword call formats."""
        # Simple keyword
        kw, args = parse_keyword_call("Get Title")
        assert kw == "Get Title"
        assert args == []

        # Keyword with CSS selector
        kw, args = parse_keyword_call("Click    button.primary[data-testid='submit']")
        assert kw == "Click"
        assert "button.primary[data-testid='submit']" in args[0]

        # Keyword with multiple arguments
        kw, args = parse_keyword_call("Fill Text    #email    user@example.com")
        assert kw == "Fill Text"
        assert len(args) == 2
        assert "#email" in args[0]
        assert "user@example.com" in args[1]

        # Keyword with RF variable
        kw, args = parse_keyword_call("Set Variable    ${RESULT}    Success")
        assert kw == "Set Variable"
        assert "${RESULT}" in args

        # Keyword with named argument
        kw, args = parse_keyword_call("New Browser    chromium    headless=true")
        assert kw == "New Browser"
        assert "chromium" in args
        assert "headless=true" in args


class TestContextBridgeIntegration:
    """Test RF context bridge integration."""

    def test_context_bridge_initialization(self):
        """Test context bridge can be initialized without RF context."""
        # Should not raise during initialization
        bridge = RFContextBridge()
        assert bridge._builtin is None

    def test_context_bridge_rf_available_flag(self):
        """Test ROBOT_AVAILABLE flag is set correctly."""
        # In test environment, RF should be available (we have robotframework installed)
        assert ROBOT_AVAILABLE is True

    def test_context_bridge_without_execution_context(self):
        """Test context bridge methods handle missing execution context gracefully."""
        bridge = RFContextBridge()

        # These should raise or return defaults when no RF context
        # In our implementation, they should catch the error
        try:
            result = bridge.get_variables()
            # If we get here, it should return a dict-like object (possibly empty, or with
            # RF built-in variables if there's a lingering execution context).
            # Note: RF returns NormalizedDict which is a Mapping but not a dict subclass.
            assert isinstance(result, Mapping)
        except RuntimeError:
            # Expected when no RF execution context
            pass


class TestExporterFormatIntegration:
    """Test exporter format compatibility."""

    def test_robot_format_compatibility(self):
        """Test generated .robot files are syntactically valid."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        # Record steps with special characters
        recorder.record_step(
            keyword="Fill Text",
            args=["input[name='email']", "test+special@example.com"],
            library="Browser",
        )
        recorder.record_step(
            keyword="Click",
            args=["button:has-text('Submit & Continue')"],
            library="Browser",
        )

        recorder.stop_recording()
        exporter = TestSuiteExporter(recorder)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".robot", delete=False) as f:
            path = exporter.export(path=f.name, format="robot")

            with open(path) as rf:
                content = rf.read()

            # Verify proper escaping/quoting
            assert "*** Test Cases ***" in content
            assert "Fill Text" in content
            assert "Click" in content

            os.unlink(f.name)

    def test_yaml_format_export(self):
        """Test YAML export maintains proper structure."""
        import yaml

        recorder = ExecutionRecorder()
        recorder.start_recording()

        recorder.record_step(
            keyword="Log",
            args=["Multi\nline\ntext"],
            library="BuiltIn",
        )

        recorder.stop_recording()
        exporter = TestSuiteExporter(recorder)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = exporter.export(path=f.name, format="yaml")

            with open(path) as yf:
                data = yaml.safe_load(yf)

            assert "metadata" in data
            assert "steps" in data

            os.unlink(f.name)


class TestProviderConfigYAMLIntegration:
    """Test YAML configuration loading."""

    def test_yaml_config_with_all_options(self):
        """Test loading comprehensive YAML config.

        Note: max_tokens and temperature are loaded from provider-specific
        sections, not from top-level config.
        """
        yaml_content = """
provider: anthropic
model: claude-sonnet-4-20250514
retries: 5
retry_delay: 2.0
log_level: DEBUG

anthropic:
  max_tokens: 8192
  temperature: 0.2
  thinking_mode: enabled

openai:
  organization: test-org
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ProviderConfig.from_yaml(f.name)

            assert config.provider == "anthropic"
            assert config.model == "claude-sonnet-4-20250514"
            assert config.retries == 5
            assert config.retry_delay == 2.0
            assert config.log_level == "DEBUG"
            # max_tokens and temperature come from provider-specific section
            assert config.max_tokens == 8192
            assert config.temperature == 0.2
            # extra_options should contain provider-specific settings
            assert config.extra_options.get("thinking_mode") == "enabled"

            os.unlink(f.name)


class TestEndToEndRecordingScenario:
    """Test end-to-end recording scenarios."""

    def test_mixed_regular_and_ai_steps(self):
        """Test recording with mixed regular and AI keyword steps."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        # Setup steps
        recorder.record_step(keyword="New Browser", args=["chromium"], library="Browser")
        recorder.record_step(keyword="New Page", args=["https://example.com"], library="Browser")

        # First AI step
        recorder.start_ai_step(prompt="Login as admin", ai_keyword="Do")
        recorder.record_step(keyword="Fill Text", args=["#username", "admin"], library="Browser")
        recorder.record_step(keyword="Fill Text", args=["#password", "secret"], library="Browser")
        recorder.record_step(keyword="Click", args=["#login"], library="Browser")
        recorder.end_ai_step(success=True, attempts=1)

        # Regular step between AI steps
        recorder.record_step(keyword="Wait For Load State", args=["networkidle"], library="Browser")

        # Second AI step (verify)
        recorder.start_ai_step(prompt="Check user is logged in", ai_keyword="Check")
        recorder.record_step(keyword="Get Text", args=["#welcome-message"], library="Browser")
        recorder.end_ai_step(success=True, attempts=1)

        # Third AI step (query)
        recorder.start_ai_step(prompt="What is the user's role?", ai_keyword="Ask")
        recorder.record_step(keyword="Get Text", args=["#user-role"], library="Browser")
        recorder.end_ai_step(success=True, result="Administrator", attempts=1)

        recorder.stop_recording()

        steps = recorder.get_steps()

        # Verify step count and types
        assert len(steps) == 6  # 2 regular + 3 AI + 1 wait

        # Count step types
        regular_count = sum(1 for s in steps if s.step_type == StepType.REGULAR)
        ai_count = sum(1 for s in steps if s.step_type == StepType.AI)

        assert regular_count == 3  # New Browser, New Page, Wait For Load State
        assert ai_count == 3  # Login, Check, Ask

        # Verify AI steps have sub-steps
        ai_steps = [s for s in steps if s.step_type == StepType.AI]
        for ai_step in ai_steps:
            assert len(ai_step.executed_keywords) > 0

    def test_metadata_tracking(self):
        """Test metadata is properly tracked during recording."""
        recorder = ExecutionRecorder()
        recorder.start_recording()

        recorder.record_step(keyword="Log", args=["Test"], library="BuiltIn")
        recorder.record_step(keyword="Click", args=["button"], library="Browser")
        recorder.record_step(keyword="Click Element", args=["locator"], library="SeleniumLibrary")

        metadata = recorder.get_metadata()

        assert "started_at" in metadata
        assert metadata["step_count"] == 3

        libs = recorder.get_libraries_used()
        assert "BuiltIn" in libs
        assert "Browser" in libs
        assert "SeleniumLibrary" in libs

        recorder.stop_recording()
