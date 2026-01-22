"""Tests for MCP Services Adapter."""

import pytest
from robotmcp.lib.mcp_adapter import (
    MCPServicesAdapter,
    KeywordInfo,
    ExecutedStep,
    get_mcp_adapter,
)


class TestKeywordInfo:
    """Tests for KeywordInfo dataclass."""

    def test_basic_keyword_info(self):
        """Test creating basic keyword info."""
        kw = KeywordInfo(name="Click", library="Browser")
        assert kw.name == "Click"
        assert kw.library == "Browser"
        assert kw.args == []
        assert kw.short_doc == ""

    def test_keyword_info_with_args(self):
        """Test keyword info with arguments."""
        kw = KeywordInfo(
            name="Fill Text",
            library="Browser",
            args=["selector", "text"],
            arg_types=["str", "str"],
            short_doc="Fill text into input element",
        )
        assert kw.name == "Fill Text"
        assert kw.args == ["selector", "text"]
        assert "Fill text" in kw.short_doc

    def test_to_context_string(self):
        """Test formatting keyword for context."""
        kw = KeywordInfo(
            name="Click",
            library="Browser",
            args=["selector"],
            short_doc="Click on element",
        )
        context_str = kw.to_context_string()
        assert "Click" in context_str
        assert "selector" in context_str
        assert "Click on element" in context_str


class TestExecutedStep:
    """Tests for ExecutedStep dataclass."""

    def test_successful_step(self):
        """Test recording a successful step."""
        step = ExecutedStep(
            keyword="Click",
            args=["button#submit"],
            success=True,
            result=None,
        )
        assert step.keyword == "Click"
        assert step.success is True
        assert step.error is None

    def test_failed_step(self):
        """Test recording a failed step."""
        step = ExecutedStep(
            keyword="Click",
            args=["button#missing"],
            success=False,
            error="Element not found",
        )
        assert step.success is False
        assert "not found" in step.error


class TestMCPServicesAdapter:
    """Tests for MCPServicesAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create fresh adapter for each test."""
        return MCPServicesAdapter()

    def test_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter is not None
        assert hasattr(adapter, "mcp_available")

    def test_record_step(self, adapter):
        """Test recording execution steps."""
        adapter.record_step(
            keyword="Click",
            args=["button#submit"],
            success=True,
        )
        history = adapter.get_execution_history(limit=5)
        assert len(history) == 1
        assert history[0]["keyword"] == "Click"
        assert history[0]["success"] is True

    def test_record_multiple_steps(self, adapter):
        """Test recording multiple steps."""
        adapter.record_step("Open Browser", ["https://example.com"], True)
        adapter.record_step("Click", ["button#login"], True)
        adapter.record_step("Fill Text", ["#username", "test"], True)

        history = adapter.get_execution_history(limit=10)
        assert len(history) == 3
        assert history[0]["keyword"] == "Open Browser"
        assert history[2]["keyword"] == "Fill Text"

    def test_history_limit(self, adapter):
        """Test execution history respects limit."""
        # Record 25 steps (max is 20)
        for i in range(25):
            adapter.record_step(f"Step{i}", [f"arg{i}"], True)

        history = adapter.get_execution_history(limit=100)
        assert len(history) == 20  # Should only keep last 20

    def test_format_history_for_context(self, adapter):
        """Test formatting history for AI context."""
        adapter.record_step("Click", ["#btn"], True)
        adapter.record_step("Fill Text", ["#input", "value"], True)

        formatted = adapter.format_history_for_context(limit=5)
        assert "Previously executed steps:" in formatted
        assert "Click" in formatted
        assert "PASS" in formatted

    def test_format_history_empty(self, adapter):
        """Test formatting empty history."""
        formatted = adapter.format_history_for_context(limit=5)
        assert formatted == ""

    def test_clear_history(self, adapter):
        """Test clearing execution history."""
        adapter.record_step("Click", ["#btn"], True)
        adapter.clear_history()

        history = adapter.get_execution_history(limit=5)
        assert len(history) == 0

    def test_get_rich_context(self, adapter):
        """Test getting rich context for AI."""
        adapter.record_step("Click", ["#btn"], True)

        context = adapter.get_rich_context(
            page_state={"url": "https://example.com", "title": "Example"},
            include_keywords=False,  # Skip keyword discovery in test
            include_history=True,
            history_limit=5,
        )

        assert "url" in context
        assert context["url"] == "https://example.com"
        assert "execution_history" in context


class TestGetMCPAdapter:
    """Tests for singleton adapter."""

    def test_singleton_instance(self):
        """Test that get_mcp_adapter returns singleton."""
        adapter1 = get_mcp_adapter()
        adapter2 = get_mcp_adapter()
        assert adapter1 is adapter2

    def test_mcp_available_property(self):
        """Test mcp_available property."""
        adapter = get_mcp_adapter()
        # MCP components should be available when imported within rf-mcp
        assert isinstance(adapter.mcp_available, bool)
