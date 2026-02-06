"""Tests for agent module."""

import pytest

from robotmcp.lib.agent import parse_keyword_call


class TestParseKeywordCall:
    """Tests for parse_keyword_call function."""

    def test_simple_keyword(self):
        """Test parsing simple keyword with no args."""
        keyword, args = parse_keyword_call("Get Title")

        assert keyword == "Get Title"
        assert args == []

    def test_keyword_with_one_arg(self):
        """Test parsing keyword with one argument."""
        keyword, args = parse_keyword_call("Click    button#submit")

        assert keyword == "Click"
        assert args == ["button#submit"]

    def test_keyword_with_multiple_args(self):
        """Test parsing keyword with multiple arguments."""
        keyword, args = parse_keyword_call("Fill Text    id=username    testuser")

        assert keyword == "Fill Text"
        assert args == ["id=username", "testuser"]

    def test_keyword_with_tab_separator(self):
        """Test parsing keyword with tab separators."""
        keyword, args = parse_keyword_call("Click\tbutton")

        assert keyword == "Click"
        assert args == ["button"]

    def test_empty_call(self):
        """Test parsing empty keyword call."""
        keyword, args = parse_keyword_call("")

        assert keyword == ""
        assert args == []

    def test_keyword_only_spaces(self):
        """Test parsing keyword with spaces in name."""
        keyword, args = parse_keyword_call("Should Be Equal    value1    value2")

        assert keyword == "Should Be Equal"
        assert args == ["value1", "value2"]

    def test_keyword_with_special_chars(self):
        """Test parsing keyword with special characters in args."""
        keyword, args = parse_keyword_call("Fill Text    input[type='email']    test@example.com")

        assert keyword == "Fill Text"
        assert len(args) == 2
        assert "input[type='email']" in args[0]
        assert "test@example.com" in args[1]

    def test_keyword_with_variable(self):
        """Test parsing keyword with RF variable."""
        keyword, args = parse_keyword_call("Log    ${MESSAGE}")

        assert keyword == "Log"
        assert args == ["${MESSAGE}"]

    def test_keyword_with_named_arg(self):
        """Test parsing keyword with named argument."""
        keyword, args = parse_keyword_call("New Browser    chromium    headless=false")

        assert keyword == "New Browser"
        assert "chromium" in args
        assert "headless=false" in args
