"""Unit tests for Tool Profile domain events (ADR-006).

Tests cover: ProfileActivated, ProfileTransitioned, ToolsHidden, ToolsRevealed,
TokenBudgetExceeded — construction and to_dict serialization.

Run with: uv run pytest tests/unit/test_tool_profile_events.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.tool_profile.events import (
    ProfileActivated,
    ProfileTransitioned,
    TokenBudgetExceeded,
    ToolsHidden,
    ToolsRevealed,
)


class TestProfileActivated:
    """Test ProfileActivated event."""

    def test_construction(self):
        event = ProfileActivated(
            profile_name="browser_exec",
            model_tier="small_context",
            description_mode="compact",
            tool_names=frozenset({"manage_session", "execute_step"}),
            tool_count=2,
            estimated_tokens=350,
            trigger="user_request",
            session_id="sess-1",
        )
        assert event.profile_name == "browser_exec"
        assert event.tool_count == 2

    def test_to_dict_has_event_type(self):
        event = ProfileActivated(
            profile_name="browser_exec",
            model_tier="small_context",
            description_mode="compact",
            tool_names=frozenset({"manage_session"}),
            tool_count=1,
            estimated_tokens=100,
            trigger="session_init",
        )
        d = event.to_dict()
        assert d["event_type"] == "ProfileActivated"
        assert d["profile_name"] == "browser_exec"
        assert d["model_tier"] == "small_context"
        assert d["trigger"] == "session_init"
        assert "timestamp" in d

    def test_to_dict_serializes_session_id(self):
        event = ProfileActivated(
            profile_name="full",
            model_tier="large_context",
            description_mode="full",
            tool_names=frozenset({"a"}),
            tool_count=1,
            estimated_tokens=100,
            trigger="user_request",
            session_id="abc-123",
        )
        assert event.to_dict()["session_id"] == "abc-123"

    def test_to_dict_no_session_id(self):
        event = ProfileActivated(
            profile_name="full",
            model_tier="large_context",
            description_mode="full",
            tool_names=frozenset({"a"}),
            tool_count=1,
            estimated_tokens=100,
            trigger="user_request",
        )
        assert event.to_dict()["session_id"] is None


class TestProfileTransitioned:
    """Test ProfileTransitioned event."""

    def test_construction(self):
        event = ProfileTransitioned(
            from_profile="discovery",
            to_profile="browser_exec",
            trigger="plan_to_execute",
            tools_added=frozenset({"execute_step"}),
            tools_removed=frozenset({"analyze_scenario"}),
        )
        assert event.from_profile == "discovery"
        assert event.to_profile == "browser_exec"

    def test_net_tool_change_positive(self):
        event = ProfileTransitioned(
            from_profile="minimal_exec",
            to_profile="browser_exec",
            trigger="escalation",
            tools_added=frozenset({"a", "b", "c"}),
            tools_removed=frozenset({"x"}),
        )
        assert event.net_tool_change == 2  # +3 - 1

    def test_net_tool_change_negative(self):
        event = ProfileTransitioned(
            from_profile="full",
            to_profile="browser_exec",
            trigger="de_escalation",
            tools_added=frozenset(),
            tools_removed=frozenset({"a", "b", "c"}),
        )
        assert event.net_tool_change == -3

    def test_net_tool_change_zero(self):
        event = ProfileTransitioned(
            from_profile="a",
            to_profile="b",
            trigger="user_request",
            tools_added=frozenset({"x"}),
            tools_removed=frozenset({"y"}),
        )
        assert event.net_tool_change == 0

    def test_to_dict_has_net_tool_change(self):
        event = ProfileTransitioned(
            from_profile="a",
            to_profile="b",
            trigger="user_request",
            tools_added=frozenset({"x", "y"}),
            tools_removed=frozenset({"z"}),
        )
        d = event.to_dict()
        assert d["event_type"] == "ProfileTransitioned"
        assert d["net_tool_change"] == 1
        assert "tools_added" in d
        assert "tools_removed" in d


class TestToolsHidden:
    """Test ToolsHidden event."""

    def test_construction_and_to_dict(self):
        event = ToolsHidden(
            tool_names=frozenset({"analyze_scenario", "recommend_libraries"}),
            reason="profile_switch",
            profile_name="browser_exec",
        )
        d = event.to_dict()
        assert d["event_type"] == "ToolsHidden"
        assert sorted(d["tool_names"]) == ["analyze_scenario", "recommend_libraries"]
        assert d["reason"] == "profile_switch"


class TestToolsRevealed:
    """Test ToolsRevealed event."""

    def test_construction_and_to_dict(self):
        event = ToolsRevealed(
            tool_names=frozenset({"get_keyword_info"}),
            reason="escalation",
            profile_name="browser_exec_recovery",
        )
        d = event.to_dict()
        assert d["event_type"] == "ToolsRevealed"
        assert d["tool_names"] == ["get_keyword_info"]
        assert d["reason"] == "escalation"


class TestTokenBudgetExceeded:
    """Test TokenBudgetExceeded event."""

    def test_construction_and_to_dict(self):
        event = TokenBudgetExceeded(
            profile_name="browser_exec",
            budget_limit=1638,
            actual_cost=2000,
            overage=362,
            suggested_action="switch_to_compact",
        )
        d = event.to_dict()
        assert d["event_type"] == "TokenBudgetExceeded"
        assert d["profile_name"] == "browser_exec"
        assert d["budget_limit"] == 1638
        assert d["actual_cost"] == 2000
        assert d["overage"] == 362
        assert d["suggested_action"] == "switch_to_compact"
        assert "timestamp" in d
