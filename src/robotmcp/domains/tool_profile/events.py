"""Tool Profile Domain Events.

This module contains domain events for the Tool Profile bounded context.
Domain events represent something that happened in the domain that
domain experts care about.

Following the event pattern from domains/instruction/events.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, FrozenSet, Optional


@dataclass
class ProfileActivated:
    """Emitted when a tool profile is activated, changing visible tools.

    Subscribers: ToolManagerAdapter (to apply add/remove), InstructionResolver
    (to switch instruction template if model tier changed).

    Attributes:
        profile_name: Name of the activated profile.
        model_tier: The model tier string value.
        description_mode: The description mode string value.
        tool_names: Set of tool names in the activated profile.
        tool_count: Number of tools in the profile.
        estimated_tokens: Estimated total token cost.
        trigger: What triggered the activation.
        session_id: Optional session context.
        timestamp: When the activation occurred.
    """
    profile_name: str
    model_tier: str
    description_mode: str
    tool_names: FrozenSet[str]
    tool_count: int
    estimated_tokens: int
    trigger: str
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": "ProfileActivated",
            "profile_name": self.profile_name,
            "model_tier": self.model_tier,
            "description_mode": self.description_mode,
            "tool_count": self.tool_count,
            "estimated_tokens": self.estimated_tokens,
            "trigger": self.trigger,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProfileTransitioned:
    """Emitted when the active profile transitions to another.

    Captures the full transition context for debugging and learning.

    Attributes:
        from_profile: Name of the profile being left.
        to_profile: Name of the profile being entered.
        trigger: What triggered the transition.
        tools_added: Tools that became visible.
        tools_removed: Tools that became hidden.
        session_id: Optional session context.
        timestamp: When the transition occurred.
    """
    from_profile: str
    to_profile: str
    trigger: str
    tools_added: FrozenSet[str]
    tools_removed: FrozenSet[str]
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def net_tool_change(self) -> int:
        """Net change in tool count (positive = more tools visible)."""
        return len(self.tools_added) - len(self.tools_removed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": "ProfileTransitioned",
            "from_profile": self.from_profile,
            "to_profile": self.to_profile,
            "trigger": self.trigger,
            "tools_added": sorted(self.tools_added),
            "tools_removed": sorted(self.tools_removed),
            "net_tool_change": self.net_tool_change,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToolsHidden:
    """Emitted when tools are removed from the visible set.

    Used by learning/optimization subsystem to track which tools
    are most frequently hidden and whether this causes escalation.

    Attributes:
        tool_names: Names of the tools that were hidden.
        reason: Why the tools were hidden (profile_switch, budget_exceeded, user_request).
        profile_name: The profile that caused the hiding.
        timestamp: When the tools were hidden.
    """
    tool_names: FrozenSet[str]
    reason: str
    profile_name: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": "ToolsHidden",
            "tool_names": sorted(self.tool_names),
            "reason": self.reason,
            "profile_name": self.profile_name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToolsRevealed:
    """Emitted when previously hidden tools become visible.

    Paired with ToolsHidden for escalation tracking.

    Attributes:
        tool_names: Names of the tools that were revealed.
        reason: Why the tools were revealed (escalation, profile_switch, user_request).
        profile_name: The profile that caused the revealing.
        timestamp: When the tools were revealed.
    """
    tool_names: FrozenSet[str]
    reason: str
    profile_name: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": "ToolsRevealed",
            "tool_names": sorted(self.tool_names),
            "reason": self.reason,
            "profile_name": self.profile_name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TokenBudgetExceeded:
    """Emitted when a profile activation would exceed its token budget.

    This is a warning event. The profile manager must decide whether
    to reject the activation, trim tools, or switch description mode.

    Attributes:
        profile_name: The profile that exceeded the budget.
        budget_limit: The maximum allowed token budget.
        actual_cost: The actual computed token cost.
        overage: How many tokens over budget.
        suggested_action: Recommended corrective action.
        timestamp: When the budget was exceeded.
    """
    profile_name: str
    budget_limit: int
    actual_cost: int
    overage: int
    suggested_action: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": "TokenBudgetExceeded",
            "profile_name": self.profile_name,
            "budget_limit": self.budget_limit,
            "actual_cost": self.actual_cost,
            "overage": self.overage,
            "suggested_action": self.suggested_action,
            "timestamp": self.timestamp.isoformat(),
        }
