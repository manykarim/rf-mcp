"""Models for E2E AI agent testing scenarios."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class ExpectedToolCall(BaseModel):
    """Expected tool call in a scenario."""

    tool_name: str = Field(..., description="Name of the MCP tool expected to be called")
    min_calls: int = Field(1, description="Minimum number of times this tool should be called")
    max_calls: Optional[int] = Field(None, description="Maximum number of times this tool should be called")
    required_params: Optional[Dict[str, Any]] = Field(None, description="Required parameters in the tool call")


class Scenario(BaseModel):
    """E2E test scenario definition."""

    id: str = Field(..., description="Unique scenario identifier")
    name: str = Field(..., description="Human-readable scenario name")
    description: str = Field(..., description="Detailed scenario description")
    context: Literal["web", "mobile", "api", "desktop", "generic"] = Field(
        "web",
        description="Test context/domain"
    )
    prompt: str = Field(..., description="Natural language prompt given to the AI agent")
    expected_tools: List[ExpectedToolCall] = Field(
        default_factory=list,
        description="Expected tool calls and their frequencies"
    )
    expected_outcome: str = Field(..., description="Expected outcome description")
    min_tool_hit_rate: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable tool hit rate (0.0-1.0)"
    )
    tags: List[str] = Field(default_factory=list, description="Scenario tags for filtering")


class ToolCallRecord(BaseModel):
    """Record of a single tool call made by the AI agent."""

    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float


class ScenarioResult(BaseModel):
    """Result of executing a test scenario."""

    scenario_id: str
    success: bool
    tool_calls: List[ToolCallRecord]
    tool_hit_rate: float = Field(
        description="Percentage of expected tools that were called correctly"
    )
    total_tool_calls: int
    expected_tool_calls_met: int
    expected_tool_calls_total: int
    errors: List[str] = Field(default_factory=list)
    execution_time_seconds: float
    agent_output: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
