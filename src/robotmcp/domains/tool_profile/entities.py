"""Tool Profile Domain Entities.

This module contains entities for the Tool Profile bounded context.
Entities have identity and lifecycle, unlike value objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional

from .value_objects import SchemaMode, ToolDescriptionMode, ToolTag


@dataclass
class ToolDescriptor:
    """Per-tool metadata entity held in the tool registry.

    Stores the full, compact, and minimal description variants
    for a single MCP tool, along with its schema and tag classification.

    This entity is NOT the FastMCP Tool object itself -- it is a
    domain-side mirror of the metadata needed for profile decisions.
    The ToolManagerAdapter translates between this and FastMCP's Tool.

    Attributes:
        tool_name: The unique name of the MCP tool.
        tags: Set of semantic tags classifying this tool.
        description_full: Full description text for FULL mode.
        description_compact: Shortened description for COMPACT mode.
        description_minimal: Single-sentence description for MINIMAL mode.
        schema_full: Complete JSON schema for FULL mode.
        schema_compact: Flattened/reduced schema for COMPACT/MINIMAL modes.
        token_estimate_full: Estimated token cost in FULL mode.
        token_estimate_compact: Estimated token cost in COMPACT mode.
        token_estimate_minimal: Estimated token cost in MINIMAL mode.
    """

    tool_name: str
    tags: FrozenSet[ToolTag]
    description_full: str
    description_compact: str
    description_minimal: str
    schema_full: Dict
    schema_compact: Optional[Dict] = None     # Flattened schema variant
    token_estimate_full: int = 0
    token_estimate_compact: int = 0
    token_estimate_minimal: int = 0
    # ADR-016: Ultra-slim variants for 7B models
    description_slim: Optional[str] = None
    schema_slim: Optional[Dict] = None
    token_estimate_slim: Optional[int] = None

    def description_for_mode(self, mode: ToolDescriptionMode) -> str:
        """Return the description string for a given mode.

        Args:
            mode: The description mode to use.

        Returns:
            The appropriate description string.
        """
        if mode == ToolDescriptionMode.FULL:
            return self.description_full
        elif mode == ToolDescriptionMode.COMPACT:
            return self.description_compact
        else:
            return self.description_minimal

    def schema_for_mode(self, mode: ToolDescriptionMode) -> Dict:
        """Return the JSON schema for a given mode.

        Compact/minimal modes use schema_compact if available,
        otherwise fall back to schema_full.

        Args:
            mode: The description mode to use.

        Returns:
            The appropriate JSON schema dict.
        """
        if mode == ToolDescriptionMode.FULL:
            return self.schema_full
        return self.schema_compact if self.schema_compact else self.schema_full

    def token_estimate_for_mode(self, mode: ToolDescriptionMode) -> int:
        """Return estimated token cost for a given mode.

        Args:
            mode: The description mode to use.

        Returns:
            The estimated token cost.
        """
        if mode == ToolDescriptionMode.FULL:
            return self.token_estimate_full
        elif mode == ToolDescriptionMode.COMPACT:
            return self.token_estimate_compact
        else:
            return self.token_estimate_minimal

    def description_for_schema_mode(self, mode: SchemaMode) -> str:
        """Return the description string appropriate for a SchemaMode (ADR-016).

        FULL -> description_full
        STANDARD -> description_compact
        MINIMAL -> description_slim (falls back to description_minimal)

        Args:
            mode: The schema mode to use.

        Returns:
            The appropriate description string.
        """
        if mode == SchemaMode.FULL:
            return self.description_full
        elif mode == SchemaMode.STANDARD:
            return self.description_compact
        else:  # MINIMAL
            return self.description_slim or self.description_minimal

    def schema_for_schema_mode(self, mode: SchemaMode) -> Dict:
        """Return the JSON schema appropriate for a SchemaMode (ADR-016).

        FULL -> schema_full
        STANDARD -> schema_compact (falls back to schema_full)
        MINIMAL -> schema_slim (falls back to schema_compact, then schema_full)

        Args:
            mode: The schema mode to use.

        Returns:
            The appropriate JSON schema dict.
        """
        if mode == SchemaMode.FULL:
            return self.schema_full
        elif mode == SchemaMode.STANDARD:
            return self.schema_compact if self.schema_compact else self.schema_full
        else:  # MINIMAL
            if self.schema_slim:
                return self.schema_slim
            return self.schema_compact if self.schema_compact else self.schema_full
