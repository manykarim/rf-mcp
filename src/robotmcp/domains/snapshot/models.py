"""
Snapshot domain models for ARIA tree representation.

This module defines the core value objects and entities for the Snapshot Context,
providing a type-safe representation of accessibility trees with element references
for token-efficient page state management.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Literal, Optional


@dataclass(frozen=True)
class SnapshotId:
    """Unique identifier for a snapshot."""
    value: str

    @classmethod
    def generate(cls) -> SnapshotId:
        """Generate a new unique snapshot ID."""
        return cls(value=f"snap_{uuid.uuid4().hex[:8]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ElementRef:
    """
    Short reference to an element (e1, e2, etc.).

    These refs replace verbose locators in MCP responses, dramatically
    reducing token consumption while maintaining actionability.
    """
    value: str  # Format: "e{number}"

    @classmethod
    def from_index(cls, index: int) -> ElementRef:
        """Create a ref from a numeric index."""
        return cls(value=f"e{index}")

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ElementRef):
            return self.value == other.value
        return False


@dataclass(frozen=True)
class AriaRole:
    """
    ARIA role value object.

    Represents the semantic role of an element in the accessibility tree.
    """
    value: str

    # Common role constants
    BUTTON = "button"
    LINK = "link"
    HEADING = "heading"
    TEXTBOX = "textbox"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    LISTITEM = "listitem"
    LIST = "list"
    NAVIGATION = "navigation"
    MAIN = "main"
    DOCUMENT = "document"
    GENERIC = "generic"
    IMG = "img"
    FORM = "form"
    COMBOBOX = "combobox"
    OPTION = "option"
    TAB = "tab"
    TABLIST = "tablist"
    TABPANEL = "tabpanel"
    DIALOG = "dialog"
    ALERT = "alert"
    MENUITEM = "menuitem"
    MENU = "menu"

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AriaRole):
            return self.value == other.value
        return False

    @property
    def is_interactive(self) -> bool:
        """Check if this role represents an interactive element."""
        interactive_roles = {
            self.BUTTON, self.LINK, self.TEXTBOX, self.CHECKBOX,
            self.RADIO, self.COMBOBOX, self.OPTION, self.TAB,
            self.MENUITEM
        }
        return self.value in interactive_roles


@dataclass
class AriaNode:
    """
    Entity representing a node in the ARIA accessibility tree.

    Each node has a unique ref that can be used to identify it across
    snapshots for diff tracking and element interaction.
    """
    ref: ElementRef
    role: AriaRole
    name: Optional[str] = None
    level: Optional[int] = None
    children: List[AriaNode] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.ref)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AriaNode):
            return self.ref == other.ref
        return False

    def semantic_equals(self, other: AriaNode) -> bool:
        """
        Check if two nodes are semantically equal.

        This compares role, name, level, and properties but ignores
        the ref (which may change between snapshots).
        """
        return (
            self.role == other.role and
            self.name == other.name and
            self.level == other.level and
            self.properties == other.properties
        )

    def to_yaml_line(self, indent: int = 0) -> str:
        """Convert this node to a YAML representation line."""
        indent_str = "  " * indent
        parts = [f"- {self.role.value}"]

        if self.name:
            # Escape quotes in name
            escaped_name = self.name.replace('"', '\\"')
            parts.append(f'"{escaped_name}"')

        if self.level is not None:
            parts.append(f"[level={self.level}]")

        # Add relevant properties
        for key, value in self.properties.items():
            if key in ("checked", "selected", "disabled", "expanded", "pressed"):
                parts.append(f"[{key}={value}]")

        parts.append(f"[ref={self.ref.value}]")

        return indent_str + " ".join(parts)


@dataclass
class AriaTree:
    """
    Entity representing a complete accessibility tree.

    The tree provides traversal and lookup capabilities for efficient
    diff computation and element reference resolution.
    """
    root: AriaNode
    node_count: int = 0
    interactive_count: int = 0

    def __post_init__(self) -> None:
        """Calculate node counts after initialization."""
        if self.node_count == 0:
            self._calculate_counts()

    def _calculate_counts(self) -> None:
        """Calculate node and interactive element counts."""
        count = 0
        interactive = 0
        for node in self.traverse():
            count += 1
            if node.role.is_interactive:
                interactive += 1
        self.node_count = count
        self.interactive_count = interactive

    def traverse(self) -> Iterator[AriaNode]:
        """
        Traverse all nodes in the tree in depth-first order.

        Yields:
            Each AriaNode in the tree.
        """
        def _traverse(node: AriaNode) -> Iterator[AriaNode]:
            yield node
            for child in node.children:
                yield from _traverse(child)

        yield from _traverse(self.root)

    def find_by_ref(self, ref: ElementRef) -> Optional[AriaNode]:
        """
        Find a node by its element reference.

        Args:
            ref: The ElementRef to search for.

        Returns:
            The matching AriaNode, or None if not found.
        """
        for node in self.traverse():
            if node.ref == ref:
                return node
        return None

    def filter_interactive_only(self) -> AriaTree:
        """
        Create a new tree containing only interactive elements.

        This is useful for reducing token count when only actionable
        elements are needed.

        Returns:
            A new AriaTree with only interactive nodes.
        """
        def _filter_node(node: AriaNode) -> Optional[AriaNode]:
            # Filter children recursively
            filtered_children = []
            for child in node.children:
                filtered = _filter_node(child)
                if filtered is not None:
                    filtered_children.append(filtered)

            # Keep this node if it's interactive or has interactive descendants
            if node.role.is_interactive or filtered_children:
                return AriaNode(
                    ref=node.ref,
                    role=node.role,
                    name=node.name,
                    level=node.level,
                    children=filtered_children,
                    properties=node.properties.copy()
                )
            return None

        filtered_root = _filter_node(self.root)
        if filtered_root is None:
            # Return empty tree with document root
            filtered_root = AriaNode(
                ref=ElementRef.from_index(0),
                role=AriaRole(AriaRole.DOCUMENT),
                name="(empty)"
            )

        return AriaTree(root=filtered_root)

    def to_yaml(self, include_non_interactive: bool = True) -> str:
        """
        Convert tree to YAML string representation.

        Args:
            include_non_interactive: Whether to include non-interactive nodes.

        Returns:
            YAML string representation of the tree.
        """
        tree = self if include_non_interactive else self.filter_interactive_only()
        lines: List[str] = []

        def _render(node: AriaNode, depth: int) -> None:
            lines.append(node.to_yaml_line(depth))
            for child in node.children:
                _render(child, depth + 1)

        _render(tree.root, 0)
        return "\n".join(lines)


@dataclass(frozen=True)
class SnapshotFormat:
    """
    Format specification for snapshot output.

    Controls how snapshots are rendered and what content is included.
    """
    mode: Literal["full", "incremental", "none"] = "incremental"
    include_hidden: bool = False
    max_depth: Optional[int] = None
    fold_lists: bool = True
    fold_threshold: float = 0.85  # SimHash similarity threshold
    interactive_only: bool = False

    @classmethod
    def full(cls) -> SnapshotFormat:
        """Create a format for full snapshot output."""
        return cls(mode="full")

    @classmethod
    def incremental(cls) -> SnapshotFormat:
        """Create a format for incremental (diff) output."""
        return cls(mode="incremental")

    @classmethod
    def none(cls) -> SnapshotFormat:
        """Create a format that suppresses snapshot output."""
        return cls(mode="none")


@dataclass(frozen=True)
class CompressionStats:
    """
    Statistics about compression achieved in a snapshot.

    Tracks the effectiveness of token optimization techniques.
    """
    original_nodes: int
    compressed_nodes: int
    folded_lists: int
    token_estimate_before: int
    token_estimate_after: int

    @property
    def compression_ratio(self) -> float:
        """Calculate the compression ratio achieved."""
        if self.token_estimate_before == 0:
            return 0.0
        return 1 - (self.token_estimate_after / self.token_estimate_before)

    @property
    def tokens_saved(self) -> int:
        """Calculate the number of tokens saved."""
        return self.token_estimate_before - self.token_estimate_after

    @property
    def node_reduction(self) -> int:
        """Number of nodes removed through compression."""
        return self.original_nodes - self.compressed_nodes

    @classmethod
    def empty(cls) -> "CompressionStats":
        """Create stats representing no compression applied."""
        return cls(
            original_nodes=0,
            compressed_nodes=0,
            folded_lists=0,
            token_estimate_before=0,
            token_estimate_after=0
        )


@dataclass
class PageSnapshot:
    """
    Aggregate root for the Snapshot Context.

    Represents a complete page snapshot including the ARIA tree,
    metadata, and compression statistics.
    """
    snapshot_id: SnapshotId
    session_id: str
    aria_tree: AriaTree
    created_at: datetime = field(default_factory=datetime.utcnow)
    format: SnapshotFormat = field(default_factory=SnapshotFormat)
    compression_stats: Optional[CompressionStats] = None
    url: Optional[str] = None
    title: Optional[str] = None

    @classmethod
    def create(
        cls,
        session_id: str,
        aria_tree: AriaTree,
        url: Optional[str] = None,
        title: Optional[str] = None,
        format: Optional[SnapshotFormat] = None
    ) -> PageSnapshot:
        """
        Factory method to create a new PageSnapshot.

        Args:
            session_id: The session this snapshot belongs to.
            aria_tree: The accessibility tree.
            url: Optional page URL.
            title: Optional page title.
            format: Optional format configuration.

        Returns:
            A new PageSnapshot instance.
        """
        return cls(
            snapshot_id=SnapshotId.generate(),
            session_id=session_id,
            aria_tree=aria_tree,
            url=url,
            title=title,
            format=format or SnapshotFormat()
        )

    def to_yaml(self) -> str:
        """
        Convert snapshot to YAML representation.

        Returns:
            YAML string representation of the snapshot.
        """
        interactive_only = self.format.interactive_only
        return self.aria_tree.to_yaml(include_non_interactive=not interactive_only)

    def estimate_tokens(self) -> int:
        """
        Estimate the token count for this snapshot.

        Uses a simple heuristic: ~4 characters per token.

        Returns:
            Estimated token count.
        """
        yaml_output = self.to_yaml()
        # Approximate: 4 characters per token on average
        return len(yaml_output) // 4
