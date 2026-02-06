"""Snapshot domain entities - PageSnapshot and AriaTree aggregates."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from robotmcp.domains.shared.kernel import AriaNode, AriaRole, ElementRef


@dataclass(frozen=True)
class SnapshotId:
    """Unique identifier for a page snapshot."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            object.__setattr__(self, 'value', str(uuid.uuid4()))

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    @classmethod
    def generate(cls) -> "SnapshotId":
        """Generate a new unique snapshot ID."""
        return cls(str(uuid.uuid4()))


@dataclass
class AriaTree:
    """ARIA accessibility tree representation.

    Parses and represents the accessibility tree from Browser Library's
    Get Aria Snapshot output, providing structured access to elements.
    """

    root: AriaNode
    """Root node of the ARIA tree."""

    snapshot_id: SnapshotId = field(default_factory=SnapshotId.generate)
    """Unique identifier for this tree instance."""

    captured_at: datetime = field(default_factory=datetime.now)
    """Timestamp when this tree was captured."""

    source_selector: str = "css=html"
    """Selector used to capture this snapshot."""

    _ref_index: Dict[str, AriaNode] = field(default_factory=dict, repr=False)
    """Internal index for fast ref lookups."""

    def __post_init__(self) -> None:
        """Build ref index after initialization."""
        self._build_ref_index()

    def _build_ref_index(self) -> None:
        """Build index mapping refs to nodes."""
        self._ref_index.clear()
        for node in self.root.traverse():
            self._ref_index[node.ref.value] = node

    def get_node_by_ref(self, ref: ElementRef) -> Optional[AriaNode]:
        """Get a node by its element reference.

        Args:
            ref: The ElementRef to look up

        Returns:
            The matching AriaNode, or None if not found
        """
        return self._ref_index.get(ref.value)

    def get_interactive_nodes(self) -> List[AriaNode]:
        """Get all interactive nodes in the tree.

        Returns:
            List of nodes that represent interactive elements
        """
        return [node for node in self.root.traverse() if node.is_interactive]

    def get_nodes_by_role(self, role: str) -> List[AriaNode]:
        """Get all nodes with a specific role.

        Args:
            role: The ARIA role to filter by

        Returns:
            List of matching nodes
        """
        return [
            node for node in self.root.traverse()
            if node.role.value == role
        ]

    @property
    def node_count(self) -> int:
        """Total number of nodes in the tree."""
        return self.root.count_nodes()

    @property
    def interactive_count(self) -> int:
        """Number of interactive nodes in the tree."""
        return self.root.count_interactive()

    def to_yaml(self, indent: int = 0) -> str:
        """Convert tree to YAML format.

        Args:
            indent: Starting indentation level

        Returns:
            YAML-formatted string representation
        """
        return self._node_to_yaml(self.root, indent)

    def _node_to_yaml(self, node: AriaNode, indent: int) -> str:
        """Convert a single node and its children to YAML.

        Args:
            node: The node to convert
            indent: Current indentation level

        Returns:
            YAML-formatted string for this node subtree
        """
        prefix = "  " * indent
        parts = [f"{prefix}- {node.role.value}"]

        if node.name:
            parts.append(f' "{node.name}"')

        if node.level is not None:
            parts.append(f" [level={node.level}]")

        parts.append(f" [ref={node.ref.value}]")

        result = "".join(parts)

        # Add children
        for child in node.children:
            result += "\n" + self._node_to_yaml(child, indent + 1)

        return result

    def get_content_hash(self) -> str:
        """Compute a hash of the tree content for comparison.

        Returns:
            MD5 hash of the tree's YAML representation
        """
        yaml_content = self.to_yaml()
        return hashlib.md5(yaml_content.encode()).hexdigest()

    @classmethod
    def from_yaml(cls, yaml_content: str, snapshot_id: Optional[SnapshotId] = None) -> "AriaTree":
        """Parse an AriaTree from YAML content.

        Args:
            yaml_content: YAML string from Get Aria Snapshot
            snapshot_id: Optional snapshot ID (generated if not provided)

        Returns:
            Parsed AriaTree instance
        """
        lines = yaml_content.strip().split('\n')
        if not lines:
            # Return empty tree with document root
            root = AriaNode(
                ref=ElementRef.from_index(0),
                role=AriaRole("document")
            )
            return cls(root=root, snapshot_id=snapshot_id or SnapshotId.generate())

        root, _ = cls._parse_yaml_lines(lines, 0, 0)
        return cls(root=root, snapshot_id=snapshot_id or SnapshotId.generate())

    @classmethod
    def _parse_yaml_lines(
        cls,
        lines: List[str],
        start_idx: int,
        parent_indent: int
    ) -> Tuple[AriaNode, int]:
        """Parse YAML lines into AriaNode tree.

        Args:
            lines: All lines of the YAML
            start_idx: Index of current line to parse
            parent_indent: Indentation level of parent

        Returns:
            Tuple of (parsed node, index of next line to process)
        """
        if start_idx >= len(lines):
            raise ValueError("Unexpected end of YAML content")

        line = lines[start_idx]
        current_indent = len(line) - len(line.lstrip())

        # Parse the node content
        node = cls._parse_node_line(line.strip())

        # Parse children
        children: List[AriaNode] = []
        next_idx = start_idx + 1

        while next_idx < len(lines):
            next_line = lines[next_idx]
            next_indent = len(next_line) - len(next_line.lstrip())

            if next_indent <= current_indent:
                # Same or lower level - not our child
                break

            if next_indent > current_indent:
                # Child node
                child, next_idx = cls._parse_yaml_lines(lines, next_idx, current_indent)
                children.append(child)
            else:
                next_idx += 1

        node.children = children
        return node, next_idx

    @classmethod
    def _parse_node_line(cls, line: str) -> AriaNode:
        """Parse a single YAML line into an AriaNode.

        Expected format: - role "name" [ref=eN] [level=N]

        Args:
            line: Stripped line content

        Returns:
            Parsed AriaNode (without children)
        """
        # Remove leading dash
        content = line.lstrip('- ').strip()

        # Extract ref
        ref_match = re.search(r'\[ref=(e\d+)\]', content)
        if ref_match:
            ref = ElementRef(ref_match.group(1))
            content = content.replace(ref_match.group(0), '').strip()
        else:
            # Generate a ref if not present
            ref = ElementRef.from_index(0)

        # Extract level
        level = None
        level_match = re.search(r'\[level=(\d+)\]', content)
        if level_match:
            level = int(level_match.group(1))
            content = content.replace(level_match.group(0), '').strip()

        # Extract name (quoted string)
        name = None
        name_match = re.search(r'"([^"]*)"', content)
        if name_match:
            name = name_match.group(1)
            content = content.replace(f'"{name}"', '').strip()

        # Extract properties (other bracketed values)
        properties: Dict[str, Any] = {}
        prop_matches = re.findall(r'\[([^=]+)=([^\]]+)\]', content)
        for prop_name, prop_value in prop_matches:
            properties[prop_name] = prop_value
            content = re.sub(rf'\[{prop_name}={re.escape(prop_value)}\]', '', content)

        # Remaining content is the role
        role_str = content.strip()
        if not role_str:
            role_str = "generic"

        return AriaNode(
            ref=ref,
            role=AriaRole(role_str),
            name=name,
            level=level,
            properties=properties
        )


@dataclass
class PageSnapshot:
    """Complete page snapshot including ARIA tree and metadata.

    Represents the full state of a page at a point in time, including
    the accessibility tree, URL, title, and optimization metadata.
    """

    snapshot_id: SnapshotId
    """Unique identifier for this snapshot."""

    session_id: str
    """Session this snapshot belongs to."""

    aria_tree: AriaTree
    """The ARIA accessibility tree."""

    url: Optional[str] = None
    """Current page URL."""

    title: Optional[str] = None
    """Page title."""

    captured_at: datetime = field(default_factory=datetime.now)
    """When this snapshot was captured."""

    capture_duration_ms: float = 0.0
    """How long the capture took."""

    compression_applied: bool = False
    """Whether list folding was applied."""

    token_estimate: int = 0
    """Estimated token count for LLM consumption."""

    _yaml_cache: Optional[str] = field(default=None, repr=False)
    """Cached YAML representation."""

    @classmethod
    def create(
        cls,
        session_id: str,
        aria_yaml: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        capture_duration_ms: float = 0.0
    ) -> "PageSnapshot":
        """Create a new PageSnapshot from ARIA YAML content.

        Args:
            session_id: The session ID
            aria_yaml: YAML content from Get Aria Snapshot
            url: Optional current URL
            title: Optional page title
            capture_duration_ms: How long capture took

        Returns:
            New PageSnapshot instance
        """
        snapshot_id = SnapshotId.generate()
        aria_tree = AriaTree.from_yaml(aria_yaml, snapshot_id)

        snapshot = cls(
            snapshot_id=snapshot_id,
            session_id=session_id,
            aria_tree=aria_tree,
            url=url,
            title=title,
            captured_at=datetime.now(),
            capture_duration_ms=capture_duration_ms
        )

        # Estimate tokens
        snapshot.token_estimate = snapshot.estimate_tokens()

        return snapshot

    def to_yaml(self) -> str:
        """Get YAML representation of the snapshot.

        Returns:
            YAML-formatted string of the ARIA tree
        """
        if self._yaml_cache is None:
            object.__setattr__(self, '_yaml_cache', self.aria_tree.to_yaml())
        return self._yaml_cache

    def estimate_tokens(self, chars_per_token: float = 4.0) -> int:
        """Estimate token count for LLM consumption.

        Args:
            chars_per_token: Average characters per token

        Returns:
            Estimated token count
        """
        yaml_content = self.to_yaml()
        return int(len(yaml_content) / chars_per_token)

    def get_node_by_ref(self, ref: ElementRef) -> Optional[AriaNode]:
        """Get a node by its element reference.

        Args:
            ref: The ElementRef to look up

        Returns:
            The matching AriaNode, or None if not found
        """
        return self.aria_tree.get_node_by_ref(ref)

    def get_all_refs(self) -> List[str]:
        """Get all element refs in this snapshot.

        Returns:
            List of ref strings (e.g., ["e0", "e1", "e2"])
        """
        return list(self.aria_tree._ref_index.keys())

    def get_incremental_diff(self, previous: "PageSnapshot") -> str:
        """Get incremental diff from a previous snapshot.

        Args:
            previous: The previous snapshot to compare against

        Returns:
            String describing the changes, or the full YAML if too different
        """
        from robotmcp.domains.snapshot.services import SnapshotDiffService
        diff_service = SnapshotDiffService()
        return diff_service.compute_diff(previous, self)

    def get_content_hash(self) -> str:
        """Get content hash for comparison.

        Returns:
            MD5 hash of the snapshot content
        """
        return self.aria_tree.get_content_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "snapshot_id": str(self.snapshot_id),
            "session_id": self.session_id,
            "url": self.url,
            "title": self.title,
            "captured_at": self.captured_at.isoformat(),
            "capture_duration_ms": self.capture_duration_ms,
            "node_count": self.aria_tree.node_count,
            "interactive_count": self.aria_tree.interactive_count,
            "token_estimate": self.token_estimate,
            "compression_applied": self.compression_applied
        }
