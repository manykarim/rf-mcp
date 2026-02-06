"""
Aggregates for the Snapshot bounded context.

Aggregates are clusters of domain objects that can be treated as a single unit.
The PageSnapshot is the aggregate root for this bounded context.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .entities import AriaNode, AriaTree
from .value_objects import (
    CompressionStats,
    ElementRef,
    SnapshotFormat,
    SnapshotId,
    AriaRole,
)


# Token estimation constants (approximate)
TOKENS_PER_NODE_BASE = 8  # Base tokens per node (role, ref brackets, etc.)
TOKENS_PER_CHAR = 0.25  # Approximate tokens per character


@dataclass
class PageSnapshot:
    """
    Aggregate root for page snapshots.

    The PageSnapshot aggregate manages all snapshot-related state including
    the accessibility tree, compression, and incremental diff computation.
    """
    snapshot_id: SnapshotId
    session_id: str
    aria_tree: AriaTree
    raw_content: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    format: SnapshotFormat = field(default_factory=lambda: SnapshotFormat.full())
    compression_stats: CompressionStats = field(default_factory=CompressionStats.empty)
    _ref_counter: int = field(default=0, repr=False)

    def capture(self, selector: Optional[str] = None) -> AriaTree:
        """
        Capture an accessibility tree for the page or a specific element.

        This is the primary method for obtaining a snapshot. The actual
        browser interaction should be performed by the infrastructure layer.

        Args:
            selector: Optional CSS selector to limit the snapshot scope

        Returns:
            The captured AriaTree
        """
        # The actual capture is delegated to the infrastructure layer
        # This method manages the domain logic around the capture
        return self.aria_tree

    def get_incremental_diff(self, previous: "PageSnapshot") -> str:
        """
        Compute an incremental diff between this snapshot and a previous one.

        Returns a YAML-formatted string showing only the changes.
        """
        if previous is None:
            return self.to_yaml()

        current_refs = self.aria_tree.get_all_refs()
        previous_refs = previous.aria_tree.get_all_refs()

        added_refs = current_refs - previous_refs
        removed_refs = previous_refs - current_refs

        # Build diff output
        lines = ["# Incremental snapshot diff"]

        if added_refs:
            lines.append("\n# Added elements:")
            for ref in sorted(added_refs, key=lambda r: r.index):
                node = self.aria_tree.find_by_ref(ref)
                if node:
                    lines.append(self._node_to_yaml_line(node, indent=0))

        if removed_refs:
            lines.append("\n# Removed elements:")
            for ref in sorted(removed_refs, key=lambda r: r.index):
                lines.append(f"  - [REMOVED] ref={ref}")

        # Find modified nodes (same ref, different content)
        common_refs = current_refs & previous_refs
        modified_nodes: List[Tuple[AriaNode, AriaNode]] = []

        for ref in common_refs:
            current_node = self.aria_tree.find_by_ref(ref)
            previous_node = previous.aria_tree.find_by_ref(ref)
            if current_node and previous_node:
                if self._nodes_differ(current_node, previous_node):
                    modified_nodes.append((current_node, previous_node))

        if modified_nodes:
            lines.append("\n# Modified elements:")
            for current_node, _ in modified_nodes:
                lines.append(self._node_to_yaml_line(current_node, indent=0, prefix="[MODIFIED] "))

        return "\n".join(lines)

    def _nodes_differ(self, node1: AriaNode, node2: AriaNode) -> bool:
        """Check if two nodes have different content (excluding children)."""
        return (
            node1.role != node2.role or
            node1.name != node2.name or
            node1.level != node2.level or
            node1.properties != node2.properties
        )

    def fold_lists(self, threshold: float = 0.85) -> "PageSnapshot":
        """
        Apply list folding compression to reduce token consumption.

        Uses SimHash-based similarity detection to identify repetitive
        list items and compress them into a summary.

        Args:
            threshold: Similarity threshold (0-1) for folding

        Returns:
            A new PageSnapshot with folded lists
        """
        folded_root, stats = self._fold_node_lists(self.aria_tree.root, threshold)

        new_tree = AriaTree(root=folded_root)
        new_tree._recalculate_counts()

        # Calculate token estimates
        token_before = self.estimate_tokens()
        new_snapshot = PageSnapshot(
            snapshot_id=self.snapshot_id,
            session_id=self.session_id,
            aria_tree=new_tree,
            raw_content=self.raw_content,
            created_at=self.created_at,
            format=self.format,
            compression_stats=CompressionStats(
                original_nodes=self.aria_tree.node_count,
                compressed_nodes=new_tree.node_count,
                folded_lists=stats["lists_folded"],
                token_estimate_before=token_before,
                token_estimate_after=new_snapshot._estimate_tokens_for_tree(new_tree)
            )
        )
        return new_snapshot

    def _fold_node_lists(
        self,
        node: AriaNode,
        threshold: float
    ) -> Tuple[AriaNode, Dict[str, int]]:
        """
        Recursively fold lists in the node tree.

        Returns the folded node and statistics.
        """
        stats = {"lists_folded": 0, "items_compressed": 0}

        # Process children first
        new_children: List[AriaNode] = []
        for child in node.children:
            folded_child, child_stats = self._fold_node_lists(child, threshold)
            new_children.append(folded_child)
            stats["lists_folded"] += child_stats["lists_folded"]
            stats["items_compressed"] += child_stats["items_compressed"]

        # Check if this node is a list that should be folded
        if node.role.value in ("list", "listbox", "menu", "grid", "tree"):
            if len(new_children) > 3:
                # Check similarity of children
                if self._children_are_similar(new_children, threshold):
                    # Fold the list
                    folded_children = self._create_folded_list(new_children)
                    stats["lists_folded"] += 1
                    stats["items_compressed"] += len(new_children) - len(folded_children)
                    new_children = folded_children

        # Create new node with processed children
        new_node = AriaNode(
            ref=node.ref,
            role=node.role,
            name=node.name,
            level=node.level,
            children=new_children,
            properties=node.properties.copy()
        )

        return new_node, stats

    def _children_are_similar(self, children: List[AriaNode], threshold: float) -> bool:
        """
        Check if list children are similar enough to fold.

        Uses a simplified similarity check based on role and structure.
        """
        if len(children) < 3:
            return False

        # Get signature of first child
        first_sig = self._node_signature(children[0])

        # Count how many children match the signature
        matching = sum(1 for child in children[1:] if self._node_signature(child) == first_sig)

        similarity = (matching + 1) / len(children)
        return similarity >= threshold

    def _node_signature(self, node: AriaNode) -> Tuple[str, int, frozenset]:
        """Generate a signature for a node based on its structure."""
        return (
            node.role.value,
            len(node.children),
            frozenset(child.role.value for child in node.children)
        )

    def _create_folded_list(self, children: List[AriaNode]) -> List[AriaNode]:
        """
        Create a folded representation of similar list items.

        Keeps first item, adds summary, and includes ref range.
        """
        if len(children) <= 3:
            return children

        # Keep first item
        first_child = children[0]

        # Create summary node
        refs = [child.ref for child in children[1:]]
        ref_range = f"e{refs[0].index}-e{refs[-1].index}" if refs else ""

        summary_node = AriaNode(
            ref=refs[0] if refs else first_child.ref,
            role=AriaRole("presentation"),
            name=f"... and {len(children) - 1} more similar items",
            properties={
                "folded": True,
                "folded_count": len(children) - 1,
                "folded_refs": ref_range,
                "original_refs": [str(r) for r in refs]
            }
        )

        return [first_child, summary_node]

    def to_yaml(self) -> str:
        """
        Convert the snapshot to YAML format.

        This is the primary output format for snapshots, optimized
        for token efficiency while remaining human-readable.
        """
        lines = ["# Page Snapshot"]
        lines.append(f"# ID: {self.snapshot_id}")
        lines.append(f"# Nodes: {self.aria_tree.node_count}")
        lines.append(f"# Interactive: {self.aria_tree.interactive_count}")
        lines.append("")

        self._node_to_yaml(self.aria_tree.root, lines, indent=0)

        return "\n".join(lines)

    def _node_to_yaml(self, node: AriaNode, lines: List[str], indent: int) -> None:
        """Recursively convert a node and its children to YAML."""
        lines.append(self._node_to_yaml_line(node, indent))

        # Respect max_depth if set
        if self.format.max_depth is not None and indent >= self.format.max_depth:
            if node.children:
                lines.append("  " * (indent + 1) + f"# ... {len(node.children)} children omitted")
            return

        for child in node.children:
            self._node_to_yaml(child, lines, indent + 1)

    def _node_to_yaml_line(self, node: AriaNode, indent: int, prefix: str = "") -> str:
        """Convert a single node to a YAML line."""
        indent_str = "  " * indent

        # Build the role part
        parts = [f"- {prefix}{node.role}"]

        # Add name if present
        if node.name:
            # Escape quotes in name
            escaped_name = node.name.replace('"', '\\"')
            parts.append(f'"{escaped_name}"')

        # Add level for headings
        if node.level is not None:
            parts.append(f"[level={node.level}]")

        # Add ref
        parts.append(f"[ref={node.ref}]")

        # Add folded indicator
        if node.properties.get("folded"):
            parts.append(f"[refs: {node.properties.get('folded_refs', '')}]")

        # Add key properties
        for key in ("checked", "disabled", "expanded", "selected", "pressed"):
            if node.properties.get(key):
                parts.append(f"[{key}]")

        return indent_str + " ".join(parts)

    def estimate_tokens(self) -> int:
        """
        Estimate the token count for this snapshot.

        Uses a heuristic based on node count and content length.
        """
        return self._estimate_tokens_for_tree(self.aria_tree)

    def _estimate_tokens_for_tree(self, tree: AriaTree) -> int:
        """Estimate tokens for a given tree."""
        total = 0
        for node in tree.traverse():
            # Base tokens per node
            total += TOKENS_PER_NODE_BASE

            # Add tokens for name
            if node.name:
                total += int(len(node.name) * TOKENS_PER_CHAR)

            # Add tokens for properties
            for key, value in node.properties.items():
                total += int((len(key) + len(str(value))) * TOKENS_PER_CHAR)

        return total

    def get_ref_by_role_and_name(
        self,
        role: str,
        name: Optional[str] = None
    ) -> Optional[ElementRef]:
        """
        Find an element reference by role and optional name.

        Useful for finding interactive elements to act upon.
        """
        for node in self.aria_tree.traverse():
            if node.role.value == role:
                if name is None or node.name == name:
                    return node.ref
        return None

    def get_interactive_elements(self) -> List[Dict[str, Any]]:
        """
        Get a list of all interactive elements with their refs.

        Returns a simplified list suitable for tool responses.
        """
        elements = []
        for node in self.aria_tree.find_interactive():
            elements.append({
                "ref": str(node.ref),
                "role": str(node.role),
                "name": node.name,
                "level": node.level
            })
        return elements

    def with_format(self, format: SnapshotFormat) -> "PageSnapshot":
        """Create a new snapshot with a different format configuration."""
        return PageSnapshot(
            snapshot_id=self.snapshot_id,
            session_id=self.session_id,
            aria_tree=self.aria_tree,
            raw_content=self.raw_content,
            created_at=self.created_at,
            format=format,
            compression_stats=self.compression_stats
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "snapshot_id": str(self.snapshot_id),
            "session_id": self.session_id,
            "aria_tree": self.aria_tree.to_dict(),
            "raw_content": self.raw_content,
            "created_at": self.created_at.isoformat(),
            "format": {
                "mode": self.format.mode,
                "include_hidden": self.format.include_hidden,
                "max_depth": self.format.max_depth,
                "fold_lists": self.format.fold_lists,
                "fold_threshold": self.format.fold_threshold
            },
            "compression_stats": {
                "original_nodes": self.compression_stats.original_nodes,
                "compressed_nodes": self.compression_stats.compressed_nodes,
                "folded_lists": self.compression_stats.folded_lists,
                "token_estimate_before": self.compression_stats.token_estimate_before,
                "token_estimate_after": self.compression_stats.token_estimate_after
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageSnapshot":
        """Create a PageSnapshot from a dictionary."""
        format_data = data.get("format", {})
        stats_data = data.get("compression_stats", {})

        return cls(
            snapshot_id=SnapshotId(data["snapshot_id"]),
            session_id=data["session_id"],
            aria_tree=AriaTree.from_dict(data["aria_tree"]),
            raw_content=data.get("raw_content"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            format=SnapshotFormat(
                mode=format_data.get("mode", "full"),
                include_hidden=format_data.get("include_hidden", False),
                max_depth=format_data.get("max_depth"),
                fold_lists=format_data.get("fold_lists", True),
                fold_threshold=format_data.get("fold_threshold", 0.85)
            ),
            compression_stats=CompressionStats(
                original_nodes=stats_data.get("original_nodes", 0),
                compressed_nodes=stats_data.get("compressed_nodes", 0),
                folded_lists=stats_data.get("folded_lists", 0),
                token_estimate_before=stats_data.get("token_estimate_before", 0),
                token_estimate_after=stats_data.get("token_estimate_after", 0)
            )
        )

    @classmethod
    def create_empty(cls, session_id: str) -> "PageSnapshot":
        """Create an empty snapshot for a session."""
        return cls(
            snapshot_id=SnapshotId.generate(),
            session_id=session_id,
            aria_tree=AriaTree.empty(),
            format=SnapshotFormat.full()
        )

    def __repr__(self) -> str:
        return (
            f"PageSnapshot(id={self.snapshot_id}, session={self.session_id}, "
            f"nodes={self.aria_tree.node_count}, tokens={self.estimate_tokens()})"
        )
