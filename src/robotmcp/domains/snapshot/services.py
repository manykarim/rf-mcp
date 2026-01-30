"""
Domain Services for the Snapshot bounded context.

Domain services encapsulate domain logic that doesn't naturally fit within
a single entity or value object. They operate on domain objects and
perform operations that span multiple aggregates.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# Import from models.py for consistency with what __init__.py exports
from .models import (
    AriaNode,
    AriaRole,
    AriaTree,
    CompressionStats,
    ElementRef,
    PageSnapshot,
    SnapshotFormat,
    SnapshotId,
)
from .events import ListsFolded, SnapshotDiffComputed


class SnapshotDiffService:
    """
    Service for computing incremental diffs between snapshots.

    This service compares two snapshots and identifies what has changed,
    enabling incremental snapshot responses that significantly reduce
    token consumption.
    """

    def compute_diff(
        self,
        current: PageSnapshot,
        previous: Optional[PageSnapshot]
    ) -> SnapshotDiffComputed:
        """
        Compute the diff between current and previous snapshots.

        Args:
            current: The current (newer) snapshot
            previous: The previous (older) snapshot, or None for full diff

        Returns:
            A SnapshotDiffComputed event containing the changes
        """
        if previous is None:
            # No previous snapshot - everything is "added"
            all_refs = list(current.aria_tree.get_all_refs())
            return SnapshotDiffComputed(
                current_snapshot_id=current.snapshot_id,
                previous_snapshot_id=SnapshotId("none"),
                added_refs=all_refs,
                removed_refs=[],
                modified_refs=[]
            )

        current_refs = current.aria_tree.get_all_refs()
        previous_refs = previous.aria_tree.get_all_refs()

        added_refs = list(current_refs - previous_refs)
        removed_refs = list(previous_refs - current_refs)

        # Find modified refs (same ref, different content)
        common_refs = current_refs & previous_refs
        modified_refs: List[ElementRef] = []

        for ref in common_refs:
            current_node = current.aria_tree.find_by_ref(ref)
            previous_node = previous.aria_tree.find_by_ref(ref)

            if current_node and previous_node:
                if self._nodes_differ(current_node, previous_node):
                    modified_refs.append(ref)

        return SnapshotDiffComputed(
            current_snapshot_id=current.snapshot_id,
            previous_snapshot_id=previous.snapshot_id,
            added_refs=added_refs,
            removed_refs=removed_refs,
            modified_refs=modified_refs
        )

    def _nodes_differ(self, node1: AriaNode, node2: AriaNode) -> bool:
        """Check if two nodes have different content (excluding children)."""
        return (
            node1.role != node2.role or
            node1.name != node2.name or
            node1.level != node2.level or
            node1.properties != node2.properties
        )

    def generate_diff_yaml(
        self,
        current: PageSnapshot,
        previous: Optional[PageSnapshot],
        diff: Optional[SnapshotDiffComputed] = None
    ) -> str:
        """
        Generate a YAML representation of the diff.

        Args:
            current: The current snapshot
            previous: The previous snapshot
            diff: Pre-computed diff (optional, will compute if not provided)

        Returns:
            YAML-formatted diff string
        """
        if diff is None:
            diff = self.compute_diff(current, previous)

        lines = ["# Incremental Snapshot Diff"]
        lines.append(f"# Current: {diff.current_snapshot_id}")
        lines.append(f"# Previous: {diff.previous_snapshot_id}")
        lines.append(f"# Changes: +{len(diff.added_refs)} -{len(diff.removed_refs)} ~{len(diff.modified_refs)}")
        lines.append("")

        if diff.added_refs:
            lines.append("# Added:")
            for ref in sorted(diff.added_refs, key=lambda r: r.index):
                node = current.aria_tree.find_by_ref(ref)
                if node:
                    lines.append(self._node_to_line(node, prefix="+ "))

        if diff.modified_refs:
            lines.append("")
            lines.append("# Modified:")
            for ref in sorted(diff.modified_refs, key=lambda r: r.index):
                node = current.aria_tree.find_by_ref(ref)
                if node:
                    lines.append(self._node_to_line(node, prefix="~ "))

        if diff.removed_refs:
            lines.append("")
            lines.append("# Removed:")
            for ref in sorted(diff.removed_refs, key=lambda r: r.index):
                lines.append(f"- [removed] ref={ref}")

        return "\n".join(lines)

    def _node_to_line(self, node: AriaNode, prefix: str = "") -> str:
        """Convert a node to a single line representation."""
        parts = [f"{prefix}{node.role}"]

        if node.name:
            escaped_name = node.name.replace('"', '\\"')
            parts.append(f'"{escaped_name}"')

        if node.level is not None:
            parts.append(f"[level={node.level}]")

        parts.append(f"[ref={node.ref}]")

        return " ".join(parts)


class ListFoldingService:
    """
    Service for SimHash-based list compression.

    This service identifies repetitive list content and folds it
    to reduce token consumption while preserving essential information.
    """

    def __init__(self, default_threshold: float = 0.85):
        """
        Initialize the list folding service.

        Args:
            default_threshold: Default similarity threshold for folding (0-1)
        """
        self.default_threshold = default_threshold

    def fold_snapshot(
        self,
        snapshot: PageSnapshot,
        threshold: Optional[float] = None
    ) -> Tuple[PageSnapshot, ListsFolded]:
        """
        Apply list folding to a snapshot.

        Args:
            snapshot: The snapshot to fold
            threshold: Similarity threshold (uses default if not provided)

        Returns:
            Tuple of (folded snapshot, ListsFolded event)
        """
        threshold = threshold or self.default_threshold

        # Track original token estimate
        tokens_before = snapshot.estimate_tokens()

        # Fold the tree
        folded_root, stats = self._fold_node(snapshot.aria_tree.root, threshold)

        # Create new tree
        new_tree = AriaTree(root=folded_root)
        # Calculate counts - method may be named differently
        if hasattr(new_tree, '_recalculate_counts'):
            new_tree._recalculate_counts()
        elif hasattr(new_tree, '_calculate_counts'):
            new_tree._calculate_counts()

        # Calculate new token estimate
        tokens_after = self._estimate_tree_tokens(new_tree)

        # Create compression stats
        compression_stats = CompressionStats(
            original_nodes=snapshot.aria_tree.node_count,
            compressed_nodes=new_tree.node_count,
            folded_lists=stats["lists_folded"],
            token_estimate_before=tokens_before,
            token_estimate_after=tokens_after
        )

        # Create new snapshot - use only fields that exist in models.py PageSnapshot
        folded_snapshot = PageSnapshot(
            snapshot_id=snapshot.snapshot_id,
            session_id=snapshot.session_id,
            aria_tree=new_tree,
            created_at=snapshot.created_at,
            format=snapshot.format,
            compression_stats=compression_stats,
            url=getattr(snapshot, 'url', None),
            title=getattr(snapshot, 'title', None)
        )

        # Create event
        event = ListsFolded(
            snapshot_id=snapshot.snapshot_id,
            lists_folded=stats["lists_folded"],
            items_compressed=stats["items_compressed"],
            compression_ratio=compression_stats.compression_ratio,
            tokens_before=tokens_before,
            tokens_after=tokens_after
        )

        return folded_snapshot, event

    def _fold_node(
        self,
        node: AriaNode,
        threshold: float
    ) -> Tuple[AriaNode, Dict[str, int]]:
        """
        Recursively fold lists in the node tree.

        Returns the folded node and statistics.
        """
        stats = {"lists_folded": 0, "items_compressed": 0}

        # Process children first (depth-first)
        new_children: List[AriaNode] = []
        for child in node.children:
            folded_child, child_stats = self._fold_node(child, threshold)
            new_children.append(folded_child)
            stats["lists_folded"] += child_stats["lists_folded"]
            stats["items_compressed"] += child_stats["items_compressed"]

        # Check if this node is a list that should be folded
        list_roles = ("list", "listbox", "menu", "grid", "tree", "group")
        if node.role.value in list_roles and len(new_children) > 3:
            # Calculate similarity using SimHash-like approach
            if self._children_are_similar(new_children, threshold):
                folded_children, items_compressed = self._fold_children(new_children)
                stats["lists_folded"] += 1
                stats["items_compressed"] += items_compressed
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
        Determine if children are similar enough to fold.

        Uses a simplified SimHash-like approach based on structural similarity.
        """
        if len(children) < 3:
            return False

        # Generate signatures for all children
        signatures = [self._compute_signature(child) for child in children]

        # Calculate pairwise similarity
        first_sig = signatures[0]
        similar_count = sum(
            1 for sig in signatures[1:]
            if self._signature_similarity(first_sig, sig) >= threshold
        )

        # Require majority to be similar
        return similar_count / (len(signatures) - 1) >= threshold

    def _compute_signature(self, node: AriaNode) -> Tuple[str, int, frozenset, int]:
        """
        Compute a structural signature for a node.

        The signature captures the essential structure without specific content.
        """
        child_roles = frozenset(child.role.value for child in node.children)
        name_length = len(node.name) if node.name else 0

        return (
            node.role.value,
            len(node.children),
            child_roles,
            name_length // 10  # Bucket name length
        )

    def _signature_similarity(
        self,
        sig1: Tuple[str, int, frozenset, int],
        sig2: Tuple[str, int, frozenset, int]
    ) -> float:
        """
        Calculate similarity between two signatures.

        Returns a value between 0 and 1.
        """
        role1, child_count1, child_roles1, name_bucket1 = sig1
        role2, child_count2, child_roles2, name_bucket2 = sig2

        # Role must match
        if role1 != role2:
            return 0.0

        # Calculate component similarities
        count_sim = 1.0 - abs(child_count1 - child_count2) / max(child_count1, child_count2, 1)

        # Child roles similarity (Jaccard)
        if child_roles1 or child_roles2:
            intersection = len(child_roles1 & child_roles2)
            union = len(child_roles1 | child_roles2)
            roles_sim = intersection / union if union > 0 else 1.0
        else:
            roles_sim = 1.0

        # Name length similarity
        name_sim = 1.0 - abs(name_bucket1 - name_bucket2) / max(name_bucket1, name_bucket2, 1)

        # Weighted average
        return 0.4 * count_sim + 0.4 * roles_sim + 0.2 * name_sim

    def _fold_children(self, children: List[AriaNode]) -> Tuple[List[AriaNode], int]:
        """
        Fold a list of similar children into a compact representation.

        Returns the folded list and count of items compressed.
        """
        if len(children) <= 3:
            return children, 0

        # Keep first item as representative
        first_child = children[0]

        # Create summary node for the rest
        remaining = children[1:]
        refs = [child.ref for child in remaining]

        if refs:
            ref_range = f"{refs[0].value}-{refs[-1].value}"
        else:
            ref_range = ""

        summary_node = AriaNode(
            ref=refs[0] if refs else first_child.ref,
            role=AriaRole("presentation"),
            name=f"... and {len(remaining)} more similar items",
            properties={
                "folded": True,
                "folded_count": len(remaining),
                "folded_refs": ref_range,
                "original_refs": [str(r) for r in refs]
            }
        )

        return [first_child, summary_node], len(remaining)

    def _estimate_tree_tokens(self, tree: AriaTree) -> int:
        """Estimate tokens for a tree."""
        total = 0
        for node in tree.traverse():
            total += 8  # Base tokens per node
            if node.name:
                total += int(len(node.name) * 0.25)
            for key, value in node.properties.items():
                total += int((len(key) + len(str(value))) * 0.25)
        return total


class SnapshotCaptureService:
    """
    Service for capturing page snapshots.

    This service orchestrates the snapshot capture process,
    including tree construction and reference assignment.
    """

    def __init__(self):
        """Initialize the capture service."""
        self._ref_counter = 0

    def reset_ref_counter(self) -> None:
        """Reset the element reference counter."""
        self._ref_counter = 0

    def create_snapshot_from_raw(
        self,
        raw_aria_content: str,
        session_id: str,
        format: Optional[SnapshotFormat] = None
    ) -> PageSnapshot:
        """
        Create a PageSnapshot from raw aria snapshot content.

        Args:
            raw_aria_content: Raw aria snapshot string from browser
            session_id: The session identifier
            format: Optional format configuration

        Returns:
            A new PageSnapshot
        """
        # Parse the raw content into an AriaTree
        tree = self._parse_aria_content(raw_aria_content)

        # Create the snapshot using the factory method for compatibility
        snapshot = PageSnapshot.create(
            session_id=session_id,
            aria_tree=tree,
            format=format or SnapshotFormat.full()
        )

        return snapshot

    def _parse_aria_content(self, content: str) -> AriaTree:
        """
        Parse raw aria content into an AriaTree.

        This is a simplified parser - the actual implementation would
        need to handle the specific format from the browser library.
        """
        # Reset ref counter for new tree
        self._ref_counter = 0

        # Create a basic tree structure
        # In practice, this would parse the actual aria snapshot format
        root = AriaNode(
            ref=ElementRef.from_index(self._next_ref()),
            role=AriaRole("document"),
            name=None,
            children=[]
        )

        # Parse content lines and build tree
        lines = content.strip().split('\n')
        if lines:
            self._parse_lines(lines, root, 0)

        tree = AriaTree(root=root)
        # Calculate counts - method may be named differently
        if hasattr(tree, '_recalculate_counts'):
            tree._recalculate_counts()
        elif hasattr(tree, '_calculate_counts'):
            tree._calculate_counts()

        return tree

    def _next_ref(self) -> int:
        """Get the next reference number."""
        ref = self._ref_counter
        self._ref_counter += 1
        return ref

    def _parse_lines(
        self,
        lines: List[str],
        parent: AriaNode,
        start_idx: int
    ) -> int:
        """
        Parse lines into the tree structure.

        Returns the index of the last processed line.
        """
        # Simplified parsing - actual implementation would be more robust
        idx = start_idx
        current_indent = self._get_indent(lines[start_idx]) if lines else 0

        while idx < len(lines):
            line = lines[idx]
            indent = self._get_indent(line)

            if indent < current_indent:
                return idx

            if indent == current_indent:
                # Parse this line as a sibling
                node = self._parse_line(line)
                if node:
                    # Append to parent's children list (models.py AriaNode uses list directly)
                    parent.children.append(node)

                    # Check for children
                    if idx + 1 < len(lines):
                        next_indent = self._get_indent(lines[idx + 1])
                        if next_indent > indent:
                            idx = self._parse_lines(lines, node, idx + 1)
                            continue

            idx += 1

        return idx

    def _get_indent(self, line: str) -> int:
        """Get the indentation level of a line."""
        stripped = line.lstrip()
        return len(line) - len(stripped)

    def _parse_line(self, line: str) -> Optional[AriaNode]:
        """
        Parse a single line into an AriaNode.

        Expected format: "- role \"name\" [level=N] [ref=eN]"
        """
        line = line.strip()
        if not line or not line.startswith('-'):
            return None

        # Remove leading dash
        line = line[1:].strip()

        # Extract role (first word)
        parts = line.split()
        if not parts:
            return None

        role = parts[0]
        name = None
        level = None
        ref_num = self._next_ref()

        # Extract name (quoted string)
        if '"' in line:
            start = line.index('"') + 1
            end = line.index('"', start) if '"' in line[start:] else len(line)
            name = line[start:end] if end > start else None

        # Extract level
        if '[level=' in line:
            level_start = line.index('[level=') + 7
            level_end = line.index(']', level_start)
            try:
                level = int(line[level_start:level_end])
            except ValueError:
                pass

        # Extract existing ref if present
        if '[ref=' in line:
            ref_start = line.index('[ref=') + 5
            ref_end = line.index(']', ref_start)
            ref_str = line[ref_start:ref_end]
            if ref_str.startswith('e'):
                try:
                    ref_num = int(ref_str[1:])
                except ValueError:
                    pass

        return AriaNode(
            ref=ElementRef.from_index(ref_num),
            role=AriaRole(role),
            name=name,
            level=level,
            children=[],
            properties={}
        )
