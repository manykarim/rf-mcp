"""
Incremental snapshot diff tracking.

Computes and returns only changed portions between snapshots,
dramatically reducing token count when page content is mostly unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# Try to import from shared kernel first, fall back to models for compatibility
try:
    from robotmcp.domains.shared.kernel import AriaNode, AriaRole, ElementRef
    from robotmcp.domains.snapshot.entities import AriaTree, PageSnapshot
except ImportError:
    # Fall back to models module for backwards compatibility
    from robotmcp.domains.snapshot.models import (
        AriaNode,
        AriaTree,
        ElementRef,
        PageSnapshot,
    )


class ChangeType(Enum):
    """Types of changes that can occur between snapshots."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class NodeChange:
    """
    Represents a change to a single node between snapshots.

    Tracks what changed (added, removed, modified) along with
    the old and new values for rendering diffs.
    """
    ref: ElementRef
    change_type: ChangeType
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    old_node: Optional[AriaNode] = None
    new_node: Optional[AriaNode] = None

    def to_yaml_line(self) -> str:
        """
        Format change for YAML output with diff prefix.

        Returns:
            YAML line with appropriate prefix (+ for added, - for removed, ~ for modified).
        """
        prefix = {
            ChangeType.ADDED: "+ ",
            ChangeType.REMOVED: "- ",
            ChangeType.MODIFIED: "~ ",
            ChangeType.UNCHANGED: "  ",
        }[self.change_type]

        value = self.new_value if self.new_value is not None else self.old_value
        return f"{prefix}{value}"

    def to_detailed_yaml(self) -> str:
        """
        Format change with both old and new values for modified nodes.

        Returns:
            Detailed YAML representation showing before/after for modifications.
        """
        if self.change_type == ChangeType.MODIFIED:
            lines = [
                f"~ [ref={self.ref.value}]",
                f"  - before: {self.old_value}",
                f"  + after:  {self.new_value}",
            ]
            return "\n".join(lines)
        return self.to_yaml_line()


@dataclass
class SnapshotDiff:
    """
    Result of comparing two snapshots.

    Contains categorized lists of changes (added, removed, modified)
    along with statistics about unchanged content.
    """
    added_nodes: List[NodeChange] = field(default_factory=list)
    removed_nodes: List[NodeChange] = field(default_factory=list)
    modified_nodes: List[NodeChange] = field(default_factory=list)
    unchanged_count: int = 0
    previous_snapshot_id: Optional[str] = None
    current_snapshot_id: Optional[str] = None

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes between snapshots."""
        return bool(self.added_nodes or self.removed_nodes or self.modified_nodes)

    @property
    def change_count(self) -> int:
        """Get total number of changes."""
        return len(self.added_nodes) + len(self.removed_nodes) + len(self.modified_nodes)

    @property
    def total_nodes(self) -> int:
        """Get total node count (changed + unchanged)."""
        return self.change_count + self.unchanged_count

    @property
    def change_ratio(self) -> float:
        """Calculate ratio of changed nodes to total nodes."""
        if self.total_nodes == 0:
            return 0.0
        return self.change_count / self.total_nodes

    def to_yaml(self, include_unchanged: bool = False, detailed: bool = False) -> str:
        """
        Convert diff to YAML representation.

        Args:
            include_unchanged: Whether to include unchanged node count in output.
            detailed: Whether to show detailed before/after for modifications.

        Returns:
            YAML string representing the diff.
        """
        lines: List[str] = []

        # Header with summary
        lines.append(f"# Diff: {self.change_count} changes ({self.unchanged_count} unchanged)")

        if self.removed_nodes:
            lines.append("")
            lines.append("# Removed:")
            for node in self.removed_nodes:
                if detailed:
                    lines.append(node.to_detailed_yaml())
                else:
                    lines.append(node.to_yaml_line())

        if self.added_nodes:
            lines.append("")
            lines.append("# Added:")
            for node in self.added_nodes:
                if detailed:
                    lines.append(node.to_detailed_yaml())
                else:
                    lines.append(node.to_yaml_line())

        if self.modified_nodes:
            lines.append("")
            lines.append("# Modified:")
            for node in self.modified_nodes:
                if detailed:
                    lines.append(node.to_detailed_yaml())
                else:
                    lines.append(node.to_yaml_line())

        if not self.has_changes:
            lines.append("")
            lines.append("# No changes detected")

        if include_unchanged:
            lines.append("")
            lines.append(f"# Unchanged: {self.unchanged_count} nodes")

        return "\n".join(lines)

    def to_compact_yaml(self) -> str:
        """
        Convert diff to compact YAML representation.

        Only shows the changes without headers or statistics,
        optimized for minimal token usage.

        Returns:
            Compact YAML string.
        """
        if not self.has_changes:
            return "[No changes]"

        lines: List[str] = []

        for node in self.removed_nodes:
            lines.append(node.to_yaml_line())

        for node in self.added_nodes:
            lines.append(node.to_yaml_line())

        for node in self.modified_nodes:
            lines.append(node.to_yaml_line())

        return "\n".join(lines)

    def estimate_token_savings(self, full_snapshot_tokens: int) -> int:
        """
        Estimate token savings compared to full snapshot.

        Args:
            full_snapshot_tokens: Token count for full snapshot.

        Returns:
            Estimated tokens saved by using diff.
        """
        diff_tokens = len(self.to_yaml()) // 4  # ~4 chars per token
        return max(0, full_snapshot_tokens - diff_tokens)


class SnapshotDiffService:
    """
    Service for computing incremental diffs between snapshots.

    Analyzes two snapshots and produces a diff containing only the
    changes, enabling significant token savings when page content
    is mostly unchanged.
    """

    def compute_diff(
        self,
        previous: PageSnapshot,
        current: PageSnapshot,
        semantic_comparison: bool = True
    ) -> SnapshotDiff:
        """
        Compute the difference between two snapshots.

        Args:
            previous: The previous/baseline snapshot.
            current: The current snapshot to compare.
            semantic_comparison: If True, compares node content semantically.
                               If False, only compares by ref.

        Returns:
            SnapshotDiff containing only the changes, dramatically
            reducing token count when page content is mostly unchanged.
        """
        # Build ref -> node maps for efficient lookup
        prev_nodes = self._build_node_map(previous.aria_tree)
        curr_nodes = self._build_node_map(current.aria_tree)

        prev_refs = set(prev_nodes.keys())
        curr_refs = set(curr_nodes.keys())

        # Find added, removed, and common refs
        added_refs = curr_refs - prev_refs
        removed_refs = prev_refs - curr_refs
        common_refs = prev_refs & curr_refs

        # Check common refs for modifications
        modified_refs: Set[ElementRef] = set()
        if semantic_comparison:
            for ref in common_refs:
                if not self._nodes_equal(prev_nodes[ref], curr_nodes[ref]):
                    modified_refs.add(ref)

        # Build change objects
        added = [
            NodeChange(
                ref=ref,
                change_type=ChangeType.ADDED,
                new_value=self._node_to_yaml(curr_nodes[ref]),
                new_node=curr_nodes[ref]
            )
            for ref in sorted(added_refs, key=lambda r: r.value)
        ]

        removed = [
            NodeChange(
                ref=ref,
                change_type=ChangeType.REMOVED,
                old_value=self._node_to_yaml(prev_nodes[ref]),
                old_node=prev_nodes[ref]
            )
            for ref in sorted(removed_refs, key=lambda r: r.value)
        ]

        modified = [
            NodeChange(
                ref=ref,
                change_type=ChangeType.MODIFIED,
                old_value=self._node_to_yaml(prev_nodes[ref]),
                new_value=self._node_to_yaml(curr_nodes[ref]),
                old_node=prev_nodes[ref],
                new_node=curr_nodes[ref]
            )
            for ref in sorted(modified_refs, key=lambda r: r.value)
        ]

        return SnapshotDiff(
            added_nodes=added,
            removed_nodes=removed,
            modified_nodes=modified,
            unchanged_count=len(common_refs) - len(modified_refs),
            previous_snapshot_id=str(previous.snapshot_id),
            current_snapshot_id=str(current.snapshot_id)
        )

    def compute_structural_diff(
        self,
        previous: PageSnapshot,
        current: PageSnapshot
    ) -> Tuple[SnapshotDiff, Dict[str, int]]:
        """
        Compute diff with additional structural change analysis.

        This method provides more detailed analysis of what changed,
        including hierarchy changes and property changes.

        Args:
            previous: The previous snapshot.
            current: The current snapshot.

        Returns:
            Tuple of (SnapshotDiff, statistics dict).
        """
        diff = self.compute_diff(previous, current)

        stats = {
            "total_added": len(diff.added_nodes),
            "total_removed": len(diff.removed_nodes),
            "total_modified": len(diff.modified_nodes),
            "unchanged": diff.unchanged_count,
            "name_changes": 0,
            "role_changes": 0,
            "property_changes": 0,
        }

        # Analyze modifications in detail
        for change in diff.modified_nodes:
            if change.old_node and change.new_node:
                if change.old_node.name != change.new_node.name:
                    stats["name_changes"] += 1
                if change.old_node.role != change.new_node.role:
                    stats["role_changes"] += 1
                if change.old_node.properties != change.new_node.properties:
                    stats["property_changes"] += 1

        return diff, stats

    def _build_node_map(self, tree: AriaTree) -> Dict[ElementRef, AriaNode]:
        """
        Build a map from ref to node for efficient lookup.

        Args:
            tree: The AriaTree to index.

        Returns:
            Dict mapping ElementRef to AriaNode.
        """
        node_map: Dict[ElementRef, AriaNode] = {}
        for node in tree.traverse():
            node_map[node.ref] = node
        return node_map

    def _nodes_equal(self, node1: AriaNode, node2: AriaNode) -> bool:
        """
        Check if two nodes are semantically equal.

        Compares role, name, level, and properties but ignores children
        since they are tracked separately by their own refs.

        Args:
            node1: First node.
            node2: Second node.

        Returns:
            True if nodes are semantically equal.
        """
        return (
            node1.role == node2.role and
            node1.name == node2.name and
            node1.level == node2.level and
            node1.properties == node2.properties
        )

    def _node_to_yaml(self, node: AriaNode) -> str:
        """
        Convert node to YAML line representation.

        Args:
            node: The AriaNode to convert.

        Returns:
            YAML string representation.
        """
        parts = [f"{node.role.value}"]

        if node.name:
            escaped_name = node.name.replace('"', '\\"')
            parts.append(f'"{escaped_name}"')

        if node.level is not None:
            parts.append(f"[level={node.level}]")

        # Include relevant properties
        for key, value in sorted(node.properties.items()):
            if key in ("checked", "selected", "disabled", "expanded", "pressed", "value"):
                parts.append(f"[{key}={value}]")

        parts.append(f"[ref={node.ref.value}]")

        return " ".join(parts)


class SnapshotManager:
    """
    Manages snapshot capture with caching for diff computation.

    Maintains a per-session cache of recent snapshots to enable
    incremental diff tracking without requiring the caller to
    manage snapshot history.
    """

    def __init__(self, max_cache_per_session: int = 5):
        """
        Initialize the snapshot manager.

        Args:
            max_cache_per_session: Maximum snapshots to cache per session (LRU eviction).
        """
        self.max_cache = max_cache_per_session
        self._cache: Dict[str, List[PageSnapshot]] = {}
        self._diff_service = SnapshotDiffService()

    def capture_and_diff(
        self,
        session_id: str,
        new_snapshot: PageSnapshot,
        mode: str = "incremental"
    ) -> str:
        """
        Capture snapshot and return appropriate output based on mode.

        Args:
            session_id: Session identifier.
            new_snapshot: Newly captured snapshot.
            mode: Output mode - "full", "incremental", or "none".

        Returns:
            YAML string - full snapshot, diff, or empty string.
        """
        if mode == "none":
            self._cache_snapshot(session_id, new_snapshot)
            return ""

        if mode == "full":
            self._cache_snapshot(session_id, new_snapshot)
            return new_snapshot.to_yaml()

        # Incremental mode
        previous = self.get_previous_snapshot(session_id)
        self._cache_snapshot(session_id, new_snapshot)

        if previous is None:
            # First snapshot for session - return full
            return new_snapshot.to_yaml()

        diff = self._diff_service.compute_diff(previous, new_snapshot)

        if not diff.has_changes:
            return "[No changes detected since last snapshot]"

        return diff.to_yaml()

    def capture_and_diff_detailed(
        self,
        session_id: str,
        new_snapshot: PageSnapshot,
        mode: str = "incremental"
    ) -> Tuple[str, Optional[SnapshotDiff]]:
        """
        Capture snapshot and return output with diff object.

        Args:
            session_id: Session identifier.
            new_snapshot: Newly captured snapshot.
            mode: Output mode - "full", "incremental", or "none".

        Returns:
            Tuple of (YAML string, SnapshotDiff or None).
        """
        if mode == "none":
            self._cache_snapshot(session_id, new_snapshot)
            return "", None

        if mode == "full":
            self._cache_snapshot(session_id, new_snapshot)
            return new_snapshot.to_yaml(), None

        previous = self.get_previous_snapshot(session_id)
        self._cache_snapshot(session_id, new_snapshot)

        if previous is None:
            return new_snapshot.to_yaml(), None

        diff = self._diff_service.compute_diff(previous, new_snapshot)

        if not diff.has_changes:
            return "[No changes detected since last snapshot]", diff

        return diff.to_yaml(), diff

    def get_previous_snapshot(self, session_id: str) -> Optional[PageSnapshot]:
        """
        Get the most recent snapshot for a session.

        Args:
            session_id: Session identifier.

        Returns:
            The most recent PageSnapshot, or None if no history.
        """
        snapshots = self._cache.get(session_id, [])
        return snapshots[-1] if snapshots else None

    def get_snapshot_history(self, session_id: str) -> List[PageSnapshot]:
        """
        Get all cached snapshots for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of PageSnapshots in chronological order.
        """
        return self._cache.get(session_id, []).copy()

    def get_snapshot_count(self, session_id: str) -> int:
        """
        Get the number of cached snapshots for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Number of cached snapshots.
        """
        return len(self._cache.get(session_id, []))

    def _cache_snapshot(self, session_id: str, snapshot: PageSnapshot) -> None:
        """
        Cache snapshot with LRU eviction.

        Args:
            session_id: Session identifier.
            snapshot: Snapshot to cache.
        """
        if session_id not in self._cache:
            self._cache[session_id] = []

        self._cache[session_id].append(snapshot)

        # Evict old snapshots (LRU - keep most recent)
        if len(self._cache[session_id]) > self.max_cache:
            self._cache[session_id] = self._cache[session_id][-self.max_cache:]

    def clear_session_cache(self, session_id: str) -> None:
        """
        Clear cache for a specific session.

        Args:
            session_id: Session identifier to clear.
        """
        self._cache.pop(session_id, None)

    def clear_all_caches(self) -> None:
        """Clear all session caches."""
        self._cache.clear()

    def get_all_session_ids(self) -> List[str]:
        """
        Get all session IDs with cached snapshots.

        Returns:
            List of session IDs.
        """
        return list(self._cache.keys())

    def compute_diff_between(
        self,
        session_id: str,
        snapshot_index_old: int,
        snapshot_index_new: int
    ) -> Optional[SnapshotDiff]:
        """
        Compute diff between two specific snapshots in the cache.

        Args:
            session_id: Session identifier.
            snapshot_index_old: Index of older snapshot (0 = oldest cached).
            snapshot_index_new: Index of newer snapshot.

        Returns:
            SnapshotDiff or None if indices are invalid.
        """
        snapshots = self._cache.get(session_id, [])

        if (snapshot_index_old < 0 or snapshot_index_old >= len(snapshots) or
            snapshot_index_new < 0 or snapshot_index_new >= len(snapshots)):
            return None

        old_snapshot = snapshots[snapshot_index_old]
        new_snapshot = snapshots[snapshot_index_new]

        return self._diff_service.compute_diff(old_snapshot, new_snapshot)
