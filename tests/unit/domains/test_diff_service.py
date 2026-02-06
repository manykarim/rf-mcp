"""
Tests for the snapshot diff service.

Tests cover:
- SnapshotDiffService: Computing diffs between snapshots
- SnapshotManager: Caching and incremental diff management
- Token reduction validation
"""

import pytest
from datetime import datetime, timedelta, timezone

from robotmcp.domains.snapshot.models import (
    AriaNode,
    AriaRole,
    AriaTree,
    ElementRef,
    PageSnapshot,
    SnapshotFormat,
    SnapshotId,
)
from robotmcp.domains.snapshot.diff_service import (
    ChangeType,
    NodeChange,
    SnapshotDiff,
    SnapshotDiffService,
    SnapshotManager,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

def create_simple_tree(
    nodes: list[tuple[int, str, str | None, dict | None]]
) -> AriaTree:
    """
    Create a simple AriaTree for testing.

    Args:
        nodes: List of (ref_index, role, name, properties) tuples.
               First node becomes the root.

    Returns:
        AriaTree with flat structure (all nodes as root's children).
    """
    if not nodes:
        root = AriaNode(
            ref=ElementRef.from_index(0),
            role=AriaRole(AriaRole.DOCUMENT),
            name="empty"
        )
        return AriaTree(root=root)

    root_data = nodes[0]
    root = AriaNode(
        ref=ElementRef.from_index(root_data[0]),
        role=AriaRole(root_data[1]),
        name=root_data[2],
        properties=root_data[3] or {}
    )

    for ref_idx, role, name, props in nodes[1:]:
        child = AriaNode(
            ref=ElementRef.from_index(ref_idx),
            role=AriaRole(role),
            name=name,
            properties=props or {}
        )
        root.children.append(child)

    return AriaTree(root=root)


def create_snapshot(
    session_id: str,
    tree: AriaTree,
    snapshot_id: str | None = None
) -> PageSnapshot:
    """Create a PageSnapshot for testing."""
    return PageSnapshot(
        snapshot_id=SnapshotId(snapshot_id or SnapshotId.generate().value),
        session_id=session_id,
        aria_tree=tree,
        created_at=datetime.now(timezone.utc)
    )


# =============================================================================
# SnapshotDiffService Tests
# =============================================================================

class TestSnapshotDiffService:
    """Tests for SnapshotDiffService."""

    @pytest.fixture
    def diff_service(self) -> SnapshotDiffService:
        """Create a diff service instance."""
        return SnapshotDiffService()

    def test_identical_snapshots_no_changes(self, diff_service: SnapshotDiffService):
        """Test that identical snapshots produce no changes."""
        tree = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.HEADING, "Title", {"level": 1}),
            (3, AriaRole.BUTTON, "Submit", None),
        ])

        snapshot1 = create_snapshot("session1", tree, "snap_001")
        snapshot2 = create_snapshot("session1", tree, "snap_002")

        diff = diff_service.compute_diff(snapshot1, snapshot2)

        assert not diff.has_changes
        assert diff.change_count == 0
        assert len(diff.added_nodes) == 0
        assert len(diff.removed_nodes) == 0
        assert len(diff.modified_nodes) == 0
        assert diff.unchanged_count == 3

    def test_detect_added_nodes(self, diff_service: SnapshotDiffService):
        """Test detection of newly added nodes."""
        tree1 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.HEADING, "Title", None),
        ])

        tree2 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.HEADING, "Title", None),
            (3, AriaRole.BUTTON, "New Button", None),
            (4, AriaRole.LINK, "New Link", None),
        ])

        snapshot1 = create_snapshot("session1", tree1)
        snapshot2 = create_snapshot("session1", tree2)

        diff = diff_service.compute_diff(snapshot1, snapshot2)

        assert diff.has_changes
        assert len(diff.added_nodes) == 2
        assert len(diff.removed_nodes) == 0
        assert len(diff.modified_nodes) == 0

        added_refs = {n.ref.value for n in diff.added_nodes}
        assert "e3" in added_refs
        assert "e4" in added_refs

        # Verify change type
        for node in diff.added_nodes:
            assert node.change_type == ChangeType.ADDED
            assert node.new_value is not None
            assert node.old_value is None

    def test_detect_removed_nodes(self, diff_service: SnapshotDiffService):
        """Test detection of removed nodes."""
        tree1 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.HEADING, "Title", None),
            (3, AriaRole.BUTTON, "Delete Me", None),
            (4, AriaRole.LINK, "Remove Me", None),
        ])

        tree2 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.HEADING, "Title", None),
        ])

        snapshot1 = create_snapshot("session1", tree1)
        snapshot2 = create_snapshot("session1", tree2)

        diff = diff_service.compute_diff(snapshot1, snapshot2)

        assert diff.has_changes
        assert len(diff.removed_nodes) == 2
        assert len(diff.added_nodes) == 0
        assert len(diff.modified_nodes) == 0

        removed_refs = {n.ref.value for n in diff.removed_nodes}
        assert "e3" in removed_refs
        assert "e4" in removed_refs

        # Verify change type
        for node in diff.removed_nodes:
            assert node.change_type == ChangeType.REMOVED
            assert node.old_value is not None
            assert node.new_value is None

    def test_detect_modified_nodes(self, diff_service: SnapshotDiffService):
        """Test detection of modified nodes (same ref, different content)."""
        tree1 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Old Text", None),
            (3, AriaRole.CHECKBOX, "Check", {"checked": False}),
        ])

        tree2 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "New Text", None),  # Name changed
            (3, AriaRole.CHECKBOX, "Check", {"checked": True}),  # Property changed
        ])

        snapshot1 = create_snapshot("session1", tree1)
        snapshot2 = create_snapshot("session1", tree2)

        diff = diff_service.compute_diff(snapshot1, snapshot2)

        assert diff.has_changes
        assert len(diff.modified_nodes) == 2
        assert len(diff.added_nodes) == 0
        assert len(diff.removed_nodes) == 0

        modified_refs = {n.ref.value for n in diff.modified_nodes}
        assert "e2" in modified_refs
        assert "e3" in modified_refs

        # Verify change type and values
        for node in diff.modified_nodes:
            assert node.change_type == ChangeType.MODIFIED
            assert node.old_value is not None
            assert node.new_value is not None
            assert node.old_value != node.new_value

    def test_detect_mixed_changes(self, diff_service: SnapshotDiffService):
        """Test detection of mixed changes (added, removed, modified)."""
        tree1 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Keep", None),
            (3, AriaRole.BUTTON, "Modify Me", None),
            (4, AriaRole.BUTTON, "Remove Me", None),
        ])

        tree2 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Keep", None),
            (3, AriaRole.BUTTON, "Modified!", None),  # Modified
            (5, AriaRole.BUTTON, "Added", None),  # Added (e4 removed, e5 added)
        ])

        snapshot1 = create_snapshot("session1", tree1)
        snapshot2 = create_snapshot("session1", tree2)

        diff = diff_service.compute_diff(snapshot1, snapshot2)

        assert diff.has_changes
        assert len(diff.added_nodes) == 1  # e5
        assert len(diff.removed_nodes) == 1  # e4
        assert len(diff.modified_nodes) == 1  # e3
        assert diff.unchanged_count == 2  # e1, e2

    def test_diff_token_reduction(self, diff_service: SnapshotDiffService):
        """Test that diff output is smaller than full snapshot for minor changes."""
        # Create a larger tree
        nodes = [(1, AriaRole.DOCUMENT, "Page", None)]
        for i in range(2, 52):  # 50 items
            nodes.append((i, AriaRole.LISTITEM, f"Item {i}", None))

        tree1 = create_simple_tree(nodes)

        # Modify just one item
        nodes2 = [(1, AriaRole.DOCUMENT, "Page", None)]
        for i in range(2, 52):
            name = "MODIFIED Item" if i == 25 else f"Item {i}"
            nodes2.append((i, AriaRole.LISTITEM, name, None))

        tree2 = create_simple_tree(nodes2)

        snapshot1 = create_snapshot("session1", tree1)
        snapshot2 = create_snapshot("session1", tree2)

        diff = diff_service.compute_diff(snapshot1, snapshot2)

        # Full snapshot token estimate
        full_tokens = snapshot2.estimate_tokens()

        # Diff token estimate
        diff_yaml = diff.to_yaml()
        diff_tokens = len(diff_yaml) // 4

        # Diff should be significantly smaller
        assert diff_tokens < full_tokens * 0.3  # At least 70% reduction
        assert diff.change_count == 1
        assert diff.unchanged_count == 50

    def test_structural_diff_statistics(self, diff_service: SnapshotDiffService):
        """Test structural diff provides detailed statistics."""
        tree1 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Click", {"disabled": False}),
        ])

        tree2 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Submit", {"disabled": True}),  # Name + property changed
        ])

        snapshot1 = create_snapshot("session1", tree1)
        snapshot2 = create_snapshot("session1", tree2)

        diff, stats = diff_service.compute_structural_diff(snapshot1, snapshot2)

        assert stats["total_modified"] == 1
        assert stats["name_changes"] == 1
        assert stats["property_changes"] == 1
        assert stats["role_changes"] == 0

    def test_diff_to_yaml_output(self, diff_service: SnapshotDiffService):
        """Test YAML output format of diff."""
        tree1 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Old", None),
        ])

        tree2 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (3, AriaRole.LINK, "New", None),
        ])

        snapshot1 = create_snapshot("session1", tree1)
        snapshot2 = create_snapshot("session1", tree2)

        diff = diff_service.compute_diff(snapshot1, snapshot2)
        yaml_output = diff.to_yaml()

        # Check structure
        assert "# Diff:" in yaml_output
        assert "# Removed:" in yaml_output
        assert "# Added:" in yaml_output
        assert "- " in yaml_output  # Removed prefix
        assert "+ " in yaml_output  # Added prefix

    def test_empty_trees(self, diff_service: SnapshotDiffService):
        """Test diffing empty trees."""
        tree1 = create_simple_tree([])
        tree2 = create_simple_tree([])

        snapshot1 = create_snapshot("session1", tree1)
        snapshot2 = create_snapshot("session1", tree2)

        diff = diff_service.compute_diff(snapshot1, snapshot2)

        assert not diff.has_changes
        assert diff.unchanged_count == 1  # Root node


# =============================================================================
# NodeChange Tests
# =============================================================================

class TestNodeChange:
    """Tests for NodeChange dataclass."""

    def test_to_yaml_line_added(self):
        """Test YAML line format for added node."""
        change = NodeChange(
            ref=ElementRef.from_index(5),
            change_type=ChangeType.ADDED,
            new_value='button "Submit" [ref=e5]'
        )

        yaml_line = change.to_yaml_line()

        assert yaml_line.startswith("+ ")
        assert "Submit" in yaml_line

    def test_to_yaml_line_removed(self):
        """Test YAML line format for removed node."""
        change = NodeChange(
            ref=ElementRef.from_index(3),
            change_type=ChangeType.REMOVED,
            old_value='link "Old Link" [ref=e3]'
        )

        yaml_line = change.to_yaml_line()

        assert yaml_line.startswith("- ")
        assert "Old Link" in yaml_line

    def test_to_yaml_line_modified(self):
        """Test YAML line format for modified node."""
        change = NodeChange(
            ref=ElementRef.from_index(2),
            change_type=ChangeType.MODIFIED,
            old_value='button "Old" [ref=e2]',
            new_value='button "New" [ref=e2]'
        )

        yaml_line = change.to_yaml_line()

        assert yaml_line.startswith("~ ")
        assert "New" in yaml_line


# =============================================================================
# SnapshotDiff Tests
# =============================================================================

class TestSnapshotDiff:
    """Tests for SnapshotDiff dataclass."""

    def test_has_changes_true(self):
        """Test has_changes returns True when there are changes."""
        diff = SnapshotDiff(
            added_nodes=[
                NodeChange(ElementRef.from_index(1), ChangeType.ADDED, new_value="x")
            ],
            unchanged_count=5
        )

        assert diff.has_changes

    def test_has_changes_false(self):
        """Test has_changes returns False when no changes."""
        diff = SnapshotDiff(unchanged_count=10)

        assert not diff.has_changes

    def test_change_count(self):
        """Test change_count sums all change types."""
        diff = SnapshotDiff(
            added_nodes=[NodeChange(ElementRef.from_index(1), ChangeType.ADDED)],
            removed_nodes=[
                NodeChange(ElementRef.from_index(2), ChangeType.REMOVED),
                NodeChange(ElementRef.from_index(3), ChangeType.REMOVED),
            ],
            modified_nodes=[NodeChange(ElementRef.from_index(4), ChangeType.MODIFIED)],
            unchanged_count=10
        )

        assert diff.change_count == 4

    def test_change_ratio(self):
        """Test change_ratio calculation."""
        diff = SnapshotDiff(
            added_nodes=[NodeChange(ElementRef.from_index(1), ChangeType.ADDED)],
            unchanged_count=9
        )

        assert diff.change_ratio == 0.1  # 1 changed / 10 total

    def test_to_compact_yaml_no_changes(self):
        """Test compact YAML output with no changes."""
        diff = SnapshotDiff(unchanged_count=5)

        assert diff.to_compact_yaml() == "[No changes]"

    def test_estimate_token_savings(self):
        """Test token savings estimation."""
        diff = SnapshotDiff(
            added_nodes=[NodeChange(ElementRef.from_index(1), ChangeType.ADDED, new_value="x")],
            unchanged_count=100
        )

        savings = diff.estimate_token_savings(full_snapshot_tokens=1000)

        # Diff is much smaller than full snapshot
        assert savings > 0


# =============================================================================
# SnapshotManager Tests
# =============================================================================

class TestSnapshotManager:
    """Tests for SnapshotManager."""

    @pytest.fixture
    def manager(self) -> SnapshotManager:
        """Create a snapshot manager instance."""
        return SnapshotManager(max_cache_per_session=5)

    def test_first_snapshot_returns_full(self, manager: SnapshotManager):
        """Test that first snapshot returns full YAML."""
        tree = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Click", None),
        ])
        snapshot = create_snapshot("session1", tree)

        result = manager.capture_and_diff("session1", snapshot, mode="incremental")

        # First snapshot should return full content
        assert "document" in result.lower()
        assert "button" in result.lower()
        assert "Click" in result
        assert "# Diff:" not in result  # Not a diff

    def test_subsequent_snapshot_returns_diff(self, manager: SnapshotManager):
        """Test that subsequent snapshots return diffs."""
        tree1 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Click", None),
        ])
        snapshot1 = create_snapshot("session1", tree1)

        tree2 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Click", None),
            (3, AriaRole.LINK, "New Link", None),
        ])
        snapshot2 = create_snapshot("session1", tree2)

        # First capture
        manager.capture_and_diff("session1", snapshot1, mode="incremental")

        # Second capture should return diff
        result = manager.capture_and_diff("session1", snapshot2, mode="incremental")

        assert "# Diff:" in result or "# Added:" in result
        assert "+ " in result  # Added node prefix

    def test_no_changes_message(self, manager: SnapshotManager):
        """Test message when no changes detected."""
        tree = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
        ])
        snapshot1 = create_snapshot("session1", tree, "snap1")
        snapshot2 = create_snapshot("session1", tree, "snap2")

        manager.capture_and_diff("session1", snapshot1, mode="incremental")
        result = manager.capture_and_diff("session1", snapshot2, mode="incremental")

        assert "No changes" in result

    def test_full_mode_always_returns_full(self, manager: SnapshotManager):
        """Test that full mode always returns full snapshot."""
        tree = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
        ])
        snapshot1 = create_snapshot("session1", tree)
        snapshot2 = create_snapshot("session1", tree)

        manager.capture_and_diff("session1", snapshot1, mode="full")
        result = manager.capture_and_diff("session1", snapshot2, mode="full")

        # Should return full content, not diff
        assert "# Diff:" not in result
        assert "document" in result.lower()

    def test_none_mode_returns_empty(self, manager: SnapshotManager):
        """Test that none mode returns empty string."""
        tree = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
        ])
        snapshot = create_snapshot("session1", tree)

        result = manager.capture_and_diff("session1", snapshot, mode="none")

        assert result == ""

    def test_cache_eviction(self, manager: SnapshotManager):
        """Test LRU cache eviction."""
        # Manager has max_cache=5

        for i in range(7):
            tree = create_simple_tree([
                (1, AriaRole.DOCUMENT, f"Page {i}", None),
            ])
            snapshot = create_snapshot("session1", tree, f"snap_{i}")
            manager.capture_and_diff("session1", snapshot)

        # Should only have 5 snapshots cached
        assert manager.get_snapshot_count("session1") == 5

        # Oldest should be evicted (snap_0 and snap_1)
        history = manager.get_snapshot_history("session1")
        snapshot_ids = [str(s.snapshot_id) for s in history]
        assert "snap_0" not in snapshot_ids
        assert "snap_1" not in snapshot_ids
        assert "snap_6" in snapshot_ids

    def test_clear_session_cache(self, manager: SnapshotManager):
        """Test clearing session cache."""
        tree = create_simple_tree([(1, AriaRole.DOCUMENT, "Page", None)])
        snapshot = create_snapshot("session1", tree)

        manager.capture_and_diff("session1", snapshot)
        assert manager.get_snapshot_count("session1") == 1

        manager.clear_session_cache("session1")
        assert manager.get_snapshot_count("session1") == 0

    def test_clear_all_caches(self, manager: SnapshotManager):
        """Test clearing all caches."""
        tree = create_simple_tree([(1, AriaRole.DOCUMENT, "Page", None)])

        for session in ["session1", "session2", "session3"]:
            snapshot = create_snapshot(session, tree)
            manager.capture_and_diff(session, snapshot)

        assert len(manager.get_all_session_ids()) == 3

        manager.clear_all_caches()
        assert len(manager.get_all_session_ids()) == 0

    def test_get_previous_snapshot(self, manager: SnapshotManager):
        """Test getting previous snapshot."""
        tree = create_simple_tree([(1, AriaRole.DOCUMENT, "Page", None)])

        # No history yet
        assert manager.get_previous_snapshot("session1") is None

        snapshot1 = create_snapshot("session1", tree, "snap_1")
        manager.capture_and_diff("session1", snapshot1)

        # After first capture
        prev = manager.get_previous_snapshot("session1")
        assert prev is not None
        assert str(prev.snapshot_id) == "snap_1"

    def test_multiple_sessions_isolation(self, manager: SnapshotManager):
        """Test that different sessions maintain separate caches."""
        tree1 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Session1 Page", None),
        ])
        tree2 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Session2 Page", None),
        ])

        snapshot1 = create_snapshot("session1", tree1)
        snapshot2 = create_snapshot("session2", tree2)

        manager.capture_and_diff("session1", snapshot1)
        manager.capture_and_diff("session2", snapshot2)

        # Each session has its own cache
        prev1 = manager.get_previous_snapshot("session1")
        prev2 = manager.get_previous_snapshot("session2")

        assert prev1 is not None
        assert prev2 is not None
        assert prev1.aria_tree.root.name == "Session1 Page"
        assert prev2.aria_tree.root.name == "Session2 Page"

    def test_compute_diff_between(self, manager: SnapshotManager):
        """Test computing diff between specific cached snapshots."""
        for i in range(3):
            tree = create_simple_tree([
                (1, AriaRole.DOCUMENT, f"Page v{i}", None),
            ])
            snapshot = create_snapshot("session1", tree)
            manager.capture_and_diff("session1", snapshot)

        # Diff between first and last
        diff = manager.compute_diff_between("session1", 0, 2)

        assert diff is not None
        assert diff.has_changes
        assert len(diff.modified_nodes) == 1

    def test_compute_diff_between_invalid_indices(self, manager: SnapshotManager):
        """Test computing diff with invalid indices."""
        tree = create_simple_tree([(1, AriaRole.DOCUMENT, "Page", None)])
        snapshot = create_snapshot("session1", tree)
        manager.capture_and_diff("session1", snapshot)

        # Invalid indices
        assert manager.compute_diff_between("session1", 0, 5) is None
        assert manager.compute_diff_between("session1", -1, 0) is None
        assert manager.compute_diff_between("nonexistent", 0, 0) is None

    def test_capture_and_diff_detailed(self, manager: SnapshotManager):
        """Test detailed capture returning diff object."""
        tree1 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Click", None),
        ])
        tree2 = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Page", None),
            (2, AriaRole.BUTTON, "Submit", None),  # Changed
        ])

        snapshot1 = create_snapshot("session1", tree1)
        snapshot2 = create_snapshot("session1", tree2)

        # First capture
        yaml1, diff1 = manager.capture_and_diff_detailed("session1", snapshot1)
        assert diff1 is None  # First snapshot has no diff

        # Second capture
        yaml2, diff2 = manager.capture_and_diff_detailed("session1", snapshot2)
        assert diff2 is not None
        assert diff2.has_changes
        assert len(diff2.modified_nodes) == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestDiffServiceIntegration:
    """Integration tests for diff service with realistic scenarios."""

    def test_realistic_page_update_scenario(self):
        """Test a realistic scenario of updating page content."""
        manager = SnapshotManager()

        # Initial page load - product listing
        initial_tree = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Products", None),
            (2, AriaRole.HEADING, "Our Products", {"level": 1}),
            (3, AriaRole.LISTITEM, "Product A - $10", None),
            (4, AriaRole.LISTITEM, "Product B - $20", None),
            (5, AriaRole.LISTITEM, "Product C - $30", None),
            (6, AriaRole.BUTTON, "Add to Cart", None),
        ])
        snapshot1 = create_snapshot("user-session", initial_tree)

        result1 = manager.capture_and_diff("user-session", snapshot1)
        assert "Products" in result1  # Full snapshot

        # User adds item to cart - cart counter updates
        updated_tree = create_simple_tree([
            (1, AriaRole.DOCUMENT, "Products", None),
            (2, AriaRole.HEADING, "Our Products", {"level": 1}),
            (3, AriaRole.LISTITEM, "Product A - $10", None),
            (4, AriaRole.LISTITEM, "Product B - $20", None),
            (5, AriaRole.LISTITEM, "Product C - $30", None),
            (6, AriaRole.BUTTON, "Add to Cart (1)", None),  # Changed
            (7, AriaRole.ALERT, "Added to cart!", None),  # New
        ])
        snapshot2 = create_snapshot("user-session", updated_tree)

        result2 = manager.capture_and_diff("user-session", snapshot2)

        # Should be a diff with only the changes
        assert "# Diff:" in result2
        assert "+ " in result2  # Added alert
        assert "~ " in result2  # Modified button

    def test_token_efficiency_large_page(self):
        """Test token efficiency on a large page with minor changes."""
        manager = SnapshotManager()

        # Create large page with 100 items
        large_nodes = [(1, AriaRole.DOCUMENT, "Large Page", None)]
        for i in range(2, 102):
            large_nodes.append((i, AriaRole.LISTITEM, f"Item {i}", None))

        tree1 = create_simple_tree(large_nodes)
        snapshot1 = create_snapshot("session", tree1)
        manager.capture_and_diff("session", snapshot1)

        # Modify just 2 items
        large_nodes_modified = [(1, AriaRole.DOCUMENT, "Large Page", None)]
        for i in range(2, 102):
            if i == 50:
                large_nodes_modified.append((i, AriaRole.LISTITEM, "SELECTED Item 50", None))
            elif i == 51:
                large_nodes_modified.append((i, AriaRole.LISTITEM, "SELECTED Item 51", None))
            else:
                large_nodes_modified.append((i, AriaRole.LISTITEM, f"Item {i}", None))

        tree2 = create_simple_tree(large_nodes_modified)
        snapshot2 = create_snapshot("session", tree2)

        # Get diff
        result, diff = manager.capture_and_diff_detailed("session", snapshot2)

        # Calculate token savings
        full_tokens = snapshot2.estimate_tokens()
        diff_tokens = len(result) // 4

        # Should achieve significant reduction
        reduction = (full_tokens - diff_tokens) / full_tokens
        assert reduction > 0.7  # At least 70% reduction
        assert diff is not None
        assert diff.change_count == 2
        assert diff.unchanged_count == 99
