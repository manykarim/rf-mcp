"""Tests for Snapshot Context bounded context.

This module tests the core snapshot functionality for token optimization:
- SnapshotId value object (unique, immutable identifiers)
- ElementRef value object (short element references)
- AriaNode entity (accessibility tree nodes)
- AriaTree aggregate (full accessibility tree operations)
- PageSnapshot aggregate (page snapshots with optimization)
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Set

import pytest


# =============================================================================
# Domain Models (to be moved to production code)
# =============================================================================


@dataclass(frozen=True)
class SnapshotId:
    """Value object representing a unique snapshot identifier.

    Immutable by design to ensure snapshot references remain stable.
    """

    value: str

    @classmethod
    def generate(cls) -> "SnapshotId":
        """Generate a new unique snapshot ID."""
        return cls(value=str(uuid.uuid4())[:8])

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ElementRef:
    """Value object for short element references.

    Format: 'e' followed by a number (e.g., e1, e2, e42)
    """

    index: int

    @classmethod
    def from_index(cls, index: int) -> "ElementRef":
        """Create an ElementRef from a numeric index."""
        if index < 0:
            raise ValueError("Element index must be non-negative")
        return cls(index=index)

    @classmethod
    def parse(cls, ref_string: str) -> "ElementRef":
        """Parse a ref string like 'e42' into an ElementRef."""
        if not ref_string.startswith("e"):
            raise ValueError(f"Invalid ref format: {ref_string}")
        try:
            index = int(ref_string[1:])
            if index < 0:
                raise ValueError(f"Invalid ref format: {ref_string}")
            return cls(index=index)
        except ValueError:
            raise ValueError(f"Invalid ref format: {ref_string}")

    @property
    def ref(self) -> str:
        """Return the string representation (e.g., 'e42')."""
        return f"e{self.index}"

    def __str__(self) -> str:
        return self.ref


@dataclass
class AriaNode:
    """Entity representing a node in the accessibility tree.

    Each node has a role, optional name, and may contain children.
    """

    role: str
    ref: ElementRef
    name: str = ""
    level: Optional[int] = None  # For headings
    value: Optional[str] = None  # For inputs
    children: List["AriaNode"] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        role: str,
        index: int,
        name: str = "",
        level: Optional[int] = None,
        children: Optional[List["AriaNode"]] = None,
    ) -> "AriaNode":
        """Factory method to create an AriaNode."""
        return cls(
            role=role,
            ref=ElementRef.from_index(index),
            name=name,
            level=level,
            children=children or [],
        )

    def to_yaml_line(self, indent: int = 0) -> str:
        """Convert node to YAML-like representation."""
        prefix = "  " * indent + "- "
        parts = [f'{self.role}']
        if self.name:
            parts.append(f'"{self.name}"')
        if self.level:
            parts.append(f"[level={self.level}]")
        parts.append(f"[ref={self.ref}]")
        return prefix + " ".join(parts)


class AriaTree:
    """Aggregate for the complete accessibility tree.

    Provides traversal, search, and filtering operations.
    """

    def __init__(self, root: AriaNode):
        self._root = root
        self._ref_index: Dict[str, AriaNode] = {}
        self._build_index(root)

    def _build_index(self, node: AriaNode) -> None:
        """Build an index of refs to nodes for fast lookup."""
        self._ref_index[str(node.ref)] = node
        for child in node.children:
            self._build_index(child)

    @property
    def root(self) -> AriaNode:
        return self._root

    def traverse(self) -> Generator[AriaNode, None, None]:
        """Traverse all nodes in depth-first order."""

        def _traverse(node: AriaNode) -> Generator[AriaNode, None, None]:
            yield node
            for child in node.children:
                yield from _traverse(child)

        yield from _traverse(self._root)

    def find_by_ref(self, ref: str) -> Optional[AriaNode]:
        """Find a node by its ref string."""
        return self._ref_index.get(ref)

    def filter_interactive_only(self) -> List[AriaNode]:
        """Return only interactive elements (buttons, links, inputs, etc.)."""
        interactive_roles = {
            "button",
            "link",
            "textbox",
            "checkbox",
            "radio",
            "combobox",
            "menuitem",
            "tab",
            "slider",
            "spinbutton",
            "switch",
        }
        return [
            node for node in self.traverse() if node.role.lower() in interactive_roles
        ]

    def count_nodes(self) -> int:
        """Count total nodes in the tree."""
        return len(self._ref_index)


class PageSnapshot:
    """Aggregate for page snapshots with optimization capabilities.

    Handles snapshot capture, YAML conversion, token estimation,
    incremental diffs, and list folding.
    """

    def __init__(
        self,
        snapshot_id: SnapshotId,
        aria_tree: AriaTree,
        url: str = "",
        title: str = "",
    ):
        self._id = snapshot_id
        self._tree = aria_tree
        self._url = url
        self._title = title
        self._yaml_cache: Optional[str] = None

    @property
    def id(self) -> SnapshotId:
        return self._id

    @property
    def tree(self) -> AriaTree:
        return self._tree

    @classmethod
    def capture(cls, tree_data: List[Dict], url: str = "", title: str = "") -> "PageSnapshot":
        """Factory method to capture a snapshot from raw tree data."""
        # Convert dict data to AriaNode structure
        def dict_to_node(data: Dict, index_counter: List[int]) -> AriaNode:
            current_index = index_counter[0]
            index_counter[0] += 1
            children = [
                dict_to_node(child, index_counter)
                for child in data.get("children", [])
            ]
            return AriaNode(
                role=data.get("role", "unknown"),
                ref=ElementRef.from_index(current_index),
                name=data.get("name", ""),
                level=data.get("level"),
                children=children,
            )

        if not tree_data:
            root = AriaNode.create("document", 0)
        else:
            counter = [1]
            root = dict_to_node(tree_data[0], counter)

        return cls(
            snapshot_id=SnapshotId.generate(),
            aria_tree=AriaTree(root),
            url=url,
            title=title,
        )

    def to_yaml(self) -> str:
        """Convert the snapshot to YAML format."""
        if self._yaml_cache:
            return self._yaml_cache

        lines = []

        def node_to_lines(node: AriaNode, indent: int = 0) -> None:
            lines.append(node.to_yaml_line(indent))
            for child in node.children:
                node_to_lines(child, indent + 1)

        node_to_lines(self._tree.root)
        self._yaml_cache = "\n".join(lines)
        return self._yaml_cache

    def estimate_tokens(self) -> int:
        """Estimate token count for this snapshot.

        Uses ~4 characters per token as a rough estimate.
        """
        yaml_str = self.to_yaml()
        return len(yaml_str) // 4

    def get_incremental_diff(self, previous: "PageSnapshot") -> str:
        """Get only the changes from a previous snapshot.

        Returns a diff representation showing added/removed/changed nodes.
        """
        current_refs = set(str(n.ref) for n in self._tree.traverse())
        previous_refs = set(str(n.ref) for n in previous._tree.traverse())

        added = current_refs - previous_refs
        removed = previous_refs - current_refs
        # Check for modified nodes (same ref, different content)
        common = current_refs & previous_refs
        modified = set()
        for ref in common:
            current_node = self._tree.find_by_ref(ref)
            prev_node = previous._tree.find_by_ref(ref)
            if current_node and prev_node:
                if current_node.name != prev_node.name:
                    modified.add(ref)

        if not added and not removed and not modified:
            return "[No changes detected]"

        lines = []
        if added:
            lines.append(f"[ADDED] refs: {', '.join(sorted(added))}")
            for ref in sorted(added):
                node = self._tree.find_by_ref(ref)
                if node:
                    lines.append(f"  + {node.role} \"{node.name}\" [{ref}]")

        if removed:
            lines.append(f"[REMOVED] refs: {', '.join(sorted(removed))}")

        if modified:
            lines.append(f"[MODIFIED] refs: {', '.join(sorted(modified))}")
            for ref in sorted(modified):
                node = self._tree.find_by_ref(ref)
                if node:
                    lines.append(f"  ~ {node.role} \"{node.name}\" [{ref}]")

        return "\n".join(lines)

    def fold_lists(self, similarity_threshold: float = 0.85, min_items: int = 3) -> str:
        """Compress similar list items into folded representation.

        Returns YAML with repetitive listitems collapsed:
        - listitem "Product 1" [ref=e10]
        - (... and 47 more similar) [refs: e11-e57]
        """
        yaml_output = []

        def process_node(node: AriaNode, indent: int = 0) -> None:
            prefix = "  " * indent + "- "

            # Check if this is a list with many similar items
            if node.role == "list" and len(node.children) >= min_items:
                listitems = [c for c in node.children if c.role == "listitem"]
                if len(listitems) >= min_items:
                    # Check similarity using SimHash-like approach
                    if self._are_items_similar(listitems, similarity_threshold):
                        # Fold the list
                        yaml_output.append(f"{prefix}{node.role} [ref={node.ref}]")
                        first_item = listitems[0]
                        yaml_output.append(
                            f"{'  ' * (indent + 1)}- {first_item.role} "
                            f'"{first_item.name}" [ref={first_item.ref}]'
                        )
                        remaining = len(listitems) - 1
                        first_ref = listitems[1].ref
                        last_ref = listitems[-1].ref
                        yaml_output.append(
                            f"{'  ' * (indent + 1)}- "
                            f"(... and {remaining} more similar) "
                            f"[refs: {first_ref}-{last_ref}]"
                        )
                        return

            # Normal processing
            yaml_output.append(node.to_yaml_line(indent))
            for child in node.children:
                process_node(child, indent + 1)

        process_node(self._tree.root)
        return "\n".join(yaml_output)

    def _are_items_similar(
        self, items: List[AriaNode], threshold: float
    ) -> bool:
        """Check if list items are similar enough to fold.

        Uses a simplified SimHash-like comparison.
        """
        if len(items) < 2:
            return False

        # Extract patterns (remove numbers, keep structure)
        def get_pattern(name: str) -> str:
            return re.sub(r"\d+", "N", name)

        patterns = [get_pattern(item.name) for item in items]
        # Check if most patterns are the same
        most_common = max(set(patterns), key=patterns.count)
        similarity = patterns.count(most_common) / len(patterns)
        return similarity >= threshold


# =============================================================================
# Tests
# =============================================================================


class TestSnapshotId:
    """Tests for SnapshotId value object."""

    def test_generate_creates_unique_id(self):
        """Test that generate() creates unique IDs."""
        ids = [SnapshotId.generate() for _ in range(100)]
        unique_values = set(str(sid) for sid in ids)
        # All 100 should be unique
        assert len(unique_values) == 100

    def test_snapshot_id_is_immutable(self):
        """Test that SnapshotId cannot be modified after creation."""
        snapshot_id = SnapshotId(value="test123")

        # frozen=True should prevent attribute modification
        with pytest.raises((AttributeError, TypeError)):
            snapshot_id.value = "modified"

    def test_snapshot_id_string_representation(self):
        """Test string conversion of SnapshotId."""
        snapshot_id = SnapshotId(value="abc123")
        assert str(snapshot_id) == "abc123"

    def test_snapshot_id_equality(self):
        """Test that SnapshotIds with same value are equal."""
        id1 = SnapshotId(value="test")
        id2 = SnapshotId(value="test")
        id3 = SnapshotId(value="other")

        assert id1 == id2
        assert id1 != id3

    def test_snapshot_id_hashable(self):
        """Test that SnapshotId can be used in sets/dicts."""
        id1 = SnapshotId(value="test")
        id2 = SnapshotId(value="test")

        # Should be hashable and work in sets
        id_set = {id1, id2}
        assert len(id_set) == 1


class TestElementRef:
    """Tests for ElementRef value object."""

    def test_from_index_creates_valid_ref(self):
        """Test creating ElementRef from numeric index."""
        ref = ElementRef.from_index(42)
        assert ref.index == 42
        assert ref.ref == "e42"

    def test_ref_format_e_followed_by_number(self):
        """Test that ref format is 'e' followed by a number."""
        for i in [0, 1, 10, 100, 9999]:
            ref = ElementRef.from_index(i)
            assert ref.ref == f"e{i}"
            assert ref.ref.startswith("e")
            assert ref.ref[1:].isdigit()

    def test_from_index_rejects_negative(self):
        """Test that negative indices are rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            ElementRef.from_index(-1)

    def test_parse_valid_ref_string(self):
        """Test parsing valid ref strings."""
        ref = ElementRef.parse("e42")
        assert ref.index == 42
        assert ref.ref == "e42"

    def test_parse_invalid_ref_string(self):
        """Test that invalid ref strings are rejected."""
        # These don't start with 'e'
        for invalid in ["42", "ref42", "E42"]:
            with pytest.raises(ValueError, match="Invalid ref format"):
                ElementRef.parse(invalid)

        # These start with 'e' but have invalid number part
        for invalid in ["e", "e-1", "e abc"]:
            with pytest.raises(ValueError, match="Invalid ref format"):
                ElementRef.parse(invalid)

    def test_element_ref_is_immutable(self):
        """Test that ElementRef is immutable (frozen dataclass)."""
        ref = ElementRef.from_index(1)
        with pytest.raises((AttributeError, TypeError)):
            ref.index = 2

    def test_element_ref_equality(self):
        """Test ElementRef equality comparison."""
        ref1 = ElementRef.from_index(5)
        ref2 = ElementRef.from_index(5)
        ref3 = ElementRef.from_index(6)

        assert ref1 == ref2
        assert ref1 != ref3


class TestAriaNode:
    """Tests for AriaNode entity."""

    def test_create_node_with_required_fields(self):
        """Test creating a node with required fields."""
        node = AriaNode.create(role="button", index=1, name="Submit")

        assert node.role == "button"
        assert node.name == "Submit"
        assert str(node.ref) == "e1"
        assert node.children == []

    def test_create_node_with_level(self):
        """Test creating a heading node with level."""
        node = AriaNode.create(role="heading", index=2, name="Title", level=1)

        assert node.role == "heading"
        assert node.level == 1

    def test_node_children_hierarchy(self):
        """Test that nodes can have children forming a hierarchy."""
        child1 = AriaNode.create(role="listitem", index=2, name="Item 1")
        child2 = AriaNode.create(role="listitem", index=3, name="Item 2")
        parent = AriaNode.create(
            role="list", index=1, name="", children=[child1, child2]
        )

        assert len(parent.children) == 2
        assert parent.children[0].name == "Item 1"
        assert parent.children[1].name == "Item 2"

    def test_to_yaml_line_basic(self):
        """Test YAML line generation for basic node."""
        node = AriaNode.create(role="button", index=5, name="Click Me")
        yaml_line = node.to_yaml_line()

        assert "button" in yaml_line
        assert '"Click Me"' in yaml_line
        assert "[ref=e5]" in yaml_line

    def test_to_yaml_line_with_level(self):
        """Test YAML line generation for heading with level."""
        node = AriaNode.create(role="heading", index=2, name="Title", level=1)
        yaml_line = node.to_yaml_line()

        assert "[level=1]" in yaml_line

    def test_to_yaml_line_with_indent(self):
        """Test YAML line generation with indentation."""
        node = AriaNode.create(role="listitem", index=3, name="Item")
        yaml_line = node.to_yaml_line(indent=2)

        assert yaml_line.startswith("    - ")  # 2 levels of indent


class TestAriaTree:
    """Tests for AriaTree aggregate."""

    @pytest.fixture
    def sample_tree(self) -> AriaTree:
        """Create a sample tree for testing."""
        list_items = [
            AriaNode.create(role="listitem", index=i, name=f"Item {i-3}")
            for i in range(4, 7)
        ]
        children = [
            AriaNode.create(role="heading", index=2, name="Page Title", level=1),
            AriaNode.create(role="button", index=3, name="Submit"),
            AriaNode.create(role="list", index=4, children=list_items),
        ]
        # Renumber the list node properly
        list_node = AriaNode.create(role="list", index=8, children=[
            AriaNode.create(role="listitem", index=5, name="Item 1"),
            AriaNode.create(role="listitem", index=6, name="Item 2"),
            AriaNode.create(role="listitem", index=7, name="Item 3"),
        ])
        root = AriaNode.create(
            role="document",
            index=1,
            children=[
                AriaNode.create(role="heading", index=2, name="Page Title", level=1),
                AriaNode.create(role="button", index=3, name="Submit"),
                AriaNode.create(role="link", index=4, name="Click here"),
                list_node,
            ],
        )
        return AriaTree(root)

    def test_traverse_visits_all_nodes(self, sample_tree):
        """Test that traverse visits every node in the tree."""
        visited = list(sample_tree.traverse())
        # Root + heading + button + link + list + 3 listitems = 8 nodes
        assert len(visited) == 8

        # Check that all roles are present
        roles = [n.role for n in visited]
        assert "document" in roles
        assert "heading" in roles
        assert "button" in roles
        assert "link" in roles
        assert "list" in roles
        assert roles.count("listitem") == 3

    def test_find_by_ref_returns_correct_node(self, sample_tree):
        """Test finding nodes by ref."""
        node = sample_tree.find_by_ref("e3")
        assert node is not None
        assert node.role == "button"
        assert node.name == "Submit"

    def test_find_by_ref_returns_none_for_invalid(self, sample_tree):
        """Test that invalid refs return None."""
        node = sample_tree.find_by_ref("e999")
        assert node is None

    def test_filter_interactive_only_removes_decorative(self, sample_tree):
        """Test filtering to only interactive elements."""
        interactive = sample_tree.filter_interactive_only()

        # Should include button and link, but not document, heading, list, listitem
        roles = [n.role for n in interactive]
        assert "button" in roles
        assert "link" in roles
        assert "document" not in roles
        assert "heading" not in roles
        assert "list" not in roles

    def test_count_nodes(self, sample_tree):
        """Test node counting."""
        count = sample_tree.count_nodes()
        assert count == 8


class TestPageSnapshot:
    """Tests for PageSnapshot aggregate."""

    @pytest.fixture
    def sample_snapshot(self) -> PageSnapshot:
        """Create a sample snapshot for testing."""
        root = AriaNode.create(
            role="document",
            index=1,
            children=[
                AriaNode.create(role="heading", index=2, name="Example", level=1),
                AriaNode.create(role="paragraph", index=3, name="Some text"),
                AriaNode.create(role="button", index=4, name="Submit"),
            ],
        )
        return PageSnapshot(
            snapshot_id=SnapshotId(value="test123"),
            aria_tree=AriaTree(root),
            url="https://example.com",
            title="Example Page",
        )

    def test_capture_creates_aria_tree(self, sample_aria_nodes):
        """Test that capture creates an AriaTree from dict data."""
        snapshot = PageSnapshot.capture(sample_aria_nodes, url="https://example.com")

        assert snapshot.tree is not None
        assert snapshot.tree.root.role == "document"
        assert snapshot.id is not None

    def test_to_yaml_produces_valid_output(self, sample_snapshot):
        """Test YAML output generation."""
        yaml = sample_snapshot.to_yaml()

        assert "document" in yaml
        assert "[ref=e1]" in yaml
        assert 'heading "Example"' in yaml
        assert "[level=1]" in yaml
        assert 'button "Submit"' in yaml

    def test_estimate_tokens_returns_reasonable_value(self, sample_snapshot):
        """Test token estimation."""
        tokens = sample_snapshot.estimate_tokens()

        # Should be positive and reasonable for the small snapshot
        assert tokens > 0
        assert tokens < 1000  # Small snapshot shouldn't have many tokens

    def test_get_incremental_diff_detects_changes(self):
        """Test incremental diff detection."""
        # Create two snapshots with differences
        root1 = AriaNode.create(
            role="document",
            index=1,
            children=[
                AriaNode.create(role="paragraph", index=2, name="Original text"),
            ],
        )
        snapshot1 = PageSnapshot(
            snapshot_id=SnapshotId(value="v1"),
            aria_tree=AriaTree(root1),
        )

        root2 = AriaNode.create(
            role="document",
            index=1,
            children=[
                AriaNode.create(role="paragraph", index=2, name="Modified text"),
                AriaNode.create(role="button", index=3, name="New button"),
            ],
        )
        snapshot2 = PageSnapshot(
            snapshot_id=SnapshotId(value="v2"),
            aria_tree=AriaTree(root2),
        )

        diff = snapshot2.get_incremental_diff(snapshot1)

        assert "[ADDED]" in diff
        assert "e3" in diff
        assert "[MODIFIED]" in diff
        assert "e2" in diff

    def test_get_incremental_diff_no_changes(self, sample_snapshot):
        """Test diff when there are no changes."""
        diff = sample_snapshot.get_incremental_diff(sample_snapshot)
        assert diff == "[No changes detected]"

    def test_fold_lists_compresses_similar_items(self):
        """Test list folding for similar items."""
        # Create a list with many similar items
        items = [
            AriaNode.create(role="listitem", index=i, name=f"Product {i}")
            for i in range(10, 60)
        ]
        root = AriaNode.create(
            role="document",
            index=1,
            children=[
                AriaNode.create(role="list", index=2, children=items),
            ],
        )
        snapshot = PageSnapshot(
            snapshot_id=SnapshotId(value="test"),
            aria_tree=AriaTree(root),
        )

        folded = snapshot.fold_lists(similarity_threshold=0.85, min_items=3)

        # Should contain folding indicator
        assert "more similar" in folded
        assert "refs:" in folded
        # Should not list all 50 items individually
        assert folded.count("listitem") < 50

    def test_fold_lists_preserves_non_list_content(self, sample_snapshot):
        """Test that folding preserves non-list content."""
        folded = sample_snapshot.fold_lists()

        # Should still contain all original elements
        assert "document" in folded
        assert "heading" in folded
        assert "button" in folded


class TestSnapshotCaching:
    """Tests for snapshot caching behavior."""

    def test_yaml_is_cached(self):
        """Test that YAML output is cached after first call."""
        root = AriaNode.create(role="document", index=1)
        snapshot = PageSnapshot(
            snapshot_id=SnapshotId(value="test"),
            aria_tree=AriaTree(root),
        )

        # First call computes
        yaml1 = snapshot.to_yaml()
        # Second call should return cached
        yaml2 = snapshot.to_yaml()

        assert yaml1 == yaml2
        assert snapshot._yaml_cache is not None


class TestSnapshotEdgeCases:
    """Tests for edge cases in snapshot handling."""

    def test_empty_tree_data(self):
        """Test handling of empty tree data."""
        snapshot = PageSnapshot.capture([], url="https://example.com")

        assert snapshot.tree is not None
        assert snapshot.tree.root.role == "document"

    def test_deeply_nested_tree(self):
        """Test handling of deeply nested trees."""
        # Create a deeply nested structure
        # Start with leaf at index 100, then nest divs 99->1, then root at 0
        current = AriaNode.create(role="div", index=100, name="leaf")
        for i in range(99, 0, -1):
            current = AriaNode.create(role="div", index=i, children=[current])
        root = AriaNode.create(role="document", index=0, children=[current])

        tree = AriaTree(root)

        # Should be able to traverse without stack overflow
        nodes = list(tree.traverse())
        # document (1) + 99 nested divs (range 99->1 = 99 items) + leaf (1) = 101
        assert len(nodes) == 101

    def test_node_with_special_characters_in_name(self):
        """Test nodes with special characters in names."""
        node = AriaNode.create(
            role="button",
            index=1,
            name='Click "here" to <submit> & continue',
        )
        yaml_line = node.to_yaml_line()

        # Should handle quotes in the output
        assert "button" in yaml_line
        assert "[ref=e1]" in yaml_line
