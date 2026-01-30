"""Shared Kernel - Core domain types shared across bounded contexts.

These types are intentionally minimal and shared between:
- Snapshot Context (produces ElementRef, AriaNode, AriaRole)
- Element Registry Context (consumes these for registration)

As per ADR-001, this shared kernel is kept minimal to reduce coupling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ElementRef:
    """Short reference to an element (e1, e2, etc.).

    Element references provide a compact way to identify elements
    in the accessibility tree, reducing token consumption compared
    to verbose XPath or CSS selectors.

    Format: "e{number}" where number is 1-10 digits (e.g., e1, e42, e9999)

    Security: Ref format is validated to prevent injection attacks.
    """
    value: str

    # Regex pattern for valid ref format: e followed by 1-10 digits
    REF_PATTERN: str = field(default=r"^e\d{1,10}$", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate ref format on creation."""
        if not re.match(self.REF_PATTERN, self.value):
            raise ValueError(
                f"Invalid ElementRef format: '{self.value}'. "
                f"Must match pattern 'e{{number}}' (e.g., e1, e42)"
            )

    @classmethod
    def from_index(cls, index: int) -> "ElementRef":
        """Create an ElementRef from a numeric index.

        Args:
            index: The element index (must be non-negative)

        Returns:
            ElementRef with value "e{index}"

        Raises:
            ValueError: If index is negative or exceeds max value
        """
        if index < 0:
            raise ValueError(f"Element index must be non-negative, got {index}")
        if index > 9999999999:  # 10 digits max
            raise ValueError(f"Element index exceeds maximum value: {index}")
        return cls(value=f"e{index}")

    def to_index(self) -> int:
        """Extract the numeric index from the ref.

        Returns:
            The numeric portion of the ref (e.g., e42 -> 42)
        """
        return int(self.value[1:])

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(frozen=True)
class AriaRole:
    """ARIA role value object.

    Represents an ARIA role attribute value, which describes the type
    of UI element for accessibility purposes.
    """
    value: str

    def __post_init__(self) -> None:
        """Validate role is non-empty."""
        if not self.value or not self.value.strip():
            raise ValueError("AriaRole value cannot be empty")

    # Common role constants
    BUTTON: str = "button"
    LINK: str = "link"
    HEADING: str = "heading"
    TEXTBOX: str = "textbox"
    CHECKBOX: str = "checkbox"
    RADIO: str = "radio"
    LISTITEM: str = "listitem"
    LIST: str = "list"
    MENU: str = "menu"
    MENUITEM: str = "menuitem"
    NAVIGATION: str = "navigation"
    MAIN: str = "main"
    BANNER: str = "banner"
    CONTENTINFO: str = "contentinfo"
    COMPLEMENTARY: str = "complementary"
    SEARCH: str = "search"
    FORM: str = "form"
    REGION: str = "region"
    ALERT: str = "alert"
    DIALOG: str = "dialog"
    TAB: str = "tab"
    TABLIST: str = "tablist"
    TABPANEL: str = "tabpanel"
    TREE: str = "tree"
    TREEITEM: str = "treeitem"
    GRID: str = "grid"
    GRIDCELL: str = "gridcell"
    ROW: str = "row"
    ROWGROUP: str = "rowgroup"
    COLUMNHEADER: str = "columnheader"
    ROWHEADER: str = "rowheader"
    TABLE: str = "table"
    CELL: str = "cell"
    IMG: str = "img"
    FIGURE: str = "figure"
    ARTICLE: str = "article"
    SEPARATOR: str = "separator"
    TOOLBAR: str = "toolbar"
    STATUS: str = "status"
    PROGRESSBAR: str = "progressbar"
    SLIDER: str = "slider"
    SPINBUTTON: str = "spinbutton"
    COMBOBOX: str = "combobox"
    OPTION: str = "option"
    LISTBOX: str = "listbox"
    DOCUMENT: str = "document"
    APPLICATION: str = "application"
    GROUP: str = "group"
    PRESENTATION: str = "presentation"
    NONE: str = "none"

    @classmethod
    def is_interactive(cls, role: str) -> bool:
        """Check if a role represents an interactive element.

        Args:
            role: The ARIA role string to check

        Returns:
            True if the role is typically interactive
        """
        interactive_roles = {
            cls.BUTTON, cls.LINK, cls.TEXTBOX, cls.CHECKBOX, cls.RADIO,
            cls.MENUITEM, cls.TAB, cls.TREEITEM, cls.OPTION, cls.SLIDER,
            cls.SPINBUTTON, cls.COMBOBOX, cls.LISTBOX, cls.GRIDCELL,
        }
        return role in interactive_roles

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass
class AriaNode:
    """Minimal aria node representation for sharing between contexts.

    This is the shared kernel representation. The Snapshot Context may
    have a richer internal representation, but this is what gets shared
    with the Element Registry Context.
    """
    ref: ElementRef
    role: AriaRole
    name: Optional[str] = None
    level: Optional[int] = None  # For headings (h1-h6)
    children: List["AriaNode"] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_interactive(self) -> bool:
        """Check if this node represents an interactive element."""
        return AriaRole.is_interactive(self.role.value)

    @property
    def display_name(self) -> str:
        """Get a display-friendly name for the node.

        Returns role and name combined for display purposes.
        """
        if self.name:
            return f'{self.role.value} "{self.name}"'
        return self.role.value

    def traverse(self) -> "AriaNodeIterator":
        """Traverse this node and all descendants.

        Returns:
            Iterator yielding this node and all descendants depth-first
        """
        return AriaNodeIterator(self)

    def find_by_ref(self, ref: ElementRef) -> Optional["AriaNode"]:
        """Find a descendant node by its ref.

        Args:
            ref: The ElementRef to search for

        Returns:
            The matching node, or None if not found
        """
        if self.ref == ref:
            return self
        for child in self.children:
            found = child.find_by_ref(ref)
            if found:
                return found
        return None

    def count_nodes(self) -> int:
        """Count total nodes in this subtree.

        Returns:
            Total number of nodes including this one
        """
        return 1 + sum(child.count_nodes() for child in self.children)

    def count_interactive(self) -> int:
        """Count interactive nodes in this subtree.

        Returns:
            Number of interactive nodes including this one if interactive
        """
        count = 1 if self.is_interactive else 0
        return count + sum(child.count_interactive() for child in self.children)


class AriaNodeIterator:
    """Iterator for depth-first traversal of AriaNode tree."""

    def __init__(self, root: AriaNode) -> None:
        """Initialize iterator with root node.

        Args:
            root: The root node to start traversal from
        """
        self._stack: List[AriaNode] = [root]

    def __iter__(self) -> "AriaNodeIterator":
        return self

    def __next__(self) -> AriaNode:
        if not self._stack:
            raise StopIteration
        node = self._stack.pop()
        # Add children in reverse order so leftmost is processed first
        self._stack.extend(reversed(node.children))
        return node
