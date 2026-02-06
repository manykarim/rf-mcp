"""
Value Objects for the Snapshot bounded context.

Value objects are immutable and defined by their attributes rather than identity.
They represent concepts in the domain that have no conceptual identity.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import uuid


@dataclass(frozen=True)
class SnapshotId:
    """
    Unique identifier for a snapshot.

    Immutable value object representing the identity of a page snapshot.
    """
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("SnapshotId value cannot be empty")

    @classmethod
    def generate(cls) -> "SnapshotId":
        """Generate a new unique snapshot ID."""
        return cls(value=f"snap_{uuid.uuid4().hex[:8]}")

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"SnapshotId({self.value!r})"


@dataclass(frozen=True)
class ElementRef:
    """
    Short reference to an element (e1, e2, etc.).

    Used to provide compact element references in snapshots instead of
    verbose locators, achieving significant token reduction.
    """
    value: str  # Format: "e{number}"

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("ElementRef value cannot be empty")
        if not self.value.startswith("e"):
            raise ValueError(f"ElementRef must start with 'e', got: {self.value}")

    @classmethod
    def from_index(cls, index: int) -> "ElementRef":
        """Create an ElementRef from a numeric index."""
        if index < 0:
            raise ValueError(f"ElementRef index must be non-negative, got: {index}")
        return cls(value=f"e{index}")

    @property
    def index(self) -> int:
        """Extract the numeric index from the ref."""
        return int(self.value[1:])

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"ElementRef({self.value!r})"


@dataclass(frozen=True)
class AriaRole:
    """
    ARIA role value object.

    Represents the accessibility role of an element, which is crucial
    for understanding the semantic meaning and interactivity of elements.
    """
    value: str

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
    TAB: str = "tab"
    TABPANEL: str = "tabpanel"
    DIALOG: str = "dialog"
    NAVIGATION: str = "navigation"
    MAIN: str = "main"
    BANNER: str = "banner"
    CONTENTINFO: str = "contentinfo"
    COMPLEMENTARY: str = "complementary"
    FORM: str = "form"
    SEARCH: str = "search"
    REGION: str = "region"
    ARTICLE: str = "article"
    SECTION: str = "section"
    IMG: str = "img"
    TABLE: str = "table"
    ROW: str = "row"
    CELL: str = "cell"
    ROWHEADER: str = "rowheader"
    COLUMNHEADER: str = "columnheader"
    GRID: str = "grid"
    GRIDCELL: str = "gridcell"
    TREE: str = "tree"
    TREEITEM: str = "treeitem"
    DOCUMENT: str = "document"
    APPLICATION: str = "application"
    ALERT: str = "alert"
    ALERTDIALOG: str = "alertdialog"
    STATUS: str = "status"
    PROGRESSBAR: str = "progressbar"
    SLIDER: str = "slider"
    SPINBUTTON: str = "spinbutton"
    COMBOBOX: str = "combobox"
    OPTION: str = "option"
    SEPARATOR: str = "separator"
    TOOLBAR: str = "toolbar"
    TOOLTIP: str = "tooltip"
    GROUP: str = "group"
    PRESENTATION: str = "presentation"
    NONE: str = "none"

    # Interactive roles that typically accept user input
    INTERACTIVE_ROLES: frozenset = frozenset({
        "button", "link", "textbox", "checkbox", "radio",
        "menuitem", "tab", "combobox", "option", "slider",
        "spinbutton", "switch", "searchbox", "menuitemcheckbox",
        "menuitemradio", "treeitem", "gridcell"
    })

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("AriaRole value cannot be empty")

    @property
    def is_interactive(self) -> bool:
        """Check if this role represents an interactive element."""
        return self.value.lower() in self.INTERACTIVE_ROLES

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"AriaRole({self.value!r})"


@dataclass(frozen=True)
class SnapshotFormat:
    """
    Format specification for snapshot output.

    Controls how snapshots are captured and formatted, including
    compression options and output modes.
    """
    mode: Literal["full", "incremental", "none"]
    include_hidden: bool = False
    max_depth: Optional[int] = None
    fold_lists: bool = True
    fold_threshold: float = 0.85  # SimHash similarity threshold

    def __post_init__(self) -> None:
        if self.mode not in ("full", "incremental", "none"):
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, got: {self.max_depth}")
        if not (0.0 <= self.fold_threshold <= 1.0):
            raise ValueError(f"fold_threshold must be between 0 and 1, got: {self.fold_threshold}")

    @classmethod
    def full(cls, include_hidden: bool = False, max_depth: Optional[int] = None) -> "SnapshotFormat":
        """Create a full snapshot format configuration."""
        return cls(mode="full", include_hidden=include_hidden, max_depth=max_depth)

    @classmethod
    def incremental(cls, fold_lists: bool = True, fold_threshold: float = 0.85) -> "SnapshotFormat":
        """Create an incremental snapshot format configuration."""
        return cls(mode="incremental", fold_lists=fold_lists, fold_threshold=fold_threshold)

    @classmethod
    def none(cls) -> "SnapshotFormat":
        """Create a configuration that suppresses snapshot output."""
        return cls(mode="none")

    def __repr__(self) -> str:
        return (
            f"SnapshotFormat(mode={self.mode!r}, include_hidden={self.include_hidden}, "
            f"max_depth={self.max_depth}, fold_lists={self.fold_lists}, "
            f"fold_threshold={self.fold_threshold})"
        )


@dataclass(frozen=True)
class CompressionStats:
    """
    Statistics about compression achieved on a snapshot.

    Tracks the effectiveness of various compression techniques
    applied to reduce token consumption.
    """
    original_nodes: int
    compressed_nodes: int
    folded_lists: int
    token_estimate_before: int
    token_estimate_after: int

    def __post_init__(self) -> None:
        if self.original_nodes < 0:
            raise ValueError(f"original_nodes cannot be negative: {self.original_nodes}")
        if self.compressed_nodes < 0:
            raise ValueError(f"compressed_nodes cannot be negative: {self.compressed_nodes}")
        if self.folded_lists < 0:
            raise ValueError(f"folded_lists cannot be negative: {self.folded_lists}")
        if self.token_estimate_before < 0:
            raise ValueError(f"token_estimate_before cannot be negative: {self.token_estimate_before}")
        if self.token_estimate_after < 0:
            raise ValueError(f"token_estimate_after cannot be negative: {self.token_estimate_after}")

    @property
    def compression_ratio(self) -> float:
        """
        Calculate the compression ratio achieved.

        Returns a value between 0 and 1, where 0.9 means 90% reduction.
        """
        if self.token_estimate_before == 0:
            return 0.0
        return 1 - (self.token_estimate_after / self.token_estimate_before)

    @property
    def node_reduction(self) -> int:
        """Number of nodes removed through compression."""
        return self.original_nodes - self.compressed_nodes

    @property
    def token_savings(self) -> int:
        """Number of tokens saved through compression."""
        return self.token_estimate_before - self.token_estimate_after

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

    def __repr__(self) -> str:
        return (
            f"CompressionStats(original_nodes={self.original_nodes}, "
            f"compressed_nodes={self.compressed_nodes}, folded_lists={self.folded_lists}, "
            f"compression_ratio={self.compression_ratio:.2%})"
        )
