"""Shared Kernel - Core domain types shared across bounded contexts.

These types are intentionally minimal and shared between:
- Snapshot Context (produces ElementRef, AriaNode, AriaRole)
- Element Registry Context (consumes these for registration)

As per ADR-001, this shared kernel is kept minimal to reduce coupling.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BeforeValidator


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


class ModelTier(Enum):
    """LLM context window capacity classification.

    Used by Tool Profile (ADR-006) for profile selection and description mode,
    Response Optimization (ADR-008) for token budget allocation,
    and Instruction context for template selection.
    """
    SMALL_CONTEXT = "small_context"
    STANDARD = "standard"
    LARGE_CONTEXT = "large_context"

    @classmethod
    def from_context_window(cls, window_size: int) -> "ModelTier":
        """Infer model tier from context window size in tokens."""
        if window_size <= 16384:
            return cls.SMALL_CONTEXT
        elif window_size <= 65536:
            return cls.STANDARD
        else:
            return cls.LARGE_CONTEXT


# ============================================================
# ADR-009: Type-Constrained Tool Parameters
# ============================================================
#
# Literal type aliases with BeforeValidator for case-insensitive
# normalization.  Produces flat {"enum": [...]} in JSON Schema
# while accepting wrong-case input at runtime.
# ============================================================


def _normalize_str(v: Any) -> Any:
    """Normalize string input: strip whitespace, lowercase."""
    return v.strip().lower() if isinstance(v, str) else v


# ── Tier 1: Action dispatchers (HIGH impact) ──────────────────

SessionAction = Annotated[
    Literal[
        "init", "initialize", "bootstrap",
        "import_library", "library",
        "import_resource", "resource",
        "set_variables", "variables",
        "import_variables", "load_variables",
        "start_test", "end_test", "start_task", "end_task",
        "list_tests",
        "set_suite_setup", "set_suite_teardown",
        "set_tool_profile", "tool_profile",
    ],
    BeforeValidator(_normalize_str),
]

TestStatus = Annotated[
    Literal["pass", "fail"],
    BeforeValidator(_normalize_str),
]

ToolProfileName = Annotated[
    Literal["browser_exec", "api_exec", "discovery", "minimal_exec", "full"],
    BeforeValidator(_normalize_str),
]

ModelTierLiteral = Annotated[
    Literal["small_context", "standard", "large_context"],
    BeforeValidator(_normalize_str),
]

PluginAction = Annotated[
    Literal["list", "reload", "diagnose"],
    BeforeValidator(_normalize_str),
]

AttachAction = Annotated[
    Literal[
        "status", "info", "stop", "shutdown",
        "cleanup", "clean", "reset", "reconnect",
        "disconnect_all", "terminate", "force_stop",
    ],
    BeforeValidator(_normalize_str),
]

IntentVerb = Annotated[
    Literal[
        "navigate", "click", "fill", "hover",
        "select", "assert_visible", "extract_text", "wait_for",
    ],
    BeforeValidator(_normalize_str),
]

# ── Tier 2: Mode/Strategy selectors (MEDIUM impact) ──────────

KeywordStrategy = Annotated[
    Literal["semantic", "pattern", "catalog", "session"],
    BeforeValidator(_normalize_str),
]

AutomationContext = Annotated[
    Literal["web", "mobile", "api", "desktop", "generic", "database"],
    BeforeValidator(_normalize_str),
]

RecommendMode = Annotated[
    Literal["direct", "sampling_prompt", "sampling", "merge_samples", "merge"],
    BeforeValidator(_normalize_str),
]

FlowStructure = Annotated[
    Literal["if", "for", "try"],
    BeforeValidator(_normalize_str),
]

ExecutionMode = Annotated[
    Literal["keyword", "evaluate"],
    BeforeValidator(_normalize_str),
]

# ── Tier 3: Verbosity/Level selectors (LOW impact) ───────────

DetailLevel = Annotated[
    Literal["minimal", "standard", "full"],
    BeforeValidator(_normalize_str),
]

FilteringLevel = Annotated[
    Literal["standard", "aggressive"],
    BeforeValidator(_normalize_str),
]

SuiteRunMode = Annotated[
    Literal["dry", "validate", "full"],
    BeforeValidator(_normalize_str),
]

ValidationLevel = Annotated[
    Literal["minimal", "standard", "strict"],
    BeforeValidator(_normalize_str),
]


# ============================================================
# ADR-010: Small LLM Resilience — Array Coercion & Guided Recovery
# ============================================================


def _coerce_string_to_list(v: Any) -> Any:
    """Coerce stringified JSON arrays and comma-separated strings to lists.

    Handles three LLM output patterns:
    1. JSON array string:  '["Browser", "BuiltIn"]' -> ["Browser", "BuiltIn"]
    2. Comma-separated:    'Browser,BuiltIn'        -> ["Browser", "BuiltIn"]
    3. Single value:       'Browser'                 -> ["Browser"]

    Non-string inputs (list, None, int, etc.) pass through unchanged.
    Schema-transparent: produces identical JSON Schema to List[str].
    """
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        v_stripped = v.strip()
        # Path 1: JSON parse '[...]'
        if v_stripped.startswith("["):
            try:
                parsed = json.loads(v_stripped)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        # Path 2: Comma split 'A,B,C'
        if "," in v_stripped:
            return [item.strip() for item in v_stripped.split(",") if item.strip()]
        # Path 3: Single value 'Browser'
        if v_stripped:
            return [v_stripped]
    return v


CoercedStringList = Annotated[List[str], BeforeValidator(_coerce_string_to_list)]
OptionalCoercedStringList = Annotated[
    Optional[List[str]], BeforeValidator(_coerce_string_to_list)
]


def _coerce_string_to_variables(v: Any) -> Any:
    """Coerce stringified JSON dicts/arrays to proper dict or list.

    Handles LLM output patterns where structured types are sent as strings:
    1. JSON dict string:   '{"headless": "true"}'  -> {"headless": "true"}
    2. JSON array string:  '["headless=true"]'      -> ["headless=true"]
    3. Comma-separated:    'headless=true,TIMEOUT=30' -> ["headless=true", "TIMEOUT=30"]

    Non-string inputs (dict, list, None) pass through unchanged.
    """
    if v is None or isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        v_stripped = v.strip()
        if not v_stripped:
            return v
        # Path 1: JSON dict '{"key": "val"}'
        if v_stripped.startswith("{"):
            try:
                parsed = json.loads(v_stripped)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        # Path 2: JSON array '["key=val"]'
        if v_stripped.startswith("["):
            try:
                parsed = json.loads(v_stripped)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        # Path 3: Comma-separated 'key=val,key2=val2'
        if "=" in v_stripped:
            return [item.strip() for item in v_stripped.split(",") if item.strip()]
    return v


CoercedVariables = Annotated[
    Union[Dict[str, Any], List[str], None],
    BeforeValidator(_coerce_string_to_variables),
]


# ── ADR-010 I4: Deprecated keyword aliases ────────────────────

DEPRECATED_KEYWORD_ALIASES: Dict[str, str] = {
    "get": "GET On Session",
    "post": "POST On Session",
    "put": "PUT On Session",
    "delete": "DELETE On Session",
    "patch": "PATCH On Session",
    "head": "HEAD On Session",
    "options": "OPTIONS On Session",
}


def extract_deprecation_suggestion(error_msg: str) -> str | None:
    """Extract suggested replacement from deprecation warning text."""
    pattern = re.compile(
        r"(?:use|Use|favor of)\s+['\"]?(\w[\w\s]*?)['\"]?\s*(?:\.|instead|$)",
        re.IGNORECASE,
    )
    match = pattern.search(error_msg)
    return match.group(1).strip() if match else None


def resolve_deprecated_alias(keyword: str) -> str | None:
    """Return modern replacement for deprecated keyword, or None."""
    return DEPRECATED_KEYWORD_ALIASES.get(keyword.lower().strip())
