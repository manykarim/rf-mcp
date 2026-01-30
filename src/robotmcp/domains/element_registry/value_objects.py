"""Value Objects for the Element Registry Context.

Value objects are immutable domain primitives that encapsulate
validation rules and provide type safety.

Security Controls:
- RegistryId: Validated format for session/version combination
- Locator: Input sanitization for locator values
- StaleRefError: Clear error messaging for recovery
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotmcp.domains.shared import ElementRef


class LocatorStrategy(Enum):
    """Supported locator strategies for element identification.

    Each strategy corresponds to a different method of locating
    elements in the DOM, supporting both Browser Library (Playwright)
    and SeleniumLibrary.
    """
    CSS = "css"
    XPATH = "xpath"
    ID = "id"
    NAME = "name"
    TEXT = "text"
    ARIA_LABEL = "aria-label"
    ROLE = "role"
    DATA_TESTID = "data-testid"

    @classmethod
    def from_string(cls, value: str) -> "LocatorStrategy":
        """Create a LocatorStrategy from a string value.

        Args:
            value: The strategy name (case-insensitive)

        Returns:
            The matching LocatorStrategy

        Raises:
            ValueError: If the strategy name is not recognized
        """
        normalized = value.lower().strip()
        for strategy in cls:
            if strategy.value == normalized:
                return strategy
        raise ValueError(
            f"Unknown locator strategy: '{value}'. "
            f"Valid strategies: {[s.value for s in cls]}"
        )


@dataclass(frozen=True)
class RegistryId:
    """Unique identifier for an element registry.

    Format: reg_{session_id}_{version}

    The registry ID uniquely identifies a specific version of the
    element registry for a session. Each time a new snapshot is
    taken, a new registry version is created.
    """
    value: str

    # Pattern: reg_ followed by session_id and version number
    REGISTRY_PATTERN: str = r"^reg_[a-zA-Z0-9_-]+_\d+$"

    def __post_init__(self) -> None:
        """Validate registry ID format."""
        if not self.value:
            raise ValueError("RegistryId value cannot be empty")
        # Allow flexible format during creation, validate on lookup
        if not self.value.startswith("reg_"):
            raise ValueError(
                f"Invalid RegistryId format: '{self.value}'. "
                "Must start with 'reg_'"
            )

    @classmethod
    def for_session(cls, session_id: str, version: int) -> "RegistryId":
        """Create a RegistryId for a specific session and version.

        Args:
            session_id: The session identifier
            version: The snapshot version number

        Returns:
            A new RegistryId

        Raises:
            ValueError: If session_id is empty or version is negative
        """
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty")
        if version < 0:
            raise ValueError(f"Version must be non-negative, got {version}")

        # Sanitize session_id to remove potentially dangerous characters
        sanitized_session = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id.strip())
        return cls(value=f"reg_{sanitized_session}_{version}")

    @property
    def session_id(self) -> str:
        """Extract the session ID from the registry ID.

        Returns:
            The session ID portion of the registry ID
        """
        # Format: reg_{session_id}_{version}
        parts = self.value.split("_")
        if len(parts) >= 3:
            # Join all parts except first (reg) and last (version)
            return "_".join(parts[1:-1])
        return ""

    @property
    def version(self) -> int:
        """Extract the version number from the registry ID.

        Returns:
            The version number, or 0 if not parseable
        """
        parts = self.value.split("_")
        if parts:
            try:
                return int(parts[-1])
            except ValueError:
                return 0
        return 0

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(frozen=True)
class Locator:
    """Abstraction over different locator strategies.

    A Locator combines a strategy (how to find the element) with
    a value (the selector or identifier to use). It provides
    conversion methods for different Robot Framework libraries.

    Security: Locator values are sanitized to prevent injection attacks.
    """
    strategy: LocatorStrategy
    value: str

    # Maximum locator value length to prevent DoS
    MAX_VALUE_LENGTH: int = 10000

    # Characters that need escaping in CSS selectors
    CSS_ESCAPE_CHARS: str = r'!"#$%&\'()*+,./:;<=>?@[\]^`{|}~'

    def __post_init__(self) -> None:
        """Validate and potentially sanitize the locator value."""
        if not self.value:
            raise ValueError("Locator value cannot be empty")
        if len(self.value) > self.MAX_VALUE_LENGTH:
            raise ValueError(
                f"Locator value exceeds maximum length of {self.MAX_VALUE_LENGTH}"
            )

    @classmethod
    def create_safe(cls, strategy: LocatorStrategy, value: str) -> "Locator":
        """Create a Locator with sanitized value.

        This factory method applies appropriate sanitization based
        on the locator strategy.

        Args:
            strategy: The locator strategy
            value: The raw locator value

        Returns:
            A Locator with sanitized value
        """
        sanitized_value = cls._sanitize_value(strategy, value)
        return cls(strategy=strategy, value=sanitized_value)

    @staticmethod
    def _sanitize_value(strategy: LocatorStrategy, value: str) -> str:
        """Sanitize locator value based on strategy.

        Args:
            strategy: The locator strategy
            value: The raw value

        Returns:
            Sanitized value
        """
        if not value:
            return value

        # Trim whitespace
        value = value.strip()

        # Strategy-specific sanitization
        if strategy == LocatorStrategy.XPATH:
            # Escape quotes in XPath - prefer single quotes for values
            # to allow use in XPath expressions
            if "'" in value and '"' in value:
                # Use concat for mixed quotes
                parts = value.split("'")
                value = "concat('" + "',\"'\",'" .join(parts) + "')"
            return value

        elif strategy == LocatorStrategy.CSS:
            # CSS selectors: escape special characters in identifiers
            # This is for ID/class names, not the full selector
            return value

        elif strategy in (LocatorStrategy.ID, LocatorStrategy.NAME,
                         LocatorStrategy.DATA_TESTID):
            # Attribute values: escape HTML entities
            return html.escape(value, quote=True)

        elif strategy == LocatorStrategy.TEXT:
            # Text content: escape HTML entities
            return html.escape(value, quote=True)

        elif strategy == LocatorStrategy.ARIA_LABEL:
            # ARIA labels: escape HTML entities
            return html.escape(value, quote=True)

        elif strategy == LocatorStrategy.ROLE:
            # Role values are predefined, just lowercase
            return value.lower()

        return value

    def to_browser_library(self) -> str:
        """Convert to Browser Library (Playwright) locator format.

        Browser Library uses the format: strategy=value

        Returns:
            Browser Library compatible locator string
        """
        strategy_map = {
            LocatorStrategy.CSS: "css",
            LocatorStrategy.XPATH: "xpath",
            LocatorStrategy.ID: "id",
            LocatorStrategy.NAME: "name",
            LocatorStrategy.TEXT: "text",
            LocatorStrategy.ARIA_LABEL: "aria-label",
            LocatorStrategy.ROLE: "role",
            LocatorStrategy.DATA_TESTID: "data-testid",
        }
        prefix = strategy_map.get(self.strategy, "css")

        # Browser Library special handling
        if self.strategy == LocatorStrategy.TEXT:
            # Browser Library uses text= prefix
            return f"text={self.value}"
        elif self.strategy == LocatorStrategy.ROLE:
            # Role selector with optional name
            return f"role={self.value}"
        elif self.strategy == LocatorStrategy.ARIA_LABEL:
            # Use attribute selector for aria-label
            return f'css=[aria-label="{self.value}"]'
        elif self.strategy == LocatorStrategy.DATA_TESTID:
            return f'css=[data-testid="{self.value}"]'
        elif self.strategy == LocatorStrategy.ID:
            return f"id={self.value}"
        elif self.strategy == LocatorStrategy.NAME:
            return f'css=[name="{self.value}"]'

        return f"{prefix}={self.value}"

    def to_selenium_library(self) -> str:
        """Convert to SeleniumLibrary locator format.

        SeleniumLibrary uses the format: strategy:value or strategy=value

        Returns:
            SeleniumLibrary compatible locator string
        """
        strategy_map = {
            LocatorStrategy.CSS: "css",
            LocatorStrategy.XPATH: "xpath",
            LocatorStrategy.ID: "id",
            LocatorStrategy.NAME: "name",
            LocatorStrategy.TEXT: "link",  # SeleniumLibrary uses 'link' for text
            LocatorStrategy.ARIA_LABEL: "css",  # Use CSS attribute selector
            LocatorStrategy.ROLE: "css",  # Use CSS attribute selector
            LocatorStrategy.DATA_TESTID: "css",  # Use CSS attribute selector
        }
        prefix = strategy_map.get(self.strategy, "css")

        # SeleniumLibrary specific handling
        if self.strategy == LocatorStrategy.ARIA_LABEL:
            return f'css:[aria-label="{self.value}"]'
        elif self.strategy == LocatorStrategy.ROLE:
            return f'css:[role="{self.value}"]'
        elif self.strategy == LocatorStrategy.DATA_TESTID:
            return f'css:[data-testid="{self.value}"]'

        return f"{prefix}:{self.value}"

    def to_playwright(self) -> str:
        """Convert to raw Playwright selector format.

        Returns:
            Playwright compatible selector string
        """
        if self.strategy == LocatorStrategy.CSS:
            return self.value
        elif self.strategy == LocatorStrategy.XPATH:
            return f"xpath={self.value}"
        elif self.strategy == LocatorStrategy.ID:
            return f"#{self.value}"
        elif self.strategy == LocatorStrategy.TEXT:
            return f'text="{self.value}"'
        elif self.strategy == LocatorStrategy.ROLE:
            return f'role={self.value}'
        elif self.strategy == LocatorStrategy.ARIA_LABEL:
            return f'[aria-label="{self.value}"]'
        elif self.strategy == LocatorStrategy.DATA_TESTID:
            return f'[data-testid="{self.value}"]'
        elif self.strategy == LocatorStrategy.NAME:
            return f'[name="{self.value}"]'
        return self.value

    def __str__(self) -> str:
        return f"{self.strategy.value}={self.value}"


class StaleRefError(Exception):
    """Error when a ref is no longer valid.

    This error is raised when attempting to use an element reference
    that belongs to an older snapshot version. The error provides
    guidance on how to recover (call get_page_snapshot).

    Attributes:
        ref: The stale ElementRef that was accessed
        snapshot_version_used: The version the ref belongs to
        current_snapshot_version: The current active version
        message: Human-readable error message
    """

    def __init__(
        self,
        ref: "ElementRef",
        snapshot_version_used: int,
        current_snapshot_version: int,
        message: str = "Ref is stale. Call get_page_snapshot for fresh refs.",
    ) -> None:
        """Initialize a StaleRefError.

        Args:
            ref: The stale ElementRef
            snapshot_version_used: Version the ref was from
            current_snapshot_version: Current snapshot version
            message: Error message (optional)
        """
        self.ref = ref
        self.snapshot_version_used = snapshot_version_used
        self.current_snapshot_version = current_snapshot_version
        self.message = message

        super().__init__(
            f"{message} "
            f"(ref={ref.value}, version_used={snapshot_version_used}, "
            f"current_version={current_snapshot_version})"
        )

    def __repr__(self) -> str:
        return (
            f"StaleRefError(ref={self.ref!r}, "
            f"snapshot_version_used={self.snapshot_version_used}, "
            f"current_snapshot_version={self.current_snapshot_version})"
        )
