"""Tests for Element Registry Context bounded context.

This module tests the element registry functionality for token optimization:
- Locator value object (browser library format conversions)
- ElementRegistry aggregate (ref management)
- StaleRefError exception
- Registry lifecycle and cleanup
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock

import pytest


# =============================================================================
# Domain Models (to be moved to production code)
# =============================================================================


class StaleRefError(Exception):
    """Exception raised when an element reference is stale.

    This occurs when the page has navigated or the DOM has changed
    since the snapshot was taken.
    """

    def __init__(self, ref: str, message: Optional[str] = None):
        self.ref = ref
        default_msg = (
            f"Element ref '{ref}' is stale. "
            "Page may have navigated. Call snapshot tool to get fresh refs."
        )
        super().__init__(message or default_msg)


class InvalidRefFormatError(Exception):
    """Exception raised when a ref string has invalid format."""

    def __init__(self, ref: str):
        self.ref = ref
        super().__init__(
            f"Invalid ref format: '{ref}'. Expected format: 'e' followed by a number (e.g., 'e42')"
        )


@dataclass(frozen=True)
class Locator:
    """Value object representing an element locator.

    Supports conversion between different locator formats used by
    various Robot Framework libraries.
    """

    strategy: str  # css, xpath, id, text, etc.
    value: str
    # Additional locator properties
    aria_role: Optional[str] = None
    aria_name: Optional[str] = None

    def to_browser_library_format(self) -> str:
        """Convert to Browser Library locator format."""
        if self.strategy == "css":
            return f"css={self.value}"
        elif self.strategy == "xpath":
            return f"xpath={self.value}"
        elif self.strategy == "id":
            return f"id={self.value}"
        elif self.strategy == "text":
            return f"text={self.value}"
        elif self.strategy == "aria":
            # Use ARIA selector format
            if self.aria_role and self.aria_name:
                return f'role={self.aria_role}[name="{self.aria_name}"]'
            elif self.aria_role:
                return f"role={self.aria_role}"
            return f'text="{self.value}"'
        else:
            return f"{self.strategy}={self.value}"

    def to_selenium_library_format(self) -> str:
        """Convert to SeleniumLibrary locator format."""
        if self.strategy == "css":
            return f"css:{self.value}"
        elif self.strategy == "xpath":
            return f"xpath:{self.value}"
        elif self.strategy == "id":
            return f"id:{self.value}"
        elif self.strategy == "text":
            # SeleniumLibrary uses link text
            return f"link:{self.value}"
        elif self.strategy == "aria":
            # Convert ARIA to XPath for SeleniumLibrary
            if self.aria_role and self.aria_name:
                return f"xpath://*[@role='{self.aria_role}' and @aria-label='{self.aria_name}']"
            elif self.aria_role:
                return f"xpath://*[@role='{self.aria_role}']"
            return f"xpath://*[contains(text(), '{self.value}')]"
        else:
            return f"{self.strategy}:{self.value}"

    @classmethod
    def from_css(cls, selector: str) -> "Locator":
        """Create a CSS locator."""
        return cls(strategy="css", value=selector)

    @classmethod
    def from_xpath(cls, xpath: str) -> "Locator":
        """Create an XPath locator."""
        return cls(strategy="xpath", value=xpath)

    @classmethod
    def from_aria(
        cls, role: str, name: Optional[str] = None
    ) -> "Locator":
        """Create an ARIA-based locator."""
        return cls(
            strategy="aria",
            value=name or "",
            aria_role=role,
            aria_name=name,
        )


@dataclass
class RegisteredElement:
    """Internal representation of a registered element."""

    ref: str
    locator: Locator
    role: str
    name: str
    stale: bool = False
    registered_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class ElementRegistry:
    """Aggregate for managing element references.

    Maps short ref IDs (e.g., 'e42') to full locators.
    Handles ref lifecycle, validation, and cleanup.
    """

    # Valid ref pattern: 'e' followed by digits
    REF_PATTERN = re.compile(r"^e\d+$")

    def __init__(
        self,
        max_refs: int = 10000,
        ref_ttl_seconds: float = 300.0,  # 5 minutes default
    ):
        self._elements: Dict[str, RegisteredElement] = {}
        self._next_index: int = 1
        self._snapshot_version: int = 0
        self._max_refs = max_refs
        self._ref_ttl = ref_ttl_seconds

    @property
    def snapshot_version(self) -> int:
        """Current snapshot version number."""
        return self._snapshot_version

    def register_element(
        self,
        locator: Locator,
        role: str,
        name: str = "",
    ) -> str:
        """Register an element and return its ref.

        Args:
            locator: The element locator
            role: ARIA role of the element
            name: Accessible name of the element

        Returns:
            The assigned ref (e.g., 'e42')

        Raises:
            RuntimeError: If max refs limit is reached
        """
        if len(self._elements) >= self._max_refs:
            # Try cleanup first
            self._cleanup_expired()
            if len(self._elements) >= self._max_refs:
                raise RuntimeError(
                    f"Element registry limit reached ({self._max_refs}). "
                    "Call invalidate() to clear stale refs."
                )

        ref = f"e{self._next_index}"
        self._next_index += 1

        self._elements[ref] = RegisteredElement(
            ref=ref,
            locator=locator,
            role=role,
            name=name,
        )
        return ref

    def get_locator(self, ref: str) -> Locator:
        """Get the locator for a ref.

        Args:
            ref: The element ref (e.g., 'e42')

        Returns:
            The Locator for the element

        Raises:
            InvalidRefFormatError: If ref format is invalid
            StaleRefError: If the ref is stale or not found
        """
        # Validate format
        if not self._validate_ref_format(ref):
            raise InvalidRefFormatError(ref)

        element = self._elements.get(ref)
        if element is None:
            raise StaleRefError(ref)

        if element.stale:
            raise StaleRefError(ref, f"Element ref '{ref}' has been marked stale.")

        # Update last accessed time
        element.last_accessed = time.time()
        return element.locator

    def invalidate(self) -> int:
        """Invalidate all refs, marking them as stale.

        This should be called after page navigation or significant DOM changes.

        Returns:
            Number of refs invalidated
        """
        count = 0
        for element in self._elements.values():
            if not element.stale:
                element.stale = True
                count += 1
        self._snapshot_version += 1
        return count

    def clear(self) -> int:
        """Clear all refs from the registry.

        Returns:
            Number of refs cleared
        """
        count = len(self._elements)
        self._elements.clear()
        self._next_index = 1
        self._snapshot_version += 1
        return count

    def _validate_ref_format(self, ref: str) -> bool:
        """Validate that a ref has the correct format."""
        return bool(self.REF_PATTERN.match(ref))

    def _cleanup_expired(self) -> int:
        """Remove expired refs based on TTL.

        Returns:
            Number of refs removed
        """
        current_time = time.time()
        expired = [
            ref
            for ref, element in self._elements.items()
            if element.stale
            or (current_time - element.last_accessed) > self._ref_ttl
        ]
        for ref in expired:
            del self._elements[ref]
        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stale_count = sum(1 for e in self._elements.values() if e.stale)
        return {
            "total_refs": len(self._elements),
            "stale_refs": stale_count,
            "active_refs": len(self._elements) - stale_count,
            "max_refs": self._max_refs,
            "snapshot_version": self._snapshot_version,
        }

    def __len__(self) -> int:
        return len(self._elements)

    def __contains__(self, ref: str) -> bool:
        element = self._elements.get(ref)
        return element is not None and not element.stale


# =============================================================================
# Tests
# =============================================================================


class TestLocator:
    """Tests for Locator value object."""

    def test_to_browser_library_format_css(self):
        """Test CSS locator conversion to Browser Library format."""
        locator = Locator(strategy="css", value="button.submit")
        assert locator.to_browser_library_format() == "css=button.submit"

    def test_to_browser_library_format_xpath(self):
        """Test XPath locator conversion to Browser Library format."""
        locator = Locator(strategy="xpath", value="//button[@type='submit']")
        assert locator.to_browser_library_format() == "xpath=//button[@type='submit']"

    def test_to_browser_library_format_id(self):
        """Test ID locator conversion to Browser Library format."""
        locator = Locator(strategy="id", value="submit-btn")
        assert locator.to_browser_library_format() == "id=submit-btn"

    def test_to_browser_library_format_text(self):
        """Test text locator conversion to Browser Library format."""
        locator = Locator(strategy="text", value="Submit")
        assert locator.to_browser_library_format() == "text=Submit"

    def test_to_browser_library_format_aria(self):
        """Test ARIA locator conversion to Browser Library format."""
        locator = Locator.from_aria(role="button", name="Submit")
        result = locator.to_browser_library_format()
        assert 'role=button[name="Submit"]' == result

    def test_to_selenium_library_format_css(self):
        """Test CSS locator conversion to SeleniumLibrary format."""
        locator = Locator(strategy="css", value="button.submit")
        assert locator.to_selenium_library_format() == "css:button.submit"

    def test_to_selenium_library_format_xpath(self):
        """Test XPath locator conversion to SeleniumLibrary format."""
        locator = Locator(strategy="xpath", value="//button")
        assert locator.to_selenium_library_format() == "xpath://button"

    def test_to_selenium_library_format_id(self):
        """Test ID locator conversion to SeleniumLibrary format."""
        locator = Locator(strategy="id", value="my-id")
        assert locator.to_selenium_library_format() == "id:my-id"

    def test_to_selenium_library_format_text(self):
        """Test text locator conversion to SeleniumLibrary format."""
        locator = Locator(strategy="text", value="Click me")
        assert locator.to_selenium_library_format() == "link:Click me"

    def test_to_selenium_library_format_aria(self):
        """Test ARIA locator conversion to SeleniumLibrary format."""
        locator = Locator.from_aria(role="button", name="Submit")
        result = locator.to_selenium_library_format()
        assert "xpath:" in result
        assert "@role='button'" in result
        assert "@aria-label='Submit'" in result

    def test_from_css_factory(self):
        """Test CSS locator factory method."""
        locator = Locator.from_css(".my-class")
        assert locator.strategy == "css"
        assert locator.value == ".my-class"

    def test_from_xpath_factory(self):
        """Test XPath locator factory method."""
        locator = Locator.from_xpath("//div[@id='test']")
        assert locator.strategy == "xpath"
        assert locator.value == "//div[@id='test']"

    def test_from_aria_factory(self):
        """Test ARIA locator factory method."""
        locator = Locator.from_aria("button", "Submit")
        assert locator.strategy == "aria"
        assert locator.aria_role == "button"
        assert locator.aria_name == "Submit"

    def test_locator_is_immutable(self):
        """Test that Locator is immutable (frozen dataclass)."""
        locator = Locator(strategy="css", value=".test")
        with pytest.raises((AttributeError, TypeError)):
            locator.strategy = "xpath"

    def test_locator_equality(self):
        """Test Locator equality comparison."""
        loc1 = Locator(strategy="css", value=".test")
        loc2 = Locator(strategy="css", value=".test")
        loc3 = Locator(strategy="css", value=".other")

        assert loc1 == loc2
        assert loc1 != loc3


class TestElementRegistry:
    """Tests for ElementRegistry aggregate."""

    @pytest.fixture
    def registry(self) -> ElementRegistry:
        """Create a fresh registry for testing."""
        return ElementRegistry(max_refs=100, ref_ttl_seconds=60.0)

    def test_register_element_returns_ref(self, registry):
        """Test that registering an element returns a valid ref."""
        locator = Locator.from_css("button.submit")
        ref = registry.register_element(locator, role="button", name="Submit")

        assert ref.startswith("e")
        assert ref[1:].isdigit()

    def test_register_multiple_elements_unique_refs(self, registry):
        """Test that multiple registrations return unique refs."""
        refs = set()
        for i in range(10):
            locator = Locator.from_css(f".item-{i}")
            ref = registry.register_element(locator, role="button", name=f"Button {i}")
            refs.add(ref)

        assert len(refs) == 10

    def test_get_locator_returns_correct_locator(self, registry):
        """Test retrieving a locator by ref."""
        locator = Locator.from_css("button.submit")
        ref = registry.register_element(locator, role="button", name="Submit")

        retrieved = registry.get_locator(ref)
        assert retrieved == locator

    def test_get_locator_raises_on_invalid_ref(self, registry):
        """Test that invalid refs raise appropriate errors."""
        # Non-existent ref
        with pytest.raises(StaleRefError):
            registry.get_locator("e999")

    def test_get_locator_raises_on_invalid_format(self, registry):
        """Test that invalid ref formats raise InvalidRefFormatError."""
        invalid_refs = [
            "invalid",
            "42",
            "E42",  # Wrong case
            "e",  # Missing number
            "e-1",  # Negative
            "e1.5",  # Decimal
            "e 42",  # Space
        ]
        for invalid in invalid_refs:
            with pytest.raises(InvalidRefFormatError):
                registry.get_locator(invalid)

    def test_invalidate_marks_all_refs_stale(self, registry):
        """Test that invalidate marks all refs as stale."""
        # Register some elements
        for i in range(5):
            locator = Locator.from_css(f".item-{i}")
            registry.register_element(locator, role="button")

        # Invalidate
        invalidated = registry.invalidate()
        assert invalidated == 5

        # All refs should now raise StaleRefError
        with pytest.raises(StaleRefError):
            registry.get_locator("e1")

    def test_max_refs_limit_enforced(self):
        """Test that max refs limit is enforced."""
        registry = ElementRegistry(max_refs=5, ref_ttl_seconds=60.0)

        # Fill up the registry
        for i in range(5):
            locator = Locator.from_css(f".item-{i}")
            registry.register_element(locator, role="button")

        # Next registration should fail
        with pytest.raises(RuntimeError, match="limit reached"):
            locator = Locator.from_css(".overflow")
            registry.register_element(locator, role="button")

    def test_ref_expiration_cleanup(self):
        """Test that expired refs are cleaned up."""
        registry = ElementRegistry(max_refs=5, ref_ttl_seconds=0.01)  # 10ms TTL

        # Register element
        locator = Locator.from_css(".item")
        ref = registry.register_element(locator, role="button")

        # Wait for expiration
        time.sleep(0.02)

        # Cleanup should remove expired ref
        removed = registry._cleanup_expired()
        assert removed == 1
        assert ref not in registry

    def test_ref_format_validation(self, registry):
        """Test ref format validation logic."""
        valid_refs = ["e0", "e1", "e42", "e999", "e10000"]
        invalid_refs = ["e", "E1", "1", "ref1", "e1a", "e-1", "e 1", ""]

        for ref in valid_refs:
            assert registry._validate_ref_format(ref) is True

        for ref in invalid_refs:
            assert registry._validate_ref_format(ref) is False

    def test_clear_removes_all_refs(self, registry):
        """Test that clear removes all refs."""
        # Register some elements
        for i in range(5):
            locator = Locator.from_css(f".item-{i}")
            registry.register_element(locator, role="button")

        assert len(registry) == 5

        # Clear
        cleared = registry.clear()
        assert cleared == 5
        assert len(registry) == 0

    def test_contains_check(self, registry):
        """Test __contains__ for checking ref existence."""
        locator = Locator.from_css(".item")
        ref = registry.register_element(locator, role="button")

        assert ref in registry
        assert "e999" not in registry

        # After invalidation, ref should not be "in" registry
        registry.invalidate()
        assert ref not in registry

    def test_get_stats(self, registry):
        """Test registry statistics."""
        # Register elements
        for i in range(5):
            locator = Locator.from_css(f".item-{i}")
            registry.register_element(locator, role="button")

        # Invalidate some
        registry._elements["e1"].stale = True
        registry._elements["e2"].stale = True

        stats = registry.get_stats()
        assert stats["total_refs"] == 5
        assert stats["stale_refs"] == 2
        assert stats["active_refs"] == 3
        assert stats["max_refs"] == 100

    def test_snapshot_version_increments(self, registry):
        """Test that snapshot version increments on invalidate/clear."""
        initial_version = registry.snapshot_version

        registry.invalidate()
        assert registry.snapshot_version == initial_version + 1

        registry.clear()
        assert registry.snapshot_version == initial_version + 2


class TestStaleRefError:
    """Tests for StaleRefError exception."""

    def test_error_contains_helpful_message(self):
        """Test that error message is helpful."""
        error = StaleRefError("e42")
        message = str(error)

        assert "e42" in message
        assert "stale" in message.lower()
        assert "snapshot" in message.lower()

    def test_error_stores_ref(self):
        """Test that error stores the ref."""
        error = StaleRefError("e42")
        assert error.ref == "e42"

    def test_custom_message(self):
        """Test custom error message."""
        error = StaleRefError("e42", "Custom message about e42")
        assert str(error) == "Custom message about e42"


class TestInvalidRefFormatError:
    """Tests for InvalidRefFormatError exception."""

    def test_error_contains_ref(self):
        """Test that error contains the invalid ref."""
        error = InvalidRefFormatError("bad_ref")
        message = str(error)

        assert "bad_ref" in message
        assert "Invalid ref format" in message

    def test_error_suggests_correct_format(self):
        """Test that error message suggests correct format."""
        error = InvalidRefFormatError("invalid")
        message = str(error)

        assert "e42" in message or "e" in message  # Shows example


class TestRegistryEdgeCases:
    """Tests for registry edge cases."""

    def test_concurrent_access_simulation(self):
        """Test simulated concurrent access patterns."""
        registry = ElementRegistry(max_refs=1000)

        # Simulate rapid registration and lookup
        refs = []
        for i in range(100):
            locator = Locator.from_css(f".item-{i}")
            ref = registry.register_element(locator, role="button", name=f"Button {i}")
            refs.append(ref)

        # All refs should be retrievable
        for ref in refs:
            locator = registry.get_locator(ref)
            assert locator is not None

    def test_reregistration_after_clear(self):
        """Test that refs start from e1 after clear."""
        registry = ElementRegistry()

        # First registration
        locator1 = Locator.from_css(".item1")
        ref1 = registry.register_element(locator1, role="button")

        # Clear
        registry.clear()

        # Next registration should start from e1 again
        locator2 = Locator.from_css(".item2")
        ref2 = registry.register_element(locator2, role="button")

        assert ref2 == "e1"

    def test_locator_with_special_characters(self):
        """Test locators with special characters."""
        registry = ElementRegistry()

        # Various special character scenarios
        locators = [
            Locator.from_css('[data-testid="submit-btn"]'),
            Locator.from_xpath("//button[contains(@class, 'btn')]"),
            Locator.from_css(".btn\\.special"),  # Escaped class
            Locator.from_aria("button", 'Click "here"'),
        ]

        for locator in locators:
            ref = registry.register_element(locator, role="button")
            retrieved = registry.get_locator(ref)
            assert retrieved == locator
