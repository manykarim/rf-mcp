"""Action Context Services.

Domain services for the Action Context, including the PreValidator service
that checks element actionability before execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, Set

from robotmcp.domains.action.value_objects import PreValidationResult
from robotmcp.domains.shared.kernel import ElementRef

if TYPE_CHECKING:
    from robotmcp.domains.shared.kernel import AriaNode

logger = logging.getLogger(__name__)


class Locator(Protocol):
    """Protocol for locator abstraction.

    Defines the interface for locators that can be used with browser adapters.
    This allows the Action Context to work with different locator implementations.
    """

    def to_browser_library(self) -> str:
        """Convert to Browser Library locator format."""
        ...

    def to_selenium_library(self) -> str:
        """Convert to SeleniumLibrary locator format."""
        ...


class ElementRegistry(Protocol):
    """Protocol for element registry abstraction.

    Defines the interface for accessing the element registry from the
    Element Registry Context.
    """

    def get_locator(self, ref: ElementRef) -> Locator:
        """Get the locator for an element reference."""
        ...

    def has_ref(self, ref: ElementRef) -> bool:
        """Check if a reference exists in the registry."""
        ...

    def is_stale(self) -> bool:
        """Check if the registry is stale."""
        ...


class BrowserAdapter(Protocol):
    """Protocol for browser adapter abstraction.

    Defines the interface for browser operations used by the PreValidator.
    Implementations handle the specifics of Browser Library vs SeleniumLibrary.
    """

    def is_element_visible(self, locator: str) -> bool:
        """Check if an element is visible."""
        ...

    def is_element_enabled(self, locator: str) -> bool:
        """Check if an element is enabled."""
        ...

    def is_element_editable(self, locator: str) -> bool:
        """Check if an element is editable."""
        ...

    def is_element_stable(self, locator: str) -> bool:
        """Check if an element is stable (not animating)."""
        ...

    def receives_pointer_events(self, locator: str) -> bool:
        """Check if an element receives pointer events."""
        ...

    def get_element_states(self, locator: str) -> Set[str]:
        """Get all current states of an element."""
        ...


class PreValidator:
    """Pre-validation service for element actionability.

    Performs pre-flight checks on elements before action execution to:
    1. Detect issues early with clear error messages
    2. Avoid wasting time on actions that will fail
    3. Provide actionable feedback for debugging

    The checks performed depend on the action type. For example:
    - click: requires visible, enabled, stable
    - fill: requires visible, enabled, editable
    - hover: requires visible, stable
    """

    # Required element states for each action type
    REQUIRED_STATES_FOR_ACTION: dict[str, Set[str]] = {
        "click": {"visible", "enabled", "stable"},
        "fill": {"visible", "enabled", "editable"},
        "type": {"visible", "enabled", "editable"},
        "select": {"visible", "enabled"},
        "check": {"visible", "enabled"},
        "uncheck": {"visible", "enabled"},
        "hover": {"visible", "stable"},
        "focus": {"visible"},
        "scroll_to": {"visible"},
        "double_click": {"visible", "enabled", "stable"},
        "right_click": {"visible", "enabled", "stable"},
        "press_keys": {"visible", "enabled"},
        "clear": {"visible", "enabled", "editable"},
    }

    # Actions that don't require element pre-validation
    NO_VALIDATION_ACTIONS: Set[str] = {
        "navigate",
        "go_to",
        "reload",
        "go_back",
        "go_forward",
        "get_url",
        "get_title",
        "get_page_source",
        "get_aria_snapshot",
        "take_screenshot",
        "wait_for_navigation",
        "wait_for_load_state",
        "close_browser",
        "close_page",
        "new_page",
        "switch_page",
        "evaluate_javascript",
        "execute_javascript",
    }

    def __init__(self) -> None:
        """Initialize the PreValidator."""
        self._check_cache: dict[str, Set[str]] = {}

    def should_validate(self, action_type: str) -> bool:
        """Determine if an action requires pre-validation.

        Args:
            action_type: The type of action to check

        Returns:
            True if the action requires element validation
        """
        action_lower = action_type.lower().replace(" ", "_")
        return action_lower not in self.NO_VALIDATION_ACTIONS

    def get_required_states(self, action_type: str) -> Set[str]:
        """Get the required element states for an action type.

        Args:
            action_type: The type of action

        Returns:
            Set of required state names
        """
        action_lower = action_type.lower().replace(" ", "_")
        return self.REQUIRED_STATES_FOR_ACTION.get(action_lower, {"visible"})

    def check_element_exists(
        self,
        ref: ElementRef,
        registry: ElementRegistry,
    ) -> bool:
        """Check if an element reference exists in the registry.

        Args:
            ref: The element reference to check
            registry: The element registry to look up

        Returns:
            True if the element exists in the registry
        """
        try:
            return registry.has_ref(ref)
        except Exception as e:
            logger.warning(f"Error checking element existence for {ref}: {e}")
            return False

    def check_registry_fresh(self, registry: ElementRegistry) -> bool:
        """Check if the element registry is still fresh.

        Args:
            registry: The element registry to check

        Returns:
            True if the registry is not stale
        """
        try:
            return not registry.is_stale()
        except Exception as e:
            logger.warning(f"Error checking registry freshness: {e}")
            return False

    def check_element_visible(
        self,
        locator: str,
        browser_adapter: BrowserAdapter,
    ) -> bool:
        """Check if an element is visible.

        Args:
            locator: The element locator string
            browser_adapter: The browser adapter for checks

        Returns:
            True if the element is visible
        """
        try:
            return browser_adapter.is_element_visible(locator)
        except Exception as e:
            logger.warning(f"Error checking element visibility: {e}")
            return False

    def check_element_enabled(
        self,
        locator: str,
        browser_adapter: BrowserAdapter,
    ) -> bool:
        """Check if an element is enabled.

        Args:
            locator: The element locator string
            browser_adapter: The browser adapter for checks

        Returns:
            True if the element is enabled
        """
        try:
            return browser_adapter.is_element_enabled(locator)
        except Exception as e:
            logger.warning(f"Error checking element enabled state: {e}")
            return False

    def check_element_editable(
        self,
        locator: str,
        browser_adapter: BrowserAdapter,
    ) -> bool:
        """Check if an element is editable.

        Args:
            locator: The element locator string
            browser_adapter: The browser adapter for checks

        Returns:
            True if the element is editable
        """
        try:
            return browser_adapter.is_element_editable(locator)
        except Exception as e:
            logger.warning(f"Error checking element editable state: {e}")
            return False

    def check_element_stable(
        self,
        locator: str,
        browser_adapter: BrowserAdapter,
    ) -> bool:
        """Check if an element is stable (not animating).

        Args:
            locator: The element locator string
            browser_adapter: The browser adapter for checks

        Returns:
            True if the element is stable
        """
        try:
            return browser_adapter.is_element_stable(locator)
        except Exception as e:
            logger.warning(f"Error checking element stability: {e}")
            # Default to true if check fails - stability is nice-to-have
            return True

    def check_receives_pointer_events(
        self,
        locator: str,
        browser_adapter: BrowserAdapter,
    ) -> bool:
        """Check if an element receives pointer events.

        Args:
            locator: The element locator string
            browser_adapter: The browser adapter for checks

        Returns:
            True if the element receives pointer events
        """
        try:
            return browser_adapter.receives_pointer_events(locator)
        except Exception as e:
            logger.warning(f"Error checking pointer events: {e}")
            # Default to true if check fails
            return True

    def get_element_states(
        self,
        locator: str,
        browser_adapter: BrowserAdapter,
    ) -> Set[str]:
        """Get all current states of an element.

        Args:
            locator: The element locator string
            browser_adapter: The browser adapter for checks

        Returns:
            Set of current state names
        """
        try:
            return browser_adapter.get_element_states(locator)
        except Exception as e:
            logger.warning(f"Error getting element states: {e}")
            return set()

    def run_all_checks(
        self,
        action_type: str,
        ref: ElementRef,
        registry: ElementRegistry,
        browser_adapter: BrowserAdapter,
    ) -> PreValidationResult:
        """Run all pre-validation checks for an action.

        Performs a complete set of pre-validation checks appropriate for
        the given action type and element.

        Args:
            action_type: The type of action to validate for
            ref: The element reference to validate
            registry: The element registry for locator lookup
            browser_adapter: The browser adapter for state checks

        Returns:
            PreValidationResult with check outcomes
        """
        # Check if validation is needed for this action
        if not self.should_validate(action_type):
            logger.debug(f"Skipping validation for action type: {action_type}")
            return PreValidationResult.skipped()

        checks_performed: list[str] = []
        failed_checks: list[str] = []

        # Check 1: Registry is fresh
        checks_performed.append("registry_fresh")
        if not self.check_registry_fresh(registry):
            failed_checks.append("registry_fresh")
            return PreValidationResult.failure(
                checks_performed=checks_performed,
                failed_checks=failed_checks,
                current_states=set(),
                missing_states={"registry_fresh"},
            )

        # Check 2: Element exists in registry
        checks_performed.append("element_exists")
        if not self.check_element_exists(ref, registry):
            failed_checks.append("element_exists")
            return PreValidationResult.failure(
                checks_performed=checks_performed,
                failed_checks=failed_checks,
                current_states=set(),
                missing_states={"exists"},
            )

        # Get locator for remaining checks
        try:
            locator = registry.get_locator(ref)
            locator_str = locator.to_browser_library()
        except Exception as e:
            logger.error(f"Failed to get locator for {ref}: {e}")
            failed_checks.append("locator_resolution")
            return PreValidationResult.failure(
                checks_performed=["locator_resolution"],
                failed_checks=["locator_resolution"],
                current_states=set(),
                missing_states={"locator"},
            )

        # Get required states for this action
        required_states = self.get_required_states(action_type)

        # Get current element states
        current_states = self.get_element_states(locator_str, browser_adapter)

        # Check each required state
        missing_states: Set[str] = set()

        if "visible" in required_states:
            checks_performed.append("visible")
            if not self.check_element_visible(locator_str, browser_adapter):
                failed_checks.append("visible")
                missing_states.add("visible")

        if "enabled" in required_states:
            checks_performed.append("enabled")
            if not self.check_element_enabled(locator_str, browser_adapter):
                failed_checks.append("enabled")
                missing_states.add("enabled")

        if "editable" in required_states:
            checks_performed.append("editable")
            if not self.check_element_editable(locator_str, browser_adapter):
                failed_checks.append("editable")
                missing_states.add("editable")

        if "stable" in required_states:
            checks_performed.append("stable")
            if not self.check_element_stable(locator_str, browser_adapter):
                failed_checks.append("stable")
                missing_states.add("stable")

        # Additional pointer events check for click actions
        if action_type.lower() in {"click", "double_click", "right_click"}:
            checks_performed.append("receives_pointer_events")
            if not self.check_receives_pointer_events(locator_str, browser_adapter):
                failed_checks.append("receives_pointer_events")
                missing_states.add("receives_pointer_events")

        # Build result
        if failed_checks:
            return PreValidationResult.failure(
                checks_performed=checks_performed,
                failed_checks=failed_checks,
                current_states=current_states,
                missing_states=missing_states,
            )

        return PreValidationResult.success(
            checks_performed=checks_performed,
            current_states=current_states,
        )

    def validate_quick(
        self,
        action_type: str,
        ref: ElementRef,
        registry: ElementRegistry,
    ) -> PreValidationResult:
        """Perform quick validation without browser state checks.

        This is a lightweight validation that only checks:
        1. Registry freshness
        2. Element existence in registry

        Use this for performance-critical paths where full validation
        is too slow.

        Args:
            action_type: The type of action to validate for
            ref: The element reference to validate
            registry: The element registry for lookup

        Returns:
            PreValidationResult with check outcomes
        """
        if not self.should_validate(action_type):
            return PreValidationResult.skipped()

        checks_performed: list[str] = []
        failed_checks: list[str] = []

        # Check registry freshness
        checks_performed.append("registry_fresh")
        if not self.check_registry_fresh(registry):
            failed_checks.append("registry_fresh")
            return PreValidationResult.failure(
                checks_performed=checks_performed,
                failed_checks=failed_checks,
                current_states=set(),
                missing_states={"registry_fresh"},
            )

        # Check element exists
        checks_performed.append("element_exists")
        if not self.check_element_exists(ref, registry):
            failed_checks.append("element_exists")
            return PreValidationResult.failure(
                checks_performed=checks_performed,
                failed_checks=failed_checks,
                current_states=set(),
                missing_states={"exists"},
            )

        return PreValidationResult.success(
            checks_performed=checks_performed,
            current_states={"exists"},
        )
