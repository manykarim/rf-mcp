"""Timeout Domain Entities.

This module contains entities for the Timeout bounded context.
Entities have identity and lifecycle, unlike value objects.
"""

from enum import Enum
from typing import Literal, FrozenSet


class ActionType(Enum):
    """Types of browser actions with their associated timeout behaviors.

    Actions are categorized by their expected duration:
    - Navigation actions: Page loads, refreshes (use long timeout)
    - Element actions: Clicks, typing, selections (use short timeout)
    - Read actions: Getting text, attributes, state (minimal timeout)
    """

    # Navigation actions (use navigation timeout - 60s default)
    NAVIGATE = "navigate"
    RELOAD = "reload"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    WAIT_FOR_NAVIGATION = "wait_for_navigation"

    # Element actions (use action timeout - 5s default)
    CLICK = "click"
    FILL = "fill"
    TYPE = "type"
    PRESS = "press"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    HOVER = "hover"
    FOCUS = "focus"
    DRAG_AND_DROP = "drag_and_drop"
    UPLOAD_FILE = "upload_file"
    SCROLL = "scroll"

    # Read actions (minimal timeout - 2s default)
    GET_TEXT = "get_text"
    GET_ATTRIBUTE = "get_attribute"
    GET_STATE = "get_state"
    GET_VALUE = "get_value"
    GET_PROPERTY = "get_property"
    IS_VISIBLE = "is_visible"
    IS_ENABLED = "is_enabled"
    IS_CHECKED = "is_checked"

    # Snapshot actions (use read timeout)
    SNAPSHOT = "snapshot"
    GET_ARIA_SNAPSHOT = "get_aria_snapshot"
    SCREENSHOT = "screenshot"

    # Wait actions (use assertion timeout)
    WAIT_FOR_ELEMENT = "wait_for_element"
    WAIT_FOR_TEXT = "wait_for_text"
    WAIT_FOR_STATE = "wait_for_state"


class TimeoutCategory:
    """Categorization of action types by their timeout behavior.

    This class provides static categorization of actions into groups
    that share the same timeout defaults. It enables the TimeoutPolicy
    to automatically select appropriate timeouts based on action type.

    Categories:
        - navigation: Long-running operations like page loads
        - action: Interactive element operations
        - read: Quick data retrieval operations
        - assertion: Wait and retry operations
    """

    NAVIGATION_ACTIONS: FrozenSet[ActionType] = frozenset({
        ActionType.NAVIGATE,
        ActionType.RELOAD,
        ActionType.GO_BACK,
        ActionType.GO_FORWARD,
        ActionType.WAIT_FOR_NAVIGATION,
    })

    ELEMENT_ACTIONS: FrozenSet[ActionType] = frozenset({
        ActionType.CLICK,
        ActionType.FILL,
        ActionType.TYPE,
        ActionType.PRESS,
        ActionType.SELECT,
        ActionType.CHECK,
        ActionType.UNCHECK,
        ActionType.HOVER,
        ActionType.FOCUS,
        ActionType.DRAG_AND_DROP,
        ActionType.UPLOAD_FILE,
        ActionType.SCROLL,
    })

    READ_ACTIONS: FrozenSet[ActionType] = frozenset({
        ActionType.GET_TEXT,
        ActionType.GET_ATTRIBUTE,
        ActionType.GET_STATE,
        ActionType.GET_VALUE,
        ActionType.GET_PROPERTY,
        ActionType.IS_VISIBLE,
        ActionType.IS_ENABLED,
        ActionType.IS_CHECKED,
        ActionType.SNAPSHOT,
        ActionType.GET_ARIA_SNAPSHOT,
        ActionType.SCREENSHOT,
    })

    ASSERTION_ACTIONS: FrozenSet[ActionType] = frozenset({
        ActionType.WAIT_FOR_ELEMENT,
        ActionType.WAIT_FOR_TEXT,
        ActionType.WAIT_FOR_STATE,
    })

    @classmethod
    def categorize(cls, action: ActionType) -> Literal["navigation", "action", "read", "assertion"]:
        """Determine the timeout category for an action type.

        Args:
            action: The ActionType to categorize.

        Returns:
            The category string: "navigation", "action", "read", or "assertion".

        Examples:
            >>> TimeoutCategory.categorize(ActionType.NAVIGATE)
            'navigation'
            >>> TimeoutCategory.categorize(ActionType.CLICK)
            'action'
            >>> TimeoutCategory.categorize(ActionType.GET_TEXT)
            'read'
        """
        if action in cls.NAVIGATION_ACTIONS:
            return "navigation"
        elif action in cls.ELEMENT_ACTIONS:
            return "action"
        elif action in cls.ASSERTION_ACTIONS:
            return "assertion"
        else:
            return "read"

    @classmethod
    def get_all_actions_for_category(cls, category: Literal["navigation", "action", "read", "assertion"]) -> FrozenSet[ActionType]:
        """Get all action types belonging to a category.

        Args:
            category: The category to look up.

        Returns:
            FrozenSet of ActionType values in the category.
        """
        category_map = {
            "navigation": cls.NAVIGATION_ACTIONS,
            "action": cls.ELEMENT_ACTIONS,
            "read": cls.READ_ACTIONS,
            "assertion": cls.ASSERTION_ACTIONS,
        }
        return category_map.get(category, frozenset())
