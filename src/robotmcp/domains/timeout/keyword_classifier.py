"""Keyword classification for automatic timeout selection.

This module provides mapping between Robot Framework keyword names and
ActionType values to enable automatic timeout selection based on the
keyword being executed.

The dual timeout strategy uses:
- 5 seconds for element actions (clicks, typing, etc.)
- 60 seconds for navigation operations (page loads, reloads)
- 2 seconds for read operations (getting text, attributes)
- 10 seconds for wait/assertion operations
"""

from typing import Dict, Set

from .entities import ActionType


# Keywords mapped to CLICK action type (element interaction - 5s default)
CLICK_KEYWORDS: Set[str] = {
    "click",
    "click_element",
    "click_button",
    "click_link",
    "double_click",
    "double_click_element",
    "right_click",
    "right_click_element",
    "context_click",
    "hover",
    "mouse_over",
    "mouse_down",
    "mouse_up",
}

# Keywords mapped to FILL action type (text input - 5s default)
FILL_KEYWORDS: Set[str] = {
    "fill_text",
    "input_text",
    "type_text",
    "fill_secret",
    "input_password",
    "clear_element_text",
    "clear_text",
    "press_keys",
    "press_key",
}

# Keywords mapped to NAVIGATE action type (page navigation - 60s default)
NAVIGATION_KEYWORDS: Set[str] = {
    "go_to",
    "go_back",
    "go_forward",
    "reload",
    "new_page",
    "new_context",
    "new_browser",
    "open_browser",
    "close_browser",
    "wait_for_navigation",
    "wait_until_network_is_idle",
}

# Keywords mapped to SELECT action type (dropdown/checkbox - 5s default)
SELECT_KEYWORDS: Set[str] = {
    "select_options",
    "select_from_list",
    "select_from_list_by_value",
    "select_from_list_by_label",
    "select_from_list_by_index",
    "select_checkbox",
    "unselect_checkbox",
    "check_checkbox",
    "uncheck_checkbox",
}

# Keywords mapped to GET_TEXT action type (read operations - 2s default)
READ_KEYWORDS: Set[str] = {
    "get_text",
    "get_value",
    "get_attribute",
    "get_element_states",
    "get_element_count",
    "get_title",
    "get_url",
    "get_location",
    "is_visible",
    "is_enabled",
    "is_checked",
    "take_screenshot",
    "get_page_source",
}

# PlatynUI Desktop keywords mapped to CLICK action type (pointer interactions - 5s)
PLATYNUI_POINTER_KEYWORDS: Set[str] = {
    "pointer_click",
    "pointer_multi_click",
    "pointer_press",
    "pointer_release",
    "pointer_move_to",
}

# PlatynUI Desktop keywords mapped to FILL action type (keyboard input - 5s)
PLATYNUI_KEYBOARD_KEYWORDS: Set[str] = {
    "keyboard_type",
    "keyboard_press",
    "keyboard_release",
}

# PlatynUI Desktop keywords mapped to CLICK action type (window management - 5s)
PLATYNUI_WINDOW_KEYWORDS: Set[str] = {
    "activate",
    "maximize",
    "minimize",
    "restore",
    "focus",
    "close",
}

# PlatynUI Desktop keywords mapped to GET_TEXT action type (read operations - 2s)
PLATYNUI_READ_KEYWORDS: Set[str] = {
    "get_attribute",
    "query",
    "get_pointer_position",
}

# PlatynUI Desktop keywords mapped to GET_TEXT action type (screenshot - 2s)
PLATYNUI_SCREENSHOT_KEYWORDS: Set[str] = {
    "take_screenshot",
}

# PlatynUI Desktop keywords mapped to GET_TEXT action type (visual - 2s)
PLATYNUI_HIGHLIGHT_KEYWORDS: Set[str] = {
    "highlight",
}

# Keywords mapped to WAIT_FOR_ELEMENT action type (wait/assertions - 10s default)
WAIT_KEYWORDS: Set[str] = {
    "wait_for_elements_state",
    "wait_for_element",
    "wait_until_element_is_visible",
    "wait_until_element_is_enabled",
    "wait_until_page_contains",
    "wait_until_page_contains_element",
    "sleep",
    "wait",
}

# Mapping from keyword sets to ActionType
_KEYWORD_TO_ACTION: Dict[str, ActionType] = {}


def _build_keyword_mapping() -> None:
    """Build the keyword to ActionType mapping on module load."""
    global _KEYWORD_TO_ACTION

    for keyword in CLICK_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.CLICK

    for keyword in FILL_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.FILL

    for keyword in NAVIGATION_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.NAVIGATE

    for keyword in SELECT_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.SELECT

    for keyword in READ_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.GET_TEXT

    for keyword in WAIT_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.WAIT_FOR_ELEMENT

    for keyword in PLATYNUI_POINTER_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.CLICK

    for keyword in PLATYNUI_KEYBOARD_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.FILL

    for keyword in PLATYNUI_WINDOW_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.CLICK

    for keyword in PLATYNUI_READ_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.GET_TEXT

    for keyword in PLATYNUI_SCREENSHOT_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.SCREENSHOT

    for keyword in PLATYNUI_HIGHLIGHT_KEYWORDS:
        _KEYWORD_TO_ACTION[keyword] = ActionType.GET_TEXT


# Build mapping on module load
_build_keyword_mapping()


def normalize_keyword(keyword: str) -> str:
    """Normalize a keyword name for lookup.

    Handles variations in keyword naming:
    - Converts to lowercase
    - Replaces spaces with underscores
    - Replaces hyphens with underscores

    Args:
        keyword: Robot Framework keyword name

    Returns:
        Normalized keyword string for lookup
    """
    return keyword.lower().replace(" ", "_").replace("-", "_")


def classify_keyword(keyword: str) -> ActionType:
    """Classify a keyword to determine appropriate timeout.

    Maps Robot Framework keyword names to ActionType values for
    automatic timeout selection.

    Args:
        keyword: Robot Framework keyword name

    Returns:
        ActionType for timeout lookup

    Examples:
        >>> classify_keyword("Click Element")
        ActionType.CLICK
        >>> classify_keyword("go_to")
        ActionType.NAVIGATE
        >>> classify_keyword("Get Text")
        ActionType.GET_TEXT
    """
    normalized = normalize_keyword(keyword)

    if normalized in _KEYWORD_TO_ACTION:
        return _KEYWORD_TO_ACTION[normalized]

    # Default to CLICK for unknown keywords (uses action timeout - 5s)
    # This is a safe default as most keywords are element interactions
    return ActionType.CLICK


def get_timeout_for_keyword(keyword: str, timeout_policy: "TimeoutPolicy") -> int:
    """Get timeout in milliseconds for a keyword.

    Combines keyword classification with timeout policy lookup
    to provide the appropriate timeout value.

    Args:
        keyword: Robot Framework keyword name
        timeout_policy: TimeoutPolicy instance for timeout lookup

    Returns:
        Timeout in milliseconds

    Examples:
        >>> from robotmcp.domains.timeout import TimeoutPolicy
        >>> policy = TimeoutPolicy.create_default("session_1")
        >>> get_timeout_for_keyword("Click Element", policy)
        5000
        >>> get_timeout_for_keyword("Go To", policy)
        60000
    """
    action_type = classify_keyword(keyword)
    return timeout_policy.get_timeout_for(action_type).value


def get_all_keywords_for_action(action_type: ActionType) -> Set[str]:
    """Get all keywords that map to a specific action type.

    Args:
        action_type: The ActionType to look up

    Returns:
        Set of keyword names that map to the action type
    """
    return {kw for kw, action in _KEYWORD_TO_ACTION.items() if action == action_type}


def is_navigation_keyword(keyword: str) -> bool:
    """Check if a keyword is a navigation operation.

    Args:
        keyword: Robot Framework keyword name

    Returns:
        True if the keyword is classified as navigation
    """
    return classify_keyword(keyword) == ActionType.NAVIGATE


def is_read_keyword(keyword: str) -> bool:
    """Check if a keyword is a read operation.

    Args:
        keyword: Robot Framework keyword name

    Returns:
        True if the keyword is classified as a read operation
    """
    return classify_keyword(keyword) == ActionType.GET_TEXT


def is_wait_keyword(keyword: str) -> bool:
    """Check if a keyword is a wait/assertion operation.

    Args:
        keyword: Robot Framework keyword name

    Returns:
        True if the keyword is classified as a wait operation
    """
    return classify_keyword(keyword) == ActionType.WAIT_FOR_ELEMENT
