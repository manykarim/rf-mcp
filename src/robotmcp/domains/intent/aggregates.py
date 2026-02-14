"""Intent Domain Aggregate Root.

The IntentRegistry is the aggregate root for the Intent bounded context.
It owns all IntentMappings and enforces invariants across the collection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .entities import IntentMapping
from .value_objects import IntentVerb


@dataclass
class IntentRegistry:
    """Registry of intent-to-keyword mappings across all libraries.

    The registry is populated at startup with the built-in mappings
    and can be extended at runtime for custom libraries.

    Invariants:
        - Each (intent_verb, library) pair has at most one mapping
        - At least one library must have a mapping for each intent verb
        - Mappings cannot be removed, only overridden

    Concurrency:
        The registry is read-heavy, write-rare. Writes occur only at
        initialization or when a new library plugin is registered.
        No locking is needed because writes are atomic dict assignments
        and reads tolerate stale data (worst case: one execution uses
        the old mapping before the new one is visible).
    """
    _mappings: Dict[Tuple[IntentVerb, str], IntentMapping] = field(
        default_factory=dict
    )

    # Track which libraries have been registered
    _registered_libraries: Set[str] = field(default_factory=set)

    def register(self, mapping: IntentMapping) -> None:
        """Register or override an intent mapping.

        Args:
            mapping: The IntentMapping to register

        Raises:
            ValueError: If mapping has invalid fields
        """
        if not mapping.keyword:
            raise ValueError("IntentMapping.keyword must not be empty")
        if not mapping.library:
            raise ValueError("IntentMapping.library must not be empty")

        self._mappings[mapping.mapping_key] = mapping
        self._registered_libraries.add(mapping.library)

    def register_all(self, mappings: List[IntentMapping]) -> None:
        """Register multiple mappings at once.

        Args:
            mappings: List of IntentMappings to register
        """
        for mapping in mappings:
            self.register(mapping)

    def resolve(
        self, intent_verb: IntentVerb, library: str
    ) -> Optional[IntentMapping]:
        """Look up mapping for an intent verb and library.

        Args:
            intent_verb: The intent to resolve
            library: The target library name

        Returns:
            IntentMapping if found, None otherwise
        """
        return self._mappings.get((intent_verb, library))

    def has_mapping(self, intent_verb: IntentVerb, library: str) -> bool:
        """Check if a mapping exists."""
        return (intent_verb, library) in self._mappings

    def get_supported_intents(self, library: str) -> List[IntentVerb]:
        """Get all intent verbs supported by a given library.

        Args:
            library: Library name

        Returns:
            List of supported IntentVerb values
        """
        return [
            verb
            for verb in IntentVerb
            if (verb, library) in self._mappings
        ]

    def get_supported_libraries(self) -> Set[str]:
        """Get all libraries that have at least one mapping."""
        return frozenset(self._registered_libraries)

    def get_all_mappings(self) -> List[IntentMapping]:
        """Get all registered mappings (for diagnostics)."""
        return list(self._mappings.values())

    @classmethod
    def with_builtins(cls) -> IntentRegistry:
        """Create a registry pre-populated with built-in mappings.

        This is the standard factory method. The built-in mappings
        encode the knowledge currently split across browser_plugin.py
        and selenium_plugin.py KEYWORD_ALTERNATIVES dicts.
        """
        registry = cls()
        registry.register_all(_builtin_browser_mappings())
        registry.register_all(_builtin_selenium_mappings())
        registry.register_all(_builtin_appium_mappings())
        return registry


# ============================================================
# Argument transformer functions (module-level)
# ============================================================

def _navigate_browser_transformer(target, value, normalized_locator, options):
    """Browser Library navigate: Go To <url>."""
    url = target.locator if target else (value or "")
    return [url]


def _navigate_selenium_transformer(target, value, normalized_locator, options):
    """SeleniumLibrary navigate: Go To <url>."""
    url = target.locator if target else (value or "")
    return [url]


def _navigate_appium_transformer(target, value, normalized_locator, options):
    """AppiumLibrary navigate: Go To Url <url>."""
    url = target.locator if target else (value or "")
    return [url]


def _select_browser_transformer(target, value, normalized_locator, options):
    """Browser Library select: Select Options By <selector> label <value>."""
    args = []
    if normalized_locator:
        args.append(normalized_locator.value)
    elif target:
        args.append(target.locator)
    # Browser Library: Select Options By <selector> <attribute> <value>
    args.append("label")
    if value:
        args.append(value)
    return args


def _select_selenium_transformer(target, value, normalized_locator, options):
    """SeleniumLibrary: Select From List By Label <locator> <label>."""
    args = []
    if normalized_locator:
        args.append(normalized_locator.value)
    elif target:
        args.append(target.locator)
    if value:
        args.append(value)
    return args


def _assert_visible_browser_transformer(target, value, normalized_locator, options):
    """Browser Library: Get Element States <selector> then check 'visible'."""
    args = []
    if normalized_locator:
        args.append(normalized_locator.value)
    elif target:
        args.append(target.locator)
    args.append("visible")
    args.append("==")
    args.append("True")
    return args


def _wait_for_browser_transformer(target, value, normalized_locator, options):
    """Browser Library: Wait For Elements State <selector> visible."""
    args = []
    if normalized_locator:
        args.append(normalized_locator.value)
    elif target:
        args.append(target.locator)
    args.append("visible")
    timeout = (options or {}).get("timeout", "10s")
    args.append(f"timeout={timeout}")
    return args


def _wait_for_selenium_transformer(target, value, normalized_locator, options):
    """SeleniumLibrary: Wait Until Element Is Visible <locator> <timeout>."""
    args = []
    if normalized_locator:
        args.append(normalized_locator.value)
    elif target:
        args.append(target.locator)
    timeout = (options or {}).get("timeout", "10s")
    args.append(timeout)
    return args


# ============================================================
# Built-in mapping definitions
# ============================================================

def _builtin_browser_mappings() -> List[IntentMapping]:
    """Built-in mappings for Browser Library (Playwright)."""
    return [
        IntentMapping(
            intent_verb=IntentVerb.NAVIGATE,
            library="Browser",
            keyword="Go To",
            requires_target=True,  # target is the URL
            requires_value=False,
            argument_transformer=_navigate_browser_transformer,
            timeout_category="navigation",
            notes="Uses Go To (not New Page) for URL navigation within existing browser",
        ),
        IntentMapping(
            intent_verb=IntentVerb.CLICK,
            library="Browser",
            keyword="Click",
            requires_target=True,
            requires_value=False,
            timeout_category="action",
        ),
        IntentMapping(
            intent_verb=IntentVerb.FILL,
            library="Browser",
            keyword="Fill Text",
            requires_target=True,
            requires_value=True,
            timeout_category="action",
        ),
        IntentMapping(
            intent_verb=IntentVerb.HOVER,
            library="Browser",
            keyword="Hover",
            requires_target=True,
            requires_value=False,
            timeout_category="action",
        ),
        IntentMapping(
            intent_verb=IntentVerb.SELECT,
            library="Browser",
            keyword="Select Options By",
            requires_target=True,
            requires_value=True,
            argument_transformer=_select_browser_transformer,
            timeout_category="action",
            notes="Adds 'label' as default attribute between selector and value",
        ),
        IntentMapping(
            intent_verb=IntentVerb.ASSERT_VISIBLE,
            library="Browser",
            keyword="Get Element States",
            requires_target=True,
            requires_value=False,
            argument_transformer=_assert_visible_browser_transformer,
            timeout_category="assertion",
            notes="Checks 'visible' state flag via Get Element States",
        ),
        IntentMapping(
            intent_verb=IntentVerb.EXTRACT_TEXT,
            library="Browser",
            keyword="Get Text",
            requires_target=True,
            requires_value=False,
            timeout_category="read",
        ),
        IntentMapping(
            intent_verb=IntentVerb.WAIT_FOR,
            library="Browser",
            keyword="Wait For Elements State",
            requires_target=True,
            requires_value=False,
            argument_transformer=_wait_for_browser_transformer,
            timeout_category="assertion",
            notes="Waits for 'visible' state by default; timeout from options",
        ),
    ]


def _builtin_selenium_mappings() -> List[IntentMapping]:
    """Built-in mappings for SeleniumLibrary."""
    return [
        IntentMapping(
            intent_verb=IntentVerb.NAVIGATE,
            library="SeleniumLibrary",
            keyword="Go To",
            requires_target=True,
            requires_value=False,
            argument_transformer=_navigate_selenium_transformer,
            timeout_category="navigation",
        ),
        IntentMapping(
            intent_verb=IntentVerb.CLICK,
            library="SeleniumLibrary",
            keyword="Click Element",
            requires_target=True,
            requires_value=False,
            timeout_category="action",
            notes="SL Click Element takes 'locator' only; no 'options' like Browser Click",
        ),
        IntentMapping(
            intent_verb=IntentVerb.FILL,
            library="SeleniumLibrary",
            keyword="Input Text",
            requires_target=True,
            requires_value=True,
            timeout_category="action",
        ),
        IntentMapping(
            intent_verb=IntentVerb.HOVER,
            library="SeleniumLibrary",
            keyword="Mouse Over",
            requires_target=True,
            requires_value=False,
            timeout_category="action",
        ),
        IntentMapping(
            intent_verb=IntentVerb.SELECT,
            library="SeleniumLibrary",
            keyword="Select From List By Label",
            requires_target=True,
            requires_value=True,
            argument_transformer=_select_selenium_transformer,
            timeout_category="action",
        ),
        IntentMapping(
            intent_verb=IntentVerb.ASSERT_VISIBLE,
            library="SeleniumLibrary",
            keyword="Element Should Be Visible",
            requires_target=True,
            requires_value=False,
            timeout_category="assertion",
        ),
        IntentMapping(
            intent_verb=IntentVerb.EXTRACT_TEXT,
            library="SeleniumLibrary",
            keyword="Get Text",
            requires_target=True,
            requires_value=False,
            timeout_category="read",
        ),
        IntentMapping(
            intent_verb=IntentVerb.WAIT_FOR,
            library="SeleniumLibrary",
            keyword="Wait Until Element Is Visible",
            requires_target=True,
            requires_value=False,
            argument_transformer=_wait_for_selenium_transformer,
            timeout_category="assertion",
        ),
    ]


def _builtin_appium_mappings() -> List[IntentMapping]:
    """Built-in mappings for AppiumLibrary."""
    return [
        IntentMapping(
            intent_verb=IntentVerb.NAVIGATE,
            library="AppiumLibrary",
            keyword="Go To Url",
            requires_target=True,
            requires_value=False,
            argument_transformer=_navigate_appium_transformer,
            timeout_category="navigation",
        ),
        IntentMapping(
            intent_verb=IntentVerb.CLICK,
            library="AppiumLibrary",
            keyword="Click Element",
            requires_target=True,
            requires_value=False,
            timeout_category="action",
        ),
        IntentMapping(
            intent_verb=IntentVerb.FILL,
            library="AppiumLibrary",
            keyword="Input Text",
            requires_target=True,
            requires_value=True,
            timeout_category="action",
        ),
        # HOVER: AppiumLibrary has no hover equivalent (touch-only)
        # SELECT: AppiumLibrary has no dropdown select equivalent
        IntentMapping(
            intent_verb=IntentVerb.ASSERT_VISIBLE,
            library="AppiumLibrary",
            keyword="Element Should Be Visible",
            requires_target=True,
            requires_value=False,
            timeout_category="assertion",
        ),
        IntentMapping(
            intent_verb=IntentVerb.EXTRACT_TEXT,
            library="AppiumLibrary",
            keyword="Get Text",
            requires_target=True,
            requires_value=False,
            timeout_category="read",
        ),
        IntentMapping(
            intent_verb=IntentVerb.WAIT_FOR,
            library="AppiumLibrary",
            keyword="Wait Until Element Is Visible",
            requires_target=True,
            requires_value=False,
            argument_transformer=_wait_for_selenium_transformer,  # same arg shape
            timeout_category="assertion",
        ),
    ]
