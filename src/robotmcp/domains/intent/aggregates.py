"""Intent Domain Aggregate Root.

The IntentRegistry is the aggregate root for the Intent bounded context.
It owns all IntentMappings and enforces invariants across the collection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .entities import IntentMapping
from .value_objects import (
    FallbackStep,
    IntentVerb,
    NavigateFallbackSequence,
)


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

    # Navigate fallback sequences: library → list of sequences (order matters)
    _navigate_fallbacks: Dict[str, List[NavigateFallbackSequence]] = field(
        default_factory=dict
    )

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

    def register_navigate_fallback(
        self, sequence: NavigateFallbackSequence
    ) -> None:
        """Register a navigate fallback sequence for a library."""
        if sequence.library not in self._navigate_fallbacks:
            self._navigate_fallbacks[sequence.library] = []
        self._navigate_fallbacks[sequence.library].append(sequence)

    def get_navigate_fallback(
        self, library: str, error_message: str
    ) -> Optional[NavigateFallbackSequence]:
        """Find a fallback sequence matching the error for this library.

        Returns the first matching sequence (order matters: more
        specific patterns should be registered first).
        """
        sequences = self._navigate_fallbacks.get(library, [])
        for seq in sequences:
            if seq.matches_error(error_message):
                return seq
        return None

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
        for seq in _builtin_navigate_fallbacks():
            registry.register_navigate_fallback(seq)
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


def _resolve_select_match(match_option: str, value: str | None) -> str:
    """Resolve the effective select-match strategy.

    Explicit strategies (label/value/index/text) pass through unchanged.
    The opt-in ``auto`` heuristic resolves to ``value`` for purely-numeric
    values and ``label`` otherwise — but is OPT-IN only because numeric
    visible labels (years, amounts, ids) would mis-route. Default is
    ``label`` to match Robot Framework's "select by visible text" semantics.
    """
    if match_option in ("label", "value", "index", "text"):
        return match_option
    # AUTO heuristic (opt-in). Strip leading minus so negative integers
    # like "-1" still classify as numeric.
    if value and value.strip().lstrip("-").isdigit():
        return "value"
    return "label"


_SELENIUM_SELECT_KEYWORDS: Dict[str, str] = {
    "label": "Select From List By Label",
    "text": "Select From List By Label",
    "value": "Select From List By Value",
    "index": "Select From List By Index",
}


def _get_selenium_select_keyword(match_opt: str, value: str | None) -> str:
    """Return the appropriate SeleniumLibrary select keyword for the
    resolved match strategy."""
    strategy = _resolve_select_match(match_opt, value)
    return _SELENIUM_SELECT_KEYWORDS.get(strategy, "Select From List By Label")


def _select_browser_transformer(target, value, normalized_locator, options):
    """Browser Library select: ``Select Options By <selector> <attr> <value>``.

    The attribute is driven by ``options["match"]`` (passed via the
    ``intent_action(match=...)`` parameter). Defaults to ``label``.
    """
    args = []
    if normalized_locator:
        args.append(normalized_locator.value)
    elif target:
        args.append(target.locator)
    match_opt = (options or {}).get("match", "label")
    args.append(_resolve_select_match(match_opt, value))
    if value:
        args.append(value)
    return args


def _select_selenium_transformer(target, value, normalized_locator, options):
    """SeleniumLibrary: builds args for the chosen Select From List By X.

    The adapter selects the actual keyword name via
    ``_get_selenium_select_keyword`` based on ``options["match"]``; this
    transformer just produces the args (locator + value).
    """
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
# OBS-06: extract intent — DOM/page-state read with mode-aware dispatch
# ============================================================

# Mode → keyword name per library. Looked up by the adapter to substitute
# the dispatched keyword after the registry returns the default mapping.
_BROWSER_EXTRACT_KEYWORDS: Dict[str, str] = {
    "text":      "Get Text",
    "attribute": "Get Attribute",
    "count":     "Get Element Count",
    "value":     "Get Property",
    "url":       "Get Url",
    "title":     "Get Title",
}

_SELENIUM_EXTRACT_KEYWORDS: Dict[str, str] = {
    "text":      "Get Text",
    "attribute": "Get Element Attribute",
    "count":     "Get Element Count",
    "value":     "Get Value",
    "url":       "Get Location",
    "title":     "Get Title",
}

_APPIUM_EXTRACT_KEYWORDS: Dict[str, str] = {
    # AppiumLibrary is more limited; only the universally-supported reads
    # are exposed. Other modes fall back to "text" with a warning.
    "text":      "Get Text",
    "attribute": "Get Element Attribute",
    "count":     "Get Matching Xpath Count",
}

# Modes that do NOT take a target. The transformer omits the locator
# from the args list; the resolver tolerates missing target because the
# EXTRACT mapping is registered with requires_target=False.
_EXTRACT_MODES_WITHOUT_TARGET: frozenset[str] = frozenset({"url", "title"})


def _get_browser_extract_keyword(mode: str) -> str:
    """Map an extract mode to its Browser library keyword.

    Unknown modes default to ``Get Text`` (safe fallback — same as the
    EXTRACT_TEXT intent's behaviour). Callers should validate ``mode``
    against the ``ExtractMode`` Literal first, so this fallback only
    fires for direct-test inputs.
    """
    return _BROWSER_EXTRACT_KEYWORDS.get((mode or "text").lower(), "Get Text")


def _get_selenium_extract_keyword(mode: str) -> str:
    """Map an extract mode to its SeleniumLibrary keyword."""
    return _SELENIUM_EXTRACT_KEYWORDS.get((mode or "text").lower(), "Get Text")


def _get_appium_extract_keyword(mode: str) -> str:
    """Map an extract mode to its AppiumLibrary keyword.

    Unsupported modes (value/url/title) fall back to ``Get Text`` — those
    aren't first-class in AppiumLibrary. Callers should normally restrict
    extract to web sessions; this is a defensive fallback.
    """
    return _APPIUM_EXTRACT_KEYWORDS.get((mode or "text").lower(), "Get Text")


def _extract_browser_transformer(target, value, normalized_locator, options):
    """Browser Library extract: builds args based on ``options["mode"]``.

    Per-mode argument shapes:
        text:      [locator]
        attribute: [locator, attribute_name]
        count:     [locator]
        value:     [locator, "value"]    — Browser uses Get Property(sel, attr)
        url:       []
        title:     []

    Raises:
        ValueError: when a mode that requires a target is invoked without one,
            or when mode=attribute is used without ``options["attribute_name"]``.
    """
    opts = options or {}
    mode = (opts.get("mode") or "text").lower()
    args: List[str] = []

    if mode in _EXTRACT_MODES_WITHOUT_TARGET:
        return args  # no locator, no other args

    # All other modes require a target.
    locator: Optional[str] = None
    if normalized_locator is not None:
        locator = normalized_locator.value
    elif target is not None:
        locator = target.locator
    if locator is None:
        raise ValueError(
            f"intent_action(intent='extract', mode='{mode}') requires a target; "
            f"only mode='url' and mode='title' may omit it."
        )
    args.append(locator)

    if mode == "attribute":
        attr_name = opts.get("attribute_name")
        if not attr_name:
            raise ValueError(
                "intent_action(intent='extract', mode='attribute') requires "
                "the attribute_name parameter (e.g. attribute_name='href')."
            )
        args.append(attr_name)
    elif mode == "value":
        # Browser's Get Property takes (selector, attribute). For the
        # `value` mode we hard-wire attribute="value" — that's the DOM
        # property name for input values.
        args.append("value")
    # text / count: no further args.

    return args


def _extract_selenium_transformer(target, value, normalized_locator, options):
    """SeleniumLibrary extract: builds args based on ``options["mode"]``.

    Per-mode argument shapes (SeleniumLibrary keyword signatures):
        text:      [locator]                — Get Text(locator)
        attribute: [locator, attribute_name] — Get Element Attribute(locator, attribute)
        count:     [locator]                — Get Element Count(locator)
        value:     [locator]                — Get Value(locator)
        url:       []                       — Get Location()
        title:     []                       — Get Title()
    """
    opts = options or {}
    mode = (opts.get("mode") or "text").lower()
    args: List[str] = []

    if mode in _EXTRACT_MODES_WITHOUT_TARGET:
        return args

    locator: Optional[str] = None
    if normalized_locator is not None:
        locator = normalized_locator.value
    elif target is not None:
        locator = target.locator
    if locator is None:
        raise ValueError(
            f"intent_action(intent='extract', mode='{mode}') requires a target; "
            f"only mode='url' and mode='title' may omit it."
        )
    args.append(locator)

    if mode == "attribute":
        attr_name = opts.get("attribute_name")
        if not attr_name:
            raise ValueError(
                "intent_action(intent='extract', mode='attribute') requires "
                "the attribute_name parameter (e.g. attribute_name='value')."
            )
        args.append(attr_name)
    # text / count / value: no further args. SeleniumLibrary's Get Value
    # takes only the locator (no attribute name, unlike Browser).

    return args


# Modes that intentionally accept multi-match (count) — pre-validation
# would falsely reject these because the test "an element matches the
# locator" is the wrong question when you're counting matches.
EXTRACT_MULTI_MATCH_MODES: frozenset[str] = frozenset({"count"})


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
            # Browser.Click(selector, button) takes no `force=` arg.
            # Click With Options(selector, *clickOptions) does. When
            # intent_action is called with force=True we swap to the
            # latter so the documented escape hatch actually works.
            force_keyword="Click With Options",
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
            intent_verb=IntentVerb.EXTRACT,
            library="Browser",
            # Default keyword for the most common mode (text). The adapter
            # overrides this based on ``options["mode"]`` via
            # ``_get_browser_extract_keyword``.
            keyword="Get Text",
            # Target is OPTIONAL — modes url/title don't need one. The
            # transformer raises if a target-requiring mode is invoked
            # without one, so the validation is mode-aware.
            requires_target=False,
            requires_value=False,
            argument_transformer=_extract_browser_transformer,
            timeout_category="read",
            notes=(
                "OBS-06. mode={text,attribute,count,value,url,title} dispatches "
                "to Get Text / Get Attribute / Get Element Count / Get Property "
                "/ Get Url / Get Title."
            ),
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
            intent_verb=IntentVerb.EXTRACT,
            library="SeleniumLibrary",
            keyword="Get Text",  # default; adapter swaps based on mode
            requires_target=False,
            requires_value=False,
            argument_transformer=_extract_selenium_transformer,
            timeout_category="read",
            notes=(
                "OBS-06. mode={text,attribute,count,value,url,title} dispatches "
                "to Get Text / Get Element Attribute / Get Element Count / "
                "Get Value / Get Location / Get Title."
            ),
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
            intent_verb=IntentVerb.EXTRACT,
            library="AppiumLibrary",
            keyword="Get Text",  # default; adapter swaps based on mode
            requires_target=False,
            requires_value=False,
            # AppiumLibrary uses the same arg shape as SeleniumLibrary for
            # the modes it supports (text / attribute / count). The
            # unsupported modes (value, url, title) fall back to Get Text
            # — see _get_appium_extract_keyword.
            argument_transformer=_extract_selenium_transformer,
            timeout_category="read",
            notes=(
                "OBS-06. mode={text,attribute,count} supported; value/url/title "
                "fall back to Get Text (AppiumLibrary has no first-class "
                "equivalents in mobile context)."
            ),
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


# ============================================================
# Built-in navigate fallback sequences
# ============================================================

def _builtin_navigate_fallbacks() -> List[NavigateFallbackSequence]:
    """Built-in fallback sequences for navigate intent recovery.

    Order within each library matters: more specific patterns first.
    """
    return [
        # Browser Library: no browser open → New Browser + New Page
        NavigateFallbackSequence(
            library="Browser",
            error_pattern=r"no browser|browser.*not.*open|no open browser",
            steps=(
                FallbackStep(
                    keyword="New Browser",
                    arguments=("headless=False",),
                    reason="Open browser first",
                ),
                FallbackStep(
                    keyword="New Page",
                    arguments=(),
                    reason="Open a page/tab",
                ),
            ),
            description="Open browser and page before navigating",
        ),
        # Browser Library: no page/tab → New Page only
        NavigateFallbackSequence(
            library="Browser",
            error_pattern=r"target closed|page.*closed|no page|no context",
            steps=(
                FallbackStep(
                    keyword="New Page",
                    arguments=(),
                    reason="Reopen page/tab",
                ),
            ),
            description="Reopen page after target closed",
        ),
        # SeleniumLibrary: no browser open → Open Browser
        NavigateFallbackSequence(
            library="SeleniumLibrary",
            error_pattern=r"No browser is open|invalid session id|Session.*not.*found",
            steps=(
                FallbackStep(
                    keyword="Open Browser",
                    arguments=("about:blank", "chrome"),
                    reason="Open browser first",
                ),
            ),
            description="Open browser before navigating",
        ),
    ]
