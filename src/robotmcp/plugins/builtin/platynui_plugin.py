"""Builtin PlatynUI desktop automation plugin."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    LibraryCapabilities,
    LibraryHints,
    LibraryMetadata,
    LibraryStateProvider,
)

logger = logging.getLogger(__name__)


class PlatynUIStateProvider(LibraryStateProvider):
    """State provider for PlatynUI desktop automation sessions.

    Desktop applications have no HTML DOM, so ``get_page_source`` always
    returns ``None``.  ``get_ui_tree`` retrieves the accessibility tree
    via PlatynUI's Query keyword (stubbed until native backend is wired).
    """

    async def get_page_source(
        self,
        session: "ExecutionSession",
        *,
        full_source: bool = False,
        filtered: bool = False,
        filtering_level: str = "standard",
        include_reduced_dom: bool = True,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Desktop applications have no HTML DOM; always returns None."""
        return None

    async def get_application_state(
        self,
        session: "ExecutionSession",
    ) -> Optional[Dict[str, Any]]:
        """Return basic application state stub for PlatynUI sessions."""
        return {
            "provider": "PlatynUI",
            "session_id": getattr(session, "session_id", None),
            "note": "Use 'Query' keyword to inspect the accessibility tree.",
        }

    async def get_ui_tree(
        self,
        session: "ExecutionSession",
        *,
        max_depth: int = 3,
        format: str = "text",
    ) -> Dict[str, Any]:
        """Retrieve the accessibility tree for the active desktop application.

        This is a stub implementation.  The real version will execute the
        PlatynUI ``Query`` keyword via the RF execution context.
        """
        return {
            "success": False,
            "provider": "PlatynUI",
            "session_id": getattr(session, "session_id", None),
            "max_depth": max_depth,
            "format": format,
            "note": (
                "UI tree retrieval requires the PlatynUI native backend "
                "(platynui_native).  Use the 'Query' keyword directly via "
                "execute_step to inspect the accessibility tree."
            ),
        }


class PlatynUIPlugin(StaticLibraryPlugin):
    """Builtin PlatynUI plugin for desktop GUI automation via accessibility APIs.

    PlatynUI provides cross-platform desktop automation using native
    accessibility frameworks (UIA on Windows, AT-SPI on Linux, AX on macOS)
    through a Rust/PyO3 backend.
    """

    __test__ = False  # Suppress pytest collection

    # Keywords shared with Browser Library (same name, different semantics).
    _BROWSER_SHARED_KEYWORDS: frozenset = frozenset({
        "focus",
        "get attribute",
        "take screenshot",
    })

    # Complete keyword set for PlatynUI.BareMetal
    _ALL_KEYWORDS: frozenset = frozenset({
        "query",
        "pointer click",
        "pointer multi click",
        "pointer press",
        "pointer release",
        "pointer move to",
        "get pointer position",
        "focus",
        "activate",
        "restore",
        "maximize",
        "minimize",
        "close",
        "get attribute",
        "keyboard type",
        "keyboard press",
        "keyboard release",
        "take screenshot",
        "highlight",
    })

    # Keywords unique to PlatynUI (not shared with web libraries).
    _UNIQUE_KEYWORDS: frozenset = _ALL_KEYWORDS - _BROWSER_SHARED_KEYWORDS

    # Mapping of common web keywords to PlatynUI equivalents.
    KEYWORD_ALTERNATIVES = {
        "click element": {
            "alternative": "Pointer Click",
            "example": "Pointer Click    /Window/Button[@Name='OK']",
            "explanation": "PlatynUI uses Pointer Click with XPath accessibility locators",
        },
        "click": {
            "alternative": "Pointer Click",
            "example": "Pointer Click    /Window/Button[@Name='Submit']",
            "explanation": "PlatynUI uses Pointer Click for all click interactions",
        },
        "input text": {
            "alternative": "Keyboard Type",
            "example": "Keyboard Type    /Window/Edit[@Name='Username']    myuser",
            "explanation": "PlatynUI uses Keyboard Type to enter text into focused elements",
        },
        "fill text": {
            "alternative": "Keyboard Type",
            "example": "Keyboard Type    /Window/Edit[@Name='Search']    search term",
            "explanation": "PlatynUI uses Keyboard Type instead of Fill Text",
        },
        "open browser": {
            "alternative": "Activate",
            "example": "Activate    /Window[@Name='My Application']",
            "explanation": "PlatynUI uses Activate to bring a desktop window to the foreground",
        },
        "get page source": {
            "alternative": "Query",
            "example": "Query    /Window",
            "explanation": "PlatynUI uses Query to inspect the accessibility tree (no HTML DOM)",
        },
        "get text": {
            "alternative": "Get Attribute",
            "example": "Get Attribute    /Window/Text[@Name='Status']    Value",
            "explanation": "PlatynUI uses Get Attribute to read element properties",
        },
        "mouse over": {
            "alternative": "Pointer Move To",
            "example": "Pointer Move To    /Window/Button[@Name='Hover Me']",
            "explanation": "PlatynUI uses Pointer Move To for hover interactions",
        },
        "set focus to element": {
            "alternative": "Focus",
            "example": "Focus    /Window/Edit[@Name='Email']",
            "explanation": "PlatynUI uses Focus to set keyboard focus on an element",
        },
        "capture page screenshot": {
            "alternative": "Take Screenshot",
            "example": "Take Screenshot    /Window",
            "explanation": "PlatynUI uses Take Screenshot for desktop window captures",
        },
        "close browser": {
            "alternative": "Close",
            "example": "Close    /Window[@Name='My Application']",
            "explanation": "PlatynUI uses Close to close a desktop window",
        },
    }

    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="PlatynUI",
            package_name="robotframework-platynui",
            import_path="PlatynUI.BareMetal",
            description="Cross-platform desktop GUI automation via accessibility APIs",
            library_type="external",
            use_cases=[
                "desktop testing",
                "gui automation",
                "accessibility testing",
                "window management",
            ],
            categories=["testing", "desktop", "gui", "accessibility"],
            contexts=["desktop"],
            installation_command="pip install robotframework-platynui",
            dependencies=["platynui_native"],
            requires_type_conversion=True,
            supports_async=False,
            load_priority=55,
            default_enabled=True,
        )
        capabilities = LibraryCapabilities(
            contexts=["desktop"],
            features=[
                "element_inspection",
                "keyboard_input",
                "pointer_input",
                "window_management",
                "screenshot",
                "accessibility_tree",
            ],
            technology=["uia", "atspi", "ax", "pyo3"],
            supports_page_source=False,
            supports_application_state=True,
            requires_type_conversion=True,
        )
        hints = LibraryHints(
            standard_keywords=[
                "Query",
                "Pointer Click",
                "Pointer Multi Click",
                "Keyboard Type",
                "Keyboard Press",
                "Keyboard Release",
                "Focus",
                "Activate",
                "Maximize",
                "Minimize",
                "Close",
                "Get Attribute",
                "Take Screenshot",
                "Highlight",
                "Pointer Move To",
                "Get Pointer Position",
            ],
            error_hints=[
                "PlatynUI uses XPath-style accessibility locators, e.g. /Window/Button[@Name='OK']",
                "Use 'Query' to inspect the accessibility tree and discover element paths",
                "Ensure the platynui_native backend is installed for your platform",
                "Use 'Activate' to bring the target window to the foreground before interacting",
            ],
            usage_examples=[
                "Query    /Window[@Name='Calculator']",
                "Activate    /Window[@Name='Calculator']",
                "Pointer Click    /Window/Button[@Name='1']",
                "Keyboard Type    /Window/Edit[@Name='Input']    Hello World",
                "Get Attribute    /Window/Text[@Name='Result']    Value",
                "Take Screenshot    /Window[@Name='Calculator']",
            ],
        )
        super().__init__(
            metadata=metadata,
            capabilities=capabilities,
            hints=hints,
            install_actions=None,
        )
        self._provider = PlatynUIStateProvider()

    def get_state_provider(self) -> LibraryStateProvider:
        return self._provider

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        """Map all PlatynUI keywords to the library name."""
        return {kw: "PlatynUI" for kw in self._ALL_KEYWORDS}

    def get_incompatible_libraries(self) -> List[str]:
        """PlatynUI has no hard incompatibilities with other libraries."""
        return []

    def get_keyword_alternatives(self) -> Dict[str, Dict[str, Any]]:
        """Return keyword alternatives mapping web keywords to PlatynUI equivalents."""
        return self.KEYWORD_ALTERNATIVES

    def validate_keyword_for_session(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        keyword_source_library: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Validate keyword compatibility for PlatynUI desktop sessions.

        Blocks web-only keywords when the session exclusively uses PlatynUI
        for desktop testing (no web libraries imported).
        """
        try:
            # Only apply validation when PlatynUI is the active library
            pref = (getattr(session, "explicit_library_preference", "") or "").lower()
            imported = getattr(session, "imported_libraries", []) or []

            is_platynui_session = (
                pref == "platynui"
                or "PlatynUI" in imported
                or "PlatynUI.BareMetal" in imported
            )
            if not is_platynui_session:
                return None

            # If web libraries are also imported, allow web keywords
            web_libraries = {"Browser", "SeleniumLibrary"}
            if web_libraries & set(imported):
                return None

            # Check if keyword is from a web-only library
            if keyword_source_library and keyword_source_library.lower() in {
                "browser",
                "seleniumlibrary",
            }:
                keyword_lower = keyword_name.lower()

                # Allow shared keywords
                if keyword_lower in self._BROWSER_SHARED_KEYWORDS:
                    return None

                alternative = self._get_desktop_alternative(keyword_lower)
                alternative_info = self.KEYWORD_ALTERNATIVES.get(keyword_lower, {})

                error_msg = (
                    f"Keyword '{keyword_name}' is from {keyword_source_library}, "
                    f"but this session uses PlatynUI for desktop automation.\n\n"
                )

                if alternative_info:
                    error_msg += (
                        f"Use '{alternative_info['alternative']}' instead:\n"
                        f"   {alternative_info['explanation']}\n\n"
                        f"Example:\n   {alternative_info['example']}\n\n"
                    )
                elif alternative:
                    error_msg += (
                        f"Try the PlatynUI equivalent: '{alternative}'\n\n"
                    )
                else:
                    error_msg += (
                        "Find the PlatynUI equivalent using:\n"
                        f"   find_keywords('{keyword_name}', strategy='catalog', "
                        f"session_id='...')\n\n"
                    )

                error_msg += (
                    "PlatynUI uses accessibility-based locators (XPath) for "
                    "desktop automation.\n"
                    "   Use 'Query' to discover available UI elements."
                )

                return {
                    "success": False,
                    "error": error_msg,
                    "keyword": keyword_name,
                    "keyword_library": keyword_source_library,
                    "session_library": "PlatynUI",
                    "alternative": alternative_info.get("alternative") or alternative,
                    "example": alternative_info.get("example"),
                    "hints": [
                        {
                            "title": "Library Mismatch",
                            "message": (
                                "Use PlatynUI keywords for desktop automation "
                                "instead of web browser keywords"
                            ),
                        }
                    ],
                }

            return None
        except Exception as exc:
            logger.debug("PlatynUI keyword validation failed: %s", exc)
            return None

    def generate_failure_hints(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        arguments: List[Any],
        error_text: str,
    ) -> List[Dict[str, Any]]:
        """Return hints for common PlatynUI errors."""
        err = (error_text or "").lower()
        hints: List[Dict[str, Any]] = []

        if "no results" in err or "returned no results" in err:
            locator = arguments[0] if arguments else "/Window/..."
            hints.append({
                "title": "PlatynUI: Element not found",
                "message": (
                    f"The locator '{locator}' did not match any elements in "
                    f"the accessibility tree. Verify the element exists using "
                    f"'Query' and check the XPath expression."
                ),
                "examples": [
                    {
                        "tool": "execute_step",
                        "keyword": "Query",
                        "arguments": ["/Window"],
                    },
                    {
                        "tool": "execute_step",
                        "keyword": "Activate",
                        "arguments": ["/Window"],
                    },
                ],
            })

        if "no module named 'platynui_native'" in err:
            hints.append({
                "title": "PlatynUI: Native backend not installed",
                "message": (
                    "The platynui_native Rust backend is not installed. "
                    "Install it with: pip install robotframework-platynui\n"
                    "Ensure the native binary is compiled for your platform "
                    "(Windows/Linux/macOS)."
                ),
            })

        if "attributenotfounderror" in err:
            attr_name = arguments[1] if len(arguments) > 1 else "<unknown>"
            hints.append({
                "title": "PlatynUI: Attribute not found",
                "message": (
                    f"The attribute '{attr_name}' does not exist on the target "
                    f"element. Common attributes: Name, Value, IsEnabled, "
                    f"IsVisible, ClassName, AutomationId, BoundingRectangle."
                ),
                "examples": [
                    {
                        "tool": "execute_step",
                        "keyword": "Get Attribute",
                        "arguments": [arguments[0] if arguments else "/Window", "Name"],
                    },
                ],
            })

        return hints

    def _get_desktop_alternative(self, kw_lower: str) -> Optional[str]:
        """Map a common web keyword to its PlatynUI equivalent.

        Used by ``validate_keyword_for_session`` to suggest alternatives.
        """
        mapping: Dict[str, str] = {
            "click element": "Pointer Click",
            "click": "Pointer Click",
            "click button": "Pointer Click",
            "click link": "Pointer Click",
            "double click element": "Pointer Multi Click",
            "input text": "Keyboard Type",
            "fill text": "Keyboard Type",
            "press keys": "Keyboard Press",
            "open browser": "Activate",
            "new browser": "Activate",
            "new page": "Activate",
            "get page source": "Query",
            "get source": "Query",
            "get text": "Get Attribute",
            "get value": "Get Attribute",
            "mouse over": "Pointer Move To",
            "set focus to element": "Focus",
            "capture page screenshot": "Take Screenshot",
            "screenshot": "Take Screenshot",
            "close browser": "Close",
            "close all browsers": "Close",
            "go to": "Activate",
            "maximize browser window": "Maximize",
            "minimize browser window": "Minimize",
        }
        return mapping.get(kw_lower)


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
