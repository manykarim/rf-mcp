"""Builtin Browser Library plugin with page source integration."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    KeywordOverrideHandler,
    LibraryCapabilities,
    LibraryMetadata,
    LibraryStateProvider,
)

logger = logging.getLogger(__name__)


class BrowserStateProvider(LibraryStateProvider):
    """Implement Browser Library page source retrieval via RF context."""

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
        service = kwargs.get("service")
        browser_library_manager = kwargs.get("browser_library_manager")

        if service is None or browser_library_manager is None:
            logger.debug("BrowserStateProvider missing required context; skipping.")
            return None

        try:
            page_source = service._get_page_source_via_rf_context(session)  # type: ignore[attr-defined]
        except AttributeError:
            logger.debug("PageSourceService helper not available for Browser provider.")
            return None

        if not page_source:
            return {"success": False, "error": "No page source available for this session"}

        aria_snapshot_info: Optional[Dict[str, Any]] = None
        if include_reduced_dom:
            try:
                aria_snapshot_info = await service._capture_browser_aria_snapshot(  # type: ignore[attr-defined]
                    session=session,
                    browser_library_manager=browser_library_manager,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(
                    "Browser reduced DOM capture failed for session %s: %s",
                    session.session_id,
                    exc,
                )
                aria_snapshot_info = {
                    "success": False,
                    "selector": "css=html",
                    "error": str(exc),
                }
        else:
            aria_snapshot_info = {
                "success": False,
                "selector": "css=html",
                "skipped": True,
            }

        if filtered:
            filtered_source = service.filter_page_source(page_source, filtering_level)
            result_source = filtered_source
            filtered_length = len(filtered_source)
        else:
            result_source = page_source
            filtered_length = None

        result: Dict[str, Any] = {
            "success": True,
            "session_id": session.session_id,
            "page_source_length": len(page_source),
            "current_url": session.browser_state.current_url,
            "page_title": session.browser_state.page_title,
            "context": await service.extract_page_context(page_source),
            "filtering_applied": filtered,
        }

        if filtered:
            result["filtered_page_source_length"] = filtered_length

        if aria_snapshot_info is not None:
            result["aria_snapshot"] = aria_snapshot_info

        if full_source:
            key = "page_source"
            result[key] = result_source
        else:
            preview_size = service.config.PAGE_SOURCE_PREVIEW_SIZE
            if len(result_source) > preview_size:
                result["page_source_preview"] = (
                    result_source[:preview_size]
                    + "...\n[Truncated - use full_source=True for complete filtered source]"
                )
            else:
                result["page_source_preview"] = result_source

        return result


class BrowserLibraryPlugin(StaticLibraryPlugin):
    """Builtin Browser plugin with custom state provider and capabilities."""

    # Mapping of SeleniumLibrary keywords to Browser Library alternatives
    KEYWORD_ALTERNATIVES = {
        "open browser": {
            "alternative": "New Browser + New Page",
            "example": "New Browser    browser=firefox    headless=${False}\nNew Page    https://example.com",
            "explanation": "Browser Library uses Playwright which requires separate browser and page creation",
        },
        "close browser": {
            "alternative": "Close Browser",
            "example": "Close Browser",
            "explanation": "Same keyword name, but use with Browser Library context",
        },
        "close all browsers": {
            "alternative": "Close Browser    ALL",
            "example": "Close Browser    ALL",
            "explanation": "Use Close Browser with ALL parameter",
        },
        "go to": {
            "alternative": "Go To",
            "example": "Go To    https://example.com",
            "explanation": "Same keyword available in Browser Library",
        },
        "get source": {
            "alternative": "Get Page Source",
            "example": "Get Page Source",
            "explanation": "Use Get Page Source for Browser Library",
        },
        "input text": {
            "alternative": "Fill Text",
            "example": "Fill Text    css=input#username    myuser",
            "explanation": "Browser Library uses Fill Text instead of Input Text",
        },
        "click element": {
            "alternative": "Click",
            "example": "Click    css=button#submit",
            "explanation": "Browser Library uses shorter Click keyword",
        },
        "click button": {
            "alternative": "Click",
            "example": "Click    css=button#submit",
            "explanation": "Browser Library uses generic Click for all elements",
        },
        "wait until element is visible": {
            "alternative": "Wait For Elements State",
            "example": "Wait For Elements State    css=.element    visible    timeout=10s",
            "explanation": "Browser Library uses Wait For Elements State with state parameter",
        },
        "wait until page contains element": {
            "alternative": "Wait For Elements State",
            "example": "Wait For Elements State    css=.element    attached    timeout=10s",
            "explanation": "Use 'attached' state to wait for element presence",
        },
    }

    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="Browser",
            package_name="robotframework-browser",
            import_path="Browser",
            description="Modern web testing with Playwright backend",
            library_type="external",
            use_cases=[
                "modern web testing",
                "playwright automation",
                "web performance",
                "mobile web",
            ],
            categories=["web", "testing"],
            contexts=["web"],
            installation_command="pip install robotframework-browser",
            post_install_commands=["rfbrowser init"],
            dependencies=["playwright", "node.js"],
            requires_type_conversion=True,
            supports_async=True,
            load_priority=5,
            default_enabled=True,
            extra_name="web",
        )
        capabilities = LibraryCapabilities(
            contexts=["web"],
            features=["playwright"],
            technology=["playwright"],
            supports_page_source=True,
            supports_application_state=False,
            requires_type_conversion=True,
            supports_async=True,
        )
        super().__init__(metadata=metadata, capabilities=capabilities)
        self._provider = BrowserStateProvider()

    def get_state_provider(self) -> LibraryStateProvider:
        return self._provider

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        return {
            "browser.new browser": "Browser",
            "browser.new page": "Browser",
            "browser.close browser": "Browser",
            "new browser": "Browser",
            "new page": "Browser",
            "close browser": "Browser",
            "open browser": "Browser",
            "get page source": "Browser",
            "get url": "Browser",
            "get title": "Browser",
        }

    def get_keyword_overrides(self) -> Dict[str, KeywordOverrideHandler]:  # type: ignore[override]
        return {"open browser": self._override_open_browser}

    def get_locator_normalizer(self):
        def normalize(locator: str) -> str:
            return locator

        return normalize

    def get_locator_validator(self):
        def validate(locator: str) -> Dict[str, Any]:
            ok = isinstance(locator, str) and bool(locator.strip())
            return {"valid": ok, "warnings": [] if ok else ["Empty locator"]}

        return validate

    def get_incompatible_libraries(self) -> list[str]:
        """Browser Library is incompatible with SeleniumLibrary."""
        return ["SeleniumLibrary"]

    def get_keyword_alternatives(self) -> Dict[str, Dict[str, Any]]:
        """Return keyword alternatives for SeleniumLibrary keywords."""
        return self.KEYWORD_ALTERNATIVES

    def validate_keyword_for_session(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        keyword_source_library: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Validate if a keyword is compatible with Browser Library session.

        Returns error with alternative if keyword is from SeleniumLibrary.
        """
        try:
            # Check if this session prefers Browser Library
            pref = (getattr(session, "explicit_library_preference", "") or "").lower()
            if not pref or pref != "browser":
                return None  # Not a Browser Library session

            # Check if keyword is from SeleniumLibrary
            if keyword_source_library and keyword_source_library.lower() == "seleniumlibrary":
                keyword_lower = keyword_name.lower()
                alternative_info = self.KEYWORD_ALTERNATIVES.get(keyword_lower, {})

                error_msg = (
                    f"Keyword '{keyword_name}' is from SeleniumLibrary, "
                    f"but this session uses Browser Library (Playwright).\n\n"
                )

                if alternative_info:
                    error_msg += f"💡 Use '{alternative_info['alternative']}' instead:\n"
                    error_msg += f"   {alternative_info['explanation']}\n\n"
                    error_msg += f"Example:\n   {alternative_info['example']}\n\n"
                else:
                    error_msg += "💡 Find the Browser Library equivalent using:\n"
                    error_msg += f"   find_keywords('{keyword_name}', strategy='catalog', session_id='...')\n\n"

                error_msg += (
                    "📚 Browser Library uses Playwright which has different keyword names.\n"
                    "   Use find_keywords to discover available keywords."
                )

                return {
                    "success": False,
                    "error": error_msg,
                    "keyword": keyword_name,
                    "keyword_library": keyword_source_library,
                    "session_library": "Browser",
                    "alternative": alternative_info.get("alternative"),
                    "example": alternative_info.get("example"),
                    "hints": [{
                        "title": "Library Mismatch",
                        "message": f"Use Browser Library keywords instead of SeleniumLibrary"
                    }]
                }

            return None
        except Exception as exc:
            logger.debug("Keyword validation failed: %s", exc)
            return None

    async def _override_open_browser(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        arguments: list[str],
        keyword_info: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Reject 'Open Browser' and suggest Browser Library alternatives."""
        try:
            pref = (getattr(session, "explicit_library_preference", "") or "").lower()
            if pref.startswith("selenium"):
                return None

            active = getattr(session, "browser_state", None)
            if active and getattr(active, "active_library", None) == "selenium":
                return None

            # If SeleniumLibrary is imported (and Browser is NOT), this is a
            # SeleniumLibrary session — do not reject Open Browser.
            imported = getattr(session, "imported_libraries", []) or []
            has_selenium = "SeleniumLibrary" in imported
            has_browser = "Browser" in imported
            if has_selenium and not has_browser:
                return None

            # Get the alternative info
            alt_info = self.KEYWORD_ALTERNATIVES.get("open browser", {})

            error_msg = (
                "'Open Browser' is a SeleniumLibrary keyword and cannot be used with Browser Library.\n\n"
                f"💡 Use '{alt_info.get('alternative', 'New Browser + New Page')}' instead:\n"
                f"   {alt_info.get('explanation', 'Browser Library requires separate browser and page creation')}\n\n"
                "Example:\n"
            )

            # Parse arguments to create helpful example
            url = arguments[0] if arguments else "https://example.com"
            browser = arguments[1] if len(arguments) > 1 else "chromium"

            error_msg += f"   New Browser    browser={browser}    headless=${{False}}\n"
            error_msg += f"   New Page       {url}\n\n"
            error_msg += (
                "📚 Browser Library uses Playwright which provides modern, fast browser automation.\n"
                "   The 'New Browser' keyword starts the browser, 'New Page' opens a URL."
            )

            return {
                "success": False,
                "error": error_msg,
                "keyword": keyword_name,
                "session_library": "Browser",
                "alternative": "New Browser + New Page",
                "guidance": [
                    "Use 'New Browser' to start Playwright browser instance.",
                    "Use 'New Page' to navigate to your target URL.",
                    f"Example: New Browser    browser={browser}    headless=${{False}}",
                    f"Then: New Page    {url}",
                ],
                "hints": [{
                    "title": "Use Browser Library Keywords",
                    "message": "Replace 'Open Browser' with 'New Browser' + 'New Page'"
                }]
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Open Browser override failed: %s", exc)
            return None


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
