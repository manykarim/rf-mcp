"""Builtin SeleniumLibrary plugin."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    KeywordOverrideHandler,
    LibraryCapabilities,
    LibraryMetadata,
    LibraryStateProvider,
)

logger = logging.getLogger(__name__)


class SeleniumStateProvider(LibraryStateProvider):
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
        if service is None:
            return None

        page_source = service._get_page_source_via_rf_context(session)  # type: ignore[attr-defined]
        if not page_source:
            return {"success": False, "error": "No page source available for this session"}

        if filtered:
            output_source = service.filter_page_source(page_source, filtering_level)
            filtered_length = len(output_source)
        else:
            output_source = page_source
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

        if full_source:
            result["page_source"] = output_source
        else:
            preview_size = service.config.PAGE_SOURCE_PREVIEW_SIZE
            if len(output_source) > preview_size:
                result["page_source_preview"] = (
                    output_source[:preview_size]
                    + "...\n[Truncated - use full_source=True for complete filtered source]"
                )
            else:
                result["page_source_preview"] = output_source

        return result


class SeleniumLibraryPlugin(StaticLibraryPlugin):
    """Builtin SeleniumLibrary plugin with reciprocal Browser Library validation."""

    # Mapping of Browser Library keywords to SeleniumLibrary alternatives
    KEYWORD_ALTERNATIVES = {
        "new browser": {
            "alternative": "Open Browser",
            "example": "Open Browser    https://example.com    chrome",
            "explanation": "SeleniumLibrary uses Open Browser to start browser and navigate",
        },
        "new page": {
            "alternative": "Go To",
            "example": "Go To    https://example.com",
            "explanation": "Use Go To to navigate to a URL in SeleniumLibrary",
        },
        "new context": {
            "alternative": "Open Browser (with options)",
            "example": "Open Browser    https://example.com    chrome    options=add_argument(\"--window-size=1280,720\")",
            "explanation": "SeleniumLibrary doesn't have contexts - configure browser in Open Browser",
        },
        "close browser": {
            "alternative": "Close Browser",
            "example": "Close Browser",
            "explanation": "Same keyword name, works in SeleniumLibrary",
        },
        "fill text": {
            "alternative": "Input Text",
            "example": "Input Text    id=username    myuser",
            "explanation": "SeleniumLibrary uses Input Text instead of Fill Text",
        },
        "click": {
            "alternative": "Click Element",
            "example": "Click Element    id=submit",
            "explanation": "SeleniumLibrary uses Click Element instead of Click",
        },
        "get text": {
            "alternative": "Get Text",
            "example": "Get Text    id=message",
            "explanation": "Same keyword available in SeleniumLibrary",
        },
        "wait for elements state": {
            "alternative": "Wait Until Element Is Visible",
            "example": "Wait Until Element Is Visible    id=element    timeout=10s",
            "explanation": "SeleniumLibrary uses specific wait keywords like Wait Until Element Is Visible",
        },
        "get page source": {
            "alternative": "Get Source",
            "example": "Get Source",
            "explanation": "SeleniumLibrary uses Get Source instead of Get Page Source",
        },
    }

    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="SeleniumLibrary",
            package_name="robotframework-seleniumlibrary",
            import_path="SeleniumLibrary",
            description="Traditional web testing with Selenium WebDriver",
            library_type="external",
            use_cases=["web testing", "browser automation", "web elements", "form filling"],
            categories=["web", "testing"],
            contexts=["web"],
            installation_command="pip install robotframework-seleniumlibrary",
            dependencies=["selenium"],
            requires_type_conversion=True,
            supports_async=False,
            load_priority=8,
            default_enabled=True,
            extra_name="web",
        )
        capabilities = LibraryCapabilities(
            contexts=["web"],
            supports_page_source=True,
            requires_type_conversion=True,
        )
        super().__init__(metadata=metadata, capabilities=capabilities)
        self._provider = SeleniumStateProvider()

    def get_state_provider(self) -> LibraryStateProvider:
        return self._provider

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        return {
            "seleniumlibrary.get source": "SeleniumLibrary",
            "get source": "SeleniumLibrary",
        }

    def get_incompatible_libraries(self) -> List[str]:
        """SeleniumLibrary is incompatible with Browser Library."""
        return ["Browser"]

    def get_keyword_alternatives(self) -> Dict[str, Dict[str, Any]]:
        """Return keyword alternatives for Browser Library keywords."""
        return self.KEYWORD_ALTERNATIVES

    def validate_keyword_for_session(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        keyword_source_library: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Validate if a keyword is compatible with SeleniumLibrary session.

        Returns error with alternative if keyword is from Browser Library.
        """
        try:
            # Check if this session prefers SeleniumLibrary
            pref = (getattr(session, "explicit_library_preference", "") or "").lower()
            if not pref or not pref.startswith("selenium"):
                return None  # Not a SeleniumLibrary session

            # Check if keyword is from Browser Library
            if keyword_source_library and keyword_source_library.lower() == "browser":
                keyword_lower = keyword_name.lower()
                alternative_info = self.KEYWORD_ALTERNATIVES.get(keyword_lower, {})

                error_msg = (
                    f"Keyword '{keyword_name}' is from Browser Library (Playwright), "
                    f"but this session uses SeleniumLibrary.\n\n"
                )

                if alternative_info:
                    error_msg += f"💡 Use '{alternative_info['alternative']}' instead:\n"
                    error_msg += f"   {alternative_info['explanation']}\n\n"
                    error_msg += f"Example:\n   {alternative_info['example']}\n\n"
                else:
                    error_msg += "💡 Find the SeleniumLibrary equivalent using:\n"
                    error_msg += f"   find_keywords('{keyword_name}', strategy='catalog', session_id='...')\n\n"

                error_msg += (
                    "📚 SeleniumLibrary uses Selenium WebDriver which has different keyword names.\n"
                    "   Use find_keywords to discover available keywords."
                )

                return {
                    "success": False,
                    "error": error_msg,
                    "keyword": keyword_name,
                    "keyword_library": keyword_source_library,
                    "session_library": "SeleniumLibrary",
                    "alternative": alternative_info.get("alternative"),
                    "example": alternative_info.get("example"),
                    "hints": [{
                        "title": "Library Mismatch",
                        "message": "Use SeleniumLibrary keywords instead of Browser Library"
                    }]
                }

            return None
        except Exception as exc:
            logger.debug("Keyword validation failed: %s", exc)
            return None


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
