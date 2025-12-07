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

    async def _override_open_browser(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        arguments: list[str],
        keyword_info: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Reject Browser's 'Open Browser' to avoid Playwright debug/pause mode."""
        try:
            pref = (getattr(session, "explicit_library_preference", "") or "").lower()
            if pref.startswith("selenium"):
                return None

            active = getattr(session, "browser_state", None)
            if active and getattr(active, "active_library", None) == "selenium":
                return None

            return {
                "success": False,
                "error": "'Open Browser' is not supported for Browser library (debug/pause mode).",
                "guidance": [
                    "Use 'New Browser' to start Playwright.",
                    "Then 'New Context' (optional) and 'New Page' with the target URL.",
                    "Example: New Browser    chromium    headless=False -> New Context    viewport={'width':1280,'height':720} -> New Page    https://demoshop.makrocode.de/",
                ],
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
