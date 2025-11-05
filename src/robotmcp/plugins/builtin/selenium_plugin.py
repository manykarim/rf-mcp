"""Builtin SeleniumLibrary plugin."""

from __future__ import annotations

from typing import Any, Dict, Optional

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    LibraryCapabilities,
    LibraryMetadata,
    LibraryStateProvider,
)


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


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
