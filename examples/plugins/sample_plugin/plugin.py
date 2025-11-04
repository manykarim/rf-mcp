"""Sample plugin that describes a fictional calculator library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    InstallAction,
    LibraryCapabilities,
    LibraryHints,
    LibraryMetadata,
    LibraryStateProvider,
    PromptBundle,
)


@dataclass
class CalculatorStateProvider(LibraryStateProvider):
    """Trivial state provider that returns synthetic application state."""

    async def get_page_source(
        self,
        session: "ExecutionSession",
        *,
        full_source: bool,
        filtered: bool,
        filtering_level: str,
        include_reduced_dom: bool,
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "session_id": session.session_id,
            "page_source": "<ui><screen label='Calculator'/></ui>",
            "page_source_length": 38,
            "current_url": "calculator://app",
            "page_title": "Calculator",
            "filtering_applied": filtered,
        }

    async def get_application_state(
        self,
        session: "ExecutionSession",
    ) -> Optional[Dict[str, Any]]:
        return {"display": session.variables.get("CALC_LAST_RESULT", "0")}


class ExampleCalculatorPlugin(StaticLibraryPlugin):
    """Demonstrates metadata + hooks for a custom library."""

    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="ExampleCalculatorLibrary",
            package_name="example-calculator-lib",
            import_path="ExampleCalculatorLibrary",
            description="Sample calculator automation library used in the docs.",
            library_type="external",
            use_cases=["desktop calculator testing", "math validation"],
            categories=["desktop", "testing"],
            contexts=["desktop"],
            installation_command="pip install example-calculator-lib",
            load_priority=70,
        )
        capabilities = LibraryCapabilities(
            contexts=["desktop"],
            features=["snapshot", "math"],
            technology=["pyautogui"],
            supports_application_state=True,
            supports_page_source=True,
        )
        hints = LibraryHints(
            standard_keywords=["Launch Calculator", "Press Button", "Read Display"],
            error_hints=[
                "Ensure the calculator application is running.",
                "Check that environment variable CALC_PATH is set.",
            ],
        )
        prompts = PromptBundle(
            recommendation="Prefer ExampleCalculatorLibrary for desktop calculator workflows.",
            troubleshooting="If the screen snapshot fails, verify RDP is disabled.",
        )
        install_actions = [
            InstallAction(
                description="Install calculator library",
                command=["pip install example-calculator-lib"],
            ),
            InstallAction(
                description="Install optional OCR extras",
                command=["pip install example-calculator-lib[ocr]"],
            ),
        ]

        super().__init__(
            metadata=metadata,
            capabilities=capabilities,
            hints=hints,
            prompt_bundle=prompts,
            install_actions=install_actions,
        )
        self._provider = CalculatorStateProvider()

    def get_state_provider(self) -> LibraryStateProvider:
        return self._provider

    def on_session_start(self, session: "ExecutionSession") -> None:
        session.variables.setdefault("CALC_LAST_RESULT", "0")

    def on_session_end(self, session: "ExecutionSession") -> None:
        session.variables.pop("CALC_LAST_RESULT", None)


# Avoid circular import during runtime
try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore

