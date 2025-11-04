"""Plugin metadata and overrides for DocTest print job comparisons."""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from robot.libraries.BuiltIn import BuiltIn

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    InstallAction,
    LibraryCapabilities,
    LibraryHints,
    LibraryMetadata,
    LibraryStateProvider,
)


class _PrintJobStateProvider(LibraryStateProvider):
    async def get_page_source(self, *args, **kwargs):  # type: ignore[override]
        return None

    async def get_application_state(  # type: ignore[override]
        self,
        session: "ExecutionSession",
    ) -> Optional[Dict[str, Any]]:
        summary = session.variables.get("_doctest_print_result")
        if not summary:
            return {
                "success": False,
                "error": "No DocTest print comparison result available.",
            }
        return {"success": True, "print_job": summary}


class DocTestPrintJobPlugin(StaticLibraryPlugin):
    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="DocTest.PrintJobTest",
            package_name="robotframework-doctestlibrary",
            import_path="DocTest.PrintJobTests",
            description="Compare printer job outputs (PCL/PS) via rendering and visual diffs.",
            library_type="external",
            use_cases=["print job comparison", "rendered document diff"],
            categories=["documents", "testing"],
            contexts=["desktop"],
            installation_command="pip install robotframework-doctestlibrary",
            platform_requirements=["GhostPCL", "Ghostscript"],
            load_priority=66,
            default_enabled=False,
        )
        capabilities = LibraryCapabilities(
            contexts=["desktop"],
            features=["print-job-diff"],
        )
        hints = LibraryHints(
            standard_keywords=["Compare Print Jobs"],
            error_hints=["Install GhostPCL and Ghostscript to render PCL/PS files."],
            usage_examples=[
                "Compare Print Jobs    reference.pcl    candidate.pcl",
            ],
        )

        install_actions = [
            InstallAction(
                description="Install DocTestLibrary core package",
                command=["pip", "install", "robotframework-doctestlibrary"],
            )
        ]

        super().__init__(
            metadata=metadata,
            capabilities=capabilities,
            hints=hints,
            install_actions=install_actions,
        )
        self._state_provider = _PrintJobStateProvider()

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        return {"compare print jobs": "DocTest.PrintJobTest"}

    def get_state_provider(self) -> Optional[LibraryStateProvider]:  # type: ignore[override]
        return self._state_provider

    def get_keyword_overrides(self):  # type: ignore[override]
        async def _override(session, keyword_name, args, keyword_info):
            return self._execute_print_keyword(session, keyword_name, args)

        return {"compare print jobs": _override}

    def _execute_print_keyword(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        args: List[str],
    ) -> Dict[str, Any]:
        session.import_library("DocTest.PrintJobTests", force=False)
        built_in = BuiltIn()

        try:
            built_in.run_keyword(keyword_name, args)
        except AssertionError as exc:
            summary = self._build_failure_summary(exc)
            session.variables["_doctest_print_result"] = summary
            return {
                "success": False,
                "output": summary.get("message", "DocTest print jobs differ."),
                "error": summary.get("message"),
                "state_updates": {"doctest": {"print_job": summary}},
            }

        summary = {
            "status": "passed",
            "message": "Print jobs comparison passed.",
        }
        session.variables["_doctest_print_result"] = summary
        return {
            "success": True,
            "output": "Print jobs comparison passed",
            "state_updates": {"doctest": {"print_job": summary}},
        }

    def _build_failure_summary(self, exc: AssertionError) -> Dict[str, Any]:
        frames: List[Any] = []
        tb = exc.__traceback__
        while tb:
            frames.append(tb)
            tb = tb.tb_next

        list_difference: List[Dict[str, Any]] = []
        reference_props = None
        candidate_props = None

        for frame in reversed(frames):
            locs = frame.tb_frame.f_locals
            if "list_difference" in locs and locs["list_difference"]:
                list_difference = locs["list_difference"]
            if "reference_print_job" in locs and reference_props is None:
                reference = locs.get("reference_print_job")
                reference_props = getattr(reference, "properties", None)
            if "test_print_job" in locs and candidate_props is None:
                candidate = locs.get("test_print_job")
                candidate_props = getattr(candidate, "properties", None)

        summary = {
            "status": "failed",
            "message": str(exc),
            "differences": list_difference,
        }

        if reference_props is not None:
            summary["reference_properties_path"] = self._write_json(
                reference_props, "reference"
            )
        if candidate_props is not None:
            summary["candidate_properties_path"] = self._write_json(
                candidate_props, "candidate"
            )

        return summary

    def _write_json(
        self, data: Optional[List[Dict[str, Any]]], prefix: str
    ) -> Optional[str]:
        if data is None:
            return None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{prefix}_print.json") as tmp:
                tmp.write(json.dumps(data, indent=2).encode("utf-8"))
                return tmp.name
        except Exception:
            return None


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
