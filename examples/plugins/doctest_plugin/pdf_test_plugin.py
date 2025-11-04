"""Plugin metadata and overrides for DocTest.PdfTest."""

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
    PromptBundle,
)


class _PdfStateProvider(LibraryStateProvider):
    async def get_page_source(self, *args, **kwargs):  # type: ignore[override]
        return None

    async def get_application_state(  # type: ignore[override]
        self,
        session: "ExecutionSession",
    ) -> Optional[Dict[str, Any]]:
        summary = session.variables.get("_doctest_pdf_result")
        if not summary:
            return {"success": False, "error": "No DocTest PDF result available."}
        return {"success": True, "pdf": summary}


class DocTestPdfPlugin(StaticLibraryPlugin):
    """Expose PDF comparison metadata and provide keyword overrides."""

    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="DocTest.PdfTest",
            package_name="robotframework-doctestlibrary",
            import_path="DocTest.PdfTest",
            description="Compare PDF text, structure, and metadata or ensure strings exist.",
            library_type="external",
            use_cases=[
                "pdf text comparison",
                "metadata verification",
                "document content assertions",
            ],
            categories=["documents", "testing"],
            contexts=["desktop"],
            installation_command="pip install robotframework-doctestlibrary",
            platform_requirements=["MuPDF", "Tesseract OCR (for scanned PDFs)"],
            dependencies=[],
            requires_type_conversion=False,
            supports_async=False,
            load_priority=65,
            default_enabled=False,
        )
        capabilities = LibraryCapabilities(
            contexts=["desktop"],
            features=["pdf-diff", "metadata"],
            technology=["mupdf"],
        )
        hints = LibraryHints(
            standard_keywords=[
                "Compare Pdf Documents",
                "Pdf Should Contain Strings",
            ],
            error_hints=[
                "Use `compare=text`, `compare=structure`, or `compare=metadata` to scope comparisons.",
                "Ensure MuPDF / PyMuPDF is installed to parse PDFs.",
            ],
            usage_examples=[
                "Compare Pdf Documents    reference.pdf    candidate.pdf    compare=text",
                "Pdf Should Contain Strings    ${strings}    candidate.pdf",
            ],
        )
        prompts = PromptBundle(
            recommendation="Use DocTest.PdfTest to compare PDF text, structure, or metadata without rendering.",
            troubleshooting="For scanned PDFs, pair with VisualTest to extract OCR before comparing.",
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
            prompt_bundle=prompts,
            install_actions=install_actions,
        )
        self._state_provider = _PdfStateProvider()

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        keywords = {
            "compare pdf documents",
            "pdf should contain strings",
        }
        return {kw: "DocTest.PdfTest" for kw in keywords}

    def get_state_provider(self) -> Optional[LibraryStateProvider]:  # type: ignore[override]
        return self._state_provider

    def get_keyword_overrides(self):  # type: ignore[override]
        async def _override(session, keyword_name, args, keyword_info):
            return self._execute_pdf_keyword(session, keyword_name, args)

        return {
            "compare pdf documents": _override,
            "pdf should contain strings": _override,
        }

    def _execute_pdf_keyword(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        args: List[str],
    ) -> Dict[str, Any]:
        session.import_library("DocTest.PdfTest", force=False)
        built_in = BuiltIn()

        try:
            built_in.run_keyword(keyword_name, args)
        except AssertionError as exc:
            summary = self._build_failure_summary(exc)
            session.variables["_doctest_pdf_result"] = summary
            return {
                "success": False,
                "output": summary.get("message", "DocTest PDF comparison failed."),
                "error": summary.get("message"),
                "state_updates": {"doctest": {"pdf": summary}},
            }

        summary = {
            "status": "passed",
            "message": "PDF comparison passed.",
        }
        session.variables["_doctest_pdf_result"] = summary
        return {
            "success": True,
            "output": "PDF comparison passed",
            "state_updates": {"doctest": {"pdf": summary}},
        }

    def _build_failure_summary(self, exc: AssertionError) -> Dict[str, Any]:
        frames: List[Any] = []
        tb = exc.__traceback__
        while tb:
            frames.append(tb)
            tb = tb.tb_next

        llm_differences: List[Dict[str, Any]] = []
        compare_facets: Optional[List[str]] = None
        mask_applied: Optional[str] = None

        for frame in reversed(frames):
            locs = frame.tb_frame.f_locals
            if "llm_differences" in locs and locs.get("llm_differences"):
                llm_differences = locs["llm_differences"]
            if "compare_set" in locs and not compare_facets:
                compare_facets = sorted(str(token) for token in locs["compare_set"])
            if "mask" in locs and mask_applied is None:
                mask_applied = "yes" if locs.get("mask") else "no"

        serialized_diffs: List[Dict[str, Any]] = []
        for diff in llm_differences:
            serialized_diffs.append({k: str(v) for k, v in diff.items()})

        summary = {
            "status": "failed",
            "message": str(exc),
            "compare_facets": compare_facets,
            "mask_used": mask_applied,
            "differences": serialized_diffs,
        }

        if serialized_diffs:
            summary["differences_path"] = self._write_json(serialized_diffs)

        return summary

    def _write_json(self, data: List[Dict[str, Any]]) -> Optional[str]:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix="_pdf_diff.json") as tmp:
                tmp.write(json.dumps(data, indent=2).encode("utf-8"))
                return tmp.name
        except Exception:
            return None


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
