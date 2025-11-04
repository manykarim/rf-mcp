"""Plugin metadata and overrides for DocTest.VisualTest."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
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


class _VisualStateProvider(LibraryStateProvider):
    async def get_page_source(  # type: ignore[override]
        self,
        session: "ExecutionSession",
        *,
        full_source: bool = False,
        filtered: bool = False,
        filtering_level: str = "standard",
        include_reduced_dom: bool = True,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        return None

    async def get_application_state(  # type: ignore[override]
        self,
        session: "ExecutionSession",
    ) -> Optional[Dict[str, Any]]:
        summary = session.variables.get("_doctest_visual_result")
        if not summary:
            return {"success": False, "error": "No DocTest visual result available."}
        return {"success": True, "visual": summary}


class DocTestVisualPlugin(StaticLibraryPlugin):
    """Expose VisualTest keywords, overrides, and application state."""

    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="DocTest.VisualTest",
            package_name="robotframework-doctestlibrary",
            import_path="DocTest.VisualTest",
            description="Visual comparisons, OCR, barcode detection, and watermark-aware image/PDF diffs.",
            library_type="external",
            use_cases=[
                "visual regression testing",
                "document OCR validation",
                "barcode extraction",
                "watermark-aware diffing",
            ],
            categories=["visual", "testing", "documents"],
            contexts=["desktop"],
            installation_command="pip install robotframework-doctestlibrary",
            platform_requirements=[
                "ImageMagick",
                "Tesseract OCR",
                "Ghostscript/GhostPCL (for PostScript/Printer control files)",
            ],
            dependencies=["Pillow"],
            requires_type_conversion=False,
            supports_async=False,
            load_priority=64,
            default_enabled=False,
        )
        capabilities = LibraryCapabilities(
            contexts=["desktop"],
            features=["visual-diff", "ocr", "barcode"],
            technology=["pillow", "pytesseract"],
        )
        hints = LibraryHints(
            standard_keywords=[
                "Compare Images",
                "Image Should Contain",
                "Get Text From Document",
                "Get Barcodes From Document",
            ],
            error_hints=[
                "Ensure Tesseract, ImageMagick, and Ghostscript binaries are installed and on PATH.",
                "Use `placeholder_file` to mask known differences.",
                "Set `show_diff=${True}` to persist diff artefacts in log.html.",
                "Leverage `watermark_file` when ignoring repeated watermarks.",
            ],
            usage_examples=[
                "Compare Images    Reference.png    Candidate.png    show_diff=${True}",
                "Image Should Contain    Candidate.png    Product logo",
                "Get Text From Document    Candidate.pdf",
            ],
        )
        prompts = PromptBundle(
            recommendation="For pixel-perfect visual regression or OCR checks, use DocTest.VisualTest.",
            troubleshooting=(
                "If comparisons fail, confirm images render at consistent DPI and external binaries are installed. "
                "Use masks/watermarks to ignore intentional differences."
            ),
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
        self._state_provider = _VisualStateProvider()

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        keywords = {
            "compare images",
            "compare images with llm",
            "image should contain",
            "image should not contain",
            "get text from document",
            "get barcodes from document",
            "compare gifs",
        }
        return {kw: "DocTest.VisualTest" for kw in keywords}

    def on_session_start(self, session: "ExecutionSession") -> None:
        session.variables.setdefault("DOCTEST_SHOW_DIFF", False)
        session.variables.setdefault("DOCTEST_WATERMARK_FILE", None)

    def on_session_end(self, session: "ExecutionSession") -> None:
        session.variables.pop("DOCTEST_SHOW_DIFF", None)
        session.variables.pop("DOCTEST_WATERMARK_FILE", None)

    def get_state_provider(self) -> Optional[LibraryStateProvider]:  # type: ignore[override]
        return self._state_provider

    def get_keyword_overrides(self):  # type: ignore[override]
        async def _override(session, keyword_name, args, keyword_info):
            return self._execute_compare_images(session, keyword_name, args)

        return {"compare images": _override}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_compare_images(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        args: List[str],
    ) -> Dict[str, Any]:
        session.import_library("DocTest.VisualTest", force=False)
        built_in = BuiltIn()

        try:
            built_in.run_keyword(keyword_name, args)
        except AssertionError as exc:
            summary = self._build_failure_summary(exc)
            session.variables["_doctest_visual_result"] = summary
            return {
                "success": False,
                "output": summary.get("message", "DocTest visual comparison failed."),
                "error": summary.get("message"),
                "state_updates": {"doctest": {"visual": summary}},
            }

        summary = {
            "status": "passed",
            "message": "Visual comparison passed.",
        }
        session.variables["_doctest_visual_result"] = summary
        return {
            "success": True,
            "output": "Visual comparison passed",
            "state_updates": {"doctest": {"visual": summary}},
        }

    def _build_failure_summary(self, exc: AssertionError) -> Dict[str, Any]:
        frames: List[Any] = []
        tb = exc.__traceback__
        while tb:
            frames.append(tb)
            tb = tb.tb_next

        diff_info: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, str]] = []
        message = str(exc)

        for frame in reversed(frames):
            locals_dict = frame.tb_frame.f_locals
            if "diff" in locals_dict and locals_dict.get("diff"):
                diff = locals_dict["diff"]
                message = diff.get("message", message)
            if "detected_differences" in locals_dict:
                for diff in locals_dict["detected_differences"] or []:
                    diff_info.append(self._serialize_diff(diff, artifacts))
                break

        summary = {
            "status": "failed",
            "message": message,
            "differences": diff_info,
            "artifacts": artifacts,
        }
        return summary

    def _serialize_diff(
        self,
        diff: Dict[str, Any],
        artifacts: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        info = {
            "message": diff.get("message"),
            "score": diff.get("score"),
            "threshold": diff.get("threshold"),
            "rectangles": diff.get("rectangles"),
            "notes": diff.get("notes"),
        }

        for label in ("combined_diff", "absolute_diff"):
            image = diff.get(label)
            if image is not None:
                path = self._write_image(image, label)
                if path:
                    artifacts.append({"label": label, "path": path})

        return info

    def _write_image(self, image: Any, label: str) -> Optional[str]:
        try:
            array = np.asarray(image)
            if array.ndim == 2:
                mode_array = array
            elif array.ndim == 3 and array.shape[2] == 3:
                mode_array = array[:, :, ::-1]  # Convert BGR -> RGB
            else:
                mode_array = array

            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{label}.png") as tmp:
                imageio.imwrite(tmp.name, mode_array)
                return tmp.name
        except Exception:
            return None



try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
