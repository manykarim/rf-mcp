"""Plugin metadata and overrides for DocTest.VisualTest."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import imageio.v2 as imageio
import numpy as np
from robot.libraries.BuiltIn import BuiltIn

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    InstallAction,
    KeywordOverrideHandler,
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
    """Expose VisualTest metadata, overrides, and application state."""

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
                "Ghostscript/GhostPCL",
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
            supports_page_source=False,
            supports_application_state=True,
        )
        hints = LibraryHints(
            standard_keywords=[
                "Compare Images",
                "Compare Images With LLM",
                "Image Should Contain",
                "Get Text From Document",
                "Get Barcodes From Document",
            ],
            error_hints=[
                "Ensure Tesseract, ImageMagick, and Ghostscript binaries are installed and on PATH.",
                "Use `placeholder_file` to mask known differences.",
                "Set `show_diff=${True}` to persist diff artefacts in log.html.",
                "Configure `watermark_file` to ignore recurring overlays.",
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
                "If comparisons fail, confirm documents render at consistent DPI and external binaries are installed. "
                "Use masks or watermarks to ignore intentional differences."
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
        self._override_keywords = ("compare images",)

    # ------------------------------------------------------------------
    # Discovery / state
    # ------------------------------------------------------------------

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        keywords = [
            "compare images",
            "compare images with llm",
            "image should contain",
            "image should not contain",
            "get text from document",
            "get barcodes from document",
            "compare gifs",
        ]
        mapping: Dict[str, str] = {}
        for keyword in keywords:
            for alias in self._expand_aliases(keyword):
                mapping[alias] = "DocTest.VisualTest"
        return mapping

    def on_session_start(self, session: "ExecutionSession") -> None:
        session.variables.setdefault("DOCTEST_SHOW_DIFF", False)
        session.variables.setdefault("DOCTEST_WATERMARK_FILE", None)

    def on_session_end(self, session: "ExecutionSession") -> None:
        session.variables.pop("DOCTEST_SHOW_DIFF", None)
        session.variables.pop("DOCTEST_WATERMARK_FILE", None)

    def get_state_provider(self) -> Optional[LibraryStateProvider]:  # type: ignore[override]
        return self._state_provider

    # ------------------------------------------------------------------
    # Keyword overrides
    # ------------------------------------------------------------------

    def get_keyword_overrides(self) -> Dict[str, KeywordOverrideHandler]:  # type: ignore[override]
        async def _override(session, keyword_name, args, keyword_info):
            return self._execute_compare_images(session, keyword_name, args)

        overrides: Dict[str, KeywordOverrideHandler] = {}
        for keyword in self._override_keywords:
            for alias in self._expand_aliases(keyword):
                overrides[alias] = _override
        return overrides

    def _execute_compare_images(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        args: List[str],
    ) -> Dict[str, Any]:
        session.import_library("DocTest.VisualTest", force=False)
        built_in = BuiltIn()

        normalised_args = [self._normalise_argument(arg) for arg in args]

        try:
            built_in.run_keyword(keyword_name, normalised_args)
        except Exception as exc:  # Catch both assertion failures and runtime errors
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_argument(self, argument: Any) -> Any:
        if not isinstance(argument, str):
            return argument

        if "=" in argument:
            name, value = argument.split("=", 1)
            if self._looks_like_path(value):
                value = self._normalise_path_string(value)
            return f"{name}={value}"

        if self._looks_like_path(argument):
            return self._normalise_path_string(argument)
        return argument

    def _looks_like_path(self, value: str) -> bool:
        lowered = value.lower()
        if any(lowered.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".pdf")):
            return True
        if "\\" in value or "/" in value:
            return True
        if len(value) > 1 and value[1] == ":":
            return True
        return False

    def _normalise_path_string(self, value: str) -> str:
        cleaned = value.replace("\r", "").replace("\n", "").strip()
        if not cleaned:
            return cleaned

        cleaned = cleaned.replace("\\", "/")
        try:
            path = Path(cleaned)
            resolved = path.resolve(strict=False)
            return resolved.as_posix()
        except Exception:
            return cleaned

    def _expand_aliases(self, keyword: str) -> Sequence[str]:
        base = keyword.strip().lower()
        qualified = f"{self.metadata.name.lower()}.{base}"
        return {base, qualified}

    def _build_failure_summary(self, exc: BaseException) -> Dict[str, Any]:
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
            "exception": exc.__class__.__name__,
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
