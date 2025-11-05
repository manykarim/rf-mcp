"""Plugin metadata and overrides for DocTest.PrintJobTests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

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


class _PrintJobStateProvider(LibraryStateProvider):
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
        summary = session.variables.get("_doctest_print_result")
        if not summary:
            return {"success": False, "error": "No DocTest print job result available."}
        return {"success": True, "print": summary}


class DocTestPrintJobPlugin(StaticLibraryPlugin):
    """Expose DocTest.PrintJobTests metadata and capture comparison diffs."""

    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="DocTest.PrintJobTests",
            package_name="robotframework-doctestlibrary",
            import_path="DocTest.PrintJobTests",
            description="Parse and compare PostScript/PCL print jobs, including per-page metadata checks.",
            library_type="external",
            use_cases=[
                "print spool regression testing",
                "printer configuration validation",
                "document preflight inspection",
            ],
            categories=["printing", "documents"],
            contexts=["desktop"],
            installation_command="pip install robotframework-doctestlibrary",
            platform_requirements=["Ghostscript (for PostScript parsing)"],
            dependencies=["parsimonious", "deepdiff"],
            requires_type_conversion=False,
            supports_async=False,
            load_priority=66,
            default_enabled=False,
        )
        capabilities = LibraryCapabilities(
            contexts=["desktop"],
            features=["print-diff", "metadata-inspection"],
            technology=["parsimonious", "deepdiff"],
            supports_application_state=True,
        )
        hints = LibraryHints(
            standard_keywords=[
                "Get Pcl Print Job",
                "Get Postscript Print Job",
                "Compare Print Jobs",
                "Check Print Job Property",
            ],
            error_hints=[
                "Ensure Ghostscript (or compatible PostScript interpreter) is installed for PostScript comparisons.",
                "Normalise printer-specific properties (paper trays, duplex) before comparing across devices.",
                "Use masks or targeted property checks when only specific metadata matters.",
            ],
            usage_examples=[
                "Compare Print Jobs    pcl    reference.pcl    candidate.pcl",
                "Check Print Job Property    ${job}    pcl_commands    ${expected}",
            ],
        )
        prompts = PromptBundle(
            recommendation="Use DocTest.PrintJobTests to diff print spool files and verify per-page printer settings.",
            troubleshooting=(
                "When comparisons fail, inspect the collected differences via get_application_state "
                "or lower the scope with Check Print Job Property."
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
        self._state_provider = _PrintJobStateProvider()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        keywords = [
            "get pcl print job",
            "get postscript print job",
            "compare print jobs",
            "check print job property",
        ]
        mapping: Dict[str, str] = {}
        for keyword in keywords:
            for alias in self._expand_aliases(keyword):
                mapping[alias] = "DocTest.PrintJobTests"
        return mapping

    def get_state_provider(self) -> Optional[LibraryStateProvider]:  # type: ignore[override]
        return self._state_provider

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def get_keyword_overrides(self) -> Dict[str, KeywordOverrideHandler]:  # type: ignore[override]
        async def _compare(session, keyword_name, args, keyword_info):
            return self._execute_compare_print_jobs(session, keyword_name, args)

        async def _check_property(session, keyword_name, args, keyword_info):
            return self._execute_check_property(session, keyword_name, args)

        overrides: Dict[str, KeywordOverrideHandler] = {}
        for alias in self._expand_aliases("compare print jobs"):
            overrides[alias] = _compare
        for alias in self._expand_aliases("check print job property"):
            overrides[alias] = _check_property
        return overrides

    def _execute_compare_print_jobs(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        args: Sequence[str],
    ) -> Dict[str, Any]:
        session.import_library("DocTest.PrintJobTests", force=False)
        built_in = BuiltIn()

        normalised_args = [self._normalise_argument(arg, index=i) for i, arg in enumerate(args)]

        try:
            built_in.run_keyword(keyword_name, list(normalised_args))
        except Exception as exc:
            summary = self._build_compare_failure_summary(exc)
            session.variables["_doctest_print_result"] = summary
            return {
                "success": False,
                "output": summary.get("message", "DocTest print job comparison failed."),
                "error": summary.get("message"),
                "state_updates": {"doctest": {"print": summary}},
            }

        summary = {
            "status": "passed",
            "message": "Print jobs comparison passed.",
        }
        session.variables["_doctest_print_result"] = summary
        return {
            "success": True,
            "output": summary["message"],
            "state_updates": {"doctest": {"print": summary}},
        }

    def _execute_check_property(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        args: Sequence[str],
    ) -> Dict[str, Any]:
        session.import_library("DocTest.PrintJobTests", force=False)
        built_in = BuiltIn()

        normalised_args = [self._normalise_argument(arg, index=i) for i, arg in enumerate(args)]

        try:
            built_in.run_keyword(keyword_name, list(normalised_args))
        except Exception as exc:
            summary = self._build_property_failure_summary(exc)
            session.variables["_doctest_print_result"] = summary
            return {
                "success": False,
                "output": summary.get("message", "DocTest print job property check failed."),
                "error": summary.get("message"),
                "state_updates": {"doctest": {"print": summary}},
            }

        summary = {
            "status": "passed",
            "message": "Print job property matches expected value.",
        }
        session.variables["_doctest_print_result"] = summary
        return {
            "success": True,
            "output": summary["message"],
            "state_updates": {"doctest": {"print": summary}},
        }

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _expand_aliases(self, keyword: str) -> Set[str]:
        base = keyword.strip().lower()
        qualified = f"{self.metadata.name.lower()}.{base}"
        return {base, qualified}

    def _build_compare_failure_summary(self, exc: BaseException) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "status": "failed",
            "message": str(exc) or "DocTest print job comparison failed.",
            "exception": exc.__class__.__name__,
        }

        frames = self._collect_frames(exc)
        differences = None
        reference_job = None
        candidate_job = None

        for frame in frames:
            locals_dict = frame.tb_frame.f_locals
            if differences is None and "list_difference" in locals_dict:
                differences = self._serialize(locals_dict.get("list_difference"))
            if reference_job is None and "reference_print_job" in locals_dict:
                reference_job = self._serialize_print_job(
                    locals_dict.get("reference_print_job")
                )
            if candidate_job is None and "test_print_job" in locals_dict:
                candidate_job = self._serialize_print_job(
                    locals_dict.get("test_print_job")
                )

        if differences:
            summary["differences"] = differences
        if reference_job:
            summary["reference"] = reference_job
        if candidate_job:
            summary["candidate"] = candidate_job
        return summary

    def _build_property_failure_summary(self, exc: BaseException) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "status": "failed",
            "message": str(exc) or "DocTest print job property check failed.",
            "exception": exc.__class__.__name__,
        }

        frames = self._collect_frames(exc)
        property_name = None
        expected_value = None
        actual_value = None

        for frame in frames:
            locals_dict = frame.tb_frame.f_locals
            if property_name is None and "property" in locals_dict:
                property_name = locals_dict.get("property")
            if expected_value is None and "value" in locals_dict:
                expected_value = locals_dict.get("value")
            if actual_value is None and "test_property_item" in locals_dict:
                actual_value = locals_dict.get("test_property_item")

        if property_name is not None:
            summary["property"] = self._serialize(property_name)
        if expected_value is not None:
            summary["expected"] = self._serialize(expected_value)
        if actual_value is not None:
            summary["actual"] = self._serialize(actual_value)
        return summary

    def _serialize_print_job(self, job: Any) -> Optional[Dict[str, Any]]:
        if job is None:
            return None
        jobtype = getattr(job, "jobtype", None)
        properties = getattr(job, "properties", None)
        return {
            "type": self._serialize(jobtype),
            "properties": self._serialize(properties),
        }

    def _serialize(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            return {
                k: self._serialize(v)
                for k, v in value.__dict__.items()
                if not k.startswith("_")
            }
        return str(value)

    def _collect_frames(self, exc: BaseException) -> List[Any]:
        frames: List[Any] = []
        tb = exc.__traceback__
        while tb:
            frames.append(tb)
            tb = tb.tb_next
        return list(reversed(frames))

    def _normalise_argument(self, argument: Any, index: int = 0) -> Any:
        if not isinstance(argument, str):
            return argument
        if "=" in argument:
            name, value = argument.split("=", 1)
            if self._looks_like_path(value):
                value = self._normalise_path_string(value)
            return f"{name}={value}"

        # Positional arguments: index 1 and 2 are typically file paths
        if self._looks_like_path(argument) or index in (1, 2):
            return self._normalise_path_string(argument)
        return argument

    def _looks_like_path(self, value: str) -> bool:
        lowered = value.lower()
        if any(lowered.endswith(ext) for ext in (".pcl", ".ps", ".pdf", ".png", ".jpg", ".jpeg")):
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


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
