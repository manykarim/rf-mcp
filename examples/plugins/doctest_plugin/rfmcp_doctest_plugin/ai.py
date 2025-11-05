"""Plugin metadata and overrides for DocTest.Ai."""

from __future__ import annotations

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

_MAX_STRING_LENGTH = 2000


class _AiStateProvider(LibraryStateProvider):
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
        summary = session.variables.get("_doctest_ai_result")
        if not summary:
            return {"success": False, "error": "No DocTest AI result available."}
        return {"success": True, "ai": summary}


class DocTestAiPlugin(StaticLibraryPlugin):
    """Expose DocTest.Ai metadata and capture LLM interactions."""

    _CONTEXT_FIELDS: Dict[str, Sequence[str]] = {
        "get text with llm": (
            "document",
            "prompt",
            "include_pdf_text",
            "model",
            "provider",
            "max_pages",
        ),
        "get text from area with llm": (
            "document",
            "area",
            "prompt",
            "model",
            "provider",
        ),
        "chat with document": (
            "prompt",
            "documents",
            "include_pdf_text",
            "model",
            "provider",
        ),
        "image should contain": (
            "document",
            "expected",
            "prompt",
            "model",
            "provider",
        ),
        "get item count from image": (
            "document",
            "item_description",
            "prompt",
            "model",
            "provider",
        ),
    }

    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="DocTest.Ai",
            package_name="robotframework-doctestlibrary",
            import_path="DocTest.Ai",
            description="LLM-assisted document analysis, transcription, vision detection, and question answering.",
            library_type="external",
            use_cases=[
                "vision-assisted regression testing",
                "document transcription",
                "LLM-powered document question answering",
                "visual object detection and counting",
            ],
            categories=["ai", "documents", "vision"],
            contexts=["desktop"],
            installation_command='pip install "robotframework-doctestlibrary[ai]"',
            dependencies=[
                "robotframework-doctestlibrary[ai]",
            ],
            requires_type_conversion=False,
            supports_async=False,
            load_priority=67,
            default_enabled=False,
            extra_name="ai",
        )
        capabilities = LibraryCapabilities(
            contexts=["desktop"],
            features=["llm-vision", "document-qa", "ocr"],
            technology=["llm", "vision"],
            supports_application_state=True,
        )
        hints = LibraryHints(
            standard_keywords=[
                "Get Text With LLM",
                "Get Text From Area With LLM",
                "Chat With Document",
                "Image Should Contain",
                "Get Item Count From Image",
            ],
            error_hints=[
                "Configure API keys (e.g. OPENAI_API_KEY) before invoking LLM-powered keywords.",
                "Reduce `max_pages` or DPI when documents exceed provider attachment limits.",
                "Check the collected diagnostics via get_application_state after a failure for model responses.",
            ],
            usage_examples=[
                "Get Text With LLM    invoice.pdf    prompt=Transcribe the invoice text",
                "Chat With Document    What is the total amount due?    documents=${docs}",
                "Image Should Contain    flyer.png    Expected logo in top-left corner",
            ],
        )
        prompts = PromptBundle(
            recommendation="Use DocTest.Ai when you need LLM-assisted OCR, document Q&A, or visual verification.",
            troubleshooting=(
                "If a keyword fails, inspect the stored response in get_application_state. "
                "Adjust prompts, model overrides, or DPI when providers reject attachments."
            ),
            sampling_notes=(
                "Include installation instructions for optional extras and mention environment variables "
                "such as OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT when guiding users."
            ),
        )
        install_actions = [
            InstallAction(
                description="Install DocTestLibrary with AI extras",
                command=["pip", "install", "robotframework-doctestlibrary[ai]"],
            )
        ]

        super().__init__(
            metadata=metadata,
            capabilities=capabilities,
            hints=hints,
            prompt_bundle=prompts,
            install_actions=install_actions,
        )
        self._state_provider = _AiStateProvider()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        keywords = [
            "get text with llm",
            "get text from area with llm",
            "chat with document",
            "image should contain",
            "get item count from image",
        ]
        mapping: Dict[str, str] = {}
        for keyword in keywords:
            for alias in self._expand_aliases(keyword):
                mapping[alias] = "DocTest.Ai"
        return mapping

    def get_state_provider(self) -> Optional[LibraryStateProvider]:  # type: ignore[override]
        return self._state_provider

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def get_keyword_overrides(self) -> Dict[str, KeywordOverrideHandler]:  # type: ignore[override]
        async def _handler(session, keyword_name, args, keyword_info):
            return self._execute_keyword(session, keyword_name, args)

        overrides: Dict[str, KeywordOverrideHandler] = {}
        for keyword in self._CONTEXT_FIELDS.keys():
            for alias in self._expand_aliases(keyword):
                overrides[alias] = _handler
        return overrides

    def _execute_keyword(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        args: Sequence[str],
    ) -> Dict[str, Any]:
        session.import_library("DocTest.Ai", force=False)
        built_in = BuiltIn()

        try:
            result = built_in.run_keyword(keyword_name, list(args))
        except Exception as exc:
            summary = self._build_failure_summary(keyword_name, args, exc)
            session.variables["_doctest_ai_result"] = summary
            return {
                "success": False,
                "output": summary.get("message", f"{keyword_name} failed."),
                "error": summary.get("message"),
                "state_updates": {"doctest": {"ai": summary}},
            }

        summary = self._build_success_summary(keyword_name, args, result)
        session.variables["_doctest_ai_result"] = summary
        message = summary.get("message") or f"{keyword_name} succeeded."
        payload: Dict[str, Any] = {
            "success": True,
            "output": message,
            "state_updates": {"doctest": {"ai": summary}},
        }
        if result is not None:
            payload["return_value"] = result
        return payload

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _expand_aliases(self, keyword: str) -> Set[str]:
        base = keyword.strip().lower()
        qualified = f"{self.metadata.name.lower()}.{base}"
        return {base, qualified}

    def _build_success_summary(
        self,
        keyword_name: str,
        args: Sequence[str],
        result: Any,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "status": "passed",
            "keyword": keyword_name,
            "arguments": self._serialize(list(args)),
        }
        if result is not None:
            serialized = self._serialize(result)
            summary["result"] = serialized
            summary["message"] = (
                serialized if isinstance(serialized, str) else "Keyword executed successfully."
            )
        else:
            summary["message"] = "Keyword executed successfully."
        return summary

    def _build_failure_summary(
        self,
        keyword_name: str,
        args: Sequence[str],
        exc: BaseException,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "status": "failed",
            "keyword": keyword_name,
            "arguments": self._serialize(list(args)),
            "message": str(exc) or f"{keyword_name} failed.",
            "exception": exc.__class__.__name__,
        }

        frames = self._collect_frames(exc)
        keyword_key = keyword_name.strip().lower()
        fields = self._CONTEXT_FIELDS.get(keyword_key, ())
        context: Dict[str, Any] = {}

        for frame in frames:
            locals_dict = frame.tb_frame.f_locals
            for field in fields:
                if field not in context and field in locals_dict:
                    context[field] = self._serialize(locals_dict[field])
            if keyword_key == "image should contain" and "decision" in locals_dict and "decision" not in context:
                context["decision"] = self._serialize(locals_dict["decision"])
            if keyword_key == "get item count from image" and "decision" in locals_dict and "decision" not in context:
                context["decision"] = self._serialize(locals_dict["decision"])
            if keyword_key == "chat with document" and "result" in locals_dict and "result" not in context:
                context["llm_response"] = self._serialize(locals_dict["result"])

        if context:
            summary["context"] = context
        return summary

    def _serialize(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace")
                except Exception:
                    value = repr(value)
            text = str(value)
            if len(text) > _MAX_STRING_LENGTH:
                return text[:_MAX_STRING_LENGTH] + "…"
            return text
        if isinstance(value, (int, float, bool)):
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


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
