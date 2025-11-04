"""Plugin metadata and overrides for DocTest.Ai."""

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
    LibraryMetadata,
    LibraryStateProvider,
    PromptBundle,
)


class _AiStateProvider(LibraryStateProvider):
    async def get_page_source(self, *args, **kwargs):  # type: ignore[override]
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
    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="DocTest.Ai",
            package_name="robotframework-doctestlibrary",
            import_path="DocTest.Ai",
            description="Optional LLM-assisted review and extraction keywords for DocTest.",
            library_type="external",
            use_cases=["llm-assisted comparison", "vision analysis"],
            categories=["ai", "testing"],
            contexts=["desktop"],
            installation_command="pip install \"robotframework-doctestlibrary[ai]\"",
            platform_requirements=["OpenAI-compatible endpoint"],
            load_priority=67,
            default_enabled=False,
        )
        capabilities = LibraryCapabilities(
            contexts=["desktop"],
            features=["llm"],
            technology=["openai"],
        )
        prompts = PromptBundle(
            recommendation="Use DocTest.Ai when human-like judgement is needed to approve or summarise differences.",
            troubleshooting="Configure OpenAI or Azure OpenAI keys via environment variables before importing the library.",
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
            prompt_bundle=prompts,
            install_actions=install_actions,
        )
        self._state_provider = _AiStateProvider()

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        keywords = {
            "chat with document",
            "get text with llm",
            "get text from area with llm",
            "image should contain",
            "get item count from image",
        }
        return {kw: "DocTest.Ai" for kw in keywords}

    def get_state_provider(self) -> Optional[LibraryStateProvider]:  # type: ignore[override]
        return self._state_provider

    def get_keyword_overrides(self):  # type: ignore[override]
        async def _override(session, keyword_name, args, keyword_info):
            if keyword_name.lower() == "image should contain":
                return self._execute_image_should_contain(session, keyword_name, args)
            return None

        return {"image should contain": _override}

    def _execute_image_should_contain(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        args: List[str],
    ) -> Dict[str, Any]:
        session.import_library("DocTest.Ai", force=False)
        built_in = BuiltIn()

        try:
            built_in.run_keyword(keyword_name, args)
        except AssertionError as exc:
            summary = self._build_failure_summary(exc)
            session.variables["_doctest_ai_result"] = summary
            return {
                "success": False,
                "output": summary.get("message", "DocTest AI keyword failed."),
                "error": summary.get("message"),
                "state_updates": {"doctest": {"ai": summary}},
            }

        summary = {
            "status": "passed",
            "message": "LLM confirmed expected content.",
        }
        session.variables["_doctest_ai_result"] = summary
        return {
            "success": True,
            "output": "LLM confirmed expected content",
            "state_updates": {"doctest": {"ai": summary}},
        }

    def _build_failure_summary(self, exc: AssertionError) -> Dict[str, Any]:
        frames: List[Any] = []
        tb = exc.__traceback__
        while tb:
            frames.append(tb)
            tb = tb.tb_next

        decision = None
        prompt = None
        attachments: List[Any] = []

        for frame in reversed(frames):
            locs = frame.tb_frame.f_locals
            if "decision" in locs and decision is None:
                decision = locs["decision"]
            if "combined_prompt" in locs and prompt is None:
                prompt = locs["combined_prompt"]
            if "attachments" in locs and not attachments:
                attachments = locs["attachments"]

        serialized_decision = None
        if decision is not None:
            serialized_decision = {
                "decision": getattr(decision, "decision", None),
                "reason": getattr(decision, "reason", None),
                "confidence": getattr(decision, "confidence", None),
            }

        artifact_paths: List[Dict[str, str]] = []
        for idx, attachment in enumerate(attachments or []):
            path = self._write_binary_attachment(attachment, idx)
            if path:
                artifact_paths.append(path)

        summary = {
            "status": "failed",
            "message": str(exc),
            "prompt": prompt,
            "decision": serialized_decision,
            "artifacts": artifact_paths,
        }
        return summary

    def _write_binary_attachment(self, attachment: Any, index: int) -> Optional[Dict[str, str]]:
        data = getattr(attachment, "data", None)
        media_type = getattr(attachment, "media_type", "application/octet-stream")
        if data is None:
            return None
        suffix = ".png" if "png" in media_type else ".bin"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                return {"index": index, "path": tmp.name, "media_type": media_type}
        except Exception:
            return None


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
