"""Builtin RequestsLibrary plugin providing session hooks."""

from __future__ import annotations

import logging
from typing import Dict, Optional, List, Any

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import LibraryCapabilities, LibraryMetadata
from robotmcp.models.session_models import ExecutionSession

logger = logging.getLogger(__name__)


class RequestsLibraryPlugin(StaticLibraryPlugin):
    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="RequestsLibrary",
            package_name="robotframework-requests",
            import_path="RequestsLibrary",
            description="HTTP API testing by wrapping Python Requests Library",
            library_type="external",
            use_cases=["api testing", "http requests", "rest api", "json validation"],
            categories=["api", "testing", "network"],
            contexts=["api"],
            installation_command="pip install robotframework-requests",
            requires_type_conversion=True,
            supports_async=False,
            load_priority=6,
            default_enabled=True,
            extra_name="api",
        )
        capabilities = LibraryCapabilities(
            contexts=["api"],
            requires_type_conversion=True,
            features=["session-management"],
        )
        super().__init__(metadata=metadata, capabilities=capabilities)

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        keywords = {
            "get",
            "post",
            "put",
            "delete",
            "patch",
            "head",
            "options",
            "get on session",
            "post on session",
            "put on session",
            "delete on session",
            "create session",
            "delete all sessions",
        }
        return {keyword: "RequestsLibrary" for keyword in keywords}

    def before_keyword_execution(  # type: ignore[override]
        self,
        session: "ExecutionSession",
        keyword_name: str,
        library_manager,
        keyword_discovery,
    ) -> None:
        try:
            library_manager.ensure_library_in_rf_context("RequestsLibrary")
        except Exception as exc:  # pragma: no cover
            logger.debug("RequestsLibrary RF registration failed: %s", exc)

        manager = getattr(session, "_session_manager", None)
        sync = getattr(manager, "synchronize_requests_library_state", None)
        if callable(sync):
            try:
                sync(session)
            except Exception as exc:  # pragma: no cover
                logger.debug("RequestsLibrary session sync failed: %s", exc)

    def on_session_start(self, session: "ExecutionSession") -> None:
        manager = getattr(session, "_session_manager", None)
        sync = getattr(manager, "synchronize_requests_library_state", None)
        if callable(sync):
            try:
                sync(session)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("RequestsLibrary session sync failed: %s", exc)

    def generate_failure_hints(
        self,
        session: ExecutionSession,
        keyword_name: str,
        arguments: List[Any],
        error_text: str,
    ) -> List[Dict[str, Any]]:
        kw_clean = (keyword_name or "").strip()
        if kw_clean.lower().startswith("requestslibrary."):
            kw_clean = kw_clean.split(".", 1)[1]
        kw_lower = kw_clean.lower()
        err = (error_text or "").lower()
        args = arguments or []

        hints: List[Dict[str, Any]] = []

        if kw_lower in {
            "get",
            "post",
            "put",
            "patch",
            "delete",
            "post on session",
            "put on session",
            "patch on session",
            "get on session",
            "delete on session",
        }:
            first_arg = args[0] if args else ""
            is_session_keyword = kw_lower.endswith(" on session")
            first_is_url = isinstance(first_arg, str) and first_arg.lower().startswith(
                ("http://", "https://")
            )
            missing_scheme = "missingschema" in err or "no scheme" in err

            if missing_scheme or (not is_session_keyword and not first_is_url):
                hints.append(self._build_session_hint())
            if "httperror" in err:
                hints.append(self._build_payload_hint(is_session_keyword))

        return [hint for hint in hints if hint]

    def _build_session_hint(self) -> Dict[str, Any]:
        base_url = "https://restful-booker.herokuapp.com"
        return {
            "title": "RequestsLibrary: Provide a full URL or use sessions",
            "message": (
                "The first argument should be a full URL (https://...) for sessionless keywords, "
                "or you should create a session and call Get/Post On Session with an alias and relative path."
            ),
            "examples": [
                {
                    "tool": "execute_step",
                    "keyword": "Create Session",
                    "arguments": ["rb", base_url],
                },
                {
                    "tool": "execute_step",
                    "keyword": "Get On Session",
                    "arguments": ["rb", "/booking/1"],
                    "use_context": True,
                },
                {
                    "tool": "execute_step",
                    "keyword": "Get",
                    "arguments": [f"{base_url}/booking/1"],
                },
            ],
        }

    def _build_payload_hint(self, is_session_keyword: bool) -> Dict[str, Any]:
        base_url = "https://restful-booker.herokuapp.com"
        target_keyword = "Post On Session" if is_session_keyword else "POST"
        target_args = (
            [
                "rb",
                "/booking",
                "json=${booking}",
                "headers=${headers}",
            ]
            if is_session_keyword
            else [
                f"{base_url}/booking",
                "json=${booking}",
                "headers=${headers}",
            ]
        )
        return {
            "title": "RequestsLibrary: Handle HTTPError responses",
            "message": (
                "HTTP errors often indicate payload or header issues. Build the payload as a Python dict, pass it via json=, "
                "and include Content-Type/Accept headers."
            ),
            "examples": [
                {
                    "tool": "execute_step",
                    "keyword": "Create Session",
                    "arguments": ["rb", base_url],
                },
                {
                    "tool": "execute_step",
                    "keyword": "Evaluate",
                    "arguments": [
                        "{\n  'firstname':'Jim',\n  'lastname':'Brown',\n  'totalprice':111,\n  'depositpaid':True,\n  'bookingdates':{'checkin':'2018-01-01','checkout':'2019-01-01'},\n  'additionalneeds':'Breakfast'\n}"
                    ],
                    "assign_to": "booking",
                    "use_context": True,
                },
                {
                    "tool": "execute_step",
                    "keyword": "Create Dictionary",
                    "arguments": [
                        "Content-Type",
                        "application/json",
                        "Accept",
                        "application/json",
                    ],
                    "assign_to": "headers",
                    "use_context": True,
                },
                {
                    "tool": "execute_step",
                    "keyword": target_keyword,
                    "arguments": target_args,
                    "use_context": True,
                },
            ],
        }


