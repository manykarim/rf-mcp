"""Builtin RequestsLibrary plugin providing session hooks."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import LibraryCapabilities, LibraryMetadata

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


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
