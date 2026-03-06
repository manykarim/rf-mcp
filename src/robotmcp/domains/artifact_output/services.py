"""Artifact Output Domain Services (ADR-015).

Application services for externalization and retrieval of artifacts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .aggregates import ArtifactStore
from .entities import Artifact, ArtifactSlice
from .events import LargeFieldExternalized
from .value_objects import (
    ArtifactPolicy,
    ExternalizationResult,
    ExternalizationRule,
    OutputMode,
)

logger = logging.getLogger(__name__)

# Default rules for tools with large outputs
DEFAULT_RULES: List[ExternalizationRule] = [
    ExternalizationRule(tool_name="build_test_suite", field_path="rf_text"),
    ExternalizationRule(
        tool_name="get_session_state", field_path="page_source"
    ),
    ExternalizationRule(
        tool_name="get_session_state", field_path="aria_snapshot"
    ),
    ExternalizationRule(
        tool_name="run_test_suite", field_path="execution_details"
    ),
    ExternalizationRule(tool_name="execute_batch", field_path="steps"),
    ExternalizationRule(tool_name="find_keywords", field_path="result"),
]


class ArtifactExternalizationService:
    """Applies externalization rules to tool responses."""

    def __init__(
        self,
        store: ArtifactStore,
        output_mode: Optional[OutputMode] = None,
        rules: Optional[List[ExternalizationRule]] = None,
    ) -> None:
        self._store = store
        self._mode = output_mode or OutputMode.from_env()
        self._rules = {
            (r.tool_name, r.field_path): r
            for r in (rules or DEFAULT_RULES)
        }

    @property
    def mode(self) -> OutputMode:
        """Current output mode."""
        return self._mode

    def externalize(
        self,
        tool_name: str,
        response: Dict[str, Any],
        session_id: str,
    ) -> Tuple[Dict[str, Any], List[ExternalizationResult]]:
        """Process a tool response, externalizing large fields per rules.

        Returns the (possibly modified) response and a list of results.
        """
        if self._mode == OutputMode.INLINE:
            return response, []

        results: List[ExternalizationResult] = []
        rules = [
            (fp, r)
            for (tn, fp), r in self._rules.items()
            if tn == tool_name
        ]

        for field_path, rule in rules:
            value = self._get_nested(response, field_path)
            if value is None:
                continue
            content = value if isinstance(value, str) else str(value)
            original_tokens = len(content) // 4

            if self._mode == OutputMode.AUTO and not self._store.policy.should_externalize(content):
                continue

            artifact = self._store.create_artifact(
                content=content,
                tool_name=tool_name,
                field_name=field_path,
                session_id=session_id,
            )
            summary = rule.summary_template.format(
                artifact_id=str(artifact.id),
                byte_size=artifact.reference.byte_size,
                token_estimate=artifact.reference.token_estimate,
            )

            self._set_nested(response, field_path, summary)
            saved = original_tokens - len(summary) // 4
            result = ExternalizationResult(
                summary=summary,
                artifact_ref=artifact.reference,
                original_token_estimate=original_tokens,
                saved_tokens=max(0, saved),
            )
            results.append(result)

        return response, results

    def _get_nested(self, d: Dict, path: str) -> Any:
        """Retrieve a value from a nested dict using dot-separated path."""
        parts = path.split(".")
        current: Any = d
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _set_nested(self, d: Dict, path: str, value: Any) -> None:
        """Set a value in a nested dict using dot-separated path."""
        parts = path.split(".")
        current = d
        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current.setdefault(part, {})
        if isinstance(current, dict):
            current[parts[-1]] = value


class ArtifactRetrievalService:
    """Provides artifact fetch and listing capabilities."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def fetch(
        self,
        artifact_id: str,
        offset: int = 0,
        limit: int = 4000,
    ) -> Dict[str, Any]:
        """Fetch artifact content with pagination support."""
        artifact = self._store.get_artifact(artifact_id)
        if artifact is None:
            return {
                "success": False,
                "error": f"Artifact '{artifact_id}' not found or expired",
                "hint": "Re-run the original tool to regenerate",
            }
        slc = self._store.read_slice(artifact_id, offset, limit)
        if slc is None:
            return {
                "success": False,
                "error": "Failed to read artifact content",
            }
        return {"success": True, **slc.to_dict()}

    def list_artifacts(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all artifacts, optionally filtered by session."""
        return [
            a.to_dict()
            for a in self._store.list_artifacts(session_id)
        ]
