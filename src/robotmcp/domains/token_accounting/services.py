"""Token Accounting Domain Services (ADR-017)."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional, Protocol

from .value_objects import (
    ProfileTokenSummary,
    TokenBudget,
    TokenCount,
    TokenizerBackend,
    TokenRegressionResult,
    TokenReport,
)

logger = logging.getLogger(__name__)


class TokenEstimator(Protocol):
    """Protocol for token estimation -- consumed by other domains."""

    def estimate(self, content: str, content_type: str = "json") -> TokenCount:
        ...

    def estimate_json(self, obj: Any) -> TokenCount:
        ...


class TokenEstimationService:
    """Pluggable token estimation with backend selection."""

    def __init__(self, backend: Optional[TokenizerBackend] = None):
        self._backend = backend or TokenizerBackend.from_env()
        self._encoder_cache: dict = {}

    @property
    def backend(self) -> TokenizerBackend:
        return self._backend

    def estimate(self, content: str, content_type: str = "json") -> TokenCount:
        """Estimate the token count for a string."""
        backend = self._resolve_backend()
        if backend == TokenizerBackend.HEURISTIC:
            return TokenCount.heuristic(len(content), content_type)
        if backend in (TokenizerBackend.CL100K_BASE, TokenizerBackend.O200K_BASE):
            try:
                enc = self._get_tiktoken_encoder(backend)
                return TokenCount.exact(len(enc.encode(content)), backend)
            except ImportError:
                logger.debug(
                    "tiktoken not installed, falling back to heuristic"
                )
                return TokenCount.heuristic(len(content), content_type)
        return TokenCount.heuristic(len(content), content_type)

    def estimate_json(self, obj: Any) -> TokenCount:
        """Estimate the token count for a JSON-serializable object."""
        try:
            json_str = json.dumps(obj, default=str)
        except (TypeError, ValueError):
            json_str = str(obj)
        return self.estimate(json_str, "json")

    def _resolve_backend(self) -> TokenizerBackend:
        if self._backend == TokenizerBackend.AUTO:
            try:
                import tiktoken  # noqa: F401

                return TokenizerBackend.CL100K_BASE
            except ImportError:
                return TokenizerBackend.HEURISTIC
        return self._backend

    def _get_tiktoken_encoder(self, backend: TokenizerBackend):
        if backend not in self._encoder_cache:
            import tiktoken

            encoding_name = backend.value  # "cl100k_base" or "o200k_base"
            self._encoder_cache[backend] = tiktoken.get_encoding(encoding_name)
        return self._encoder_cache[backend]


class TokenReportingService:
    """Generates per-profile token reports."""

    def __init__(self, estimator: TokenEstimationService):
        self._estimator = estimator

    def measure_tool(
        self, tool_name: str, description: str, schema_json: str
    ) -> TokenReport:
        """Measure tokens for a single tool."""
        desc_count = self._estimator.estimate(description, "text")
        schema_count = self._estimator.estimate(schema_json, "json")
        return TokenReport(
            tool_name=tool_name,
            description_tokens=desc_count.count,
            schema_tokens=schema_count.count,
            total=desc_count.count + schema_count.count,
        )

    def measure_profile(
        self,
        profile_name: str,
        description_mode: str,
        tool_data: Dict[str, Dict[str, str]],
    ) -> ProfileTokenSummary:
        """Measure tokens for an entire profile.

        Args:
            profile_name: Name of the tool profile.
            description_mode: Description mode used (e.g. "full", "compact").
            tool_data: {tool_name: {"description": "...", "schema": "..."}}.
        """
        reports = []
        for name in sorted(tool_data):
            data = tool_data[name]
            report = self.measure_tool(
                name,
                data.get("description", ""),
                data.get("schema", "{}"),
            )
            reports.append(report)
        total = sum(r.total for r in reports)
        return ProfileTokenSummary(
            profile_name=profile_name,
            description_mode=description_mode,
            total_schema_tokens=total,
            tool_reports=tuple(reports),
            backend_used=self._estimator.backend,
        )


class TokenRegressionService:
    """CI-compatible regression checking."""

    def __init__(
        self,
        threshold: int = 50,
        threshold_percent: float = 5.0,
        event_publisher: Optional[Callable] = None,
    ):
        self._threshold = threshold
        self._threshold_percent = threshold_percent
        self._event_publisher = event_publisher

    def check_regression(
        self,
        profile_name: str,
        current_summary: ProfileTokenSummary,
        baseline: Optional[int] = None,
    ) -> TokenRegressionResult:
        """Check whether token usage has regressed.

        Regression is detected when BOTH absolute and percentage thresholds
        are exceeded.
        """
        if baseline is None or baseline == 0:
            return TokenRegressionResult(
                profile=profile_name,
                baseline_tokens=0,
                current_tokens=current_summary.total_schema_tokens,
                delta=current_summary.total_schema_tokens,
                delta_percent=100.0,
                threshold=self._threshold,
                passed=True,
            )

        current = current_summary.total_schema_tokens
        delta = current - baseline
        delta_pct = (delta / baseline * 100.0) if baseline > 0 else 0.0
        abs_exceeded = delta > self._threshold
        pct_exceeded = delta_pct > self._threshold_percent
        passed = not (abs_exceeded and pct_exceeded)

        result = TokenRegressionResult(
            profile=profile_name,
            baseline_tokens=baseline,
            current_tokens=current,
            delta=delta,
            delta_percent=delta_pct,
            threshold=self._threshold,
            passed=passed,
        )

        if not passed and self._event_publisher:
            from .events import TokenRegressionDetected

            self._event_publisher(
                TokenRegressionDetected(
                    profile=profile_name,
                    baseline_tokens=baseline,
                    current_tokens=current,
                    delta=delta,
                    threshold=self._threshold,
                    backend_used=current_summary.backend_used.value,
                )
            )

        return result


# Module-level singleton
_estimation_service: Optional[TokenEstimationService] = None


def get_estimation_service() -> TokenEstimationService:
    """Return a module-level singleton estimation service."""
    global _estimation_service
    if _estimation_service is None:
        _estimation_service = TokenEstimationService()
    return _estimation_service
