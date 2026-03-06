"""Token Accounting Domain Value Objects (ADR-017)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class TokenizerBackend(Enum):
    """Supported tokenizer backends."""

    HEURISTIC = "heuristic"
    CL100K_BASE = "cl100k_base"
    O200K_BASE = "o200k_base"
    LLAMA = "llama"
    MISTRAL = "mistral"
    AUTO = "auto"

    @classmethod
    def from_env(cls) -> "TokenizerBackend":
        """Read backend from ROBOTMCP_TOKENIZER env var."""
        val = os.getenv("ROBOTMCP_TOKENIZER", "heuristic").lower().strip()
        try:
            return cls(val)
        except ValueError:
            return cls.HEURISTIC


@dataclass(frozen=True)
class TokenCount:
    """Immutable token count with provenance."""

    count: int
    backend_used: TokenizerBackend
    is_exact: bool
    confidence: float  # 0.0-1.0

    def __post_init__(self):
        if self.count < 0:
            raise ValueError(f"Token count cannot be negative: {self.count}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be 0.0-1.0, got: {self.confidence}"
            )

    @classmethod
    def heuristic(cls, char_count: int, content_type: str = "json") -> "TokenCount":
        """Create a heuristic estimate (chars / 4)."""
        confidence = {"json": 0.80, "yaml": 0.85, "text": 0.75}.get(
            content_type, 0.75
        )
        return cls(
            count=char_count // 4,
            backend_used=TokenizerBackend.HEURISTIC,
            is_exact=False,
            confidence=confidence,
        )

    @classmethod
    def exact(cls, count: int, backend: TokenizerBackend) -> "TokenCount":
        """Create an exact token count from a real tokenizer."""
        return cls(
            count=count, backend_used=backend, is_exact=True, confidence=1.0
        )

    def to_dict(self):
        return {
            "count": self.count,
            "backend": self.backend_used.value,
            "is_exact": self.is_exact,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TokenBudget:
    """Budget constraint for token usage."""

    max_tokens: int
    warn_threshold: float = 0.8
    hard_limit: Optional[int] = None

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not (0.0 < self.warn_threshold <= 1.0):
            raise ValueError("warn_threshold must be (0, 1]")
        if self.hard_limit is not None and self.hard_limit < self.max_tokens:
            raise ValueError("hard_limit must be >= max_tokens")

    def check(self, token_count: int) -> "BudgetCheckResult":
        """Check a token count against this budget."""
        warn_at = int(self.max_tokens * self.warn_threshold)
        hard = self.hard_limit or self.max_tokens
        if token_count > hard:
            return BudgetCheckResult(
                status="exceeded",
                token_count=token_count,
                budget=self.max_tokens,
                suggestion=f"Reduce by {token_count - self.max_tokens} tokens",
            )
        if token_count > warn_at:
            return BudgetCheckResult(
                status="warning",
                token_count=token_count,
                budget=self.max_tokens,
                suggestion="Approaching budget limit",
            )
        return BudgetCheckResult(
            status="ok",
            token_count=token_count,
            budget=self.max_tokens,
            suggestion=None,
        )


@dataclass(frozen=True)
class BudgetCheckResult:
    """Result of checking token usage against a budget."""

    status: str  # "ok", "warning", "exceeded"
    token_count: int
    budget: int
    suggestion: Optional[str] = None

    def to_dict(self):
        d = {
            "status": self.status,
            "token_count": self.token_count,
            "budget": self.budget,
        }
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


@dataclass(frozen=True)
class TokenReport:
    """Per-tool token breakdown."""

    tool_name: str
    description_tokens: int
    schema_tokens: int
    total: int

    def __post_init__(self):
        if self.total != self.description_tokens + self.schema_tokens:
            raise ValueError(
                f"total ({self.total}) must equal "
                f"description_tokens + schema_tokens"
            )

    def to_dict(self):
        return {
            "tool": self.tool_name,
            "desc": self.description_tokens,
            "schema": self.schema_tokens,
            "total": self.total,
        }


@dataclass(frozen=True)
class ProfileTokenSummary:
    """Aggregate token summary for a tool profile."""

    profile_name: str
    description_mode: str
    total_schema_tokens: int
    tool_reports: Tuple[TokenReport, ...]
    backend_used: TokenizerBackend

    def __post_init__(self):
        expected = sum(r.total for r in self.tool_reports)
        if self.total_schema_tokens != expected:
            raise ValueError(
                f"total_schema_tokens ({self.total_schema_tokens}) "
                f"!= sum of reports ({expected})"
            )

    def to_dict(self):
        return {
            "profile": self.profile_name,
            "mode": self.description_mode,
            "total": self.total_schema_tokens,
            "tools": [r.to_dict() for r in self.tool_reports],
            "backend": self.backend_used.value,
        }


@dataclass(frozen=True)
class TokenRegressionResult:
    """Result of a regression check against a baseline."""

    profile: str
    baseline_tokens: int
    current_tokens: int
    delta: int
    delta_percent: float
    threshold: int
    passed: bool

    def __post_init__(self):
        if self.delta != self.current_tokens - self.baseline_tokens:
            raise ValueError("delta must equal current - baseline")

    def to_dict(self):
        return {
            "profile": self.profile,
            "baseline": self.baseline_tokens,
            "current": self.current_tokens,
            "delta": self.delta,
            "delta_pct": round(self.delta_percent, 2),
            "passed": self.passed,
        }
