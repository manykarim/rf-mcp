"""Tests for Token Accounting Value Objects (ADR-017)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from robotmcp.domains.token_accounting.value_objects import (
    BudgetCheckResult,
    ProfileTokenSummary,
    TokenBudget,
    TokenCount,
    TokenizerBackend,
    TokenRegressionResult,
    TokenReport,
)


# ── TokenizerBackend ──────────────────────────────────────────────────


class TestTokenizerBackend:
    """Tests for TokenizerBackend enum."""

    def test_all_values(self):
        assert TokenizerBackend.HEURISTIC.value == "heuristic"
        assert TokenizerBackend.CL100K_BASE.value == "cl100k_base"
        assert TokenizerBackend.O200K_BASE.value == "o200k_base"
        assert TokenizerBackend.LLAMA.value == "llama"
        assert TokenizerBackend.MISTRAL.value == "mistral"
        assert TokenizerBackend.AUTO.value == "auto"

    def test_from_env_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ROBOTMCP_TOKENIZER", None)
            assert TokenizerBackend.from_env() == TokenizerBackend.HEURISTIC

    def test_from_env_set(self):
        with patch.dict(os.environ, {"ROBOTMCP_TOKENIZER": "cl100k_base"}):
            assert TokenizerBackend.from_env() == TokenizerBackend.CL100K_BASE

    def test_from_env_auto(self):
        with patch.dict(os.environ, {"ROBOTMCP_TOKENIZER": "auto"}):
            assert TokenizerBackend.from_env() == TokenizerBackend.AUTO

    def test_from_env_case_insensitive(self):
        with patch.dict(os.environ, {"ROBOTMCP_TOKENIZER": "CL100K_BASE"}):
            assert TokenizerBackend.from_env() == TokenizerBackend.CL100K_BASE

    def test_from_env_with_whitespace(self):
        with patch.dict(os.environ, {"ROBOTMCP_TOKENIZER": "  heuristic  "}):
            assert TokenizerBackend.from_env() == TokenizerBackend.HEURISTIC

    def test_from_env_invalid_falls_back(self):
        with patch.dict(os.environ, {"ROBOTMCP_TOKENIZER": "nonexistent"}):
            assert TokenizerBackend.from_env() == TokenizerBackend.HEURISTIC

    def test_from_env_empty_string(self):
        with patch.dict(os.environ, {"ROBOTMCP_TOKENIZER": ""}):
            # Empty string is not a valid enum value, falls back
            assert TokenizerBackend.from_env() == TokenizerBackend.HEURISTIC


# ── TokenCount ────────────────────────────────────────────────────────


class TestTokenCount:
    """Tests for TokenCount value object."""

    def test_basic_creation(self):
        tc = TokenCount(
            count=100,
            backend_used=TokenizerBackend.HEURISTIC,
            is_exact=False,
            confidence=0.8,
        )
        assert tc.count == 100
        assert tc.backend_used == TokenizerBackend.HEURISTIC
        assert tc.is_exact is False
        assert tc.confidence == 0.8

    def test_frozen(self):
        tc = TokenCount(
            count=10,
            backend_used=TokenizerBackend.HEURISTIC,
            is_exact=False,
            confidence=0.5,
        )
        with pytest.raises(AttributeError):
            tc.count = 20  # type: ignore[misc]

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match="negative"):
            TokenCount(
                count=-1,
                backend_used=TokenizerBackend.HEURISTIC,
                is_exact=False,
                confidence=0.5,
            )

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="Confidence"):
            TokenCount(
                count=10,
                backend_used=TokenizerBackend.HEURISTIC,
                is_exact=False,
                confidence=-0.1,
            )

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="Confidence"):
            TokenCount(
                count=10,
                backend_used=TokenizerBackend.HEURISTIC,
                is_exact=False,
                confidence=1.1,
            )

    def test_confidence_boundary_zero(self):
        tc = TokenCount(
            count=0,
            backend_used=TokenizerBackend.HEURISTIC,
            is_exact=False,
            confidence=0.0,
        )
        assert tc.confidence == 0.0

    def test_confidence_boundary_one(self):
        tc = TokenCount(
            count=0,
            backend_used=TokenizerBackend.HEURISTIC,
            is_exact=True,
            confidence=1.0,
        )
        assert tc.confidence == 1.0

    def test_count_zero(self):
        tc = TokenCount(
            count=0,
            backend_used=TokenizerBackend.HEURISTIC,
            is_exact=False,
            confidence=0.5,
        )
        assert tc.count == 0

    def test_heuristic_json(self):
        tc = TokenCount.heuristic(400, "json")
        assert tc.count == 100
        assert tc.backend_used == TokenizerBackend.HEURISTIC
        assert tc.is_exact is False
        assert tc.confidence == 0.80

    def test_heuristic_yaml(self):
        tc = TokenCount.heuristic(400, "yaml")
        assert tc.confidence == 0.85

    def test_heuristic_text(self):
        tc = TokenCount.heuristic(400, "text")
        assert tc.confidence == 0.75

    def test_heuristic_unknown_content_type(self):
        tc = TokenCount.heuristic(400, "xml")
        assert tc.confidence == 0.75  # default fallback

    def test_heuristic_zero_chars(self):
        tc = TokenCount.heuristic(0, "json")
        assert tc.count == 0

    def test_heuristic_small_chars(self):
        tc = TokenCount.heuristic(3, "json")
        assert tc.count == 0  # 3 // 4 == 0

    def test_exact_creation(self):
        tc = TokenCount.exact(42, TokenizerBackend.CL100K_BASE)
        assert tc.count == 42
        assert tc.backend_used == TokenizerBackend.CL100K_BASE
        assert tc.is_exact is True
        assert tc.confidence == 1.0

    def test_to_dict(self):
        tc = TokenCount(
            count=50,
            backend_used=TokenizerBackend.CL100K_BASE,
            is_exact=True,
            confidence=1.0,
        )
        d = tc.to_dict()
        assert d == {
            "count": 50,
            "backend": "cl100k_base",
            "is_exact": True,
            "confidence": 1.0,
        }

    def test_to_dict_heuristic(self):
        tc = TokenCount.heuristic(100, "json")
        d = tc.to_dict()
        assert d["backend"] == "heuristic"
        assert d["is_exact"] is False

    def test_equality(self):
        tc1 = TokenCount(
            count=10,
            backend_used=TokenizerBackend.HEURISTIC,
            is_exact=False,
            confidence=0.8,
        )
        tc2 = TokenCount(
            count=10,
            backend_used=TokenizerBackend.HEURISTIC,
            is_exact=False,
            confidence=0.8,
        )
        assert tc1 == tc2

    def test_inequality(self):
        tc1 = TokenCount(
            count=10,
            backend_used=TokenizerBackend.HEURISTIC,
            is_exact=False,
            confidence=0.8,
        )
        tc2 = TokenCount(
            count=11,
            backend_used=TokenizerBackend.HEURISTIC,
            is_exact=False,
            confidence=0.8,
        )
        assert tc1 != tc2


# ── TokenBudget ───────────────────────────────────────────────────────


class TestTokenBudget:
    """Tests for TokenBudget value object."""

    def test_basic_creation(self):
        b = TokenBudget(max_tokens=1000)
        assert b.max_tokens == 1000
        assert b.warn_threshold == 0.8
        assert b.hard_limit is None

    def test_custom_warn_threshold(self):
        b = TokenBudget(max_tokens=1000, warn_threshold=0.9)
        assert b.warn_threshold == 0.9

    def test_with_hard_limit(self):
        b = TokenBudget(max_tokens=1000, hard_limit=1200)
        assert b.hard_limit == 1200

    def test_hard_limit_equals_max(self):
        b = TokenBudget(max_tokens=1000, hard_limit=1000)
        assert b.hard_limit == 1000

    def test_frozen(self):
        b = TokenBudget(max_tokens=1000)
        with pytest.raises(AttributeError):
            b.max_tokens = 2000  # type: ignore[misc]

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TokenBudget(max_tokens=0)

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TokenBudget(max_tokens=-1)

    def test_warn_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="warn_threshold"):
            TokenBudget(max_tokens=1000, warn_threshold=0.0)

    def test_warn_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="warn_threshold"):
            TokenBudget(max_tokens=1000, warn_threshold=1.1)

    def test_warn_threshold_one_ok(self):
        b = TokenBudget(max_tokens=1000, warn_threshold=1.0)
        assert b.warn_threshold == 1.0

    def test_hard_limit_below_max_raises(self):
        with pytest.raises(ValueError, match="hard_limit"):
            TokenBudget(max_tokens=1000, hard_limit=999)

    def test_check_ok(self):
        b = TokenBudget(max_tokens=1000)
        result = b.check(500)
        assert result.status == "ok"
        assert result.suggestion is None

    def test_check_at_boundary_ok(self):
        b = TokenBudget(max_tokens=1000, warn_threshold=0.8)
        # warn_at = 800, so 800 is still ok
        result = b.check(800)
        assert result.status == "ok"

    def test_check_warning(self):
        b = TokenBudget(max_tokens=1000, warn_threshold=0.8)
        result = b.check(801)
        assert result.status == "warning"
        assert result.suggestion == "Approaching budget limit"

    def test_check_warning_at_max(self):
        b = TokenBudget(max_tokens=1000, warn_threshold=0.8)
        result = b.check(1000)
        assert result.status == "warning"

    def test_check_exceeded_no_hard_limit(self):
        b = TokenBudget(max_tokens=1000)
        result = b.check(1001)
        assert result.status == "exceeded"
        assert "Reduce by 1 tokens" in result.suggestion

    def test_check_exceeded_with_hard_limit(self):
        b = TokenBudget(max_tokens=1000, hard_limit=1200)
        # Between max and hard_limit: still warning (not exceeded)
        result = b.check(1100)
        assert result.status == "warning"

    def test_check_exceeded_past_hard_limit(self):
        b = TokenBudget(max_tokens=1000, hard_limit=1200)
        result = b.check(1201)
        assert result.status == "exceeded"

    def test_check_zero_tokens(self):
        b = TokenBudget(max_tokens=1000)
        result = b.check(0)
        assert result.status == "ok"


# ── BudgetCheckResult ─────────────────────────────────────────────────


class TestBudgetCheckResult:
    """Tests for BudgetCheckResult value object."""

    def test_to_dict_ok(self):
        r = BudgetCheckResult(
            status="ok", token_count=100, budget=1000, suggestion=None
        )
        d = r.to_dict()
        assert d == {"status": "ok", "token_count": 100, "budget": 1000}
        assert "suggestion" not in d

    def test_to_dict_with_suggestion(self):
        r = BudgetCheckResult(
            status="warning",
            token_count=900,
            budget=1000,
            suggestion="Approaching budget limit",
        )
        d = r.to_dict()
        assert d["suggestion"] == "Approaching budget limit"

    def test_frozen(self):
        r = BudgetCheckResult(status="ok", token_count=0, budget=100)
        with pytest.raises(AttributeError):
            r.status = "warning"  # type: ignore[misc]


# ── TokenReport ───────────────────────────────────────────────────────


class TestTokenReport:
    """Tests for TokenReport value object."""

    def test_basic_creation(self):
        r = TokenReport(
            tool_name="execute_step",
            description_tokens=50,
            schema_tokens=150,
            total=200,
        )
        assert r.tool_name == "execute_step"
        assert r.total == 200

    def test_total_mismatch_raises(self):
        with pytest.raises(ValueError, match="total"):
            TokenReport(
                tool_name="x",
                description_tokens=50,
                schema_tokens=150,
                total=100,
            )

    def test_zero_tokens(self):
        r = TokenReport(
            tool_name="x",
            description_tokens=0,
            schema_tokens=0,
            total=0,
        )
        assert r.total == 0

    def test_to_dict(self):
        r = TokenReport(
            tool_name="execute_step",
            description_tokens=50,
            schema_tokens=150,
            total=200,
        )
        d = r.to_dict()
        assert d == {
            "tool": "execute_step",
            "desc": 50,
            "schema": 150,
            "total": 200,
        }

    def test_frozen(self):
        r = TokenReport(
            tool_name="x",
            description_tokens=10,
            schema_tokens=20,
            total=30,
        )
        with pytest.raises(AttributeError):
            r.total = 50  # type: ignore[misc]


# ── ProfileTokenSummary ──────────────────────────────────────────────


class TestProfileTokenSummary:
    """Tests for ProfileTokenSummary value object."""

    def _make_reports(self):
        return (
            TokenReport(
                tool_name="a",
                description_tokens=10,
                schema_tokens=20,
                total=30,
            ),
            TokenReport(
                tool_name="b",
                description_tokens=5,
                schema_tokens=15,
                total=20,
            ),
        )

    def test_basic_creation(self):
        reports = self._make_reports()
        s = ProfileTokenSummary(
            profile_name="full",
            description_mode="verbose",
            total_schema_tokens=50,
            tool_reports=reports,
            backend_used=TokenizerBackend.HEURISTIC,
        )
        assert s.total_schema_tokens == 50

    def test_total_mismatch_raises(self):
        reports = self._make_reports()
        with pytest.raises(ValueError, match="total_schema_tokens"):
            ProfileTokenSummary(
                profile_name="full",
                description_mode="verbose",
                total_schema_tokens=999,
                tool_reports=reports,
                backend_used=TokenizerBackend.HEURISTIC,
            )

    def test_empty_reports(self):
        s = ProfileTokenSummary(
            profile_name="empty",
            description_mode="none",
            total_schema_tokens=0,
            tool_reports=(),
            backend_used=TokenizerBackend.HEURISTIC,
        )
        assert s.total_schema_tokens == 0
        assert len(s.tool_reports) == 0

    def test_to_dict(self):
        reports = self._make_reports()
        s = ProfileTokenSummary(
            profile_name="full",
            description_mode="verbose",
            total_schema_tokens=50,
            tool_reports=reports,
            backend_used=TokenizerBackend.HEURISTIC,
        )
        d = s.to_dict()
        assert d["profile"] == "full"
        assert d["mode"] == "verbose"
        assert d["total"] == 50
        assert len(d["tools"]) == 2
        assert d["backend"] == "heuristic"

    def test_frozen(self):
        reports = self._make_reports()
        s = ProfileTokenSummary(
            profile_name="full",
            description_mode="verbose",
            total_schema_tokens=50,
            tool_reports=reports,
            backend_used=TokenizerBackend.HEURISTIC,
        )
        with pytest.raises(AttributeError):
            s.profile_name = "x"  # type: ignore[misc]


# ── TokenRegressionResult ────────────────────────────────────────────


class TestTokenRegressionResult:
    """Tests for TokenRegressionResult value object."""

    def test_basic_creation(self):
        r = TokenRegressionResult(
            profile="full",
            baseline_tokens=100,
            current_tokens=120,
            delta=20,
            delta_percent=20.0,
            threshold=50,
            passed=True,
        )
        assert r.passed is True
        assert r.delta == 20

    def test_delta_mismatch_raises(self):
        with pytest.raises(ValueError, match="delta must equal"):
            TokenRegressionResult(
                profile="full",
                baseline_tokens=100,
                current_tokens=120,
                delta=999,
                delta_percent=20.0,
                threshold=50,
                passed=True,
            )

    def test_negative_delta(self):
        """Token reduction (improvement) yields negative delta."""
        r = TokenRegressionResult(
            profile="full",
            baseline_tokens=100,
            current_tokens=80,
            delta=-20,
            delta_percent=-20.0,
            threshold=50,
            passed=True,
        )
        assert r.delta == -20

    def test_to_dict(self):
        r = TokenRegressionResult(
            profile="full",
            baseline_tokens=100,
            current_tokens=120,
            delta=20,
            delta_percent=20.123456,
            threshold=50,
            passed=True,
        )
        d = r.to_dict()
        assert d == {
            "profile": "full",
            "baseline": 100,
            "current": 120,
            "delta": 20,
            "delta_pct": 20.12,
            "passed": True,
        }

    def test_to_dict_rounds_percent(self):
        r = TokenRegressionResult(
            profile="x",
            baseline_tokens=100,
            current_tokens=105,
            delta=5,
            delta_percent=5.555555,
            threshold=50,
            passed=True,
        )
        assert r.to_dict()["delta_pct"] == 5.56

    def test_frozen(self):
        r = TokenRegressionResult(
            profile="x",
            baseline_tokens=100,
            current_tokens=100,
            delta=0,
            delta_percent=0.0,
            threshold=50,
            passed=True,
        )
        with pytest.raises(AttributeError):
            r.passed = False  # type: ignore[misc]

    def test_zero_delta(self):
        r = TokenRegressionResult(
            profile="x",
            baseline_tokens=100,
            current_tokens=100,
            delta=0,
            delta_percent=0.0,
            threshold=50,
            passed=True,
        )
        assert r.delta == 0
        assert r.delta_percent == 0.0
