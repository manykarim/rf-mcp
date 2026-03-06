"""Tests for Token Accounting Services (ADR-017)."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.domains.token_accounting.services import (
    TokenEstimationService,
    TokenReportingService,
    TokenRegressionService,
    get_estimation_service,
)
from robotmcp.domains.token_accounting.value_objects import (
    ProfileTokenSummary,
    TokenCount,
    TokenizerBackend,
    TokenReport,
)


# ── TokenEstimationService ────────────────────────────────────────────


class TestTokenEstimationServiceHeuristic:
    """Tests for heuristic backend."""

    def test_estimate_json(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        tc = svc.estimate("a" * 400, "json")
        assert tc.count == 100
        assert tc.backend_used == TokenizerBackend.HEURISTIC
        assert tc.is_exact is False
        assert tc.confidence == 0.80

    def test_estimate_text(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        tc = svc.estimate("hello world", "text")
        assert tc.count == len("hello world") // 4
        assert tc.confidence == 0.75

    def test_estimate_yaml(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        tc = svc.estimate("key: value", "yaml")
        assert tc.confidence == 0.85

    def test_estimate_unknown_content_type(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        tc = svc.estimate("data", "xml")
        assert tc.confidence == 0.75

    def test_estimate_empty_string(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        tc = svc.estimate("", "json")
        assert tc.count == 0

    def test_estimate_single_char(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        tc = svc.estimate("a", "json")
        assert tc.count == 0  # 1 // 4 == 0

    def test_estimate_json_object(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        obj = {"key": "value", "num": 42}
        tc = svc.estimate_json(obj)
        expected_str = json.dumps(obj, default=str)
        assert tc.count == len(expected_str) // 4

    def test_estimate_json_non_serializable(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        obj = object()
        tc = svc.estimate_json(obj)
        # Falls back to str(obj)
        assert tc.count >= 0

    def test_estimate_json_list(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        tc = svc.estimate_json([1, 2, 3])
        assert tc.count >= 0

    def test_estimate_json_nested(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        obj = {"a": {"b": {"c": [1, 2, 3]}}}
        tc = svc.estimate_json(obj)
        expected_str = json.dumps(obj, default=str)
        assert tc.count == len(expected_str) // 4

    def test_backend_property(self):
        svc = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        assert svc.backend == TokenizerBackend.HEURISTIC


class TestTokenEstimationServiceTiktoken:
    """Tests for tiktoken-backed estimation using mocks."""

    def _make_mock_tiktoken(self, token_count: int):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = list(range(token_count))
        mock_tiktoken = MagicMock()
        mock_tiktoken.get_encoding.return_value = mock_enc
        return mock_tiktoken, mock_enc

    def test_cl100k_base_exact(self):
        mock_tiktoken, mock_enc = self._make_mock_tiktoken(42)
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            svc = TokenEstimationService(backend=TokenizerBackend.CL100K_BASE)
            # Clear any cached encoder
            svc._encoder_cache = {}
            tc = svc.estimate("hello world", "text")
            assert tc.count == 42
            assert tc.backend_used == TokenizerBackend.CL100K_BASE
            assert tc.is_exact is True
            assert tc.confidence == 1.0
            mock_tiktoken.get_encoding.assert_called_with("cl100k_base")

    def test_o200k_base_exact(self):
        mock_tiktoken, mock_enc = self._make_mock_tiktoken(99)
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            svc = TokenEstimationService(backend=TokenizerBackend.O200K_BASE)
            svc._encoder_cache = {}
            tc = svc.estimate("test content", "json")
            assert tc.count == 99
            assert tc.backend_used == TokenizerBackend.O200K_BASE
            assert tc.is_exact is True
            mock_tiktoken.get_encoding.assert_called_with("o200k_base")

    def test_tiktoken_import_error_fallback(self):
        """Falls back to heuristic when tiktoken is not installed."""
        svc = TokenEstimationService(backend=TokenizerBackend.CL100K_BASE)
        svc._encoder_cache = {}

        def _raise_import(*args, **kwargs):
            raise ImportError("no tiktoken")

        with patch.object(svc, "_get_tiktoken_encoder", side_effect=_raise_import):
            tc = svc.estimate("a" * 100, "json")
            assert tc.backend_used == TokenizerBackend.HEURISTIC
            assert tc.is_exact is False
            assert tc.count == 25  # 100 // 4

    def test_encoder_caching(self):
        mock_tiktoken, mock_enc = self._make_mock_tiktoken(10)
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            svc = TokenEstimationService(backend=TokenizerBackend.CL100K_BASE)
            svc._encoder_cache = {}
            svc.estimate("a", "text")
            svc.estimate("b", "text")
            # get_encoding should only be called once due to caching
            assert mock_tiktoken.get_encoding.call_count == 1

    def test_different_backends_different_cache_entries(self):
        mock_tiktoken, _ = self._make_mock_tiktoken(10)
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            svc = TokenEstimationService(backend=TokenizerBackend.CL100K_BASE)
            svc._encoder_cache = {}
            svc._get_tiktoken_encoder(TokenizerBackend.CL100K_BASE)
            svc._get_tiktoken_encoder(TokenizerBackend.O200K_BASE)
            assert mock_tiktoken.get_encoding.call_count == 2


class TestTokenEstimationServiceAuto:
    """Tests for AUTO backend resolution."""

    def test_auto_resolves_to_cl100k_when_tiktoken_available(self):
        mock_tiktoken, _ = self._make_mock_tiktoken(5)
        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            svc = TokenEstimationService(backend=TokenizerBackend.AUTO)
            svc._encoder_cache = {}
            tc = svc.estimate("hello", "text")
            assert tc.count == 5
            assert tc.backend_used == TokenizerBackend.CL100K_BASE

    def test_auto_resolves_to_heuristic_when_no_tiktoken(self):
        svc = TokenEstimationService(backend=TokenizerBackend.AUTO)

        # Simulate tiktoken not importable
        with patch.dict(sys.modules, {"tiktoken": None}):
            with patch("builtins.__import__", side_effect=ImportError("no tiktoken")):
                tc = svc.estimate("a" * 40, "json")
                assert tc.backend_used == TokenizerBackend.HEURISTIC
                assert tc.count == 10

    def _make_mock_tiktoken(self, token_count: int):
        mock_enc = MagicMock()
        mock_enc.encode.return_value = list(range(token_count))
        mock_tiktoken = MagicMock()
        mock_tiktoken.get_encoding.return_value = mock_enc
        return mock_tiktoken, mock_enc


class TestTokenEstimationServiceEnvDefault:
    """Tests for env-based default backend."""

    def test_default_from_env_heuristic(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ROBOTMCP_TOKENIZER", None)
            svc = TokenEstimationService()
            assert svc.backend == TokenizerBackend.HEURISTIC

    def test_default_from_env_cl100k(self):
        with patch.dict(os.environ, {"ROBOTMCP_TOKENIZER": "cl100k_base"}):
            svc = TokenEstimationService()
            assert svc.backend == TokenizerBackend.CL100K_BASE


class TestTokenEstimationServiceUnsupportedBackend:
    """Tests for unsupported backends that fall back to heuristic."""

    def test_llama_falls_back(self):
        svc = TokenEstimationService(backend=TokenizerBackend.LLAMA)
        tc = svc.estimate("a" * 80, "json")
        assert tc.backend_used == TokenizerBackend.HEURISTIC
        assert tc.count == 20

    def test_mistral_falls_back(self):
        svc = TokenEstimationService(backend=TokenizerBackend.MISTRAL)
        tc = svc.estimate("a" * 80, "text")
        assert tc.backend_used == TokenizerBackend.HEURISTIC
        assert tc.count == 20


# ── TokenReportingService ────────────────────────────────────────────


class TestTokenReportingService:
    """Tests for TokenReportingService."""

    def _make_service(self):
        estimator = TokenEstimationService(backend=TokenizerBackend.HEURISTIC)
        return TokenReportingService(estimator)

    def test_measure_tool_basic(self):
        svc = self._make_service()
        report = svc.measure_tool(
            "execute_step",
            "A description of the tool",
            '{"type": "object"}',
        )
        assert report.tool_name == "execute_step"
        assert report.description_tokens == len("A description of the tool") // 4
        assert report.schema_tokens == len('{"type": "object"}') // 4
        assert report.total == report.description_tokens + report.schema_tokens

    def test_measure_tool_empty_description(self):
        svc = self._make_service()
        report = svc.measure_tool("tool", "", '{"type": "object"}')
        assert report.description_tokens == 0
        assert report.total == report.schema_tokens

    def test_measure_tool_empty_schema(self):
        svc = self._make_service()
        report = svc.measure_tool("tool", "description", "")
        assert report.schema_tokens == 0
        assert report.total == report.description_tokens

    def test_measure_tool_both_empty(self):
        svc = self._make_service()
        report = svc.measure_tool("tool", "", "")
        assert report.total == 0

    def test_measure_tool_returns_token_report(self):
        svc = self._make_service()
        report = svc.measure_tool("x", "desc", "{}")
        assert isinstance(report, TokenReport)

    def test_measure_profile_basic(self):
        svc = self._make_service()
        tool_data = {
            "tool_a": {"description": "desc A", "schema": '{"a": 1}'},
            "tool_b": {"description": "desc B", "schema": '{"b": 2}'},
        }
        summary = svc.measure_profile("full", "verbose", tool_data)
        assert summary.profile_name == "full"
        assert summary.description_mode == "verbose"
        assert len(summary.tool_reports) == 2
        assert summary.total_schema_tokens == sum(r.total for r in summary.tool_reports)
        assert summary.backend_used == TokenizerBackend.HEURISTIC

    def test_measure_profile_sorted_tools(self):
        svc = self._make_service()
        tool_data = {
            "z_tool": {"description": "z", "schema": "{}"},
            "a_tool": {"description": "a", "schema": "{}"},
            "m_tool": {"description": "m", "schema": "{}"},
        }
        summary = svc.measure_profile("p", "mode", tool_data)
        names = [r.tool_name for r in summary.tool_reports]
        assert names == ["a_tool", "m_tool", "z_tool"]

    def test_measure_profile_empty_tools(self):
        svc = self._make_service()
        summary = svc.measure_profile("empty", "none", {})
        assert summary.total_schema_tokens == 0
        assert len(summary.tool_reports) == 0

    def test_measure_profile_missing_description_key(self):
        svc = self._make_service()
        tool_data = {"tool_a": {"schema": '{"x": 1}'}}
        summary = svc.measure_profile("p", "mode", tool_data)
        # description defaults to ""
        assert summary.tool_reports[0].description_tokens == 0

    def test_measure_profile_missing_schema_key(self):
        svc = self._make_service()
        tool_data = {"tool_a": {"description": "desc"}}
        summary = svc.measure_profile("p", "mode", tool_data)
        # schema defaults to "{}"
        assert summary.tool_reports[0].schema_tokens == len("{}") // 4

    def test_measure_profile_returns_summary_type(self):
        svc = self._make_service()
        summary = svc.measure_profile("p", "mode", {})
        assert isinstance(summary, ProfileTokenSummary)

    def test_measure_profile_single_tool(self):
        svc = self._make_service()
        tool_data = {"only_tool": {"description": "d" * 100, "schema": "s" * 200}}
        summary = svc.measure_profile("single", "full", tool_data)
        assert len(summary.tool_reports) == 1
        assert summary.total_schema_tokens == summary.tool_reports[0].total

    def test_measure_profile_large_tool_set(self):
        svc = self._make_service()
        tool_data = {
            f"tool_{i}": {"description": f"desc {i}", "schema": f'{{"id": {i}}}'}
            for i in range(50)
        }
        summary = svc.measure_profile("large", "full", tool_data)
        assert len(summary.tool_reports) == 50

    def test_measure_tool_long_description(self):
        svc = self._make_service()
        desc = "x" * 10000
        report = svc.measure_tool("tool", desc, "{}")
        assert report.description_tokens == 2500  # 10000 // 4

    def test_measure_tool_long_schema(self):
        svc = self._make_service()
        schema = "y" * 8000
        report = svc.measure_tool("tool", "", schema)
        assert report.schema_tokens == 2000  # 8000 // 4


# ── TokenRegressionService ───────────────────────────────────────────


class TestTokenRegressionService:
    """Tests for TokenRegressionService."""

    def _make_summary(self, total: int, profile: str = "test") -> ProfileTokenSummary:
        """Helper to create a ProfileTokenSummary with a given total."""
        if total == 0:
            reports = ()
        else:
            reports = (
                TokenReport(
                    tool_name="tool_a",
                    description_tokens=total // 2,
                    schema_tokens=total - total // 2,
                    total=total,
                ),
            )
        return ProfileTokenSummary(
            profile_name=profile,
            description_mode="full",
            total_schema_tokens=total,
            tool_reports=reports,
            backend_used=TokenizerBackend.HEURISTIC,
        )

    def test_no_baseline_passes(self):
        svc = TokenRegressionService(threshold=50)
        result = svc.check_regression("test", self._make_summary(200), baseline=None)
        assert result.passed is True
        assert result.baseline_tokens == 0
        assert result.delta == 200

    def test_zero_baseline_passes(self):
        svc = TokenRegressionService(threshold=50)
        result = svc.check_regression("test", self._make_summary(200), baseline=0)
        assert result.passed is True

    def test_within_threshold_passes(self):
        svc = TokenRegressionService(threshold=50, threshold_percent=5.0)
        result = svc.check_regression("test", self._make_summary(1020), baseline=1000)
        # delta=20, delta_pct=2.0% - both below threshold
        assert result.passed is True
        assert result.delta == 20

    def test_abs_exceeded_pct_not_passes(self):
        """Only absolute threshold exceeded -> still passes."""
        svc = TokenRegressionService(threshold=50, threshold_percent=50.0)
        result = svc.check_regression("test", self._make_summary(1060), baseline=1000)
        # delta=60 > 50 (abs exceeded), but 6% < 50% (pct not exceeded)
        assert result.passed is True

    def test_pct_exceeded_abs_not_passes(self):
        """Only percentage threshold exceeded -> still passes."""
        svc = TokenRegressionService(threshold=100, threshold_percent=5.0)
        result = svc.check_regression("test", self._make_summary(1060), baseline=1000)
        # delta=60 < 100 (abs not exceeded), but 6% > 5% (pct exceeded)
        assert result.passed is True

    def test_both_exceeded_fails(self):
        """Both thresholds exceeded -> regression detected."""
        svc = TokenRegressionService(threshold=50, threshold_percent=5.0)
        result = svc.check_regression("test", self._make_summary(1100), baseline=1000)
        # delta=100 > 50 (abs), 10% > 5% (pct)
        assert result.passed is False
        assert result.delta == 100
        assert result.delta_percent == pytest.approx(10.0)

    def test_exact_threshold_not_exceeded(self):
        """delta == threshold is NOT exceeded (need >)."""
        svc = TokenRegressionService(threshold=50, threshold_percent=5.0)
        result = svc.check_regression("test", self._make_summary(1050), baseline=1000)
        # delta=50 == threshold (not >), so abs not exceeded
        assert result.passed is True

    def test_exact_pct_threshold_not_exceeded(self):
        svc = TokenRegressionService(threshold=1, threshold_percent=5.0)
        result = svc.check_regression("test", self._make_summary(1050), baseline=1000)
        # delta=50 > 1 (abs exceeded), pct=5.0% == 5.0% (not >)
        assert result.passed is True

    def test_negative_delta_passes(self):
        """Token reduction always passes."""
        svc = TokenRegressionService(threshold=50, threshold_percent=5.0)
        result = svc.check_regression("test", self._make_summary(900), baseline=1000)
        assert result.passed is True
        assert result.delta == -100

    def test_zero_delta_passes(self):
        svc = TokenRegressionService(threshold=50)
        result = svc.check_regression("test", self._make_summary(1000), baseline=1000)
        assert result.passed is True
        assert result.delta == 0
        assert result.delta_percent == 0.0

    def test_event_published_on_failure(self):
        events = []
        svc = TokenRegressionService(
            threshold=10, threshold_percent=1.0, event_publisher=events.append
        )
        svc.check_regression("test", self._make_summary(1200), baseline=1000)
        assert len(events) == 1
        e = events[0]
        assert e.profile == "test"
        assert e.delta == 200
        assert e.threshold == 10

    def test_no_event_on_pass(self):
        events = []
        svc = TokenRegressionService(
            threshold=50, threshold_percent=5.0, event_publisher=events.append
        )
        svc.check_regression("test", self._make_summary(1020), baseline=1000)
        assert len(events) == 0

    def test_event_type(self):
        from robotmcp.domains.token_accounting.events import TokenRegressionDetected

        events = []
        svc = TokenRegressionService(
            threshold=10, threshold_percent=1.0, event_publisher=events.append
        )
        svc.check_regression("test", self._make_summary(1200), baseline=1000)
        assert isinstance(events[0], TokenRegressionDetected)

    def test_event_backend_value(self):
        events = []
        svc = TokenRegressionService(
            threshold=10, threshold_percent=1.0, event_publisher=events.append
        )
        svc.check_regression("test", self._make_summary(1200), baseline=1000)
        assert events[0].backend_used == "heuristic"

    def test_result_profile_name(self):
        svc = TokenRegressionService(threshold=50)
        result = svc.check_regression("my_profile", self._make_summary(100), baseline=None)
        assert result.profile == "my_profile"

    def test_result_threshold_in_result(self):
        svc = TokenRegressionService(threshold=42)
        result = svc.check_regression("test", self._make_summary(100), baseline=None)
        assert result.threshold == 42

    def test_large_regression(self):
        svc = TokenRegressionService(threshold=50, threshold_percent=5.0)
        result = svc.check_regression("test", self._make_summary(5000), baseline=1000)
        assert result.passed is False
        assert result.delta == 4000
        assert result.delta_percent == pytest.approx(400.0)

    def test_no_event_publisher(self):
        """No publisher set, no error on failure."""
        svc = TokenRegressionService(threshold=10, threshold_percent=1.0, event_publisher=None)
        result = svc.check_regression("test", self._make_summary(1200), baseline=1000)
        assert result.passed is False  # Still detects regression

    def test_custom_thresholds(self):
        svc = TokenRegressionService(threshold=200, threshold_percent=20.0)
        # delta=150, 15% - both below custom thresholds
        result = svc.check_regression("test", self._make_summary(1150), baseline=1000)
        assert result.passed is True

    def test_very_small_baseline(self):
        svc = TokenRegressionService(threshold=50, threshold_percent=5.0)
        result = svc.check_regression("test", self._make_summary(200), baseline=1)
        # delta=199 > 50, delta_pct=19900% > 5%
        assert result.passed is False


# ── get_estimation_service singleton ──────────────────────────────────


class TestGetEstimationService:
    """Tests for module-level singleton."""

    def test_returns_service(self):
        import robotmcp.domains.token_accounting.services as svc_mod

        svc_mod._estimation_service = None
        svc = get_estimation_service()
        assert isinstance(svc, TokenEstimationService)

    def test_singleton(self):
        import robotmcp.domains.token_accounting.services as svc_mod

        svc_mod._estimation_service = None
        svc1 = get_estimation_service()
        svc2 = get_estimation_service()
        assert svc1 is svc2

    def test_reset_singleton(self):
        import robotmcp.domains.token_accounting.services as svc_mod

        svc_mod._estimation_service = None
        svc1 = get_estimation_service()
        svc_mod._estimation_service = None
        svc2 = get_estimation_service()
        assert svc1 is not svc2
