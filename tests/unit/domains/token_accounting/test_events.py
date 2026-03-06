"""Tests for Token Accounting Events (ADR-017)."""

from __future__ import annotations

from datetime import datetime

import pytest

from robotmcp.domains.token_accounting.events import (
    TokenBudgetExceeded,
    TokenMeasurementRecorded,
    TokenRegressionDetected,
)


class TestTokenBudgetExceeded:
    """Tests for TokenBudgetExceeded event."""

    def test_basic_creation(self):
        e = TokenBudgetExceeded(
            tool_name="execute_step",
            token_count=1200,
            budget=1000,
            overage=200,
        )
        assert e.tool_name == "execute_step"
        assert e.token_count == 1200
        assert e.budget == 1000
        assert e.overage == 200

    def test_timestamp_auto_set(self):
        before = datetime.now()
        e = TokenBudgetExceeded(
            tool_name="x", token_count=100, budget=50, overage=50
        )
        after = datetime.now()
        assert before <= e.timestamp <= after

    def test_frozen(self):
        e = TokenBudgetExceeded(
            tool_name="x", token_count=100, budget=50, overage=50
        )
        with pytest.raises(AttributeError):
            e.overage = 999  # type: ignore[misc]

    def test_to_dict(self):
        e = TokenBudgetExceeded(
            tool_name="execute_step",
            token_count=1200,
            budget=1000,
            overage=200,
        )
        d = e.to_dict()
        assert d["event_type"] == "TokenBudgetExceeded"
        assert d["tool"] == "execute_step"
        assert d["count"] == 1200
        assert d["budget"] == 1000
        assert d["overage"] == 200
        assert "ts" in d

    def test_to_dict_ts_is_iso(self):
        e = TokenBudgetExceeded(
            tool_name="x", token_count=100, budget=50, overage=50
        )
        datetime.fromisoformat(e.to_dict()["ts"])


class TestTokenRegressionDetected:
    """Tests for TokenRegressionDetected event."""

    def test_basic_creation(self):
        e = TokenRegressionDetected(
            profile="full",
            baseline_tokens=500,
            current_tokens=600,
            delta=100,
            threshold=50,
            backend_used="heuristic",
        )
        assert e.profile == "full"
        assert e.delta == 100

    def test_timestamp_auto_set(self):
        before = datetime.now()
        e = TokenRegressionDetected(
            profile="x",
            baseline_tokens=100,
            current_tokens=200,
            delta=100,
            threshold=50,
            backend_used="heuristic",
        )
        after = datetime.now()
        assert before <= e.timestamp <= after

    def test_frozen(self):
        e = TokenRegressionDetected(
            profile="x",
            baseline_tokens=100,
            current_tokens=200,
            delta=100,
            threshold=50,
            backend_used="heuristic",
        )
        with pytest.raises(AttributeError):
            e.delta = 0  # type: ignore[misc]

    def test_to_dict(self):
        e = TokenRegressionDetected(
            profile="full",
            baseline_tokens=500,
            current_tokens=600,
            delta=100,
            threshold=50,
            backend_used="cl100k_base",
        )
        d = e.to_dict()
        assert d["event_type"] == "TokenRegressionDetected"
        assert d["profile"] == "full"
        assert d["baseline"] == 500
        assert d["current"] == 600
        assert d["delta"] == 100
        assert d["threshold"] == 50
        assert d["backend"] == "cl100k_base"
        assert "ts" in d

    def test_to_dict_ts_is_iso(self):
        e = TokenRegressionDetected(
            profile="x",
            baseline_tokens=100,
            current_tokens=200,
            delta=100,
            threshold=50,
            backend_used="heuristic",
        )
        datetime.fromisoformat(e.to_dict()["ts"])


class TestTokenMeasurementRecorded:
    """Tests for TokenMeasurementRecorded event."""

    def test_basic_creation(self):
        e = TokenMeasurementRecorded(
            tool_name="execute_step",
            measurement_type="schema",
            count=150,
            backend="heuristic",
        )
        assert e.tool_name == "execute_step"
        assert e.measurement_type == "schema"
        assert e.count == 150

    def test_timestamp_auto_set(self):
        before = datetime.now()
        e = TokenMeasurementRecorded(
            tool_name="x",
            measurement_type="total",
            count=0,
            backend="heuristic",
        )
        after = datetime.now()
        assert before <= e.timestamp <= after

    def test_frozen(self):
        e = TokenMeasurementRecorded(
            tool_name="x",
            measurement_type="total",
            count=0,
            backend="heuristic",
        )
        with pytest.raises(AttributeError):
            e.count = 100  # type: ignore[misc]

    def test_to_dict(self):
        e = TokenMeasurementRecorded(
            tool_name="execute_step",
            measurement_type="schema",
            count=150,
            backend="cl100k_base",
        )
        d = e.to_dict()
        assert d["event_type"] == "TokenMeasurementRecorded"
        assert d["tool"] == "execute_step"
        assert d["type"] == "schema"
        assert d["count"] == 150
        assert d["backend"] == "cl100k_base"
        assert "ts" in d

    def test_to_dict_ts_is_iso(self):
        e = TokenMeasurementRecorded(
            tool_name="x",
            measurement_type="total",
            count=0,
            backend="heuristic",
        )
        datetime.fromisoformat(e.to_dict()["ts"])
