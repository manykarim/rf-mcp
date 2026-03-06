"""Tests for Token Accounting Aggregates (ADR-017)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from robotmcp.domains.token_accounting.aggregates import TokenAccountant
from robotmcp.domains.token_accounting.entities import TokenMeasurement
from robotmcp.domains.token_accounting.value_objects import TokenizerBackend


class TestTokenAccountant:
    """Tests for TokenAccountant aggregate."""

    def test_create_default_backend(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ROBOTMCP_TOKENIZER", None)
            ta = TokenAccountant.create()
            assert ta.backend == TokenizerBackend.HEURISTIC

    def test_create_with_backend(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.CL100K_BASE)
        assert ta.backend == TokenizerBackend.CL100K_BASE

    def test_create_from_env(self):
        with patch.dict(os.environ, {"ROBOTMCP_TOKENIZER": "o200k_base"}):
            ta = TokenAccountant.create()
            assert ta.backend == TokenizerBackend.O200K_BASE

    def test_create_empty_measurements(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        assert len(ta.measurements) == 0

    def test_record_single(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        m = TokenMeasurement.create("tool_a", "full", TokenizerBackend.HEURISTIC, "schema", 100)
        ta.record(m)
        assert len(ta.measurements) == 1
        assert ta.measurements[0] == m

    def test_record_multiple(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        for i in range(5):
            m = TokenMeasurement.create(f"tool_{i}", "full", TokenizerBackend.HEURISTIC, "schema", i * 10)
            ta.record(m)
        assert len(ta.measurements) == 5

    def test_ring_buffer_maxlen(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        assert ta.measurements.maxlen == 1000

    def test_ring_buffer_eviction(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        # Fill to capacity + 1
        for i in range(1001):
            m = TokenMeasurement.create(f"tool_{i}", "full", TokenizerBackend.HEURISTIC, "schema", i)
            ta.record(m)
        assert len(ta.measurements) == 1000
        # First measurement should have been evicted
        assert ta.measurements[0].tool_name == "tool_1"

    def test_get_measurements_all(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        for i in range(3):
            m = TokenMeasurement.create(f"tool_{i}", "full", TokenizerBackend.HEURISTIC, "schema", i * 10)
            ta.record(m)
        all_m = ta.get_measurements()
        assert len(all_m) == 3

    def test_get_measurements_returns_list(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        result = ta.get_measurements()
        assert isinstance(result, list)

    def test_get_measurements_by_tool_name(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        ta.record(TokenMeasurement.create("tool_a", "full", TokenizerBackend.HEURISTIC, "schema", 10))
        ta.record(TokenMeasurement.create("tool_b", "full", TokenizerBackend.HEURISTIC, "schema", 20))
        ta.record(TokenMeasurement.create("tool_a", "full", TokenizerBackend.HEURISTIC, "description", 30))

        result = ta.get_measurements(tool_name="tool_a")
        assert len(result) == 2
        assert all(m.tool_name == "tool_a" for m in result)

    def test_get_measurements_nonexistent_tool(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        ta.record(TokenMeasurement.create("tool_a", "full", TokenizerBackend.HEURISTIC, "schema", 10))
        result = ta.get_measurements(tool_name="nonexistent")
        assert result == []

    def test_get_measurements_empty(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        assert ta.get_measurements() == []
        assert ta.get_measurements(tool_name="x") == []

    def test_max_measurements_constant(self):
        assert TokenAccountant.MAX_MEASUREMENTS == 1000

    def test_measurements_order_preserved(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        names = ["first", "second", "third"]
        for name in names:
            ta.record(TokenMeasurement.create(name, "full", TokenizerBackend.HEURISTIC, "schema", 10))
        result = ta.get_measurements()
        assert [m.tool_name for m in result] == names

    def test_ring_buffer_oldest_evicted_first(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        # Record exactly maxlen items
        for i in range(1000):
            ta.record(TokenMeasurement.create(f"t{i}", "p", TokenizerBackend.HEURISTIC, "schema", 1))
        assert len(ta.measurements) == 1000
        assert ta.measurements[0].tool_name == "t0"
        # Record one more
        ta.record(TokenMeasurement.create("t1000", "p", TokenizerBackend.HEURISTIC, "schema", 1))
        assert len(ta.measurements) == 1000
        assert ta.measurements[0].tool_name == "t1"
        assert ta.measurements[-1].tool_name == "t1000"

    def test_filter_preserves_order(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        ta.record(TokenMeasurement.create("a", "p", TokenizerBackend.HEURISTIC, "schema", 1))
        ta.record(TokenMeasurement.create("b", "p", TokenizerBackend.HEURISTIC, "schema", 2))
        ta.record(TokenMeasurement.create("a", "p", TokenizerBackend.HEURISTIC, "description", 3))
        result = ta.get_measurements(tool_name="a")
        assert result[0].measurement_type == "schema"
        assert result[1].measurement_type == "description"

    def test_backend_attribute_set(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.MISTRAL)
        assert ta.backend == TokenizerBackend.MISTRAL

    def test_all_backends(self):
        for b in TokenizerBackend:
            ta = TokenAccountant.create(backend=b)
            assert ta.backend == b

    def test_create_env_invalid_falls_back(self):
        with patch.dict(os.environ, {"ROBOTMCP_TOKENIZER": "garbage"}):
            ta = TokenAccountant.create()
            assert ta.backend == TokenizerBackend.HEURISTIC

    def test_measurement_types_in_aggregate(self):
        ta = TokenAccountant.create(backend=TokenizerBackend.HEURISTIC)
        for mtype in ("schema", "description", "response", "total"):
            ta.record(TokenMeasurement.create("x", "p", TokenizerBackend.HEURISTIC, mtype, 10))
        assert len(ta.measurements) == 4
