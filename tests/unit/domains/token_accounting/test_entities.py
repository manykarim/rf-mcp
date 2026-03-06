"""Tests for Token Accounting Entities (ADR-017)."""

from __future__ import annotations

from datetime import datetime

import pytest

from robotmcp.domains.token_accounting.entities import TokenMeasurement
from robotmcp.domains.token_accounting.value_objects import TokenizerBackend


class TestTokenMeasurement:
    """Tests for TokenMeasurement entity."""

    def test_create_factory(self):
        m = TokenMeasurement.create(
            tool_name="execute_step",
            profile="full",
            backend=TokenizerBackend.HEURISTIC,
            measurement_type="schema",
            count=150,
        )
        assert m.tool_name == "execute_step"
        assert m.profile == "full"
        assert m.backend == TokenizerBackend.HEURISTIC
        assert m.measurement_type == "schema"
        assert m.count == 150
        assert m.id.startswith("tm_")
        assert len(m.id) == 11  # "tm_" + 8 hex chars

    def test_create_unique_ids(self):
        m1 = TokenMeasurement.create("a", "full", TokenizerBackend.HEURISTIC, "schema", 10)
        m2 = TokenMeasurement.create("a", "full", TokenizerBackend.HEURISTIC, "schema", 10)
        assert m1.id != m2.id

    def test_create_sets_timestamp(self):
        before = datetime.now()
        m = TokenMeasurement.create("a", "full", TokenizerBackend.HEURISTIC, "schema", 10)
        after = datetime.now()
        assert before <= m.timestamp <= after

    def test_valid_measurement_types(self):
        for t in ("schema", "description", "response", "total"):
            m = TokenMeasurement.create("x", "p", TokenizerBackend.HEURISTIC, t, 0)
            assert m.measurement_type == t

    def test_invalid_measurement_type_raises(self):
        with pytest.raises(ValueError, match="Invalid measurement_type"):
            TokenMeasurement.create("x", "p", TokenizerBackend.HEURISTIC, "invalid", 0)

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match="negative"):
            TokenMeasurement.create("x", "p", TokenizerBackend.HEURISTIC, "schema", -1)

    def test_zero_count(self):
        m = TokenMeasurement.create("x", "p", TokenizerBackend.HEURISTIC, "schema", 0)
        assert m.count == 0

    def test_mutable_count(self):
        """Entities are mutable (not frozen dataclasses)."""
        m = TokenMeasurement.create("x", "p", TokenizerBackend.HEURISTIC, "schema", 10)
        m.count = 20
        assert m.count == 20

    def test_to_dict(self):
        m = TokenMeasurement.create(
            tool_name="execute_step",
            profile="full",
            backend=TokenizerBackend.CL100K_BASE,
            measurement_type="total",
            count=200,
        )
        d = m.to_dict()
        assert d["tool"] == "execute_step"
        assert d["profile"] == "full"
        assert d["backend"] == "cl100k_base"
        assert d["type"] == "total"
        assert d["count"] == 200
        assert "id" in d
        assert "ts" in d

    def test_to_dict_id_format(self):
        m = TokenMeasurement.create("x", "p", TokenizerBackend.HEURISTIC, "schema", 0)
        d = m.to_dict()
        assert d["id"].startswith("tm_")

    def test_to_dict_ts_is_iso(self):
        m = TokenMeasurement.create("x", "p", TokenizerBackend.HEURISTIC, "schema", 0)
        d = m.to_dict()
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(d["ts"])

    def test_all_backends(self):
        for b in TokenizerBackend:
            m = TokenMeasurement.create("x", "p", b, "schema", 10)
            assert m.backend == b

    def test_direct_construction(self):
        m = TokenMeasurement(
            id="tm_custom01",
            tool_name="my_tool",
            profile="compact",
            backend=TokenizerBackend.HEURISTIC,
            measurement_type="description",
            count=42,
        )
        assert m.id == "tm_custom01"
        assert m.count == 42

    def test_direct_construction_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid measurement_type"):
            TokenMeasurement(
                id="tm_bad",
                tool_name="x",
                profile="p",
                backend=TokenizerBackend.HEURISTIC,
                measurement_type="bad",
                count=0,
            )

    def test_valid_types_is_frozenset(self):
        assert isinstance(TokenMeasurement.VALID_TYPES, frozenset)
        assert TokenMeasurement.VALID_TYPES == frozenset(
            {"schema", "description", "response", "total"}
        )

    def test_large_count(self):
        m = TokenMeasurement.create("x", "p", TokenizerBackend.HEURISTIC, "total", 999_999)
        assert m.count == 999_999

    def test_response_type(self):
        m = TokenMeasurement.create("x", "p", TokenizerBackend.HEURISTIC, "response", 50)
        assert m.measurement_type == "response"
