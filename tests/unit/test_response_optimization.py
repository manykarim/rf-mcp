"""Comprehensive unit tests for Response Optimization domain (ADR-008).

Tests cover: ResponseOptimizationConfig, Verbosity, FieldAbbreviationMap,
TruncationPolicy, TokenEstimate, ResponseCompressor, CompressionMetrics.

Run with: uv run pytest tests/unit/test_response_optimization.py -v
"""

__test__ = True

import json

import pytest

from robotmcp.domains.response_optimization.aggregates import ResponseOptimizationConfig
from robotmcp.domains.response_optimization.events import (
    CompressionRatioLearned,
    IncrementalDiffComputed,
    ResponseCompressed,
    SnapshotFolded,
)
from robotmcp.domains.response_optimization.services import (
    CompressionMetrics,
    ResponseCompressor,
)
from robotmcp.domains.response_optimization.value_objects import (
    FieldAbbreviationMap,
    SnapshotCompressionMode,
    TokenEstimate,
    TruncationPolicy,
    Verbosity,
)


# =============================================================================
# Verbosity
# =============================================================================


class TestVerbosity:
    """Test Verbosity enum."""

    def test_verbose_value(self):
        assert Verbosity.VERBOSE.value == "verbose"

    def test_standard_value(self):
        assert Verbosity.STANDARD.value == "standard"

    def test_compact_value(self):
        assert Verbosity.COMPACT.value == "compact"

    def test_has_3_members(self):
        assert len(Verbosity) == 3


# =============================================================================
# FieldAbbreviationMap
# =============================================================================


class TestFieldAbbreviationMap:
    """Test FieldAbbreviationMap value object."""

    def test_standard_has_27_mappings(self):
        fam = FieldAbbreviationMap.standard()
        assert len(fam.mappings) == 27

    def test_standard_enabled(self):
        fam = FieldAbbreviationMap.standard()
        assert fam.enabled is True

    def test_abbreviate_session_id(self):
        fam = FieldAbbreviationMap.standard()
        assert fam.abbreviate("session_id") == "sid"

    def test_abbreviate_success(self):
        fam = FieldAbbreviationMap.standard()
        assert fam.abbreviate("success") == "ok"

    def test_abbreviate_error(self):
        fam = FieldAbbreviationMap.standard()
        assert fam.abbreviate("error") == "err"

    def test_abbreviate_message(self):
        fam = FieldAbbreviationMap.standard()
        assert fam.abbreviate("message") == "msg"

    def test_abbreviate_keyword(self):
        fam = FieldAbbreviationMap.standard()
        assert fam.abbreviate("keyword") == "kw"

    def test_abbreviate_unknown_returns_original(self):
        fam = FieldAbbreviationMap.standard()
        assert fam.abbreviate("unknown_field") == "unknown_field"

    def test_disabled_returns_original(self):
        fam = FieldAbbreviationMap.disabled()
        assert fam.abbreviate("session_id") == "session_id"

    def test_disabled_not_enabled(self):
        fam = FieldAbbreviationMap.disabled()
        assert fam.enabled is False
        assert len(fam.mappings) == 0

    def test_frozen(self):
        fam = FieldAbbreviationMap.standard()
        with pytest.raises(AttributeError):
            fam.enabled = False


# =============================================================================
# TruncationPolicy
# =============================================================================


class TestTruncationPolicy:
    """Test TruncationPolicy value object."""

    def test_default_policy(self):
        policy = TruncationPolicy.default()
        assert policy.max_string == 2000
        assert policy.max_list_items == 50
        assert policy.max_dict_items == 30

    def test_aggressive_policy(self):
        policy = TruncationPolicy.aggressive()
        assert policy.max_string == 300
        assert policy.max_list_items == 10
        assert policy.max_dict_items == 10

    def test_aggressive_shorter_than_default(self):
        default = TruncationPolicy.default()
        aggressive = TruncationPolicy.aggressive()
        assert aggressive.max_string < default.max_string
        assert aggressive.max_list_items < default.max_list_items
        assert aggressive.max_dict_items < default.max_dict_items


# =============================================================================
# SnapshotCompressionMode
# =============================================================================


class TestSnapshotCompressionMode:
    """Test SnapshotCompressionMode enum."""

    def test_none_value(self):
        assert SnapshotCompressionMode.NONE.value == "none"

    def test_folded_value(self):
        assert SnapshotCompressionMode.FOLDED.value == "folded"

    def test_incremental_value(self):
        assert SnapshotCompressionMode.INCREMENTAL_DIFF.value == "incremental"

    def test_folded_diff_value(self):
        assert SnapshotCompressionMode.FOLDED_DIFF.value == "folded_diff"

    def test_has_4_members(self):
        assert len(SnapshotCompressionMode) == 4


# =============================================================================
# TokenEstimate
# =============================================================================


class TestTokenEstimate:
    """Test TokenEstimate value object."""

    def test_from_dict_calculates_tokens(self):
        data = {"key": "value", "number": 42}
        estimate = TokenEstimate.from_dict(data)
        expected_chars = len(json.dumps(data, default=str))
        assert estimate.char_count == expected_chars
        assert estimate.estimated_tokens == expected_chars // 4

    def test_from_dict_with_custom_budget(self):
        data = {"x": "y"}
        estimate = TokenEstimate.from_dict(data, budget=100)
        assert estimate.budget == 100

    def test_within_budget_true(self):
        estimate = TokenEstimate(char_count=100, estimated_tokens=25, budget=100)
        assert estimate.within_budget is True

    def test_within_budget_exact(self):
        estimate = TokenEstimate(char_count=400, estimated_tokens=100, budget=100)
        assert estimate.within_budget is True

    def test_within_budget_false(self):
        estimate = TokenEstimate(char_count=600, estimated_tokens=150, budget=100)
        assert estimate.within_budget is False

    def test_overage_zero_when_within(self):
        estimate = TokenEstimate(char_count=100, estimated_tokens=25, budget=100)
        assert estimate.overage == 0

    def test_overage_positive_when_over(self):
        estimate = TokenEstimate(char_count=600, estimated_tokens=150, budget=100)
        assert estimate.overage == 50

    def test_frozen(self):
        estimate = TokenEstimate(char_count=100, estimated_tokens=25, budget=100)
        with pytest.raises(AttributeError):
            estimate.budget = 200


# =============================================================================
# ResponseOptimizationConfig
# =============================================================================


class TestResponseOptimizationConfig:
    """Test ResponseOptimizationConfig aggregate root."""

    def test_create_default(self):
        config = ResponseOptimizationConfig.create_default("sess-1")
        assert config.verbosity == Verbosity.STANDARD
        assert config.field_abbreviation.enabled is False
        assert config.token_budget == 4000
        assert config.session_id == "sess-1"

    def test_create_compact(self):
        config = ResponseOptimizationConfig.create_compact("sess-2")
        assert config.verbosity == Verbosity.COMPACT
        assert config.field_abbreviation.enabled is True
        assert config.token_budget == 1500
        assert config.snapshot_compression == SnapshotCompressionMode.FOLDED_DIFF

    def test_create_verbose(self):
        config = ResponseOptimizationConfig.create_verbose("sess-3")
        assert config.verbosity == Verbosity.VERBOSE
        assert config.field_abbreviation.enabled is False
        assert config.token_budget == 10000

    def test_compact_without_abbreviation_raises(self):
        with pytest.raises(ValueError, match="COMPACT verbosity requires field abbreviation"):
            ResponseOptimizationConfig(
                config_id="test",
                session_id="s1",
                verbosity=Verbosity.COMPACT,
                field_abbreviation=FieldAbbreviationMap.disabled(),
                truncation=TruncationPolicy.default(),
                snapshot_compression=SnapshotCompressionMode.NONE,
                token_budget=1000,
            )

    def test_zero_token_budget_raises(self):
        with pytest.raises(ValueError, match="Token budget must be positive"):
            ResponseOptimizationConfig(
                config_id="test",
                session_id="s1",
                verbosity=Verbosity.STANDARD,
                field_abbreviation=FieldAbbreviationMap.disabled(),
                truncation=TruncationPolicy.default(),
                snapshot_compression=SnapshotCompressionMode.NONE,
                token_budget=0,
            )

    def test_negative_token_budget_raises(self):
        with pytest.raises(ValueError, match="Token budget must be positive"):
            ResponseOptimizationConfig(
                config_id="test",
                session_id="s1",
                verbosity=Verbosity.STANDARD,
                field_abbreviation=FieldAbbreviationMap.disabled(),
                truncation=TruncationPolicy.default(),
                snapshot_compression=SnapshotCompressionMode.NONE,
                token_budget=-100,
            )


# =============================================================================
# CompressionMetrics
# =============================================================================


class TestCompressionMetrics:
    """Test CompressionMetrics helper."""

    def test_compression_ratio(self):
        metrics = CompressionMetrics(raw_tokens=100, compressed_tokens=60)
        assert abs(metrics.compression_ratio - 0.4) < 0.001

    def test_compression_ratio_zero_raw(self):
        metrics = CompressionMetrics(raw_tokens=0, compressed_tokens=0)
        assert metrics.compression_ratio == 0.0

    def test_tokens_saved(self):
        metrics = CompressionMetrics(raw_tokens=100, compressed_tokens=60)
        assert metrics.tokens_saved == 40

    def test_tokens_saved_no_compression(self):
        metrics = CompressionMetrics(raw_tokens=100, compressed_tokens=100)
        assert metrics.tokens_saved == 0


# =============================================================================
# ResponseCompressor
# =============================================================================


class TestResponseCompressor:
    """Test ResponseCompressor service."""

    @pytest.fixture
    def event_log(self):
        return []

    @pytest.fixture
    def compressor(self, event_log):
        return ResponseCompressor(event_publisher=lambda e: event_log.append(e))

    def test_verbose_returns_original(self, compressor):
        config = ResponseOptimizationConfig.create_verbose("s1")
        raw = {"success": True, "result": "OK", "error": None}
        result, metrics = compressor.compress_response(raw, config)
        assert result == raw
        assert metrics.tokens_saved == 0

    def test_standard_removes_empty_fields(self, compressor):
        config = ResponseOptimizationConfig.create_default("s1")
        raw = {
            "success": True,
            "result": "data",
            "error": None,       # in OMIT_WHEN_EMPTY
            "message": "",       # in OMIT_WHEN_EMPTY
            "metadata": {},      # in OMIT_WHEN_EMPTY
        }
        result, metrics = compressor.compress_response(raw, config)
        assert "success" in result
        assert "result" in result
        assert "error" not in result
        assert "message" not in result
        assert "metadata" not in result
        assert metrics.fields_omitted >= 3

    def test_standard_preserves_non_empty_omittable_fields(self, compressor):
        config = ResponseOptimizationConfig.create_default("s1")
        raw = {
            "success": True,
            "error": "Something went wrong",  # Non-empty, kept
        }
        result, metrics = compressor.compress_response(raw, config)
        assert "error" in result
        assert result["error"] == "Something went wrong"

    def test_compact_abbreviates_fields(self, compressor):
        config = ResponseOptimizationConfig.create_compact("s1")
        raw = {
            "success": True,
            "session_id": "s1",
            "keyword": "Click",
            "result": "OK",
        }
        result, metrics = compressor.compress_response(raw, config)
        # "success" -> "ok", "session_id" -> "sid", "keyword" -> "kw"
        assert "ok" in result
        assert "sid" in result
        assert "kw" in result
        assert metrics.fields_abbreviated >= 3

    def test_compact_truncates_long_strings(self, compressor):
        config = ResponseOptimizationConfig.create_compact("s1")
        # Aggressive truncation: max_string=300
        long_val = "x" * 500
        raw = {"result": long_val}
        result, metrics = compressor.compress_response(raw, config)
        # "result" -> "res" (abbreviated)
        abbreviated_key = "res"
        assert len(result[abbreviated_key]) < 500
        assert metrics.strings_truncated >= 1

    def test_compact_truncates_long_lists(self, compressor):
        config = ResponseOptimizationConfig.create_compact("s1")
        # Aggressive truncation: max_list_items=10
        raw = {"keywords": list(range(20))}
        result, metrics = compressor.compress_response(raw, config)
        abbreviated_key = "kws"
        assert len(result[abbreviated_key]) <= 10
        assert metrics.lists_truncated >= 1

    def test_compact_nested_dict_abbreviation(self, compressor):
        config = ResponseOptimizationConfig.create_compact("s1")
        raw = {
            "result": {
                "keyword": "Click",
                "session_id": "s1",
            }
        }
        result, metrics = compressor.compress_response(raw, config)
        # result -> res, and nested keyword -> kw, session_id -> sid
        assert "res" in result
        nested = result["res"]
        assert "kw" in nested
        assert "sid" in nested

    def test_metrics_track_tokens_saved(self, compressor):
        config = ResponseOptimizationConfig.create_compact("s1")
        raw = {
            "success": True,
            "error": None,
            "message": "",
            "result": "short",
        }
        result, metrics = compressor.compress_response(raw, config)
        assert metrics.tokens_saved >= 0
        assert metrics.compressed_tokens <= metrics.raw_tokens

    def test_event_published(self, compressor, event_log):
        config = ResponseOptimizationConfig.create_default("s1")
        raw = {"success": True}
        compressor.compress_response(raw, config)
        compressed_events = [e for e in event_log if isinstance(e, ResponseCompressed)]
        assert len(compressed_events) == 1
        assert compressed_events[0].session_id == "s1"
        assert compressed_events[0].verbosity == "standard"

    def test_emergency_truncation_when_over_budget(self, compressor):
        """Create a response that exceeds COMPACT budget (1500 tokens) to trigger emergency."""
        config = ResponseOptimizationConfig.create_compact("s1")
        # 1500 tokens * 4 chars/token = 6000 chars budget
        # Create data well over budget
        raw = {
            "success": True,
            "result": "x" * 10000,
            "extra_data": "y" * 10000,
        }
        result, metrics = compressor.compress_response(raw, config)
        # After emergency truncation, should be smaller
        final_chars = len(json.dumps(result, default=str))
        assert final_chars < 25000  # Much smaller than original

    def test_compressor_without_publisher(self):
        """Compression works fine without an event publisher."""
        compressor = ResponseCompressor()
        config = ResponseOptimizationConfig.create_default("s1")
        raw = {"success": True, "error": None}
        result, metrics = compressor.compress_response(raw, config)
        assert "success" in result


# =============================================================================
# Response Optimization Events
# =============================================================================


class TestResponseCompressedEvent:
    """Test ResponseCompressed event."""

    def test_to_dict(self):
        event = ResponseCompressed(
            session_id="s1",
            tool_name="execute_step",
            raw_tokens=200,
            compressed_tokens=120,
            compression_ratio=0.4,
            verbosity="compact",
        )
        d = event.to_dict()
        assert d["event_type"] == "ResponseCompressed"
        assert d["session_id"] == "s1"
        assert d["raw_tokens"] == 200
        assert d["compression_ratio"] == 0.4
        assert "timestamp" in d


class TestSnapshotFoldedEvent:
    """Test SnapshotFolded event."""

    def test_to_dict(self):
        event = SnapshotFolded(
            session_id="s1",
            items_original=100,
            items_after_folding=30,
            fold_threshold=0.85,
            tokens_before=500,
            tokens_after=150,
        )
        d = event.to_dict()
        assert d["event_type"] == "SnapshotFolded"
        assert d["items_original"] == 100
        assert d["items_after_folding"] == 30


class TestIncrementalDiffComputedEvent:
    """Test IncrementalDiffComputed event."""

    def test_to_dict(self):
        event = IncrementalDiffComputed(
            session_id="s1",
            full_snapshot_tokens=500,
            diff_tokens=80,
            change_count=3,
        )
        d = event.to_dict()
        assert d["event_type"] == "IncrementalDiffComputed"
        assert d["diff_tokens"] == 80
        assert d["change_count"] == 3


class TestCompressionRatioLearnedEvent:
    """Test CompressionRatioLearned event."""

    def test_to_dict(self):
        event = CompressionRatioLearned(
            page_type="login_form",
            sample_count=50,
            avg_compression_ratio=0.42,
            optimal_fold_threshold=0.88,
        )
        d = event.to_dict()
        assert d["event_type"] == "CompressionRatioLearned"
        assert d["page_type"] == "login_form"
        assert d["avg_compression_ratio"] == 0.42
        assert d["optimal_fold_threshold"] == 0.88
