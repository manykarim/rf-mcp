"""Tests for memory domain value objects.

Covers: MemoryType, MemoryTypeEnum, EmbeddingVector, SimilarityScore,
ConfidenceScore, TimeDecayFactor, MemoryQuery, MemoryEntry, RecallResult,
StorageConfig.
"""

from __future__ import annotations

import math
import os
from unittest.mock import patch

import pytest

from robotmcp.domains.memory.value_objects import (
    ConfidenceScore,
    EmbeddingVector,
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    MemoryTypeEnum,
    RecallResult,
    SimilarityScore,
    StorageConfig,
    TimeDecayFactor,
)


# =========================================================================
# MemoryTypeEnum
# =========================================================================


class TestMemoryTypeEnum:
    def test_all_values(self):
        expected = {
            "working_steps",
            "keywords",
            "documentation",
            "common_errors",
            "domain_knowledge",
            "locators",
        }
        assert {m.value for m in MemoryTypeEnum} == expected

    def test_is_str_enum(self):
        assert isinstance(MemoryTypeEnum.WORKING_STEPS, str)
        assert MemoryTypeEnum.KEYWORDS == "keywords"


# =========================================================================
# MemoryType
# =========================================================================


class TestMemoryType:
    # -- Valid construction --------------------------------------------------

    def test_valid_working_steps(self):
        mt = MemoryType("working_steps")
        assert mt.value == "working_steps"

    def test_valid_keywords(self):
        mt = MemoryType("keywords")
        assert mt.value == "keywords"

    def test_valid_documentation(self):
        mt = MemoryType("documentation")
        assert mt.value == "documentation"

    def test_valid_common_errors(self):
        mt = MemoryType("common_errors")
        assert mt.value == "common_errors"

    def test_valid_domain_knowledge(self):
        mt = MemoryType("domain_knowledge")
        assert mt.value == "domain_knowledge"

    # -- Invalid construction ------------------------------------------------

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid memory type 'garbage'"):
            MemoryType("garbage")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid memory type"):
            MemoryType("")

    def test_case_sensitive(self):
        with pytest.raises(ValueError):
            MemoryType("WORKING_STEPS")

    # -- Frozen --------------------------------------------------------------

    def test_frozen(self):
        mt = MemoryType("keywords")
        with pytest.raises(AttributeError):
            mt.value = "documentation"  # type: ignore[misc]

    # -- Factory methods -----------------------------------------------------

    def test_factory_working_steps(self):
        mt = MemoryType.working_steps()
        assert mt.value == "working_steps"

    def test_factory_keywords(self):
        mt = MemoryType.keywords()
        assert mt.value == "keywords"

    def test_factory_documentation(self):
        mt = MemoryType.documentation()
        assert mt.value == "documentation"

    def test_factory_common_errors(self):
        mt = MemoryType.common_errors()
        assert mt.value == "common_errors"

    def test_factory_domain_knowledge(self):
        mt = MemoryType.domain_knowledge()
        assert mt.value == "domain_knowledge"

    # -- from_string ---------------------------------------------------------

    def test_from_string_lowercase(self):
        mt = MemoryType.from_string("keywords")
        assert mt.value == "keywords"

    def test_from_string_strips_whitespace(self):
        mt = MemoryType.from_string("  keywords  ")
        assert mt.value == "keywords"

    def test_from_string_lowercases(self):
        mt = MemoryType.from_string("KEYWORDS")
        assert mt.value == "keywords"

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            MemoryType.from_string("nonsense")

    # -- all_types -----------------------------------------------------------

    def test_all_types_count(self):
        all_t = MemoryType.all_types()
        assert len(all_t) == 6

    def test_all_types_sorted(self):
        all_t = MemoryType.all_types()
        values = [mt.value for mt in all_t]
        assert values == sorted(values)

    def test_all_types_are_memory_type(self):
        for mt in MemoryType.all_types():
            assert isinstance(mt, MemoryType)

    # -- Properties ----------------------------------------------------------

    def test_collection_name_prefix(self):
        mt = MemoryType("keywords")
        assert mt.collection_name == "rfmcp_keywords"

    def test_collection_name_all_types(self):
        for mt in MemoryType.all_types():
            assert mt.collection_name.startswith("rfmcp_")

    def test_is_executable_working_steps(self):
        assert MemoryType.working_steps().is_executable is True

    def test_is_executable_keywords(self):
        assert MemoryType.keywords().is_executable is True

    def test_is_executable_false_for_docs(self):
        assert MemoryType.documentation().is_executable is False

    def test_is_executable_false_for_errors(self):
        assert MemoryType.common_errors().is_executable is False

    def test_is_executable_false_for_domain(self):
        assert MemoryType.domain_knowledge().is_executable is False

    def test_is_reference_documentation(self):
        assert MemoryType.documentation().is_reference is True

    def test_is_reference_domain_knowledge(self):
        assert MemoryType.domain_knowledge().is_reference is True

    def test_is_reference_false_for_executable(self):
        assert MemoryType.working_steps().is_reference is False
        assert MemoryType.keywords().is_reference is False

    def test_is_reference_false_for_errors(self):
        assert MemoryType.common_errors().is_reference is False

    # -- Equality ------------------------------------------------------------

    def test_equality(self):
        a = MemoryType("keywords")
        b = MemoryType("keywords")
        assert a == b

    def test_inequality(self):
        a = MemoryType("keywords")
        b = MemoryType("documentation")
        assert a != b


# =========================================================================
# EmbeddingVector
# =========================================================================


class TestEmbeddingVector:
    # -- Valid creation ------------------------------------------------------

    def test_create_simple(self):
        ev = EmbeddingVector(values=(1.0, 0.0, 0.0), model_name="test", dimensions=3)
        assert ev.dimensions == 3
        assert ev.model_name == "test"
        assert ev.values == (1.0, 0.0, 0.0)

    def test_from_list(self):
        ev = EmbeddingVector.from_list([1.0, 2.0, 3.0], "test-model")
        assert ev.values == (1.0, 2.0, 3.0)
        assert ev.model_name == "test-model"
        assert ev.dimensions == 3

    def test_to_list(self):
        ev = EmbeddingVector(values=(0.5, 0.3, 0.1), model_name="m", dimensions=3)
        assert ev.to_list() == [0.5, 0.3, 0.1]

    def test_to_list_from_list_roundtrip(self):
        original = [0.1, 0.2, 0.3, 0.4]
        ev = EmbeddingVector.from_list(original, "m")
        assert ev.to_list() == original

    # -- Validation ----------------------------------------------------------

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimensions=2 != len"):
            EmbeddingVector(values=(1.0, 0.0, 0.0), model_name="m", dimensions=2)

    def test_zero_dimensions_raises(self):
        with pytest.raises(ValueError, match="dimensions must be positive"):
            EmbeddingVector(values=(), model_name="m", dimensions=0)

    def test_negative_dimensions_raises(self):
        with pytest.raises(ValueError):
            EmbeddingVector(values=(), model_name="m", dimensions=-1)

    def test_nan_value_raises(self):
        with pytest.raises(ValueError, match="not finite"):
            EmbeddingVector(
                values=(1.0, float("nan"), 0.0), model_name="m", dimensions=3
            )

    def test_inf_value_raises(self):
        with pytest.raises(ValueError, match="not finite"):
            EmbeddingVector(
                values=(float("inf"), 0.0, 0.0), model_name="m", dimensions=3
            )

    def test_negative_inf_value_raises(self):
        with pytest.raises(ValueError, match="not finite"):
            EmbeddingVector(
                values=(0.0, float("-inf"), 0.0), model_name="m", dimensions=3
            )

    # -- Frozen --------------------------------------------------------------

    def test_frozen(self):
        ev = EmbeddingVector(values=(1.0,), model_name="m", dimensions=1)
        with pytest.raises(AttributeError):
            ev.values = (2.0,)  # type: ignore[misc]

    # -- Cosine similarity ---------------------------------------------------

    def test_cosine_identical_vectors(self):
        ev = EmbeddingVector(values=(1.0, 0.0, 0.0), model_name="m", dimensions=3)
        assert ev.cosine_similarity(ev) == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self):
        a = EmbeddingVector(values=(1.0, 0.0, 0.0), model_name="m", dimensions=3)
        b = EmbeddingVector(values=(0.0, 1.0, 0.0), model_name="m", dimensions=3)
        assert a.cosine_similarity(b) == pytest.approx(0.0)

    def test_cosine_opposite_vectors(self):
        a = EmbeddingVector(values=(1.0, 0.0, 0.0), model_name="m", dimensions=3)
        b = EmbeddingVector(values=(-1.0, 0.0, 0.0), model_name="m", dimensions=3)
        assert a.cosine_similarity(b) == pytest.approx(-1.0)

    def test_cosine_45_degree(self):
        a = EmbeddingVector(values=(1.0, 0.0), model_name="m", dimensions=2)
        b = EmbeddingVector(values=(1.0, 1.0), model_name="m", dimensions=2)
        expected = 1.0 / math.sqrt(2.0)
        assert a.cosine_similarity(b) == pytest.approx(expected, abs=1e-9)

    def test_cosine_dimension_mismatch_raises(self):
        a = EmbeddingVector(values=(1.0, 0.0), model_name="m", dimensions=2)
        b = EmbeddingVector(values=(1.0, 0.0, 0.0), model_name="m", dimensions=3)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            a.cosine_similarity(b)

    def test_cosine_zero_vector_returns_zero(self):
        a = EmbeddingVector(values=(0.0, 0.0, 0.0), model_name="m", dimensions=3)
        b = EmbeddingVector(values=(1.0, 0.0, 0.0), model_name="m", dimensions=3)
        assert a.cosine_similarity(b) == 0.0

    def test_cosine_both_zero_vectors(self):
        a = EmbeddingVector(values=(0.0, 0.0), model_name="m", dimensions=2)
        b = EmbeddingVector(values=(0.0, 0.0), model_name="m", dimensions=2)
        assert a.cosine_similarity(b) == 0.0

    def test_cosine_symmetry(self):
        a = EmbeddingVector(values=(1.0, 2.0, 3.0), model_name="m", dimensions=3)
        b = EmbeddingVector(values=(4.0, 5.0, 6.0), model_name="m", dimensions=3)
        assert a.cosine_similarity(b) == pytest.approx(b.cosine_similarity(a))

    # -- Supported models class var ------------------------------------------

    def test_supported_models(self):
        assert "potion-base-8M" in EmbeddingVector.SUPPORTED_MODELS
        assert EmbeddingVector.SUPPORTED_MODELS["potion-base-8M"] == 256
        assert EmbeddingVector.SUPPORTED_MODELS["all-MiniLM-L6-v2"] == 384
        assert EmbeddingVector.SUPPORTED_MODELS["all-mpnet-base-v2"] == 768


# =========================================================================
# SimilarityScore
# =========================================================================


class TestSimilarityScore:
    # -- Valid creation ------------------------------------------------------

    def test_create_default_metric(self):
        ss = SimilarityScore(value=0.75)
        assert ss.value == 0.75
        assert ss.distance_metric == "cosine"

    def test_create_euclidean(self):
        ss = SimilarityScore(value=0.5, distance_metric="euclidean")
        assert ss.distance_metric == "euclidean"

    def test_create_dot_product(self):
        ss = SimilarityScore(value=0.9, distance_metric="dot_product")
        assert ss.distance_metric == "dot_product"

    # -- Range validation ----------------------------------------------------

    def test_boundary_zero(self):
        ss = SimilarityScore(value=0.0)
        assert ss.value == 0.0

    def test_boundary_one(self):
        ss = SimilarityScore(value=1.0)
        assert ss.value == 1.0

    def test_below_zero_raises(self):
        with pytest.raises(ValueError, match="Score must be 0.0"):
            SimilarityScore(value=-0.01)

    def test_above_one_raises(self):
        with pytest.raises(ValueError, match="Score must be 0.0"):
            SimilarityScore(value=1.01)

    # -- Metric validation ---------------------------------------------------

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Invalid metric"):
            SimilarityScore(value=0.5, distance_metric="hamming")

    # -- Properties ----------------------------------------------------------

    def test_is_high_at_0_8(self):
        assert SimilarityScore(value=0.8).is_high is True

    def test_is_high_at_1_0(self):
        assert SimilarityScore(value=1.0).is_high is True

    def test_is_high_below_threshold(self):
        assert SimilarityScore(value=0.79).is_high is False

    def test_is_moderate_at_0_5(self):
        assert SimilarityScore(value=0.5).is_moderate is True

    def test_is_moderate_at_0_79(self):
        assert SimilarityScore(value=0.79).is_moderate is True

    def test_is_moderate_below_0_5(self):
        assert SimilarityScore(value=0.49).is_moderate is False

    def test_is_moderate_at_0_8(self):
        # 0.8 is_high, not is_moderate
        assert SimilarityScore(value=0.8).is_moderate is False

    def test_is_low_at_0(self):
        assert SimilarityScore(value=0.0).is_low is True

    def test_is_low_at_0_49(self):
        assert SimilarityScore(value=0.49).is_low is True

    def test_is_low_at_0_5(self):
        assert SimilarityScore(value=0.5).is_low is False

    # -- Exceeds -------------------------------------------------------------

    def test_exceeds_true(self):
        assert SimilarityScore(value=0.9).exceeds(0.5) is True

    def test_exceeds_exact(self):
        assert SimilarityScore(value=0.5).exceeds(0.5) is True

    def test_exceeds_false(self):
        assert SimilarityScore(value=0.4).exceeds(0.5) is False

    # -- Factory methods -----------------------------------------------------

    def test_cosine_factory(self):
        ss = SimilarityScore.cosine(0.85)
        assert ss.value == 0.85
        assert ss.distance_metric == "cosine"

    def test_zero_factory(self):
        ss = SimilarityScore.zero()
        assert ss.value == 0.0
        assert ss.distance_metric == "cosine"

    # -- Frozen --------------------------------------------------------------

    def test_frozen(self):
        ss = SimilarityScore(value=0.5)
        with pytest.raises(AttributeError):
            ss.value = 0.9  # type: ignore[misc]

    # -- Mutually exclusive categories ---------------------------------------

    def test_categories_mutually_exclusive(self):
        """Exactly one of is_high/is_moderate/is_low is True for any score."""
        for val in [0.0, 0.25, 0.49, 0.5, 0.79, 0.8, 0.99, 1.0]:
            ss = SimilarityScore(value=val)
            cats = [ss.is_high, ss.is_moderate, ss.is_low]
            assert sum(cats) == 1, f"Multiple categories for {val}: {cats}"


# =========================================================================
# ConfidenceScore
# =========================================================================


class TestConfidenceScore:
    # -- Valid creation ------------------------------------------------------

    def test_create(self):
        cs = ConfidenceScore(value=0.75)
        assert cs.value == 0.75

    def test_boundary_zero(self):
        cs = ConfidenceScore(value=0.0)
        assert cs.value == 0.0

    def test_boundary_one(self):
        cs = ConfidenceScore(value=1.0)
        assert cs.value == 1.0

    # -- Validation ----------------------------------------------------------

    def test_below_zero_raises(self):
        with pytest.raises(ValueError, match="Confidence must be 0.0"):
            ConfidenceScore(value=-0.1)

    def test_above_one_raises(self):
        with pytest.raises(ValueError, match="Confidence must be 0.0"):
            ConfidenceScore(value=1.1)

    # -- Action thresholds ---------------------------------------------------

    def test_auto_apply_at_0_9(self):
        cs = ConfidenceScore(value=0.9)
        assert cs.action == "auto_apply"
        assert cs.should_auto_apply is True
        assert cs.should_suggest is False
        assert cs.is_low is False

    def test_auto_apply_at_1_0(self):
        cs = ConfidenceScore(value=1.0)
        assert cs.action == "auto_apply"
        assert cs.should_auto_apply is True

    def test_suggest_at_0_5(self):
        cs = ConfidenceScore(value=0.5)
        assert cs.action == "suggest"
        assert cs.should_suggest is True
        assert cs.should_auto_apply is False
        assert cs.is_low is False

    def test_suggest_at_0_89(self):
        cs = ConfidenceScore(value=0.89)
        assert cs.action == "suggest"
        assert cs.should_suggest is True

    def test_deprioritize_at_0_49(self):
        cs = ConfidenceScore(value=0.49)
        assert cs.action == "deprioritize"
        assert cs.is_low is True
        assert cs.should_auto_apply is False
        assert cs.should_suggest is False

    def test_deprioritize_at_0(self):
        cs = ConfidenceScore(value=0.0)
        assert cs.action == "deprioritize"
        assert cs.is_low is True

    # -- Frozen --------------------------------------------------------------

    def test_frozen(self):
        cs = ConfidenceScore(value=0.5)
        with pytest.raises(AttributeError):
            cs.value = 0.9  # type: ignore[misc]

    # -- Class constants -----------------------------------------------------

    def test_threshold_constants(self):
        assert ConfidenceScore.AUTO_APPLY_THRESHOLD == 0.9
        assert ConfidenceScore.SUGGEST_THRESHOLD == 0.5


# =========================================================================
# TimeDecayFactor
# =========================================================================


class TestTimeDecayFactor:
    # -- Validation ----------------------------------------------------------

    def test_default_half_life(self):
        td = TimeDecayFactor()
        assert td.half_life_days == 30.0

    def test_custom_half_life(self):
        td = TimeDecayFactor(half_life_days=60.0)
        assert td.half_life_days == 60.0

    def test_zero_half_life_raises(self):
        with pytest.raises(ValueError, match="half_life_days must be positive"):
            TimeDecayFactor(half_life_days=0.0)

    def test_negative_half_life_raises(self):
        with pytest.raises(ValueError, match="half_life_days must be positive"):
            TimeDecayFactor(half_life_days=-1.0)

    # -- Frozen --------------------------------------------------------------

    def test_frozen(self):
        td = TimeDecayFactor()
        with pytest.raises(AttributeError):
            td.half_life_days = 60.0  # type: ignore[misc]

    # -- decay_factor --------------------------------------------------------

    def test_decay_factor_at_zero_days(self):
        td = TimeDecayFactor(half_life_days=30.0)
        # 0.5 + 0.5 * e^0 = 0.5 + 0.5 = 1.0
        assert td.decay_factor(0.0) == pytest.approx(1.0)

    def test_decay_factor_at_half_life(self):
        td = TimeDecayFactor(half_life_days=30.0)
        # 0.5 + 0.5 * e^(-1) ~= 0.5 + 0.5 * 0.3679 = 0.684
        expected = 0.5 + 0.5 * math.exp(-1.0)
        assert td.decay_factor(30.0) == pytest.approx(expected, abs=1e-6)

    def test_decay_factor_at_large_days(self):
        td = TimeDecayFactor(half_life_days=30.0)
        # At very large days, e^(-large) -> 0, so factor -> 0.5
        factor = td.decay_factor(10000.0)
        assert factor == pytest.approx(0.5, abs=1e-6)

    def test_decay_factor_monotonically_decreasing(self):
        td = TimeDecayFactor(half_life_days=30.0)
        prev = td.decay_factor(0.0)
        for days in [1, 5, 10, 30, 60, 90, 180, 365]:
            cur = td.decay_factor(float(days))
            assert cur < prev or (cur == prev and days == 0)
            prev = cur

    def test_decay_factor_always_between_0_5_and_1(self):
        td = TimeDecayFactor(half_life_days=30.0)
        for days in [0.0, 0.01, 1.0, 10.0, 100.0, 1000.0]:
            f = td.decay_factor(days)
            assert 0.5 <= f <= 1.0, f"decay_factor({days}) = {f}"

    # -- compute -------------------------------------------------------------

    def test_compute_at_zero_days(self):
        td = TimeDecayFactor(half_life_days=30.0)
        sim = SimilarityScore.cosine(0.8)
        result = td.compute(sim, 0.0)
        # factor=1.0, adjusted = 0.8 * 1.0 = 0.8
        assert result.value == pytest.approx(0.8)
        assert result.distance_metric == "cosine"

    def test_compute_negative_days_clamped_to_zero(self):
        td = TimeDecayFactor(half_life_days=30.0)
        sim = SimilarityScore.cosine(0.8)
        # Negative days should be treated as 0 via max(0.0, days_old)
        result = td.compute(sim, -5.0)
        assert result.value == pytest.approx(0.8)

    def test_compute_old_record(self):
        td = TimeDecayFactor(half_life_days=30.0)
        sim = SimilarityScore.cosine(0.8)
        result = td.compute(sim, 1000.0)
        # factor ~ 0.5, adjusted ~ 0.8 * 0.5 = 0.4
        assert result.value == pytest.approx(0.4, abs=0.01)

    def test_compute_preserves_metric(self):
        td = TimeDecayFactor()
        sim = SimilarityScore(value=0.6, distance_metric="euclidean")
        result = td.compute(sim, 10.0)
        assert result.distance_metric == "euclidean"

    def test_compute_clamps_to_valid_range(self):
        td = TimeDecayFactor()
        sim = SimilarityScore.cosine(1.0)
        result = td.compute(sim, 0.0)
        assert 0.0 <= result.value <= 1.0


# =========================================================================
# MemoryQuery
# =========================================================================


class TestMemoryQuery:
    # -- Valid creation ------------------------------------------------------

    def test_create_minimal(self):
        mq = MemoryQuery(query_text="hello world")
        assert mq.query_text == "hello world"
        assert mq.memory_type is None
        assert mq.top_k == 5
        assert mq.min_similarity == 0.3
        assert mq.apply_time_decay is True
        assert mq.session_id is None

    def test_create_with_all_params(self):
        mt = MemoryType.keywords()
        mq = MemoryQuery(
            query_text="test query",
            memory_type=mt,
            top_k=10,
            min_similarity=0.5,
            apply_time_decay=False,
            session_id="sess-1",
        )
        assert mq.memory_type == mt
        assert mq.top_k == 10
        assert mq.min_similarity == 0.5
        assert mq.apply_time_decay is False
        assert mq.session_id == "sess-1"

    # -- Validation ----------------------------------------------------------

    def test_empty_query_text_raises(self):
        with pytest.raises(ValueError, match="query_text must be non-empty"):
            MemoryQuery(query_text="")

    def test_whitespace_only_query_text_raises(self):
        with pytest.raises(ValueError, match="query_text must be non-empty"):
            MemoryQuery(query_text="   ")

    def test_query_text_exceeds_max_length(self):
        with pytest.raises(ValueError, match="query_text exceeds"):
            MemoryQuery(query_text="a" * 2001)

    def test_query_text_at_max_length(self):
        # Should not raise
        mq = MemoryQuery(query_text="a" * 2000)
        assert len(mq.query_text) == 2000

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError, match="top_k must be 1"):
            MemoryQuery(query_text="test", top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValueError, match="top_k must be 1"):
            MemoryQuery(query_text="test", top_k=-1)

    def test_top_k_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="top_k must be 1"):
            MemoryQuery(query_text="test", top_k=51)

    def test_top_k_at_max(self):
        mq = MemoryQuery(query_text="test", top_k=50)
        assert mq.top_k == 50

    def test_top_k_at_min(self):
        mq = MemoryQuery(query_text="test", top_k=1)
        assert mq.top_k == 1

    def test_min_similarity_below_zero_raises(self):
        with pytest.raises(ValueError, match="min_similarity must be 0.0"):
            MemoryQuery(query_text="test", min_similarity=-0.1)

    def test_min_similarity_above_one_raises(self):
        with pytest.raises(ValueError, match="min_similarity must be 0.0"):
            MemoryQuery(query_text="test", min_similarity=1.1)

    def test_min_similarity_boundaries(self):
        MemoryQuery(query_text="test", min_similarity=0.0)
        MemoryQuery(query_text="test", min_similarity=1.0)

    # -- Properties ----------------------------------------------------------

    def test_is_scoped_true(self):
        mq = MemoryQuery(query_text="test", memory_type=MemoryType.keywords())
        assert mq.is_scoped is True

    def test_is_scoped_false(self):
        mq = MemoryQuery(query_text="test")
        assert mq.is_scoped is False

    def test_is_session_scoped_true(self):
        mq = MemoryQuery(query_text="test", session_id="s1")
        assert mq.is_session_scoped is True

    def test_is_session_scoped_false(self):
        mq = MemoryQuery(query_text="test")
        assert mq.is_session_scoped is False

    def test_collection_names_scoped(self):
        mq = MemoryQuery(query_text="test", memory_type=MemoryType.keywords())
        assert mq.collection_names == ["rfmcp_keywords"]

    def test_collection_names_unscoped(self):
        mq = MemoryQuery(query_text="test")
        names = mq.collection_names
        assert len(names) == 6
        for n in names:
            assert n.startswith("rfmcp_")

    # -- Factory methods -----------------------------------------------------

    def test_for_error_fix(self):
        mq = MemoryQuery.for_error_fix("some error text")
        assert mq.query_text == "some error text"
        assert mq.memory_type == MemoryType.common_errors()
        assert mq.top_k == 3
        assert mq.min_similarity == 0.3

    def test_for_error_fix_with_session(self):
        mq = MemoryQuery.for_error_fix("err", session_id="s1")
        assert mq.session_id == "s1"

    def test_for_error_fix_truncates_long_text(self):
        long_text = "x" * 3000
        mq = MemoryQuery.for_error_fix(long_text)
        assert len(mq.query_text) == MemoryQuery.MAX_QUERY_LENGTH

    def test_for_keyword_recall(self):
        mq = MemoryQuery.for_keyword_recall("Click Element")
        assert mq.memory_type == MemoryType.keywords()
        assert mq.top_k == 5
        assert mq.min_similarity == 0.2

    def test_for_step_recall(self):
        mq = MemoryQuery.for_step_recall("login to the website")
        assert mq.memory_type == MemoryType.working_steps()
        assert mq.top_k == 10
        assert mq.min_similarity == 0.05

    # -- Frozen --------------------------------------------------------------

    def test_frozen(self):
        mq = MemoryQuery(query_text="test")
        with pytest.raises(AttributeError):
            mq.query_text = "other"  # type: ignore[misc]


# =========================================================================
# MemoryEntry
# =========================================================================


class TestMemoryEntry:
    # -- Valid creation ------------------------------------------------------

    def test_create_minimal(self):
        me = MemoryEntry(content="hello", memory_type=MemoryType.keywords())
        assert me.content == "hello"
        assert me.memory_type == MemoryType.keywords()
        assert me.metadata == {}
        assert me.embedding is None
        assert me.tags == ()

    def test_create_with_all_params(self):
        ev = EmbeddingVector.from_list([1.0, 0.0, 0.0], "m")
        me = MemoryEntry(
            content="test content",
            memory_type=MemoryType.documentation(),
            metadata={"source": "test"},
            embedding=ev,
            tags=("tag1", "tag2"),
        )
        assert me.metadata == {"source": "test"}
        assert me.embedding == ev
        assert me.tags == ("tag1", "tag2")

    # -- Validation ----------------------------------------------------------

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="content must be non-empty"):
            MemoryEntry(content="", memory_type=MemoryType.keywords())

    def test_whitespace_only_content_raises(self):
        with pytest.raises(ValueError, match="content must be non-empty"):
            MemoryEntry(content="   \n\t  ", memory_type=MemoryType.keywords())

    def test_content_exceeds_max_length(self):
        with pytest.raises(ValueError, match="content exceeds"):
            MemoryEntry(
                content="x" * 50_001, memory_type=MemoryType.keywords()
            )

    def test_content_at_max_length(self):
        me = MemoryEntry(content="x" * 50_000, memory_type=MemoryType.keywords())
        assert len(me.content) == 50_000

    # -- Properties ----------------------------------------------------------

    def test_has_embedding_true(self):
        ev = EmbeddingVector.from_list([1.0], "m")
        me = MemoryEntry(
            content="c", memory_type=MemoryType.keywords(), embedding=ev
        )
        assert me.has_embedding is True

    def test_has_embedding_false(self):
        me = MemoryEntry(content="c", memory_type=MemoryType.keywords())
        assert me.has_embedding is False

    def test_content_preview_short(self):
        me = MemoryEntry(content="short text", memory_type=MemoryType.keywords())
        assert me.content_preview == "short text"

    def test_content_preview_long(self):
        me = MemoryEntry(content="a" * 200, memory_type=MemoryType.keywords())
        assert len(me.content_preview) == 100
        assert me.content_preview == "a" * 100

    def test_word_count(self):
        me = MemoryEntry(
            content="one two three four five",
            memory_type=MemoryType.keywords(),
        )
        assert me.word_count == 5

    def test_word_count_single(self):
        me = MemoryEntry(content="word", memory_type=MemoryType.keywords())
        assert me.word_count == 1

    # -- with_embedding ------------------------------------------------------

    def test_with_embedding(self):
        me = MemoryEntry(
            content="c",
            memory_type=MemoryType.keywords(),
            metadata={"k": "v"},
            tags=("t1",),
        )
        ev = EmbeddingVector.from_list([1.0, 2.0], "m")
        new_me = me.with_embedding(ev)

        assert new_me.embedding == ev
        assert new_me.content == "c"
        assert new_me.memory_type == MemoryType.keywords()
        assert new_me.metadata == {"k": "v"}
        assert new_me.tags == ("t1",)
        # Original unchanged
        assert me.embedding is None

    # -- with_tags -----------------------------------------------------------

    def test_with_tags(self):
        me = MemoryEntry(content="c", memory_type=MemoryType.keywords())
        new_me = me.with_tags("a", "b", "c")
        assert new_me.tags == ("a", "b", "c")
        # Original unchanged
        assert me.tags == ()

    def test_with_tags_replaces_all(self):
        me = MemoryEntry(
            content="c", memory_type=MemoryType.keywords(), tags=("old",)
        )
        new_me = me.with_tags("new")
        assert new_me.tags == ("new",)

    # -- Frozen --------------------------------------------------------------

    def test_frozen(self):
        me = MemoryEntry(content="c", memory_type=MemoryType.keywords())
        with pytest.raises(AttributeError):
            me.content = "other"  # type: ignore[misc]


# =========================================================================
# RecallResult
# =========================================================================


class TestRecallResult:
    def _make_result(self, **kwargs):
        defaults = {
            "record_id": "rec-1",
            "content": "test content",
            "memory_type": MemoryType.keywords(),
            "similarity": SimilarityScore.cosine(0.9),
            "adjusted_similarity": SimilarityScore.cosine(0.85),
            "age_days": 5.123,
            "metadata": {},
            "confidence": None,
            "rank": 1,
        }
        defaults.update(kwargs)
        return RecallResult(**defaults)

    def test_create(self):
        rr = self._make_result()
        assert rr.record_id == "rec-1"
        assert rr.content == "test content"
        assert rr.rank == 1

    def test_to_dict_without_confidence(self):
        rr = self._make_result()
        d = rr.to_dict()
        assert d["record_id"] == "rec-1"
        assert d["content"] == "test content"
        assert d["memory_type"] == "keywords"
        assert d["similarity"] == 0.9
        assert d["adjusted_similarity"] == 0.85
        assert d["age_days"] == 5.1  # rounded
        assert d["rank"] == 1
        assert "confidence" not in d
        assert "action" not in d

    def test_to_dict_with_confidence(self):
        rr = self._make_result(confidence=ConfidenceScore(value=0.95))
        d = rr.to_dict()
        assert d["confidence"] == 0.95
        assert d["action"] == "auto_apply"

    def test_to_dict_with_metadata(self):
        rr = self._make_result(metadata={"source": "test"})
        d = rr.to_dict()
        assert d["metadata"] == {"source": "test"}

    def test_to_dict_without_metadata(self):
        rr = self._make_result(metadata={})
        d = rr.to_dict()
        assert "metadata" not in d

    def test_frozen(self):
        rr = self._make_result()
        with pytest.raises(AttributeError):
            rr.record_id = "other"  # type: ignore[misc]


# =========================================================================
# StorageConfig
# =========================================================================


class TestStorageConfig:
    # -- Defaults ------------------------------------------------------------

    def test_defaults(self):
        sc = StorageConfig()
        assert sc.db_path == ""
        assert sc.embedding_model == "potion-base-8M"
        assert sc.dimension == 256
        assert sc.max_records_per_collection == 10_000
        assert sc.prune_age_days == 90.0
        assert sc.time_decay_half_life == 30.0
        assert sc.enabled is False
        assert sc.project_id == "default"

    # -- Validation ----------------------------------------------------------

    def test_zero_dimension_raises(self):
        with pytest.raises(ValueError, match="dimension must be positive"):
            StorageConfig(dimension=0)

    def test_negative_dimension_raises(self):
        with pytest.raises(ValueError, match="dimension must be positive"):
            StorageConfig(dimension=-1)

    def test_zero_max_records_raises(self):
        with pytest.raises(ValueError, match="max_records_per_collection must be positive"):
            StorageConfig(max_records_per_collection=0)

    def test_negative_prune_age_raises(self):
        with pytest.raises(ValueError, match="prune_age_days must be positive"):
            StorageConfig(prune_age_days=-1.0)

    def test_zero_prune_age_raises(self):
        with pytest.raises(ValueError, match="prune_age_days must be positive"):
            StorageConfig(prune_age_days=0.0)

    def test_zero_time_decay_half_life_raises(self):
        with pytest.raises(ValueError, match="time_decay_half_life must be positive"):
            StorageConfig(time_decay_half_life=0.0)

    def test_negative_time_decay_half_life_raises(self):
        with pytest.raises(ValueError, match="time_decay_half_life must be positive"):
            StorageConfig(time_decay_half_life=-5.0)

    # -- Frozen --------------------------------------------------------------

    def test_frozen(self):
        sc = StorageConfig()
        with pytest.raises(AttributeError):
            sc.dimension = 512  # type: ignore[misc]

    # -- from_env defaults ---------------------------------------------------

    def test_from_env_defaults(self):
        env = {}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.enabled is False
        assert sc.embedding_model == "potion-base-8M"
        assert sc.dimension == 256
        assert sc.max_records_per_collection == 10_000
        assert sc.prune_age_days == 90.0
        assert sc.time_decay_half_life == 30.0
        assert sc.project_id == "default"
        assert sc.db_path.endswith("memory.db")

    # -- from_env with custom values -----------------------------------------

    def test_from_env_enabled_true(self):
        env = {"ROBOTMCP_MEMORY_ENABLED": "true"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.enabled is True

    def test_from_env_enabled_1(self):
        env = {"ROBOTMCP_MEMORY_ENABLED": "1"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.enabled is True

    def test_from_env_enabled_yes(self):
        env = {"ROBOTMCP_MEMORY_ENABLED": "yes"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.enabled is True

    def test_from_env_enabled_case_insensitive(self):
        env = {"ROBOTMCP_MEMORY_ENABLED": "TRUE"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.enabled is True

    def test_from_env_enabled_false_explicit(self):
        env = {"ROBOTMCP_MEMORY_ENABLED": "false"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.enabled is False

    def test_from_env_custom_model(self):
        env = {"ROBOTMCP_MEMORY_MODEL": "all-MiniLM-L6-v2"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.embedding_model == "all-MiniLM-L6-v2"
        assert sc.dimension == 384

    def test_from_env_unknown_model_falls_back_to_256(self):
        env = {"ROBOTMCP_MEMORY_MODEL": "unknown-model"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.dimension == 256

    def test_from_env_custom_db_path(self):
        env = {"ROBOTMCP_MEMORY_DB_PATH": "/custom/path/memory.db"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.db_path == "/custom/path/memory.db"

    def test_from_env_custom_max_records(self):
        env = {"ROBOTMCP_MEMORY_MAX_RECORDS": "5000"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.max_records_per_collection == 5000

    def test_from_env_custom_prune_days(self):
        env = {"ROBOTMCP_MEMORY_PRUNE_DAYS": "30"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.prune_age_days == 30.0

    def test_from_env_custom_decay_half_life(self):
        env = {"ROBOTMCP_MEMORY_DECAY_HALF_LIFE": "60"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.time_decay_half_life == 60.0

    def test_from_env_custom_project_id(self):
        env = {"ROBOTMCP_PROJECT_ID": "my-project"}
        with patch.dict(os.environ, env, clear=True):
            sc = StorageConfig.from_env()
        assert sc.project_id == "my-project"

    # -- default() -----------------------------------------------------------

    def test_default_factory(self):
        sc = StorageConfig.default()
        assert sc.db_path.endswith("memory.db")
        assert sc.embedding_model == "potion-base-8M"
        assert sc.dimension == 256
        assert sc.enabled is False
