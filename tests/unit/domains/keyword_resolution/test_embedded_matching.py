"""Tests for embedded argument matching (ADR-019 Phase 2)."""
import pytest

from robotmcp.domains.keyword_resolution.value_objects import (
    BddPrefix,
    EmbeddedMatch,
    EmbeddedPattern,
)
from robotmcp.domains.keyword_resolution.services import (
    BddPrefixService,
    EmbeddedMatcherService,
)
from robotmcp.domains.keyword_resolution.events import EmbeddedArgMatched
from robotmcp.models.library_models import KeywordInfo


def _make_kw_info(name: str, library: str = "TestLib") -> KeywordInfo:
    return KeywordInfo(name=name, library=library, method_name=name.lower().replace(" ", "_"))


class TestEmbeddedMatcherService:
    """Tests for EmbeddedMatcherService."""

    def test_is_embedded_keyword_true(self):
        assert EmbeddedMatcherService.is_embedded_keyword("Click ${element}") is True
        assert EmbeddedMatcherService.is_embedded_keyword("Set ${var} to ${value}") is True

    def test_is_embedded_keyword_false(self):
        assert EmbeddedMatcherService.is_embedded_keyword("Click Button") is False
        assert EmbeddedMatcherService.is_embedded_keyword("Open Browser") is False

    def test_create_pattern_simple(self):
        pattern = EmbeddedMatcherService.create_pattern("Click ${element}")
        assert pattern is not None
        assert pattern.template_name == "Click ${element}"
        assert pattern.arg_names == ("element",)
        assert pattern.regex_pattern  # non-empty

    def test_create_pattern_multiple_args(self):
        pattern = EmbeddedMatcherService.create_pattern(
            "Set ${variable} to ${value}"
        )
        assert pattern is not None
        assert pattern.arg_names == ("variable", "value")

    def test_create_pattern_no_embedded_returns_none(self):
        assert EmbeddedMatcherService.create_pattern("Click Button") is None

    def test_match_simple(self):
        pattern = EmbeddedMatcherService.create_pattern("Click ${element}")
        kw_info = _make_kw_info("Click ${element}")
        result = EmbeddedMatcherService.match(
            "Click login button", [(pattern, kw_info)]
        )
        assert result is not None
        match, info = result
        assert match.concrete_name == "Click login button"
        assert match.template_name == "Click ${element}"
        assert info.library == "TestLib"

    def test_match_extracts_args(self):
        pattern = EmbeddedMatcherService.create_pattern("Click ${element}")
        kw_info = _make_kw_info("Click ${element}")
        result = EmbeddedMatcherService.match(
            "Click submit button", [(pattern, kw_info)]
        )
        assert result is not None
        match, _ = result
        assert match.extracted_args == ("submit button",)

    def test_match_multiple_args(self):
        pattern = EmbeddedMatcherService.create_pattern(
            "Set ${variable} to ${value}"
        )
        kw_info = _make_kw_info("Set ${variable} to ${value}")
        result = EmbeddedMatcherService.match(
            "Set username to admin", [(pattern, kw_info)]
        )
        assert result is not None
        match, _ = result
        assert match.extracted_args == ("username", "admin")

    def test_match_no_match_returns_none(self):
        pattern = EmbeddedMatcherService.create_pattern("Click ${element}")
        kw_info = _make_kw_info("Click ${element}")
        result = EmbeddedMatcherService.match(
            "Open Browser", [(pattern, kw_info)]
        )
        assert result is None

    def test_match_case_insensitive(self):
        """RF embedded matching is case-insensitive by default."""
        pattern = EmbeddedMatcherService.create_pattern("Click ${element}")
        kw_info = _make_kw_info("Click ${element}")
        # RF normalizes keyword names to lowercase for matching
        result = EmbeddedMatcherService.match(
            "click login button", [(pattern, kw_info)]
        )
        # RF's EmbeddedArguments.matches() is case-insensitive
        assert result is not None

    def test_bdd_prefix_then_embedded(self):
        """Strip BDD prefix first, then match embedded pattern."""
        bdd = BddPrefixService.strip_prefix("Given Click login button")
        assert bdd.stripped_name == "Click login button"

        pattern = EmbeddedMatcherService.create_pattern("Click ${element}")
        kw_info = _make_kw_info("Click ${element}")
        result = EmbeddedMatcherService.match(
            bdd.stripped_name, [(pattern, kw_info)]
        )
        assert result is not None
        match, _ = result
        assert match.extracted_args == ("login button",)

    def test_embedded_pattern_invariant_no_dollar(self):
        """EmbeddedPattern must contain ${...} in template_name."""
        with pytest.raises(ValueError, match="template_name must contain embedded args"):
            EmbeddedPattern(
                template_name="Click Button",
                arg_names=(),
                regex_pattern=".*",
            )

    def test_embedded_match_value_object(self):
        match = EmbeddedMatch(
            template_name="Click ${element}",
            concrete_name="Click login button",
            extracted_args=("login button",),
            library="Browser",
        )
        assert match.template_name == "Click ${element}"
        assert match.concrete_name == "Click login button"
        assert match.extracted_args == ("login button",)
        assert match.library == "Browser"

    def test_event_embedded_arg_matched_to_dict(self):
        event = EmbeddedArgMatched(
            call_name="Click login button",
            template_name="Click ${element}",
            extracted_args=("login button",),
            library="Browser",
            source_tool="execute_step",
        )
        d = event.to_dict()
        assert d["event_type"] == "embedded_arg_matched"
        assert d["call_name"] == "Click login button"
        assert d["template_name"] == "Click ${element}"
        assert d["extracted_args"] == ["login button"]
        assert d["library"] == "Browser"
        assert d["source_tool"] == "execute_step"

    def test_multi_match_prefers_more_specific_template(self):
        """When multiple embedded patterns match, the longer (more specific) wins."""
        generic_pattern = EmbeddedMatcherService.create_pattern("Click ${element}")
        specific_pattern = EmbeddedMatcherService.create_pattern(
            "Click ${element} with ${modifier}"
        )
        assert generic_pattern is not None
        assert specific_pattern is not None

        generic_kw = _make_kw_info("Click ${element}", library="GenericLib")
        specific_kw = _make_kw_info("Click ${element} with ${modifier}", library="SpecificLib")

        result = EmbeddedMatcherService.match(
            "Click button with shift",
            [(generic_pattern, generic_kw), (specific_pattern, specific_kw)],
        )
        assert result is not None
        match, kw_info = result
        # The more specific template (longer) should win
        assert match.template_name == "Click ${element} with ${modifier}"
        assert kw_info.library == "SpecificLib"
        assert match.extracted_args == ("button", "shift")
