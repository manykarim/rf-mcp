"""Tests for BDD prefix stripping (ADR-019 Phase 1)."""
import pytest

from robotmcp.domains.keyword_resolution.value_objects import BddPrefix, BddPrefixType
from robotmcp.domains.keyword_resolution.services import BddPrefixService
from robotmcp.domains.keyword_resolution.events import BddPrefixStripped


class TestBddPrefixValueObject:
    """Tests for BddPrefix value object."""

    def test_strip_given_prefix(self):
        result = BddPrefix.from_keyword("Given Open Browser")
        assert result.stripped_name == "Open Browser"
        assert result.prefix_type == BddPrefixType.GIVEN
        assert result.has_prefix is True

    def test_strip_when_prefix(self):
        result = BddPrefix.from_keyword("When Click Button")
        assert result.stripped_name == "Click Button"
        assert result.prefix_type == BddPrefixType.WHEN

    def test_strip_then_prefix(self):
        result = BddPrefix.from_keyword("Then Page Should Contain")
        assert result.stripped_name == "Page Should Contain"
        assert result.prefix_type == BddPrefixType.THEN

    def test_strip_and_prefix(self):
        result = BddPrefix.from_keyword("And Fill Text")
        assert result.stripped_name == "Fill Text"
        assert result.prefix_type == BddPrefixType.AND

    def test_strip_but_prefix(self):
        result = BddPrefix.from_keyword("But Element Should Not Be Visible")
        assert result.stripped_name == "Element Should Not Be Visible"
        assert result.prefix_type == BddPrefixType.BUT

    def test_case_insensitive(self):
        result = BddPrefix.from_keyword("given open browser")
        assert result.stripped_name == "open browser"
        assert result.prefix_type == BddPrefixType.GIVEN

        result2 = BddPrefix.from_keyword("WHEN Click Button")
        assert result2.stripped_name == "Click Button"
        assert result2.prefix_type == BddPrefixType.WHEN

    def test_no_prefix(self):
        result = BddPrefix.from_keyword("Open Browser")
        assert result.stripped_name == "Open Browser"
        assert result.prefix_type is None
        assert result.has_prefix is False

    def test_no_space_after_prefix(self):
        """GivenOpenBrowser should NOT be stripped — requires space after prefix."""
        result = BddPrefix.from_keyword("GivenOpenBrowser")
        assert result.stripped_name == "GivenOpenBrowser"
        assert result.prefix_type is None
        assert result.has_prefix is False

    def test_empty_keyword_raises_value_error(self):
        with pytest.raises(ValueError, match="original_name must not be empty"):
            BddPrefix.from_keyword("")

    def test_has_prefix_property(self):
        with_prefix = BddPrefix.from_keyword("Given Open Browser")
        assert with_prefix.has_prefix is True

        without_prefix = BddPrefix.from_keyword("Open Browser")
        assert without_prefix.has_prefix is False

    def test_bdd_prefix_type_enum_values(self):
        assert BddPrefixType.GIVEN.value == "Given"
        assert BddPrefixType.WHEN.value == "When"
        assert BddPrefixType.THEN.value == "Then"
        assert BddPrefixType.AND.value == "And"
        assert BddPrefixType.BUT.value == "But"


class TestBddPrefixService:
    """Tests for BddPrefixService."""

    def test_service_strip_prefix(self):
        result = BddPrefixService.strip_prefix("Given Open Browser")
        assert result.stripped_name == "Open Browser"
        assert result.prefix_type == BddPrefixType.GIVEN

    def test_service_is_bdd_prefixed(self):
        assert BddPrefixService.is_bdd_prefixed("Given Open Browser") is True
        assert BddPrefixService.is_bdd_prefixed("Open Browser") is False
        assert BddPrefixService.is_bdd_prefixed("When Click") is True
        assert BddPrefixService.is_bdd_prefixed("ThenSomething") is False


class TestBddPrefixStrippedEvent:
    """Tests for BddPrefixStripped event."""

    def test_event_to_dict(self):
        event = BddPrefixStripped(
            original_name="Given Open Browser",
            stripped_name="Open Browser",
            prefix="Given",
            source_tool="execute_step",
            session_id="s1",
        )
        d = event.to_dict()
        assert d["event_type"] == "bdd_prefix_stripped"
        assert d["original_name"] == "Given Open Browser"
        assert d["stripped_name"] == "Open Browser"
        assert d["prefix"] == "Given"
        assert d["source_tool"] == "execute_step"
