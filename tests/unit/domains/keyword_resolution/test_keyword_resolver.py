"""Tests for KeywordResolver aggregate (ADR-019 Phase 5)."""
import pytest

from robotmcp.domains.keyword_resolution.aggregates import KeywordResolver
from robotmcp.domains.keyword_resolution.value_objects import (
    BddPrefixType,
    EmbeddedMatch,
)
from robotmcp.models.library_models import KeywordInfo


def _make_kw(name: str, library: str = "BuiltIn") -> KeywordInfo:
    return KeywordInfo(name=name, library=library, method_name=name.lower().replace(" ", "_"))


class TestKeywordResolver:
    def test_resolve_plain_keyword(self):
        """Plain keyword, no BDD prefix, no embedded match."""
        kw = _make_kw("Log")
        info, bdd, embedded = KeywordResolver.resolve(
            "Log", lambda name, **kw: _make_kw("Log") if name == "Log" else None
        )
        assert info is not None
        assert info.name == "Log"
        assert not bdd.has_prefix
        assert embedded is None

    def test_resolve_bdd_prefixed_keyword(self):
        """BDD prefix stripped, then keyword found."""
        kw = _make_kw("Open Browser")

        def find(name, **kwargs):
            return kw if name == "Open Browser" else None

        info, bdd, embedded = KeywordResolver.resolve("Given Open Browser", find)
        assert info is not None
        assert info.name == "Open Browser"
        assert bdd.has_prefix
        assert bdd.prefix_type == BddPrefixType.GIVEN
        assert bdd.stripped_name == "Open Browser"
        assert embedded is None

    def test_resolve_embedded_keyword(self):
        """Embedded match returned from find_keyword_fn."""
        kw = _make_kw("Select ${animal} from list", library="MyLib")
        em = EmbeddedMatch(
            template_name="Select ${animal} from list",
            concrete_name="Select dog from list",
            extracted_args=("dog",),
            library="MyLib",
        )
        # Simulate find_keyword returning kw with _embedded_match attached
        kw._embedded_match = em  # type: ignore[attr-defined]

        def find(name, **kwargs):
            return kw if "select" in name.lower() else None

        info, bdd, embedded = KeywordResolver.resolve("Select dog from list", find)
        assert info is not None
        assert not bdd.has_prefix
        assert embedded is not None
        assert embedded.extracted_args == ("dog",)

    def test_resolve_bdd_plus_embedded(self):
        """BDD prefix stripped, then embedded match found."""
        kw = _make_kw("user selects ${item}", library="MyLib")
        em = EmbeddedMatch(
            template_name="user selects ${item}",
            concrete_name="user selects apple",
            extracted_args=("apple",),
            library="MyLib",
        )
        kw._embedded_match = em  # type: ignore[attr-defined]

        def find(name, **kwargs):
            return kw if "user selects" in name.lower() else None

        info, bdd, embedded = KeywordResolver.resolve(
            "Given user selects apple", find
        )
        assert info is not None
        assert bdd.has_prefix
        assert bdd.prefix_type == BddPrefixType.GIVEN
        assert bdd.stripped_name == "user selects apple"
        assert embedded is not None
        assert embedded.extracted_args == ("apple",)

    def test_resolve_not_found(self):
        """Keyword not found returns (None, bdd_result, None)."""
        info, bdd, embedded = KeywordResolver.resolve(
            "Nonexistent Keyword", lambda name, **kw: None
        )
        assert info is None
        assert not bdd.has_prefix
        assert embedded is None

    def test_resolve_passes_kwargs(self):
        """Extra kwargs forwarded to find_keyword_fn."""
        received_kwargs = {}

        def find(name, **kwargs):
            received_kwargs.update(kwargs)
            return _make_kw("Log")

        KeywordResolver.resolve(
            "Log", find, active_library="Browser", session_libraries=["Browser"]
        )
        assert received_kwargs["active_library"] == "Browser"
        assert received_kwargs["session_libraries"] == ["Browser"]

    def test_resolve_when_prefix(self):
        kw = _make_kw("Click Element")

        def find(name, **kwargs):
            return kw if name == "Click Element" else None

        info, bdd, _ = KeywordResolver.resolve("When Click Element", find)
        assert info is not None
        assert bdd.prefix_type == BddPrefixType.WHEN

    def test_resolve_then_prefix(self):
        kw = _make_kw("Should Be Equal")

        def find(name, **kwargs):
            return kw if name == "Should Be Equal" else None

        info, bdd, _ = KeywordResolver.resolve("Then Should Be Equal", find)
        assert info is not None
        assert bdd.prefix_type == BddPrefixType.THEN

    def test_resolve_and_prefix(self):
        kw = _make_kw("Fill Text")

        def find(name, **kwargs):
            return kw if name == "Fill Text" else None

        info, bdd, _ = KeywordResolver.resolve("And Fill Text", find)
        assert info is not None
        assert bdd.prefix_type == BddPrefixType.AND

    def test_resolve_but_prefix(self):
        kw = _make_kw("Element Should Not Be Visible")

        def find(name, **kwargs):
            return kw if name == "Element Should Not Be Visible" else None

        info, bdd, _ = KeywordResolver.resolve(
            "But Element Should Not Be Visible", find
        )
        assert info is not None
        assert bdd.prefix_type == BddPrefixType.BUT
