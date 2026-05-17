"""``intent="extract_text"`` deprecation in favour of ``intent="extract"``.

Empirical analysis in ``docs/reviews/extract_vs_extract_text_overlap.md``
showed that ``extract_text`` and ``extract(mode="text")`` produce
byte-identical RF dispatch (``Get Text`` with the same args) — the
only difference is that ``extract`` additionally surfaces
``extracted_value`` at the top level of the response and supports
five other modes (attribute / count / value / url / title).

Having both verbs in the public IntentVerb Literal alias and the
intent_action docstring's "Valid intents:" list creates real agent
confusion (5 failure modes documented in the review). Option A from
the review: deprecate ``extract_text`` cleanly while preserving the
existing mappings + tests so no breaking change for callers.

These tests pin the deprecation surface:

1. ``intent="extract_text"`` still resolves successfully (no breaking
   change) but emits a ``DeprecationWarning``.
2. ``intent="extract"`` does NOT emit any deprecation warning.
3. The ``IntentResolutionError`` hint (server.py:6571) lists
   ``extract`` AND documents ``extract_text`` as deprecated.
4. The ``intent_action`` docstring marks ``extract_text`` deprecated
   with a clear migration path to ``extract`` with ``mode="text"``.
5. The ``IntentVerb`` enum docstring around ``EXTRACT_TEXT`` references
   the deprecation.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from robotmcp.domains.intent.value_objects import IntentVerb


# ---------------------------------------------------------------------------
# Layer 1: DeprecationWarning fires for extract_text resolution
# ---------------------------------------------------------------------------


def _build_real_adapter():
    """Build an IntentActionAdapter with the real registry + a
    minimal-but-valid resolver dependency surface. Lets us exercise
    the warning emission without spinning up a real RF session."""
    from robotmcp.domains.intent.adapters.mcp_tool import IntentActionAdapter
    from robotmcp.domains.intent.aggregates import IntentRegistry
    from robotmcp.domains.intent.services import IntentResolver
    from robotmcp.domains.intent.value_objects import NormalizedLocator

    class _SL:
        def get_active_library(self, sid): return "Browser"
        def get_active_web_library(self, sid): return "Browser"
        def get_imported_libraries(self, sid): return ["Browser"]

    class _Norm:
        def normalize(self, target, library):
            return NormalizedLocator(
                value=target.locator, source_locator=target.locator,
                target_library=library, strategy_applied="auto",
                was_transformed=False,
            )

    return IntentActionAdapter(
        resolver=IntentResolver(
            registry=IntentRegistry.with_builtins(),
            session_lookup=_SL(),
            normalizer=_Norm(),
        ),
    )


class TestDeprecationWarningEmitted:
    """``intent_action(intent="extract_text", ...)`` must emit a
    DeprecationWarning. The warning's message must include the
    migration path (``extract`` with ``mode="text"``) so the LLM /
    user can fix the call without reading separate docs."""

    def test_warning_emitted_when_intent_is_extract_text(self):
        adapter = _build_real_adapter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            adapter.resolve_intent(intent="extract_text", target="id=foo")
        # Find our specific deprecation warning (other code paths may
        # emit unrelated DeprecationWarnings in the same call).
        ours = [w for w in caught
                if issubclass(w.category, DeprecationWarning)
                and "extract_text" in str(w.message)]
        assert len(ours) == 1, (
            f"expected exactly one extract_text DeprecationWarning, "
            f"got {len(ours)}: {[str(w.message) for w in caught]}"
        )

    def test_warning_message_includes_migration_path(self):
        adapter = _build_real_adapter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            adapter.resolve_intent(intent="extract_text", target="id=foo")
        message = str(caught[0].message)
        # Must point the caller at the canonical replacement spelling.
        assert "extract" in message
        assert 'mode="text"' in message or "mode='text'" in message

    def test_warning_message_notes_extracted_value_field(self):
        """The migration message should sell the upside (top-level
        ``extracted_value`` field + multi-mode support) so callers
        understand WHY to move, not just that they should."""
        adapter = _build_real_adapter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            adapter.resolve_intent(intent="extract_text", target="id=foo")
        message = str(caught[0].message)
        assert "extracted_value" in message

    def test_no_warning_for_extract_intent(self):
        """The canonical ``extract`` intent must NOT emit any
        deprecation warning — that's the migration target."""
        adapter = _build_real_adapter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            adapter.resolve_intent(
                intent="extract", target="id=foo", mode="text",
            )
        ours = [w for w in caught
                if issubclass(w.category, DeprecationWarning)
                and "extract_text" in str(w.message)]
        assert ours == []

    @pytest.mark.parametrize("other_intent", [
        "click", "navigate", "fill", "hover", "select",
        "assert_visible", "wait_for",
    ])
    def test_no_warning_for_other_intents(self, other_intent):
        """Other intents must NOT trip the extract_text deprecation
        warning — the check must key on intent identity, not coincidence."""
        adapter = _build_real_adapter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            try:
                # Some intents need a value (fill, select); pass one to
                # avoid the resolver tripping on missing required args.
                kwargs = {"intent": other_intent, "target": "id=foo"}
                if other_intent in ("fill", "select"):
                    kwargs["value"] = "x"
                adapter.resolve_intent(**kwargs)
            except Exception:
                # Resolution-error path is fine — we only care about
                # whether OUR deprecation warning was incorrectly raised.
                pass
        ours = [w for w in caught
                if issubclass(w.category, DeprecationWarning)
                and "extract_text" in str(w.message)]
        assert ours == [], (
            f"extract_text deprecation warning leaked for intent={other_intent!r}"
        )


# ---------------------------------------------------------------------------
# Layer 2: extract_text still resolves correctly (no breaking change)
# ---------------------------------------------------------------------------


class TestExtractTextStillResolves:
    """Backward-compat invariant: deprecating must NOT break callers.
    ``extract_text`` resolves to the same RF dispatch as before; only
    the warning emission is new."""

    def test_extract_text_resolves_to_get_text(self):
        adapter = _build_real_adapter()
        # Suppress the deprecation warning we just pinned in Layer 1 so
        # this test doesn't fail with -Werror.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = adapter.resolve_intent(
                intent="extract_text", target="id=foo",
            )
        assert result["keyword"] == "Get Text"
        assert result["arguments"] == ["id=foo"]
        assert result["library"] == "Browser"

    def test_extract_and_extract_text_produce_same_keyword_dispatch(self):
        """Empirical contract from the overlap-review: identical RF
        dispatch for the text-extraction case. Pinning here so a
        future change to either path doesn't silently break the
        deprecation promise."""
        adapter = _build_real_adapter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            extract_text_result = adapter.resolve_intent(
                intent="extract_text", target="id=foo",
            )
            extract_result = adapter.resolve_intent(
                intent="extract", target="id=foo", mode="text",
            )
        assert extract_text_result["keyword"] == extract_result["keyword"]
        assert extract_text_result["arguments"] == extract_result["arguments"]
        assert extract_text_result["library"] == extract_result["library"]


# ---------------------------------------------------------------------------
# Layer 3: User-facing surfaces document the deprecation
# ---------------------------------------------------------------------------


class TestDocumentationSurfaces:
    """Three user-facing strings must call out the deprecation:
    the intent_action docstring, the IntentResolutionError hint, and
    the IntentVerb enum's EXTRACT_TEXT entry."""

    def test_intent_action_docstring_marks_extract_text_deprecated(self):
        from robotmcp.server import intent_action
        fn = getattr(intent_action, "fn", intent_action)
        doc = inspect.getdoc(fn) or ""
        # The "Valid intents:" listing must not silently list both
        # — the deprecated alias should be flagged.
        assert "DEPRECATED" in doc, (
            "intent_action docstring must call out extract_text as DEPRECATED"
        )
        assert "extract_text" in doc
        # And it must point at the replacement spelling explicitly.
        assert 'mode="text"' in doc or "mode='text'" in doc

    def test_intent_resolution_error_hint_lists_extract_and_flags_extract_text(self):
        """Failure-path hint at server.py:6571 — confirmed bug in the
        overlap review: the original hint listed extract_text and
        OMITTED extract entirely. Fix: list extract AND document
        extract_text's deprecated status in the same hint."""
        import pathlib
        server_src = pathlib.Path(
            "src/robotmcp/server.py",
        ).read_text(encoding="utf-8")
        # Locate the IntentResolutionError hint block (anchored on the
        # distinctive prefix from the existing message).
        anchor = '"Use execute_step for direct keyword access, or check "'
        idx = server_src.find(anchor)
        assert idx != -1, "IntentResolutionError hint anchor not found"
        hint_block = server_src[idx:idx + 600]
        # extract must be in the canonical list:
        assert "extract," in hint_block or "extract " in hint_block
        # extract_text must be flagged as deprecated:
        assert "extract_text" in hint_block
        assert "deprecated" in hint_block.lower()

    def test_intent_verb_enum_docstring_marks_extract_text_deprecated(self):
        """The IntentVerb enum value docstring (a comment immediately
        above ``EXTRACT_TEXT = "extract_text"``) must reference the
        deprecation so anyone editing the enum sees the migration
        rationale without having to cross-reference the docs file."""
        import pathlib
        value_objects_src = pathlib.Path(
            "src/robotmcp/domains/intent/value_objects.py",
        ).read_text(encoding="utf-8")
        # Find the EXTRACT_TEXT line and look at the preceding comment.
        lines = value_objects_src.splitlines()
        extract_text_line_idx = next(
            (i for i, line in enumerate(lines)
             if 'EXTRACT_TEXT = "extract_text"' in line),
            None,
        )
        assert extract_text_line_idx is not None
        # Look at the 6 lines immediately preceding for the comment block.
        preceding = "\n".join(lines[max(0, extract_text_line_idx - 6):extract_text_line_idx])
        assert "DEPRECATED" in preceding or "deprecated" in preceding.lower()
        # Migration path mentioned:
        assert "EXTRACT" in preceding or "extract" in preceding.lower()

    def test_kernel_literal_alias_marks_extract_text_deprecated(self):
        """The IntentVerb Literal alias in kernel.py — the type-hint
        surface that downstream OpenAPI / ADR-009 schemas inherit
        from — must also flag extract_text as deprecated."""
        import pathlib
        kernel_src = pathlib.Path(
            "src/robotmcp/domains/shared/kernel.py",
        ).read_text(encoding="utf-8")
        # Find the IntentVerb alias block.
        idx = kernel_src.find("IntentVerb = Annotated[")
        assert idx != -1
        # Look in a generous window after the start (the Literal spans
        # a few lines and may include comments).
        alias_block = kernel_src[idx:idx + 800]
        assert '"extract_text"' in alias_block
        assert "DEPRECATED" in alias_block or "deprecated" in alias_block.lower()
