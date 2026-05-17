"""OBS-10 — Browser ``Drag And Drop`` pre-scrolls source + target.

The 2026-05-17 post-OBS validation benchmark surfaced this silent
failure on Obstacle 10 (Todolist):

    Drag And Drop    css=tr[task='1']    css=tbody.droparea

returns success (no error) but the DOM doesn't update when the source
element is below the viewport (y = 737.5px in that case). Cost on
the benchmark: ~5 minutes of Sonnet's debugging cycles + 27 tool calls
on this single obstacle.

The fix: a Browser-plugin override pre-scrolls source + target into
view via ``Browser.Scroll To Element`` BEFORE dispatching the actual
Drag And Drop. The override returns ``None`` so the normal Drag And
Drop dispatch proceeds — only difference is the elements are now
guaranteed to be in the viewport.

These tests pin:
(1) The override is registered for ``"drag and drop"``.
(2) When invoked with two string locator args, both are passed to
    ``Browser.Scroll To Element`` (in order).
(3) The override returns ``None`` (delegation, not short-circuit).
(4) Scroll-to-element failures are swallowed (logged, not re-raised) —
    the drag should still get a chance to run with its own error.
(5) Non-string args are skipped (defensive against routing of other
    drag-variant keywords).
(6) Missing BuiltIn (unit-test context without RF) doesn't crash.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin


# ---------------------------------------------------------------------------
# Layer 1: override registration
# ---------------------------------------------------------------------------


class TestDragAndDropOverrideRegistered:
    """``drag and drop`` must appear in the plugin's override map AND
    point at ``_override_drag_and_drop``."""

    def test_keyword_registered_in_override_map(self):
        plugin = BrowserLibraryPlugin()
        overrides = plugin.get_keyword_overrides()
        assert "drag and drop" in overrides

    def test_keyword_routed_to_drag_and_drop_override(self):
        plugin = BrowserLibraryPlugin()
        overrides = plugin.get_keyword_overrides()
        assert overrides["drag and drop"] == plugin._override_drag_and_drop

    def test_open_browser_override_still_registered(self):
        # Regression-guard: OBS-10 adds an override; it must NOT
        # remove the pre-existing Open Browser override.
        plugin = BrowserLibraryPlugin()
        overrides = plugin.get_keyword_overrides()
        assert "open browser" in overrides


# ---------------------------------------------------------------------------
# Layer 2: pre-scroll behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPreScrollHappyPath:
    """ONLY the source locator (arg 0) gets pre-scrolled.

    Rationale: scrolling target THEN source is a net no-op when both
    are far apart (the second scroll undoes the first). Playwright's
    drag mechanics handle target-side scrolling DURING the drag
    motion, so pre-scrolling target isn't needed. Sonnet's proven
    Obstacle-10 recovery scrolled only the source — match that."""

    async def test_scrolls_source_only(self):
        plugin = BrowserLibraryPlugin()
        mock_builtin = MagicMock()
        with patch("robot.libraries.BuiltIn.BuiltIn",
                   return_value=mock_builtin):
            result = await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=["css=tr[task='1']", "css=tbody.droparea"],
            )
        # Override delegates back to normal dispatch (no short-circuit).
        assert result is None
        # Exactly ONE scroll call: the source. Target is NOT scrolled
        # — see docstring on _override_drag_and_drop.
        scroll_calls = [
            c for c in mock_builtin.run_keyword.call_args_list
            if c.args[0] == "Browser.Scroll To Element"
        ]
        assert len(scroll_calls) == 1
        assert scroll_calls[0].args[1] == "css=tr[task='1']"

    async def test_scrolls_id_source(self):
        plugin = BrowserLibraryPlugin()
        mock_builtin = MagicMock()
        with patch("robot.libraries.BuiltIn.BuiltIn",
                   return_value=mock_builtin):
            await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=["id=toscabot", "id=to"],
            )
        scroll_calls = [
            c for c in mock_builtin.run_keyword.call_args_list
            if c.args[0] == "Browser.Scroll To Element"
        ]
        assert len(scroll_calls) == 1
        assert scroll_calls[0].args[1] == "id=toscabot"

    async def test_target_not_scrolled(self):
        """Regression-guard: the target (arg 1) must NOT be scrolled.
        See OBS-10 design note on why target-then-source order is a
        net no-op when both elements are far apart."""
        plugin = BrowserLibraryPlugin()
        mock_builtin = MagicMock()
        with patch("robot.libraries.BuiltIn.BuiltIn",
                   return_value=mock_builtin):
            await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=["id=src", "id=tgt"],
            )
        scroll_calls = [
            c for c in mock_builtin.run_keyword.call_args_list
            if c.args[0] == "Browser.Scroll To Element"
        ]
        target_scrolls = [c for c in scroll_calls if c.args[1] == "id=tgt"]
        assert target_scrolls == [], (
            "target locator must NOT be pre-scrolled; Playwright handles "
            "target-side scroll during drag motion"
        )

    async def test_steps_arg_is_not_scrolled(self):
        """Drag And Drop takes (source, target, [steps]). Only the
        source (arg 0) is scrolled regardless of how many trailing
        args the call has."""
        plugin = BrowserLibraryPlugin()
        mock_builtin = MagicMock()
        with patch("robot.libraries.BuiltIn.BuiltIn",
                   return_value=mock_builtin):
            await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=["id=src", "id=tgt", "5"],   # 5 = steps
            )
        scroll_calls = [
            c for c in mock_builtin.run_keyword.call_args_list
            if c.args[0] == "Browser.Scroll To Element"
        ]
        # Exactly one scroll (source only).
        assert len(scroll_calls) == 1
        assert scroll_calls[0].args[1] == "id=src"


# ---------------------------------------------------------------------------
# Layer 3: defensive behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPreScrollDefensive:
    """The override must NEVER crash the drag. Failures in scroll
    should be swallowed; the normal drag dispatch then runs and
    produces its own (more useful) error if there's a real problem."""

    async def test_scroll_failure_does_not_propagate(self):
        plugin = BrowserLibraryPlugin()
        mock_builtin = MagicMock()
        mock_builtin.run_keyword.side_effect = RuntimeError(
            "Element not found: id=missing",
        )
        with patch("robot.libraries.BuiltIn.BuiltIn",
                   return_value=mock_builtin):
            result = await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=["id=missing", "id=target"],
            )
        # Returns None (delegation), no exception bubbled.
        assert result is None

    async def test_scroll_failure_on_source_does_not_block_dispatch(self):
        """When the source scroll fails, the override must still
        return None so the normal Drag And Drop dispatch can run and
        produce its own (more useful) error message."""
        plugin = BrowserLibraryPlugin()
        mock_builtin = MagicMock()
        mock_builtin.run_keyword.side_effect = RuntimeError(
            "source scroll failed",
        )
        with patch("robot.libraries.BuiltIn.BuiltIn",
                   return_value=mock_builtin):
            result = await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=["id=bad", "id=target"],
            )
        assert result is None
        # Exactly one scroll attempt (source); the failure is
        # swallowed but the call WAS made.
        assert mock_builtin.run_keyword.call_count == 1

    async def test_missing_builtin_does_not_crash(self):
        """Unit-test contexts may not have ``robot`` installed or
        importable. The override must degrade to a no-op instead of
        crashing the executor."""
        plugin = BrowserLibraryPlugin()
        # Force the import to raise inside the override.
        with patch.dict("sys.modules", {"robot.libraries.BuiltIn": None}):
            result = await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=["id=src", "id=tgt"],
            )
        assert result is None

    @pytest.mark.parametrize("non_locator", [None, 42, 3.14, [], {}, True])
    async def test_non_string_args_are_skipped(self, non_locator):
        """If someone routes a coords-only drag variant here by
        mistake, integer args must NOT be passed to Scroll To Element."""
        plugin = BrowserLibraryPlugin()
        mock_builtin = MagicMock()
        with patch("robot.libraries.BuiltIn.BuiltIn",
                   return_value=mock_builtin):
            await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=[non_locator, non_locator],
            )
        scroll_calls = [
            c for c in mock_builtin.run_keyword.call_args_list
            if c.args[0] == "Browser.Scroll To Element"
        ]
        assert len(scroll_calls) == 0

    async def test_empty_string_locator_is_skipped(self):
        plugin = BrowserLibraryPlugin()
        mock_builtin = MagicMock()
        with patch("robot.libraries.BuiltIn.BuiltIn",
                   return_value=mock_builtin):
            await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=["", "  "],
            )
        scroll_calls = [
            c for c in mock_builtin.run_keyword.call_args_list
            if c.args[0] == "Browser.Scroll To Element"
        ]
        assert len(scroll_calls) == 0

    async def test_zero_args_does_not_crash(self):
        plugin = BrowserLibraryPlugin()
        mock_builtin = MagicMock()
        with patch("robot.libraries.BuiltIn.BuiltIn",
                   return_value=mock_builtin):
            result = await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=[],
            )
        assert result is None
        # No scroll calls when there are no args to scroll.
        assert mock_builtin.run_keyword.call_count == 0

    async def test_only_one_arg_still_scrolls_the_source(self):
        plugin = BrowserLibraryPlugin()
        mock_builtin = MagicMock()
        with patch("robot.libraries.BuiltIn.BuiltIn",
                   return_value=mock_builtin):
            await plugin._override_drag_and_drop(
                session=MagicMock(),
                keyword_name="Drag And Drop",
                arguments=["id=onlyone"],
            )
        scroll_calls = [
            c for c in mock_builtin.run_keyword.call_args_list
            if c.args[0] == "Browser.Scroll To Element"
        ]
        assert len(scroll_calls) == 1
        assert scroll_calls[0].args[1] == "id=onlyone"


# ---------------------------------------------------------------------------
# Layer 4: helper predicate
# ---------------------------------------------------------------------------


class TestLooksLikeLocatorPredicate:
    """The ``_looks_like_locator`` helper used by the override."""

    @pytest.mark.parametrize("locator", [
        "id=foo", "css=.x", "xpath=//y", "text=Login",
        "button:text-is('Save')", "id=foo >> nth=0", "#submit",
    ])
    def test_strings_with_content_are_locators(self, locator):
        assert BrowserLibraryPlugin._looks_like_locator(locator) is True

    @pytest.mark.parametrize("non_locator", [
        None, 42, 3.14, [], {}, True, False, "", "   ", "\t\n",
    ])
    def test_non_strings_and_empty_strings_are_not_locators(self, non_locator):
        assert BrowserLibraryPlugin._looks_like_locator(non_locator) is False
