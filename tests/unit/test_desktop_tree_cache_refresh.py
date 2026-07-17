"""Unit tests for the desktop-tree-cache-refresh change.

The PlatynUI runtime caches the desktop accessibility tree, so an app launched
after the last snapshot stays invisible to keyword queries until the cache is
cleared. This change clears it on a desktop launch and refreshes before the
first post-launch tree-resolving keyword.
"""

from __future__ import annotations

import pytest


# ── D3: shared clear_runtime_tree_cache helper ──────────────────────


class TestClearRuntimeTreeCacheHelper:
    def test_clears_when_available(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        class _RT:
            def __init__(self):
                self.cleared = 0

            def clear_cache(self):
                self.cleared += 1

        rt = _RT()
        monkeypatch.setattr(p, "get_runtime", lambda: rt)
        assert p.clear_runtime_tree_cache() is True
        assert rt.cleared == 1

    def test_false_when_no_clear_cache(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        class _RT:  # no clear_cache attr
            pass

        monkeypatch.setattr(p, "get_runtime", lambda: _RT())
        assert p.clear_runtime_tree_cache() is False

    def test_false_when_runtime_none(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        monkeypatch.setattr(p, "get_runtime", lambda: None)
        assert p.clear_runtime_tree_cache() is False

    def test_never_raises_on_disposed_runtime(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        def _boom():
            raise RuntimeError("broker disposed")

        monkeypatch.setattr(p, "get_runtime", _boom)
        # Must swallow the disposed-broker error and return False.
        assert p.clear_runtime_tree_cache() is False

    def test_swallows_clear_cache_exception(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        class _RT:
            def clear_cache(self):
                raise RuntimeError("clear failed")

        monkeypatch.setattr(p, "get_runtime", lambda: _RT())
        assert p.clear_runtime_tree_cache() is False


# ── tree-resolving keyword predicate ────────────────────────────────


class TestTreeResolvingKeyword:
    @pytest.mark.parametrize(
        "keyword",
        [
            "Query",
            "Get Attribute",
            "Set Root",
            "Pointer Click",
            "Keyboard Type",
            "PlatynUI.BareMetal.Query",
        ],
    )
    def test_tree_resolving_true(self, keyword):
        from robotmcp.components.execution.desktop_execution_signals import (
            is_tree_resolving_keyword,
        )

        assert is_tree_resolving_keyword(keyword) is True

    @pytest.mark.parametrize(
        "keyword", ["Start Process", "Run Process", "Log", "Sleep", "Is Process Running"]
    )
    def test_tree_resolving_false(self, keyword):
        from robotmcp.components.execution.desktop_execution_signals import (
            is_tree_resolving_keyword,
        )

        assert is_tree_resolving_keyword(keyword) is False


# ── D1/D2: session flag lifecycle + gate logic ──────────────────────


class TestDesktopTreeDirtyFlag:
    def test_flag_defaults_false(self):
        from robotmcp.models.session_models import ExecutionSession

        assert ExecutionSession(session_id="t").desktop_tree_dirty is False

    def test_launch_gate_sets_dirty(self):
        # Mirrors the D1 condition: a desktop launch keyword marks the tree dirty.
        from robotmcp.components.execution.desktop_execution_signals import (
            is_launch_keyword,
        )
        from robotmcp.models.session_models import ExecutionSession

        s = ExecutionSession(session_id="d1")
        s.configure_from_scenario("desktop calculator", context="desktop")
        if s.is_desktop_session() and is_launch_keyword("Start Process"):
            s.desktop_tree_dirty = True
        assert s.desktop_tree_dirty is True

    def test_first_post_launch_query_consumes_flag(self):
        # Mirrors the D2 gate: dirty + tree-resolving keyword -> refresh once,
        # flag cleared so the next query does NOT re-refresh.
        from robotmcp.components.execution.desktop_execution_signals import (
            is_tree_resolving_keyword,
        )
        from robotmcp.models.session_models import ExecutionSession

        s = ExecutionSession(session_id="d2")
        s.configure_from_scenario("desktop calculator", context="desktop")
        s.desktop_tree_dirty = True

        refreshes = 0
        for kw in ["Query", "Get Attribute", "Pointer Click"]:
            if s.desktop_tree_dirty and is_tree_resolving_keyword(kw):
                refreshes += 1
                s.desktop_tree_dirty = False
        # Only the FIRST tree-resolving keyword refreshed.
        assert refreshes == 1
        assert s.desktop_tree_dirty is False

    def test_steady_state_no_refresh(self):
        from robotmcp.components.execution.desktop_execution_signals import (
            is_tree_resolving_keyword,
        )
        from robotmcp.models.session_models import ExecutionSession

        s = ExecutionSession(session_id="d3")
        s.configure_from_scenario("desktop calculator", context="desktop")
        # No launch -> flag stays False -> no refresh for any query.
        refreshes = sum(
            1
            for kw in ["Query", "Get Attribute", "Query"]
            if s.desktop_tree_dirty and is_tree_resolving_keyword(kw)
        )
        assert refreshes == 0

    def test_non_launch_desktop_keyword_does_not_consume_before_query(self):
        # A Process keyword (Is Process Running) after launch must NOT consume
        # the dirty flag — only a tree-resolving keyword does.
        from robotmcp.components.execution.desktop_execution_signals import (
            is_tree_resolving_keyword,
        )
        from robotmcp.models.session_models import ExecutionSession

        s = ExecutionSession(session_id="d4")
        s.configure_from_scenario("desktop calculator", context="desktop")
        s.desktop_tree_dirty = True
        # Non-tree-resolving keyword runs first — flag survives.
        kw = "Is Process Running"
        if s.desktop_tree_dirty and is_tree_resolving_keyword(kw):
            s.desktop_tree_dirty = False
        assert s.desktop_tree_dirty is True  # still dirty for the real Query


# ── D4: guidance documents the tree-refresh recovery ────────────────


class TestTreeFreshnessGuidance:
    def _guidance(self):
        from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

        return RobotFrameworkNativeConverter().get_platynui_locator_guidance()

    def test_tree_freshness_section_present(self):
        g = self._guidance()
        assert "tree_freshness" in g
        joined = (g["tree_freshness"]["description"] + " " + " ".join(g["tree_freshness"]["rules"])).lower()
        assert "cache" in joined or "cached" in joined

    def test_documents_auto_refresh_after_launch(self):
        rules = " ".join(self._guidance()["tree_freshness"]["rules"]).lower()
        assert "after a desktop launch" in rules or "after a launch" in rules

    def test_documents_ui_tree_force_refresh(self):
        rules = " ".join(self._guidance()["tree_freshness"]["rules"])
        assert "get_session_state" in rules
        assert "ui_tree" in rules

    def test_discourages_coordinate_ocr_first_resort(self):
        rules = " ".join(self._guidance()["tree_freshness"]["rules"]).lower()
        assert "coordinate" in rules or "ocr" in rules
