"""Desktop actionable-controls view (change: desktop-actionable-controls).

ui_tree is depth-3-bounded and GTK controls nest 4-6 deep, forcing per-element
Query probing. The actionable_controls view returns a FLAT, AUT-scoped,
budget-bounded list of interactive controls with ready descriptors — the desktop
analog of the web P8 actionable_elements view.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from robotmcp.components.execution import ui_tree_service as u


class FakeNode:
    """Minimal stand-in for a PlatynUI UiNode (role/name/children()/attribute())."""

    def __init__(self, role="", name="", children=None, attrs=None):
        self.role = role
        self.name = name
        self.namespace = None
        self._children = children or []
        self._attrs = attrs or {}

    def children(self):
        return list(self._children)

    def attribute(self, key):
        return self._attrs.get(key)


def _nested(depth, leaf):
    node = leaf
    for _ in range(depth):
        node = FakeNode("control:Panel", "", children=[node])
    return node


class TestWalk:
    def test_deeply_nested_control_appears_flat(self):
        button = FakeNode("control:Button", "OK", attrs={"IsEnabled": True, "IsVisible": True})
        app = FakeNode("app", "calc", children=[_nested(4, button)])  # button at depth 5
        r = u.walk_actionable_controls(app, "calc")
        assert r["control_count"] == 1
        c = r["controls"][0]
        assert c["role"] == "Button" and c["name"] == "OK" and c["depth"] == 5
        assert c["enabled"] is True and c["visible"] is True

    def test_descriptor_shape_and_index_disambiguation(self):
        app = FakeNode("app", "calc", children=[
            FakeNode("control:Button", "7"),
            FakeNode("control:Button", "7"),
            FakeNode("control:Text", ""),  # nameless
        ])
        d = [c["descriptor"] for c in u.walk_actionable_controls(app, "calc")["controls"]]
        assert d[0] == "/app:*[@Name='calc']//control:Button[@Name='7']"
        assert d[1] == "(/app:*[@Name='calc']//control:Button[@Name='7'])[2]"
        assert d[2] == "/app:*[@Name='calc']//control:Text"

    def test_role_filter_default_excludes_label(self):
        app = FakeNode("app", "calc", children=[
            FakeNode("control:Button", "B"),
            FakeNode("control:Label", "L"),
            FakeNode("control:Text", "T"),
        ])
        roles = {c["role"] for c in u.walk_actionable_controls(app, "calc")["controls"]}
        assert roles == {"Button", "Text"}  # Label not interactive by default

    def test_custom_role_filter(self):
        app = FakeNode("app", "calc", children=[
            FakeNode("control:Button", "B"), FakeNode("control:Label", "L"),
        ])
        r = u.walk_actionable_controls(app, "calc", roles=frozenset({"label"}))
        assert [c["role"] for c in r["controls"]] == ["Label"]

    def test_element_budget_truncation(self):
        app = FakeNode("app", "calc", children=[FakeNode("control:Button", f"b{i}") for i in range(10)])
        r = u.walk_actionable_controls(app, "calc", max_elements=3)
        assert r["control_count"] == 3 and r["truncated"]["reason"] == "max_elements"

    def test_node_budget_truncation(self):
        app = FakeNode("app", "calc", children=[FakeNode("control:Button", f"b{i}") for i in range(10)])
        r = u.walk_actionable_controls(app, "calc", max_nodes=2)
        assert r["truncated"]["reason"] == "max_nodes"

    def test_per_node_provider_error_is_skipped_not_raised(self):
        class Boom(FakeNode):
            def children(self):
                raise RuntimeError("provider hiccup")
        app = FakeNode("app", "calc", children=[Boom("control:Button", "OK")])
        r = u.walk_actionable_controls(app, "calc")  # must not raise
        assert r["success"] is True and r["control_count"] == 1

    def test_only_walks_given_subtree(self):
        # The walk uses node.children() exclusively — no runtime // expression,
        # never leaves the anchor subtree (spec req 2/2.3).
        visited = []

        class Tracking(FakeNode):
            def children(self):
                visited.append(self.name)
                return super().children()

        app = Tracking("app", "calc", children=[Tracking("control:Button", "OK")])
        u.walk_actionable_controls(app, "calc")
        assert visited == ["calc", "OK"]  # only the anchor subtree


class TestAnchorResolution:
    def test_multi_app_without_filter_refuses(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        class FakeRuntime:
            def evaluate(self, xpath):
                return [FakeNode("app", "calc"), FakeNode("app", "mousepad")]

        monkeypatch.setattr(p, "get_runtime", lambda: FakeRuntime())
        monkeypatch.setattr(p, "clear_runtime_tree_cache", lambda: None, raising=False)
        r = u._collect_actionable_controls_sync(
            None, roles=u._INTERACTIVE_ROLES, max_nodes=100, max_elements=80, time_budget_s=5
        )
        assert r.get("requires_app_filter") is True
        assert set(r["applications"]) == {"calc", "mousepad"}
        assert "controls" not in r  # no walk happened

    def test_filter_selects_single_anchor(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        class FakeRuntime:
            def evaluate(self, xpath):
                return [
                    FakeNode("app", "calc", children=[FakeNode("control:Button", "7")]),
                    FakeNode("app", "mousepad"),
                ]

        monkeypatch.setattr(p, "get_runtime", lambda: FakeRuntime())
        monkeypatch.setattr(p, "clear_runtime_tree_cache", lambda: None, raising=False)
        r = u._collect_actionable_controls_sync(
            ["calc"], roles=u._INTERACTIVE_ROLES, max_nodes=100, max_elements=80, time_budget_s=5
        )
        assert r["application"] == "calc" and r["control_count"] == 1


class TestNonDesktopRejection:
    def test_web_session_rejected(self):
        web = types.SimpleNamespace(is_desktop_session=lambda: False)
        r = asyncio.run(u.get_actionable_controls(web))
        assert r["success"] is False and "desktop" in r["error"].lower()
