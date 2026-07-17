"""Unit tests: PlatynUI runtime broker
(change: platynui-desktop-safety-isolation, tasks 5.1/5.4).

The broker fixes the proven root cause of "ProviderError ... not available
after shutdown": ui_tree_service used to create + shut down a Runtime per
call. These tests use a fake platynui_native so no display is required.
"""

import sys
import threading
import types

import pytest


@pytest.fixture
def fake_pn(monkeypatch):
    counter = {"n": 0}

    class FakeRuntime:
        def __init__(self):
            counter["n"] += 1
            self.shut = False

        def shutdown(self):
            self.shut = True

    mod = types.ModuleType("platynui_native")
    mod.Runtime = FakeRuntime
    monkeypatch.setitem(sys.modules, "platynui_native", mod)
    import robotmcp.plugins.builtin.platynui_plugin as plugin
    plugin._reset_runtime_broker_for_tests()
    yield plugin, counter
    plugin._reset_runtime_broker_for_tests()


def test_runtime_bound_once_and_reused(fake_pn):
    plugin, counter = fake_pn
    r1 = plugin.get_runtime()
    r2 = plugin.get_runtime()
    assert r1 is r2
    assert counter["n"] == 1
    assert plugin.runtime_state() == "open"


def test_dispose_then_refuse_reinit(fake_pn):
    plugin, _ = fake_pn
    plugin.get_runtime()
    plugin.shutdown_runtime()
    assert plugin.runtime_state() == "disposed"
    with pytest.raises(RuntimeError, match="disposed"):
        plugin.get_runtime()


def test_concurrent_first_use_binds_once(fake_pn):
    plugin, counter = fake_pn
    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(plugin.get_runtime())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counter["n"] == 1
    assert all(r is results[0] for r in results)


def test_ui_tree_reuses_broker_no_new_runtime(fake_pn):
    """get_ui_tree (sync core) must not create/shutdown its own runtime."""
    plugin, counter = fake_pn

    # Make the fake runtime answer the calls _collect_ui_tree_sync makes.
    class _App:
        role = "Application"
        name = "x"
        namespace = "app"

        def children(self):
            return iter([])

        def attribute(self, *_a, **_k):
            return None

    def evaluate(expr):
        return [_App()]

    def desktop_info():
        return {"bounds": {"x": 0, "y": 0, "width": 100, "height": 100}}

    def clear_cache():
        pass

    plugin.get_runtime()  # bind once
    rt = plugin.get_runtime()
    rt.evaluate = evaluate
    rt.desktop_info = desktop_info
    rt.clear_cache = clear_cache

    from robotmcp.components.execution.ui_tree_service import _collect_ui_tree_sync

    before = counter["n"]
    out1 = _collect_ui_tree_sync(None, 2, 10, 50)
    out2 = _collect_ui_tree_sync(None, 2, 10, 50)
    assert out1["success"] is True and out2["success"] is True
    # No additional Runtime() constructions across two calls.
    assert counter["n"] == before
    assert plugin.runtime_state() == "open"
    assert rt.shut is False  # never shut down per-call
