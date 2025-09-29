import types
import os
import sys
from pathlib import Path


def get_server_module():
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from robotmcp import server as mcp_server

    return mcp_server


class DummyClient:
    def __init__(self, ok: bool):
        self._ok = ok

    def diagnostics(self):
        return {"success": self._ok}


def test_compute_effective_use_context_auto_reachable(monkeypatch):
    server = get_server_module()
    monkeypatch.setenv("ROBOTMCP_ATTACH_DEFAULT", "auto")
    monkeypatch.delenv("ROBOTMCP_ATTACH_STRICT", raising=False)
    eff, mode, strict = server._compute_effective_use_context(
        use_context=None, client=DummyClient(True), keyword="KW"
    )
    assert eff is True and mode == "auto" and strict is False


def test_compute_effective_use_context_auto_unreachable(monkeypatch):
    server = get_server_module()
    monkeypatch.setenv("ROBOTMCP_ATTACH_DEFAULT", "auto")
    eff, mode, strict = server._compute_effective_use_context(
        use_context=None, client=DummyClient(False), keyword="KW"
    )
    assert eff is False and mode == "auto"


def test_compute_effective_use_context_force_overrides_false(monkeypatch):
    server = get_server_module()
    monkeypatch.setenv("ROBOTMCP_ATTACH_DEFAULT", "force")
    eff, mode, strict = server._compute_effective_use_context(
        use_context=False, client=DummyClient(False), keyword="KW"
    )
    assert eff is True and mode == "force"


def test_compute_effective_use_context_off(monkeypatch):
    server = get_server_module()
    monkeypatch.setenv("ROBOTMCP_ATTACH_DEFAULT", "off")
    eff, mode, strict = server._compute_effective_use_context(
        use_context=None, client=DummyClient(True), keyword="KW"
    )
    assert eff is False and mode == "off"

