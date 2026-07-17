"""Unit tests: visual-inspection-guidance (change)."""
import asyncio, os, tempfile
import pytest


def _call(fn_holder, **kw):
    from robotmcp import server
    fn = getattr(getattr(server, fn_holder), "fn", getattr(server, fn_holder))
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(fn(**kw))
    finally:
        loop.close()


# ── §2 guidance topic ──────────────────────────────────────────────────────
def test_visual_guidance_topic():
    r = _call("get_locator_guidance", library="visual")
    assert r["success"] is True and r["library"] == "visual"
    blob = str(r).lower()
    assert "canvas" in blob and "overlap" in blob            # vision-only cases
    assert "get text" in blob                                # dual read-back
    assert "multimodal" in blob and "file access" in blob    # caveats


def test_visual_guidance_aliases():
    for lib in ("screenshot", "vision", "image", "VISUAL"):
        r = _call("get_locator_guidance", library=lib)
        assert r.get("library") == "visual", lib


# ── §1 success-hint function ───────────────────────────────────────────────
def test_visual_validation_hint():
    from robotmcp.components.execution.desktop_execution_signals import visual_validation_hint
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False); f.write(b"x"); f.close()
    try:
        h = visual_validation_hint("Take Screenshot", [f"filename={f.name}"], f.name)
        assert h and h["type"] == "visual_validation" and h["screenshot_path"] == f.name
        assert visual_validation_hint("Take Screenshot", ["filename=/tmp/nope_zzz.png"], "/tmp/nope_zzz.png") is None
        assert visual_validation_hint("Click", ["x"], None) is None   # non-screenshot
    finally:
        os.unlink(f.name)


# ── §3 visual_check tool + §5.4 mode gating ────────────────────────────────
class _FakeSess:  pass

def _patch_engine(monkeypatch, tmp_png):
    from robotmcp import server
    monkeypatch.setattr(server.execution_engine.session_manager, "get_session", lambda sid: _FakeSess())
    async def _exec(kw, args, sid, **k):
        # emulate a successful screenshot capture writing the file
        return {"success": kw == "Take Screenshot", "screenshot_path": tmp_png}
    monkeypatch.setattr(server.execution_engine, "execute_step", _exec)


def test_visual_check_default_returns_path_not_image(monkeypatch):
    monkeypatch.setenv("ROBOTMCP_SCREENSHOT_MODE", "file")
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False); f.write(b"x" * 50); f.close()
    try:
        _patch_engine(monkeypatch, f.name)
        r = _call("visual_check", session_id="s")
        assert isinstance(r, dict) and r["success"] is True
        assert r["screenshot_path"] == f.name and r["size_bytes"] == 50
        assert "visual_hint" in r
    finally:
        os.unlink(f.name)


def test_visual_check_image_gated_by_mode(monkeypatch):
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False); f.write(b"x" * 20); f.close()
    try:
        _patch_engine(monkeypatch, f.name)
        # mode=file: even return_image=True stays text-only (protect text-only drivers)
        monkeypatch.setenv("ROBOTMCP_SCREENSHOT_MODE", "file")
        r = _call("visual_check", session_id="s", return_image=True)
        assert isinstance(r, dict) and "screenshot_path" in r
        # mode=image + return_image: returns [dict, Image] list
        monkeypatch.setenv("ROBOTMCP_SCREENSHOT_MODE", "image")
        r2 = _call("visual_check", session_id="s", return_image=True)
        assert isinstance(r2, list) and any(type(x).__name__ == "Image" for x in r2)
    finally:
        os.unlink(f.name)


def test_visual_check_capture_failure_degrades(monkeypatch):
    from robotmcp import server
    monkeypatch.setattr(server.execution_engine.session_manager, "get_session", lambda sid: _FakeSess())
    async def _exec(kw, args, sid, **k):
        return {"success": False, "error": "no browser"}
    monkeypatch.setattr(server.execution_engine, "execute_step", _exec)
    r = _call("visual_check", session_id="s")
    assert r["success"] is False and "hint" in r


def test_mode_reader():
    from robotmcp.server import _screenshot_mode
    import os as o
    for v, exp in [("file","file"),("image","image"),("auto","auto"),("bogus","file")]:
        o.environ["ROBOTMCP_SCREENSHOT_MODE"] = v
        assert _screenshot_mode() == exp
    o.environ.pop("ROBOTMCP_SCREENSHOT_MODE", None)
