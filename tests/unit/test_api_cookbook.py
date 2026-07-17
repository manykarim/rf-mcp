"""Unit tests: proactive RequestsLibrary cookbook (change: api-cookbook)."""

import asyncio

from robotmcp.utils.requests_guidance import (
    build_requests_cookbook,
    EVALUATE_VAR_RULE,
)
from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter


def _text(payload):
    import json
    return json.dumps(payload).lower()


def test_cookbook_has_key_recipes():
    cb = build_requests_cookbook()
    blob = _text(cb)
    assert "status should be" in blob            # native status assertion
    assert "${resp.json()" in _text(cb)          # response field access
    assert "cookie" in blob and "token=" in blob # auth header
    assert "expected_status=" in blob            # non-2xx assertion
    assert "$resp" in _text(cb)                  # evaluate rule
    assert set(cb) >= {"tips", "warnings", "examples"}


def test_converter_delegates():
    conv = RobotFrameworkNativeConverter()
    cb = conv.get_requests_guidance()
    assert "tips" in cb and any("Status Should Be" in t for t in cb["tips"])


def _call_guidance(**kw):
    from robotmcp.server import get_locator_guidance
    fn = getattr(get_locator_guidance, "fn", get_locator_guidance)  # unwrap @mcp.tool
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(fn(**kw))
    finally:
        loop.close()


def test_get_locator_guidance_requests():
    r = _call_guidance(library="requests")
    assert r["success"] is True
    assert r["library"] == "RequestsLibrary"
    assert "tips" in r


def test_get_locator_guidance_aliases():
    for lib in ("api", "requestslibrary", "Requests", "REST"):
        r = _call_guidance(library=lib)
        assert r.get("library") == "RequestsLibrary", lib
        assert r.get("success") is True, lib


def test_shape_parity_with_browser():
    req = _call_guidance(library="requests")
    br = _call_guidance(library="browser")
    # both carry tips; both resolve a library + success
    assert "tips" in req and "tips" in br
    assert req["success"] is br["success"] is True


def test_unknown_library_still_errors():
    r = _call_guidance(library="nonsense-lib")
    assert r["success"] is False
    assert "error" in r


def test_shared_source_used_in_hints():
    # the reactive hint text shares the canonical Evaluate rule constant
    import robotmcp.utils.hints as h
    assert h.EVALUATE_VAR_RULE == EVALUATE_VAR_RULE
    assert "inside evaluate" in EVALUATE_VAR_RULE.lower()
