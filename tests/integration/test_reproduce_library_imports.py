"""Quick reproducer for Debug Bridge library imports issue.

This test verifies that keywords from libraries imported in the test suite
Settings section are available through the Debug Bridge.
"""
import sys
import time
import threading
from pathlib import Path

import json
import http.client


def _post(host: str, port: int, token: str, path: str, payload: dict) -> dict:
    conn = http.client.HTTPConnection(host, port, timeout=10)
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "X-MCP-Token": token,
    }
    conn.request("POST", path, body, headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {"success": False, "error": f"invalid response: {data!r}"}


def test_library_imports_available_through_bridge(tmp_path: Path):
    """Test that libraries imported in Settings are usable via the bridge.

    This reproduces the issue from docs/issues/debug_bridge_imports.txt
    where libraries like Browser and String imported in Settings are
    not available when using keywords through the Debug Bridge.
    """
    # Ensure src is on path for Robot to import robotmcp.attach
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    # Create a suite that imports BOTH McpAttach AND String library
    suite = tmp_path / "bridge_with_libraries.robot"
    port = 7324
    token = "secret-token"
    suite.write_text(
        f"""
*** Settings ***
Library    robotmcp.attach.McpAttach    host=127.0.0.1    port={port}    token={token}
Library    String
Library    Collections

*** Test Cases ***
Bridge With Library Imports
    Log    Starting bridge with String and Collections already imported
    MCP Serve    mode=blocking
    Log    Bridge stopped
""".strip()
    )

    # Run Robot in a background thread
    rc_holder = {"rc": None, "error": None}

    def _run_robot():
        from robot import run
        try:
            rc_holder["rc"] = run(str(suite), outputdir=str(tmp_path / "out"))
        except Exception as e:
            rc_holder["error"] = str(e)

    t = threading.Thread(target=_run_robot, daemon=True)
    t.start()

    # Wait for server to be up
    deadline = time.time() + 10
    ready = False
    while time.time() < deadline:
        try:
            resp = _post("127.0.0.1", port, token, "/diagnostics", {})
            if resp.get("success"):
                ready = True
                print(f"Diagnostics response: {resp}")
                break
        except Exception:
            pass
        time.sleep(0.2)

    assert ready, "Bridge did not start in time"

    # First check what libraries are visible in the context
    diag = _post("127.0.0.1", port, token, "/diagnostics", {})
    print(f"Libraries visible in context: {diag.get('result', {}).get('libraries', [])}")

    # List all keywords to see what's available
    kw_list = _post("127.0.0.1", port, token, "/list_keywords", {})
    print(f"Keyword list response success: {kw_list.get('success')}")
    if kw_list.get("result"):
        for lib in kw_list["result"]:
            print(f"  Library '{lib.get('library')}': {len(lib.get('keywords', []))} keywords")

    # TEST 1: Try to use String library keyword (should work since imported in Settings)
    # Using "Convert To Upper Case" from String library
    resp1 = _post(
        "127.0.0.1",
        port,
        token,
        "/run_keyword",
        {"name": "Convert To Upper Case", "args": ["hello world"]},
    )
    print(f"String library keyword result: {resp1}")

    # TEST 2: Try Collections library keyword
    resp2 = _post(
        "127.0.0.1",
        port,
        token,
        "/run_keyword",
        {"name": "Create List", "args": ["a", "b", "c"]},
    )
    print(f"Collections library keyword result: {resp2}")

    # TEST 3: Try BuiltIn keyword (should always work)
    resp3 = _post(
        "127.0.0.1",
        port,
        token,
        "/run_keyword",
        {"name": "Log", "args": ["Test from bridge"]},
    )
    print(f"BuiltIn keyword result: {resp3}")

    # Stop the serve loop
    _post("127.0.0.1", port, token, "/stop", {})

    # Wait for robot to finish
    t.join(timeout=20)

    # Now check assertions
    assert resp3.get("success"), f"BuiltIn Log failed: {resp3}"
    assert resp1.get("success"), f"String Convert To Upper Case failed (library imports issue!): {resp1}"
    assert resp2.get("success"), f"Collections Create List failed: {resp2}"

    # Verify the result
    if resp1.get("success"):
        assert resp1.get("result") == "HELLO WORLD", f"Expected 'HELLO WORLD', got {resp1.get('result')}"

    assert rc_holder["rc"] == 0, f"Robot Framework run failed: {rc_holder}"
