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


def test_bridge_e2e_runs_keyword_in_live_context(tmp_path: Path):
    # Ensure src is on path for Robot to import robotmcp.attach
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    # Create a minimal suite that starts the serve loop
    suite = tmp_path / "bridge_suite.robot"
    port = 7321
    token = "secret-token"
    suite.write_text(
        f"""
*** Settings ***
Library    robotmcp.attach.McpAttach    host=127.0.0.1    port={port}    token={token}

*** Test Cases ***
Bridge Processes External Requests
    Log    Starting bridge
    MCP Serve    mode=blocking
    Log    Bridge stopped
""".strip()
    )

    # Run Robot in a background thread using robot.run API
    rc_holder = {"rc": None}

    def _run_robot():
        from robot import run

        rc_holder["rc"] = run(str(suite), outputdir=str(tmp_path / "out"))

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
                break
        except Exception:
            pass
        time.sleep(0.2)

    assert ready, "Bridge did not start in time"

    # Run a BuiltIn keyword in the live context and assign a variable
    resp = _post(
        "127.0.0.1",
        port,
        token,
        "/run_keyword",
        {"name": "Set Test Variable", "args": ["${X}", "123"], "assign_to": None},
    )
    assert resp.get("success"), f"run_keyword failed: {resp}"

    # Log via BuiltIn (no exception means execution worked)
    resp2 = _post("127.0.0.1", port, token, "/run_keyword", {"name": "Log", "args": ["Hello from bridge"]})
    assert resp2.get("success"), f"Log failed: {resp2}"

    # Stop the serve loop and finish the test
    _post("127.0.0.1", port, token, "/stop", {})

    # Wait for robot to finish
    t.join(timeout=20)
    assert rc_holder["rc"] == 0


def test_bridge_import_library_and_variables(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    suite = tmp_path / "bridge_suite.robot"
    port = 7322
    token = "secret-token"
    suite.write_text(
        f"""
*** Settings ***
Library    robotmcp.attach.McpAttach    host=127.0.0.1    port={port}    token={token}

*** Test Cases ***
Bridge Import Library
    MCP Serve    mode=blocking
""".strip()
    )

    rc_holder = {"rc": None}

    def _run_robot():
        from robot import run

        rc_holder["rc"] = run(str(suite), outputdir=str(tmp_path / "out2"))

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
                break
        except Exception:
            pass
        time.sleep(0.2)
    assert ready, "Bridge did not start in time"

    # Import Collections and use a keyword
    r1 = _post("127.0.0.1", port, token, "/import_library", {"name_or_path": "Collections"})
    assert r1.get("success"), r1

    r2 = _post("127.0.0.1", port, token, "/run_keyword", {"name": "Create List", "args": ["a", "b", "c"], "assign_to": ["${L}"]})
    assert r2.get("success"), r2

    # Validate variable by using it in another keyword
    r3 = _post("127.0.0.1", port, token, "/run_keyword", {"name": "Length Should Be", "args": ["${L}", "3"]})
    assert r3.get("success"), r3

    # Set a variable and check via get_variables
    r4 = _post("127.0.0.1", port, token, "/set_variable", {"name": "${X}", "value": 42})
    assert r4.get("success"), r4
    r5 = _post("127.0.0.1", port, token, "/get_variables", {"names": ["${X}"]})
    assert r5.get("success") and r5["result"].get("${X}") == 42, r5

    _post("127.0.0.1", port, token, "/stop", {})
    t.join(timeout=20)
    assert rc_holder["rc"] == 0
