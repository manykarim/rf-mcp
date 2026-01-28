"""
Test to reproduce Debug Bridge library imports issue.

Problem: When using Debug Bridge (McpAttach) from within a Test Suite with Library
imports (like Browser, String), those library imports are not available in the
session of the Debug Bridge.

This test verifies that libraries imported in the Robot Framework test suite's
Settings section are available when executing keywords through the Debug Bridge.
"""

import sys
import time
import threading
from pathlib import Path
import json
import http.client


def _post(host: str, port: int, token: str, path: str, payload: dict) -> dict:
    """Send a POST request to the Debug Bridge HTTP server."""
    try:
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
    except Exception as e:
        return {"success": False, "error": f"connection error: {e}"}


def test_pre_imported_string_library_available_via_bridge(tmp_path: Path):
    """
    Test that String library imported in Settings is available through Debug Bridge.

    This test reproduces the issue where libraries imported in the test suite's
    Settings section are not accessible when executing keywords via the Debug Bridge.

    Expected behavior: Keywords from pre-imported libraries (like String) should
    be callable through the bridge without needing to import them again.
    """
    # Ensure src is on path for Robot to import robotmcp.attach
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    # Create a test suite that imports BOTH McpAttach AND String library
    suite = tmp_path / "library_import_suite.robot"
    port = 7323
    token = "secret-token"
    suite.write_text(
        f"""
*** Settings ***
Library    String
Library    robotmcp.attach.McpAttach    host=127.0.0.1    port={port}    token={token}

*** Test Cases ***
Test Pre-Imported Library Via Bridge
    # First verify String library works directly in Robot
    ${{result}}=    Convert To Upper Case    hello world
    Should Be Equal    ${{result}}    HELLO WORLD
    Log    String library works directly, starting bridge...
    MCP Serve    mode=blocking
    Log    Bridge stopped
""".strip()
    )

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

    # First, check diagnostics to see what libraries are available
    diag_resp = _post("127.0.0.1", port, token, "/diagnostics", {})
    print(f"DEBUG: Diagnostics response: {diag_resp}")
    if diag_resp.get("success"):
        libraries = diag_resp.get("result", {}).get("libraries", [])
        print(f"DEBUG: Available libraries in session: {libraries}")
        # Verify String library is in the list
        assert "String" in libraries, (
            f"BUG: String library not found in session libraries. "
            f"Available: {libraries}"
        )

    # Also check list_keywords to see what keywords are discoverable
    kw_resp = _post("127.0.0.1", port, token, "/list_keywords", {})
    print(f"DEBUG: list_keywords response success: {kw_resp.get('success')}")
    if kw_resp.get("success"):
        result = kw_resp.get("result", [])
        string_lib = next((lib for lib in result if lib.get("library") == "String"), None)
        if string_lib:
            print(f"DEBUG: String library keywords: {string_lib.get('keywords', [])[:5]}...")
        else:
            print(f"DEBUG: String library NOT found in list_keywords result")
            print(f"DEBUG: Available libraries: {[lib.get('library') for lib in result]}")

    # Try to use String library keyword through the bridge
    # This is the key test - String library was imported in Settings,
    # so it should be available without explicit import via bridge
    resp = _post(
        "127.0.0.1",
        port,
        token,
        "/run_keyword",
        {"name": "Convert To Upper Case", "args": ["test string"], "assign_to": None},
    )

    # Check if the keyword executed successfully
    # If the bug exists, this will fail because String library is not available
    keyword_success = resp.get("success", False)

    if not keyword_success:
        # Log the error for debugging
        error_msg = resp.get("error", "Unknown error")
        print(f"DEBUG: String library keyword failed with error: {error_msg}")
        print(f"DEBUG: Full response: {resp}")

    # Also try to get the result value if it succeeded
    if keyword_success:
        result = resp.get("result")
        print(f"DEBUG: Convert To Upper Case result: {result}")
        # Verify the result is correct
        assert result == "TEST STRING", f"Expected 'TEST STRING', got {result}"

    # Stop the serve loop
    _post("127.0.0.1", port, token, "/stop", {})

    # Wait for robot to finish
    t.join(timeout=20)

    # Assert the keyword succeeded - this is where the bug manifests
    assert keyword_success, (
        f"Pre-imported String library keyword 'Convert To Upper Case' failed via bridge. "
        f"Error: {resp.get('error', 'Unknown')}. "
        f"This confirms the bug: libraries imported in Settings are not available in Debug Bridge session."
    )

    assert rc_holder["rc"] == 0, f"Robot Framework test failed with rc={rc_holder['rc']}"


def test_pre_imported_collections_library_available_via_bridge(tmp_path: Path):
    """
    Test that Collections library imported in Settings is available through Debug Bridge.

    Similar to the String library test, but uses Collections to verify the issue
    is not specific to a single library.
    """
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    suite = tmp_path / "collections_import_suite.robot"
    port = 7326  # Use different port to avoid conflicts
    token = "secret-token"
    suite.write_text(
        f"""
*** Settings ***
Library    Collections
Library    robotmcp.attach.McpAttach    host=127.0.0.1    port={port}    token={token}

*** Test Cases ***
Test Pre-Imported Collections Via Bridge
    # Verify Collections works directly
    @{{mylist}}=    Create List    a    b    c
    Length Should Be    ${{mylist}}    3
    Log    Collections library works directly, starting bridge...
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

    # Try to use Collections library keyword through the bridge
    # WITHOUT explicitly importing it via /import_library
    resp = _post(
        "127.0.0.1",
        port,
        token,
        "/run_keyword",
        {"name": "Create List", "args": ["x", "y", "z"], "assign_to": None},
    )

    keyword_success = resp.get("success", False)

    if not keyword_success:
        error_msg = resp.get("error", "Unknown error")
        print(f"DEBUG: Collections library keyword failed with error: {error_msg}")

    # Stop the serve loop
    _post("127.0.0.1", port, token, "/stop", {})

    # Wait for robot to finish
    t.join(timeout=20)

    assert keyword_success, (
        f"Pre-imported Collections library keyword 'Create List' failed via bridge. "
        f"Error: {resp.get('error', 'Unknown')}. "
        f"This confirms the bug: libraries imported in Settings are not available in Debug Bridge session."
    )

    assert rc_holder["rc"] == 0


def test_compare_dynamic_vs_preimported_library(tmp_path: Path):
    """
    Compare behavior: dynamically imported library vs pre-imported library.

    This test demonstrates the difference between:
    1. Library imported in Robot Settings (pre-imported) - may not work
    2. Library imported via bridge /import_library endpoint - should work

    This helps confirm whether the issue is specifically with pre-imported libraries.
    """
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    suite = tmp_path / "compare_import_suite.robot"
    port = 7327  # Use different port to avoid conflicts
    token = "secret-token"
    # Only import String in Settings, NOT Collections
    suite.write_text(
        f"""
*** Settings ***
Library    String
Library    robotmcp.attach.McpAttach    host=127.0.0.1    port={port}    token={token}

*** Test Cases ***
Compare Import Methods
    ${{upper}}=    Convert To Upper Case    pre-import test
    Log    String library (pre-imported) works directly: ${{upper}}
    MCP Serve    mode=blocking
""".strip()
    )

    rc_holder = {"rc": None}

    def _run_robot():
        from robot import run

        rc_holder["rc"] = run(str(suite), outputdir=str(tmp_path / "out3"))

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

    # Test 1: Try pre-imported String library keyword (may fail due to bug)
    resp_preimport = _post(
        "127.0.0.1",
        port,
        token,
        "/run_keyword",
        {"name": "Convert To Upper Case", "args": ["via bridge"], "assign_to": None},
    )
    preimport_success = resp_preimport.get("success", False)

    # Test 2: Dynamically import Collections and use it (should work)
    import_resp = _post(
        "127.0.0.1", port, token, "/import_library", {"name_or_path": "Collections"}
    )
    dynamic_import_success = import_resp.get("success", False)

    resp_dynamic = _post(
        "127.0.0.1",
        port,
        token,
        "/run_keyword",
        {"name": "Create List", "args": ["1", "2", "3"], "assign_to": None},
    )
    dynamic_keyword_success = resp_dynamic.get("success", False)

    # Stop the serve loop
    _post("127.0.0.1", port, token, "/stop", {})

    # Wait for robot to finish
    t.join(timeout=20)

    # Report findings
    print(f"\n=== Library Import Comparison ===")
    print(f"Pre-imported (String) keyword success: {preimport_success}")
    print(f"Dynamic import (Collections) success: {dynamic_import_success}")
    print(f"Dynamic library keyword success: {dynamic_keyword_success}")

    if not preimport_success and dynamic_keyword_success:
        print(
            "\nBUG CONFIRMED: Pre-imported libraries are NOT available, "
            "but dynamically imported libraries work fine."
        )

    # The test passes if we can demonstrate the comparison
    # The key assertion is that dynamic import works
    assert dynamic_import_success, f"Dynamic import failed: {import_resp}"
    assert dynamic_keyword_success, f"Dynamic library keyword failed: {resp_dynamic}"

    # This assertion will fail if the bug exists, helping to reproduce the issue
    assert preimport_success, (
        f"BUG REPRODUCED: Pre-imported String library keyword failed via bridge "
        f"while dynamically imported Collections worked. "
        f"Pre-import error: {resp_preimport.get('error', 'Unknown')}"
    )

    assert rc_holder["rc"] == 0


def test_library_visibility_diagnostics(tmp_path: Path):
    """
    Test to diagnose library visibility in the Debug Bridge session.

    This test provides detailed diagnostics about which libraries are visible
    in the Debug Bridge session to help understand the scope of the issue.
    """
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    suite = tmp_path / "diagnostics_suite.robot"
    port = 7328
    token = "secret-token"
    # Import multiple libraries in a specific order like the original issue
    suite.write_text(
        f"""
*** Settings ***
Library    robotmcp.attach.McpAttach    token={token}    port={port}
Library    String
Library    Collections

*** Variables ***
${{DEBUG_TOKEN}}    {token}

*** Test Cases ***
Diagnose Library Visibility
    # Use libraries before starting bridge
    ${{upper}}=    Convert To Upper Case    test
    @{{list}}=    Create List    a    b    c
    Log    Libraries work directly
    MCP Serve    port={port}    token=${{DEBUG_TOKEN}}    mode=blocking
    [Teardown]    Log    Done
""".strip()
    )

    rc_holder = {"rc": None}
    diagnostics_info = {"libraries": None, "keywords": None}

    def _run_robot():
        from robot import run

        rc_holder["rc"] = run(str(suite), outputdir=str(tmp_path / "out4"))

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

    # Get detailed diagnostics
    diag_resp = _post("127.0.0.1", port, token, "/diagnostics", {})
    assert diag_resp.get("success"), f"Diagnostics failed: {diag_resp}"
    diagnostics_info["libraries"] = diag_resp.get("result", {}).get("libraries", [])

    # Get list of keywords
    kw_resp = _post("127.0.0.1", port, token, "/list_keywords", {})
    if kw_resp.get("success"):
        diagnostics_info["keywords"] = kw_resp.get("result", [])

    # Test each library's keywords
    test_results = {}

    # Test String library
    resp1 = _post(
        "127.0.0.1", port, token, "/run_keyword",
        {"name": "Convert To Upper Case", "args": ["diagnostics"], "assign_to": None}
    )
    test_results["String.Convert To Upper Case"] = resp1

    # Test Collections library
    resp2 = _post(
        "127.0.0.1", port, token, "/run_keyword",
        {"name": "Create List", "args": ["1", "2"], "assign_to": None}
    )
    test_results["Collections.Create List"] = resp2

    # Test BuiltIn library (should always work)
    resp3 = _post(
        "127.0.0.1", port, token, "/run_keyword",
        {"name": "Log", "args": ["Test log via bridge"], "assign_to": None}
    )
    test_results["BuiltIn.Log"] = resp3

    # Stop the serve loop
    _post("127.0.0.1", port, token, "/stop", {})

    # Wait for robot to finish
    t.join(timeout=20)

    # Output diagnostics for analysis
    print("\n" + "=" * 60)
    print("LIBRARY VISIBILITY DIAGNOSTICS")
    print("=" * 60)
    print(f"\nLibraries visible in session: {diagnostics_info['libraries']}")
    print(f"\nKeyword results:")
    for kw_name, result in test_results.items():
        status = "OK" if result.get("success") else "FAILED"
        print(f"  {kw_name}: {status}")
        if not result.get("success"):
            print(f"    Error: {result.get('error', 'Unknown')}")
    print("=" * 60)

    # Assertions - all should pass
    assert "String" in diagnostics_info["libraries"], (
        f"String library not visible. Libraries: {diagnostics_info['libraries']}"
    )
    assert "Collections" in diagnostics_info["libraries"], (
        f"Collections library not visible. Libraries: {diagnostics_info['libraries']}"
    )
    for kw_name, result in test_results.items():
        assert result.get("success"), f"{kw_name} failed: {result}"

    assert rc_holder["rc"] == 0


def test_issue_reproduction_exact_scenario(tmp_path: Path):
    """
    Attempt to reproduce the exact scenario from the reported issue.

    From docs/issues/debug_bridge_imports.txt:
    - Library robotmcp.attach.McpAttach with token
    - Library Browser
    - Library String
    - MCP Serve with specific parameters

    Since Browser requires Playwright and is complex, we simulate with
    a library that might exhibit similar behavior (imported but not available).
    """
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    suite = tmp_path / "issue_reproduction.robot"
    port = 7329
    token = "change-me"  # As in the original issue

    # Replicate the exact structure from the issue, but without Browser
    # (Browser library requires playwright installation)
    suite.write_text(
        f"""
*** Settings ***
Library    robotmcp.attach.McpAttach    token=${{DEBUG_TOKEN}}
Library    String

*** Variables ***
${{DEBUG_TOKEN}}    {token}

*** Test Cases ***
Serve From Debugger
    MCP Serve    port={port}    token=${{DEBUG_TOKEN}}    mode=blocking    poll_ms=100
    [Teardown]    MCP Stop
""".strip()
    )

    rc_holder = {"rc": None}

    def _run_robot():
        from robot import run

        rc_holder["rc"] = run(str(suite), outputdir=str(tmp_path / "out5"))

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

    # Check diagnostics
    diag_resp = _post("127.0.0.1", port, token, "/diagnostics", {})
    libraries = diag_resp.get("result", {}).get("libraries", [])
    print(f"\nIssue reproduction - Libraries visible: {libraries}")

    # Try String library keyword
    resp = _post(
        "127.0.0.1", port, token, "/run_keyword",
        {"name": "Convert To Upper Case", "args": ["issue test"], "assign_to": None}
    )

    if not resp.get("success"):
        print(f"ISSUE REPRODUCED: String keyword failed - {resp.get('error')}")
        print(f"Available libraries: {libraries}")

    # Stop the serve loop
    _post("127.0.0.1", port, token, "/stop", {})

    # Wait for robot to finish
    t.join(timeout=20)

    # If String library is not visible, the bug is reproduced
    assert "String" in libraries, (
        f"BUG REPRODUCED: String library not visible in session. "
        f"Available: {libraries}"
    )

    assert resp.get("success"), (
        f"BUG REPRODUCED: String keyword 'Convert To Upper Case' failed. "
        f"Error: {resp.get('error', 'Unknown')}. "
        f"Libraries visible: {libraries}"
    )

    # If we get here, the bug is NOT reproduced with String library
    # The issue might be specific to Browser library
    assert rc_holder["rc"] == 0


def test_browser_library_import_via_bridge(tmp_path: Path):
    """
    Test specifically for Browser library which was mentioned in the original issue.

    The Browser library has a different initialization pattern than standard
    Robot Framework libraries and may behave differently with the Debug Bridge.

    Note: This test may fail if Browser/Playwright is not properly configured.
    """
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    suite = tmp_path / "browser_import_suite.robot"
    port = 7330
    token = "secret-token"

    # Test with Browser library as in the original issue
    suite.write_text(
        f"""
*** Settings ***
Library    robotmcp.attach.McpAttach    token=${{DEBUG_TOKEN}}    port={port}
Library    Browser
Library    String

*** Variables ***
${{DEBUG_TOKEN}}    {token}

*** Test Cases ***
Browser Library Via Bridge
    # Don't actually launch a browser, just start the bridge
    # and test if Browser library keywords are visible
    MCP Serve    mode=blocking    poll_ms=100
    [Teardown]    MCP Stop
""".strip()
    )

    rc_holder = {"rc": None}

    def _run_robot():
        from robot import run

        rc_holder["rc"] = run(str(suite), outputdir=str(tmp_path / "out_browser"))

    t = threading.Thread(target=_run_robot, daemon=True)
    t.start()

    # Wait for server to be up
    deadline = time.time() + 15  # Browser library takes longer to load
    ready = False
    while time.time() < deadline:
        try:
            resp = _post("127.0.0.1", port, token, "/diagnostics", {})
            if resp.get("success"):
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.3)

    if not ready:
        # Try to stop gracefully and report
        try:
            _post("127.0.0.1", port, token, "/stop", {})
        except Exception:
            pass
        t.join(timeout=5)
        assert False, "Bridge did not start in time (Browser library may have failed to load)"

    # Check diagnostics to see what libraries are visible
    diag_resp = _post("127.0.0.1", port, token, "/diagnostics", {})
    libraries = diag_resp.get("result", {}).get("libraries", [])
    print(f"\nBrowser test - Libraries visible: {libraries}")

    # Check if Browser library is visible
    browser_visible = "Browser" in libraries
    string_visible = "String" in libraries

    print(f"Browser library visible: {browser_visible}")
    print(f"String library visible: {string_visible}")

    # Try String library keyword (should work based on previous tests)
    string_resp = _post(
        "127.0.0.1", port, token, "/run_keyword",
        {"name": "Convert To Upper Case", "args": ["browser test"], "assign_to": None}
    )

    # Try a Browser library keyword that doesn't require a browser
    # "Get Browser Catalog" just returns the catalog of available browsers
    browser_resp = _post(
        "127.0.0.1", port, token, "/run_keyword",
        {"name": "Get Browser Catalog", "args": [], "assign_to": None}
    )

    # Stop the serve loop
    _post("127.0.0.1", port, token, "/stop", {})

    # Wait for robot to finish
    t.join(timeout=20)

    # Report results
    print("\n" + "=" * 60)
    print("BROWSER LIBRARY TEST RESULTS")
    print("=" * 60)
    print(f"Libraries visible: {libraries}")
    print(f"String keyword success: {string_resp.get('success')}")
    if not string_resp.get("success"):
        print(f"  String error: {string_resp.get('error')}")
    print(f"Browser keyword success: {browser_resp.get('success')}")
    if not browser_resp.get("success"):
        print(f"  Browser error: {browser_resp.get('error')}")
    print("=" * 60)

    # Assertions
    assert "Browser" in libraries, (
        f"BUG REPRODUCED: Browser library not visible in session. "
        f"Available: {libraries}"
    )
    assert "String" in libraries, (
        f"String library not visible when Browser is imported. "
        f"Available: {libraries}"
    )

    # String should work
    assert string_resp.get("success"), (
        f"String keyword failed when Browser library is also imported. "
        f"Error: {string_resp.get('error', 'Unknown')}"
    )

    # Browser keyword should work (or at least be recognized)
    # Note: Get Browser Catalog might fail if Playwright is not initialized
    # but the error should be about Playwright, not about missing library
    if not browser_resp.get("success"):
        error = browser_resp.get("error", "")
        # If the error is about Playwright not being initialized, that's expected
        # If the error is "No keyword with name..." then the bug is reproduced
        if "No keyword with name" in error or "keyword not found" in error.lower():
            assert False, (
                f"BUG REPRODUCED: Browser keyword not found via bridge. "
                f"Error: {error}. Libraries visible: {libraries}"
            )
        else:
            # Other errors (like Playwright not initialized) are acceptable
            print(f"Browser keyword failed with expected error: {error}")

    # Robot test should complete (may have rc != 0 if Browser setup fails)
    # But the test case itself should not crash the bridge
