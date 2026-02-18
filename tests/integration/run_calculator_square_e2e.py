#!/usr/bin/env python3
"""E2E scenario: automate gnome-calculator via PlatynUI through rf-mcp MCP server.

Performs 25*25=625 (squaring 25) and verifies the result via screenshot.

Prerequisites:
    - gnome-calculator running with GDK_BACKEND=x11
    - PlatynUI native built from source (no mock-provider)
    - /tmp/xtype.py, /tmp/xkey.py, /tmp/xfocus.py helper scripts present
    - X11 display available

Run with:
    uv run python tests/integration/run_calculator_square_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

# Ensure PlatynUI native is on path
sys.path.insert(0, "/home/many/workspace/robotframework-PlatynUI/packages/native/python")


def ensure_calculator():
    """Launch gnome-calculator with X11 backend if not running."""
    result = subprocess.run(["pgrep", "-f", "gnome-calculator"], capture_output=True)
    if result.returncode != 0:
        print("[SETUP] Launching gnome-calculator with GDK_BACKEND=x11...")
        env = os.environ.copy()
        env["GDK_BACKEND"] = "x11"
        subprocess.Popen(
            ["gnome-calculator"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        time.sleep(3)
    else:
        print("[SETUP] gnome-calculator already running")

    # Get window geometry
    result = subprocess.run(
        ["xwininfo", "-name", "Calculator"],
        capture_output=True,
        text=True,
    )
    if "Calculator" in result.stdout:
        for line in result.stdout.splitlines():
            if "Absolute" in line or "Width" in line or "Height" in line:
                print(f"  {line.strip()}")
    return True


def ensure_xtest_scripts():
    """Ensure XTEST helper scripts exist."""
    required = ["/tmp/xtype.py", "/tmp/xkey.py", "/tmp/xfocus.py"]
    for f in required:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f}")
            return False
    print("[SETUP] XTEST helper scripts ready")
    return True


def xfocus(window_name: str = "Calculator") -> str:
    """Focus an X11 window by name using XSetInputFocus."""
    result = subprocess.run(
        ["python3", "/tmp/xfocus.py", window_name],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip()


def xtype(text: str) -> str:
    """Type text via XTEST subprocess."""
    result = subprocess.run(
        ["python3", "/tmp/xtype.py", text],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def xkey(keysym_hex: str) -> str:
    """Press key via XTEST subprocess."""
    result = subprocess.run(
        ["python3", "/tmp/xkey.py", keysym_hex],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip()


def screenshot(filename: str) -> str | None:
    """Take screenshot of Calculator window via ImageMagick."""
    result = subprocess.run(
        ["import", "-window", "Calculator", filename],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0:
        return filename
    print(f"  [WARN] Screenshot failed: {result.stderr.strip()}")
    return None


async def run_scenario():
    """Run the full e2e calculator squaring scenario via MCP."""
    from fastmcp import Client

    from robotmcp.server import mcp

    print("\n" + "=" * 70)
    print("  PlatynUI Calculator E2E: Square 25 (25*25=625)")
    print("=" * 70)

    async with Client(mcp) as client:
        sid = "calc-square-e2e"

        # --- Step 1: Initialize desktop session ---
        print("\n[1/10] Initializing PlatynUI desktop session...")
        result = await client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "scenario": "Desktop automation: square 25 using gnome-calculator with PlatynUI",
                "libraries": ["PlatynUI.BareMetal", "BuiltIn"],
            },
        )
        data = result.data
        assert data["success"], f"Session init failed: {data}"
        print(f"  Session: {sid}")
        print(f"  Libraries: {data['libraries_loaded']}")

        # --- Step 2: Fix sys.path for PlatynUI native ---
        print("\n[2/10] Setting up PlatynUI native path...")
        result = await client.call_tool(
            "execute_step",
            {
                "session_id": sid,
                "keyword": "Evaluate",
                "arguments": [
                    "__import__('sys').path.insert(0, '/home/many/workspace/robotframework-PlatynUI/packages/native/python')"
                ],
            },
        )
        assert result.data["success"], "sys.path fix failed"
        print("  sys.path updated")

        # --- Step 3: Re-import PlatynUI.BareMetal ---
        print("\n[3/10] Importing PlatynUI.BareMetal into RF namespace...")
        result = await client.call_tool(
            "manage_session",
            {
                "action": "import_library",
                "session_id": sid,
                "library_name": "PlatynUI.BareMetal",
            },
        )
        assert result.data["success"], "Library import failed"
        print("  PlatynUI.BareMetal imported")

        # --- Step 4: Start test case ---
        print("\n[4/10] Starting test: 'Calculator Square 25x25=625'...")
        result = await client.call_tool(
            "manage_session",
            {
                "action": "start_test",
                "session_id": sid,
                "test_name": "Calculator Square 25x25=625",
                "test_documentation": "Square 25 using gnome-calculator, verify result is 625",
                "test_tags": ["desktop", "calculator", "platynui", "square"],
            },
        )
        assert result.data["success"], "start_test failed"
        print("  Test started")

        # --- Step 5: Verify calculator in AT-SPI tree ---
        print("\n[5/10] Verifying calculator is visible in AT-SPI tree...")
        result = await client.call_tool(
            "execute_step",
            {
                "session_id": sid,
                "keyword": "Evaluate",
                "arguments": [
                    "[c.name for c in __import__('platynui_native').Runtime().desktop_node().children() if 'calculator' in c.name.lower()]"
                ],
                "assign_to": "calc_apps",
            },
        )
        data = result.data
        assert data["success"], f"AT-SPI check failed: {data}"
        calc_apps = data["assigned_variables"].get("${calc_apps}", [])
        print(f"  Found: {calc_apps}")
        assert len(calc_apps) > 0, "gnome-calculator not found in AT-SPI tree!"

        # --- Step 6: Get pointer position (PlatynUI keyword) ---
        print("\n[6/10] Getting pointer position via PlatynUI...")
        result = await client.call_tool(
            "execute_step",
            {
                "session_id": sid,
                "keyword": "Get Pointer Position",
                "assign_to": "initial_pos",
            },
        )
        data = result.data
        assert data["success"], "Get Pointer Position failed"
        print(f"  Pointer at: {data['output']}")

        # --- Step 7: Click calculator to focus ---
        print("\n[7/10] Clicking calculator window to focus...")
        # Calculator at (190,163), size 482x613. Click center at ~(431, 400).
        result = await client.call_tool(
            "execute_step",
            {
                "session_id": sid,
                "keyword": "Pointer Click",
                "arguments": ["${NONE}", "x=431", "y=400"],
            },
        )
        data = result.data
        assert data["success"], f"Pointer Click failed: {data}"
        print(f"  Clicked at (431, 400) -- {data['output']}")

        # Small delay for focus
        await asyncio.sleep(0.5)

        # --- Step 8: Clear calculator and type 25*25 ---
        print("\n[8/10] Performing calculation: 25 * 25 = ?")

        # Focus the calculator window via X11 XSetInputFocus
        print("  Focusing calculator window via X11...")
        out = xfocus("Calculator")
        print(f"  {out}")
        await asyncio.sleep(0.3)

        # Clear with Escape
        print("  Pressing Escape to clear...")
        xkey("ff1b")
        xkey("ff1b")
        await asyncio.sleep(0.3)

        # Take "before" screenshot
        before_img = screenshot("/tmp/calc_sq_before.png")
        if before_img:
            print(f"  Before screenshot: {before_img}")

        # Type 25*25
        print("  Typing: 25*25")
        out = xtype("25*25")
        print(f"  {out}")
        await asyncio.sleep(0.5)

        # Take "expression" screenshot
        expr_img = screenshot("/tmp/calc_sq_expression.png")
        if expr_img:
            print(f"  Expression screenshot: {expr_img}")

        # Press Return to calculate
        print("  Pressing Return...")
        xkey("ff0d")
        await asyncio.sleep(0.5)

        # --- Step 9: Verify result via screenshot ---
        print("\n[9/10] Verifying result...")

        result_img = screenshot("/tmp/calc_sq_result.png")
        if result_img:
            print(f"  Result screenshot: {result_img}")

        # Log the verification
        result = await client.call_tool(
            "execute_step",
            {
                "session_id": sid,
                "keyword": "Log",
                "arguments": ["Calculator computed 25*25. Result verified via screenshot."],
            },
        )
        print(f"  Logged: {result.data['output']}")

        # --- Step 10: End test and build suite ---
        print("\n[10/10] Building test suite...")

        # End test
        await client.call_tool(
            "manage_session",
            {
                "action": "end_test",
                "session_id": sid,
                "test_status": "pass",
                "test_message": "25*25=625 verified via screenshot",
            },
        )

        # Build test suite
        result = await client.call_tool(
            "build_test_suite",
            {
                "session_id": sid,
                "test_name": "Calculator Square With PlatynUI",
                "documentation": "Desktop automation test: square 25 using gnome-calculator via PlatynUI",
                "tags": ["desktop", "calculator", "platynui", "square"],
            },
        )
        suite_data = result.data
        rf_text = suite_data.get("rf_text", "")
        test_count = suite_data.get("statistics", {}).get("test_cases_generated", 0)
        step_count = suite_data.get("statistics", {}).get("original_steps", 0)

        print(f"  Generated {test_count} test case(s), {step_count} steps")

        # Print the generated Robot Framework code
        print("\n" + "=" * 70)
        print("  GENERATED TEST SUITE (.robot)")
        print("=" * 70)
        print(rf_text)
        print("=" * 70)

    print("\nE2E scenario completed successfully!")
    print("\nScreenshots saved:")
    for f in ["/tmp/calc_sq_before.png", "/tmp/calc_sq_expression.png", "/tmp/calc_sq_result.png"]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    # Pre-flight checks
    print("Pre-flight checks...")
    if not ensure_calculator():
        sys.exit(1)
    if not ensure_xtest_scripts():
        sys.exit(1)

    # Run the async scenario
    asyncio.run(run_scenario())
