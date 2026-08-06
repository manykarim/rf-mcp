#!/usr/bin/env python3
"""Deterministic desktop steering + read-back smoke (no LLM).
change: desktop-docker-agent-harness.

Proves the harness end-to-end against gnome-calculator:
  0. AT-SPI2 provider is active (fail fast, never hang, if not).
  1. launch the calculator, resolve it, confirm native pattern API imports.
  2. bring it to front (exercises WindowSurface via the fluxbox EWMH WM),
     type 7*6=, and READ the result display back via node attributes.
  3. screenshot to the artifacts dir; write a machine-checkable JSON record.

Exit 0 iff the read-back value is 42 and a non-trivial screenshot was written.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ARTIFACTS = os.environ.get("ROBOTMCP_SCREENSHOT_DIR", "/artifacts")
RECORD = os.path.join(ARTIFACTS, "smoke_result.json")
SHOT = os.path.join(ARTIFACTS, "calc.png")
result: dict = {"rungs": {}, "provider_ok": False, "read_back": None, "passed": False}


def _try(fn):
    """Call fn(), returning its value or None on any exception."""
    try:
        return fn()
    except Exception:
        return None


def finish(code: int) -> None:
    # The result ALWAYS prints to stdout (captured by CI) so a non-writable
    # artifacts mount can never mask the real pass/fail — the JSON record is a
    # best-effort side artifact, not the gate.
    try:
        os.makedirs(ARTIFACTS, exist_ok=True)
        with open(RECORD, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except OSError as e:
        result["record_write_error"] = repr(e)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(code)


try:
    import platynui_native as pn
except Exception as e:  # native runtime missing / skewed
    result["error"] = f"import platynui_native failed: {e!r}"
    finish(3)

rt = pn.Runtime()

# --- Rung 0: AT-SPI provider must be active ---------------------------------
providers = []
try:
    providers = [dict(p) if isinstance(p, dict) else {"repr": str(p)} for p in rt.providers()]
except Exception as e:
    result["error"] = f"providers() failed: {e!r}"
    finish(2)
result["providers"] = providers
atspi = [p for p in providers if "spi" in json.dumps(p).lower()]
result["provider_ok"] = bool(atspi)
result["rungs"]["0_provider"] = result["provider_ok"]
if not result["provider_ok"]:
    result["error"] = "no AT-SPI2 provider active — accessibility bus not up"
    finish(2)

# --- Rung 1: launch + resolve the calculator --------------------------------
proc = subprocess.Popen(["gnome-calculator"], env=dict(os.environ))
result["aut_pid"] = proc.pid
app = None
for _ in range(40):
    time.sleep(0.5)
    try:
        rt.clear_cache()
        apps = rt.evaluate("/app:*")
        for a in apps:
            if "calc" in (a.name or "").lower():
                app = a
                break
    except Exception:
        pass
    if app is not None:
        break
if app is None:
    result["error"] = "calculator did not appear in the AT-SPI tree"
    finish(4)
result["aut_name"] = app.name
result["aut_atspi_pid"] = _try(lambda: app.attribute("ProcessId"))
result["rungs"]["1_resolved"] = True

# native pattern API must import cleanly (no WindowSurface symbol skew)
frame = None
try:
    frame = rt.evaluate_single(f"/app:*[@Name='{app.name}']/*[1]")
    result["frame_role"] = frame.role
    result["frame_patterns"] = list(frame.supported_patterns())
    result["rungs"]["1_pattern_api"] = True
except Exception as e:
    result["error"] = f"pattern API failed: {e!r}"
    finish(5)

# --- Rung 2: steer (bring-to-front + type) then READ BACK --------------------
try:
    rt.bring_to_front(frame)
    result["rungs"]["2_bring_to_front"] = True
except Exception as e:
    # not fatal: fluxbox may focus on map; record and continue
    result["rungs"]["2_bring_to_front"] = f"warn: {e!r}"
time.sleep(0.8)

try:
    rt.keyboard_type("7*6=")
except Exception:
    try:
        rt.keyboard_type("7*6\n")
    except Exception as e:
        result["error"] = f"keyboard_type failed: {e!r}"
        finish(6)
time.sleep(1.0)

# read the result display back: scan the calc subtree for a text node == 42
rt.clear_cache()
found = None
texts = []
try:
    for n in rt.evaluate(f"/app:*[@Name='{app.name}']//control:*"):
        for attr in ("Text", "Value", "Name"):
            v = _try(lambda a=attr: n.attribute(a))
            if v is None:
                continue
            s = str(v).strip()
            if s:
                texts.append(f"{n.role}:{attr}={s}")
            if s.replace(" ", "") in ("42", "7×6=42", "7*6=42", "42."):
                found = s
        if found:
            break
except Exception as e:
    result["error"] = f"read-back scan failed: {e!r}"
result["sample_texts"] = texts[:25]
result["read_back"] = found
result["rungs"]["3_read_back_42"] = found is not None

# --- Rung 3: screenshot -----------------------------------------------------
try:
    png = rt.screenshot(None, "image/png")
    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(SHOT, "wb") as f:
        f.write(png)
    result["screenshot"] = SHOT
    result["screenshot_bytes"] = len(png)
    result["rungs"]["4_screenshot"] = len(png) > 2000
except Exception as e:
    result["error"] = f"screenshot failed: {e!r}"

try:
    proc.terminate()
except Exception:
    pass

result["passed"] = bool(result["provider_ok"] and found is not None
                        and result["rungs"].get("4_screenshot"))
finish(0 if result["passed"] else 7)
