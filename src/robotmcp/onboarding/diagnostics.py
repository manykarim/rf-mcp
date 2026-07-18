"""`robotmcp init`, `doctor`, and `--version` — the manual onboarding surface for a
tool-installed rf-mcp. None of these start the MCP server."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# (module, human label, extra that provides it)
TEST_LIBRARIES: List[Tuple[str, str, str]] = [
    ("RequestsLibrary", "API (RequestsLibrary)", "api"),
    ("SeleniumLibrary", "Web (SeleniumLibrary)", "web"),
    ("Browser", "Web (Browser/Playwright)", "web"),
    ("AppiumLibrary", "Mobile (AppiumLibrary)", "mobile"),
    ("DatabaseLibrary", "Database (DatabaseLibrary)", "database"),
]

MCP_CONFIG_SNIPPET = (
    '{\n'
    '  "mcpServers": {\n'
    '    "robotmcp": { "command": "robotmcp" }\n'
    '  }\n'
    '}'
)


def get_version() -> str:
    try:
        from importlib.metadata import version
        return version("rf-mcp")
    except Exception:
        return "unknown"


def library_status() -> Dict[str, bool]:
    return {mod: importlib.util.find_spec(mod) is not None for mod, _, _ in TEST_LIBRARIES}


def node_present() -> bool:
    return shutil.which("node") is not None


def browser_initialized() -> bool:
    """Best-effort: robotframework-browser's node wrapper exists only after
    `rfbrowser init` has run."""
    spec = importlib.util.find_spec("Browser")
    if not spec or not spec.origin:
        return False
    wrapper = Path(spec.origin).parent / "wrapper" / "node_modules"
    return wrapper.exists()


def browser_init_argv() -> Optional[List[str]]:
    """Resolve how to invoke robotframework-browser's initializer in *this*
    interpreter's environment (so Playwright lands where the installed Browser
    library imports from), version-robustly:

    1. the ``rfbrowser`` console script installed next to this Python, else
    2. ``python -m Browser.entry`` when that module exists.

    Returns None when robotframework-browser is not actually installed (a bare
    importable ``Browser`` shadow does not count)."""
    rfb = Path(sys.executable).with_name("rfbrowser")
    if rfb.exists():
        return [str(rfb)]
    if importlib.util.find_spec("Browser.entry") is not None:
        return [sys.executable, "-m", "Browser.entry"]
    return None


def run_browser_init() -> Tuple[bool, str]:
    argv = browser_init_argv()
    if not argv:
        return False, "robotframework-browser (rfbrowser) not found in this environment"
    r = subprocess.run(argv + ["init"], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr)


def cmd_version() -> int:
    print(get_version())
    return 0


def cmd_doctor() -> int:
    print(f"rf-mcp {get_version()}")
    print(f"executable: {shutil.which('robotmcp') or sys.argv[0]}")
    print("test libraries:")
    for mod, label, extra in TEST_LIBRARIES:
        ok = importlib.util.find_spec(mod) is not None
        print(f"  [{'x' if ok else ' '}] {label}"
              + ("" if ok else f"   (add with rf-mcp[{extra}])"))
    print(f"Browser initialized (Playwright): {'yes' if browser_initialized() else 'no'}")
    print(f"Node.js present: {'yes' if node_present() else 'no (required by the Browser library)'}")
    return 0


def cmd_init(*, browsers: bool = False) -> int:
    """Idempotent, non-destructive. Reports libraries, optionally runs browser
    init, and always prints the MCP config to paste into a coding agent."""
    libs = library_status()
    print(f"rf-mcp {get_version()} — init")
    print("test libraries:")
    for mod, label, extra in TEST_LIBRARIES:
        ok = libs[mod]
        print(f"  [{'x' if ok else ' '}] {label}"
              + ("" if ok else f"   (add with rf-mcp[{extra}])"))

    want_browsers = browsers or libs.get("Browser")
    if want_browsers:
        if not libs.get("Browser"):
            print('\nBrowser (Playwright) not installed. Add it with:')
            print('  uv tool install "rf-mcp[web]"')
        else:
            if not node_present():
                print("\nWARNING: Node.js not found on PATH — the Browser library "
                      "needs it at runtime. Install Node.js, then re-run init.")
            if browser_initialized():
                print("\nPlaywright browser already initialized.")
            else:
                print("\nInitializing the Playwright browser (this downloads a browser, "
                      "may take ~1 minute)…")
                ok, out = run_browser_init()
                print("  " + ("done." if ok else "FAILED — see output below:\n" + out[-600:]))

    print("\nAdd this to your coding agent's MCP configuration:\n")
    print(MCP_CONFIG_SNIPPET)
    print('\nOr run `robotmcp install` to register it into detected agents '
          'automatically. See `robotmcp list` for supported agents.')
    return 0
