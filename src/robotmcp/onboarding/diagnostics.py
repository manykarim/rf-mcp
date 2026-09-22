"""`robotmcp init`, `doctor`, and `--version` - the manual onboarding surface for a
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
    # Desktop was missing entirely, so `doctor` could not diagnose the product's
    # most common install failure - and on Python 3.10/3.11 the marker drops
    # PlatynUI silently, with nothing to reveal it (change: installer-cli-safety sec 9).
    ("PlatynUI", "Desktop (PlatynUI)", "desktop"),
]

# Extras that cannot be installed with a plain `uv ... "rf-mcp[<extra>]"`, mapped to
# the command that actually works. `desktop` pins a PlatynUI pre-release, which
# uv/uvx/pipx refuse transitively (change: install-extra-resolvability).
EXTRA_INSTALL_HINT = {
    "desktop": 'uv tool install --prerelease=allow "rf-mcp[desktop]"  '
               '(or: pip install "rf-mcp[desktop]"; needs Python 3.12+, '
               'glibc>=2.34 / macOS arm64 / Windows)',
}


def _extra_hint(extra: str) -> str:
    """The actionable install hint for a missing extra."""
    return EXTRA_INSTALL_HINT.get(extra, f"add with rf-mcp[{extra}]")

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


def cmd_doctor(*, project_dir: Optional[str] = None, strict: bool = False) -> int:
    """Report installation health.

    Returns 0 by default so plain `doctor` stays a report. With ``strict`` it
    returns non-zero when a checked capability is missing, so CI can gate on it
    (change: installer-cli-safety sec 2).
    """
    print(f"rf-mcp {get_version()}")
    print(f"executable: {shutil.which('robotmcp') or sys.argv[0]}")
    print("test libraries:")
    missing = []
    for mod, label, extra in TEST_LIBRARIES:
        ok = importlib.util.find_spec(mod) is not None
        if not ok:
            missing.append(label)
        print(f"  [{'x' if ok else ' '}] {label}"
              + ("" if ok else f"   ({_extra_hint(extra)})"))
    browser_ok = browser_initialized()
    node_ok = node_present()
    print(f"Browser initialized (Playwright): {'yes' if browser_ok else 'no'}")
    print(f"Node.js present: {'yes' if node_ok else 'no (required by the Browser library)'}")
    if project_dir is not None:
        _doctor_project(project_dir)
    if strict and missing:
        print(f"\nstrict: {len(missing)} capability/capabilities missing: "
              f"{', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


def _doctor_project(project_dir: str) -> None:
    """Read-only: report which of the project's RF libraries the resolved rf-mcp
    launch would see (change: installer-project-aware-launch)."""
    from robotmcp.onboarding import installer as I
    from robotmcp.onboarding import project_env as pe

    env = pe.detect(Path(project_dir).expanduser())
    print(f"\nproject: {env.project_dir}")
    print(f"  environment: {env.type}" + (" (virtualenv)" if env.is_venv else ""))
    print(f"  interpreter: {env.python or '-'}")
    print(f"  rf-mcp already in project env: "
          f"{'yes' if pe.rfmcp_in_project(env.python) else 'no'}")
    conflict = pe.rf_conflict(env)
    if conflict:
        print(f"  version conflict: {conflict}")
    else:
        # Not a conflict (that routes to attach); just which RF will execute.
        vnote = pe.rf_version_note(env)
        if vnote:
            print(f"  robot framework: {vnote}")
    plan = I.resolve_launch(scope="project", project_dir=env.project_dir)
    print(f"  resolved launch: [{plan.strategy}] {plan.command} "
          f"{' '.join(plan.args)}".rstrip())
    if plan.note:
        print(f"    {plan.note}")
    extras = pe.project_extra_libraries(env)
    if not extras:
        print("  extra project libraries: none (rf-mcp[all]'s bundle suffices)")
        return
    interp = I._plan_interpreter(plan)
    print("  project libraries rf-mcp would see:")
    for lib in extras:
        ok = False
        if interp is not None:
            try:
                r = subprocess.run([*interp, "-c", f"import {lib}"],
                                   capture_output=True, text=True, timeout=90,
                                   stdin=subprocess.DEVNULL)
                ok = r.returncode == 0
            except Exception:
                ok = False
        print(f"    [{'x' if ok else ' '}] {lib}")


def cmd_init(*, browsers: bool = False) -> int:
    """Idempotent, non-destructive. Reports libraries, optionally runs browser
    init, and always prints the MCP config to paste into a coding agent."""
    libs = library_status()
    rc = 0
    print(f"rf-mcp {get_version()} - init")
    print("test libraries:")
    for mod, label, extra in TEST_LIBRARIES:
        ok = libs[mod]
        print(f"  [{'x' if ok else ' '}] {label}"
              + ("" if ok else f"   ({_extra_hint(extra)})"))

    # Only --browsers downloads. This used to read `browsers or libs.get("Browser")`,
    # so a plain `robotmcp init` on any rf-mcp[web]/[all] install started a ~500MB
    # Playwright download with no flag and no prompt - while `--browsers`' own help
    # and the README present it as opt-in (change: installer-cli-safety sec 8).
    if browsers:
        if not libs.get("Browser"):
            print('\nBrowser (Playwright) not installed. Add it with:')
            print('  uv tool install "rf-mcp[web]"')
            rc = 1
        else:
            if not node_present():
                print("\nWARNING: Node.js not found on PATH - the Browser library "
                      "needs it at runtime. Install Node.js, then re-run init.")
            if browser_initialized():
                print("\nPlaywright browser already initialized.")
            else:
                print("\nInitializing the Playwright browser (this downloads a browser, "
                      "may take ~1 minute)...")
                ok, out = run_browser_init()
                print("  " + ("done." if ok else "FAILED - see output below:\n" + out[-600:]))
                if not ok:
                    rc = 1
    elif libs.get("Browser") and not browser_initialized():
        # Report it; do not act on it.
        print("\nPlaywright browser not initialized yet. The Browser library needs it.")
        print("  Run:  robotmcp init --browsers      (downloads a browser, ~1 minute)")
        if not node_present():
            print("  Note: Node.js was not found on PATH; the Browser library needs it.")

    print("\nAdd this to your coding agent's MCP configuration:\n")
    print(MCP_CONFIG_SNIPPET)
    print('\nOr run `robotmcp install` to register it into detected agents '
          'automatically. See `robotmcp list` for supported agents.')
    # The snippet launches rf-mcp in ITS OWN environment. For a project with its own
    # libraries that is the wrong environment, and nothing else says so
    # (change: launch-env-fidelity sec 5).
    print('\nNOTE: that snippet runs rf-mcp in its own environment. If you have an '
          'existing Robot Framework project with its own libraries, run '
          '`robotmcp install` from the project directory instead - it writes a launch '
          'that can also see your project. Check it with `robotmcp doctor -C .`.')
    return rc
