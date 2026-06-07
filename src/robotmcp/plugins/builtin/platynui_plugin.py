"""Builtin PlatynUI.BareMetal plugin (new Rust core, ADR-025).

Targets the new-core PlatynUI (https://github.com/imbus/robotframework-PlatynUI,
branch ``new_core``): Rust runtime + ``platynui-native`` PyO3 bindings +
``platynui-cli`` diagnostic tool. The Robot Framework library surface is
``PlatynUI.BareMetal`` (24 keywords, descriptor/XPath locator model).

Key integration concerns handled here (see ADR-025):

* **Wayland portal hang** — on Linux Wayland sessions the PlatynUI runtime
  blocks indefinitely on an interactive ``org.freedesktop.portal.RemoteDesktop``
  consent handshake. ``ensure_x11_session_env()`` forces the X11/XWayland
  backend (``XDG_SESSION_TYPE=x11``) before the first ``Runtime`` is created
  in this process. Opt out with ``ROBOTMCP_PLATYNUI_KEEP_WAYLAND=1``.
* **Query scoping** — desktop-wide descendant queries (``//control:...``) can
  take ~1 s per AT-SPI node timeout and minutes on busy desktops. Hints and
  failure guidance push app-scoped queries
  (``/app:*[@Name='X']//control:Button[@Name='OK']``).
* **Matched-set requirement** — the RF library and ``platynui-native`` wheel
  must come from the same source commit until upstream stabilizes; mismatch
  surfaces as ``ImportError`` on native symbols and gets a structured hint.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    InstallAction,
    LibraryCapabilities,
    LibraryHints,
    LibraryMetadata,
    PromptBundle,
)

logger = logging.getLogger(__name__)

# Environment variable to opt out of the X11 session forcing shim.
KEEP_WAYLAND_ENV = "ROBOTMCP_PLATYNUI_KEEP_WAYLAND"

# The 24 PlatynUI.BareMetal keywords of the new core (lowercase RF names).
PLATYNUI_KEYWORDS = (
    # Query / context
    "set root",
    "query",
    "get attribute",
    # Pointer
    "pointer click",
    "pointer multi click",
    "pointer press",
    "pointer release",
    "pointer move to",
    "get pointer position",
    # Keyboard
    "keyboard type",
    "keyboard press",
    "keyboard release",
    # Focus / window management
    "focus",
    "activate window",
    "maximize window",
    "minimize window",
    "restore window",
    "close window",
    "move window",
    "resize window",
    "move and resize window",
    "bring to front",
    # Diagnostics
    "take screenshot",
    "highlight",
)

# Keywords PlatynUI shares (by name) with Browser Library. Both define
# "Focus", "Get Attribute" and "Take Screenshot"; desktop sessions must not
# block them and web sessions must not route them to PlatynUI.
_SHARED_WITH_BROWSER = frozenset({"focus", "get attribute", "take screenshot"})

_ACTIONABILITY_HINT = (
    "PlatynUI queries that start with '//' walk the WHOLE desktop tree and can "
    "take minutes on busy desktops (AT-SPI applies a 1s timeout per "
    "unresponsive node). Scope queries to the target application instead: "
    "/app:*[@Name='myapp']//control:Button[@Name='OK'] — or call 'Set Root' "
    "once with the application/window node."
)

_LINUX_FRAME_HINT = (
    "On Linux (AT-SPI2), application top-level windows usually expose the "
    "role 'Frame', not 'Window'. Use //control:Frame[@Name='...'] (or the "
    "window-management keywords, which accept Frame/Window/Dialog alike). "
    "control:Window typically only matches compositor/shell elements."
)

_MATCHED_SET_HINT = (
    "PlatynUI.BareMetal and platynui-native must be built from the SAME "
    "source commit (upstream is preview quality; PyPI dev wheels lag the "
    "source tree). If you see ImportError for native symbols (e.g. "
    "'WindowSurface'), rebuild: `maturin develop --release --manifest-path "
    "packages/native/Cargo.toml` in the robotframework-PlatynUI checkout, "
    "then `pip install --no-deps <checkout>`."
)


# Serializes the process-global os.environ mutation across the three
# trigger points (library import chokepoint, keyword-execution chokepoint,
# session-start hook) — concurrent first-touches must not interleave the
# check-and-set (cross-LLM review finding, ADR-025).
_ENV_SHIM_LOCK = threading.Lock()


def ensure_x11_session_env(environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Force the X11 backend for PlatynUI on Linux Wayland sessions.

    The PlatynUI runtime resolves the session type once per process from
    ``XDG_SESSION_TYPE`` (authoritative) / ``WAYLAND_DISPLAY`` / ``DISPLAY``.
    The Wayland input backend performs an xdg-desktop-portal RemoteDesktop
    handshake with **no timeout** and (on first run) an interactive consent
    dialog — fatal for a headless MCP server. The X11/XWayland backend has no
    such handshake and provides full keyboard/pointer/screenshot support.

    Must run BEFORE the first ``platynui_native.Runtime`` is created in this
    process (the session type is cached process-wide).

    Returns a human-readable note when the environment was changed, else None.
    """
    env = environ if environ is not None else os.environ
    if sys.platform != "linux":
        return None

    # The mutation is set-once / never-restored (same constant value), but
    # the check-and-set must not interleave across threads — and CPython's
    # putenv from concurrent threads should be serialized anyway.
    with _ENV_SHIM_LOCK:
        if env.get(KEEP_WAYLAND_ENV, "").strip() in {"1", "true", "yes"}:
            return None

        session_type = env.get("XDG_SESSION_TYPE", "").strip().lower()
        wayland = bool(env.get("WAYLAND_DISPLAY"))
        display = bool(env.get("DISPLAY"))

        if session_type == "x11":
            return None
        if not display:
            # No X server available — forcing X11 would break init outright.
            if session_type == "wayland" or wayland:
                logger.warning(
                    "PlatynUI: Wayland session without DISPLAY — runtime init may "
                    "block on an xdg-desktop-portal consent dialog. Approve the "
                    "dialog once (a restore token is persisted) or provide an "
                    "X11/XWayland DISPLAY."
                )
            return None
        if session_type == "wayland" or (not session_type and wayland):
            env["XDG_SESSION_TYPE"] = "x11"
            note = (
                "PlatynUI: forced XDG_SESSION_TYPE=x11 (XWayland) to avoid the "
                "Wayland xdg-desktop-portal consent handshake which blocks "
                f"indefinitely in headless contexts. Set {KEEP_WAYLAND_ENV}=1 to "
                "keep the native Wayland backend."
            )
            logger.warning(note)
            return note
        return None


class PlatynUILibraryPlugin(StaticLibraryPlugin):
    """Builtin plugin for the new-core PlatynUI.BareMetal desktop library."""

    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="PlatynUI.BareMetal",
            package_name="robotframework-platynui",
            import_path="PlatynUI.BareMetal",
            description=(
                "Cross-platform native desktop UI automation (Windows UIA, "
                "Linux AT-SPI2) with XPath locators, backed by a Rust runtime"
            ),
            library_type="external",
            use_cases=[
                "desktop automation",
                "native application testing",
                "window management",
                "desktop ui inspection",
            ],
            categories=["desktop", "testing"],
            contexts=["desktop"],
            installation_command=(
                "pip install --pre platynui-native platynui-cli "
                "(RF library: install robotframework-PlatynUI from source, "
                "branch new_core — matched commit with platynui-native)"
            ),
            dependencies=["platynui-native"],
            platform_requirements=["python>=3.12"],
            requires_type_conversion=True,
            supports_async=False,
            load_priority=42,
            default_enabled=True,
            extra_name="desktop",
            technology_tags=["desktop", "uia", "atspi", "xpath"],
            aliases=["PlatynUI"],
        )
        capabilities = LibraryCapabilities(
            contexts=["desktop"],
            features=["window-management", "pointer", "keyboard", "screenshot"],
            technology=["UIA", "AT-SPI2"],
            supports_page_source=False,
            supports_application_state=True,
            requires_type_conversion=True,
        )
        hints = LibraryHints(
            standard_keywords=[
                "Query",
                "Pointer Click",
                "Keyboard Type",
                "Activate Window",
                "Get Attribute",
                "Take Screenshot",
            ],
            error_hints=[
                _ACTIONABILITY_HINT,
                _LINUX_FRAME_HINT,
                _MATCHED_SET_HINT,
            ],
            usage_examples=[
                "Query    /app:*[@Name='gnome-calculator']//control:Frame    only_first=True",
                "Pointer Click    /app:*[@Name='myapp']//control:Button[@Name='OK']",
                "Keyboard Type    /app:*[@Name='myapp']//control:Text[@Name='Input']    Hello <Ctrl+A>",
                "Activate Window    /app:*[@Name='myapp']//control:Frame",
                "Get Attribute    /app:*[@Name='myapp']//control:Frame    Bounds",
                "Take Screenshot    filename=EMBED",
            ],
        )
        prompt_bundle = PromptBundle(
            recommendation=(
                "Use PlatynUI.BareMetal for native desktop application "
                "automation (NOT web pages). Locators are XPath over the "
                "desktop accessibility tree with namespaces app:/control:/"
                "item:/native: and PascalCase attributes (@Name, @Bounds, "
                "@AutomationId). ALWAYS scope queries to an application: "
                "/app:*[@Name='X']//control:Button[@Name='OK']."
            ),
            troubleshooting=(
                "Element not found: list applications first with Query "
                "/app:* then inspect one level at a time. Slow queries: "
                "never start a locator with // (full-desktop walk). On "
                "Linux, windows are control:Frame, not control:Window. "
                "Keyboard sequences support chords: <Ctrl+A>, <Enter>."
            ),
        )
        install_actions = [
            InstallAction(
                description="Install PlatynUI native runtime + CLI (pre-release)",
                command=["pip", "install", "--pre", "platynui-native", "platynui-cli"],
            ),
            InstallAction(
                description=(
                    "Install RF library from source (matched commit with the "
                    "native wheel — see ADR-025)"
                ),
                command=[
                    "pip",
                    "install",
                    "--no-deps",
                    "git+https://github.com/imbus/robotframework-PlatynUI.git@new_core",
                ],
            ),
        ]
        super().__init__(
            metadata=metadata,
            capabilities=capabilities,
            install_actions=install_actions,
            hints=hints,
            prompt_bundle=prompt_bundle,
        )

    # -- keyword routing -------------------------------------------------

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        mapping: Dict[str, str] = {}
        for keyword in PLATYNUI_KEYWORDS:
            mapping[f"platynui.baremetal.{keyword}"] = "PlatynUI.BareMetal"
            if keyword not in _SHARED_WITH_BROWSER:
                mapping[keyword] = "PlatynUI.BareMetal"
        return mapping

    # -- session hooks ---------------------------------------------------

    def on_session_start(self, session: "ExecutionSession") -> None:
        """Force the X11 backend before the PlatynUI runtime exists.

        NOTE: this hook fires at session *creation*, which usually happens
        before the library list is populated — the deterministic trigger is
        the desktop check in ``KeywordExecutor._execute_keyword_serialized``
        (ADR-025). This hook covers restore/attach flows where the session
        already carries PlatynUI.
        """
        try:
            libraries = list(getattr(session, "imported_libraries", None) or [])
            libraries += list(getattr(session, "search_order", None) or [])
            preference = getattr(session, "explicit_library_preference", "") or ""
            libraries.append(preference)
            if any("platynui" in str(lib).lower() for lib in libraries):
                ensure_x11_session_env()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("PlatynUI session-start hook failed: %s", exc)

    def before_keyword_execution(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        library_manager: Any,
        keyword_discovery: Any,
    ) -> None:
        """Safety net: ensure the env shim ran before any PlatynUI keyword."""
        if keyword_name and keyword_name.lower() in self.get_keyword_library_map():
            ensure_x11_session_env()

    # -- failure guidance ------------------------------------------------

    def generate_failure_hints(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        arguments: List[Any],
        error_text: str,
    ) -> List[Dict[str, Any]]:
        hints: List[Dict[str, Any]] = []
        error_lower = (error_text or "").lower()
        args_text = " ".join(str(a) for a in arguments)

        if "importerror" in error_lower or "cannot import name" in error_lower:
            hints.append(
                {
                    "type": "platynui_matched_set",
                    "title": "PlatynUI version mismatch",
                    "message": _MATCHED_SET_HINT,
                }
            )
        if "elementnotfound" in error_lower or "no nodes" in error_lower or (
            "not found" in error_lower and "element" in error_lower
        ):
            message = _ACTIONABILITY_HINT
            if "control:window" in args_text.lower():
                message = f"{_LINUX_FRAME_HINT} {message}"
            hints.append(
                {
                    "type": "platynui_locator",
                    "title": "PlatynUI locator guidance",
                    "message": message,
                }
            )
        if "timeout" in error_lower or "timed out" in error_lower:
            hints.append(
                {
                    "type": "platynui_query_scope",
                    "title": "Scope desktop queries",
                    "message": _ACTIONABILITY_HINT,
                }
            )
        if "providererror" in error_lower and "mock" in error_lower:
            hints.append(
                {
                    "type": "platynui_mock_provider",
                    "title": "Mock provider not linked",
                    "message": (
                        "use_mock=True needs a platynui-native wheel built "
                        "with `--features mock-provider`; published wheels "
                        "link only the real OS providers."
                    ),
                }
            )
        return hints


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
