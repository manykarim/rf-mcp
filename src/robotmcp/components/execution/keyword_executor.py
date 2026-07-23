"""Keyword execution service."""

import asyncio
import logging
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from robotmcp.components.execution.locator_arg_introspection import (
    LocatorArgIntrospector,
)
from robotmcp.components.execution.rf_native_context_manager import (
    get_rf_native_context_manager,
)
from robotmcp.core.event_bus import FrontendEvent, event_bus
from robotmcp.components.variables.variable_resolver import VariableResolver
from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.execution_models import ExecutionStep
from robotmcp.models.session_models import ExecutionSession
from robotmcp.utils.argument_processor import ArgumentProcessor
from robotmcp.utils.response_serializer import MCPResponseSerializer
from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter
from robotmcp.plugins import get_library_plugin_manager

# Import timeout domain components for proper timeout handling
from robotmcp.domains.timeout import ActionType, TimeoutPolicy, DefaultTimeouts
from robotmcp.domains.timeout.keyword_classifier import classify_keyword
from robotmcp.container import get_container

logger = logging.getLogger(__name__)

# Import Robot Framework components
try:
    from robot.libraries.BuiltIn import BuiltIn

    ROBOT_AVAILABLE = True
except ImportError:
    BuiltIn = None
    ROBOT_AVAILABLE = False


# F-N12: keywords that are TRULY inspection-only — they take no locator,
# return ambient page/session state, and cannot serve as an implicit
# existence assertion. When the caller passes record=None (the default),
# these are auto-flagged record=False so build_test_suite produces a
# clean narrative instead of every page-state probe an LLM does between
# actions.
#
# IMPORTANT — what is NOT in this set, and why:
#
#   Locator-taking getters (Get Text, Get Value, Get Attribute,
#   Get Element Count, Get Element States, Get Property, Get Style,
#   Get Classes, Get Bounding Box, Get Element Attribute, Get Element
#   Size, Get Element Tag Name, Get List Selected Labels, Get List
#   Selected Values, ...) are deliberately RECORDED by default because
#   in Robot Framework they double as implicit existence assertions:
#   they raise on missing element, and the RF assertion-engine pattern
#   lets a call like
#       Get Text    id=cart-badge    ==    2 items
#   serve as an explicit assertion. Silently dropping such calls would
#   remove load-bearing assertions from the generated suite.
#
#   Log / Log To Console / Log Many are RECORDED because they emit
#   intentional narrative into the test report — agents and humans
#   both use them as deliberate test steps, not probes.
#
# CARVE-OUTS preserved even when keyword is in this set:
#   - assign_to is set (the recorded suite needs `${var}= Get Text ...`).
#   - A named test is currently open (after start_test). The user
#     explicitly opened a multi-test scope; steps inside it must NOT be
#     dropped silently — that produced empty test cases in CI (F3/F4).
_INSPECTION_ONLY_KEYWORDS: frozenset[str] = frozenset({
    # Browser — page/viewport ambient state (no locator, never throws)
    "get title",
    "get url",
    "get viewport size",
    "get scroll size",
    # SeleniumLibrary — page ambient state (no locator)
    "get location",
    # AppiumLibrary — session/window ambient state (no locator)
    "get capability",
    "get contexts",
    "get current context",
    "get window height",
    "get window width",
    "get window size",
})


def _resolve_record_gate(
    *,
    keyword: str,
    record: bool | None,
    assign_to: Union[str, List[str]] | None,
    session: Any,
) -> bool:
    """Decide whether a successful step should be appended to session.steps.

    Explicit ``record`` wins over everything. When ``record`` is ``None``
    (the default), the gate auto-classifies, with two CARVE-OUTS that
    always preserve the step:

    1. ``assign_to`` is set — the recorded suite needs ``${var}= Get Text``
       for subsequent assertions to compile.
    2. A named test is currently open (after ``start_test``) — the user
       explicitly opened a multi-test scope; dropping inspection-only
       steps inside it produces empty test cases (CI F3/F4 regression).

    Outside the carve-outs, keywords in ``_INSPECTION_ONLY_KEYWORDS`` are
    dropped; everything else is recorded (conservative default-on).

    Note: locator-taking getters (Get Text, Get Value, Get Attribute,
    Get Element Count, ...) are deliberately NOT in the inspection set
    even though they "read" — in Robot Framework they double as implicit
    existence assertions (raise on missing element, and the assertion-
    engine pattern ``Get Text  id=foo  ==  bar`` makes them explicit
    assertions). They are recorded by default to preserve load-bearing
    test logic. See ``_INSPECTION_ONLY_KEYWORDS`` in this module for
    the full rationale.
    """
    if record is not None:
        return bool(record)
    if assign_to is not None:
        return True
    try:
        if session.test_registry.get_current_test() is not None:
            return True
    except Exception:
        # Defensive: a malformed session must not crash the executor.
        pass
    return keyword.lower().strip() not in _INSPECTION_ONLY_KEYWORDS


def _read_native_character_count(node):
    """Read ``native:Text.CharacterCount`` from a PlatynUI node, returning an int
    or None. Best-effort; never raises.

    The string accessor ``node.attribute("native:Text.CharacterCount")`` returns
    None on the PlatynUI new-core runtime (confirmed via the docker AT-SPI probe,
    2026-07-17: it resolves the metadata list but not a bare colon-string). The
    working read enumerates ``node.attributes()`` and calls ``.value()`` on the
    matching descriptor — that returns the live count (e.g. 5 for "hello"). We
    try the string accessor first for forward-compat, then fall back.
    """
    def _val(attr):
        if attr is None:
            return None
        v = attr.value() if hasattr(attr, "value") and callable(attr.value) else attr
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    # 1) string accessor (forward-compat; currently returns None on new-core)
    attr_fn = getattr(node, "attribute", None)
    if callable(attr_fn):
        got = _val(attr_fn("native:Text.CharacterCount"))
        if got is not None:
            return got
    # 2) enumerate the node's attribute descriptors and read the match
    attrs_fn = getattr(node, "attributes", None)
    if callable(attrs_fn):
        for a in attrs_fn() or []:
            if getattr(a, "name", None) == "Text.CharacterCount" and getattr(
                a, "namespace", "native"
            ) in ("native", None, ""):
                got = _val(a)
                if got is not None:
                    return got
    return None


class KeywordExecutor:
    """Handles keyword execution with proper library routing and error handling."""

    # Keywords that require element pre-validation before execution
    # These keywords interact with elements and benefit from fast visibility/state checks
    ELEMENT_INTERACTION_KEYWORDS: Set[str] = {
        # Click operations (Browser + Selenium + Appium)
        "click",
        "click element",
        "double click",
        "double click element",
        "right click",
        "right click element",
        "click with options",       # Browser: click with modifiers
        "click button",             # SeleniumLibrary: button-specific click
        "click link",               # SeleniumLibrary: link-specific click
        "click image",              # SeleniumLibrary: image-specific click
        "click element at coordinates",  # SeleniumLibrary: offset click
        # Text input operations
        "fill text",
        "fill secret",
        "type text",
        "type secret",
        "input text",
        "input password",
        "input value",              # AppiumLibrary: mobile text input
        "clear text",
        "clear element value",
        "clear element text",       # SeleniumLibrary: clear form field text
        # Checkbox/Radio operations
        "check checkbox",
        "uncheck checkbox",
        "select checkbox",
        "unselect checkbox",
        # Select/Dropdown operations
        "select options",
        "select options by",        # Browser: select by attribute+value
        "deselect options",         # Browser: deselect from dropdown
        "select from list",
        "select from list by value",
        "select from list by label",
        "select from list by index",
        "deselect from list",
        # Keyboard operations
        "press keys",
        "press key",
        "keyboard key",
        # Focus/Hover operations
        "focus",
        "hover",
        "mouse over",
        "mouse out",                # SeleniumLibrary: mouse leave
        "set focus to element",     # SeleniumLibrary: focus keyword
        # Scroll operations
        "scroll to element",
        "scroll element into view",
        "scroll by",                # Browser: scroll near element
        "scroll to",                # Browser: scroll to position
        # Drag operations
        "drag and drop",            # Selenium + Appium
        "drag and drop by offset",  # SeleniumLibrary: drag by pixel offset
        # Touch/Gesture operations (Appium)
        "tap",                      # Browser (mobile) + Appium
        "long press",               # AppiumLibrary: touch-and-hold
        # Mouse operations
        "mouse down",               # SeleniumLibrary: mouse press
        "mouse up",                 # SeleniumLibrary: mouse release
        "mouse down on image",      # SeleniumLibrary: mouse press on image
        "mouse down on link",       # SeleniumLibrary: mouse press on link
        "mouse move relative to",   # Browser: move relative to element
        # Form operations
        "submit form",              # SeleniumLibrary: form submission
        "open context menu",        # SeleniumLibrary: right-click menu
        # File upload operations
        "choose file",              # SeleniumLibrary: file upload
        "upload file by selector",  # Browser: file upload to element
    }

    # Substrings that indicate an editable text field across web/mobile.
    # ``element.tag_name`` returns:
    #   - HTML tag (web/Selenium): ``input``, ``textarea``
    #   - Full Java class (Android/Appium): ``android.widget.EditText``,
    #     ``android.widget.AutoCompleteTextView``,
    #     ``androidx.appcompat.widget.AppCompatEditText``
    #   - XCUI element type (iOS/Appium): ``XCUIElementTypeTextField``,
    #     ``XCUIElementTypeSecureTextField``, ``XCUIElementTypeSearchField``
    # Matching uses substring containment after lower-casing so all three
    # styles map to the same editability heuristic.
    _EDITABLE_TAG_SUBSTRINGS: tuple = (
        "edittext",         # Android: *.EditText / *.AppCompatEditText / AutoCompleteTextView
        "textfield",        # iOS: XCUIElementTypeTextField, XCUIElementTypeSecureTextField
        "searchfield",      # iOS: XCUIElementTypeSearchField
        "input",            # Web fallback: <input>
        "textarea",         # Web fallback: <textarea>
        "textview",         # Some Android variants expose editable TextView
    )

    # Required element states for different action types
    REQUIRED_STATES_FOR_ACTION: Dict[str, Set[str]] = {
        "click": {"visible", "enabled"},
        "fill": {"visible", "enabled", "editable"},
        "input": {"visible", "enabled", "editable"},
        "type": {"visible", "enabled", "editable"},
        "check": {"visible", "enabled"},
        "uncheck": {"visible", "enabled"},
        "select": {"visible", "enabled"},
        "press": {"visible", "enabled"},
        "focus": {"visible"},
        "hover": {"visible"},
        "scroll": {"attached"},
        "clear": {"visible", "enabled", "editable"},
        "drag": {"visible", "enabled"},
        "tap": {"visible", "enabled"},
        "submit": {"visible", "enabled"},
        "upload": {"visible", "enabled"},
        "open": {"visible", "enabled"},
    }

    def __init__(
        self, config: Optional[ExecutionConfig] = None, override_registry=None
    ):
        self.config = config or ExecutionConfig()
        self.keyword_discovery = get_keyword_discovery()
        self.argument_processor = ArgumentProcessor()
        self.rf_converter = RobotFrameworkNativeConverter()
        self.override_registry = override_registry
        self.variable_resolver = VariableResolver()
        self.response_serializer = MCPResponseSerializer()
        # Legacy RobotContextManager is deprecated; use RF native context only
        self.rf_native_context = get_rf_native_context_manager()
        self.plugin_manager = get_library_plugin_manager()
        # Feature flag: route RequestsLibrary session operations via RF runner
        # Default ON; set ROBOTMCP_RF_RUNNER_REQUESTS=0 to disable
        self.rf_runner_requests = os.getenv("ROBOTMCP_RF_RUNNER_REQUESTS", "1") in (
            "1",
            "true",
            "True",
        )
        # Default to context-only execution unless explicitly disabled
        self.context_only = os.getenv("ROBOTMCP_RF_CONTEXT_ONLY", "1") in (
            "1",
            "true",
            "True",
        )
        # Feature flag: enable/disable pre-validation (default ON)
        self.pre_validation_enabled = os.getenv("ROBOTMCP_PRE_VALIDATION", "1") in (
            "1",
            "true",
            "True",
        )
        # Focus-before-act manager for PlatynUI desktop sessions
        # (change: platynui-focused-execution). Lazily created on first use.
        self._platynui_focus_manager = None
        # Lock to serialize Browser timeout mutations during pre-validation.
        # Browser.Set Browser Timeout is a Playwright-global setting; without
        # a lock, concurrent pre-validations in different threads could see
        # the 500ms pre-validation timeout instead of the action timeout.
        self._browser_timeout_lock = threading.Lock()
        # Global asyncio lock to serialize keyword execution.
        # Prevents concurrent asyncio.to_thread() dispatches which cause
        # _suppress_stdout() reference counting to keep fd 1 redirected
        # to stderr while FastMCP writes JSON-RPC responses → lost responses.
        # Also prevents concurrent Selenium WebDriver access (not thread-safe).
        self._execution_lock = asyncio.Lock()
        # Library-aware locator-arg introspector. Used as a CONFIDENT VETO
        # over the curated ELEMENT_INTERACTION_KEYWORDS positive list:
        # when the introspector resolves the keyword to a SPECIFIC library
        # and that library's signature explicitly has no locator-style arg,
        # the entry is stale and pre-validation is skipped.  Ambiguous or
        # unresolved lookups (no session context, multiple libraries match,
        # keyword name not found) do NOT veto — we trust the positive list.
        self._locator_introspector = LocatorArgIntrospector(self.keyword_discovery)

    def _maybe_sanitize_desktop_launch(self, session, keyword, arguments):
        """For a desktop-session ``Start Process`` of a known GUI binary,
        append RF ``env:`` overrides that strip snap-contaminated loader vars
        and set the bound display (change: platynui-desktop-safety-isolation).

        Returns possibly-augmented arguments. Non-GUI / non-desktop launches
        are returned unchanged. Honors a ``platynui_no_sanitize`` session attr.
        """
        try:
            # Sanitize both Start Process and Run Process GUI launches for parity
            # with the resolution hook and the spec (Codex review #3).
            if keyword.strip().lower().rsplit(".", 1)[-1] not in (
                "start process",
                "run process",
            ):
                return arguments
            _is_desktop = getattr(session, "is_desktop_session", None)
            if not (callable(_is_desktop) and _is_desktop() is True):
                return arguments
            from robotmcp.components.execution.desktop_launch_env import (
                build_desktop_launch_env,
                gui_launch_overrides,
                is_desktop_gui_launch,
            )

            # Evidence-based detection: recognize any GUI AUT in a desktop
            # session, not just the 8-binary gnome allow-set (change:
            # desktop-launch-env-generalization).
            binary = is_desktop_gui_launch(
                list(arguments or []), is_desktop_session=True
            )
            if binary is None:
                return arguments
            # Already carries explicit env: overrides? leave it to the author.
            if any(isinstance(a, str) and a.startswith("env:") for a in (arguments or [])):
                return arguments
            sanitize = not bool(getattr(session, "platynui_no_sanitize", False))
            # Base display pins + accessibility overlay so ANY GTK/Qt AUT comes
            # up on X11 with a populated AT-SPI tree (GTK_A11Y=atspi), not just
            # the gnome binaries the image happens to export env for.
            display_env = {
                "DISPLAY": os.environ.get("DISPLAY", ""),
                "WAYLAND_DISPLAY": "",
            }
            display_env.update(gui_launch_overrides(binary))
            clean = build_desktop_launch_env(
                arguments[0], display_env=display_env, sanitize=sanitize
            )
            # Inject only the vars that matter for the snap failure + display,
            # as RF env: overrides (per-var; the rest of the env is inherited).
            inject_vars = (
                "LD_LIBRARY_PATH", "GTK_PATH", "GIO_MODULE_DIR",
                "GIO_EXTRA_MODULES", "GSETTINGS_SCHEMA_DIR", "QT_PLUGIN_PATH",
                "XDG_DATA_DIRS", "DISPLAY", "XDG_SESSION_TYPE", "GDK_BACKEND",
                "GTK_A11Y", "QT_QPA_PLATFORM", "NO_AT_BRIDGE",
            )
            extra = []
            for var in inject_vars:
                if var in clean and clean[var]:
                    extra.append(f"env:{var}={clean[var]}")
            # Neutralize snap single-path vars that were dropped.
            for var in ("LD_PRELOAD", "GTK_EXE_PREFIX"):
                if var in os.environ and var not in clean:
                    extra.append(f"env:{var}=")
            if extra:
                applied_keys = [
                    e.split("=", 1)[0].split(":", 1)[1]
                    for e in extra
                    if isinstance(e, str) and e.startswith("env:") and "=" in e
                ]
                logger.info(
                    "PlatynUI desktop launch: %s recognized as GUI AUT; applied "
                    "%d env overrides (a11y/backend: %s)",
                    binary,
                    len(extra),
                    ",".join(
                        k for k in applied_keys
                        if k in ("GTK_A11Y", "GDK_BACKEND", "QT_QPA_PLATFORM",
                                 "XDG_SESSION_TYPE", "NO_AT_BRIDGE")
                    )
                    or "none",
                )
                return list(arguments) + extra
            return arguments
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("desktop launch sanitize skipped: %s", exc)
            return arguments

    def _maybe_resolve_desktop_executable(self, session, keyword, arguments):
        """For a desktop-session Process launch/probe, resolve the executable
        (``arguments[0]``) to an absolute path via ``shutil.which`` against the
        server-process PATH, so a tool present for the server is found from step
        execution (maintainer-report #7 — ``xdotool FileNotFoundError``).

        Resolution uses the server PATH only — it deliberately does NOT inherit
        an interactive shell's startup environment (a security regression). When
        a tool is genuinely unresolvable, logs a WARNING with the effective PATH
        so the failure is diagnosable rather than opaque.
        change: desktop-mcp-workflow-correctness.
        """
        try:
            kw = keyword.strip().lower().rsplit(".", 1)[-1]
            if kw not in ("start process", "run process"):
                return arguments
            _is_desktop = getattr(session, "is_desktop_session", None)
            if not (callable(_is_desktop) and _is_desktop() is True):
                return arguments
            if not arguments:
                return arguments
            exe = arguments[0]
            # Skip RF config tokens (env:..., key=value options) — only resolve a
            # plain executable name/path in the first positional.
            if (
                not isinstance(exe, str)
                or not exe
                or exe.startswith("env:")
                or "=" in exe
            ):
                return arguments
            from robotmcp.components.execution.desktop_launch_env import (
                get_effective_path,
                resolve_executable,
            )

            resolved = resolve_executable(exe)
            if resolved and resolved != exe:
                logger.info(
                    "PlatynUI desktop launch: resolved executable %r -> %r",
                    exe, resolved,
                )
                return [resolved] + list(arguments[1:])
            if resolved is None and not os.path.isabs(exe):
                logger.warning(
                    "PlatynUI desktop launch: executable %r is not resolvable on "
                    "the server PATH=%s",
                    exe, get_effective_path(),
                )
            return arguments
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("desktop executable resolution skipped: %s", exc)
            return arguments

    def _desktop_text_count_before(self, keyword, arguments):
        """Read the target text node's ``native:Text.CharacterCount`` via the
        shared native runtime DIRECTLY (non-reentrant — no RF re-execution under
        the lock), for the input-effect check
        (change: desktop-input-and-runtime-diagnostics).

        Only for keyboard interaction keywords with an explicit, resolvable
        target descriptor in ``arguments[0]`` (skips the focused-typing
        ``${None}`` form, where the target is ambiguous). Returns an int, or
        None when not applicable / unreadable. Best-effort; never raises.
        """
        try:
            base = keyword.strip().lower().rsplit(".", 1)[-1]
            if base not in ("keyboard type", "keyboard press", "keyboard release"):
                return None
            if not arguments:
                return None
            locator = arguments[0]
            if (
                not isinstance(locator, str)
                or not locator.strip()
                or locator.strip() in ("${None}", "${none}", "None")
                or "=" in locator
            ):
                return None
            from robotmcp.plugins.builtin.platynui_plugin import get_runtime

            rt = get_runtime()
            if rt is None:
                return None
            node = None
            ev = getattr(rt, "evaluate_single", None) or getattr(rt, "evaluate", None)
            if ev is None:
                return None
            res = ev(locator)
            node = res[0] if isinstance(res, (list, tuple)) and res else res
            if node is None:
                return None
            val = _read_native_character_count(node)
            return int(val) if val is not None else None
        except Exception as exc:  # pragma: no cover - env dependent
            logger.debug("desktop input-effect snapshot skipped: %s", exc)
            return None

    @staticmethod
    def _is_desktop_keyboard_keyword(session, keyword) -> bool:
        """True when this is a PlatynUI keyboard keyword on a desktop session
        (change: fix-platynui-windows-runtime, F16). Used to release held
        modifiers when a keyboard keyword fails or is killed mid-chord."""
        try:
            is_desktop = getattr(session, "is_desktop_session", None)
            if not (callable(is_desktop) and is_desktop() is True):
                return False
            norm = str(keyword or "").lower().rsplit(".", 1)[-1].strip()
            return norm in (
                "keyboard press", "keyboard type", "keyboard release",
            )
        except Exception:  # pragma: no cover - defensive
            return False

    def _platynui_safety_guard(self, session, keyword):
        """Active-desktop safety guard for PlatynUI interaction keywords
        (change: platynui-desktop-safety-isolation).

        Returns the safety outcome dict for interaction keywords (so the caller
        can refuse non-isolated displays), or None when the guard does not
        apply. Bypassed/warn runs are logged at WARNING for auditability.
        """
        from robotmcp.components.execution.platynui_focus import (
            is_interaction_keyword,
        )

        if not is_interaction_keyword(keyword):
            return None
        from robotmcp.components.execution.desktop_display_safety import (
            evaluate_safety,
        )

        outcome = evaluate_safety(session)
        if outcome["bypassed"] or (not outcome["enforcing"]):
            logger.warning(
                "PlatynUI safety guard: %s desktop (%s) — %s",
                "bypassed on" if outcome["bypassed"] else "warn-only on",
                outcome["classification"], outcome["reason"],
            )
        elif not outcome["allowed"]:
            logger.warning(
                "PlatynUI safety guard: refused on %s desktop",
                outcome["classification"],
            )
        elif outcome["classification"] == "windows":
            # F4: one-time warning that Windows automation drives the active
            # desktop (there is no isolated-display model on Windows).
            if not getattr(session, "_platynui_windows_warned", False):
                try:
                    session._platynui_windows_warned = True
                except Exception:
                    pass
                logger.warning(
                    "PlatynUI: driving the ACTIVE Windows desktop — %s",
                    outcome.get("reason") or "",
                )
        return outcome

    def _platynui_focus_before_act(self, session, keyword, arguments):
        """Ensure the AUT window is focused/visible/in-scope before a
        PlatynUI pointer/keyboard keyword (change: platynui-focused-execution).

        Reads policy from the session (defaults: focus ON, warn-not-fail):
          - ``platynui_no_focus`` (bool): per-session escape hatch
          - ``platynui_fail_on_hidden`` (bool): fail fast on non-visible AUT
          - ``platynui_strict_scope`` (bool): fail fast on cross-window target

        Returns a FocusOutcome (or None when not applicable). Raises
        FocusError when a strict/fail-fast precondition is violated.
        """
        from robotmcp.components.execution.platynui_focus import (
            PlatynUIFocusManager,
            is_interaction_keyword,
        )

        if not is_interaction_keyword(keyword):
            return None
        if self._platynui_focus_manager is None:
            self._platynui_focus_manager = PlatynUIFocusManager()
        # ADR-031 dirty flag: a desktop launch may have changed window
        # identities — drop the verified-activation cache (task 2.5).
        if getattr(session, "desktop_tree_dirty", False):
            try:
                self._platynui_focus_manager.invalidate_focus_cache()
            except Exception:
                pass
        focus = not bool(getattr(session, "platynui_no_focus", False))
        fail_on_hidden = bool(getattr(session, "platynui_fail_on_hidden", False))
        strict_scope = bool(getattr(session, "platynui_strict_scope", False))
        aut_pid = getattr(session, "desktop_aut_pid", None)
        aut_sid = getattr(session, "desktop_aut_sid", None)
        # Target highlighting: default ON, per-session opt-out + env kill
        # switch (change: platynui-visible-safe-targeting, task 3.1).
        highlight = bool(getattr(session, "platynui_highlight", True))
        outcome = self._platynui_focus_manager.ensure_focused(
            keyword,
            list(arguments or []),
            focus=focus,
            check_scope=True,
            strict_scope=strict_scope,
            fail_on_hidden=fail_on_hidden,
            aut_pid=aut_pid if isinstance(aut_pid, int) else None,
            aut_sid=aut_sid if isinstance(aut_sid, int) else None,
            highlight=highlight,
        )
        # D7 one-shot (change: desktop-evidence-and-display-scoping): the
        # blind type-at-focus warning fires once per session, mirroring
        # desktop_wayland_warned.
        try:
            from robotmcp.components.execution.platynui_focus import (
                UNFOCUSED_TYPING_WARNING,
            )

            if outcome is not None and UNFOCUSED_TYPING_WARNING in outcome.warnings:
                if getattr(session, "desktop_unfocused_typing_warned", False):
                    outcome.warnings = [
                        w for w in outcome.warnings
                        if w != UNFOCUSED_TYPING_WARNING
                    ]
                else:
                    session.desktop_unfocused_typing_warned = True
        except Exception:  # pragma: no cover - defensive
            pass
        return outcome

    @staticmethod
    def _screenshot_allowed_roots() -> list:
        """Roots a desktop screenshot may be written under (D3)."""
        import tempfile

        roots = [tempfile.gettempdir()]
        extra = os.environ.get("ROBOTMCP_SCREENSHOT_DIR", "").strip()
        if extra:
            roots.append(extra)
        return roots

    def _screenshot_path_guard(self, keyword, arguments):
        """Refuse desktop screenshot paths outside the allowed roots with an
        actionable hint (change: desktop-evidence-and-display-scoping, D3).
        Returns an error result dict, or None to proceed."""
        from robotmcp.components.execution.desktop_execution_signals import (
            screenshot_request_path,
        )

        requested = screenshot_request_path(keyword, arguments)
        if not requested or not os.path.isabs(requested):
            return None
        real = os.path.realpath(requested)
        for root in self._screenshot_allowed_roots():
            root_real = os.path.realpath(root)
            if real == root_real or real.startswith(root_real + os.sep):
                return None
        roots = ", ".join(self._screenshot_allowed_roots())
        return {
            "success": False,
            "error": (
                f"screenshot path '{requested}' is outside the allowed "
                f"roots ({roots})"
            ),
            "keyword": keyword,
            "hints": [{
                "type": "screenshot_path_refused",
                "message": (
                    f"Desktop screenshots may only be written under: {roots}. "
                    "Set ROBOTMCP_SCREENSHOT_DIR to allow an additional root."
                ),
            }],
        }

    def _unscoped_locator_guard(self, session, keyword, arguments):
        """Refuse a desktop Query/Evaluate whose XPath is unscoped (//-rooted),
        before the native walk runs (change: desktop-unscoped-locator-guardrail).

        A leading // re-walks the whole session AT-SPI tree (every desktop app),
        taking tens of seconds on a busy desktop — long enough to exceed the MCP
        client's request timeout and kill the transport. Returns an error dict
        to refuse, None to proceed. Honors an explicit opt-out (env or session)
        that downgrades the refusal to a one-time warning.
        """
        from robotmcp.components.execution.desktop_execution_signals import (
            is_query_keyword,
            is_unscoped_locator,
        )

        # Desktop-only (defensive — the call site already gates on this).
        _is_desktop = getattr(session, "is_desktop_session", None)
        if not (callable(_is_desktop) and _is_desktop() is True):
            return None
        if not is_query_keyword(keyword):
            return None
        args = list(arguments or [])
        if not args or not is_unscoped_locator(args[0]):
            return None

        locator = str(args[0]).strip()
        app = self._infer_session_app_name(session)
        rewrite = (
            f"/app:*[@Name='{app}']{locator}" if app
            else f"/app:*[@Name='<app>']{locator}"
        )

        # Opt-out: deliberate desktop-wide search. Downgrade to a one-time
        # warning rather than a refusal (mirrors desktop_wayland_warned).
        opted_in = (
            os.environ.get("ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED", "").strip().lower()
            in {"1", "true", "yes"}
            or bool(getattr(session, "platynui_allow_unscoped", False))
        )
        if opted_in:
            # Proceed, but surface a ONE-TIME warning (consumed + flag-flipped
            # in the post-execution result block so it rides on success OR
            # failure).
            if not getattr(session, "desktop_unscoped_warned", False):
                try:
                    session._pending_unscoped_hint = {
                        "type": "unscoped_desktop_locator_allowed",
                        "message": (
                            f"unscoped locator '{locator}' is walking the whole "
                            "session UI tree (ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED "
                            "opt-in) — this can take tens of seconds on a busy "
                            "desktop. Scope to /app:*[@Name='X']//… when possible."
                        ),
                    }
                except Exception:
                    pass
            return None  # proceed

        return {
            "success": False,
            "error": (
                f"unscoped desktop locator '{locator}' would walk the whole "
                "session UI tree (every desktop application) and can take tens "
                "of seconds — refused before dispatch"
            ),
            "keyword": keyword,
            "hints": [{
                "type": "unscoped_desktop_locator",
                "message": (
                    "NEVER start a desktop locator with // — it is absolute "
                    "XPath and ignores Set Root, re-walking every application "
                    "on the session AT-SPI bus. Scope to the application "
                    f"instead, e.g. '{rewrite}'. To size a subtree first, "
                    "count() is allowed (e.g. count(//control:Button)). "
                    "Set ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED=1 only for a "
                    "deliberate desktop-wide search."
                ),
            }],
        }

    def _screenshot_signature_guard(self, session, keyword, arguments):
        """Refuse a desktop Take Screenshot whose DESCRIPTOR slot holds a bare
        image path — the filename-as-descriptor trap that resolves for ~30s
        before ElementNotFound (change: desktop-screenshot-failfast). Returns an
        error dict to refuse, None to proceed. Opt-out downgrades to a one-time
        warning.
        """
        from robotmcp.components.execution.desktop_execution_signals import (
            screenshot_path_in_descriptor_slot,
        )

        _is_desktop = getattr(session, "is_desktop_session", None)
        if not (callable(_is_desktop) and _is_desktop() is True):
            return None
        path = screenshot_path_in_descriptor_slot(keyword, arguments)
        if path is None:
            return None

        opted_in = (
            os.environ.get("ROBOTMCP_PLATYNUI_ALLOW_PATH_DESCRIPTOR", "").strip().lower()
            in {"1", "true", "yes"}
            or bool(getattr(session, "platynui_allow_path_descriptor", False))
        )
        if opted_in:
            if not getattr(session, "desktop_path_descriptor_warned", False):
                try:
                    session.desktop_path_descriptor_warned = True
                    session._pending_path_descriptor_hint = {
                        "type": "screenshot_path_descriptor_allowed",
                        "message": (
                            f"'{path}' sits in the Take Screenshot DESCRIPTOR slot "
                            "(ALLOW_PATH_DESCRIPTOR opt-in); if it is a filename, "
                            "pass filename= instead."
                        ),
                    }
                except Exception:
                    pass
            return None

        return {
            "success": False,
            "error": (
                f"Take Screenshot received the path '{path}' in its DESCRIPTOR slot "
                "(first positional) — it resolves as a UI node and hangs ~30s "
                "before failing. Refused before dispatch."
            ),
            "keyword": keyword,
            "hints": [{
                "type": "screenshot_signature",
                "message": (
                    "Take Screenshot signature is (descriptor, filename, rect). "
                    f"For the whole desktop:  Take Screenshot  filename={path} . "
                    f"For one element:  Take Screenshot  <descriptor>  {path} . "
                    "Set ROBOTMCP_PLATYNUI_ALLOW_PATH_DESCRIPTOR=1 to bypass."
                ),
            }],
        }

    def _control_window_guard(self, session, keyword, arguments):
        """Refuse a Linux desktop tree-resolving keyword whose locator uses
        control:Window — AT-SPI windows are control:Frame, so control:Window
        matches nothing and hangs ~30s (change: desktop-screenshot-failfast).
        Refuse-with-hint (never auto-rewrite: recorded steps must match what
        executed). Opt-out downgrades to a one-time warning.
        """
        import sys

        if sys.platform != "linux":
            return None
        from robotmcp.components.execution.desktop_execution_signals import (
            control_window_locator,
        )

        _is_desktop = getattr(session, "is_desktop_session", None)
        if not (callable(_is_desktop) and _is_desktop() is True):
            return None
        locator = control_window_locator(keyword, arguments)
        if locator is None:
            return None

        opted_in = (
            os.environ.get("ROBOTMCP_PLATYNUI_ALLOW_CONTROL_WINDOW", "").strip().lower()
            in {"1", "true", "yes"}
            or bool(getattr(session, "platynui_allow_control_window", False))
        )
        if opted_in:
            if not getattr(session, "desktop_control_window_warned", False):
                try:
                    session.desktop_control_window_warned = True
                    session._pending_control_window_hint = {
                        "type": "control_window_allowed",
                        "message": (
                            f"'{locator}' uses control:Window on Linux "
                            "(ALLOW_CONTROL_WINDOW opt-in); AT-SPI windows are "
                            "control:Frame."
                        ),
                    }
                except Exception:
                    pass
            return None

        rewrite = locator.replace("control:Window", "control:Frame").replace(
            "control:window", "control:Frame"
        )
        return {
            "success": False,
            "error": (
                f"desktop locator '{locator}' uses control:Window, which matches "
                "nothing on Linux AT-SPI (windows are control:Frame) and hangs "
                "~30s — refused before dispatch."
            ),
            "keyword": keyword,
            "hints": [{
                "type": "control_window_on_linux",
                "message": (
                    "On Linux AT-SPI, application windows are control:Frame, not "
                    f"control:Window. Use '{rewrite}'. Set "
                    "ROBOTMCP_PLATYNUI_ALLOW_CONTROL_WINDOW=1 to bypass."
                ),
            }],
        }

    @staticmethod
    def _infer_session_app_name(session) -> Optional[str]:
        """Best-effort AUT application name for a scoped-locator rewrite hint:
        the launched-process basename when known, else None."""
        try:
            import os as _os

            pid = getattr(session, "desktop_aut_pid", None)
            if isinstance(pid, int):
                comm = f"/proc/{pid}/comm"
                if _os.path.exists(comm):
                    with open(comm) as f:
                        name = f.read().strip()
                    if name:
                        return name
        except Exception:
            pass
        return None

    @staticmethod
    def _maybe_recover_screenshot_result(session, keyword, arguments, result) -> None:
        """Flip a desktop screenshot failure to success when the requested
        file verifiably exists (change: desktop-evidence-and-display-scoping,
        D3). Upstream PlatynUI ``take_screenshot`` WRITES the file, then
        crashes building the html log link (``filepath.relative_to(outputdir)``)
        for absolute paths outside the RF output dir — the artifact is real,
        only the log link failed. Mutates ``result`` in place; never raises.
        """
        if not isinstance(result, dict) or result.get("success"):
            return
        try:
            err_txt = str(result.get("error") or "")
            if "is not in the subpath of" not in err_txt:
                return
            from robotmcp.components.execution.desktop_execution_signals import (
                screenshot_request_path,
            )

            requested = screenshot_request_path(keyword, arguments)
            is_desktop = getattr(session, "is_desktop_session", None)
            if (
                requested
                and os.path.isabs(requested)
                and callable(is_desktop)
                and is_desktop() is True
                and os.path.isfile(requested)
                and os.path.getsize(requested) > 0
            ):
                result["success"] = True
                result["error"] = None
                result["result"] = requested
                result.setdefault("hints", []).append({
                    "type": "screenshot_path_recovered",
                    "message": (
                        f"Screenshot was written to '{requested}'; the "
                        "keyword's failure was an upstream log-link quirk "
                        "for paths outside the RF output dir and has been "
                        "reclassified as success."
                    ),
                })
        except Exception:  # pragma: no cover - defensive
            pass

    def _requires_pre_validation(
        self,
        keyword: str,
        session: object | None = None,
    ) -> bool:
        """Check if a keyword requires element pre-validation.

        Policy: the curated ``ELEMENT_INTERACTION_KEYWORDS`` set is the
        source of truth. The library-aware introspector vetoes ONLY when
        it returns a definitive False (keyword resolved to a specific
        library whose signature has no locator-style arg). On ``None``
        (ambiguous / not found / unresolved), the positive list wins.
        """
        keyword_lower = keyword.lower().strip()
        if keyword_lower not in self.ELEMENT_INTERACTION_KEYWORDS:
            return False
        # Desktop sessions (PlatynUI, ADR-025): no DOM, no JS probes —
        # the native runtime handles descriptor resolution with its own
        # retry. Pre-validation never applies.
        # (Strict `is True` so MagicMock sessions in tests don't match.)
        try:
            checker = getattr(session, "is_desktop_session", None)
            if callable(checker) and checker() is True:
                return False
        except Exception:
            pass
        # Definitive veto only: if the introspector confidently says the
        # keyword's library has no locator arg, skip pre-validation. None /
        # ambiguous results do NOT veto — preserves the curated list.
        try:
            takes_locator = self._locator_introspector.keyword_takes_locator(
                keyword, session=session,
            )
        except Exception:
            takes_locator = None
        if takes_locator is False:
            return False
        return True

    def _get_action_type_from_keyword_for_states(self, keyword: str) -> str:
        """Extract the action type from a keyword name for state requirements."""
        keyword_lower = keyword.lower()
        if "clear" in keyword_lower:
            return "clear"
        elif "fill" in keyword_lower or "input" in keyword_lower or "type" in keyword_lower:
            return "fill"
        elif "uncheck" in keyword_lower:
            return "uncheck"
        elif "check" in keyword_lower:
            return "check"
        elif "upload" in keyword_lower or "choose file" in keyword_lower:
            return "upload"
        elif "deselect" in keyword_lower or "select" in keyword_lower:
            return "select"
        elif "drag" in keyword_lower:
            return "drag"
        elif "tap" in keyword_lower or "long press" in keyword_lower:
            return "tap"
        elif "click" in keyword_lower:
            return "click"
        elif "submit" in keyword_lower:
            return "submit"
        elif "open context" in keyword_lower:
            return "open"
        elif "press" in keyword_lower or "key" in keyword_lower:
            return "press"
        elif "focus" in keyword_lower:
            return "focus"
        elif "hover" in keyword_lower or "mouse" in keyword_lower:
            return "hover"
        elif "scroll" in keyword_lower:
            return "scroll"
        return "click"

    # Locator strategies that rely on Selenium's internal link-text matching.
    # Pre-validation uses Get WebElements + JS visibility checks which don't
    # handle link text with embedded child elements (SVG icons, badge spans)
    # correctly — the text content includes children's text and whitespace,
    # causing false "not found" failures.  Skip pre-validation for these.
    # Note: SeleniumLibrary accepts both ":" and "=" as prefix separators.
    _SKIP_PRE_VALIDATION_LOCATOR_PREFIXES = (
        "link=", "partial link=", "link:", "partial link:",
    )

    # Keywords that use keyword-specific locator resolution (tag-constrained
    # default strategy with extended key_attrs) which pre-validation's generic
    # Get WebElements cannot replicate.  For these keywords, bare-text locators
    # (no explicit prefix like id=, css=, xpath=) are skipped from pre-validation.
    _KEYWORD_SPECIFIC_LOCATOR_KEYWORDS = frozenset({
        "click link", "click image", "click button",
    })

    # Generic locator prefixes that Get WebElements understands — if one of
    # these is present, pre-validation CAN run even for Click Link/Click Image.
    _GENERIC_LOCATOR_PREFIXES = (
        "id=", "id:", "css=", "css:", "xpath=", "xpath:",
        "name=", "name:", "class=", "class:", "tag=", "tag:",
        "dom=", "dom:", "jquery=", "jquery:", "sizzle=", "sizzle:",
        "data=", "data:", "identifier=", "identifier:",
        "//",   # implicit xpath
    )

    def _extract_locator_from_args(self, keyword: str, arguments: List[Any]) -> Optional[str]:
        """Extract the element locator from keyword arguments.

        Returns None (skip pre-validation) when the locator uses a strategy
        that pre-validation's generic element lookup cannot replicate:

        1. Explicit link-text prefixes (link=, link:, partial link=, partial link:)
        2. Click Link / Click Image / Click Button with bare text (no prefix) —
           these keywords use tag-constrained default strategies (searching
           @href, @src, @alt, @value, normalize-space text) that
           Get WebElements (tag=None, key_attrs=[@id, @name]) does not support.
        """
        if not arguments:
            return None
        first_arg = arguments[0]
        if not isinstance(first_arg, str):
            return None
        # Skip pre-validation for link-text locator prefixes
        if first_arg.lower().startswith(self._SKIP_PRE_VALIDATION_LOCATOR_PREFIXES):
            logger.debug(
                f"Skipping pre-validation for link-text locator: {first_arg}"
            )
            return None
        # Click Link / Click Image / Click Button: bare text (no generic prefix)
        # uses keyword-specific resolution with tag-constrained key_attrs
        # (e.g. tag="a" → @href/text, tag="button" → @value/text,
        # tag="img" → @src/@alt).  Pre-validation's Get WebElements (tag=None)
        # only searches @id/@name, so it would false-reject valid locators.
        # Let SeleniumLibrary handle these natively for correct error messages.
        if keyword.lower() in self._KEYWORD_SPECIFIC_LOCATOR_KEYWORDS:
            if not first_arg.lower().startswith(self._GENERIC_LOCATOR_PREFIXES):
                logger.debug(
                    f"Skipping pre-validation for {keyword} bare-text locator: "
                    f"{first_arg}"
                )
                return None
        return first_arg

    @staticmethod
    def _rank_and_deduplicate_hints(
        hints: List[Dict[str, Any]], detail_level: str = "minimal"
    ) -> List[Dict[str, Any]]:
        """Deduplicate and rank hints: 1 primary + up to 2 secondary for non-full levels."""
        if not hints or detail_level == "full":
            return hints

        # Deduplicate by message content (exact + substring containment)
        seen_messages: list[str] = []
        unique: list[Dict[str, Any]] = []
        for h in hints:
            msg = (h.get("message") or h.get("title") or str(h)).lower()
            is_dup = False
            for seen in seen_messages:
                if msg in seen or seen in msg:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(h)
                seen_messages.append(msg)

        # Cap: 1 primary + 2 secondary for minimal/standard
        max_hints = 1 if detail_level == "minimal" else 3
        return unique[:max_hints]

    def _add_link_image_locator_guidance(
        self,
        keyword: str,
        arguments: List[Any],
        hints: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add locator guidance hints when Click Link/Image/Button fails.

        SeleniumLibrary keywords use tag-specific locator strategies that
        differ from the generic ``Get WebElements`` default (which only
        checks ``@id`` and ``@name``).  When bare-text locators fail,
        these hints guide users toward working alternatives.

        Returns the hints list (possibly extended).
        """
        keyword_lower = keyword.lower()
        if keyword_lower not in self._KEYWORD_SPECIFIC_LOCATOR_KEYWORDS:
            return hints
        if not arguments:
            return hints
        locator = arguments[0]
        if not isinstance(locator, str):
            return hints
        # Only add guidance for bare-text locators (no explicit prefix).
        # Locators with prefixes (css:, id=, xpath:, link:, etc.) already
        # indicate the user knows which strategy to use.
        if locator.lower().startswith(
            self._GENERIC_LOCATOR_PREFIXES
            + self._SKIP_PRE_VALIDATION_LOCATOR_PREFIXES
        ):
            return hints

        if keyword_lower == "click link":
            hints = list(hints)  # copy to avoid mutating caller's list
            hints.append({
                "type": "link_locator_guidance",
                "message": (
                    "Click Link with bare text uses SeleniumLibrary's default "
                    "XPath strategy which can fail on links containing embedded "
                    "child elements (SVGs, badge spans, icons)"
                ),
                "suggestion": (
                    f"Try 'partial link:{locator}' for substring text match, "
                    f"or use a css/xpath locator targeting the href attribute"
                ),
                "alternatives": [
                    f"partial link:{locator}",
                    "css:a[href='/your-path']",
                ],
            })
        elif keyword_lower == "click image":
            hints = list(hints)
            hints.append({
                "type": "image_locator_guidance",
                "message": (
                    "Click Image with bare text uses SeleniumLibrary's default "
                    "XPath strategy searching @id, @name, @src, @alt which may "
                    "not match dynamically loaded or lazily-rendered images"
                ),
                "suggestion": (
                    f"Try a css or xpath locator targeting the image's src or "
                    f"alt attribute directly"
                ),
                "alternatives": [
                    f"css:img[alt='{locator}']",
                    f"xpath://img[contains(@src, '{locator}')]",
                ],
            })
        elif keyword_lower == "click button":
            hints = list(hints)
            hints.append({
                "type": "button_locator_guidance",
                "message": (
                    "Click Button with bare text searches <input> by "
                    "id/name/value then <button> by id/name/value/text. "
                    "If the button has no id, name, or value attributes, "
                    "use a css or xpath locator targeting it directly"
                ),
                "suggestion": (
                    f"Try 'xpath://button[normalize-space()=\"{locator}\"]' "
                    f"or a css selector like 'css:button[type=submit]'"
                ),
                "alternatives": [
                    f"xpath://button[normalize-space()='{locator}']",
                    "css:button[type=submit]",
                ],
            })
        return hints

    async def _pre_validate_element(
        self,
        locator: str,
        session: "ExecutionSession",
        keyword: str,
        timeout_ms: Optional[int] = None,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Quick pre-validation check if element is actionable before attempting action.

        This performs a fast check (default 500ms timeout) to verify element state
        before the full keyword execution, enabling early failure detection.

        Returns:
            Tuple of (is_valid, error_message, details_dict)
        """
        if timeout_ms is None:
            timeout_ms = self.config.PRE_VALIDATION_TIMEOUT

        start_time = time.time()
        action_type = self._get_action_type_from_keyword_for_states(keyword)
        required_states = self.REQUIRED_STATES_FOR_ACTION.get(action_type, {"visible"})

        details: Dict[str, Any] = {
            "locator": locator,
            "keyword": keyword,
            "action_type": action_type,
            "required_states": list(required_states),
            "timeout_ms": timeout_ms,
        }

        try:
            # Ensure ctx.test is set for BuiltIn.run_keyword() support.
            # All pre-validation paths call BuiltIn.run_keyword() which internally
            # does kw.run(result, ctx).  When ctx.test is None, RF falls back to
            # ctx.suite.setup — a running-model Keyword without .body — causing
            # AttributeError: 'Keyword' object has no attribute 'body'.
            try:
                from robot.running.context import EXECUTION_CONTEXTS as _EC
                _ctx = _EC.current
                if _ctx and not _ctx.test:
                    from robot.result.model import TestCase as _ResTest
                    _ctx.test = _ResTest(name="MCP_PreValidation")
            except Exception:
                pass

            # Determine owning library directly from RF namespace resolution.
            # This is authoritative: it uses the same resolution RF will use
            # to execute the keyword, respecting search order and library
            # imports.  Replaces fragile active_library / pattern matching.
            active_library = None
            try:
                if _ctx:
                    _runner = _ctx.namespace.get_runner(keyword)
                    _owner = getattr(getattr(_runner, 'keyword', None), 'owner', None)
                    _owner_name = getattr(_owner, 'name', None)
                    if _owner_name == "Browser":
                        active_library = "browser"
                    elif _owner_name == "SeleniumLibrary":
                        active_library = "selenium"
                    elif _owner_name == "AppiumLibrary":
                        active_library = "appium"
                    # BuiltIn, Collections, etc. → None → skip pre-validation
            except Exception:
                pass  # No RF context → skip pre-validation

            if active_library == "browser":
                result = await self._pre_validate_browser_element(locator, required_states, timeout_ms)
            elif active_library == "selenium":
                result = await self._pre_validate_selenium_element(locator, required_states, timeout_ms)
            elif active_library == "appium":
                result = await self._pre_validate_appium_element(locator, required_states, timeout_ms)
            else:
                logger.debug(f"Pre-validation skipped: no active browser for {keyword}")
                return True, None, {"skipped": True, "reason": "no_active_browser"}

            elapsed_ms = (time.time() - start_time) * 1000
            details["elapsed_ms"] = round(elapsed_ms, 2)

            if result["valid"]:
                details["current_states"] = result.get("states", [])
                logger.debug(f"Pre-validation passed for '{locator}' in {elapsed_ms:.1f}ms")
                return True, None, details
            else:
                details["current_states"] = result.get("states", [])
                details["missing_states"] = result.get("missing", [])
                error_msg = result.get("error", f"Element not actionable: {locator}")
                logger.warning(f"Pre-validation failed for '{locator}' in {elapsed_ms:.1f}ms: {error_msg}")
                return False, error_msg, details

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            details["elapsed_ms"] = round(elapsed_ms, 2)
            details["exception"] = str(e)
            error_msg = f"Element not found or inaccessible: {locator}"
            logger.warning(f"Pre-validation exception for '{locator}': {e}")
            return False, error_msg, details

    # Retry tunables for transient pre-validation failures (slow page settle,
    # late-mounting elements). Bounded to one extra attempt with a short gap
    # so the worst-case cost on a genuine miss is roughly 2x the single-shot
    # budget plus the gap — small enough to not change end-to-end UX, large
    # enough to absorb the ~700ms settle windows we saw on real sites in
    # the 2026-05-17 Tricentis benchmark.
    PRE_VALIDATION_RETRY_GAP_MS: int = 200
    PRE_VALIDATION_MAX_RETRIES: int = 1

    async def _pre_validate_element_with_retry(
        self,
        locator: str,
        session: "ExecutionSession",
        keyword: str,
        timeout_ms: Optional[int] = None,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Wrap ``_pre_validate_element`` with a single-shot retry on transient
        failures (slow-loading pages, brief animations, late-mounting elements).

        Behaviour:
        - First call uses the configured/passed ``timeout_ms`` exactly as
          before — happy path is byte-for-byte identical and pays NO extra
          latency.
        - If the first call returns ``is_valid=False``, sleep
          ``PRE_VALIDATION_RETRY_GAP_MS`` (default 200 ms) and call again.
        - On the retry, surface ``details["retries"] = 1`` and
          ``details["first_attempt_error"] = <previous error_msg>`` so the
          failure-hint builder and tests can distinguish a real miss from
          a transient settle.
        - The retry count is capped by ``PRE_VALIDATION_MAX_RETRIES`` (1).
          This is deliberately tight; if a page truly needs >1.5 s to
          settle, the user should pass ``pre_validate_timeout_ms`` explicitly
          rather than burning latency on every step.
        """
        is_valid, error_msg, details = await self._pre_validate_element(
            locator, session, keyword, timeout_ms=timeout_ms,
        )
        if is_valid:
            return True, None, details

        for attempt in range(1, self.PRE_VALIDATION_MAX_RETRIES + 1):
            logger.debug(
                f"Pre-validation retry {attempt} for '{locator}' after {self.PRE_VALIDATION_RETRY_GAP_MS}ms"
            )
            await asyncio.sleep(self.PRE_VALIDATION_RETRY_GAP_MS / 1000.0)
            retry_is_valid, retry_error, retry_details = await self._pre_validate_element(
                locator, session, keyword, timeout_ms=timeout_ms,
            )
            if retry_details is None:
                retry_details = {}
            retry_details["retries"] = attempt
            retry_details["first_attempt_error"] = error_msg
            if retry_is_valid:
                logger.debug(
                    f"Pre-validation recovered for '{locator}' on retry {attempt}"
                )
                return True, None, retry_details
            # Still failing — update the error message but keep iterating
            # until we hit the cap.
            is_valid, error_msg, details = retry_is_valid, retry_error, retry_details

        return False, error_msg, details

    # Playwright's cascaded-selector separator — when present, the
    # locator is a compound that the Browser library's own selector
    # parser handles, NOT a plain ``id=X`` form.
    _CASCADED_SEPARATOR: str = ">>"

    @classmethod
    def _normalize_locator_for_browser_prevalidation(cls, locator: str) -> str:
        """Rewrite ``id=X`` to its CSS attribute-selector equivalent
        ``[id="X"]`` for the Browser-library pre-validation call only.

        The Browser library documents ``id=X`` and ``css=[id="X"]`` as
        equivalent (per its libdoc explicit-strategies table). In practice
        they take different code paths through the selector parser and
        have been observed to produce DIFFERENT pre-validation verdicts
        on the same DOM element under slow-load conditions (see OBS-01 /
        2026-05-17 Tricentis Obstacle 3 — ``id=generate`` reported
        'detached' while ``css=#generate`` passed immediately).

        This rewrite forces ``id=X`` through the CSS engine path, the
        same path ``css=#X`` (rewritten to bare ``#X`` by the locator
        converter) already uses. The attribute-selector form
        ``[id="X"]`` is safer than ``#X`` because it handles id values
        containing CSS-special characters (dots, colons) without
        needing per-char escapes — the double-quoted attribute value
        accepts them literally.

        Important: this rewrite is local to the pre-validation gate.
        The original ``id=X`` form is still what the actual keyword
        executes against AND what ``build_test_suite`` records — RF
        suite convention is preserved.

        Implementation note: the original draft used a regex
        (``^\\s*id\\s*=\\s*(?P<value>\\S.*?)\\s*$``) which SonarCloud's
        S5852 rule correctly flagged as a polynomial-backtracking
        shape (lazy ``.*?`` overlapping with trailing ``\\s*$``).
        Empirically bounded to ~0.6ms at the 10k-char input limit but
        still the kind of pattern the rule warns about. Replaced with
        explicit string parsing: provably O(n), no regex engine
        involved, 10–30× faster on adversarial inputs.
        """
        if not locator:
            return locator
        # Cascaded forms (`id=foo >> nth=0`) go through Browser library's
        # own selector parser — that path doesn't have the `id=` flake,
        # and trying to rewrite would mangle the cascade. Leave alone.
        if cls._CASCADED_SEPARATOR in locator:
            return locator
        # Strip outer whitespace, then parse `id<ws>?=<ws>?<value>` by
        # explicit string operations. Each step is O(n) in the input
        # length, and there is no backtracking surface.
        stripped = locator.strip()
        if not stripped.startswith("id"):
            return locator
        # Whitespace between "id" and "=" is allowed (matches the
        # original regex's `\s*` semantics).
        after_id = stripped[2:].lstrip()
        if not after_id.startswith("="):
            return locator
        value = after_id[1:].strip()
        if not value:
            return locator
        # Escape embedded double quotes; ids containing literal " are
        # vanishingly rare but the rewrite must not produce malformed CSS.
        escaped = value.replace('"', '\\"')
        return f'[id="{escaped}"]'

    async def _pre_validate_browser_element(
        self, locator: str, required_states: Set[str], timeout_ms: int
    ) -> Dict[str, Any]:
        """Pre-validate element using Browser Library's Get Element States."""
        try:
            timeout_str = f"{timeout_ms}ms"
            # OBS-01: normalize id=X → [id="X"] so the pre-validation
            # gate's verdict is identical for id=X and css=#X / css=[id="X"].
            # The original `locator` is preserved for error messages
            # (constructed by the caller from the unmodified parameter).
            pv_locator = self._normalize_locator_for_browser_prevalidation(locator)
            result, error_info = await asyncio.to_thread(self._run_browser_get_states, pv_locator, timeout_str)

            if result is None:
                # Use the actual error message if available, otherwise generic message
                error_msg = error_info if error_info else f"Element not found: {locator}"
                return {"valid": False, "states": [], "missing": list(required_states),
                        "error": error_msg}

            current_states = set()
            if hasattr(result, "__iter__"):
                for state in result:
                    state_str = str(state).lower()
                    if "." in state_str:
                        state_str = state_str.split(".")[-1]
                    current_states.add(state_str)

            missing = required_states - current_states
            if missing:
                return {"valid": False, "states": list(current_states), "missing": list(missing),
                        "error": f"Element missing required states: {', '.join(sorted(missing))}"}

            return {"valid": True, "states": list(current_states), "missing": [], "error": None}

        except Exception as e:
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                return {"valid": False, "states": [], "missing": list(required_states),
                        "error": f"Element not found within {timeout_ms}ms: {locator}"}
            elif "not found" in error_str or "no element" in error_str:
                return {"valid": False, "states": [], "missing": list(required_states),
                        "error": f"Element not found: {locator}"}
            return {"valid": False, "states": [], "missing": list(required_states),
                    "error": f"Pre-validation error: {str(e)}"}

    def _run_browser_get_states(self, locator: str, timeout: str) -> tuple[Optional[Any], Optional[str]]:
        """Run Browser Library's Get Element States keyword.

        Returns:
            tuple: (result, error_info) where result is the states or None,
                   and error_info contains the actual error message if failed.

        Note: Get Element States doesn't accept timeout directly.
        We set browser timeout temporarily before the call.

        IMPORTANT: The browser timeout MUST be restored after pre-validation,
        otherwise subsequent keyword executions (like Click) will use the
        pre-validation timeout (500ms) instead of the intended action timeout.

        Thread safety: Uses _browser_timeout_lock to serialize timeout
        mutations so concurrent threads don't see the pre-validation timeout.
        """
        builtin = BuiltIn()

        # Acquire lock to prevent concurrent timeout mutations
        self._browser_timeout_lock.acquire()

        # Set timeout temporarily (Browser Library uses global timeout for element operations)
        # Note: Set Browser Timeout returns the previous timeout value, so we use that
        original_timeout = None
        timeout_was_set = False
        try:
            # Set new timeout and capture the previous value (returned by the keyword)
            original_timeout = builtin.run_keyword("Browser.Set Browser Timeout", timeout)
            timeout_was_set = True
        except Exception as e:
            # If we can't set timeout, log but continue anyway
            logger.debug(f"Failed to set browser timeout to {timeout}: {e}")

        def try_get_states(loc: str) -> tuple[Optional[Any], Optional[str]]:
            """Try to get element states with the given locator."""
            try:
                # Get Element States doesn't take timeout - uses global browser timeout
                result = builtin.run_keyword("Browser.Get Element States", loc)
                return result, None
            except Exception as e1:
                try:
                    result = builtin.run_keyword("Get Element States", loc)
                    return result, None
                except Exception as e2:
                    return None, str(e2)

        try:
            # First attempt with original locator
            result, error = try_get_states(locator)
            if result is not None:
                return result, None

            # Check if error is due to strict mode violation (multiple elements)
            if error and ("strict mode" in error.lower() or "resolved to" in error.lower() and "elements" in error.lower()):
                logger.debug(f"Strict mode violation for '{locator}': {error}. Trying with visible filter.")

                # Use shorter timeout for retry attempts to stay within budget
                try:
                    builtin.run_keyword("Browser.Set Browser Timeout", "200ms")
                except Exception:
                    pass

                # Try with >> visible=true filter to get only visible element
                visible_locator = f"{locator} >> visible=true"
                result, visible_error = try_get_states(visible_locator)
                if result is not None:
                    return result, None

                # If visible filter also failed, try with nth=0 as last resort
                nth_locator = f"{locator} >> nth=0"
                result, nth_error = try_get_states(nth_locator)
                if result is not None:
                    logger.debug(f"Got states using nth=0 selector for '{locator}'")
                    return result, None

                # Return informative error about multiple elements
                return None, f"Multiple elements found for '{locator}'. Tried visible filter and nth=0 but both failed. Original error: {error}"

            # Return the original error for other failure cases
            return None, error

        finally:
            # CRITICAL: Restore original timeout if we changed it
            # Failure to restore leaves browser at 500ms, causing subsequent
            # actions (like Click) to fail with timeout even if they succeed
            if timeout_was_set and original_timeout is not None:
                restore_success = False
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        builtin.run_keyword("Browser.Set Browser Timeout", original_timeout)
                        restore_success = True
                        break
                    except Exception as e:
                        if attempt < 2:
                            logger.debug(f"Retry {attempt + 1}/3: Failed to restore browser timeout to {original_timeout}: {e}")
                        else:
                            logger.warning(
                                f"CRITICAL: Failed to restore browser timeout to {original_timeout} after 3 attempts. "
                                f"Browser timeout may be stuck at {timeout}. Subsequent keyword executions may fail. "
                                f"Error: {e}"
                            )
                if restore_success:
                    logger.debug(f"Browser timeout restored to {original_timeout}")
            # Release lock after timeout is restored (or if it was never set)
            self._browser_timeout_lock.release()

    async def _pre_validate_selenium_element(
        self, locator: str, required_states: Set[str], timeout_ms: int
    ) -> Dict[str, Any]:
        """Pre-validate element using SeleniumLibrary checks."""
        try:
            return await asyncio.to_thread(
                self._run_selenium_state_check, locator, required_states, timeout_ms
            )
        except Exception as e:
            return {"valid": False, "states": [], "missing": list(required_states),
                    "error": f"Pre-validation error: {str(e)}"}

    def _run_selenium_state_check(
        self, locator: str, required_states: Set[str], timeout_ms: int
    ) -> Dict[str, Any]:
        """Run SeleniumLibrary state check using JavaScript.

        Handles multiple elements by finding the first visible one.

        Note: We temporarily set implicit wait for element lookup, then restore
        the original value to avoid affecting subsequent keyword executions.
        """
        builtin = BuiltIn()
        original_implicit_wait = None
        implicit_wait_was_set = False

        try:
            # Save and set implicit wait temporarily
            try:
                # Set Selenium Implicit Wait returns the previous value, so capture it
                original_implicit_wait = builtin.run_keyword(
                    "SeleniumLibrary.Set Selenium Implicit Wait", f"{timeout_ms / 1000}s"
                )
                implicit_wait_was_set = True
            except Exception as e:
                logger.debug(f"Failed to set Selenium implicit wait: {e}")

            # Try to get all matching elements to handle duplicates
            elements = []
            try:
                elements = builtin.run_keyword("SeleniumLibrary.Get WebElements", locator)
            except Exception:
                pass

            if not elements:
                # Fallback to single element lookup
                try:
                    element = builtin.run_keyword("SeleniumLibrary.Get WebElement", locator)
                    elements = [element] if element else []
                except Exception:
                    return {"valid": False, "states": [], "missing": list(required_states),
                            "error": f"Element not found: {locator}"}

            if not elements:
                return {"valid": False, "states": [], "missing": list(required_states),
                        "error": f"Element not found: {locator}"}

            # If multiple elements, find the first visible one
            element = None
            element_count = len(elements) if hasattr(elements, '__len__') else 1

            if element_count > 1:
                logger.debug(f"Found {element_count} elements for '{locator}', checking for visible one")
                for idx, el in enumerate(elements):
                    try:
                        is_visible = builtin.run_keyword("SeleniumLibrary.Execute Javascript",
                            "var el = arguments[0]; var style = window.getComputedStyle(el); "
                            "var rect = el.getBoundingClientRect(); "
                            "return style.display !== 'none' && style.visibility !== 'hidden' && "
                            "rect.width > 0 && rect.height > 0;", "ARGUMENTS", el)
                        if is_visible:
                            element = el
                            logger.debug(f"Using visible element at index {idx} for '{locator}'")
                            break
                    except Exception:
                        continue

                if element is None:
                    # No visible element found, use first one and let it fail with proper message
                    element = elements[0]
                    logger.debug(f"No visible element found among {element_count} elements for '{locator}'")
            else:
                element = elements[0] if elements else None

            js_check = """
            var el = arguments[0];
            var states = [];
            if (document.body.contains(el)) states.push('attached');
            var style = window.getComputedStyle(el);
            var rect = el.getBoundingClientRect();
            if (style.display !== 'none' && style.visibility !== 'hidden' &&
                rect.width > 0 && rect.height > 0) states.push('visible');
            if (!el.disabled) states.push('enabled');
            if (el.isContentEditable) states.push('editable');
            else if ((el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') && !el.readOnly)
                states.push('editable');
            if (el.checked !== undefined) states.push(el.checked ? 'checked' : 'unchecked');
            return states;
            """

            try:
                states_list = builtin.run_keyword("SeleniumLibrary.Execute Javascript", js_check, "ARGUMENTS", element)
                current_states = set(states_list) if states_list else set()
            except Exception:
                current_states = {"attached"}

            missing = required_states - current_states
            if missing:
                return {"valid": False, "states": list(current_states), "missing": list(missing),
                        "error": f"Element missing required states: {', '.join(sorted(missing))}"}

            return {"valid": True, "states": list(current_states), "missing": [], "error": None}

        except Exception as e:
            return {"valid": False, "states": [], "missing": list(required_states),
                    "error": f"Pre-validation error: {str(e)}"}

        finally:
            # Restore implicit wait if we changed it
            if implicit_wait_was_set and original_implicit_wait is not None:
                try:
                    builtin.run_keyword("SeleniumLibrary.Set Selenium Implicit Wait", original_implicit_wait)
                except Exception as e:
                    logger.debug(f"Failed to restore Selenium implicit wait to {original_implicit_wait}: {e}")

    async def _pre_validate_appium_element(
        self, locator: str, required_states: Set[str], timeout_ms: int
    ) -> Dict[str, Any]:
        """Pre-validate element using AppiumLibrary checks.

        Similar to SeleniumLibrary but uses AppiumLibrary keywords.
        Handles multiple elements by finding the first visible one.
        """
        try:
            return await asyncio.to_thread(
                self._run_appium_state_check, locator, required_states, timeout_ms
            )
        except Exception as e:
            return {"valid": False, "states": [], "missing": list(required_states),
                    "error": f"Pre-validation error: {str(e)}"}

    def _run_appium_state_check(
        self, locator: str, required_states: Set[str], timeout_ms: int
    ) -> Dict[str, Any]:
        """Run AppiumLibrary state check.

        Handles multiple elements by finding the first visible/enabled one.
        Temporarily sets Appium implicit wait to the pre-validation timeout
        to avoid blocking for the full default wait duration.
        """
        try:
            builtin = BuiltIn()

            # Set Appium implicit wait temporarily for fast pre-validation
            original_implicit_wait = None
            implicit_wait_was_set = False
            try:
                original_implicit_wait = builtin.run_keyword(
                    "AppiumLibrary.Set Appium Implicit Wait", f"{timeout_ms / 1000}"
                )
                implicit_wait_was_set = True
            except Exception as e:
                logger.debug(f"Failed to set Appium implicit wait: {e}")

            # Try to get all matching elements to handle duplicates
            elements = []
            try:
                elements = builtin.run_keyword("AppiumLibrary.Get Webelements", locator)
            except Exception:
                pass

            if not elements:
                # Fallback to single element lookup
                try:
                    element = builtin.run_keyword("AppiumLibrary.Get Webelement", locator)
                    elements = [element] if element else []
                except Exception:
                    return {"valid": False, "states": [], "missing": list(required_states),
                            "error": f"Element not found: {locator}"}

            if not elements:
                return {"valid": False, "states": [], "missing": list(required_states),
                        "error": f"Element not found: {locator}"}

            # If multiple elements, find the first visible one
            element = None
            element_count = len(elements) if hasattr(elements, '__len__') else 1

            if element_count > 1:
                logger.debug(f"Found {element_count} Appium elements for '{locator}', checking for visible one")
                for idx, el in enumerate(elements):
                    try:
                        # Check if element is displayed
                        is_displayed = el.is_displayed() if hasattr(el, 'is_displayed') else True
                        if is_displayed:
                            element = el
                            logger.debug(f"Using visible Appium element at index {idx} for '{locator}'")
                            break
                    except Exception:
                        continue

                if element is None:
                    element = elements[0]
                    logger.debug(f"No visible element found among {element_count} Appium elements for '{locator}'")
            else:
                element = elements[0] if elements else None

            # Check element states
            current_states = set()
            try:
                if element is not None:
                    current_states.add("attached")
                    if hasattr(element, 'is_displayed') and element.is_displayed():
                        current_states.add("visible")
                    if hasattr(element, 'is_enabled') and element.is_enabled():
                        current_states.add("enabled")
                    # For mobile, most editable elements are enabled input fields.
                    # ``tag_name`` may be a full Java class (Android:
                    # ``android.widget.EditText``) or an XCUI element type
                    # (iOS: ``XCUIElementTypeTextField``), so substring match
                    # against _EDITABLE_TAG_SUBSTRINGS rather than exact equality.
                    tag_name = element.tag_name.lower() if hasattr(element, 'tag_name') else ""
                    is_editable_tag = any(
                        token in tag_name for token in self._EDITABLE_TAG_SUBSTRINGS
                    )
                    if not is_editable_tag and hasattr(element, 'get_attribute'):
                        # React Native / Compose / Flutter wrappers can hide the
                        # underlying widget behind a generic tag while still
                        # exposing the real class via the ``className`` attr.
                        try:
                            class_attr = (element.get_attribute("className") or "").lower()
                        except Exception:
                            class_attr = ""
                        if class_attr and any(
                            token in class_attr for token in self._EDITABLE_TAG_SUBSTRINGS
                        ):
                            is_editable_tag = True
                    if is_editable_tag and "enabled" in current_states:
                        current_states.add("editable")
            except Exception:
                current_states = {"attached"}

            missing = required_states - current_states
            if missing:
                return {"valid": False, "states": list(current_states), "missing": list(missing),
                        "error": f"Element missing required states: {', '.join(sorted(missing))}"}

            return {"valid": True, "states": list(current_states), "missing": [], "error": None}

        except Exception as e:
            return {"valid": False, "states": [], "missing": list(required_states),
                    "error": f"Pre-validation error: {str(e)}"}

        finally:
            # Restore Appium implicit wait if we changed it
            if implicit_wait_was_set and original_implicit_wait is not None:
                try:
                    builtin.run_keyword(
                        "AppiumLibrary.Set Appium Implicit Wait", original_implicit_wait
                    )
                except Exception as e:
                    logger.debug(
                        f"Failed to restore Appium implicit wait to {original_implicit_wait}: {e}"
                    )

    async def execute_keyword(
        self,
        session: ExecutionSession,
        keyword: str,
        arguments: List[str],
        browser_library_manager: Any,  # BrowserLibraryManager
        detail_level: str = "minimal",
        library_prefix: str = None,
        assign_to: Union[str, List[str]] = None,
        use_context: bool = False,
        timeout_ms: Optional[int] = None,
        record: bool | None = None,
        pre_validate_timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single Robot Framework keyword step with optional library prefix.

        Args:
            session: ExecutionSession to run in
            keyword: Robot Framework keyword name (supports Library.Keyword syntax)
            arguments: List of arguments for the keyword
            browser_library_manager: BrowserLibraryManager instance
            detail_level: Level of detail in response ('minimal', 'standard', 'full')
            library_prefix: Optional explicit library name to override session search order
            assign_to: Optional variable assignment
            use_context: If True, execute within full RF context
            timeout_ms: Optional timeout in milliseconds. If not provided, uses smart
                       defaults based on keyword type:
                       - Element actions (Click, Fill): 5000ms
                       - Navigation (Go To, New Page): 60000ms
                       - Read operations (Get Text): 2000ms
                       - API calls (GET, POST): 30000ms
                       Set to 0 or negative to disable timeout.

        Returns:
            Execution result with status, output, and state
        """

        # Serialize keyword execution globally to prevent concurrent
        # asyncio.to_thread() dispatches.  When two threads are inside
        # _suppress_stdout() simultaneously, the reference-counted fd 1
        # redirect stays active while FastMCP writes the first response,
        # causing it to go to stderr instead of the MCP transport.
        # The lock also prevents concurrent Selenium WebDriver access.
        async with self._execution_lock:
            # F16 + harden-platynui-stuck-key-release: a desktop keyboard keyword
            # that raises or returns failure may have been killed mid-chord,
            # leaving a key physically held at the OS level (the operator's
            # keyboard wedged). We track the EXACT keys held (incl. non-modifiers)
            # so teardown/failure/exit release precisely those. A SUCCESSFUL
            # Keyboard Press legitimately holds a key for a later Keyboard
            # Release, so we do NOT release on success — but we DO record it.
            _kbd = self._is_desktop_keyboard_keyword(session, keyword)
            _kbd_kind, _kbd_seq = (
                self._desktop_keyboard_op(keyword, arguments) if _kbd else (None, None)
            )
            # Record intended-held keys BEFORE dispatch so a kill mid-call is
            # covered by the persisted state file (Press holds; Type holds then
            # releases within the same call).
            if _kbd_kind in ("press", "type") and _kbd_seq:
                self._record_pressed_keys(_kbd_seq)
            try:
                result = await self._execute_keyword_serialized(
                    session, keyword, arguments, browser_library_manager,
                    detail_level, library_prefix, assign_to, use_context,
                    timeout_ms,
                    record=record,
                    pre_validate_timeout_ms=pre_validate_timeout_ms,
                )
            except BaseException:
                # An exception (incl. CancelledError from a killed run) is the
                # canonical "killed mid-chord" case. Release SYNCHRONOUSLY here
                # so it is guaranteed to run even while the task is being
                # cancelled (an awaited offload could be interrupted before it
                # dispatches). keyboard_release is a fast input dispatch, not a
                # tree query, so the brief on-loop cost is acceptable.
                if _kbd:
                    self._release_desktop_keys()
                raise
            _released_here = False
            if (
                _kbd
                and isinstance(result, dict)
                and not result.get("success", True)
                # F2: a steering-confidence downgrade means the native keyword
                # actually SUCCEEDED (the key is held as intended) — releasing
                # would corrupt a deliberate Press/Release chord. Only release
                # on a genuine execution failure.
                and not self._is_steering_downgrade(result)
            ):
                # F4: offload the release off the event loop on the normal
                # failure-return path (no cancellation concern here). Releases
                # the exact tracked held set (incl. non-modifiers) + clears state.
                await asyncio.to_thread(self._release_desktop_keys)
                _released_here = True
                # Steering: point the agent at the atomic Keyboard Type, which
                # cannot leave a key held.
                try:
                    hints = result.setdefault("hints", [])
                    if isinstance(hints, list):
                        hints.append({
                            "type": "platynui_keyboard_release_safety",
                            "message": (
                                "Released all held keyboard keys (including any "
                                "non-modifier such as a letter/F-key/Escape) "
                                "after this keyboard keyword failed. Prefer the "
                                "atomic 'Keyboard Type <Ctrl+A>' (self-contained "
                                "press+release) over a bare 'Keyboard Press', "
                                "which sends key-DOWN only and MUST be paired "
                                "with 'Keyboard Release'."
                            ),
                        })
                except Exception:  # pragma: no cover - defensive
                    pass
            # On a SUCCESSFUL Keyboard Type / Keyboard Release, the paired key-UP
            # already happened — clear those keys from the registry. (Press keeps
            # its recorded keys held until an explicit Keyboard Release.)
            if not _released_here and _kbd_kind in ("type", "release") and _kbd_seq:
                self._record_released_keys(_kbd_seq)
            return result

    @staticmethod
    def _desktop_keyboard_op(keyword, arguments):
        """Return ``(kind, sequence)`` for a desktop keyboard keyword, where
        kind is ``"press"``/``"type"``/``"release"`` and sequence is the key
        text argument (2nd positional, or a ``text=`` named arg). Non-raising;
        returns ``(None, None)`` when it cannot classify. change:
        harden-platynui-stuck-key-release."""
        try:
            norm = str(keyword or "").lower().rsplit(".", 1)[-1].strip()
            kind = {
                "keyboard press": "press",
                "keyboard type": "type",
                "keyboard release": "release",
            }.get(norm)
            if kind is None:
                return (None, None)
            seq = None
            args = list(arguments or [])
            for a in args:
                if isinstance(a, str) and a.startswith("text="):
                    seq = a[len("text="):]
                    break
            if seq is None and len(args) >= 2:
                seq = args[1]
            return (kind, seq)
        except Exception:  # pragma: no cover - defensive
            return (None, None)

    @staticmethod
    def _record_pressed_keys(sequence) -> None:
        """Record keys a desktop Press/Type is about to hold. Never raises.
        change: harden-platynui-stuck-key-release."""
        try:
            from robotmcp.plugins.builtin.platynui_plugin import record_pressed_keys

            record_pressed_keys(sequence)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("record_pressed_keys failed: %s", exc)

    @staticmethod
    def _record_released_keys(sequence) -> None:
        """Remove keys released by a desktop Release/Type from the registry.
        Never raises. change: harden-platynui-stuck-key-release."""
        try:
            from robotmcp.plugins.builtin.platynui_plugin import record_released_keys

            record_released_keys(sequence)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("record_released_keys failed: %s", exc)

    @staticmethod
    def _is_steering_downgrade(result: dict) -> bool:
        """True when a desktop keyword's success was flipped to failure by the
        steering-confidence gate rather than a genuine execution failure
        (change: fix-platynui-windows-runtime, F16/F2).

        A steering downgrade means the native keyword DID execute (its key is
        held as intended), so F16 must not release its modifiers. The gate only
        downgrades in the RF-success path, so this marker uniquely identifies
        the case."""
        try:
            from robotmcp.components.execution.desktop_execution_signals import (
                SC_CONTRADICTED,
            )
        except Exception:  # pragma: no cover - defensive
            SC_CONTRADICTED = "contradicted"
        return bool(result.get("steering_confidence") == SC_CONTRADICTED)

    @staticmethod
    def _release_desktop_keys() -> None:
        """Best-effort release of the EXACT tracked held-key set (incl.
        non-modifiers) via the PlatynUI runtime, clearing the registry + state
        file. Falls back to a modifier blast internally. Never raises.
        change: harden-platynui-stuck-key-release."""
        try:
            from robotmcp.plugins.builtin.platynui_plugin import (
                release_tracked_keys,
            )

            release_tracked_keys()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("desktop key release failed: %s", exc)

    async def _execute_keyword_serialized(
        self,
        session: ExecutionSession,
        keyword: str,
        arguments: List[str],
        browser_library_manager: Any,
        detail_level: str = "minimal",
        library_prefix: str = None,
        assign_to: Union[str, List[str]] = None,
        use_context: bool = False,
        timeout_ms: Optional[int] = None,
        record: bool | None = None,
        pre_validate_timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Inner keyword execution, called under _execution_lock."""
        try:
            # ADR-025: desktop (PlatynUI) sessions must force the X11
            # backend BEFORE the first native Runtime is created in this
            # process — the Wayland portal handshake blocks indefinitely
            # in headless contexts. Idempotent; opt-out via
            # ROBOTMCP_PLATYNUI_KEEP_WAYLAND=1.
            _focus_outcome = None
            _input_effect_before = None  # D1: native CharacterCount before a keyboard step
            try:
                _is_desktop = getattr(session, "is_desktop_session", None)
                if callable(_is_desktop) and _is_desktop() is True:
                    from robotmcp.plugins.builtin.platynui_plugin import (
                        ensure_x11_session_env,
                    )

                    ensure_x11_session_env()
                    # F3 (change: fix-platynui-windows-runtime): a caller who
                    # passes an explicit timeout_ms wants desktop query/wait
                    # keywords to wait that long; stash it so the query-settings
                    # default (applied in _execute_keyword_with_context) honors
                    # it. This is PER-CALL — reset to None when absent so a
                    # one-off long timeout never becomes a session-sticky
                    # default that defeats the fast-fail behaviour on later
                    # steps (the short default is restored automatically).
                    session._platynui_query_timeout_ms = (
                        timeout_ms
                        if isinstance(timeout_ms, int) and timeout_ms > 0
                        else None
                    )
                    # D2: if a recent desktop launch marked the tree stale,
                    # refresh the cached accessibility tree before the FIRST
                    # tree-resolving keyword so a newly-launched app resolves by
                    # name (change: desktop-tree-cache-refresh). The flag is
                    # consumed once so steady-state queries keep the warm cache.
                    try:
                        from robotmcp.components.execution.desktop_execution_signals import (
                            is_tree_resolving_keyword,
                        )

                        if (
                            getattr(session, "desktop_tree_dirty", False)
                            and is_tree_resolving_keyword(keyword)
                        ):
                            from robotmcp.plugins.builtin.platynui_plugin import (
                                clear_runtime_tree_cache,
                            )

                            clear_runtime_tree_cache()
                            session.desktop_tree_dirty = False
                            # Window identities may have changed — drop the
                            # verified-activation cache too (task 2.5).
                            if self._platynui_focus_manager is not None:
                                self._platynui_focus_manager.invalidate_focus_cache()
                    except Exception as _refq_exc:  # pragma: no cover - defensive
                        logger.debug(
                            "pre-query tree refresh skipped: %s", _refq_exc
                        )
                    # SAFETY GUARD (change: platynui-desktop-safety-isolation):
                    # before any focus/dispatch, refuse pointer/keyboard input
                    # unless the bound display is provably isolated — so input
                    # cannot leak onto the user's active desktop. Fail closed on
                    # active/unknown; opt-in + warn-mode honored.
                    _safety_outcome = self._platynui_safety_guard(
                        session, keyword
                    )
                    if _safety_outcome is not None and not _safety_outcome["allowed"]:
                        _guard_hint = {
                            "type": "platynui_active_desktop_guard",
                            "message": _safety_outcome["reason"],
                        }
                        # D3: attach the actionable isolation recipe so the agent
                        # has a guided path to an isolated display, not just the
                        # bypass env var. change: desktop-stepwise-execution-fidelity.
                        _recipe = _safety_outcome.get("isolation_recipe")
                        if _recipe:
                            _guard_hint["isolation_recipe"] = _recipe
                        return {
                            "success": False,
                            "error": _safety_outcome["reason"],
                            "keyword": keyword,
                            "platynui_safety": {
                                "classification": _safety_outcome["classification"],
                                "enforcing": _safety_outcome["enforcing"],
                            },
                            "hints": [_guard_hint],
                        }
                    # Clear any active highlight overlay before a screenshot
                    # so the evidence image shows the app, not the marker
                    # (change: platynui-visible-safe-targeting, task 3.2).
                    try:
                        from robotmcp.components.execution.platynui_focus import (
                            normalize_keyword as _pfx_norm,
                        )

                        if (
                            _pfx_norm(keyword) == "take screenshot"
                            and self._platynui_focus_manager is not None
                        ):
                            self._platynui_focus_manager.clear_highlight()
                    except Exception:  # pragma: no cover - defensive
                        pass
                    # Evidence path guard (D3): refuse screenshot paths
                    # outside the allowed roots with an actionable hint
                    # (change: desktop-evidence-and-display-scoping).
                    _shot_guard = self._screenshot_path_guard(keyword, arguments)
                    if _shot_guard is not None:
                        return _shot_guard
                    # Unscoped-locator guardrail: refuse //-rooted desktop
                    # Query/Evaluate before the multi-second session-wide walk
                    # exceeds the MCP client timeout (change:
                    # desktop-unscoped-locator-guardrail).
                    _unscoped_guard = self._unscoped_locator_guard(
                        session, keyword, arguments
                    )
                    if _unscoped_guard is not None:
                        return _unscoped_guard
                    # Screenshot filename-as-descriptor trap + Linux control:Window
                    # — both convert a silent 30s hang into a fast, actionable
                    # refusal (change: desktop-screenshot-failfast).
                    _shot_sig_guard = self._screenshot_signature_guard(
                        session, keyword, arguments
                    )
                    if _shot_sig_guard is not None:
                        return _shot_sig_guard
                    _ctrl_win_guard = self._control_window_guard(
                        session, keyword, arguments
                    )
                    if _ctrl_win_guard is not None:
                        return _ctrl_win_guard
                    # Focus-before-act: ensure pointer/keyboard input targets
                    # the AUT window, not whatever window is active, and that
                    # the AUT window is visible/in-scope (change:
                    # platynui-focused-execution). FocusError fails fast under
                    # opt-in policy; otherwise warnings ride along on the step.
                    # F14 (change: fix-platynui-windows-runtime): the focus
                    # manager drives raw runtime.evaluate() (find/activate/
                    # highlight the AUT window) with NO timeout knob — a broad
                    # or busy-tree query blocks the event loop for tens of
                    # seconds, wedging metadata calls. Offload to a worker
                    # thread so the loop stays free; the global _execution_lock
                    # already serializes desktop dispatch so no native
                    # concurrency is introduced. FocusError still propagates
                    # out of to_thread to the except below.
                    _focus_outcome = await asyncio.to_thread(
                        self._platynui_focus_before_act,
                        session, keyword, arguments,
                    )
                    # Resolve a desktop Process launch/recovery executable to an
                    # absolute path against the server PATH (maintainer-report
                    # #7). change: desktop-mcp-workflow-correctness.
                    arguments = self._maybe_resolve_desktop_executable(
                        session, keyword, arguments
                    )
                    # Snap-decontaminate a desktop GUI launch so a snap-confined
                    # app does not inherit snap-rooted loader vars and exit 127
                    # (maintainer-report finding #2). change:
                    # desktop-stepwise-execution-fidelity. (is_desktop_gui_launch
                    # matches on basename, so it still fires after absolute-path
                    # resolution above.)
                    arguments = self._maybe_sanitize_desktop_launch(
                        session, keyword, arguments
                    )
                    # D1: snapshot the target text node's CharacterCount BEFORE a
                    # keyboard interaction with an explicit (resolvable) target,
                    # so a success-with-no-effect can be flagged afterward
                    # (change: desktop-input-and-runtime-diagnostics). Native /
                    # non-reentrant; best-effort.
                    # F14: native CharacterCount query — also offload off the
                    # loop (change: fix-platynui-windows-runtime).
                    _input_effect_before = await asyncio.to_thread(
                        self._desktop_text_count_before, keyword, arguments
                    )
            except Exception as _focus_exc:
                from robotmcp.components.execution.platynui_focus import FocusError

                if isinstance(_focus_exc, FocusError):
                    return {
                        "success": False,
                        "error": str(_focus_exc),
                        "keyword": keyword,
                        "hints": [{
                            "type": "platynui_focus_precondition",
                            "message": str(_focus_exc),
                        }],
                    }
                # defensive: never let focus logic break execution

            # PHASE 1.2: Pre-execution Library Registration
            # Ensure required library is registered before keyword execution
            self._ensure_library_registration(keyword, session)

            # Create execution step
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                keyword=keyword,
                arguments=arguments,
                start_time=datetime.now(),
            )
            event_bus.publish_sync(
                FrontendEvent(
                    event_type="step_started",
                    session_id=session.session_id,
                    step_id=step.step_id,
                    payload={"keyword": keyword, "arguments": arguments},
                )
            )

            # Update session activity
            session.update_activity()

            # Mark step as running
            step.status = "running"

            # NOTE: Library keyword validation is handled by plugin overrides in _execute_keyword_internal
            # The BrowserLibraryPlugin._override_open_browser handles "Open Browser" rejection with detailed guidance

            # Check if we should use context mode
            # Enable context mode for keywords that require RF execution context
            context_required_keywords = [
                "evaluate",
                "set test variable",
                "set suite variable",
                "set global variable",
                "create dictionary",
                "get variable value",
                "variable should exist",
                "call method",
                "run keyword if",
                "run keyword unless",
                "run keywords",
                # NOTE: Input Password removed - works fine in normal execution with name normalization
            ]

            # RequestsLibrary: route session-scoped operations through RF native context
            requests_library_context_keywords = [
                "create session",
                "delete session",
                "get on session",
                "post on session",
                "put on session",
                "delete on session",
                "patch on session",
                "head on session",
                "options on session",
            ]

            # Browser Library keywords should NOT use RF native context due to import issues
            # They work perfectly in regular execution mode
            browser_library_keywords = [
                "open browser",
                "close browser",
                "new browser",
                "new context",
                "new page",
                "go to",
                "click",
                "fill text",
                "take screenshot",
                "get text",
                "wait for elements state",
                "get title",
                "get url",
                "input text",
                "click element",
                "wait until element is visible",
            ]

            # KEYWORD NAME NORMALIZATION AND OVERRIDES - General solution for keyword name variations
            # NOTE: Input Password override is now handled in _execute_selenium_keyword method
            # to ensure proper execution while preserving original keyword for step recording
            keyword_name_mappings = {
                # Add other common mappings as needed (Input Password removed - handled in _execute_selenium_keyword)
                # "click element": "click_element",  # Usually handled by dynamic resolution
            }

            # Apply normalization if mapping exists (Input Password override removed from here)
            original_keyword = keyword
            if keyword in keyword_name_mappings:
                logger.info(
                    f"Keyword name normalized: '{original_keyword}' -> '{keyword_name_mappings[keyword]}'"
                )
                keyword = keyword_name_mappings[keyword]

            keyword_requires_context = keyword.lower() in context_required_keywords
            is_requests_keyword = keyword.lower() in requests_library_context_keywords
            is_browser_keyword = keyword.lower() in browser_library_keywords

            # PRE-VALIDATION: Fast check for element actionability before execution
            # This detects "element not visible/enabled" in ~500ms instead of waiting 10s
            if self.pre_validation_enabled and self._requires_pre_validation(keyword, session):
                locator = self._extract_locator_from_args(keyword, arguments)
                if locator:
                    # Derive pre-validation timeout. Precedence:
                    # 1. Explicit pre_validate_timeout_ms wins (per-call override).
                    #    <= 0 disables the gate just like timeout_ms <= 0 does.
                    # 2. Else fall back to user's timeout_ms-derived calculation
                    #    (existing behaviour).
                    # 3. Else None → uses config.PRE_VALIDATION_TIMEOUT default.
                    preval_timeout = None
                    skip_preval = False
                    if pre_validate_timeout_ms is not None:
                        if pre_validate_timeout_ms <= 0:
                            skip_preval = True
                        else:
                            preval_timeout = pre_validate_timeout_ms
                    elif timeout_ms is not None:
                        if timeout_ms <= 0:
                            skip_preval = True
                        else:
                            preval_timeout = min(
                                self.config.PRE_VALIDATION_TIMEOUT, timeout_ms
                            )

                    if skip_preval:
                        logger.debug(
                            f"Pre-validation skipped: timeout disabled by user for '{keyword}'"
                        )
                    else:
                        # Auto-retry on transient failure (slow page settle,
                        # late-mounting elements). Happy path is unchanged —
                        # only failing calls pay the +200ms retry gap.
                        is_valid, error_msg, pre_validation_details = await self._pre_validate_element_with_retry(
                            locator, session, keyword, timeout_ms=preval_timeout
                        )
                        if not is_valid:
                            # Pre-validation failed - return early with helpful error
                            step.end_time = datetime.now()
                            step.mark_failure(error_msg)
                            hints: List[Dict[str, Any]] = [
                                {
                                    "type": "pre_validation_failure",
                                    "message": "Element is not in an actionable state",
                                    "suggestion": "Ensure the element is visible and enabled before interaction",
                                    "details": pre_validation_details,
                                }
                            ]
                            # Add specific hints based on missing states
                            if pre_validation_details and "missing_states" in pre_validation_details:
                                missing = pre_validation_details.get("missing_states", [])
                                if "visible" in missing:
                                    hints.append({
                                        "type": "visibility_hint",
                                        "message": "Element is not visible",
                                        "suggestion": "Use 'Wait For Elements State' with 'visible' before clicking",
                                        "example": f"Wait For Elements State    {locator}    visible    timeout=5s",
                                    })
                                if "enabled" in missing:
                                    hints.append({
                                        "type": "enabled_hint",
                                        "message": "Element is not enabled",
                                        "suggestion": "Check if the element is disabled or if a previous action is required",
                                    })

                            # Per-call timeout override hint. Surfaced
                            # whenever pre-validation fails so the agent
                            # knows how to extend the gate for a genuinely
                            # slow page without disabling it everywhere.
                            # Includes the real escape knob (timeout_ms=0)
                            # as a last resort — NOT a non-existent
                            # pre_validate=False parameter.
                            hints.append({
                                "type": "pre_validate_timeout_hint",
                                "message": (
                                    "Pre-validation timed out after the retry. "
                                    "If the page genuinely takes longer to "
                                    "settle, extend the gate for this call."
                                ),
                                "suggestion": (
                                    "Pass pre_validate_timeout_ms=2000 (or "
                                    "longer) on the next execute_step call. "
                                    "Use timeout_ms=0 only as a last resort — "
                                    "it disables BOTH the pre-validation gate "
                                    "AND the keyword timeout."
                                ),
                                "example_extend": (
                                    f"execute_step(..., pre_validate_timeout_ms=2000)"
                                ),
                                "example_skip": (
                                    f"execute_step(..., timeout_ms=0)"
                                ),
                            })

                            # OBS-15 — the OBS-12 role-prefix hint
                            # otherwise lives in the keyword-execution
                            # failure path via generate_hints. ``button=X``
                            # against Browser library typically fails HERE
                            # at pre-validation, never reaching that path.
                            # Run the same checker inline so the diagnosis
                            # surfaces where the failure actually occurs.
                            try:
                                from robotmcp.utils.hints import (
                                    HintContext as _HintContext,
                                    _check_browser_role_prefix_misuse,
                                )
                                _ctx = _HintContext(
                                    session_id=session.session_id,
                                    keyword=keyword,
                                    arguments=list(arguments or []),
                                    error_text=error_msg or "",
                                    session_search_order=getattr(
                                        session, "search_order", None
                                    ),
                                )
                                for _h in _check_browser_role_prefix_misuse(
                                    _ctx, _ctx.error_text
                                ):
                                    hints.append({
                                        "type": "browser_role_prefix_misuse",
                                        "title": _h.title,
                                        "message": _h.message,
                                        "examples": _h.examples,
                                    })
                            except Exception:
                                # Hint enrichment is best-effort — never
                                # mask the underlying pre-validation
                                # failure on import / context-build error.
                                pass

                            # Dedupe by ``type`` (idempotency safety —
                            # the checker should only fire once per call,
                            # but if a future caller appends another
                            # role-prefix hint via a different path we
                            # keep the response clean).
                            _seen_types: set = set()
                            _deduped: List[Dict[str, Any]] = []
                            for _h_entry in hints:
                                _t = _h_entry.get("type")
                                if _t and _t in _seen_types:
                                    continue
                                if _t:
                                    _seen_types.add(_t)
                                _deduped.append(_h_entry)
                            hints = _deduped

                            event_bus.publish_sync(
                                FrontendEvent(
                                    event_type="step_failed",
                                    session_id=session.session_id,
                                    step_id=step.step_id,
                                    payload={
                                        "status": "fail",
                                        "keyword": keyword,
                                        "arguments": arguments,
                                        "error": error_msg,
                                        "pre_validation_failed": True,
                                    },
                                )
                            )
                            return {
                                "success": False,
                                "error": f"Pre-validation failed: {error_msg}",
                                "hint": "Element is not in an actionable state. Previous steps may not have completed.",
                                "pre_validation_failed": True,
                                "pre_validation_details": pre_validation_details,
                                "step_id": step.step_id,
                                "keyword": keyword,
                                "arguments": arguments,
                                "status": "fail",
                                "execution_time": step.execution_time,
                                "hints": hints,
                            }
                        else:
                            logger.debug(
                                f"Pre-validation passed for '{keyword}' with locator '{locator}'"
                            )

            # Context-only execution: route all keywords through RF native context
            if True:
                # Determine effective timeout:
                # 1. User-provided timeout_ms takes highest precedence
                # 2. If not provided (None), use TimeoutPolicy based on keyword classification
                # 3. If timeout_ms <= 0, disable timeout entirely
                action_type = classify_keyword(keyword)
                if timeout_ms is not None:
                    effective_timeout_ms = timeout_ms if timeout_ms > 0 else None
                    timeout_source = "user-specified" if timeout_ms > 0 else "disabled"
                else:
                    container = get_container()
                    timeout_policy = container.get_timeout_policy(session.session_id)
                    effective_timeout_ms = timeout_policy.get_timeout_for(action_type).value
                    timeout_source = f"policy ({action_type.value})"

                logger.info(
                    f"Executing keyword in RF native context mode: {keyword} with args: {arguments}, "
                    f"timeout: {effective_timeout_ms}ms ({timeout_source})"
                )

                # Snapshot variables before execution for changed_variables tracking
                _vars_before = dict(session.variables)

                # Use RF native context mode for keywords that require it
                result = await self._execute_keyword_with_context(
                    session, keyword, arguments, assign_to, browser_library_manager,
                    timeout_ms=effective_timeout_ms
                )
                resolved_arguments = (
                    arguments  # For logging - RF handles variable resolution
                )
            else:
                # Unreachable in context-only mode
                result = {"success": False, "error": "Non-context path disabled"}

            # Update step status
            step.end_time = datetime.now()
            step.result = result.get("output")

            # Surface focus-before-act outcome (warnings/bypass/strategy) so
            # silent off-window or non-visible operations are not lost
            # (change: platynui-focused-execution, tasks 2.2/4.4).
            if _focus_outcome is not None and isinstance(result, dict):
                fo = _focus_outcome.to_dict()
                if _focus_outcome.bypassed:
                    result["focus_bypassed"] = True
                if fo.get("warnings") or fo.get("attempted"):
                    result["platynui_focus"] = fo
                    if fo.get("warnings"):
                        _fhints = result.get("hints") or []
                        for _w in fo["warnings"]:
                            _fhints.append({
                                "type": "platynui_focus_warning",
                                "message": _w,
                            })
                        result["hints"] = _fhints

            # One-time unscoped-locator warning on the opt-in path (change:
            # desktop-unscoped-locator-guardrail): attached for success OR
            # failure, then the per-session one-shot flag is flipped.
            _pending_unscoped = getattr(session, "_pending_unscoped_hint", None)
            if _pending_unscoped is not None and isinstance(result, dict):
                result.setdefault("hints", []).append(_pending_unscoped)
                try:
                    session.desktop_unscoped_warned = True
                    session._pending_unscoped_hint = None
                except Exception:
                    pass

            # Screenshot path recovery (D3) — see _maybe_recover_screenshot_result.
            self._maybe_recover_screenshot_result(session, keyword, arguments, result)

            # Step accounting: every execution counts, recorded or not, so
            # build_test_suite can warn on silently-empty stepwise suites
            # (change: platynui-visible-safe-targeting, I-1).
            try:
                session.executed_step_count += 1
                if not result["success"]:
                    session.failed_step_count += 1
            except Exception:
                pass

            if result["success"]:
                step_result_value = result.get("result")
                if step_result_value is None and "output" in result:
                    step_result_value = result.get("output")
                step.mark_success(step_result_value)
                # F-N12: gate step recording so build_test_suite emits a clean
                # narrative instead of every page-state probe.
                record_resolved = _resolve_record_gate(
                    keyword=keyword,
                    record=record,
                    assign_to=assign_to,
                    session=session,
                )
                if record_resolved:
                    session.add_step(step)
                    logger.debug(f"Added successful step to session: {keyword}")
                else:
                    logger.debug(
                        f"Step not recorded (inspection-only or record=False): {keyword}"
                    )
                result["recorded"] = record_resolved

                # D2a: surface a Process-vs-discovery disagreement for a desktop
                # launch — the handle is dead (e.g. snap exec 127) while
                # PlatynUI may still observe stale nodes (finding #2). Uses the
                # Visual-validation hint (change: visual-inspection-guidance):
                # for ANY successful screenshot step (web or desktop), advertise
                # the saved artifact + a one-line pointer so a multimodal agent
                # knows it can read the file for checks the DOM/ARIA can't do.
                # Token-cheap: a path + one line, no image bytes.
                try:
                    from robotmcp.components.execution.desktop_execution_signals import (
                        visual_validation_hint,
                    )

                    _vv = visual_validation_hint(keyword, arguments, step_result_value)
                    if _vv:
                        result.setdefault("hints", []).append(_vv)
                        result["screenshot_path"] = _vv["screenshot_path"]
                except Exception:  # pragma: no cover - defensive
                    pass

                # handle's own ``poll()`` — NON-reentrant, no RF re-execution
                # under the lock. change: desktop-stepwise-execution-fidelity.
                try:
                    _is_desktop = getattr(session, "is_desktop_session", None)
                    if callable(_is_desktop) and _is_desktop() is True:
                        from robotmcp.components.execution.desktop_execution_signals import (
                            close_liveness_hint,
                            evidence_missing_hint,
                            input_effect_hint,
                            is_close_keyword,
                            is_interaction_keyword,
                            is_launch_keyword,
                            launch_liveness_hint,
                            steering_confidence,
                            steering_confidence_mode,
                            wayland_input_warning,
                            SC_CONTRADICTED,
                        )

                        # Close liveness (change: desktop-test-scoping-and-
                        # close-lifecycle, D5): the window closed but the AUT
                        # process may have survived (start-center frames).
                        if is_close_keyword(keyword):
                            # The display's window population changed — drop
                            # the display-scoping PID cache so post-close
                            # diagnostics (display_empty) see fresh state
                            # (run-4 finding: stale cache made the empty
                            # display report "(X11 probe unavailable)").
                            try:
                                from robotmcp.components.execution.ui_tree_service import (
                                    clear_display_pid_cache,
                                )

                                clear_display_pid_cache()
                            except Exception:  # pragma: no cover - defensive
                                pass
                            _aut_pid = getattr(session, "desktop_aut_pid", None)
                            _alive = None
                            if isinstance(_aut_pid, int):
                                try:
                                    os.kill(_aut_pid, 0)
                                    _alive = True
                                except ProcessLookupError:
                                    _alive = False
                                except PermissionError:
                                    _alive = True
                                except Exception:
                                    _alive = None
                            _ch = close_liveness_hint(_alive)
                            if _ch:
                                result.setdefault("hints", []).append(_ch)

                        # Evidence integrity (change: desktop-evidence-and-
                        # display-scoping, D2): a successful screenshot step
                        # must be backed by a file on disk or carry a warning.
                        _ev_hint = evidence_missing_hint(
                            keyword, arguments, step_result_value
                        )
                        if _ev_hint:
                            result.setdefault("hints", []).append(_ev_hint)

                        # D2: once per session, warn that synthetic X11 input may
                        # be blocked when the session originated as Wayland.
                        if (
                            is_interaction_keyword(keyword)
                            and not getattr(session, "desktop_wayland_warned", False)
                        ):
                            _wh = wayland_input_warning(keyword)
                            if _wh:
                                result.setdefault("hints", []).append(_wh)
                                session.desktop_wayland_warned = True

                        # D1: success-with-no-effect — compare the BEFORE snapshot
                        # to an AFTER snapshot; warn when a keyboard step succeeded
                        # but the target's CharacterCount did not change.
                        _after = None
                        if _input_effect_before is not None:
                            # F14 (change: fix-platynui-windows-runtime): the
                            # AFTER CharacterCount probe is the same native
                            # runtime.evaluate() as the offloaded BEFORE probe —
                            # offload it off the loop too for symmetry.
                            _after = await asyncio.to_thread(
                                self._desktop_text_count_before, keyword, arguments
                            )
                            _eh = input_effect_hint(
                                keyword=keyword,
                                success=True,
                                state_before=_input_effect_before,
                                state_after=_after,
                            )
                            if _eh:
                                result.setdefault("hints", []).append(_eh)

                        # Steering-confidence verdict (change: desktop-steering-
                        # confidence-gate): compose the verified-focus, input-
                        # effect and Wayland-drop signals into one machine-
                        # parseable verdict; a `contradicted` interaction fails
                        # the step by default (opt-out via
                        # ROBOTMCP_PLATYNUI_STEERING_CONFIDENCE=warn), so
                        # "success" means the input demonstrably reached the app.
                        if is_interaction_keyword(keyword):
                            try:
                                _vf = None
                                if _focus_outcome is not None and hasattr(
                                    _focus_outcome, "has_verified_focus"
                                ):
                                    _vf = bool(_focus_outcome.has_verified_focus())
                                _wr = False
                                try:
                                    from robotmcp.plugins.builtin.platynui_plugin import (
                                        was_wayland_session,
                                    )

                                    _wr = bool(was_wayland_session())
                                except Exception:
                                    _wr = False
                                _sc = steering_confidence(
                                    keyword=keyword,
                                    success=True,
                                    verified_focus=_vf,
                                    state_before=_input_effect_before,
                                    state_after=_after,
                                    wayland_risk=_wr,
                                )
                                if _sc is not None:
                                    result["steering_confidence"] = _sc["verdict"]
                                    result.setdefault("hints", []).append(_sc)
                                    if (
                                        _sc["verdict"] == SC_CONTRADICTED
                                        and steering_confidence_mode() == "enforce"
                                    ):
                                        # Downgrade the false success to a failure:
                                        # the input did not demonstrably land.
                                        result["success"] = False
                                        result["error"] = _sc["message"]
                                        try:
                                            step.mark_failure(_sc["message"])
                                        except Exception:
                                            pass
                                        # Un-record the step optimistically added
                                        # above (failed steps are not kept).
                                        if result.get("recorded") and getattr(
                                            session, "steps", None
                                        ):
                                            try:
                                                if session.steps and session.steps[-1] is step:
                                                    session.steps.pop()
                                                    result["recorded"] = False
                                            except Exception:
                                                pass
                            except Exception as _sc_exc:  # pragma: no cover - defensive
                                logger.debug(
                                    "steering_confidence skipped: %s", _sc_exc
                                )

                        if is_launch_keyword(keyword):
                            proc = step_result_value
                            running = None
                            try:
                                if hasattr(proc, "poll"):
                                    running = proc.poll() is None
                            except Exception:
                                running = None
                            # Remember the AUT pid so focus-before-act can
                            # verify the resolved target belongs to the
                            # launched application (change:
                            # platynui-visible-safe-targeting, task 2.4).
                            try:
                                _pid = getattr(proc, "pid", None)
                                if isinstance(_pid, int):
                                    session.desktop_aut_pid = _pid
                                    # Session id survives wrapper exits,
                                    # daemonization and single-instance
                                    # handoff — the lineage signal for the
                                    # scope check (change:
                                    # desktop-aut-process-lineage).
                                    try:
                                        session.desktop_aut_sid = os.getsid(_pid)
                                    except Exception:
                                        session.desktop_aut_sid = None
                            except Exception:
                                pass
                            _hint = launch_liveness_hint(
                                process_running=running, discovery_node_count=None
                            )
                            if _hint:
                                result.setdefault("hints", []).append(_hint)
                            # I-4: a dash-prefixed `=`-containing launch arg
                            # (e.g. -env:UserInstallation=file://…) was likely
                            # swallowed by RF as a named argument — the app
                            # started WITHOUT it. Detect-and-hint only; the
                            # command line is never rewritten (change:
                            # platynui-visible-safe-targeting, task 5.1).
                            try:
                                from robotmcp.utils.hints import (
                                    _escaped_form,
                                    detect_process_eq_arg_misparse,
                                )

                                _flagged = detect_process_eq_arg_misparse(
                                    keyword, arguments
                                )
                                if _flagged:
                                    result.setdefault("hints", []).append({
                                        "type": "process_named_arg_misparse",
                                        "message": (
                                            "Robot Framework may have parsed "
                                            + ", ".join(f"'{a}'" for a in _flagged)
                                            + " as a named argument and dropped "
                                            "it from the command line. Escape "
                                            "the first '=' as '\\=' to keep it "
                                            "positional, e.g. "
                                            + _escaped_form(_flagged[0])
                                        ),
                                    })
                            except Exception:  # pragma: no cover - defensive
                                pass
                            # D1: a freshly-launched app is invisible to the
                            # cached desktop tree until cleared. Clear now and
                            # mark the session dirty so the first post-launch
                            # tree-resolving keyword re-reads live AT-SPI
                            # (change: desktop-tree-cache-refresh).
                            try:
                                from robotmcp.plugins.builtin.platynui_plugin import (
                                    clear_runtime_tree_cache,
                                )

                                clear_runtime_tree_cache()
                                session.desktop_tree_dirty = True
                                # The display's window/PID population changed —
                                # drop the display-scoping PID cache so the new
                                # AUT is never filtered as a "host app" (change:
                                # desktop-evidence-and-display-scoping, D4).
                                from robotmcp.components.execution.ui_tree_service import (
                                    clear_display_pid_cache,
                                )

                                clear_display_pid_cache()
                            except Exception as _ref_exc:  # pragma: no cover - defensive
                                logger.debug(
                                    "post-launch tree refresh skipped: %s", _ref_exc
                                )
                except Exception as _sig_exc:  # pragma: no cover - defensive
                    logger.debug("desktop launch signal skipped: %s", _sig_exc)
            else:
                step.mark_failure(result.get("error"))
                logger.debug(
                    f"Failed step not added to session: {keyword} - {result.get('error')}"
                )

            # Surface running executed/recorded counts so an agent can detect
            # the divergence while still executing (I-1).
            try:
                result["steps_executed"] = session.executed_step_count
                result["steps_recorded"] = session.recorded_step_count()
            except Exception:
                pass

            # Update session variables if any were set
            if "variables" in result:
                session.variables.update(result["variables"])
                try:
                    step.variables.update(result["variables"])
                except Exception:
                    pass

            # Validate assignment compatibility
            if assign_to:
                self._validate_assignment_compatibility(keyword, assign_to)

            # Process variable assignment if assign_to is specified
            if assign_to and result.get("success"):
                assignment_vars = self._process_variable_assignment(
                    assign_to, result.get("result"), keyword, result.get("output")
                )
                if assignment_vars:
                    # DUAL STORAGE IMPLEMENTATION:
                    # 1. Store ORIGINAL objects in session variables for RF execution context
                    session.variables.update(assignment_vars)

                    # NEW: Store assignment info in ExecutionStep for test suite generation
                    step.assigned_variables = list(assignment_vars.keys())
                    step.assignment_type = (
                        "multiple" if isinstance(assign_to, list) else "single"
                    )

                    # DEBUG: Verify what we actually stored in session variables
                    for var_name, var_value in assignment_vars.items():
                        logger.info(
                            f"STORED IN SESSION: {var_name} = {type(var_value).__name__}"
                        )
                        logger.debug(
                            f"Session storage detail: {var_name} -> {str(var_value)[:100]}"
                        )
                        # Verify what's actually in session.variables after update
                        actual_stored = session.variables.get(var_name)
                        logger.info(
                            f"SESSION VERIFICATION: {var_name} stored as {type(actual_stored).__name__}"
                        )

                    # 2. Store raw objects for RF Variables system (needed for ${response.json()})
                    result["assigned_variables_raw"] = assignment_vars

                    # 3. Add serialized assignment info to result for MCP response compatibility
                    # This prevents serialization errors with complex objects
                    serialized_assigned_vars = (
                        self.response_serializer.serialize_assigned_variables(
                            assignment_vars
                        )
                    )
                    result["assigned_variables"] = serialized_assigned_vars
                    try:
                        step.variables.update(serialized_assigned_vars)
                    except Exception:
                        pass

                    # Log assignment for debugging
                    for var_name, var_value in assignment_vars.items():
                        logger.info(
                            f"Assigned variable {var_name} = {type(var_value).__name__} (serialized for response)"
                        )
                        logger.debug(
                            f"Assignment detail: {var_name} -> {str(var_value)[:200]}"
                        )

            # Build response based on detail level
            response = await self._build_response_by_detail_level(
                detail_level,
                result,
                step,
                keyword,
                arguments,
                session,
                resolved_arguments,
                vars_before=_vars_before,
            )

            def _serialize_event_value(value: Any) -> Any:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    return value
                if isinstance(value, (list, tuple)):
                    return [_serialize_event_value(item) for item in value]
                if isinstance(value, dict):
                    return {str(k): _serialize_event_value(v) for k, v in value.items()}
                return str(value)

            event_payload = {
                "status": step.status,
                "keyword": keyword,
                "arguments": arguments,
            }

            if result["success"]:
                event_payload["result"] = _serialize_event_value(step.result)
                if step.assigned_variables:
                    event_payload["assigned_variables"] = list(step.assigned_variables)
                    event_payload["assignment_type"] = step.assignment_type
                    assigned_values = {}
                    for var_name in step.assigned_variables:
                        value = step.variables.get(var_name)
                        if value is None:
                            value = session.variables.get(var_name)
                        assigned_values[var_name] = _serialize_event_value(value)
                    event_payload["assigned_values"] = assigned_values
            else:
                event_payload["error"] = result.get("error")

            event_bus.publish_sync(
                FrontendEvent(
                    event_type="step_completed" if result["success"] else "step_failed",
                    session_id=session.session_id,
                    step_id=step.step_id,
                    payload=event_payload,
                )
            )
            # F-N12: surface the record gate decision so callers (and tests)
            # can observe whether the step was added to session.steps for
            # build_test_suite output.
            if "recorded" in result:
                response["recorded"] = result["recorded"]
            return response

        except Exception as e:
            logger.error(f"Error executing step {keyword}: {e}")

            # Create a failed step for error reporting
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                keyword=keyword,
                arguments=arguments,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            step.mark_failure(str(e))

            hints: List[Dict[str, Any]] = []
            library_name = self._get_library_for_keyword(keyword)
            plugin_hints = self.plugin_manager.generate_failure_hints(
                library_name,
                session,
                keyword,
                list(arguments or []),
                str(e),
            )
            if plugin_hints:
                hints.extend(plugin_hints)
            try:
                from robotmcp.utils.hints import HintContext, generate_hints

                if not hints:
                    hctx = HintContext(
                        session_id=session.session_id,
                        keyword=keyword,
                        arguments=list(arguments or []),
                        error_text=str(e),
                        session_search_order=getattr(session, "search_order", None),
                    )
                    hints = generate_hints(hctx)
            except Exception:
                if not hints:
                    hints = []

            return {
                "success": False,
                "error": str(e),
                "step_id": step.step_id,
                "keyword": keyword,
                "arguments": arguments,
                "status": "fail",
                "execution_time": step.execution_time,
                "hints": hints,
            }

    def _process_variable_assignment(
        self,
        assign_to: Union[str, List[str]],
        result_value: Any,
        keyword: str,
        output: str,
    ) -> Dict[str, Any]:
        """Process variable assignment from keyword execution result.

        Args:
            assign_to: Variable name(s) to assign to
            result_value: The actual return value from the keyword
            keyword: The keyword name (for logging)
            output: The output string representation

        Returns:
            Dictionary of variables to assign to session
        """
        if not assign_to:
            return {}

        # DEBUG: Log what we receive for tracing serialization issue
        logger.debug(
            f"VARIABLE_ASSIGNMENT_DEBUG: {keyword} result_value type: {type(result_value)}"
        )
        logger.debug(
            f"VARIABLE_ASSIGNMENT_DEBUG: {keyword} result_value: {str(result_value)[:200]}"
        )

        # Check if result_value is already serialized (RequestsLibrary Response issue)
        if (
            isinstance(result_value, dict)
            and result_value.get("_type") == "requests_response"
        ):
            logger.warning(
                f"SERIALIZATION_WARNING: {keyword} result_value is already serialized Response object!"
            )

        # If result_value is None but output exists, try to use output
        # This handles cases where the result is in output but not result field
        value_to_assign = result_value
        if value_to_assign is None and output:
            try:
                # Try to parse output as the actual value
                import ast

                # Handle simple cases like numbers, strings, lists
                if output.isdigit():
                    value_to_assign = int(output)
                elif output.replace(".", "").isdigit():
                    value_to_assign = float(output)
                elif output.startswith("[") and output.endswith("]"):
                    value_to_assign = ast.literal_eval(output)
                else:
                    value_to_assign = output
            except:
                value_to_assign = output

        variables = {}

        try:
            if isinstance(assign_to, str):
                # Single assignment
                var_name = self._normalize_variable_name(assign_to)
                variables[var_name] = value_to_assign
                logger.info(f"Assigned {var_name} = {value_to_assign}")

            elif isinstance(assign_to, list):
                # Multi-assignment
                if isinstance(value_to_assign, (list, tuple)):
                    for i, var_name in enumerate(assign_to):
                        normalized_name = self._normalize_variable_name(var_name)
                        if i < len(value_to_assign):
                            variables[normalized_name] = value_to_assign[i]
                        else:
                            variables[normalized_name] = None
                        logger.info(
                            f"Assigned {normalized_name} = {variables[normalized_name]}"
                        )
                else:
                    # Single value assigned to multiple variables (first gets value, rest get None)
                    for i, var_name in enumerate(assign_to):
                        normalized_name = self._normalize_variable_name(var_name)
                        variables[normalized_name] = value_to_assign if i == 0 else None
                        logger.info(
                            f"Assigned {normalized_name} = {variables[normalized_name]}"
                        )

        except Exception as e:
            logger.warning(
                f"Error processing variable assignment for keyword '{keyword}': {e}"
            )
            # Fallback: assign the raw value to first variable name
            if isinstance(assign_to, str):
                var_name = self._normalize_variable_name(assign_to)
                variables[var_name] = value_to_assign
            elif isinstance(assign_to, list) and assign_to:
                var_name = self._normalize_variable_name(assign_to[0])
                variables[var_name] = value_to_assign

        return variables

    def _ensure_library_registration(self, keyword: str, session: Any) -> None:
        """
        Ensure required library is registered in RF context before keyword execution.

        This is Phase 1.2 of the RequestsLibrary fix: Pre-execution Library Registration.
        We determine which library is needed for a keyword and ensure it's registered
        in the Robot Framework execution context.
        """
        try:
            # Determine library from keyword
            library_name = self._get_library_for_keyword(keyword)

            # Honor explicit preference for overlapping keywords
            pref = (getattr(session, "explicit_library_preference", "") or "").lower()
            if keyword.lower() == "open browser":
                if pref.startswith("selenium"):
                    library_name = "SeleniumLibrary"
                elif pref.startswith("browser"):
                    library_name = "Browser"

            # If the scenario explicitly prefers Selenium, avoid registering Browser for
            # overlapping keywords like 'Open Browser' so SeleniumLibrary stays in control.
            if library_name and library_name.lower() == "browser" and pref.startswith("selenium"):
                logger.debug(
                    "Skipping Browser registration for keyword '%s' due to Selenium preference",
                    keyword,
                )
                return

            if library_name:
                # Get the library manager from keyword discovery
                library_manager = self.keyword_discovery.library_manager

                # Ensure RequestsLibrary is loaded in our manager
                if library_name not in library_manager.libraries:
                    logger.info(
                        f"Loading {library_name} on demand for keyword: {keyword}"
                    )
                    library_manager.load_library_on_demand(
                        library_name, self.keyword_discovery
                    )

                # Ensure RequestsLibrary is properly registered in RF context
                registration_success = library_manager.ensure_library_in_rf_context(
                    library_name
                )

                if registration_success:
                    logger.debug(
                        f"Successfully ensured {library_name} registration for keyword: {keyword}"
                    )
                    self.plugin_manager.run_before_keyword_execution(
                        library_name,
                        session,
                        keyword,
                        library_manager,
                        self.keyword_discovery,
                    )

                else:
                    # Recoverable: the keyword still resolves via on-demand import
                    # into the freshly-created context, so this is a lazy-bootstrap
                    # note, not a failure.
                    logger.debug(
                        f"Deferred registration of {library_name} in RF context for keyword: {keyword} (will import on demand)"
                    )

        except Exception as e:
            logger.error(f"Library registration check failed for {keyword}: {e}")
            # Don't fail execution for this - let the keyword execution handle library issues

    def _get_library_for_keyword(self, keyword: str) -> Optional[str]:
        """Determine which library provides a given keyword."""

        # Handle explicit library prefixes (e.g., "RequestsLibrary.POST")
        if "." in keyword:
            parts = keyword.split(".")
            if len(parts) == 2:
                library_name, _ = parts
                return library_name

        mapped = self.plugin_manager.get_library_for_keyword(keyword)
        if mapped:
            return mapped
        return None

    # Short default desktop query timeout (F3). PlatynUI.BareMetal's
    # QuerySettings default is 30s, so an honest-miss Query/wait waits the full
    # 30s and stacks across retries (up to ~180s in the Windows eval). A short
    # default fails fast; a caller can still request a longer wait via
    # timeout_ms. Overridable via ROBOTMCP_PLATYNUI_QUERY_TIMEOUT_MS.
    _DEFAULT_PLATYNUI_QUERY_TIMEOUT_MS = 1500

    def _ensure_desktop_query_timeout(self, session) -> None:
        """Give desktop PlatynUI query/wait keywords a short default timeout so
        an honest-miss fails fast (change: fix-platynui-windows-runtime, F3).

        Mirrors ``Set Query Settings scope=SUITE`` by writing a short-timeout
        ``QuerySettings`` into ``${PLATYNUI_QUERY_SETTINGS}`` in the live RF
        context, once per session (re-applied only if the desired value
        changed). A caller-supplied ``timeout_ms`` (stashed on the session)
        overrides the short default. Best-effort; never raises — on any failure
        the library keeps its 30s default (no worse than before)."""
        try:
            checker = getattr(session, "is_desktop_session", None)
            if not (callable(checker) and checker() is True):
                return
        except Exception:
            return
        try:
            default_ms = int(
                os.environ.get(
                    "ROBOTMCP_PLATYNUI_QUERY_TIMEOUT_MS",
                    str(self._DEFAULT_PLATYNUI_QUERY_TIMEOUT_MS),
                )
            )
        except (ValueError, TypeError):
            default_ms = self._DEFAULT_PLATYNUI_QUERY_TIMEOUT_MS
        desired_ms = getattr(session, "_platynui_query_timeout_ms", None) or default_ms
        if desired_ms <= 0:
            return
        if getattr(session, "_platynui_query_timeout_applied_ms", None) == desired_ms:
            return
        try:
            from dataclasses import replace as _dc_replace

            from robot.running.context import EXECUTION_CONTEXTS as _EC
            from PlatynUI.BareMetal import (  # library-sanctioned mechanism
                PLATYNUI_QUERY_SETTINGS,
                QuerySettings,
            )

            ctx = _EC.current
            if ctx is None:
                return
            variables = ctx.variables
            name = f"${{{PLATYNUI_QUERY_SETTINGS}}}"
            # Preserve any non-timeout query settings already in scope
            # (retry_interval / ignore_exceptions) instead of resetting them to
            # class defaults — only the timeout is ours to change.
            base = None
            try:
                if PLATYNUI_QUERY_SETTINGS in variables:
                    existing = variables[name]
                    if isinstance(existing, QuerySettings):
                        base = existing
            except Exception:
                base = None
            timeout_s = desired_ms / 1000.0
            settings = (
                _dc_replace(base, timeout=timeout_s)
                if base is not None
                else QuerySettings(timeout=timeout_s)
            )
            # set_suite(children=True): a bare write into the SUITE store is
            # invisible because rf-mcp already holds a live TEST scope copied
            # from suite BEFORE the write, and PlatynUI resolves the variable
            # via variables.current (the test scope). set_suite(children=True)
            # both persists at SUITE (surviving per-step test-scope resets) AND
            # propagates down to the current/test scope so the library sees it
            # immediately (mirrors Set Query Settings scope=SUITE).
            variables.set_suite(name, settings, children=True)
            session._platynui_query_timeout_applied_ms = desired_ms
            logger.debug(
                "Applied desktop query timeout %sms via ${PLATYNUI_QUERY_SETTINGS}",
                desired_ms,
            )
        except Exception as exc:  # pragma: no cover - env/internals dependent
            logger.debug("desktop query-settings apply skipped: %s", exc)

    def _inject_timeout_into_arguments(
        self,
        keyword: str,
        arguments: List[Any],
        timeout_ms: Optional[int],
        session: ExecutionSession,
    ) -> List[Any]:
        """Inject timeout into keyword arguments for keywords that support it.

        This method adds timeout argument to Browser Library and SeleniumLibrary
        keywords that accept a timeout parameter. The timeout is only injected if:
        1. A timeout_ms value is provided
        2. The keyword supports timeout parameter
        3. No timeout argument is already present

        Args:
            keyword: The keyword name
            arguments: The original arguments list
            timeout_ms: Timeout in milliseconds from TimeoutPolicy
            session: The execution session (for library detection)

        Returns:
            Arguments list with timeout injected if applicable
        """
        if not timeout_ms:
            return arguments

        # Desktop sessions (PlatynUI, ADR-025): keywords take descriptor
        # locators, not timeout= named args. The native runtime applies its
        # own descriptor-resolution retry (default 30s). Never inject.
        # (Strict `is True` so MagicMock sessions in tests don't match.)
        try:
            checker = getattr(session, "is_desktop_session", None)
            if callable(checker) and checker() is True:
                return arguments
        except Exception:
            pass

        keyword_lower = keyword.lower().replace(" ", "_").replace("-", "_")

        # Remove library prefix if present
        if "." in keyword_lower:
            keyword_lower = keyword_lower.split(".", 1)[1]

        # Check if timeout is already in arguments (as named argument)
        for arg in arguments:
            if isinstance(arg, str) and arg.lower().startswith("timeout="):
                logger.debug(f"Timeout already present in arguments for {keyword}")
                return arguments

        # Browser Library keywords that ACTUALLY accept timeout parameter
        # NOTE: Most action keywords (click, fill_text, etc.) do NOT accept timeout
        # They use global browser timeout set via "Set Browser Timeout"
        # Only explicit wait keywords accept timeout parameter
        browser_library_timeout_keywords = {
            # Wait operations - these actually accept timeout
            "wait_for_elements_state": "timeout",
            "wait_for_condition": "timeout",
            "wait_for_navigation": "timeout",
            "wait_for_request": "timeout",
            "wait_for_response": "timeout",
            "wait_for_function": "timeout",
            "wait_for_load_state": "timeout",
            "wait_until_network_is_idle": "timeout",
        }
        # NOTE: These keywords do NOT accept timeout parameter directly:
        # - click, fill_text, fill_secret, type_text, press_keys
        # - check_checkbox, uncheck_checkbox, select_options
        # - hover, focus, scroll_to_element
        # - get_text, get_attribute, get_property, get_element_count, get_element_states
        # They all use the global browser timeout

        # SeleniumLibrary keywords that accept timeout parameter.
        # NOTE: Action keywords (click_element, click_button, input_text, etc.)
        # do NOT accept a timeout= named parameter. SeleniumLibrary interprets
        # unknown named args as Selenium Keys modifiers, causing:
        #   ValueError: 'TIMEOUT=5.0' modifier does not match to Selenium Keys
        # Only explicit wait keywords accept a timeout parameter.
        #
        # Value = positional index of the timeout argument in the keyword
        # signature.  Used to detect when timeout is already provided as a
        # positional arg (prevents "got multiple values for argument 'timeout'").
        selenium_library_timeout_keywords = {
            # signature: (locator, timeout=None, error=None)  → timeout at idx 1
            "wait_until_element_is_visible": 1,
            "wait_until_element_is_not_visible": 1,
            "wait_until_element_is_enabled": 1,
            "wait_until_element_is_not_enabled": 1,
            # signature: (locator, text, timeout=None, error=None)  → timeout at idx 2
            "wait_until_element_contains": 2,
            "wait_until_element_does_not_contain": 2,
            # signature: (text, timeout=None, error=None)  → timeout at idx 1
            "wait_until_page_contains": 1,
            "wait_until_page_does_not_contain": 1,
            # signature: (locator, timeout=None, error=None)  → timeout at idx 1
            "wait_until_page_contains_element": 1,
            "wait_until_page_does_not_contain_element": 1,
            # signature: (condition, timeout=None, error=None)  → timeout at idx 1
            # Shared with Browser Library — owner detection routes correctly.
            "wait_for_condition": 1,
        }

        # Convert timeout to seconds (most RF libraries use seconds)
        timeout_seconds = timeout_ms / 1000.0

        # Determine the ACTUAL owning library via RF namespace resolution.
        # Shared keywords (e.g. Wait For Condition) exist in both Browser
        # Library and SeleniumLibrary with different timeout formats.  The
        # static keyword maps below can't distinguish which library will
        # execute the keyword, so we ask the RF namespace directly.
        owner_library = None
        try:
            from robot.running.namespace import EXECUTION_CONTEXTS as _EC  # noqa: N811
            _ctx = _EC.current
            if _ctx:
                _runner = _ctx.namespace.get_runner(keyword)
                _owner = getattr(getattr(_runner, 'keyword', None), 'owner', None)
                owner_library = getattr(_owner, 'name', None)
        except Exception:
            pass

        # Route to the correct timeout map based on authoritative owner.
        # When owner_library is known, skip the other library's map entirely
        # to avoid format mismatches on shared keywords.
        if owner_library == "Browser" and keyword_lower in browser_library_timeout_keywords:
            timeout_arg = f"timeout={timeout_ms}ms"
            logger.debug(f"Injecting Browser Library timeout: {timeout_arg} for {keyword}")
            return list(arguments) + [timeout_arg]

        if owner_library == "SeleniumLibrary" and keyword_lower in selenium_library_timeout_keywords:
            timeout_pos_idx = selenium_library_timeout_keywords[keyword_lower]
            if len(arguments) > timeout_pos_idx:
                logger.debug(
                    f"Timeout already provided as positional arg at index {timeout_pos_idx} for {keyword}"
                )
                return arguments
            timeout_arg = f"timeout={timeout_seconds}"
            logger.debug(f"Injecting SeleniumLibrary timeout: {timeout_arg} for {keyword}")
            return list(arguments) + [timeout_arg]

        # Fallback when RF context is unavailable (e.g. bridge mode):
        # check Browser Library map first, then SeleniumLibrary map.
        if owner_library is None:
            if keyword_lower in browser_library_timeout_keywords:
                timeout_arg = f"timeout={timeout_ms}ms"
                logger.debug(f"Injecting Browser Library timeout (fallback): {timeout_arg} for {keyword}")
                return list(arguments) + [timeout_arg]

            if keyword_lower in selenium_library_timeout_keywords:
                timeout_pos_idx = selenium_library_timeout_keywords[keyword_lower]
                if len(arguments) > timeout_pos_idx:
                    logger.debug(
                        f"Timeout already provided as positional arg at index {timeout_pos_idx} for {keyword}"
                    )
                    return arguments
                timeout_arg = f"timeout={timeout_seconds}"
                logger.debug(f"Injecting SeleniumLibrary timeout (fallback): {timeout_arg} for {keyword}")
                return list(arguments) + [timeout_arg]

        # For navigation keywords, no timeout injection needed as they have implicit timeouts
        # For other keywords, return original arguments
        return arguments

    def _normalize_variable_name(self, name: str) -> str:
        """Normalize variable name to Robot Framework format."""
        if not name.startswith("${") or not name.endswith("}"):
            return f"${{{name}}}"
        return name

    def _validate_assignment_compatibility(
        self, keyword: str, assign_to: Union[str, List[str]]
    ) -> None:
        """Validate if keyword is appropriate for variable assignment."""
        if not assign_to:
            return

        # Keywords that typically return useful values for assignment
        returnable_keywords = {
            # String operations
            "Get Length",
            "Get Substring",
            "Replace String",
            "Split String",
            "Convert To Uppercase",
            "Convert To Lowercase",
            "Strip String",
            # Web automation - element queries
            "Get Text",
            "Get Title",
            "Get Location",
            "Get Element Count",
            "Get Element Attribute",
            "Get Element Size",
            "Get Element Position",
            "Get Window Size",
            "Get Window Position",
            "Get Page Source",
            # Web automation - Browser Library
            "Get Url",
            "Get Title",
            "Get Text",
            "Get Attribute",
            "Get Property",
            "Get Element Count",
            "Get Page Source",
            "Evaluate JavaScript",
            # Conversions
            "Convert To Integer",
            "Convert To Number",
            "Convert To String",
            "Convert To Boolean",
            "Evaluate",
            # Collections
            "Get From List",
            "Get Slice From List",
            "Get Length",
            "Get Index",
            "Create List",
            "Create Dictionary",
            "Get Dictionary Keys",
            "Get Dictionary Values",
            # Built-in
            "Set Variable",
            "Get Variable Value",
            "Get Time",
            "Get Environment Variable",
            # System operations
            "Run Process",
            "Run",
            "Get Environment Variable",
            # API - RequestsLibrary: request keywords return the Response object
            "On Session",  # matches GET/POST/PUT/DELETE/PATCH/HEAD/OPTIONS On Session
            "GET On Session",
            "POST On Session",
            "PUT On Session",
            "DELETE On Session",
            "PATCH On Session",
            "HEAD On Session",
            "OPTIONS On Session",
        }

        keyword_lower = keyword.lower()
        found_match = False

        for returnable in returnable_keywords:
            if (
                returnable.lower() in keyword_lower
                or keyword_lower in returnable.lower()
            ):
                found_match = True
                break

        if not found_match:
            logger.warning(
                f"Keyword '{keyword}' may not return a useful value for assignment. "
                f"Typical returnable keywords include: Get Text, Get Length, Get Title, etc."
            )

        # Validate assignment count for known multi-return keywords
        multi_return_keywords = {
            "Split String": "Can return multiple parts when max_split is used",
            "Get Time": "Can return multiple time components",
            "Run Process": "Returns stdout and stderr",
            "Get Slice From List": "Can return multiple items",
        }

        for multi_keyword, description in multi_return_keywords.items():
            if multi_keyword.lower() in keyword_lower:
                if isinstance(assign_to, str):
                    logger.info(
                        f"'{keyword}' {description}. Consider using list assignment: ['part1', 'part2']"
                    )
                break

    async def _execute_keyword_internal(
        self,
        session: ExecutionSession,
        step: ExecutionStep,
        browser_library_manager: Any,
        library_prefix: str = None,
        resolved_arguments: List[str] = None,
    ) -> Dict[str, Any]:
        """Execute a specific keyword with error handling and library prefix support."""
        try:
            keyword_name = step.keyword
            # Use resolved arguments if provided, otherwise fall back to step arguments
            args = (
                resolved_arguments if resolved_arguments is not None else step.arguments
            )

            orchestrator = self.keyword_discovery
            session_libraries = self._get_session_libraries(session)
            web_automation_lib = session.get_web_automation_library()
            keyword_info = None

            if session_libraries:
                keyword_info = orchestrator.find_keyword(
                    keyword_name, session_libraries=session_libraries
                )
                logger.debug(
                    f"Session-aware keyword discovery: '{keyword_name}' in session libraries {session_libraries} → {keyword_info.library if keyword_info else None}"
                )
            elif web_automation_lib:
                active_library = (
                    web_automation_lib
                    if web_automation_lib in ["Browser", "SeleniumLibrary"]
                    else None
                )
                keyword_info = orchestrator.find_keyword(
                    keyword_name, active_library=active_library
                )
                logger.debug(
                    f"Active library keyword discovery: '{keyword_name}' with active_library='{active_library}' → {keyword_info.library if keyword_info else None}"
                )
            else:
                keyword_info = orchestrator.find_keyword(keyword_name)
                logger.debug(
                    f"Global keyword discovery: '{keyword_name}' → {keyword_info.library if keyword_info else None}"
                )

            if keyword_info is None:
                logger.debug(
                    f"Keyword '{keyword_name}' not found; ensuring session libraries are loaded"
                )
                await orchestrator._ensure_session_libraries(
                    session.session_id, keyword_name
                )
                session_libraries = self._get_session_libraries(session)
                web_automation_lib = session.get_web_automation_library()
                if session_libraries:
                    keyword_info = orchestrator.find_keyword(
                        keyword_name, session_libraries=session_libraries
                    )
                    logger.debug(
                        f"Post-loading session-aware discovery: '{keyword_name}' in session libraries {session_libraries} → {keyword_info.library if keyword_info else None}"
                    )
                elif web_automation_lib:
                    active_library = (
                        web_automation_lib
                        if web_automation_lib in ["Browser", "SeleniumLibrary"]
                        else None
                    )
                    keyword_info = orchestrator.find_keyword(
                        keyword_name, active_library=active_library
                    )
                    logger.debug(
                        f"Post-loading active library discovery: '{keyword_name}' with active_library='{active_library}' → {keyword_info.library if keyword_info else None}"
                    )
                else:
                    keyword_info = orchestrator.find_keyword(keyword_name)
                    logger.debug(
                        f"Post-loading global discovery: '{keyword_name}' → {keyword_info.library if keyword_info else None}"
                    )

            if keyword_info and keyword_info.library == "Browser":
                logger.info(
                    f"Browser Library keyword detected: {keyword_name} - forcing regular execution mode"
                )

            library_from_map = self._get_library_for_keyword(keyword_name)
            plugin_override = self.plugin_manager.get_keyword_override(
                keyword_info.library if keyword_info else library_from_map,
                keyword_name,
            )
            if plugin_override:
                override_result = await asyncio.to_thread(
                    plugin_override, session, keyword_name, args, keyword_info
                )
                if override_result is not None:
                    library_to_import = (
                        keyword_info.library if keyword_info else library_from_map
                    )
                    if library_to_import:
                        session.import_library(library_to_import, force=True)
                    return override_result

            if self.override_registry and keyword_info:
                override_handler = self.override_registry.get_override(
                    keyword_name, keyword_info.library
                )
                if override_handler:
                    logger.info(
                        f"OVERRIDE: Using override handler {type(override_handler).__name__} for {keyword_name} from {keyword_info.library}"
                    )
                    override_result = await override_handler.execute(
                        session, keyword_name, args, keyword_info
                    )
                    if override_result is not None:
                        session.import_library(keyword_info.library, force=True)
                        logger.info(
                            f"OVERRIDE: Successfully executed {keyword_name} with {keyword_info.library}, imported to session - RETURNING EARLY"
                        )
                        return {
                            "success": override_result.success,
                            "output": override_result.output
                            or f"Executed {keyword_name}",
                            "error": override_result.error,
                            "variables": {},
                            "state_updates": override_result.state_updates or {},
                        }

            # Determine library to use based on session configuration
            web_automation_lib = session.get_web_automation_library()
            current_active = session.get_active_library()
            session_type = session.get_session_type()

            # CRITICAL FIX: Respect session type boundaries
            if session_type.value in [
                "xml_processing",
                "api_testing",
                "data_processing",
                "system_testing",
            ]:
                # Typed sessions should not use web automation auto-detection
                logger.debug(
                    f"Session type '{session_type.value}' - skipping web automation auto-detection"
                )

            elif web_automation_lib:
                # Session has a specific web automation library imported - use it
                if web_automation_lib == "Browser" and (
                    not current_active or current_active == "auto"
                ):
                    browser_library_manager.set_active_library(session, "browser")
                    logger.debug("Using session's web automation library: Browser")
                elif web_automation_lib == "SeleniumLibrary" and (
                    not current_active or current_active == "auto"
                ):
                    browser_library_manager.set_active_library(session, "selenium")
                    logger.debug(
                        "Using session's web automation library: SeleniumLibrary"
                    )

            # Non-context branches removed in context-only mode

        except Exception as e:
            logger.error(f"Error in keyword execution: {e}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {},
            }

    async def _execute_keyword_with_context(
        self,
        session: ExecutionSession,
        keyword: str,
        arguments: List[Any],
        assign_to: Optional[Union[str, List[str]]] = None,
        browser_library_manager: Any = None,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute keyword within full Robot Framework native context.

        This uses RF's native execution context to enable proper execution of
        keywords like Evaluate, Set Test Variable, etc. that require RF context.

        Args:
            session: ExecutionSession to run in
            keyword: Robot Framework keyword name
            arguments: List of arguments for the keyword
            assign_to: Optional variable assignment
            browser_library_manager: BrowserLibraryManager instance
            timeout_ms: Timeout in milliseconds from TimeoutPolicy

        Returns:
            Execution result with status, output, and state
        """
        try:
            session_id = session.session_id

            # Check for plugin keyword overrides BEFORE execution
            # This allows plugins to intercept and reject incompatible keywords
            library_from_map = self._get_library_for_keyword(keyword)
            plugin_override = self.plugin_manager.get_keyword_override(
                library_from_map, keyword
            )
            if plugin_override:
                try:
                    # Call the plugin override (it's an async function)
                    override_result = await plugin_override(
                        session, keyword, arguments, None
                    )
                    if override_result is not None:
                        # Plugin returned a result - use it (may be an error)
                        logger.info(
                            f"Plugin override handled keyword '{keyword}': success={override_result.get('success', False)}"
                        )
                        return override_result
                except Exception as e:
                    logger.debug(f"Plugin override for '{keyword}' failed: {e}")

            logger.info(
                f"RF NATIVE CONTEXT: Executing {keyword} with native RF context for session {session_id}"
            )

            # Inject timeout into arguments for keywords that support it
            arguments_with_timeout = self._inject_timeout_into_arguments(
                keyword, list(arguments), timeout_ms, session
            )
            if arguments_with_timeout != arguments:
                logger.debug(f"Timeout injected into arguments: {arguments_with_timeout}")

            # Create or get RF native context for session
            context_info = self.rf_native_context.get_session_context_info(session_id)
            if not context_info["context_exists"]:
                # Create RF native context with session's library search order
                # Use search_order if available, otherwise try loaded_libraries
                if hasattr(session, "search_order") and session.search_order:
                    libraries = list(session.search_order)
                elif hasattr(session, "loaded_libraries") and session.loaded_libraries:
                    libraries = list(session.loaded_libraries)
                else:
                    libraries = []

                # If keyword has explicit library prefix (e.g., 'XML.Parse XML'),
                # ensure it's imported. RF keyword names contain no dots, so the
                # library is everything before the LAST dot (split-on-first-dot
                # truncated dotted names like 'PlatynUI.BareMetal').
                try:
                    if "." in keyword:
                        prefix = keyword.rsplit(".", 1)[0]
                        if prefix and prefix not in libraries:
                            libraries.append(prefix)
                except Exception:
                    pass

                logger.info(f"Creating RF native context with libraries: {libraries}")
                context_result = self.rf_native_context.create_context_for_session(
                    session_id, libraries
                )
                if not context_result.get("success"):
                    logger.error(
                        f"RF native context creation failed: {context_result.get('error')}"
                    )
                    return {
                        "success": False,
                        "error": f"Failed to create RF native context: {context_result.get('error')}",
                        "keyword": keyword,
                        "arguments": arguments,
                    }
                logger.info(
                    f"Created RF native context for session {session_id} with libraries: {libraries}"
                )

            # Sync session search order into the stored RF context so the
            # per-execution search-order restore in execute_keyword_with_context
            # uses the most up-to-date library priority for this session.
            try:
                ctx_info = self.rf_native_context._session_contexts.get(session_id)
                if ctx_info is not None:
                    session_order = getattr(session, "search_order", None)
                    if session_order:
                        ctx_info["imported_libraries"] = list(session_order)
                        # Ensure every session library is ACTUALLY imported
                        # into the live RF namespace. The context may have
                        # been created before the session's libraries were
                        # fully configured (e.g. start_test auto-creation, or
                        # init/import_library after the first step) — run 4
                        # (2026-06-11) hit "No keyword with name 'Take
                        # Screenshot' found" because PlatynUI.BareMetal was
                        # in the session but not in the namespace. Idempotent:
                        # dict lookup only when already imported.
                        _ns = ctx_info.get("namespace")
                        _kw_store = getattr(_ns, "_kw_store", None)
                        _loaded = (
                            getattr(_kw_store, "libraries", {}) if _kw_store else {}
                        )
                        for _lib in session_order:
                            if _lib and _lib not in _loaded:
                                try:
                                    _ns.import_library(_lib, args=(), alias=None)
                                    logger.info(
                                        "On-demand import of session library "
                                        f"'{_lib}' into existing RF context"
                                    )
                                except Exception as _imp_exc:
                                    logger.debug(
                                        "on-demand import of %s failed: %s",
                                        _lib, _imp_exc,
                                    )
            except Exception:
                pass

            # F3: give desktop query/wait keywords a short fast-fail timeout
            # now that the RF context exists (no-op for non-desktop sessions).
            self._ensure_desktop_query_timeout(session)

            # Execute keyword using RF native context with session variables
            result = await asyncio.to_thread(
                self.rf_native_context.execute_keyword_with_context,
                session_id=session_id,
                keyword_name=keyword,
                arguments=arguments_with_timeout,
                assign_to=assign_to,
                session_variables=dict(
                    session.variables
                ),  # Pass original objects to RF Variables
            )

            # Update session variables from RF native context
            if result.get("success") and "variables" in result:
                session.variables.update(result["variables"])
                logger.debug(
                    f"Updated session variables from RF native context: {len(result['variables'])} variables"
                )

            # Bridge RF-context browser state back to session for downstream services
            try:
                if result.get("success") and browser_library_manager is not None:
                    # Detect owning library from RF namespace (authoritative).
                    # Replaces pattern-based detect_library_type_from_keyword()
                    # which returned wrong results for shared keywords like
                    # "Get Text" (exists in both Browser and SeleniumLibrary).
                    lib_type = None
                    try:
                        from robot.running.context import EXECUTION_CONTEXTS as _EC
                        _post_ctx = _EC.current
                        if _post_ctx:
                            _runner = _post_ctx.namespace.get_runner(keyword)
                            _owner = getattr(getattr(_runner, 'keyword', None), 'owner', None)
                            _owner_name = getattr(_owner, 'name', None)
                            if _owner_name == "Browser":
                                lib_type = "browser"
                            elif _owner_name == "SeleniumLibrary":
                                lib_type = "selenium"
                            elif _owner_name == "AppiumLibrary":
                                lib_type = "appium"
                    except Exception:
                        pass  # No RF context → lib_type stays None

                    if lib_type:
                        browser_library_manager.set_active_library(session, lib_type)
                        if lib_type == "browser":
                            state_updates = self._extract_browser_state_updates(
                                keyword, arguments, result.get("output")
                            )
                            self._apply_state_updates(session, state_updates)
                            # Capture page source if applicable
                            if keyword.lower().endswith("get page source") or keyword.lower() == "get page source":
                                out = result.get("output") or result.get("result")
                                if isinstance(out, str) and out:
                                    session.browser_state.page_source = out
                        elif lib_type == "selenium":
                            state_updates = self._extract_selenium_state_updates(
                                keyword, arguments, result.get("output")
                            )
                            self._apply_state_updates(session, state_updates)
                            if keyword.lower().endswith("get source") or keyword.lower() == "get source":
                                out = result.get("output") or result.get("result")
                                if isinstance(out, str) and out:
                                    session.browser_state.page_source = out
            except Exception as _bridge_err:
                # Non-fatal; page source tool has additional fallbacks
                pass

            logger.info(
                f"RF NATIVE CONTEXT: {keyword} executed with result: {result.get('success')}"
            )
            return result

        except Exception as e:
            logger.error(f"RF native context execution failed: {e}")
            import traceback

            logger.error(f"RF native context traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"RF native context execution failed: {str(e)}",
                "keyword": keyword,
                "arguments": arguments,
            }


    async def _execute_builtin_keyword(
        self, session: ExecutionSession, keyword: str, args: List[str]
    ) -> Dict[str, Any]:
        """Execute a built-in Robot Framework keyword."""
        try:
            # First, attempt dynamic execution via orchestrator for non-built-in libraries.
            # This path supports full argument parsing (incl. **kwargs) and works for AppiumLibrary, RequestsLibrary, etc.
            try:
                from robotmcp.core.dynamic_keyword_orchestrator import (
                    get_keyword_discovery,
                )

                orchestrator = get_keyword_discovery()
                dyn_result = await orchestrator.execute_keyword(
                    keyword_name=keyword,
                    args=args,
                    session_variables=session.variables,
                    active_library=None,
                    session_id=session.session_id,
                    library_prefix=None,
                )

                # If orchestrator could resolve and execute the keyword, return immediately
                if dyn_result and dyn_result.get("success"):
                    return dyn_result
            except Exception as dyn_error:
                logger.debug(
                    f"Dynamic orchestrator path failed for '{keyword}': {dyn_error}. Falling back to BuiltIn."
                )

            if not ROBOT_AVAILABLE:
                return {
                    "success": False,
                    "error": "Robot Framework not available for built-in keywords",
                    "output": "",
                    "variables": {},
                    "state_updates": {},
                }

            builtin = BuiltIn()
            keyword_lower = keyword.lower()

            # Handle common built-in keywords
            if keyword_lower == "set variable":
                if args:
                    var_value = args[0]
                    return {
                        "success": True,
                        "result": var_value,  # Store actual return value
                        "output": var_value,
                        "variables": {"${VARIABLE}": var_value},
                        "state_updates": {},
                    }

            elif keyword_lower == "log":
                message = args[0] if args else ""
                logger.info(f"Robot Log: {message}")
                return {
                    "success": True,
                    "result": None,  # Log doesn't return a value
                    "output": message,
                    "variables": {},
                    "state_updates": {},
                }

            elif keyword_lower == "should be equal":
                if len(args) >= 2:
                    if args[0] == args[1]:
                        return {
                            "success": True,
                            "result": True,  # Assertion passed
                            "output": f"'{args[0]}' == '{args[1]}'",
                            "variables": {},
                            "state_updates": {},
                        }
                    else:
                        return {
                            "success": False,
                            "result": False,  # Assertion failed
                            "error": f"'{args[0]}' != '{args[1]}'",
                            "output": "",
                            "variables": {},
                            "state_updates": {},
                        }

            # Try to execute using BuiltIn library
            try:
                # ENHANCEMENT: Use RF native type converter for proper argument processing
                # This handles RequestsLibrary and other complex libraries with named arguments
                logger.info(
                    f"BUILTIN KEYWORD EXECUTION PATH: {keyword} with args: {args}"
                )
                logger.debug("BUILTIN PATH: %s with args: %s", keyword, args)
                logger.debug("BUILTIN ARGS TYPES: %s", [type(arg).__name__ for arg in args])
                try:
                    processed_args = self.rf_converter.parse_and_convert_arguments(
                        keyword,
                        args,
                        library_name=None,
                        session_variables=session.variables,
                    )
                    logger.info(
                        f"RF converter processed {keyword} args: {args} → {processed_args}"
                    )
                    logger.debug("RF CONVERTER SUCCESS: %s", processed_args)
                except Exception as converter_error:
                    logger.warning(
                        f"RF converter failed for {keyword}: {converter_error}, falling back to basic processing"
                    )
                    logger.debug("RF CONVERTER FAILED: %s", converter_error)
                    processed_args = args

                # DUAL HANDLING: RequestsLibrary needs object arguments, others need string arguments
                # FINAL SOLUTION: Inject objects directly before keyword execution
                final_args = self._inject_objects_for_execution(processed_args, session)

                result = builtin.run_keyword(keyword, *final_args)
                return {
                    "success": True,
                    "result": result,  # Store the actual return value
                    "output": str(result) if result is not None else "OK",
                    "variables": {},
                    "state_updates": {},
                }
            except Exception as e:
                # Phase 4: Add comprehensive diagnostics for keyword execution failures
                diagnostics = self._get_keyword_failure_diagnostics(
                    keyword, args, str(e), session
                )
                return {
                    "success": False,
                    "error": f"Built-in keyword execution failed: {str(e)}",
                    "output": "",
                    "variables": {},
                    "state_updates": {},
                    "diagnostics": diagnostics,  # Phase 4: Enhanced diagnostics
                }

        except Exception as e:
            logger.error(f"Error executing built-in keyword {keyword}: {e}")
            # Phase 4: Add diagnostics for outer exception handler too
            diagnostics = self._get_keyword_failure_diagnostics(
                keyword, args, str(e), session
            )
            return {
                "success": False,
                "error": f"Built-in keyword execution failed: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {},
                "diagnostics": diagnostics,  # Phase 4: Enhanced diagnostics
            }

    def _get_keyword_failure_diagnostics(
        self,
        keyword: str,
        args: List[str],
        error_message: str,
        session: ExecutionSession,
    ) -> Dict[str, Any]:
        """
        Phase 4: Get comprehensive diagnostic information for keyword execution failures.

        Args:
            keyword: The keyword that failed
            args: Arguments provided to the keyword
            error_message: The error message from the failure
            session: ExecutionSession for context

        Returns:
            Dictionary with diagnostic information
        """
        # Use the orchestrator's diagnostic capabilities
        from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery

        orchestrator = get_keyword_discovery()

        # Get comprehensive diagnostics from the orchestrator
        diagnostics = orchestrator._get_diagnostic_info(
            keyword_name=keyword,
            session_id=session.session_id,
            active_library=session.get_active_library(),
        )

        # Add keyword executor specific information
        diagnostics["execution_context"] = {
            "execution_path": "builtin_keyword_executor",
            "provided_arguments": args,
            "argument_count": len(args),
            "execution_error": error_message,
            "session_type": session.get_session_type().value,
        }

        # Add Robot Framework specific diagnostics
        try:
            from robot.running.context import EXECUTION_CONTEXTS

            rf_context_available = bool(EXECUTION_CONTEXTS.current)
            diagnostics["robot_framework_context"] = {
                "execution_context_available": rf_context_available
            }
        except:
            diagnostics["robot_framework_context"] = {
                "execution_context_available": False
            }

        return diagnostics

    def _keyword_expects_object_arguments(
        self, keyword: str, arg_index: int, arg_value: Any
    ) -> bool:
        """
        Determine if a keyword expects object arguments at a specific position.

        This is critical for RequestsLibrary which expects dict/list objects for json/data parameters,
        while most other keywords expect string arguments.
        """
        keyword_lower = keyword.lower()

        logger.debug("OBJECT CHECK: keyword=%s, arg_index=%s, arg_value=%s, type=%s",
                     keyword_lower, arg_index, arg_value, type(arg_value).__name__)

        # RequestsLibrary keywords that accept object parameters
        requests_keywords_with_objects = {
            "post": ["json", "data"],
            "put": ["json", "data"],
            "patch": ["json", "data"],
            "post on session": ["json", "data"],
            "put on session": ["json", "data"],
            "patch on session": ["json", "data"],
        }

        if keyword_lower in requests_keywords_with_objects:
            # Check if this is a dict or list object that should be preserved
            if isinstance(arg_value, (dict, list)):
                logger.debug(
                    "PRESERVING OBJECT: RequestsLibrary keyword %s with %s argument",
                    keyword, type(arg_value).__name__,
                )
                return True

        logger.debug("CONVERTING TO STRING: keyword=%s, arg will be converted", keyword)
        # For other complex argument structures that might need objects
        # Add more library-specific logic here as needed

        return False

    def _process_object_preserving_arguments(self, args: List[Any]) -> List[Any]:
        """
        Handle ObjectPreservingArgument objects for Robot Framework execution.

        Robot Framework's argument resolver expects named parameters to be handled
        differently than simple string formatting. For object values, we need to
        pass them as separate arguments or use RF's native parameter handling.
        """
        from robotmcp.components.variables.variable_resolver import (
            ObjectPreservingArgument,
        )

        processed_args = []

        for arg in args:
            if isinstance(arg, ObjectPreservingArgument):
                # CORRECT APPROACH: For RF execution, we need to preserve the object
                # and pass it in a way that RF's ArgumentResolver can handle.
                # Instead of converting to string, we store the object and use a reference
                # that will be resolved during actual keyword execution.

                # Store object in temporary session storage for later injection
                processed_args.append(arg)  # Keep the ObjectPreservingArgument object
            else:
                processed_args.append(arg)

        return processed_args

    def _store_and_reference_objects(self, args: List[Any], session: Any) -> List[str]:
        """
        FINAL SOLUTION: Store ObjectPreservingArgument objects in session and replace with references.

        This stores the actual objects in the session's temporary storage and replaces
        them with placeholder references that can be injected back later.
        """
        from robotmcp.components.variables.variable_resolver import (
            ObjectPreservingArgument,
        )

        processed_args = []

        for arg in args:
            if isinstance(arg, ObjectPreservingArgument):
                # Create a unique reference ID for this object
                import uuid

                ref_id = f"__OBJ_REF_{uuid.uuid4().hex[:8]}"

                # Store the actual object in session temporary storage
                if not hasattr(session, "_temp_objects"):
                    session._temp_objects = {}
                session._temp_objects[ref_id] = arg.value

                # Replace with a reference that includes the parameter name
                processed_args.append(f"{arg.param_name}=${{{ref_id}}}")

                # Also store the reference in session variables for RF to resolve
                session.variables[ref_id] = arg.value

            else:
                processed_args.append(arg)

        return processed_args

    def _inject_objects_for_execution(self, args: List[str], session: Any) -> List[Any]:
        """
        FINAL SOLUTION: Inject actual objects directly at execution time.

        This replaces object reference placeholders with the actual objects
        right before the keyword is executed, bypassing all the complex
        variable resolution issues.
        """
        # Inject objects for RequestsLibrary and other libraries expecting object parameters
        final_args = []

        for arg in args:
            # Handle URL parameter conversion to positional format
            if (
                isinstance(arg, str)
                and arg.startswith("url=")
                and "${__OBJ_REF_" not in arg
            ):
                # Convert URL from named to positional for RequestsLibrary
                url_value = arg[4:]  # Remove 'url=' prefix
                final_args.append(url_value)
            elif isinstance(arg, str) and "${__OBJ_REF_" in arg:
                # This argument contains an object reference - extract and inject the object
                import re

                # Find object reference patterns in the argument
                ref_pattern = r"\$\{(__OBJ_REF_[^}]+)\}"
                matches = re.findall(ref_pattern, arg)

                if matches:
                    # Replace each reference with the actual object
                    processed_arg = arg
                    for ref_id in matches:
                        if (
                            hasattr(session, "_temp_objects")
                            and ref_id in session._temp_objects
                        ):
                            actual_object = session._temp_objects[ref_id]

                            # If the entire argument is just the reference, replace with the object
                            if processed_arg == f"${{{ref_id}}}":
                                final_args.append(actual_object)
                                break
                            # If it's a named parameter, inject the object as the value
                            elif (
                                "=" in processed_arg
                                and f"${{{ref_id}}}" in processed_arg
                            ):
                                param_name = processed_arg.split("=")[0]
                                # Use tuple format for RF named args with objects
                                final_args.append((param_name, actual_object))
                                break
                    else:
                        # No replacement made, keep as string
                        final_args.append(arg)
                else:
                    final_args.append(arg)
            else:
                final_args.append(arg)

        return final_args

    def _process_arguments_with_rf_native_resolver(
        self, keyword: str, args: List[Any], session: Any
    ) -> List[Any]:
        """
        Process arguments using Robot Framework's native ArgumentResolver patterns.

        This is the general solution that handles:
        1. ObjectPreservingArgument objects from variable resolution
        2. Proper argument formatting (named vs positional parameters)
        3. Type preservation for object parameters

        This works for ANY library that expects object parameters, not just RequestsLibrary.
        """
        from robotmcp.components.variables.variable_resolver import (
            ObjectPreservingArgument,
        )

        processed_args = []

        for i, arg in enumerate(args):
            if isinstance(arg, ObjectPreservingArgument):
                # This is a named parameter with an object value
                logger.debug("PROCESSING OBJECT ARG: %s=%s (type: %s)",
                             arg.param_name, arg.value, type(arg.value).__name__)

                # For Robot Framework, we need to handle named parameters properly
                # The RF ArgumentResolver expects either:
                # 1. Positional args followed by named args like: ['value1', 'param2=value2']
                # 2. Or kwargs-style processing

                # Keep it as named parameter but preserve the object
                processed_args.append(f"{arg.param_name}={arg.value}")

            elif isinstance(arg, str) and "=" in arg and arg.count("=") == 1:
                # This is a string-based named parameter, handle URL parameter specially
                param_name, param_value = arg.split("=", 1)

                # For common first positional parameters like 'url', convert to positional
                if param_name == "url" and i == 0:  # First argument and it's URL
                    logger.debug("CONVERTING URL TO POSITIONAL: %s", param_value)
                    processed_args.append(param_value)
                else:
                    # Keep as named parameter
                    processed_args.append(arg)

            else:
                # Regular argument (positional or already processed)
                if not isinstance(arg, str):
                    # Convert non-string args to string
                    processed_args.append(str(arg))
                else:
                    processed_args.append(arg)

        return processed_args

    def _fix_stringified_objects_for_requests_library(
        self,
        keyword: str,
        original_args: List[str],
        resolved_args: List[str],
        session_variables: Dict[str, Any],
    ) -> List[Any]:
        """
        Fix stringified objects and argument format for RequestsLibrary keywords.

        This fixes two issues:
        1. Variable resolution converts objects to strings (e.g., json=${body} becomes "json={'key': 'value'}")
        2. Named parameters need proper formatting for RequestsLibrary (e.g., "url=value" → "value", "json=object" → object)
        """
        keyword_lower = keyword.lower()

        # Only apply this fix for RequestsLibrary keywords that expect object parameters
        requests_keywords_with_objects = {
            "post",
            "put",
            "patch",
            "post on session",
            "put on session",
            "patch on session",
        }

        if keyword_lower not in requests_keywords_with_objects:
            return resolved_args

        # Get the expected signature for this keyword
        from robotmcp.utils.rf_native_type_converter import REQUESTS_LIBRARY_SIGNATURES

        signature = REQUESTS_LIBRARY_SIGNATURES.get(keyword.upper(), [])

        logger.debug("REQUESTS SIGNATURE: %s -> %s", keyword.upper(), signature)

        fixed_args = []
        for i, (orig_arg, resolved_arg) in enumerate(zip(original_args, resolved_args)):
            # Check if this was a named parameter
            if (
                "=" in orig_arg
                and "=" in str(resolved_arg)
                and orig_arg.count("=") == 1
                and str(resolved_arg).count("=") == 1
            ):
                orig_param_name, orig_param_value = orig_arg.split("=", 1)
                resolved_param_name, resolved_param_value = str(resolved_arg).split(
                    "=", 1
                )

                logger.debug("PROCESSING PARAM: %s=%s", orig_param_name, orig_param_value)

                # Handle URL parameter (first positional parameter for session-less methods)
                if orig_param_name == "url" and keyword_lower in [
                    "post",
                    "put",
                    "patch",
                    "get",
                    "delete",
                ]:
                    # URL should be positional, not named
                    logger.debug("CONVERTING URL TO POSITIONAL: %s", resolved_param_value)
                    fixed_args.append(resolved_param_value)
                    continue

                # Handle object parameters (json, data)
                if orig_param_name in ["json", "data"]:
                    # Check if original was a variable reference that should have been an object
                    if (
                        orig_param_value.startswith("${")
                        and orig_param_value.endswith("}")
                        and "[" not in orig_param_value
                    ):
                        var_name = orig_param_value[2:-1]  # Remove ${ and }
                        if var_name in session_variables:
                            original_value = session_variables[var_name]

                            # If the original value is a dict/list but got stringified, restore it
                            if isinstance(original_value, (dict, list)):
                                logger.debug("RESTORING OBJECT FOR %s: %s -> object",
                                             orig_param_name, orig_param_value)
                                # Keep it as named parameter but with restored object
                                fixed_args.append(f"{orig_param_name}={original_value}")
                                continue

                # Default: keep named parameter as-is
                fixed_args.append(resolved_arg)
            else:
                # Non-named parameter, keep as-is
                fixed_args.append(resolved_arg)

        logger.debug("FINAL FIXED ARGS: %s", fixed_args)
        return fixed_args

    def _extract_browser_state_updates(
        self, keyword: str, args: List[str], result: Any
    ) -> Dict[str, Any]:
        """Extract state updates from Browser Library keyword execution."""
        state_updates = {}
        keyword_lower = keyword.lower()

        # Extract state changes based on keyword
        if "new browser" in keyword_lower:
            browser_type = args[0] if args else "chromium"
            state_updates["current_browser"] = {"type": browser_type}
        elif "new context" in keyword_lower:
            state_updates["current_context"] = {
                "id": str(result) if result else "context"
            }
        elif "new page" in keyword_lower:
            url = args[0] if args else ""
            state_updates["current_page"] = {
                "id": str(result) if result else "page",
                "url": url,
            }
        elif "go to" in keyword_lower:
            url = args[0] if args else ""
            state_updates["current_page"] = {"url": url}

        return state_updates

    def _extract_selenium_state_updates(
        self, keyword: str, args: List[str], result: Any
    ) -> Dict[str, Any]:
        """Extract state updates from SeleniumLibrary keyword execution."""
        state_updates = {}
        keyword_lower = keyword.lower()

        # Extract state changes based on keyword
        if "open browser" in keyword_lower:
            state_updates["current_browser"] = {
                "type": args[1] if len(args) > 1 else "firefox"
            }
        elif "go to" in keyword_lower:
            state_updates["current_page"] = {"url": args[0] if args else ""}

        return state_updates

    def _apply_state_updates(
        self, session: ExecutionSession, state_updates: Dict[str, Any]
    ) -> None:
        """Apply state updates to session browser state."""
        if not state_updates:
            return

        browser_state = session.browser_state

        for key, value in state_updates.items():
            if key == "current_browser":
                if isinstance(value, dict):
                    browser_state.browser_type = value.get("type")
            elif key == "current_context":
                if isinstance(value, dict):
                    browser_state.context_id = value.get("id")
            elif key == "current_page":
                if isinstance(value, dict):
                    browser_state.current_url = value.get("url")
                    browser_state.page_id = value.get("id")

    async def _build_response_by_detail_level(
        self,
        detail_level: str,
        result: Dict[str, Any],
        step: ExecutionStep,
        keyword: str,
        arguments: List[str],
        session: ExecutionSession,
        resolved_arguments: List[str] = None,
        vars_before: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Build execution response based on requested detail level."""
        base_response = {
            "success": result["success"],
            "step_id": step.step_id,
            "keyword": keyword,
            "arguments": arguments,  # Show original arguments in response
            "status": step.status,
            "execution_time": step.execution_time,
        }
        # Carry the focus-before-act outcome to the MCP response so off-window/
        # non-visible operations are visible to the agent (change:
        # platynui-focused-execution).
        if result.get("focus_bypassed"):
            base_response["focus_bypassed"] = True
        if result.get("platynui_focus") is not None:
            base_response["platynui_focus"] = result["platynui_focus"]
        # On SUCCESS the failure-only hint path below is skipped, so surface
        # focus warnings (off-window/non-visible) here too.
        if result["success"]:
            _pf = result.get("platynui_focus") or {}
            _pf_warnings = _pf.get("warnings") if isinstance(_pf, dict) else None
            if _pf_warnings:
                base_response["hints"] = [
                    {"type": "platynui_focus_warning", "message": _w}
                    for _w in _pf_warnings
                ]

        if not result["success"]:
            base_response["error"] = result.get("error", "Unknown error")
            # Propagate hints from lower layers or generate as fallback
            hints = result.get("hints") or []
            library_name = result.get("library_name") or self._get_library_for_keyword(
                keyword
            )
            plugin_hints = self.plugin_manager.generate_failure_hints(
                library_name,
                session,
                keyword,
                list(arguments or []),
                str(base_response["error"]),
            )
            if plugin_hints:
                hints = list(plugin_hints) + list(hints)
            if not hints:
                try:
                    from robotmcp.utils.hints import HintContext, generate_hints

                    hctx = HintContext(
                        session_id=session.session_id,
                        keyword=keyword,
                        arguments=list(arguments or []),
                        error_text=str(base_response["error"]),
                        session_search_order=getattr(session, "search_order", None),
                    )
                    hints = generate_hints(hctx)
                except Exception:
                    hints = []
            # Click Link / Click Image guidance: when bare-text locator fails,
            # suggest robust alternatives.  SeleniumLibrary's default strategy
            # uses normalize-space(descendant-or-self::text()) which fails on
            # <a>/<img> elements with embedded children (SVGs, badges, icons)
            # because XPath 1.0 evaluates only the first text node.
            hints = self._add_link_image_locator_guidance(
                keyword, arguments, hints
            )
            # Deduplicate and rank hints: 1 primary + up to 2 secondary
            hints = self._rank_and_deduplicate_hints(hints, detail_level)
            base_response["hints"] = hints

        if detail_level == "minimal":
            # Serialize output to prevent MCP serialization errors with complex objects
            raw_output = result.get("output", "")
            base_response["output"] = self.response_serializer.serialize_for_response(
                raw_output
            )
            # Include assigned variables in all detail levels for debugging
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]

        elif detail_level == "standard":
            # Serialize output for standard detail level
            raw_output = result.get("output", "")
            serialized_output = self.response_serializer.serialize_for_response(
                raw_output
            )

            base_response.update(
                {
                    "output": serialized_output,
                    "active_library": session.get_active_library(),
                }
            )
            # Include assigned variables in standard detail level (serialized for MCP)
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]
            # Compute changed_variables: vars that were added or modified since before execution
            if vars_before is not None:
                changed = {}
                for k, v in session.variables.items():
                    if k not in vars_before or vars_before[k] is not v:
                        # Skip RF built-in constants (start with special chars or are well-known)
                        if k.startswith("${") and k.endswith("}"):
                            inner = k[2:-1]
                        else:
                            inner = k
                        _RF_BUILTINS = frozenset({
                            "DEBUG_FILE", "LOG_FILE", "LOG_LEVEL", "OUTPUT_DIR",
                            "OUTPUT_FILE", "REPORT_FILE", "CURDIR", "EXECDIR",
                            "TEMPDIR", "/", ":", "\\n", "\\t", " ", "SPACE",
                            "EMPTY", "True", "False", "None", "null", "OPTIONS",
                            "PREV_TEST_NAME", "PREV_TEST_STATUS", "PREV_TEST_MESSAGE",
                            "SUITE_NAME", "SUITE_SOURCE", "SUITE_DOCUMENTATION",
                            "SUITE_METADATA", "TEST_NAME", "TEST_DOCUMENTATION",
                            "TEST_TAGS", "TEST_STATUS", "TEST_MESSAGE",
                        })
                        if inner not in _RF_BUILTINS:
                            changed[k] = self.response_serializer.serialize_for_response(v)
                if changed:
                    base_response["changed_variables"] = changed
            # Add resolved arguments for debugging if they differ from original (serialized)
            if resolved_arguments is not None and resolved_arguments != arguments:
                serialized_resolved_args = [
                    self.response_serializer.serialize_for_response(arg)
                    for arg in resolved_arguments
                ]
                base_response["resolved_arguments"] = serialized_resolved_args

        elif detail_level == "full":
            # DUAL STORAGE: Keep ORIGINAL objects in session for RF, serialize ONLY for MCP response
            session_vars_for_response = {}
            for var_name, var_value in session.variables.items():
                # Only serialize for MCP response display, but keep originals in session.variables
                session_vars_for_response[var_name] = (
                    self.response_serializer.serialize_for_response(var_value)
                )

            # Serialize output for full detail level
            raw_output = result.get("output", "")
            serialized_output = self.response_serializer.serialize_for_response(
                raw_output
            )

            # Serialize state_updates to prevent MCP serialization errors
            raw_state_updates = result.get("state_updates", {})
            serialized_state_updates = {}
            for key, value in raw_state_updates.items():
                serialized_state_updates[key] = (
                    self.response_serializer.serialize_for_response(value)
                )

            base_response.update(
                {
                    "output": serialized_output,
                    "session_variables": session_vars_for_response,  # Serialized for MCP response only
                    "state_updates": serialized_state_updates,
                    "active_library": session.get_active_library(),
                    "browser_state": {
                        "browser_type": session.browser_state.browser_type,
                        "current_url": session.browser_state.current_url,
                        "context_id": session.browser_state.context_id,
                        "page_id": session.browser_state.page_id,
                    },
                    "step_count": session.step_count,
                    "duration": session.duration,
                }
            )
            # Include assigned variables in full detail level
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]
            # Always include resolved arguments in full detail for debugging (serialized)
            if resolved_arguments is not None:
                serialized_resolved_args = [
                    self.response_serializer.serialize_for_response(arg)
                    for arg in resolved_arguments
                ]
                base_response["resolved_arguments"] = serialized_resolved_args

        else:
            # Unrecognized detail_level — fall back to minimal (includes output)
            logger.warning(
                f"Unrecognized detail_level '{detail_level}', "
                f"falling back to 'minimal'. Valid values: minimal, standard, full"
            )
            raw_output = result.get("output", "")
            base_response["output"] = self.response_serializer.serialize_for_response(
                raw_output
            )
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]

        return base_response

    def get_supported_detail_levels(self) -> List[str]:
        """Get list of supported detail levels."""
        return ["minimal", "standard", "full"]

    def validate_detail_level(self, detail_level: str) -> bool:
        """Validate that the detail level is supported."""
        return detail_level in self.get_supported_detail_levels()

    def _get_selenium_error_guidance(
        self, keyword: str, args: List[str], error_message: str
    ) -> Dict[str, Any]:
        """Generate SeleniumLibrary-specific error guidance for agents."""
        # Get base locator guidance
        guidance = self.rf_converter.get_selenium_locator_guidance(
            error_message, keyword
        )

        # Add keyword-specific guidance
        keyword_lower = keyword.lower()

        if any(
            term in keyword_lower
            for term in ["click", "input", "select", "clear", "wait"]
        ):
            # Element interaction keywords
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' requires a valid element locator as the first argument",
                "Common locator patterns: 'id:elementId', 'name:fieldName', 'css:.className'",
                "Ensure the element is visible and interactable before interaction",
            ]

            # Analyze the locator argument if provided
            if args:
                locator = args[0]
                if not any(strategy in locator for strategy in [":", "="]):
                    guidance["locator_analysis"] = {
                        "provided_locator": locator,
                        "issue": "Locator appears to be missing strategy prefix",
                        "suggestions": [
                            f"Try 'id:{locator}' if it's an ID",
                            f"Try 'name:{locator}' if it's a name attribute",
                            f"Try 'css:{locator}' if it's a CSS selector",
                            f"Try 'xpath://*[@id=\"{locator}\"]' for XPath",
                        ],
                    }
                elif "=" in locator and ":" not in locator:
                    guidance["locator_analysis"] = {
                        "provided_locator": locator,
                        "issue": "Contains '=' but no strategy prefix - may be parsed as named argument",
                        "correct_format": f"name:{locator}"
                        if locator.startswith("name=")
                        else "Use appropriate strategy prefix",
                        "note": "SeleniumLibrary requires 'strategy:value' format, not 'strategy=value'",
                    }

        elif "open" in keyword_lower or "browser" in keyword_lower:
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' manages browser/session state",
                "Ensure proper browser initialization before element interactions",
                "Check browser driver compatibility and installation",
            ]

        return guidance

    def _get_browser_error_guidance(
        self, keyword: str, args: List[str], error_message: str
    ) -> Dict[str, Any]:
        """Generate Browser Library-specific error guidance for agents."""
        # Get base locator guidance
        guidance = self.rf_converter.get_browser_locator_guidance(
            error_message, keyword
        )

        # Add keyword-specific guidance
        keyword_lower = keyword.lower()

        if any(
            term in keyword_lower
            for term in ["click", "fill", "select", "check", "type", "press", "hover"]
        ):
            # Element interaction keywords
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' requires a valid element selector",
                "Browser Library uses CSS selectors by default (no prefix needed)",
                "Common patterns: '.class', '#id', 'button', 'input[type=\"submit\"]'",
                "For complex elements, use cascaded selectors: 'div.container >> .button'",
            ]

            # Analyze the selector argument if provided
            if args:
                selector = args[0]
                guidance.update(self._analyze_browser_selector(selector))

        elif any(
            term in keyword_lower
            for term in ["new browser", "new page", "new context", "go to"]
        ):
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' manages browser/page state",
                "Ensure proper browser initialization sequence",
                "Check browser installation and dependencies",
                "Verify URL accessibility for navigation keywords",
            ]

        elif "wait" in keyword_lower:
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' handles dynamic content and timing",
                "Adjust timeout values for slow-loading elements",
                "Use appropriate wait conditions (visible, hidden, enabled, etc.)",
                "Consider page load states for complete readiness",
            ]

        return guidance

    def _analyze_browser_selector(self, selector: str) -> Dict[str, Any]:
        """Analyze a Browser Library selector and provide specific guidance."""
        analysis = {}

        # Detect selector patterns and provide guidance (order matters - check >>> before >>)
        if ">>>" in selector:
            analysis["iframe_selector_detected"] = {
                "type": "iFrame piercing selector",
                "explanation": "Using >>> to access elements inside frames",
                "tip": "Left side selects frame, right side selects element inside frame",
            }

        elif selector.startswith("#") and not selector.startswith("\\#"):
            analysis["selector_warning"] = {
                "issue": "ID selector may need escaping in Robot Framework",
                "provided_selector": selector,
                "recommended": f"\\{selector}",
                "explanation": "# is a comment character in Robot Framework, use \\# for ID selectors",
            }

        elif ">>" in selector:
            analysis["cascaded_selector_detected"] = {
                "type": "Cascaded selector (good practice)",
                "explanation": "Using >> to chain multiple selector strategies",
                "tip": "Each part of the chain is relative to the previous match",
            }

        elif selector.startswith('"') and selector.endswith('"'):
            analysis["text_selector_detected"] = {
                "type": "Text selector (implicit)",
                "explanation": "Quoted strings are treated as text selectors",
                "equivalent_explicit": f"text={selector}",
                "tip": "Use for exact text matching",
            }

        elif selector.startswith("//") or selector.startswith(".."):
            analysis["xpath_selector_detected"] = {
                "type": "XPath selector (implicit)",
                "explanation": "Selectors starting with // or .. are treated as XPath",
                "equivalent_explicit": f"xpath={selector}",
                "tip": "XPath provides powerful element traversal capabilities",
            }

        elif "=" in selector and any(
            selector.startswith(prefix) for prefix in ["css=", "xpath=", "text=", "id="]
        ):
            strategy = selector.split("=", 1)[0]
            analysis["explicit_strategy_detected"] = {
                "type": f"Explicit {strategy} selector",
                "explanation": f"Using explicit {strategy} strategy",
                "tip": "Good practice to be explicit with selector strategies",
            }

        else:
            analysis["implicit_css_detected"] = {
                "type": "CSS selector (implicit default)",
                "explanation": "Plain selectors are treated as CSS by default",
                "equivalent_explicit": f"css={selector}",
                "tip": "Browser Library defaults to CSS selectors",
            }

        return analysis


    def _get_session_libraries(self, session: ExecutionSession) -> List[str]:
        """Get list of libraries loaded in the session for session-aware keyword resolution.

        Args:
            session: ExecutionSession to get libraries from

        Returns:
            List of library names loaded in the session
        """
        session_libraries = []

        # Try to get loaded libraries from session
        if hasattr(session, "loaded_libraries") and session.loaded_libraries:
            session_libraries = list(session.loaded_libraries)
        elif hasattr(session, "search_order") and session.search_order:
            session_libraries = list(session.search_order)
        elif hasattr(session, "imported_libraries") and session.imported_libraries:
            session_libraries = list(session.imported_libraries)

        # Always include core built-in libraries
        builtin_libraries = ["BuiltIn", "Collections", "String"]
        for lib in builtin_libraries:
            if lib not in session_libraries:
                session_libraries.append(lib)

        logger.debug(f"Session libraries for keyword resolution: {session_libraries}")
        return session_libraries
