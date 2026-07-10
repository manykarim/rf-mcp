"""Desktop UI tree retrieval for PlatynUI sessions (ADR-025).

Provides the ``ui_tree`` section of ``get_session_state`` for
DESKTOP_TESTING sessions. Desktop has no DOM — instead we expose a
scoped snapshot of the accessibility tree via ``platynui_native``.

Performance contract (ADR-025 E3):
* The application list (``/app:*``) is a single-level query (~1s on a
  busy desktop) — always safe.
* Subtrees are expanded ONLY for explicitly requested applications and
  bounded by ``max_depth`` / ``max_children`` — never the whole desktop
  (an unscoped ``//`` walk can take minutes on AT-SPI).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bounds for subtree expansion. Conservative: a 14-app desktop with one
# expanded application stays < ~2s and < ~1.5K tokens.
DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_CHILDREN = 50
DEFAULT_MAX_NODES = 200

# Attributes read per expanded node (PascalCase, PlatynUI convention).
_NODE_ATTRIBUTES = ("Bounds", "IsVisible", "IsEnabled")

# Display-scoping PID cache (change: desktop-evidence-and-display-scoping,
# D4): the batched _NET_WM_PID probe result per DISPLAY, invalidated together
# with the runtime tree cache (a desktop launch changes both).
_DISPLAY_PIDS_CACHE: Dict[str, frozenset] = {}


def clear_display_pid_cache() -> None:
    _DISPLAY_PIDS_CACHE.clear()


def _display_scoped_pids() -> Optional[frozenset]:
    """Cached set of pids owning an X window on the bound display, or None
    when the probe is unavailable."""
    import os as _os

    display = _os.environ.get("DISPLAY", "")
    if not display:
        return None
    cached = _DISPLAY_PIDS_CACHE.get(display)
    if cached is not None:
        return cached
    from robotmcp.components.execution.platynui_focus import x11_display_pids

    pids = x11_display_pids()
    if pids is not None:
        _DISPLAY_PIDS_CACHE[display] = pids
    return pids


def _app_pid(app: Any) -> Optional[int]:
    """ProcessId of an application node (Application pattern attribute)."""
    try:
        value = app.attribute("ProcessId")
        return int(value)
    except Exception:
        return None


def _node_summary(node: Any, include_attributes: bool = False) -> Dict[str, Any]:
    """Serialize a UiNode into a compact dict."""
    summary: Dict[str, Any] = {
        "role": getattr(node, "role", None),
        "name": getattr(node, "name", None) or "",
    }
    namespace = getattr(node, "namespace", None)
    if namespace is not None:
        summary["namespace"] = str(namespace)
    if include_attributes:
        for attr in _NODE_ATTRIBUTES:
            try:
                value = node.attribute(attr)
            except Exception:
                continue
            if value is None:
                continue
            # Rect/Point objects stringify compactly
            summary[attr.lower()] = str(value) if not isinstance(
                value, (str, int, float, bool)
            ) else value
    return summary


def _expand_subtree(
    node: Any,
    depth: int,
    budget: Dict[str, int],
    max_children: int,
) -> Dict[str, Any]:
    """Depth-first expansion of a node, bounded by depth and node budget."""
    entry = _node_summary(node, include_attributes=True)
    if depth <= 0 or budget["nodes"] <= 0:
        return entry

    children: List[Dict[str, Any]] = []
    truncated = False
    try:
        for i, child in enumerate(node.children()):
            if i >= max_children or budget["nodes"] <= 0:
                truncated = True
                break
            budget["nodes"] -= 1
            children.append(
                _expand_subtree(child, depth - 1, budget, max_children)
            )
    except Exception as exc:  # pragma: no cover - provider hiccups
        entry["children_error"] = str(exc)[:200]
    if children:
        entry["children"] = children
    if truncated:
        entry["children_truncated"] = True
    return entry


def _desktop_bounds(runtime: Any):
    """Return (x, y, w, h) of the desktop, or None."""
    try:
        info = runtime.desktop_info()
    except Exception:
        return None
    b = info.get("bounds") if isinstance(info, dict) else None
    if b is None:
        return None
    try:
        if isinstance(b, dict):
            return (float(b.get("x", 0)), float(b.get("y", 0)),
                    float(b.get("width", 0)), float(b.get("height", 0)))
        return (float(b.x()), float(b.y()), float(b.width()), float(b.height()))
    except Exception:
        return None


def _window_visibility(app_node: Any, desktop_bounds) -> Optional[Dict[str, Any]]:
    """Visibility/on-screen state of an application's top-level window.

    Reuses the focus manager's resolution + visibility logic so ui_tree and
    the execution-time precondition agree (change: platynui-focused-execution).
    """
    try:
        from robotmcp.components.execution.platynui_focus import (
            PlatynUIFocusManager,
        )
    except Exception:
        return None
    # Find the first window-surface descendant of the application node.
    window = None
    try:
        for child in app_node.children():
            window = child
            break
    except Exception:
        window = app_node
    if window is None:
        return None
    mgr = PlatynUIFocusManager()
    try:
        visible, warnings = mgr.window_visibility(window)
    except Exception:
        return None
    if visible is None:
        return None
    out: Dict[str, Any] = {"visible": visible}
    if warnings:
        out["reasons"] = warnings
    return out


def _build_exposure_diagnostic(
    app_filters: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    """When no application resolved in the AT-SPI tree, distinguish
    "app window present but no accessibility tree" from "app window absent",
    using the guarded EWMH probe + native providers
    (change: desktop-native-platynui-alignment).

    Returns a structured diagnostic dict, or None when nothing useful can be
    said. Best-effort; never raises.
    """
    try:
        from robotmcp.components.execution.platynui_focus import x11_window_present
        from robotmcp.plugins.builtin.platynui_plugin import native_providers

        presence = x11_window_present(app_names=app_filters or None)

        def _provider_name(p: Any) -> Optional[str]:
            # Native runtime.providers() dicts use keys id/display_name/
            # technology (packages/native/src/runtime.rs) — NOT "name". Prefer
            # the human-readable display_name; fall back defensively.
            if isinstance(p, dict):
                return (
                    p.get("display_name")
                    or p.get("technology")
                    or p.get("id")
                    or p.get("name")
                    or str(p)
                )
            return str(p)

        providers = [
            _provider_name(p) for p in (native_providers() or [])
        ]
        if presence == "present":
            return {
                "type": "accessibility_not_exposed",
                "window_present": True,
                "providers": providers,
                "message": (
                    "A matching application WINDOW is present on the display, "
                    "but it exposes NO accessibility (AT-SPI) tree, so name-based "
                    "locators cannot resolve. This is a GTK/AT-SPI accessibility "
                    "bridge or environment issue — NOT a locator problem."
                ),
                "remediation": [
                    "Ensure the application's accessibility bridge is enabled "
                    "(e.g. launch GTK apps with GTK_A11Y=atspi — the backend "
                    "NAME, since modern GTK rejects GTK_A11Y=1 with "
                    "'Unrecognized accessibility backend' and then exposes NO "
                    "AT-SPI tree — and/or gsettings set "
                    "org.gnome.desktop.interface toolkit-accessibility true).",
                    "Ensure an AT-SPI bus is running (at-spi-bus-launcher / "
                    "at-spi2-registryd).",
                    "Relaunch the app and allow a moment for it to register; "
                    "then refresh via get_session_state(sections=['ui_tree']).",
                ],
            }
        if presence == "absent":
            return {
                "type": "app_window_absent",
                "window_present": False,
                "providers": providers,
                "message": (
                    "No matching application window is present on the bound "
                    "display — the app likely did not start, exited, or launched "
                    "on a different display."
                ),
                "remediation": [
                    "Verify the launch succeeded (the process is running) and "
                    "targets this DISPLAY.",
                ],
            }
        # presence == "unknown" — with no app filters the name-based probe
        # has nothing to match by contract, NOT a probe failure. Consult the
        # batched display-PID probe to tell "display reachable but empty"
        # apart from "probe genuinely unavailable" (change: desktop-test-
        # scoping-and-close-lifecycle, D6). FRESH probe, not the scoping
        # cache: this path runs when ZERO apps resolved — typically right
        # after the AUT exited (e.g. Ctrl+Q), exactly when the cache is
        # stale (run-4 finding: stale cache reported "(X11 probe
        # unavailable)" on a freshly-emptied display).
        if not app_filters:
            from robotmcp.components.execution.platynui_focus import (
                x11_display_pids,
            )

            display_pids = x11_display_pids()
            if display_pids is not None and len(display_pids) == 0:
                return {
                    "type": "display_empty",
                    "window_present": False,
                    "providers": providers,
                    "message": (
                        "The display is reachable but has no application "
                        "windows — the AUT has not been launched on this "
                        "display yet."
                    ),
                    "remediation": [
                        "Launch the application (e.g. Start Process with the "
                        "session's DISPLAY), then refresh via "
                        "get_session_state(sections=['ui_tree']).",
                    ],
                }
        return {
            "type": "accessibility_exposure_undetermined",
            "window_present": None,
            "providers": providers,
            "message": (
                "No application resolved in the accessibility tree and window "
                "presence could not be determined (X11 probe unavailable)."
            ),
        }
    except Exception:  # pragma: no cover - defensive
        return None


def _collect_ui_tree_sync(
    app_filters: Optional[List[str]],
    max_depth: int,
    max_children: int,
    max_nodes: int,
    *,
    aut_pid: Optional[int] = None,
) -> Dict[str, Any]:
    """Blocking collection — run via asyncio.to_thread.

    Lists applications (single-level, fast) and expands subtrees only
    for applications whose Name matches one of ``app_filters``
    (case-insensitive substring match).
    """
    # Force the X11 backend BEFORE any Runtime exists in this process
    # (Wayland portal handshake blocks headless contexts — ADR-025 E2).
    # Use the shared runtime broker (change: platynui-desktop-safety-isolation).
    # Previously this created a fresh pn.Runtime() and shut it down per call —
    # the proven proximate cause of "ProviderError ... not available after
    # shutdown" on the MCP/Robot path. The broker binds once and is reused.
    from robotmcp.plugins.builtin.platynui_plugin import (
        get_runtime,
        runtime_unavailable_reason,
    )

    runtime = get_runtime()
    if runtime is None:
        # Classify WHY the runtime is unavailable, so the caller gets an
        # actionable message instead of a misleading "not installed"
        # (change: desktop-input-and-runtime-diagnostics).
        reason = runtime_unavailable_reason() or "not_installed"
        if reason == "display_connect_failed":
            return {
                "success": False,
                "error": "runtime_display_connect_failed",
                "reason": reason,
                "message": (
                    "The PlatynUI runtime is installed but could not connect to "
                    "the display. Check DISPLAY / XAUTHORITY / XDG_RUNTIME_DIR "
                    "for the MCP server process. The native runtime is one-shot "
                    "(it cannot re-initialize in this process), so after fixing "
                    "the environment you must RESTART the MCP server."
                ),
                "hint": (
                    "Ensure DISPLAY and XAUTHORITY (e.g. the session's Xauthority "
                    "file) are set in the MCP server env, then restart it."
                ),
            }
        if reason == "disposed":
            return {
                "success": False,
                "error": "runtime_disposed",
                "reason": reason,
                "message": (
                    "The PlatynUI runtime was disposed and cannot be "
                    "re-initialized in this process (one-shot native module). "
                    "Restart the MCP server."
                ),
                "hint": "Restart the MCP server.",
            }
        return {
            "success": False,
            "error": "platynui-native not installed or runtime unavailable",
            "reason": reason,
            "hint": "pip install --pre platynui-native",
        }

    try:
        # Always re-read live AT-SPI (the runtime caches the desktop tree).
        # change: desktop-tree-cache-refresh — shared helper.
        from robotmcp.plugins.builtin.platynui_plugin import (
            clear_runtime_tree_cache,
        )

        clear_runtime_tree_cache()
        apps = runtime.evaluate("/app:*")
        filters_lower = [f.lower() for f in (app_filters or [])]
        budget = {"nodes": max_nodes}
        desktop_bounds = _desktop_bounds(runtime)

        # Display scoping (change: desktop-evidence-and-display-scoping, D4):
        # the AT-SPI bus is session-global, NOT display-scoped — on an
        # isolation-marked display the raw /app:* list contains every host
        # desktop application (browsers, password managers, the shell). Drop
        # apps whose process owns no X window on the bound display; keep
        # PID-less apps fail-open (annotated) so the AUT is never hidden.
        scoped_pids: Optional[frozenset] = None
        scoping_active = False
        scoping_unavailable = False
        host_apps_filtered = 0
        try:
            from robotmcp.components.execution.desktop_display_safety import (
                classify_bound_display_detailed,
            )

            if classify_bound_display_detailed()["isolation_source"] == "marker":
                scoping_active = True
                scoped_pids = _display_scoped_pids()
                if scoped_pids is None:
                    scoping_unavailable = True
                elif aut_pid is not None:
                    # The launched AUT is ALWAYS in scope — its window may
                    # not be mapped yet when the probe runs post-launch.
                    scoped_pids = scoped_pids | {int(aut_pid)}
        except Exception as _scope_exc:  # pragma: no cover - defensive
            logger.debug("display scoping skipped: %s", _scope_exc)

        applications: List[Dict[str, Any]] = []
        expanded_count = 0
        for app in apps:
            entry = _node_summary(app)
            if scoping_active and scoped_pids is not None:
                pid = _app_pid(app)
                if pid is None:
                    entry["display_scoped"] = False  # fail-open, annotated
                elif pid not in scoped_pids:
                    host_apps_filtered += 1
                    continue
            name_lower = (entry.get("name") or "").lower()
            if filters_lower and any(f in name_lower for f in filters_lower):
                # Expand this application's subtree within budget
                expanded = _expand_subtree(app, max_depth, budget, max_children)
                entry.update(expanded)
                entry["expanded"] = True
                expanded_count += 1
                # Window visibility/on-screen state for the AUT window
                # (change: platynui-focused-execution, task 4.1).
                _vis = _window_visibility(app, desktop_bounds)
                if _vis is not None:
                    entry["window_visible"] = _vis
            applications.append(entry)

        result: Dict[str, Any] = {
            "success": True,
            "application_count": len(applications),
            "applications": applications,
        }
        if scoping_active:
            if scoping_unavailable:
                result["display_scoping"] = "unavailable"
            else:
                result["host_apps_filtered"] = host_apps_filtered
        if filters_lower:
            result["expanded_applications"] = expanded_count
            if expanded_count == 0:
                result["hint"] = (
                    f"No application matched {app_filters}. Names are "
                    "matched case-insensitively as substrings; see the "
                    "'applications' list for exact names."
                )
                # Distinguish "app window present but no AT-SPI tree" from "app
                # window absent" (change: desktop-native-platynui-alignment).
                _diag = _build_exposure_diagnostic(app_filters)
                if _diag is not None:
                    result["accessibility_diagnostic"] = _diag
        elif not applications:
            # Zero applications at all, no filter — still worth diagnosing
            # whether SOME desktop window exists but nothing is exposed.
            _diag = _build_exposure_diagnostic(None)
            if _diag is not None:
                result["accessibility_diagnostic"] = _diag
        else:
            result["hint"] = (
                "Pass application names via elements_of_interest to expand "
                "their subtrees (bounded depth). Example: "
                "get_session_state(sections=['ui_tree'], "
                "elements_of_interest=['gnome-calculator'])"
            )
        return result
    except Exception as exc:
        return {
            "success": False,
            "error": f"UI tree retrieval failed: {exc}",
            "hint": (
                "Ensure a display is available (X11/XWayland). On pure "
                "Wayland the first run needs an xdg-desktop-portal consent."
            ),
        }
    # NOTE: the runtime is broker-owned and process-shared — do NOT shut it
    # down here (that re-introduced the "not available after shutdown" bug).


async def get_ui_tree(
    session: Any,
    app_filters: Optional[List[str]] = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_children: int = DEFAULT_MAX_CHILDREN,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> Dict[str, Any]:
    """Async entry point for the ``ui_tree`` section.

    Args:
        session: ExecutionSession (must be a desktop session)
        app_filters: Application Name filters to expand (substring,
            case-insensitive). None/empty lists applications only.
        max_depth: Max subtree depth per expanded application.
        max_children: Max children listed per node.
        max_nodes: Global node budget across all expansions.
    """
    is_desktop = getattr(session, "is_desktop_session", None)
    if not (callable(is_desktop) and is_desktop() is True):
        return {
            "success": False,
            "error": "ui_tree is only available for desktop (PlatynUI) sessions",
            "hint": "Web sessions: use sections=['page_source'] instead.",
        }

    _aut_pid = getattr(session, "desktop_aut_pid", None)
    result = await asyncio.to_thread(
        lambda: _collect_ui_tree_sync(
            list(app_filters) if app_filters else None,
            max_depth,
            max_children,
            max_nodes,
            aut_pid=_aut_pid if isinstance(_aut_pid, int) else None,
        )
    )
    # Surface the bound-display safety state (change:
    # platynui-desktop-safety-isolation, task 4.4). Computed via a lightweight
    # EWMH probe — NOT a throwaway PlatynUI runtime.
    try:
        from robotmcp.components.execution.desktop_display_safety import (
            evaluate_safety,
        )

        s = evaluate_safety(session)
        if isinstance(result, dict):
            result["desktop_safety"] = {
                "classification": s["classification"],
                "enforcing": s["enforcing"],
                "operations_allowed": s["allowed"],
            }
    except Exception:  # pragma: no cover - defensive
        pass
    return result


def _rect_to_dict(value: Any) -> Optional[Dict[str, float]]:
    """Normalize an upstream Rect (method-style accessors) or dict to a
    plain JSON-able dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            return {
                "x": float(value.get("x", 0)),
                "y": float(value.get("y", 0)),
                "width": float(value.get("width", 0)),
                "height": float(value.get("height", 0)),
            }
        except Exception:
            return None
    try:
        return {
            "x": float(value.x()),
            "y": float(value.y()),
            "width": float(value.width()),
            "height": float(value.height()),
        }
    except Exception:
        return None


def get_desktop_environment(session: Any) -> Dict[str, Any]:
    """``desktop_environment`` section: prove display identity + input
    confinement for a desktop session (change: platynui-visible-safe-
    targeting, task 3.3).

    Combines the ADR-027 isolation classification (with provenance) and the
    upstream ``Runtime.desktop_info()`` report (technology, bounds, os,
    monitors) so an agent can verify "the app is visible on display :N and
    synthetic input is confined to it" before interacting.
    """
    from robotmcp.components.execution.desktop_display_safety import (
        classify_bound_display_detailed,
    )

    out: Dict[str, Any] = dict(classify_bound_display_detailed())
    try:
        from robotmcp.plugins.builtin.platynui_plugin import get_runtime

        runtime = get_runtime()
        info = runtime.desktop_info() if runtime is not None else None
        if isinstance(info, dict):
            monitors = []
            for mon in info.get("monitors") or []:
                if not isinstance(mon, dict):
                    continue
                monitors.append({
                    "id": mon.get("id"),
                    "name": mon.get("name"),
                    "bounds": _rect_to_dict(mon.get("bounds")),
                })
            out["desktop_info"] = {
                "technology": info.get("technology"),
                "name": info.get("name"),
                "os_name": info.get("os_name"),
                "os_version": info.get("os_version"),
                "bounds": _rect_to_dict(info.get("bounds")),
                "monitors": monitors,
            }
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("desktop_environment: desktop_info unavailable: %s", exc)
    return out
