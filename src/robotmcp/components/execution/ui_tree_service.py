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
                "elements_of_interest=['gnome-calculator']). To enumerate "
                "interactive controls (Buttons/Text/Edit…) with ready-to-use "
                "descriptors — instead of per-element Query probing — request "
                "the 'actionable_controls' section."
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


# ── Desktop actionable-controls view (change: desktop-actionable-controls) ──
# ui_tree is depth-bounded (DEFAULT_MAX_DEPTH=3) and GTK controls nest 4-6 deep,
# so agents fall back to per-element Query probing (5-8 calls/run) or //control:*
# dumps. This is the desktop analog of the web P8 actionable_elements view: a
# FLAT, role-filtered list of interactive controls with ready app-scoped
# descriptors — depth-unbounded but budget-bounded, and always AUT-scoped.

_INTERACTIVE_ROLES = frozenset({
    "button", "togglebutton", "radiobutton", "checkbox", "menuitem", "menu",
    "combobox", "listitem", "tabitem", "slider", "spinbutton", "link",
    "text", "edit", "entry", "textbox", "textfield",
})


def _role_token(role: Any) -> str:
    """Reduce a node role to its control token (``control:Button`` -> ``Button``)."""
    if not role:
        return ""
    return str(role).rsplit(":", 1)[-1]


def _is_interactive_role(role: Any, roles: frozenset) -> bool:
    tok = _role_token(role).lower()
    return bool(tok) and any(r in tok for r in roles)


def _control_descriptor(app_name: str, role: Any, name: str, occurrence: int) -> str:
    """Build an app-scoped, index-disambiguated control descriptor."""
    role_tok = _role_token(role) or "*"
    if name:
        base = f"/app:*[@Name='{app_name}']//control:{role_tok}[@Name='{name}']"
    else:
        base = f"/app:*[@Name='{app_name}']//control:{role_tok}"
    if occurrence > 1:
        return f"({base})[{occurrence}]"
    return base


def walk_actionable_controls(
    app_node: Any,
    app_name: str,
    *,
    roles: frozenset = _INTERACTIVE_ROLES,
    max_nodes: int = 1500,
    max_elements: int = 80,
    time_budget_s: float = 5.0,
) -> Dict[str, Any]:
    """Flat depth-first walk of ONE application subtree, collecting interactive
    controls with ready descriptors. Pure w.r.t. the node API (role/name/
    children()/attribute()) so it is unit-testable with a fake tree. Never
    raises on a per-node provider error (skips the node); returns partial
    results with a ``truncated`` reason on any exhausted budget.
    """
    import time

    deadline = time.monotonic() + time_budget_s
    controls: List[Dict[str, Any]] = []
    occ: Dict[tuple, int] = {}
    node_budget = max_nodes
    truncated: Optional[str] = None
    # DFS via an explicit stack (children pushed reversed to preserve order).
    stack: List[tuple] = [(app_node, 0)]
    while stack:
        if node_budget <= 0:
            truncated = "max_nodes"
            break
        if time.monotonic() > deadline:
            truncated = "time_budget"
            break
        node, depth = stack.pop()
        node_budget -= 1
        role = getattr(node, "role", None)
        if depth > 0 and _is_interactive_role(role, roles):  # skip the app root itself
            summ = _node_summary(node, include_attributes=True)
            name = summ.get("name") or ""
            key = (_role_token(role).lower(), name)
            occ[key] = occ.get(key, 0) + 1
            controls.append({
                "role": _role_token(role),
                "name": name,
                "descriptor": _control_descriptor(app_name, role, name, occ[key]),
                "enabled": summ.get("isenabled"),
                "visible": summ.get("isvisible"),
                "bounds": summ.get("bounds"),
                "depth": depth,
            })
            if len(controls) >= max_elements:
                truncated = "max_elements"
                break
        try:
            children = list(node.children())
        except Exception:
            continue  # per-node provider hiccup — skip, never raise
        for child in reversed(children):
            stack.append((child, depth + 1))

    result: Dict[str, Any] = {
        "success": True,
        "application": app_name,
        "control_count": len(controls),
        "controls": controls,
    }
    if truncated:
        result["truncated"] = {"reason": truncated}
    return result


def _collect_actionable_controls_sync(
    app_filters: Optional[List[str]],
    *,
    roles: frozenset,
    max_nodes: int,
    max_elements: int,
    time_budget_s: float,
    aut_pid: Optional[int] = None,
) -> Dict[str, Any]:
    """Blocking collection — resolve a SINGLE anchor application (display-scoped)
    and walk only its subtree. Refuses to walk when >1 app matches and no filter
    is given (spec req 2). Run via asyncio.to_thread."""
    from robotmcp.plugins.builtin.platynui_plugin import (
        get_runtime,
        runtime_unavailable_reason,
        clear_runtime_tree_cache,
    )

    runtime = get_runtime()
    if runtime is None:
        return {
            "success": False,
            "error": "platynui runtime unavailable",
            "reason": runtime_unavailable_reason() or "not_installed",
        }
    try:
        clear_runtime_tree_cache()
        apps = list(runtime.evaluate("/app:*"))
        filters_lower = [f.lower() for f in (app_filters or [])]

        # Display scoping (D4): drop host apps on an isolation-marked display.
        scoped_pids: Optional[frozenset] = None
        scoping_active = False
        try:
            from robotmcp.components.execution.desktop_display_safety import (
                classify_bound_display_detailed,
            )

            if classify_bound_display_detailed()["isolation_source"] == "marker":
                scoping_active = True
                scoped_pids = _display_scoped_pids()
                if scoped_pids is not None and aut_pid is not None:
                    scoped_pids = scoped_pids | {int(aut_pid)}
        except Exception:  # pragma: no cover - defensive
            pass

        candidates: List[tuple] = []
        for app in apps:
            name = getattr(app, "name", None) or ""
            if scoping_active and scoped_pids is not None:
                pid = _app_pid(app)
                if pid is not None and pid not in scoped_pids:
                    continue  # host app on isolated display — never the anchor
            if filters_lower:
                if any(f in name.lower() for f in filters_lower):
                    candidates.append((name, app))
            else:
                candidates.append((name, app))

        if not candidates:
            return {
                "success": True,
                "application_count": 0,
                "control_count": 0,
                "controls": [],
                "hint": (
                    f"No application matched {app_filters}."
                    if filters_lower
                    else "No desktop application found on the bound display."
                ),
            }
        if len(candidates) > 1 and not filters_lower:
            # Never walk more than one application (spec req 2).
            return {
                "success": True,
                "requires_app_filter": True,
                "applications": [n for n, _ in candidates],
                "hint": (
                    "Multiple applications present — pass one via "
                    "elements_of_interest (e.g. ['gnome-calculator']). Refusing "
                    "to walk more than one application subtree."
                ),
            }

        app_name, app_node = candidates[0]
        return walk_actionable_controls(
            app_node,
            app_name,
            roles=roles,
            max_nodes=max_nodes,
            max_elements=max_elements,
            time_budget_s=time_budget_s,
        )
    except Exception as exc:
        return {"success": False, "error": f"actionable_controls retrieval failed: {exc}"}


async def get_actionable_controls(
    session: Any,
    app_filters: Optional[List[str]] = None,
    *,
    roles: Optional[List[str]] = None,
    max_nodes: int = 1500,
    max_elements: int = 80,
    time_budget_s: float = 5.0,
) -> Dict[str, Any]:
    """Flat, AUT-scoped, budget-bounded interactive-control view for desktop
    sessions — the desktop analog of the web actionable_elements view. Desktop
    -only; a non-desktop session gets a structured rejection."""
    import asyncio

    is_desktop = getattr(session, "is_desktop_session", None)
    if not (callable(is_desktop) and is_desktop() is True):
        return {
            "success": False,
            "error": "actionable_controls is a desktop (PlatynUI) section only",
            "hint": "Use the 'page_source' section for web/API sessions.",
        }
    aut_pid = getattr(session, "desktop_aut_pid", None)
    role_set = frozenset(r.lower() for r in roles) if roles else _INTERACTIVE_ROLES
    return await asyncio.to_thread(
        _collect_actionable_controls_sync,
        app_filters,
        roles=role_set,
        max_nodes=max_nodes,
        max_elements=max_elements,
        time_budget_s=time_budget_s,
        aut_pid=aut_pid,
    )
