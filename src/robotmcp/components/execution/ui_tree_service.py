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


def _collect_ui_tree_sync(
    app_filters: Optional[List[str]],
    max_depth: int,
    max_children: int,
    max_nodes: int,
) -> Dict[str, Any]:
    """Blocking collection — run via asyncio.to_thread.

    Lists applications (single-level, fast) and expands subtrees only
    for applications whose Name matches one of ``app_filters``
    (case-insensitive substring match).
    """
    # Force the X11 backend BEFORE any Runtime exists in this process
    # (Wayland portal handshake blocks headless contexts — ADR-025 E2).
    from robotmcp.plugins.builtin.platynui_plugin import ensure_x11_session_env

    ensure_x11_session_env()

    try:
        import platynui_native as pn
    except ImportError as exc:
        return {
            "success": False,
            "error": f"platynui-native not installed: {exc}",
            "hint": "pip install --pre platynui-native",
        }

    runtime = None
    try:
        runtime = pn.Runtime()
        apps = runtime.evaluate("/app:*")
        filters_lower = [f.lower() for f in (app_filters or [])]
        budget = {"nodes": max_nodes}

        applications: List[Dict[str, Any]] = []
        expanded_count = 0
        for app in apps:
            entry = _node_summary(app)
            name_lower = (entry.get("name") or "").lower()
            if filters_lower and any(f in name_lower for f in filters_lower):
                # Expand this application's subtree within budget
                expanded = _expand_subtree(app, max_depth, budget, max_children)
                entry.update(expanded)
                entry["expanded"] = True
                expanded_count += 1
            applications.append(entry)

        result: Dict[str, Any] = {
            "success": True,
            "application_count": len(applications),
            "applications": applications,
        }
        if filters_lower:
            result["expanded_applications"] = expanded_count
            if expanded_count == 0:
                result["hint"] = (
                    f"No application matched {app_filters}. Names are "
                    "matched case-insensitively as substrings; see the "
                    "'applications' list for exact names."
                )
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
    finally:
        if runtime is not None:
            try:
                runtime.shutdown()
            except Exception:  # pragma: no cover
                pass


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

    return await asyncio.to_thread(
        _collect_ui_tree_sync,
        list(app_filters) if app_filters else None,
        max_depth,
        max_children,
        max_nodes,
    )
