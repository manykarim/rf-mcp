"""Desktop init guidance bundle (change: desktop-turn-economy-guidance).

Spike lever #1: desktop agents spent ~60% of their tool calls on DISCOVERY
(``find_keywords`` is near-useless for PlatynUI — 13 consecutive calls in one
run — and agents never called ``get_locator_guidance``). Delivering the PlatynUI
keyword surface + a locator crib directly in the ``manage_session(init)``
response removes that discovery share at its root.

The cheat-sheet is derived once from ``LibraryDocumentation("PlatynUI.BareMetal")``
and cached process-wide. The crib restates the authoritative locator rules from
``rf_native_type_converter.get_platynui_locator_guidance`` in a compact form.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DESKTOP_LIBRARY = "PlatynUI.BareMetal"

# Authoritative locator rules, kept aligned with get_platynui_locator_guidance
# (rf_native_type_converter.py). One rule set, restated compactly for delivery.
_LOCATOR_CRIB: List[str] = [
    "Scope EVERY locator to the app: /app:*[@Name='<app>']//control:... — NEVER start a locator with // (it walks the whole desktop and the guard refuses it).",
    "Set Root to the app once, then use relative control:/item: locators.",
    "On Linux, windows are control:Frame, NOT control:Window (control:Window matches nothing and hangs 30s).",
    "Launch the app first (Process.Start Process), then Query for its control:Frame before acting.",
    "Read a value back with Get Attribute on the scoped node; GTK text views expose native:Text.CharacterCount (not live text content).",
    "Take Screenshot signature is (descriptor, filename, rect) — the FIRST positional is a node descriptor, not a filename; pass filename= for a path.",
    "For the full locator cookbook call get_locator_guidance (topic defaults to the desktop/PlatynUI chapter).",
]

_cache_lock = threading.Lock()
_cached_bundle: Optional[Dict[str, Any]] = None


def _compress_arg(arg: str) -> str:
    """Reduce a libdoc arg spec to ``name`` or ``name=<short default>``.

    ``descriptor: UiNodeDescriptor | None = None`` -> ``descriptor=None``;
    ``filename: ... = platynui-screenshot-{index}.png`` -> ``filename=…``.
    Type annotations are dropped; argument order and defaults are preserved.
    """
    text = str(arg)
    name = text.split(":", 1)[0].split("=", 1)[0].strip()
    if "=" in text:
        default = text.split("=", 1)[1].strip()
        if len(default) > 14:
            default = "…"
        return f"{name}={default}"
    return name


def _compress_signature(name: str, args: Any) -> str:
    """One-line signature preserving argument order, e.g. ``Get Attribute(descriptor, name)``."""
    parts = [_compress_arg(a) for a in (args or [])]
    return f"{name}({', '.join(parts)})"


def _build_cheat_sheet() -> List[str]:
    """Derive the PlatynUI keyword cheat-sheet from libdoc as compact one-line
    signatures (each begins with the keyword name), preserving argument order."""
    from robot.libdocpkg import LibraryDocumentation  # local import: libdoc is heavy

    lib = LibraryDocumentation(_DESKTOP_LIBRARY)
    return [
        _compress_signature(kw.name, getattr(kw, "args", None)) for kw in lib.keywords
    ]


def get_desktop_guidance() -> Optional[Dict[str, Any]]:
    """Return the cached desktop guidance bundle, or None if libdoc is unavailable.

    Soft-fail by contract: any libdoc error yields None so init never fails.
    """
    global _cached_bundle
    if _cached_bundle is not None:
        return _cached_bundle
    with _cache_lock:
        if _cached_bundle is not None:
            return _cached_bundle
        try:
            cheat_sheet = _build_cheat_sheet()
        except Exception as e:  # soft-fail: never break session init
            logger.debug(f"Desktop guidance unavailable (libdoc): {e}")
            return None
        _cached_bundle = {
            "library": _DESKTOP_LIBRARY,
            "keyword_count": len(cheat_sheet),
            "keywords": cheat_sheet,  # compact one-line signatures (name first)
            "locator_crib": _LOCATOR_CRIB,
            "note": "Desktop keyword surface + locator rules — use these instead of find_keywords; call get_locator_guidance for the full cookbook.",
        }
        return _cached_bundle
