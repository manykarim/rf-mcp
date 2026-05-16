"""Wrapper-locator suggester for pre-validation visibility failures.

When a Browser Library element fails pre-validation with
"missing required states: visible", the element itself may be hidden
(e.g., a native checkbox/radio wrapped in a styled ``<label>``), while
a visible parent *is* the intended interaction target.

This module runs a cheap (<100 ms) JavaScript ancestor-walk and returns
ordered suggestions pointing the LLM at patterns like:

    *css=label >> id=gendermale
    Click    text=Cliff Diving
    section[style="display: block;"] >> id=fieldname

Known gap: only Browser Library (Playwright) is supported.
Selenium/Appium callers receive None and are unaffected.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum ancestor hops to walk before giving up.
_MAX_ANCESTOR_HOPS = 10

# Budget (seconds) for the entire suggest() call. Soft-enforced.
_BUDGET_S = 0.10

# JavaScript that walks up the ancestor chain looking for a visible
# wrapper element.  Runs entirely in the page context so it is cheap.
_ANCESTOR_PROBE_JS = """
(locatorCss) => {
    const MAX_HOPS = 10;

    // Resolve the element from a simplified CSS expression.
    // We strip Browser-library prefixes (id=, css=, *css=) so that
    // document.querySelector can parse the expression.
    function toCss(raw) {
        raw = raw.trim();
        // Remove *css= or css= prefix
        raw = raw.replace(/^\\*?css=/, '');
        // Convert id=foo  to  #foo
        raw = raw.replace(/^id=([\\w-]+)$/, '#$1');
        // Convert text=foo to a querySelectorAll fallback (handled below)
        return raw;
    }

    // Check if element is visually rendered (getBoundingClientRect + style).
    function isVisible(el) {
        if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    }

    const css = toCss(locatorCss);
    let el = null;
    try {
        el = document.querySelector(css);
    } catch (e) {
        return null;  // unparseable CSS — give up
    }
    if (!el) return null;

    const suggestions = [];
    let current = el.parentElement;
    let hops = 0;

    while (current && hops < MAX_HOPS) {
        if (isVisible(current)) {
            const tag = current.tagName.toLowerCase();
            const role = current.getAttribute('role') || '';
            const ariaLabel = current.getAttribute('aria-label') || '';
            const hasOnclick = !!current.getAttribute('onclick');
            const isButton = tag === 'button' || role === 'button';
            const labelText = (tag === 'label') ? (current.textContent || '').trim() : '';

            // Priority 1: <label> — most common SPA wrapper for custom checkbox/radio
            if (tag === 'label') {
                // Prefer *css=label >> original_locator pattern
                suggestions.push({
                    type: 'label_wrapper',
                    labelText: labelText.substring(0, 60),
                    forAttr: current.getAttribute('for') || null,
                    selectorPrefix: '*css=label',
                    action_keyword: 'Check Checkbox',
                });
                // Also suggest Click text= if label has short, unique text
                if (labelText && labelText.length <= 60) {
                    suggestions.push({
                        type: 'label_text',
                        labelText: labelText,
                        selectorPrefix: null,
                        action_keyword: 'Click',
                    });
                }
            }

            // Priority 2: <section> or <div> with display:block (wizard step pattern)
            if ((tag === 'section' || tag === 'div') && isVisible(current)) {
                const inlineStyle = current.getAttribute('style') || '';
                if (inlineStyle.includes('display') && inlineStyle.includes('block')) {
                    // Normalise to canonical form
                    const styleAttr = tag + '[style="display: block;"]';
                    suggestions.push({
                        type: 'scoped_visible_container',
                        containerSelector: styleAttr,
                        action_keyword: 'Click',
                    });
                }
            }

            // Priority 3: any clickable ancestor (button, role=button, onclick, aria-label)
            if (isButton || hasOnclick || ariaLabel) {
                const ident = ariaLabel
                    ? ('[aria-label="' + ariaLabel + '"]')
                    : (current.id ? '#' + current.id : tag);
                suggestions.push({
                    type: 'clickable_ancestor',
                    selector: ident,
                    action_keyword: 'Click',
                });
            }

            // Stop once we have 3 suggestions
            if (suggestions.length >= 3) break;
        }
        current = current.parentElement;
        hops++;
    }

    return suggestions.length ? suggestions : null;
}
"""


@dataclass(frozen=True)
class WrapperSuggestion:
    """Immutable value object for a single wrapper-locator suggestion."""

    description: str
    selector: str
    action_keyword: str


def _strip_locator_prefix(locator: str) -> str:
    """Return the CSS-queryable portion of a Browser Library locator string.

    Handles common prefixes: ``*css=``, ``css=``, ``id=``.
    Composite locators (e.g. ``section >> id=foo``) are simplified to the
    rightmost segment which is what ``document.querySelector`` can resolve.
    """
    # Use rightmost segment of a chain locator (>> separator)
    parts = locator.split(">>")
    raw = parts[-1].strip()

    # Strip *css= / css=
    if raw.startswith("*css="):
        return raw[5:]
    if raw.startswith("css="):
        return raw[4:]
    # Keep id= — the JS converts it to #foo
    return raw


def _build_hint_from_raw(raw: dict, original_locator: str) -> WrapperSuggestion | None:
    """Convert a raw JS result dict into a WrapperSuggestion."""
    suggestion_type = raw.get("type", "")

    if suggestion_type == "label_wrapper":
        prefix = raw.get("selectorPrefix", "*css=label")
        selector = f"{prefix} >> {original_locator}"
        kw = raw.get("action_keyword", "Check Checkbox")
        label_text = raw.get("labelText", "")
        desc = f"Use wrapper-label locator — the label is visible, the input is hidden"
        if label_text:
            desc += f" (label text: '{label_text}')"
        return WrapperSuggestion(description=desc, selector=selector, action_keyword=kw)

    if suggestion_type == "label_text":
        label_text = raw.get("labelText", "")
        if not label_text:
            return None
        selector = f"text={label_text}"
        return WrapperSuggestion(
            description=f"Click the visible label text directly",
            selector=selector,
            action_keyword="Click",
        )

    if suggestion_type == "scoped_visible_container":
        container = raw.get("containerSelector", "section[style=\"display: block;\"]")
        # Simplify original locator to just the last segment
        inner = _strip_locator_prefix(original_locator)
        selector = f"{container} >> {inner}"
        return WrapperSuggestion(
            description="Scope to the visible wizard step / container and locate within it",
            selector=selector,
            action_keyword="Click",
        )

    if suggestion_type == "clickable_ancestor":
        sel = raw.get("selector", "")
        if not sel:
            return None
        return WrapperSuggestion(
            description="Click the nearest visible clickable ancestor instead",
            selector=sel,
            action_keyword="Click",
        )

    return None


class WrapperSuggester:
    """Stateless service that probes the page for visible wrapper elements.

    Only operates when the active library is Browser (Playwright).
    All failures are swallowed and return None to preserve the
    "never fail a step on a failed suggester probe" invariant.
    """

    @staticmethod
    async def suggest(
        session: object,
        locator: str,
        keyword: str,
    ) -> Optional[dict]:
        """Run a JavaScript ancestor-walk and return wrapper hints.

        Args:
            session: The active ``ExecutionSession`` (used to check library type).
            locator: The Browser Library locator that failed visibility check.
            keyword: The RF keyword being executed (for context in hints).

        Returns:
            A hint dict suitable for insertion into the ``hints`` list, or None.
            The dict shape is::

                {
                    "type": "wrapper_suggestion",
                    "message": "...",
                    "suggestions": [
                        {"description": "...", "selector": "...", "action_keyword": "..."},
                        ...
                    ]
                }
        """
        start = time.monotonic()

        try:
            # Only attempt for Browser Library sessions.
            active_lib = getattr(
                getattr(session, "browser_state", None), "active_library", None
            ) or ""
            imported = getattr(session, "imported_libraries", []) or []
            is_browser = active_lib.lower() == "browser" or "Browser" in imported

            if not is_browser:
                return None

            raw_suggestions = await asyncio.to_thread(
                _run_js_probe, locator
            )

            elapsed = time.monotonic() - start
            if elapsed > _BUDGET_S:
                logger.debug(
                    "WrapperSuggester exceeded budget (%.0fms) for locator '%s'",
                    elapsed * 1000,
                    locator,
                )

            if not raw_suggestions:
                return None

            suggestions: list[dict] = []
            for raw in raw_suggestions[:3]:
                ws = _build_hint_from_raw(raw, locator)
                if ws is not None:
                    suggestions.append(
                        {
                            "description": ws.description,
                            "selector": ws.selector,
                            "action_keyword": ws.action_keyword,
                        }
                    )

            if not suggestions:
                return None

            first = suggestions[0]
            message = (
                f"Element '{locator}' is hidden inside a visible wrapper. "
                f"Try '{first['action_keyword']}    {first['selector']}' instead."
            )

            return {
                "type": "wrapper_suggestion",
                "message": message,
                "suggestions": suggestions,
            }

        except Exception as exc:
            logger.debug("WrapperSuggester.suggest() soft-failed: %s", exc)
            return None


def _run_js_probe(locator: str) -> Optional[list]:
    """Execute the JS ancestor probe synchronously via Browser Library BuiltIn.

    Called in a thread via ``asyncio.to_thread`` so the event loop is not blocked.
    Returns the raw list of suggestion dicts from JS, or None on any failure.

    Why IIFE: Browser library's ``Evaluate JavaScript(selector, function)`` accepts
    exactly two positional arguments. Passing a third positional to supply the locator
    string causes an argument-count error that is silently swallowed by the broad
    ``except``, returning None and suppressing the wrapper_suggestion hint entirely.
    The IIFE pattern bakes the locator into the JS body so the call is self-contained
    and requires no extra argument.
    """
    try:
        from robot.libraries.BuiltIn import BuiltIn
    except ImportError:
        return None

    try:
        builtin = BuiltIn()
        css_portion = _strip_locator_prefix(locator)
        # Wrap _ANCESTOR_PROBE_JS in an IIFE so the locator is inlined — no third arg needed.
        js_iife = f"() => (({_ANCESTOR_PROBE_JS}))({_json.dumps(css_portion)})"
        result = builtin.run_keyword(
            "Browser.Evaluate JavaScript",
            "NONE",   # selector — Browser ignores this when the function is an IIFE
            js_iife,  # function — exactly the second positional, no third
        )
        if result and isinstance(result, list):
            return result
        return None
    except Exception as exc:
        logger.debug("_run_js_probe failed for locator '%s': %s", locator, exc)
        return None
