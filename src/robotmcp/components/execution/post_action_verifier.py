"""Post-action verification for Browser Library action keywords.

P5: After a keyword reports success, run a cheap (<50ms) check that the
action actually had an observable effect.  Detects:

1. Fill/Type on a hidden-parent element — element is inside display:none or
   visibility:hidden, so the page never registered the value.
2. Value mismatch after Fill/Type — the input rejected the value (masked
   inputs, read-only via JS, auto-format strips characters).
3. Select mismatch — the requested option was not actually selected.
4. Checked-state mismatch after Check/Uncheck.

Architecture:
- Stateless class; all methods are classmethods or staticmethods.
- Returns a list of warning dicts (empty list = no issues found).
- All failures are soft: they attach warnings to a successful response, they
  never raise and never flip success=False.
- Browser Library only for now; Selenium/Appium are a known gap (see ADR-021).
- The verification timeout budget is 50 ms to stay under the overall latency
  budget for a single step.

Usage::

    warnings = await PostActionVerifier.verify(
        keyword="Fill Text",
        arguments=["css=input#email", "user@example.com"],
        result={"success": True, ...},
        session=session,
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum wall-clock time for the entire post-action verification pass.
_VERIFY_BUDGET_MS: int = 50

# Keywords that target a form field and whose value can be read back.
_FILL_TYPE_KEYWORDS: frozenset = frozenset({
    "fill text",
    "fill secret",
    "type text",
    "type secret",
})

# Keywords that set a checkbox/radio state.
_CHECK_KEYWORDS: frozenset = frozenset({"check checkbox"})
_UNCHECK_KEYWORDS: frozenset = frozenset({"uncheck checkbox"})

# Keywords that choose a <select> option.
_SELECT_KEYWORDS: frozenset = frozenset({
    "select options by",
    "select options",
})


class PostActionVerifier:
    """Stateless post-action verification service for Browser Library keywords."""

    @classmethod
    async def verify(
        cls,
        keyword: str,
        arguments: List[Any],
        result: Dict[str, Any],
        session: Any,
    ) -> List[Dict[str, Any]]:
        """Run all applicable post-action checks and return a list of warnings.

        Returns an empty list when:
        - The keyword is not an action keyword covered here.
        - No Browser Library RF context is available.
        - All checks pass.
        - Any internal error occurs (fail-safe: never raise).
        """
        try:
            return await asyncio.wait_for(
                cls._run_checks(keyword, arguments, result, session),
                timeout=_VERIFY_BUDGET_MS / 1000.0,
            )
        except asyncio.TimeoutError:
            logger.debug(
                f"PostActionVerifier timed out after {_VERIFY_BUDGET_MS}ms for '{keyword}'"
            )
            return []
        except Exception as exc:
            logger.debug(f"PostActionVerifier error for '{keyword}': {exc}")
            return []

    @classmethod
    async def _run_checks(
        cls,
        keyword: str,
        arguments: List[Any],
        result: Dict[str, Any],
        session: Any,
    ) -> List[Dict[str, Any]]:
        """Dispatch to the appropriate checker based on keyword type."""
        kw_lower = keyword.lower().strip()
        warnings: List[Dict[str, Any]] = []

        if not arguments:
            return warnings

        locator = arguments[0] if isinstance(arguments[0], str) else None
        if locator is None:
            return warnings

        # Determine active library from session — Browser-only for P5.
        active_lib = cls._detect_browser_library(session)
        if active_lib != "browser":
            # Selenium/Appium gap noted in ADR-021; skip silently.
            return warnings

        # Hidden-parent check applies to all action keywords on an element.
        if kw_lower in _FILL_TYPE_KEYWORDS | _CHECK_KEYWORDS | _UNCHECK_KEYWORDS | _SELECT_KEYWORDS:
            hidden_warning = await cls._check_hidden_parent(locator)
            if hidden_warning:
                warnings.append(hidden_warning)

        if kw_lower in _FILL_TYPE_KEYWORDS:
            expected_value = arguments[1] if len(arguments) > 1 else None
            if expected_value is not None:
                mismatch = await cls._check_fill_value(locator, str(expected_value))
                if mismatch:
                    warnings.append(mismatch)

        elif kw_lower in _CHECK_KEYWORDS:
            w = await cls._check_checked_state(locator, expected_checked=True)
            if w:
                warnings.append(w)

        elif kw_lower in _UNCHECK_KEYWORDS:
            w = await cls._check_checked_state(locator, expected_checked=False)
            if w:
                warnings.append(w)

        elif kw_lower in _SELECT_KEYWORDS:
            # arguments: [locator, attribute, value] or [locator, value]
            expected_text = arguments[-1] if len(arguments) >= 2 else None
            if expected_text is not None:
                w = await cls._check_select_value(locator, str(expected_text))
                if w:
                    warnings.append(w)

        return warnings

    # ------------------------------------------------------------------
    # Individual check methods — all return a warning dict or None.
    # ------------------------------------------------------------------

    @staticmethod
    async def _check_hidden_parent(locator: str) -> Optional[Dict[str, Any]]:
        """Warn when the element's ancestor chain is display:none or visibility:hidden.

        A successful keyword on such an element usually has no visible effect on
        the page because the browser never processes the interaction.
        """
        js = """
        (locator) => {
            const selector = locator;
            let el;
            try { el = document.querySelector(selector); } catch(e) { return null; }
            if (!el) return null;
            let node = el.parentElement;
            while (node && node !== document.body) {
                const s = window.getComputedStyle(node);
                if (s.display === 'none' || s.visibility === 'hidden') {
                    return node.tagName + (node.id ? '#' + node.id : '');
                }
                node = node.parentElement;
            }
            return null;
        }
        """
        try:
            hidden_ancestor = await asyncio.to_thread(
                _run_browser_js, js, locator
            )
            if hidden_ancestor:
                return {
                    "type": "hidden_parent_warning",
                    "severity": "warning",
                    "message": (
                        f"Action succeeded but element is inside a hidden ancestor "
                        f"('{hidden_ancestor}'). The page may not have registered the action."
                    ),
                    "locator": locator,
                    "hidden_ancestor": hidden_ancestor,
                    "suggestion": (
                        "Ensure the parent container is visible before interacting with "
                        "nested elements. Use 'Wait For Elements State' on the container first."
                    ),
                }
        except Exception as exc:
            logger.debug(f"_check_hidden_parent error for '{locator}': {exc}")
        return None

    @staticmethod
    async def _check_fill_value(
        locator: str, expected_value: str
    ) -> Optional[Dict[str, Any]]:
        """Warn when the input value after Fill does not match the expected value.

        Secrets are redacted: only the fact of mismatch is reported, not the values.
        """
        js = """
        (locator) => {
            let el;
            try { el = document.querySelector(locator); } catch(e) { return null; }
            if (!el) return null;
            if (typeof el.value !== 'undefined') return el.value;
            if (el.isContentEditable) return el.innerText;
            return null;
        }
        """
        try:
            actual_value = await asyncio.to_thread(_run_browser_js, js, locator)
            if actual_value is None:
                return None
            # Normalise: strip trailing whitespace that browsers sometimes add.
            if str(actual_value).strip() != expected_value.strip():
                return {
                    "type": "fill_value_mismatch",
                    "severity": "warning",
                    "message": (
                        "Fill/Type succeeded but the element value does not match "
                        "the expected input. The field may have auto-formatted, "
                        "rejected, or masked the value."
                    ),
                    "locator": locator,
                    "suggestion": (
                        "Read the field value back with 'Get Property' to confirm "
                        "what was actually stored, or use 'Press Keys' for masked inputs."
                    ),
                }
        except Exception as exc:
            logger.debug(f"_check_fill_value error for '{locator}': {exc}")
        return None

    @staticmethod
    async def _check_checked_state(
        locator: str, expected_checked: bool
    ) -> Optional[Dict[str, Any]]:
        """Warn when the checkbox/radio checked state differs from what was requested."""
        js = """
        (locator) => {
            let el;
            try { el = document.querySelector(locator); } catch(e) { return null; }
            if (!el) return null;
            return el.checked;
        }
        """
        try:
            actual_checked = await asyncio.to_thread(_run_browser_js, js, locator)
            if actual_checked is None:
                return None
            if bool(actual_checked) != expected_checked:
                action = "Check" if expected_checked else "Uncheck"
                return {
                    "type": "checked_state_mismatch",
                    "severity": "warning",
                    "message": (
                        f"{action} succeeded but the element checked state is "
                        f"{'unchecked' if expected_checked else 'checked'}. "
                        f"The element may be controlled by JavaScript or require "
                        f"a different interaction."
                    ),
                    "locator": locator,
                    "suggestion": (
                        "Verify the element is a standard checkbox. Some custom "
                        "checkboxes require clicking a label or container instead."
                    ),
                }
        except Exception as exc:
            logger.debug(f"_check_checked_state error for '{locator}': {exc}")
        return None

    @staticmethod
    async def _check_select_value(
        locator: str, expected_text: str
    ) -> Optional[Dict[str, Any]]:
        """Warn when the selected <select> option does not match the expected value."""
        js = """
        (locator) => {
            let el;
            try { el = document.querySelector(locator); } catch(e) { return null; }
            if (!el || el.tagName !== 'SELECT') return null;
            const opt = el.options[el.selectedIndex];
            return opt ? (opt.value + '|||' + opt.text) : null;
        }
        """
        try:
            selected_raw = await asyncio.to_thread(_run_browser_js, js, locator)
            if selected_raw is None:
                return None
            parts = str(selected_raw).split("|||", 1)
            selected_value = parts[0].strip()
            selected_text = parts[1].strip() if len(parts) > 1 else parts[0].strip()
            if (
                expected_text.strip() != selected_value
                and expected_text.strip() != selected_text
            ):
                return {
                    "type": "select_value_mismatch",
                    "severity": "warning",
                    "message": (
                        f"Select succeeded but the selected option "
                        f"(value='{selected_value}', text='{selected_text}') "
                        f"does not match expected '{expected_text}'."
                    ),
                    "locator": locator,
                    "suggestion": (
                        "Use 'Get Selected Options' to inspect available options, "
                        "or verify the attribute+value match for 'Select Options By'."
                    ),
                }
        except Exception as exc:
            logger.debug(f"_check_select_value error for '{locator}': {exc}")
        return None

    @staticmethod
    def _detect_browser_library(session: Any) -> Optional[str]:
        """Detect which browser library owns the session (browser/selenium/appium/None)."""
        # Browser state attribute set by BrowserPlugin during session init.
        browser_state = getattr(session, "browser_state", None)
        if browser_state is not None:
            lib = getattr(browser_state, "active_library", None)
            if lib:
                return str(lib).lower()
        # Fall back to imported_libraries list.
        imported = getattr(session, "imported_libraries", None) or []
        if "Browser" in imported:
            return "browser"
        if "SeleniumLibrary" in imported:
            return "selenium"
        if "AppiumLibrary" in imported:
            return "appium"
        return None


def _run_browser_js(js: str, locator: str) -> Any:
    """Execute JavaScript via Browser Library using BuiltIn.run_keyword.

    This runs in a thread (via asyncio.to_thread) because BuiltIn.run_keyword
    is synchronous.  Returns the JS return value or raises on error.
    """
    try:
        from robot.libraries.BuiltIn import BuiltIn
    except ImportError:
        raise RuntimeError("Robot Framework not available")

    builtin = BuiltIn()
    # Use CSS selector when the locator looks like one; xpath otherwise.
    # For simplicity we pass the locator as-is to the JS function — the JS
    # itself uses document.querySelector which handles CSS selectors.
    # For xpath-style locators we skip JS and return None (unsupported).
    if locator.startswith("//") or locator.lower().startswith("xpath="):
        return None
    # Strip css= prefix if present so querySelector gets a clean selector.
    clean_locator = locator
    for prefix in ("css=", "css:"):
        if locator.lower().startswith(prefix):
            clean_locator = locator[len(prefix):]
            break

    return builtin.run_keyword("Browser.Evaluate Javascript", js, clean_locator)
