"""Context-aware hint generation for failed tool calls.

Produces short, actionable guidance with concrete examples based on the
keyword, arguments, error text, and (optionally) session metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


@dataclass
class Hint:
    title: str
    message: str
    examples: List[Dict[str, Any]]
    relevance: int = 0  # Higher is more relevant


@dataclass
class HintContext:
    session_id: str
    keyword: str
    arguments: List[Any]
    error_text: str = ""
    use_context: Optional[bool] = None
    session_search_order: Optional[List[str]] = None


def _is_url(value: Any) -> bool:
    try:
        return isinstance(value, str) and value.lower().startswith(("http://", "https://"))
    except Exception:
        return False


def _first_arg(ctx: HintContext) -> Any:
    return ctx.arguments[0] if ctx.arguments else None


def _base_url(url: str) -> Optional[str]:
    try:
        pr = urlparse(url)
        if pr.scheme and pr.netloc:
            return f"{pr.scheme}://{pr.netloc}"
    except Exception:
        pass
    return None


def _detect_library(ctx: HintContext) -> str:
    """Return the primary UI library from the session search order."""
    for lib in (ctx.session_search_order or []):
        if lib == "Browser":
            return "Browser"
        if lib == "SeleniumLibrary":
            return "SeleniumLibrary"
        if lib == "AppiumLibrary":
            return "AppiumLibrary"
    return "unknown"


def _has_library(ctx: HintContext, name: str) -> bool:
    return name in (ctx.session_search_order or [])


def _selector_arg(ctx: HintContext) -> str:
    arg = _first_arg(ctx)
    return str(arg) if arg else "<selector>"


# ---------------------------------------------------------------------------
# Element interaction error sub-checks
# ---------------------------------------------------------------------------

def _check_click_intercepted(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"intercepts pointer events|element click intercepted|is not clickable at point|Other element would receive the click", err, re.IGNORECASE):
        return []
    lib = _detect_library(ctx)
    sel = _selector_arg(ctx)
    if lib == "Browser":
        return [Hint(
            title="Click intercepted by overlapping element",
            message="Another element is covering the target element. The most common cause is styled form controls (radio buttons, checkboxes) where a decorative element sits on top of the actual input. Use Click With Options with force=True to bypass actionability checks, or click the overlapping element instead.",
            examples=[
                {"keyword": "Click With Options", "arguments": [sel, "force=True"]},
                {"keyword": "Scroll To Element", "arguments": [sel]},
            ],
            relevance=92,
        )]
    if lib == "SeleniumLibrary":
        return [Hint(
            title="Click intercepted by overlapping element",
            message="Another element is covering the target element at the click point. Common with styled form controls, modals, or sticky headers. Use JavaScript click to bypass, scroll the element into view, or use action_chain=True.",
            examples=[
                {"keyword": "Execute JavaScript", "arguments": ["arguments[0].click()", "ARGUMENTS", "${element}"]},
                {"keyword": "Scroll Element Into View", "arguments": [sel]},
                {"keyword": "Click Element", "arguments": [sel, "action_chain=True"]},
            ],
            relevance=92,
        )]
    if lib == "AppiumLibrary":
        return [Hint(
            title="Tap intercepted by overlay or keyboard",
            message="Another element is covering the target. On mobile, this is commonly caused by the software keyboard, a toast notification, or a permission dialog. Hide the keyboard first, or wait for overlays to disappear.",
            examples=[
                {"keyword": "Hide Keyboard", "arguments": []},
                {"keyword": "Click Element", "arguments": [sel]},
            ],
            relevance=92,
        )]
    return []


def _check_strict_mode_violation(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"strict mode violation.*resolved to \d+ elements", err, re.IGNORECASE):
        return []
    if _detect_library(ctx) != "Browser":
        return []
    sel = _selector_arg(ctx)
    return [Hint(
        title="Selector matches multiple elements",
        message="The selector matched more than one element. Playwright requires exactly one match in strict mode. Use a more specific selector by adding text filters, nth index, or visibility constraints.",
        examples=[
            {"keyword": "Click", "arguments": [f"{sel} >> visible=true"]},
            {"keyword": "Click", "arguments": [f"{sel} >> nth=0"]},
        ],
        relevance=90,
    )]


def _check_invalid_selector(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"invalid selector|not a valid XPath expression|Unexpected token|selector syntax error", err, re.IGNORECASE):
        return []
    return [Hint(
        title="Invalid selector syntax",
        message="The locator expression has a syntax error. Verify the XPath or CSS selector syntax. This cannot be fixed at runtime \u2014 the locator itself must be corrected.",
        examples=[],
        relevance=90,
    )]


def _check_element_outside_viewport(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"element is outside of the viewport|outside.*viewport", err, re.IGNORECASE):
        return []
    lib = _detect_library(ctx)
    sel = _selector_arg(ctx)
    if lib == "Browser":
        return [Hint(
            title="Element is outside the visible viewport",
            message="The element exists but is positioned off-screen (common with hidden inputs styled via CSS). Scroll the element into view, click the associated label instead, or use force=True.",
            examples=[
                {"keyword": "Scroll To Element", "arguments": [sel]},
                {"keyword": "Click With Options", "arguments": [sel, "force=True"]},
            ],
            relevance=88,
        )]
    if lib == "AppiumLibrary":
        return [Hint(
            title="Element is outside the visible viewport",
            message="The element is not in the visible area on screen. On mobile, scroll to bring it into view.",
            examples=[
                {"keyword": "Scroll Down", "arguments": [sel, "timeout=10"]},
                {"keyword": "Swipe", "arguments": ["start_x=500", "start_y=1500", "end_x=500", "end_y=500"]},
            ],
            relevance=88,
        )]
    return []


def _check_element_not_interactable(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"element not interactable|ElementNotInteractableException|not currently visible|element is not visible|element is not enabled", err, re.IGNORECASE):
        return []
    lib = _detect_library(ctx)
    sel = _selector_arg(ctx)
    if lib == "Browser":
        return [Hint(
            title="Element not interactable",
            message="The element exists but cannot be interacted with (hidden, disabled, or not stable). Wait for the element to reach the required state before interacting.",
            examples=[
                {"keyword": "Wait For Elements State", "arguments": [sel, "visible", "timeout=10s"]},
                {"keyword": "Wait For Elements State", "arguments": [sel, "enabled", "timeout=10s"]},
                {"keyword": "Scroll To Element", "arguments": [sel]},
            ],
            relevance=88,
        )]
    if lib == "SeleniumLibrary":
        return [Hint(
            title="Element not interactable",
            message="The element exists in the DOM but is hidden, disabled, or not scrolled into view. Wait for visibility/enabled state or scroll into view.",
            examples=[
                {"keyword": "Wait Until Element Is Visible", "arguments": [sel, "timeout=10s"]},
                {"keyword": "Wait Until Element Is Enabled", "arguments": [sel, "timeout=10s"]},
                {"keyword": "Scroll Element Into View", "arguments": [sel]},
            ],
            relevance=88,
        )]
    if lib == "AppiumLibrary":
        return [Hint(
            title="Element not interactable",
            message="The element exists but cannot be interacted with. On mobile, this is often caused by the keyboard covering the element or the element being off-screen. Hide the keyboard and scroll to the element.",
            examples=[
                {"keyword": "Hide Keyboard", "arguments": []},
                {"keyword": "Scroll Down", "arguments": [sel, "timeout=10"]},
                {"keyword": "Expect Element", "arguments": [sel, "visible", "timeout=10"]},
            ],
            relevance=88,
        )]
    return []


def _check_stale_element(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"stale element reference|element is not attached|not attached to the page document|element reference.*is stale|not present in the current view|expired from the internal cache|Target closed", err, re.IGNORECASE):
        return []
    lib = _detect_library(ctx)
    sel = _selector_arg(ctx)
    if lib == "Browser":
        return [Hint(
            title="Stale element \u2014 DOM was rebuilt",
            message="The element was removed and re-added to the DOM (page navigation, AJAX update, SPA route change). Wait for the element to be re-attached before interacting.",
            examples=[
                {"keyword": "Wait For Elements State", "arguments": [sel, "attached", "timeout=10s"]},
            ],
            relevance=85,
        )]
    if lib == "SeleniumLibrary":
        return [Hint(
            title="Stale element \u2014 DOM was rebuilt",
            message="The element reference is no longer valid because the page was refreshed or updated. Re-locate the element by waiting for it again.",
            examples=[
                {"keyword": "Wait Until Page Contains Element", "arguments": [sel, "timeout=10s"]},
                {"keyword": "Wait Until Element Is Visible", "arguments": [sel, "timeout=10s"]},
            ],
            relevance=85,
        )]
    if lib == "AppiumLibrary":
        return [Hint(
            title="Stale element \u2014 screen changed",
            message="The element reference expired after a screen transition or navigation. Re-find the element by waiting for it to appear.",
            examples=[
                {"keyword": "Expect Element", "arguments": [sel, "visible", "timeout=10"]},
            ],
            relevance=85,
        )]
    return []


def _check_unexpected_alert(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"unexpected alert open|UnexpectedAlertPresentException|alert.*present", err, re.IGNORECASE):
        return []
    lib = _detect_library(ctx)
    if lib == "SeleniumLibrary":
        return [Hint(
            title="Unexpected JavaScript alert blocking interaction",
            message="A JavaScript alert, confirm, or prompt dialog appeared and is blocking all WebDriver commands. Accept or dismiss it before continuing.",
            examples=[
                {"keyword": "Handle Alert", "arguments": ["ACCEPT"]},
                {"keyword": "Handle Alert", "arguments": ["DISMISS"]},
            ],
            relevance=85,
        )]
    if lib == "AppiumLibrary":
        return [Hint(
            title="System dialog blocking interaction",
            message="A system permission dialog or alert is blocking the app. On iOS, use Click Alert Button. On Android, click the button text.",
            examples=[
                {"keyword": "Click Alert Button", "arguments": ["Allow"]},
                {"keyword": "Click Text", "arguments": ["Allow"]},
            ],
            relevance=85,
        )]
    return []


def _check_mobile_context_mismatch(ctx: HintContext, err: str) -> List[Hint]:
    if not _has_library(ctx, "AppiumLibrary"):
        return []
    if not re.search(r"Unknown.*command|unknown command|UnknownCommandError", err, re.IGNORECASE):
        return []
    return [Hint(
        title="Wrong context for mobile command",
        message="The command is not available in the current context. Hybrid apps have both NATIVE_APP and WEBVIEW contexts. Check and switch to the correct context.",
        examples=[
            {"keyword": "Get Current Context", "arguments": []},
            {"keyword": "Switch To Context", "arguments": ["NATIVE_APP"]},
            {"keyword": "Get Contexts", "arguments": []},
        ],
        relevance=85,
    )]


_ELEMENT_NOT_FOUND_RE = re.compile(
    r"no such element|could not be located|did not match any elements|Page should have contained element|waiting for locator",
    re.IGNORECASE,
)


def _check_mobile_scroll_needed(ctx: HintContext, err: str) -> List[Hint]:
    if not _has_library(ctx, "AppiumLibrary"):
        return []
    if not _ELEMENT_NOT_FOUND_RE.search(err):
        return []
    sel = _selector_arg(ctx)
    return [Hint(
        title="Element not found \u2014 try scrolling or inspect page",
        message=(
            "Mobile apps only render visible elements. Use get_session_state to inspect "
            "the page source and verify the locator is correct. The element may be "
            "off-screen (scroll to find it) or you may be in the wrong context "
            "(NATIVE_APP vs WEBVIEW)."
        ),
        examples=[
            {
                "tool": "get_session_state",
                "arguments": {"sections": ["page_source"]},
                "note": "Inspect the page source to verify the locator and visible elements",
            },
            {"keyword": "Scroll Down", "arguments": [sel, "timeout=10"]},
            {"keyword": "Swipe", "arguments": ["start_x=500", "start_y=1500", "end_x=500", "end_y=500"]},
            {"keyword": "Get Current Context", "arguments": []},
        ],
        relevance=85,
    )]


def _check_element_not_found(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"no such element|Unable to locate element|could not be located|did not match any elements|Page should have contained element|waiting for locator", err, re.IGNORECASE):
        return []
    # Skip if AppiumLibrary is present -- mobile scroll hint takes priority
    if _has_library(ctx, "AppiumLibrary"):
        return []
    lib = _detect_library(ctx)
    sel = _selector_arg(ctx)
    # Common tool suggestion: inspect the page to find the correct locator
    inspect_tool = {
        "tool": "get_session_state",
        "arguments": {"sections": ["page_source"], "include_reduced_dom": True},
        "note": "Inspect the ARIA/DOM snapshot to find the correct locator for the element",
    }
    if lib == "Browser":
        return [Hint(
            title="Element not found — inspect the page",
            message=(
                "No element matches the selector. Use get_session_state to inspect the "
                "page DOM/ARIA snapshot and find the correct locator. The element may have "
                "a different id/selector, the page may not be fully loaded, or the element "
                "may be inside an iframe."
            ),
            examples=[
                inspect_tool,
                {"keyword": "Wait For Elements State", "arguments": [sel, "attached", "timeout=15s"]},
            ],
            relevance=82,
        )]
    if lib == "SeleniumLibrary":
        return [Hint(
            title="Element not found — inspect the page",
            message=(
                "No element matches the locator. Use get_session_state to inspect the "
                "page source and find the correct locator. The element may have a different "
                "id/selector, the page may not be fully loaded, or the element may be inside "
                "an iframe."
            ),
            examples=[
                inspect_tool,
                {"keyword": "Wait Until Page Contains Element", "arguments": [sel, "timeout=15s"]},
                {"keyword": "Select Frame", "arguments": ["<frame_locator>"]},
            ],
            relevance=82,
        )]
    return []


def _check_invalid_element_state(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"invalid element state|InvalidElementStateException|may not be manipulated|element is not editable", err, re.IGNORECASE):
        return []
    lib = _detect_library(ctx)
    sel = _selector_arg(ctx)
    if lib == "Browser":
        return [Hint(
            title="Element in wrong state for this operation",
            message="The element cannot accept this operation in its current state (e.g., input is readonly or disabled). Wait for it to become editable.",
            examples=[
                {"keyword": "Wait For Elements State", "arguments": [sel, "editable", "timeout=10s"]},
            ],
            relevance=82,
        )]
    if lib == "SeleniumLibrary":
        return [Hint(
            title="Element in wrong state for this operation",
            message="The element's state does not allow this action (e.g., clearing a readonly field). Wait for it to become enabled.",
            examples=[
                {"keyword": "Wait Until Element Is Enabled", "arguments": [sel, "timeout=10s"]},
            ],
            relevance=82,
        )]
    return []


def _check_frame_issues(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"no such frame|NoSuchFrameException|frame was detached|ERR_ABORTED.*frame", err, re.IGNORECASE):
        return []
    lib = _detect_library(ctx)
    if lib == "Browser":
        return [Hint(
            title="Frame detached or not found",
            message="The target frame was removed or does not exist. This often happens during SPA navigation. Wait for the page to stabilize.",
            examples=[],
            relevance=80,
        )]
    if lib == "SeleniumLibrary":
        return [Hint(
            title="Frame not found",
            message="The target frame/iframe does not exist or has not loaded. Return to the main document first, then switch to the correct frame.",
            examples=[
                {"keyword": "Unselect Frame", "arguments": []},
                {"keyword": "Select Frame", "arguments": ["<frame_locator>"]},
            ],
            relevance=80,
        )]
    return []


def _check_window_issues(ctx: HintContext, err: str) -> List[Hint]:
    if not re.search(r"no such window|NoSuchWindowException", err, re.IGNORECASE):
        return []
    if _detect_library(ctx) != "SeleniumLibrary":
        return []
    return [Hint(
        title="Window or tab not found",
        message="The target browser window or tab was closed or never existed. Switch to the main window or a new window.",
        examples=[
            {"keyword": "Switch Window", "arguments": ["MAIN"]},
            {"keyword": "Switch Window", "arguments": ["NEW"]},
        ],
        relevance=80,
    )]


def _check_timeout(ctx: HintContext, err: str, already_matched: bool) -> List[Hint]:
    if already_matched:
        return []
    if not re.search(r"Timeout \d+ms exceeded|TimeoutException|timed out", err, re.IGNORECASE):
        return []
    lib = _detect_library(ctx)
    # When timeout mentions "waiting for locator", the element likely doesn't exist —
    # suggest inspecting the page rather than just increasing timeout
    is_locator_wait = bool(re.search(r"waiting for locator|waiting for selector", err, re.IGNORECASE))
    inspect_tool = {
        "tool": "get_session_state",
        "arguments": {"sections": ["page_source"], "include_reduced_dom": True},
        "note": "Inspect the ARIA/DOM snapshot to verify the locator and page structure",
    }
    if lib == "Browser":
        if is_locator_wait:
            return [Hint(
                title="Element not found within timeout — inspect the page",
                message=(
                    "The operation timed out waiting for an element. The locator may be "
                    "incorrect or the element may not exist on the page. Use get_session_state "
                    "to inspect the DOM/ARIA snapshot and find the correct locator before retrying."
                ),
                examples=[
                    inspect_tool,
                    {"keyword": "Set Browser Timeout", "arguments": ["30s"]},
                ],
                relevance=75,
            )]
        return [Hint(
            title="Operation timed out",
            message="The operation exceeded its timeout. Consider increasing the browser timeout or using explicit waits with longer durations.",
            examples=[
                {"keyword": "Set Browser Timeout", "arguments": ["30s"]},
            ],
            relevance=70,
        )]
    if lib == "SeleniumLibrary":
        if is_locator_wait:
            return [Hint(
                title="Element not found within timeout — inspect the page",
                message=(
                    "The operation timed out waiting for an element. The locator may be "
                    "incorrect or the element may not exist. Use get_session_state to inspect "
                    "the page source and find the correct locator."
                ),
                examples=[
                    inspect_tool,
                    {"keyword": "Set Selenium Timeout", "arguments": ["30s"]},
                ],
                relevance=75,
            )]
        return [Hint(
            title="Operation timed out",
            message="The wait exceeded its time limit. Increase the timeout or verify the element/condition is achievable.",
            examples=[
                {"keyword": "Set Selenium Timeout", "arguments": ["30s"]},
            ],
            relevance=70,
        )]
    if lib == "AppiumLibrary":
        if is_locator_wait:
            return [Hint(
                title="Element not found within timeout — inspect the page",
                message=(
                    "The operation timed out waiting for an element. The locator may be "
                    "incorrect or the element may not be on screen. Use get_session_state to "
                    "inspect the page source and verify what elements are available."
                ),
                examples=[
                    inspect_tool,
                    {"keyword": "Set Appium Timeout", "arguments": ["30s"]},
                ],
                relevance=75,
            )]
        return [Hint(
            title="Operation timed out",
            message="The operation exceeded its timeout. Check device responsiveness and consider increasing the timeout.",
            examples=[
                {"keyword": "Set Appium Timeout", "arguments": ["30s"]},
            ],
            relevance=70,
        )]
    return []


def _check_element_interaction_errors(ctx: HintContext) -> List[Hint]:
    """Detect element interaction errors and return library-aware hints."""
    err = ctx.error_text or ""
    if not err:
        return []

    hints: List[Hint] = []

    # Check in priority order (most specific first)
    checkers = [
        _check_click_intercepted,
        _check_strict_mode_violation,
        _check_invalid_selector,
        _check_element_outside_viewport,
        _check_element_not_interactable,
        _check_stale_element,
        _check_unexpected_alert,
        _check_mobile_context_mismatch,
        _check_mobile_scroll_needed,
        _check_element_not_found,
        _check_invalid_element_state,
        _check_frame_issues,
        _check_window_issues,
    ]

    for checker in checkers:
        hints.extend(checker(ctx, err))

    # Timeout is a catch-all -- only fire if nothing more specific matched
    hints.extend(_check_timeout(ctx, err, already_matched=len(hints) > 0))

    return hints


def generate_hints(ctx: HintContext) -> List[Dict[str, Any]]:
    """Generate a prioritized list of hint dicts suitable for inclusion in responses."""
    hints: List[Hint] = []
    kw_lower = (ctx.keyword or "").strip().lower()
    err = (ctx.error_text or "")
    args = ctx.arguments or []

    # 1) Control structures misused as keywords
    control_structs = {"try", "if", "for", "end", "except", "while"}
    if kw_lower in control_structs:
        # Specialized FOR guidance
        if kw_lower == "for" or "old for loop syntax" in err.lower():
            # Attempt to extract items and a collection expression from arguments
            items_list: List[Any] = []
            collection_expr: Optional[str] = None
            try:
                args_lower = [str(a) for a in (args or [])]
                if "IN" in args or "in" in args_lower:
                    # Find index of IN (case-insensitive)
                    idx = -1
                    for i, a in enumerate(args):
                        if isinstance(a, str) and a.strip().lower() == "in":
                            idx = i
                            break
                    if idx >= 0:
                        tail = list(args[idx + 1 :])
                        # Stop items at first likely keyword or END
                        stop_tokens = {"end", "should", "log", "evaluate", "set", "dictionary", "create"}
                        for t in tail:
                            ts = str(t).strip()
                            if ts.lower() in stop_tokens or ts.upper() == "END":
                                break
                            items_list.append(ts)
                        # Heuristic: scan remaining for a ${...} json() like expression
                        for t in tail[len(items_list) :]:
                            ts = str(t)
                            if "json()" in ts and "${" in ts:
                                collection_expr = ts
                                break
            except Exception:
                items_list = []
                collection_expr = None
            hints.append(
                Hint(
                    title="Flow Control: Use execute_for_each",
                    message=(
                        "FOR is a control structure. Use the execute_for_each tool with items and steps. "
                        "During each iteration, ${item} (or your chosen item_var) is set in RF context."
                    ),
                    examples=[
                        {
                            "tool": "execute_for_each",
                            "arguments": {
                                "item_var": "key",
                                "items": items_list[:6] or ["firstname", "lastname", "totalprice", "depositpaid"],
                                "steps": [
                                    (
                                        {
                                            "keyword": "Should Contain",
                                            "arguments": [collection_expr or "${post_response.json()['booking']}", "${item}"]
                                        }
                                    )
                                ],
                            },
                        }
                    ],
                    relevance=92,
                )
            )
        else:
            hints.append(
                Hint(
                    title="Flow Control: Use flow tools",
                    message=(
                        "TRY/IF/FOR are control structures. Use flow tools like "
                        "execute_try_except / execute_if / execute_for_each to build flows."
                    ),
                    examples=[
                        {
                            "tool": "execute_try_except",
                            "arguments": {
                                "try_steps": [{"keyword": "Fail", "arguments": ["boom"]}],
                                "except_patterns": ["*"],
                                "except_steps": [{"keyword": "Log", "arguments": ["handled"]}],
                            },
                        },
                        {
                            "tool": "execute_if",
                            "arguments": {
                                "condition": "int($X) == 1",
                                "then_steps": [{"keyword": "Log", "arguments": ["ok"]}],
                            },
                        },
                        {
                            "tool": "execute_for_each",
                            "arguments": {
                                "items": [1, 2],
                                "steps": [{"keyword": "Log", "arguments": ["loop"]}],
                            },
                        },
                    ],
                    relevance=90,
                )
            )

    # 2) Evaluate with non-Python literals or wrong variable syntax
    if kw_lower == "evaluate" and args:
        arg0 = str(args[0])
        if (" : true" in arg0) or (": true" in arg0) or (" true}" in arg0) or (" false" in arg0) or (
            "name 'true' is not defined" in err
        ):
            hints.append(
                Hint(
                    title="Evaluate: Use Python booleans or json.loads",
                    message=(
                        "Evaluate executes Python. Use True/False in dicts or parse JSON with json.loads(...)."
                    ),
                    examples=[
                        {
                            "tool": "execute_step",
                            "keyword": "Evaluate",
                            "arguments": ["{'depositpaid': True}"]
                        },
                        {
                            "tool": "execute_step",
                            "keyword": "Evaluate",
                            "arguments": [
                                "__import__('json').loads('{\"ok\": true}')"
                            ],
                        },
                    ],
                    relevance=85,
                )
            )
        if "${" in arg0 or "Try using '$" in err or ("NameError: name" in err and "not defined" in err):
            hints.append(
                Hint(
                    title="Evaluate: Use $var inside expressions",
                    message=(
                        "Use $var (not ${var}) inside Evaluate. For method calls in other keywords, use ${resp.json()}. "
                        "When indexing with a loop variable, use $dict[$item] or $dict[$item[0]] inside Evaluate."
                    ),
                    examples=[
                        {
                            "tool": "execute_step",
                            "keyword": "Evaluate",
                            "arguments": ["int($resp.status_code)"]
                        },
                        {
                            "tool": "execute_step",
                            "keyword": "Set Variable",
                            "arguments": ["${resp.json()}"]
                        },
                        {
                            "tool": "execute_step",
                            "keyword": "Evaluate",
                            "arguments": ["$created_booking[$item]"]
                        },
                    ],
                    relevance=80,
                )
            )

    # 2b) Non-Evaluate variable resolution with dynamic index (old RF syntax)
    if "resolving variable '${" in err.lower() and "name 'item' is not defined" in err.lower():
        hints.append(
            Hint(
                title="Variables: Use nested ${item} in index or Evaluate",
                message=(
                    "Use nested variable syntax like ${dict[${item}]} in keywords, or switch to Evaluate with $dict[$item] for clarity."
                ),
                examples=[
                    {"tool": "execute_step", "keyword": "Should Be Equal As Strings", "arguments": ["${created_booking[${item}]}", "${expected}"]},
                    {"tool": "execute_step", "keyword": "Evaluate", "arguments": ["$created_booking[$item]"], "assign_to": "actual"},
                ],
                relevance=78,
            )
        )

    # 3) RequestsLibrary shapes
    if kw_lower.endswith(" on session") and args:
        # First arg must be alias, not URL
        if _is_url(_first_arg(ctx)):
            hints.append(
                Hint(
                    title="RequestsLibrary: Alias first, then relative path",
                    message=(
                        "Use session alias first and a relative path, or use 'Get' with a full URL."
                    ),
                    examples=[
                        {
                            "tool": "execute_step",
                            "keyword": "Create Session",
                            "arguments": ["rb", "https://restful-booker.herokuapp.com"],
                        },
                        {
                            "tool": "execute_step",
                            "keyword": "Get On Session",
                            "arguments": ["rb", "/booking/1"],
                            "use_context": True,
                        },
                        {
                            "tool": "execute_step",
                            "keyword": "Get",
                            "arguments": [
                                "https://restful-booker.herokuapp.com/booking/1"
                            ],
                        },
                    ],
                    relevance=75,
                )
            )
    if kw_lower == "get" and args and isinstance(args[0], str) and args[0].startswith("/"):
        hints.append(
            Hint(
                title="RequestsLibrary: Use Create Session + Get On Session",
                message=(
                    "For relative paths, create a session and use 'Get On Session'; otherwise pass a full URL to 'Get'."
                ),
                examples=[
                    {
                        "tool": "execute_step",
                        "keyword": "Create Session",
                        "arguments": ["rb", "https://restful-booker.herokuapp.com"],
                    },
                    {
                        "tool": "execute_step",
                        "keyword": "Get On Session",
                        "arguments": ["rb", "/booking"],
                        "use_context": True,
                    },
                ],
                relevance=70,
            )
        )
    if "session less" in kw_lower:
        hints.append(
            Hint(
                title="RequestsLibrary: Use 'Get' or 'Get On Session'",
                message=(
                    "Use 'Get' with a full URL or 'Get On Session' with an alias and relative path."
                ),
                examples=[
                    {"tool": "execute_step", "keyword": "Get", "arguments": [
                        "https://restful-booker.herokuapp.com/booking/1"
                    ]},
                    {"tool": "execute_step", "keyword": "Get On Session", "arguments": [
                        "rb", "/booking/1"
                    ], "use_context": True},
                ],
                relevance=68,
            )
        )

    # 4) Named args guidance for dicts
    dict_like = any(isinstance(a, str) and "=" in a and ("{" in a or "[" in a) for a in args)
    if kw_lower.endswith(" on session") and dict_like:
        hints.append(
            Hint(
                title="RequestsLibrary: Pass named args as 'name=value'",
                message=(
                    "Pass named args as strings like params=..., headers=...; Python literals are supported inside."
                ),
                examples=[
                    {
                        "tool": "execute_step",
                        "keyword": "Get On Session",
                        "arguments": [
                            "rb",
                            "/booking",
                            "params={'checkin':'2014-01-01','checkout':'2014-02-01'}",
                        ],
                        "use_context": True,
                    }
                ],
                relevance=60,
            )
        )

    # 5) RequestsLibrary POST/PUT/PATCH payload/headers guidance on 400/415
    if kw_lower in {"post", "put", "patch", "post on session", "put on session", "patch on session"}:
        err_low = err.lower()
        if any(code in err_low for code in ["httperror: 400", "bad request", "415", "unsupported media", "unsupported media type"]):
            # Detect json= passed as a quoted string that likely isn't parsed into a dict
            has_json_arg = any(isinstance(a, str) and a.strip().lower().startswith("json=") for a in args)
            json_looks_quoted = any(
                isinstance(a, str)
                and a.strip().lower().startswith("json=")
                and ("{" in a or "[" in a)
            for a in args)

            examples: List[Dict[str, Any]] = []
            # Option 1: Use json= with a real Python dict variable
            examples.append(
                {
                    "tool": "execute_step",
                    "keyword": "Evaluate",
                    "arguments": [
                        "{'firstname':'Jim','lastname':'Brown','totalprice':111,'depositpaid':True, 'bookingdates':{'checkin':'2018-01-01','checkout':'2019-01-01'}, 'additionalneeds':'Breakfast'}"
                    ],
                    "assign_to": "booking",
                    "use_context": True,
                }
            )
            examples.append(
                {
                    "tool": "execute_step",
                    "keyword": "POST" if " on session" not in kw_lower else "Post On Session",
                    "arguments": (
                        [
                            args[0],
                            "json=${booking}",
                        ]
                        if " on session" not in kw_lower
                        else [
                            _first_arg(ctx) if not _is_url(_first_arg(ctx)) else "rb",
                            args[1] if len(args) > 1 else "/booking",
                            "json=${booking}",
                        ]
                    ),
                    "use_context": True,
                }
            )
            # Option 1b: Sessionful pattern (Create Session + Post On Session)
            base = _base_url(args[0]) if args else None
            if base:
                examples.append(
                    {
                        "tool": "execute_step",
                        "keyword": "Create Session",
                        "arguments": ["rb", base],
                    }
                )
                examples.append(
                    {
                        "tool": "execute_step",
                        "keyword": "Post On Session",
                        "arguments": ["rb", "/booking", "json=${booking}"],
                        "use_context": True,
                    }
                )
            # Option 2: Use data= JSON string and proper headers
            examples.append(
                {
                    "tool": "execute_step",
                    "keyword": "Create Dictionary",
                    "arguments": ["Content-Type", "application/json", "Accept", "application/json"],
                    "assign_to": "headers",
                    "use_context": True,
                }
            )
            examples.append(
                {
                    "tool": "execute_step",
                    "keyword": "POST" if " on session" not in kw_lower else "Post On Session",
                    "arguments": (
                        [
                            args[0],
                            "data={\"firstname\":\"Jim\",\"lastname\":\"Brown\",\"totalprice\":111,\"depositpaid\":true,\"bookingdates\":{\"checkin\":\"2018-01-01\",\"checkout\":\"2019-01-01\"},\"additionalneeds\":\"Breakfast\"}",
                            "headers=${headers}",
                        ]
                        if " on session" not in kw_lower
                        else [
                            _first_arg(ctx) if not _is_url(_first_arg(ctx)) else "rb",
                            args[1] if len(args) > 1 else "/booking",
                            "data={\"firstname\":\"Jim\",\"lastname\":\"Brown\",\"totalprice\":111,\"depositpaid\":true,\"bookingdates\":{\"checkin\":\"2018-01-01\",\"checkout\":\"2019-01-01\"},\"additionalneeds\":\"Breakfast\"}",
                            "headers=${headers}",
                        ]
                    ),
                    "use_context": True,
                }
            )
            # Option 2b: Sessionful POST with data= and headers
            if base:
                examples.append(
                    {
                        "tool": "execute_step",
                        "keyword": "Create Session",
                        "arguments": ["rb", base],
                    }
                )
                examples.append(
                    {
                        "tool": "execute_step",
                        "keyword": "Post On Session",
                        "arguments": [
                            "rb",
                            "/booking",
                            "data={\"firstname\":\"Jim\",\"lastname\":\"Brown\",\"totalprice\":111,\"depositpaid\":true,\"bookingdates\":{\"checkin\":\"2018-01-01\",\"checkout\":\"2019-01-01\"},\"additionalneeds\":\"Breakfast\"}",
                            "headers=${headers}",
                        ],
                        "use_context": True,
                    }
                )

            hints.append(
                Hint(
                    title="RequestsLibrary: POST/PUT payload guidance",
                    message=(
                        "400/415 errors often indicate payload/headers issues. "
                        "CRITICAL: Always use NAMED arguments (json=, headers=, data=) - positional args cause misalignment! "
                        "Either: (1) build a nested Python dict and pass via json=${body}, or "
                        "(2) pass a JSON string via data= and include headers= with Content-Type."
                    ),
                    examples=examples,
                    relevance=85,  # Increased relevance
                )
            )

    # 5b) Headers passed as string instead of dictionary
    # This happens when models try to pass headers as a string like "Content-Type=application/json"
    if "'str' object has no attribute 'items'" in err or (
        "expected argument" in err.lower() and "dictionary" in err.lower() and "got" in err.lower()
    ):
        hints.append(
            Hint(
                title="RequestsLibrary: Headers must be a DICTIONARY, not a string",
                message=(
                    "Headers must be passed as a Robot Framework dictionary variable, NOT a string. "
                    "Use Create Dictionary to build headers, then reference with ${headers}. "
                    "The same applies to JSON body - use Create Dictionary, not a string representation."
                ),
                examples=[
                    {
                        "comment": "CORRECT: Create headers as dictionary",
                        "tool": "execute_step",
                        "keyword": "Create Dictionary",
                        "arguments": [
                            "Content-Type", "application/json",
                            "Accept", "application/json",
                        ],
                        "assign_to": "headers",
                    },
                    {
                        "tool": "execute_step",
                        "keyword": "POST On Session",
                        "arguments": [
                            "alias",
                            "/endpoint",
                            "json=${body}",
                            "headers=${headers}",  # Variable reference, not string!
                        ],
                    },
                    {
                        "comment": "WRONG: Don't pass headers as string",
                        "incorrect": "headers=Content-Type=application/json",
                        "note": "This creates a string, not a dict!",
                    },
                ],
                relevance=95,
            )
        )

    # 5c) UnknownStatusError - Headers/dict passed as positional argument to wrong parameter
    # This happens when models don't use named arguments for POST On Session
    if "unknownstatuserror" in err.lower() or (
        "invalid literal for int" in err.lower() and ("{" in err or "content-type" in err.lower())
    ):
        hints.append(
            Hint(
                title="RequestsLibrary: Use NAMED arguments for POST On Session",
                message=(
                    "UnknownStatusError indicates arguments were passed in wrong positions. "
                    "POST On Session has many optional parameters (data, json, params, headers, expected_status). "
                    "ALWAYS use named arguments (name=value) to avoid positional mismatches. "
                    "The signature is: POST On Session(alias, url, data=, json=, params=, headers=, expected_status=)"
                ),
                examples=[
                    {
                        "comment": "CORRECT: All arguments are named",
                        "tool": "execute_step",
                        "keyword": "POST On Session",
                        "arguments": [
                            "alias_name",
                            "/endpoint",
                            "json=${body}",
                            "headers=${headers}",
                        ],
                        "note": "Use json= and headers= named arguments",
                    },
                    {
                        "comment": "WRONG: Positional arguments cause misalignment",
                        "incorrect": "POST On Session    alias    /url    ${body}    ${headers}",
                        "note": "Without names, headers goes to wrong param!",
                    },
                    {
                        "comment": "Build body and headers separately first",
                        "tool": "execute_step",
                        "keyword": "Create Dictionary",
                        "arguments": [
                            "firstname", "Jim",
                            "lastname", "Brown",
                            "totalprice", "111",
                        ],
                        "assign_to": "body",
                    },
                    {
                        "tool": "execute_step",
                        "keyword": "Create Dictionary",
                        "arguments": [
                            "Content-Type", "application/json",
                            "Accept", "application/json",
                        ],
                        "assign_to": "headers",
                    },
                    {
                        "tool": "execute_step",
                        "keyword": "POST On Session",
                        "arguments": [
                            "rb",
                            "/booking",
                            "json=${body}",
                            "headers=${headers}",
                        ],
                        "assign_to": "response",
                    },
                ],
                relevance=95,  # High relevance - this is a critical error pattern
            )
        )

    # 6) Keyword ambiguity - Multiple keywords with same name found
    if "multiple keywords with name" in err.lower():
        # Extract the keyword name and the library options from the error
        ambig_match = re.search(r"multiple keywords with name '([^']+)' found", err.lower())
        kw_name = ambig_match.group(1) if ambig_match else ctx.keyword

        # Extract library options if present
        lib_options = []
        for lib in ["Browser", "SeleniumLibrary", "BuiltIn", "Collections", "String", "RequestsLibrary", "AppiumLibrary"]:
            if f"{lib.lower()}.{kw_name.lower()}" in err.lower() or f"{lib}." in err:
                lib_options.append(lib)

        hints.append(
            Hint(
                title="Keyword Ambiguity: Specify full library.keyword name",
                message=(
                    f"Multiple libraries provide '{kw_name}'. Use the full 'Library.Keyword' syntax to resolve, "
                    "or use set_library_search_order tool to set precedence."
                ),
                examples=[
                    {
                        "comment": f"Option 1: Use full keyword name",
                        "tool": "execute_step",
                        "keyword": f"{lib_options[0]}.{kw_name.title().replace(' ', ' ')}" if lib_options else f"Browser.{kw_name.title()}",
                        "arguments": list(args),
                    },
                    {
                        "comment": "Option 2: Set library search order",
                        "tool": "set_library_search_order",
                        "arguments": {"libraries": lib_options or ["Browser", "BuiltIn"]},
                    },
                ],
                relevance=95,
            )
        )

    # 6b) Dictionary access via dot notation - 'dict' object has no attribute
    # Models often try ${dict.key} instead of ${dict}[key]
    if "'dict' object has no attribute" in err or "dict object has no attribute" in err.lower():
        # Extract the attempted attribute name
        attr_match = re.search(r"'dict' object has no attribute '([^']+)'", err)
        attr_name = attr_match.group(1) if attr_match else "key"
        hints.append(
            Hint(
                title="Dictionary Access: Use bracket notation, not dot notation",
                message=(
                    f"Robot Framework dictionaries use bracket notation for access. "
                    f"Use ${{dict}}['{attr_name}'] or ${{dict.{attr_name}}} (for response objects), not ${{dict.{attr_name}}}. "
                    "For API responses, use ${response.json()['key']} to access JSON data."
                ),
                examples=[
                    {
                        "correct": "${booking}['firstname']",
                        "incorrect": "${booking.firstname}",
                        "note": "Use brackets for dict access",
                    },
                    {
                        "correct": "${response.json()['booking']['firstname']}",
                        "incorrect": "${response.json.booking.firstname}",
                        "note": "Call json() method, then bracket notation",
                    },
                    {
                        "tool": "execute_step",
                        "keyword": "Should Be Equal",
                        "arguments": ["${response.json()['firstname']}", "Jim"],
                    },
                ],
                relevance=92,
            )
        )

    # 7) Method call syntax error - 'method' object is not subscriptable OR got method instead of dict
    if ("'method' object is not subscriptable" in err or
        "method object is not subscriptable" in err.lower() or
        "got method instead" in err.lower()):
        hints.append(
            Hint(
                title="Method Call Syntax: Call methods with parentheses",
                message=(
                    "In Robot Framework, methods must be called with parentheses. "
                    "Use ${response.json()} instead of ${response.json}. "
                    "For indexing: ${response.json()['key']} not ${response.json['key']}."
                ),
                examples=[
                    {
                        "correct": "${response.json()}",
                        "incorrect": "${response.json}",
                        "note": "Methods need () to be called",
                    },
                    {
                        "correct": "${response.json()['booking']}",
                        "incorrect": "${response.json['booking']}",
                        "note": "Call method first, then index",
                    },
                    {
                        "tool": "execute_step",
                        "keyword": "Should Be Equal As Strings",
                        "arguments": ["${response.json()['firstname']}", "Jim"],
                    },
                ],
                relevance=90,
            )
        )

    # 8) No keyword found - provide better suggestions
    if "no keyword with name" in err.lower():
        # Extract the keyword name attempted
        kw_not_found_match = re.search(r"no keyword with name '([^']+)'", err.lower())
        attempted_kw = kw_not_found_match.group(1) if kw_not_found_match else kw_lower

        # Common keyword corrections
        keyword_corrections = {
            "create list": ("BuiltIn.Create List", "Use BuiltIn.Create List, not Collections.Create List"),
            "collections.create list": ("BuiltIn.Create List", "Create List is in BuiltIn, not Collections"),
            "get library search order": ("set_library_search_order tool", "There is no 'Get Library Search Order' keyword - use set_library_search_order tool instead"),
            "press button": ("Click Button or Click", "Use 'Click Button' for SeleniumLibrary or 'Click' for Browser Library"),
            "verify": ("Should Be Equal / Should Contain", "RF uses 'Should' keywords for assertions, not 'Verify'"),
            "validate json": ("Should Be Equal or Dictionary Should Contain", "Use 'Should' keywords or dictionary/list keywords for validation"),
            "wait until page contains": ("Wait Until Page Contains Element", "Use 'Wait Until Page Contains Element' or 'Wait For Elements State' (Browser)"),
        }

        correction = keyword_corrections.get(attempted_kw.lower())
        if correction:
            hints.append(
                Hint(
                    title=f"Keyword Correction: {correction[0]}",
                    message=correction[1],
                    examples=[
                        {
                            "tool": "find_keywords",
                            "arguments": {"pattern": attempted_kw.split(".")[-1]},
                            "note": "Search for the correct keyword",
                        },
                    ],
                    relevance=88,
                )
            )
        else:
            # Generic suggestion to use find_keywords
            hints.append(
                Hint(
                    title="Keyword Not Found: Use find_keywords to discover",
                    message=(
                        f"Keyword '{attempted_kw}' was not found. Use find_keywords tool to search "
                        "for available keywords. Common issues: wrong library prefix, keyword doesn't exist, "
                        "or library not imported."
                    ),
                    examples=[
                        {
                            "tool": "find_keywords",
                            "arguments": {"pattern": attempted_kw.split(".")[-1] if "." in attempted_kw else attempted_kw},
                        },
                        {
                            "tool": "get_keyword_info",
                            "arguments": {"keyword": attempted_kw.split(".")[-1] if "." in attempted_kw else attempted_kw},
                        },
                    ],
                    relevance=85,
                )
            )

    # Element interaction errors (library-aware)
    hints.extend(_check_element_interaction_errors(ctx))

    # Transform to serializable list of dicts, limited to top 3 by relevance
    hints_sorted = sorted(hints, key=lambda h: h.relevance, reverse=True)[:3]
    return [
        {
            "title": h.title,
            "message": h.message,
            "examples": h.examples,
        }
        for h in hints_sorted
    ]
