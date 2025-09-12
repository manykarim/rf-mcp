"""Context-aware hint generation for failed tool calls.

Produces short, actionable guidance with concrete examples based on the
keyword, arguments, error text, and (optionally) session metadata.
"""

from __future__ import annotations

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
                        "400/415 errors often indicate payload/headers issues. Either: "
                        "(1) build a nested Python dict and pass via json=, or (2) pass a JSON string via data= and include headers with Content-Type and Accept."
                    ),
                    examples=examples,
                    relevance=72,
                )
            )

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
