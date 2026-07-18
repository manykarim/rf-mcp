"""Proactive RequestsLibrary request/response cookbook (change: api-cookbook).

The single source of the RequestsLibrary guidance recipes. Exposed proactively
through ``get_locator_guidance(library="requests")`` (the web analog agents
already pull before interacting) and referenced by the reactive on-failure hints
in ``utils/hints.py`` so proactive and reactive guidance cannot drift.

Motivation (F-API1): the restful-booker eval (2026-07-17) achieved 6/6 CRUD
assertions but hit max-turns before writing artifacts — burned by 178 ``Evaluate``
calls + 43 ``resp.json()`` rediscovering RequestsLibrary response access one
failed step at a time. This cookbook gives the patterns UP FRONT.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ── Shared canonical recipe texts (single source; also used by hints.py) ──────

# The Evaluate variable-syntax rule — the single most-cited RequestsLibrary
# gotcha (the ${resp.json()} vs $resp tarpit).
EVALUATE_VAR_RULE = (
    "Inside Evaluate use the BARE name ($resp, $token); OUTSIDE Evaluate use "
    "${resp.json()}. Do NOT reach for Evaluate to compare status/fields — use "
    "the native assertion keywords."
)

ON_SESSION_RULE = (
    "Create Session with an alias + base URL, then use GET/POST/PUT/PATCH/DELETE "
    "On Session with that alias and a RELATIVE path. A bare Get/Post takes a "
    "full URL."
)

JSON_BODY_RULE = (
    "Send a JSON body with json=<dict> — NOT data=. Set headers with "
    "headers=${headers}. A 400/415 usually means data= was used, the body was a "
    "string, or Content-Type: application/json is missing."
)

# How to CONSTRUCT the body dict — the step weak models thrash on. Two ways:
JSON_BODY_BUILD_RULE = (
    "Build the JSON body dict, do NOT pass a repr-string. Easiest: inline Python "
    'eval — json=${{ {"firstname": "Jane", "totalprice": 111, "depositpaid": True, '
    '"bookingdates": {"checkin": "2024-01-01"}} }} (a real Python literal, so True/'
    "False/None and nested dicts work). Or define it first and pass it: assign "
    '${body}= via Evaluate  {"firstname": "Jane", ...}  THEN reference json=${body} '
    "in the POST — never POST before ${body} exists. Avoid Set Variable/Create "
    "Dictionary gymnastics for JSON bodies."
)

NAMED_ARGS_RULE = (
    "RequestsLibrary named args are positional name=value tokens "
    "(expected_status=200, json=${body}, headers=${headers}), not RF &{dict} kwargs."
)


def build_requests_cookbook(
    error_message: Optional[str] = None, keyword_name: Optional[str] = None
) -> Dict[str, Any]:
    """Return the RequestsLibrary cookbook payload (tips/warnings/examples).

    Mirrors the shape of ``get_browser_locator_guidance``. The full cookbook is
    always returned; ``error_message``/``keyword_name`` only influence ordering
    (surfacing the most-relevant recipe first).
    """
    tips = [
        f"SESSION: {ON_SESSION_RULE}",
        "RESPONSE: On-Session keywords RETURN the response — capture with "
        "${resp}=. Read status via ${resp.status_code}, body via ${resp.json()}, "
        'a field via ${resp.json()["bookingid"]}.',
        f"EVALUATE: {EVALUATE_VAR_RULE}",
        "STATUS ASSERT: use 'Status Should Be    200    ${resp}' — NOT an "
        "Evaluate equality on ${resp.status_code}. This is the biggest "
        "Evaluate-call remover.",
        f"BODY/HEADERS: {JSON_BODY_RULE}",
        f"BODY CONSTRUCTION: {JSON_BODY_BUILD_RULE}",
        "AUTH TOKEN: capture the token from ${resp.json()[\"token\"]}, then send "
        'it as a cookie header: headers=${{"Cookie": "token=" + $token}} '
        "(or a &{headers} dict). Required for PUT/DELETE on restful-booker.",
        "NON-2XX: pass expected_status=404 (or expected_status=anything) so an "
        "error response is RETURNED instead of raising — needed to assert a "
        "deleted resource returns 404.",
        f"NAMED ARGS: {NAMED_ARGS_RULE}",
    ]
    warnings = [
        "${resp.json()} works OUTSIDE Evaluate; inside Evaluate it is $resp.json() "
        "(bare) — mixing them is the #1 API time sink.",
        "Without expected_status=, a non-2xx response RAISES — you cannot then "
        "assert a 404; pass expected_status to capture error responses.",
        "PUT/DELETE that return 403/Forbidden usually mean the auth token cookie "
        "header was not sent.",
    ]
    examples = [
        {"keyword": "Create Session", "arguments": ["rb", "https://restful-booker.herokuapp.com"],
         "note": "alias + base URL"},
        {"keyword": "POST On Session",
         "arguments": ["rb", "/booking",
                       'json=${{ {"firstname": "Jane", "lastname": "Doe", "totalprice": 111, "depositpaid": True, "bookingdates": {"checkin": "2024-01-01", "checkout": "2024-01-05"}} }}'],
         "assign_to": "resp", "note": "build the JSON body inline with ${{ }} — no Set Variable/Create Dictionary"},
        {"keyword": "POST On Session", "arguments": ["rb", "/auth", "json=${creds}"],
         "assign_to": "resp", "note": "or define ${creds}= first, then reference it (never POST before it exists)"},
        {"keyword": "Status Should Be", "arguments": ["200", "${resp}"],
         "note": "native status assertion (not Evaluate)"},
        {"keyword": "Set Variable", "arguments": ['${resp.json()["token"]}'],
         "assign_to": "token", "note": "read a JSON field outside Evaluate"},
        {"keyword": "GET On Session", "arguments": ["rb", "/booking/${id}", "expected_status=404"],
         "assign_to": "resp", "note": "assert a deleted resource → 404 without raising"},
        {"keyword": "DELETE On Session",
         "arguments": ["rb", "/booking/${id}", 'headers=${{"Cookie": "token=" + $token}}'],
         "assign_to": "resp", "note": "auth token cookie header"},
    ]

    payload: Dict[str, Any] = {"tips": tips, "warnings": warnings, "examples": examples}

    # Ordering nudge: surface the most-relevant recipe first when we know the error.
    err = (error_message or "").lower()
    if err:
        if "expected_status" in err or "40" in err or "raise" in err:
            payload["most_relevant"] = "NON-2XX / expected_status="
        elif "evaluate" in err or "not defined" in err or "${" in (error_message or ""):
            payload["most_relevant"] = "EVALUATE variable syntax"
        elif "on session" in err or "url" in err or "session" in err:
            payload["most_relevant"] = "SESSION setup"

    return payload
