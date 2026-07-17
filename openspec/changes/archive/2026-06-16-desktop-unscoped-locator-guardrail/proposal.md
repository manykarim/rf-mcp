# Proposal: desktop-unscoped-locator-guardrail

## Why

An independent Kilo/MiniMax session (tests/libreoffice/REPORT_FOR_RF_MCP_MAINTAINER.md, 2026-06-16) reported the rf-mcp MCP server "dies unrecoverably" during a LibreOffice test and that "LibreOffice's Paragraph does not expose Focusable." Reproducing both on the isolated `:100` display traced them to **one** cause and corrected the diagnoses:

- The AT-SPI tree is on the D-Bus **session** bus, not the X display — even an isolated-display process sees all 16 desktop apps (gnome-shell, Chrome×2, VS Code, …).
- A leading `//` in an XPath is **absolute** — it ignores `Set Root`/context and re-walks the whole session tree.
- Measured: unscoped `//control:Paragraph` → 83 desktop-wide nodes in **36.9 s**; app-scoped `/app:*[@Name='soffice']//control:Paragraph` → the **1** real document node in **2.8 s**.

So the "death" is **not a server crash** (the engine completed the 36.9 s walk and returned) — it is the unscoped walk exceeding the MCP client's ~30 s request timeout, after which the client closes stdio and the server dies on the broken pipe; the client then never respawns (client-side). And the "Paragraph not Focusable" was the unscoped `only_first` grabbing a *foreign app's* paragraph — the real LibreOffice paragraph **does** expose Focusable.

rf-mcp's `get_locator_guidance` already documents the rule verbatim ("NEVER start a locator with //; ALWAYS scope to an application"), and the report author acknowledged reading it — yet used `//*` and `//control:Paragraph` anyway. The documentation is correct; the gap is that it is **advisory, not enforced**: a desktop `Query`/`Evaluate` with no wall-clock bound turns a known-bad locator into a transport death instead of a fast, actionable error.

## What Changes

- A pre-flight guardrail in the desktop keyword path: when a desktop session runs `Query` / `Evaluate` whose XPath argument is **unscoped** (starts with `//` or `descendant-or-self::`, i.e. not anchored at `/app:`, not relative, not a count/aggregate-only expression) and no escape flag is set, the call is **refused before dispatch** with a structured error that names the offending locator, restates the performance rule, and offers a concrete app-scoped rewrite (`/app:*[@Name='<app>']//…`).
- An escape hatch for the rare deliberate desktop-wide search: env `ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED=1` (and a per-session attribute), which downgrades the refusal to a one-time warning.
- The refusal aligns with the user's own non-negotiable rule ("fail loudly; never silently fall back to the whole desktop").

Out of scope (captured as notes, not built here): a wall-clock watchdog around the native call (can't interrupt a synchronous C call cleanly); the Wayland `Take Screenshot` fallback (upstream PlatynUI provider gap); the upstream `Query only_first` TypeError (PlatynUI keyword binding, not rf-mcp); the report's suggested `Keyboard Type` auto-fallback to the AUT root (contradicts the user's fail-loud rule — the focusable target exists via app-scoped addressing, so no fallback is warranted).

## Capabilities

### New Capabilities

- `desktop-unscoped-locator-guardrail`: desktop `Query`/`Evaluate` with an unscoped (`//`-rooted) locator is refused pre-flight with an actionable, guidance-citing error, unless explicitly allowed.

### Modified Capabilities

(none — additive pre-flight gate; the existing `get_locator_guidance` content is unchanged and is what the error cites)

## Impact

- `src/robotmcp/components/execution/keyword_executor.py` — new `_unscoped_locator_guard(session, keyword, arguments)` returning an error dict, wired into the desktop pre-flight block beside `_screenshot_path_guard` (~line 1726).
- `src/robotmcp/components/execution/desktop_execution_signals.py` (or a small helper) — `is_query_keyword` / unscoped-xpath detection (leading `//`, `descendant-or-self::`, bare `*`), with carve-outs for `/app:`-anchored, relative, and pure-`count()` expressions.
- `src/robotmcp/models/session_models.py` — `platynui_allow_unscoped: bool` per-session opt-out.
- Tests: `tests/unit/test_unscoped_locator_guardrail.py` (refuses `//control:Paragraph`; allows `/app:*…`, relative, `count(//…)`; honors env + session escape; non-desktop untouched); baseline (6817 passed + 1 skipped) stays green.
