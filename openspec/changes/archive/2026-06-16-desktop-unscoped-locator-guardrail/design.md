# Design: desktop-unscoped-locator-guardrail

## Context

Reproduced facts (engine driven directly on isolated `:100`, soffice launched):

| Locator | Nodes | Time |
|---|---|---|
| `//control:Paragraph` (unscoped) | 83 (desktop-wide) | 36.9 s |
| `/app:*[@Name='soffice']//control:Paragraph` (scoped) | 1 (real doc body) | 2.8 s |
| scoped paragraph `has_pattern(Focusable)` | True | — |

The desktop pre-flight block in `_execute_keyword_serialized` already chains `_platynui_safety_guard` → `_screenshot_path_guard` → `_platynui_focus_before_act`, each returning a structured error dict to refuse before dispatch. `_inject_timeout_into_arguments` early-returns for desktop sessions, so desktop `Query`/`Evaluate` run with no wall-clock bound; the native `evaluate` runs via `asyncio.to_thread` (uninterruptible C call). The only safe lever is **pre-flight refusal**, not mid-flight cancellation.

## Goals / Non-Goals

**Goals:** convert a known-bad unscoped desktop locator into an immediate, actionable error (citing the existing guidance + a rewrite) instead of a 30 s+ transport death; keep the cheap discovery patterns (`/app:*`, `count(//…)`) working; provide an explicit opt-out for deliberate desktop-wide searches.

**Non-Goals:** interrupting an in-flight native walk; capping response size; web/mobile sessions (gate on `is_desktop_session`); the screenshot/Wayland and upstream-keyword items noted in the proposal.

## Decisions

### D1 — Pre-flight refusal, beside the existing guards
Add `_unscoped_locator_guard(session, keyword, arguments) -> Optional[dict]`; wire into the desktop pre-flight block right after `_screenshot_path_guard`. Returns the same error-dict shape (`success: False`, `error`, `hints`) so the call never reaches the native layer. Only runs for desktop sessions (strict `is_desktop_session() is True`).

### D2 — What counts as "unscoped" (the detection)
Applies to keywords whose first string argument is an XPath: `Query`, `Evaluate` (basename-matched). Strip leading whitespace; the locator is **unscoped** when it:
- starts with `//` or `descendant-or-self::`, OR
- is a bare wildcard walk (`*`, `//*`).

It is **allowed** (not unscoped) when it:
- starts with `/app:` (anchored to an application) or `/` other than `//` (absolute from desktop root but explicit),
- is **relative** (starts with `control:`, `item:`, `.`, `(`, an axis like `child::`),
- is a **pure aggregate** that returns a scalar, not a node set — detected by an outer `count(` / `string(` / `number(` / `boolean(` wrapper (the report's recommended `count(//…)` discovery stays allowed; counting is the sanctioned way to size a subtree before scoping).
*Alternative rejected*: blocking `count(//…)` too — it is exactly the discovery step the guidance recommends, and returns one number, not a giant node set.
*Risk*: a deeply nested `count(//a///b)` still walks the tree; accepted — counting is bounded by AT-SPI's own per-node timeout and returns a scalar, so it cannot trigger the large-serialization transport death.

### D3 — The refusal message reuses existing guidance + a concrete rewrite
The error hint restates the `performance_rules` line ("NEVER start with //; scope to /app:*[@Name='X']//…") and, when an app can be inferred (a single `/app:*` already resolved this session, or the session's launched AUT name), suggests the exact rewrite `/app:*[@Name='<app>']//<rest>`. Hint type `unscoped_desktop_locator`. Matches the existing fail-loud + hint convention (screenshot guard, I-4 process arg).

### D4 — Escape hatch
`ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED=1` (env) or `session.platynui_allow_unscoped` downgrades refusal to a single per-session warning (`desktop_unscoped_warned` one-shot, mirroring `desktop_wayland_warned`). Covers the genuine "search the whole desktop" case without removing the guardrail for everyone.

### D5 — Maintainer corrections recorded
The proposal context states the two diagnosis corrections (Issue 1 = client timeout, not server crash; Issue 3 = the real LO paragraph is Focusable, the unscoped query grabbed a foreign one) so the change is understood as scoping-enforcement, not focus-fallback.

## Risks / Trade-offs

- [False refusal of a legitimate `//` a user truly wants] → the escape hatch (D4) + the rewrite suggestion keep it unblocking; the default protects the 99% case.
- [Detection misclassifies an exotic XPath] → conservative: only the clear `//`/`descendant-or-self`/bare-`*` prefixes refuse; anything ambiguous is allowed (fail-open toward running).
- [`count(//…)` still walks the tree] → accepted; returns a scalar, cannot cause the large-response death, and is the documented discovery step.

## Migration Plan

Additive pre-flight gate; revert to roll back. Baseline 6817 passed + 1 skipped stays green.

## Open Questions

- Should the guard also fire on `Set Root //…` (an unscoped root that then makes every relative query desktop-wide)? Leaning yes — same detection on the `Set Root` argument — but `Set Root` is lower-traffic; could ship Query/Evaluate first and add Set Root if it recurs.
