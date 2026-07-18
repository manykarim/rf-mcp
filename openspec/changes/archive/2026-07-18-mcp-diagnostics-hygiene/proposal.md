## Why

A docker experiment round (Claude Code CLI + MiniMax M3/M2.5 driving a
`uv tool`-installed rf-mcp, robotmcp stderr captured — see
`experiments/uv-tool-install/FINDINGS_ROUND2.md`) confirmed the server works and
the stdio channel is clean, but stderr forensics surfaced a set of avoidable
problems. Two were genuine bugs, already fixed this session; the rest are logging
noise, one self-inflicted inefficiency, and a guidance gap that measurably costs
weaker models. This change lands those follow-ups.

The dominant issue is **logging hygiene**: expected, correctly-handled failures
(a 404, an unknown keyword, an unresolved variable) each dump a full multi-frame
Python traceback at ERROR level (30 in one run alone), and normal lazy-context
bootstrap steps log as WARNING/"failed" even though they immediately recover.
This buries the rare real signal and makes rf-mcp look broken when it is not.

Two functional items: a `get_session_state` on an API/Requests session cascades
through 7 browser DOM keywords (7 tracebacks of wasted work) because the
page-source service is not gated on whether a web library is loaded; and the
requests cookbook is silent on constructing a JSON request body — exactly where
both models struggled (M2.5 burned ~14 calls, M3 slipped on `${body}` ordering).

## What Changes

**Logging hygiene**
- Expected, agent-driven keyword-execution failures log a single-line
  ERROR/WARNING summary; the full traceback moves to DEBUG.
- Lazy RF-context bootstrap messages ("No active RF execution context",
  "Failed to register … in RF context", "BuiltIn library import failed during
  context creation") are downgraded to DEBUG / reworded as recoverable attempts.
- Optional-library-not-installed fallbacks log at DEBUG/INFO, not WARNING.
- The keyword-shadowing notice is de-duplicated (was logged twice per load).

**Correctness / ergonomics**
- The page-source service short-circuits when no web library is loaded (API /
  desktop sessions), so state inspection stops cascading browser DOM keywords.
- The return-value assignment heuristic recognizes RequestsLibrary response
  keywords (`GET/POST/PUT/DELETE/PATCH On Session`) so it stops false-warning on
  correct captures.
- Session init aliases the common mistake `Requests` → `RequestsLibrary` and logs
  the correction at WARNING rather than an ERROR-level import failure.
- The requests guidance cookbook gains a JSON-request-body pattern
  (`json=${{ {...} }}` inline eval + define-body-before-POST ordering).

**Already landed (from the same round; included for the record)**
- Fixed `Dialogs`/`STDLIBS` frozenset-subscript `TypeError`.
- Fixed `import_library(notify=True)` breaking on RF 7.4 (kwarg removed upstream).

Non-goals: no change to the MCP tool surface or execution semantics; not
suppressing genuine errors — only relabeling expected/recovered paths and
trimming duplicate/verbose output. FastMCP-owned output (its startup banner, rich
traceback panels, PyPI update check) is out of scope except where an env toggle
already exists.

## Capabilities

### New Capabilities
- `mcp-diagnostics-hygiene`: how rf-mcp reports diagnostics — log levels for
  expected vs unexpected failures and lazy-bootstrap paths, page-source gating by
  loaded libraries, the assignment-heuristic allowlist, the RequestsLibrary alias,
  and the requests JSON-body guidance.

### Modified Capabilities
<!-- none -->
