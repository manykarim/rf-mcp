# Session strategy schema redesign — OBS-23B

**Story**: OBS-23B (session strategy schema unification, design phase)
**Predecessor**: OBS-23 (single story, split per round-2 Codex review)
**Sibling**: OBS-23A (honour query+limit, backward-compat; ships
independently)
**Status**: proposed (this is the deliverable for OBS-23B)
**Date**: 2026-05-18

## Problem

`find_keywords(strategy="session")` returns a payload shape that does
not match the other three strategies (semantic / pattern / catalog).
The schema mismatch has three concrete consequences:

1. **Schema sprawl**: agents that handle session strategy must
   special-case its response. The other three return
   `{success, strategy, query, result: {matches: [...]}, ...}`.
   Session returns `{success, libraries_count, library_keywords,
   resource_keywords, ...}` — different top-level keys, different
   nesting, different list naming.
2. **Externalisation gap**: the existing
   `ExternalizationRule(tool_name="find_keywords", field_path="result")`
   does NOT cover `library_keywords` / `resource_keywords` (they're
   top-level siblings of `result`, not under it). Large session
   namespaces (Browser + BuiltIn + SeleniumLibrary together ≈ 480
   keywords) leak inline even WITH `session_id`.
3. **No recommendations**: agents lose the
   `recommendations: ["Best match: X (confidence: Y)", ...]` prose
   that semantic/pattern provide. The session strategy returns a raw
   list and the agent has to derive its own "what should I use" from
   the dump.

Round-2 Codex story-review verdict:

> "OBS-23 WRONG-SCOPE — bundles query handling, limit handling,
> response-shape unification, and externalization. AC #3 left the
> backward-compat decision ambiguous ('may stay as siblings OR be
> merged'), which blocks a fresh engineer from making a contract
> decision."

OBS-23A handled the query/limit honouring (backward-compat). OBS-23B
is the schema redesign with full backward-compat treatment.

## Goal

Unify the session-strategy response shape with semantic / pattern /
catalog so:

- The top-level shape is `{success, strategy, query, result: {...}, ...}` —
  identical across all four strategies.
- Large session payloads externalise via the existing
  `field_path="result"` rule (no new rule needed).
- Recommendations surface for the session strategy (top match,
  alternatives — same prose as semantic).
- Existing callers reading the legacy shape (`library_keywords` /
  `resource_keywords` siblings) keep working through a deprecation
  window.

Non-goals (deferred):
- Schema unification for `manage_session` namespace operations
  (separate concern).
- Glob/regex support in the session-strategy `query` parameter
  (OBS-23A handles substring; pattern strategy already handles
  glob).

---

## Current readers — survey

Pre-design, I surveyed every reader of `library_keywords` /
`resource_keywords` across the project:

| Path | Type | Notes |
|---|---|---|
| `rf_native_context_manager.py:1641-1686` | **Producer** | Builds the dict returned by `list_available_keywords`. Currently the source of truth. |
| `src/robotmcp/components/test_builder.py:1855` | **Internal reader** | Calls `mgr.list_available_keywords(session_id)` directly (NOT via the MCP `find_keywords` tool). Reads `library_keywords` to build a keyword→library map for the test suite generator. Bypasses MCP entirely. |
| `tests/e2e/test_openai_fastmcp.py:294` | **External reader** | Calls the MCP `find_keywords(strategy="session")` and reads `keywords.data.get("library_keywords", [])`. Production-facing — represents how external agents consume the response. |
| `tests/e2e/metrics/autonomous/*.json` | Historical records | Recorded responses from past autonomous test runs. Won't break (these are static JSON files, not active readers). |

**Key insight**: only ONE external reader (the e2e test) consumes the
legacy shape via the MCP tool. The internal reader
(`test_builder.py:1855`) calls the lower-level
`list_available_keywords` directly and is unaffected by changes to
the MCP wrapper. So the migration path can be:

1. Change the MCP wrapper (`find_keywords` session branch in
   server.py) to emit the unified shape with optional legacy
   fields.
2. Update `test_openai_fastmcp.py` to consume the unified shape.
3. Leave `test_builder.py`'s direct call to
   `list_available_keywords` unchanged — it still gets the legacy
   shape from the lower-level method.

This is a much smaller change than rewriting
`list_available_keywords` itself.

---

## Proposed unified shape

```python
# After OBS-23B fully migrates:
{
  "success": True,
  "strategy": "session",
  "query": "click",
  "result": {
    "matches": [
      {
        "keyword_name": "Click",
        "library": "Browser",
        "full_name": "Browser.Click",
        # Optional fields when available (no doc/confidence in
        # session-strategy responses; matches are name-only lookups).
        "source_type": "library",  # "library" | "resource"
        "source": None,  # path-string when type=="resource"
      },
      {
        "keyword_name": "Click Element",
        "library": "SeleniumLibrary",
        "full_name": "SeleniumLibrary.Click Element",
        "source_type": "library",
        "source": None,
      },
      # ... up to limit (post-OBS-23A trim)
    ],
    "match_count": 12,  # actual returned count after trim
    "library_count": 3,  # distinct libraries with matches
    "resource_count": 0,  # distinct resources with matches
    "recommendations": [
      "Best match: Click (confidence: 1.00)",
      "Required arguments: selector, button",
      "Alternative options: Click Element, Click Button, Tap",
    ],
  },
  # Legacy fields — preserved during deprecation window (see migration).
  "library_keywords": [...],   # mirror of matches where source_type=="library"
  "resource_keywords": [...],  # mirror of matches where source_type=="resource"
  "libraries_count": 3,        # legacy alias for library_count
}
```

### Why `recommendations` for session strategy

The session strategy doesn't compute semantic similarity — it's a
namespace listing. But callers benefit from a deterministic "best
match for this query" line. Decision: when `query` is non-empty,
`recommendations[0]` names the *exact name match* first if one
exists; otherwise the alphabetically-first match. This is cheap
(no scoring) and matches the existing semantic-recommendation prose
shape for consistent agent UX.

Pseudocode:

```python
def session_recommendations(matches: List[dict], query: str) -> List[str]:
    if not matches:
        return [
            "No keywords match the query in this session.",
            "- Verify the query string",
            "- Check the session's imported libraries",
        ]
    # Prefer exact name match if present.
    q_lower = query.lower().strip() if query else ""
    exact = next(
        (m for m in matches if m["keyword_name"].lower() == q_lower),
        None,
    )
    top = exact or matches[0]
    recs = [f"Best match: {top['keyword_name']} (confidence: 1.00)"]
    # Required arguments are unavailable in session-strategy responses
    # (the manager returns name + library only). Skip the args line —
    # agents fetch full info via get_keyword_info if needed.
    if len(matches) > 1:
        alt_names = [m["keyword_name"] for m in matches[1:4]]
        recs.append(f"Alternative options: {', '.join(alt_names)}")
    return recs
```

Confidence is `1.00` because session-strategy matches are exact
substring hits, not ranked candidates. The fixed value tells the
agent "this is a literal lookup, not a ranked guess" — useful
signal.

### Externalisation impact

With the unified shape, the existing rule

```python
ExternalizationRule(tool_name="find_keywords", field_path="result")
```

automatically covers the session strategy too. No new field-path
rule needed. Large session namespaces (Browser + BuiltIn ≈ 350+
keywords) will externalise when the serialised `result` dict exceeds
the 500-token threshold (default).

This is why OBS-27 ("Externalise pattern `results` + session payload
fields") becomes partially redundant after OBS-23B: the session half
is automatically covered. OBS-27 still needs to add the
`field_path="results"` rule for pattern strategy.

---

## Migration path

Three-phase rollout to preserve backwards compat:

### Phase 1 — dual-emit (this PR / OBS-23B implementation)

The session branch emits BOTH the unified shape AND the legacy
shape:

```python
# server.py find_keywords session branch (post-OBS-23B):
payload = mgr.list_available_keywords(session_id, ...)
# Build unified matches list
unified_matches = []
for kw in (payload.get("library_keywords") or []):
    unified_matches.append({
        "keyword_name": kw["name"],
        "library": kw.get("library"),
        "full_name": kw.get("full_name", kw["name"]),
        "source_type": "library",
        "source": None,
    })
for kw in (payload.get("resource_keywords") or []):
    unified_matches.append({
        "keyword_name": kw["name"],
        "library": None,
        "full_name": kw.get("full_name", kw["name"]),
        "source_type": "resource",
        "source": kw.get("resource"),
    })
# Apply OBS-23A query+limit trim
unified_matches = apply_name_filter_and_limit(
    unified_matches, query=query, limit=limit_value
)
result = {
    "success": payload.get("success", True),
    "strategy": "session",
    "query": query,
    "result": {
        "matches": unified_matches,
        "match_count": len(unified_matches),
        "library_count": len({m["library"] for m in unified_matches
                              if m["library"]}),
        "resource_count": len({m["source"] for m in unified_matches
                               if m["source"]}),
        "recommendations": session_recommendations(unified_matches, query),
    },
    # Legacy fields — preserved during the deprecation window.
    "library_keywords": payload.get("library_keywords") or [],
    "resource_keywords": payload.get("resource_keywords") or [],
    "libraries_count": payload.get("libraries_count", 0),
}
# ADR-015: externalize large result fields
if session_id:
    result = _externalize_response("find_keywords", session_id, result)
```

Phase 1 emits:
- New: `result.matches`, `result.match_count`, `result.library_count`,
  `result.resource_count`, `result.recommendations`.
- Legacy preserved: `library_keywords`, `resource_keywords`,
  `libraries_count` (top-level).

Phase 1 ships in OBS-23B implementation. The `test_openai_fastmcp.py`
test is updated to read `result.matches` instead of
`library_keywords` to validate the new shape end-to-end. Other
clients (currently none confirmed) keep working via the legacy
fields.

### Phase 2 — deprecation warning (1-2 release cycles after Phase 1)

Emit the legacy fields with a one-time warning log per session
indicating they'll be removed. Optional env var
`ROBOTMCP_SESSION_LEGACY_FIELDS=false` lets operators verify their
agents work on the unified shape only before Phase 3.

### Phase 3 — remove legacy fields (post-deprecation, follow-up story)

Drop `library_keywords`, `resource_keywords`, `libraries_count` from
the response. Filed as a separate story (OBS-Future, not yet
numbered). Phase 3 is NOT part of OBS-23B — only the migration plan
documentation is.

### Phase rollout decision

OBS-23B implements **Phase 1 only**. Phases 2 + 3 are documented
here for completeness but tracked as future stories. This keeps
OBS-23B small enough to land as a single PR while preserving the
upgrade path.

---

## API contract — unified shape per strategy

Post-OBS-23B + Phase 1, all four strategies emit a common outer
shape:

```python
{
  "success": bool,
  "strategy": "semantic" | "pattern" | "catalog" | "session",
  "query": str,
  "result": {
    "matches": List[Dict],  # strategy-specific entry shape
    "recommendations": List[str],  # always present, strategy-specific prose
    # Other strategy-specific fields under `result`:
    # - semantic: total_matches, filtered_count, action_type, action_description
    # - pattern: (uses top-level results; see below for full alignment notes)
    # - catalog: top_matches, catalog_truncated, full_catalog_size
    # - session: match_count, library_count, resource_count
  },
  # Strategy-specific top-level fields (mostly diagnostic):
  "library_filter": {...},        # all strategies when filter applied
  "excluded_alternatives": [...], # all strategies when filter applied
}
```

### Pattern strategy contract drift (NOT in scope for OBS-23B)

The pattern branch currently puts `results` at the top level
(`server.py:~2417`):

```python
result = {
    "success": True,
    "strategy": "pattern",
    "query": query,
    "match_count": len(matches),
    "top_matches": top_names,
    "results": matches,  # ← top-level, not under result
}
```

That's inconsistent with the unified shape. Fixing it would change
the pattern API surface — out of scope for OBS-23B. Filed as
follow-up:

> **OBS-Future-pattern-unification**: align pattern strategy with
> the unified `result.matches` shape. Same Phase 1/2/3 migration
> plan. Filed independently because the pattern strategy has more
> readers (used by `find_keywords` agent-facing workflow more often
> than session).

OBS-23B is intentionally session-only. After it lands, the next
follow-up can align pattern using the same migration template.

---

## Test plan (OBS-23B implementation will build)

### Unit tests

1. **Unified shape**: synthetic `list_available_keywords` payload
   with both `library_keywords` and `resource_keywords` →
   `result.matches` contains both, classified by `source_type`.
2. **Field-path coverage**: `result.match_count` equals
   `len(result.matches)` after filter/limit applied.
   `result.library_count` and `result.resource_count` match the
   distinct counts.
3. **Backward compat**: legacy `library_keywords`,
   `resource_keywords`, `libraries_count` still present at top level.
4. **Recommendations**: query that has an exact match → top
   recommendation names that keyword.  Empty query →
   alphabetically-first match wins.  No matches → "No keywords match"
   prose.
5. **Empty namespace**: session with no imported libraries →
   `result.matches: []`, recommendations carry the no-matches prose.
6. **Externalisation**: large session payload (synthetic 500
   keywords) with `session_id` → `result` field externalises via
   the existing rule.

### Integration tests

7. **e2e test update**: `tests/e2e/test_openai_fastmcp.py:294`
   migrated to read `result.matches` from the unified shape. New
   assertion: the legacy `library_keywords` field is still present
   (Phase 1 backward compat).
8. **MCP wire-format test**: real MCP call against a real Browser
   session, assert both new and legacy fields shape correctly.

---

## Open questions for OBS-23B reviewer

1. **Confidence value for session matches**: I chose `1.00` because
   they're exact substring matches. Alternative: omit the
   confidence field entirely (no scoring happened). **Recommendation**:
   keep `1.00` for shape consistency with semantic — agents
   uniformly read `confidence` across strategies. The fixed value
   tells the agent "no ranking, take at face value".

2. **`source_type` enum values**: `"library"` vs `"resource"`.
   Alternatives considered: `"library_kw"`, `"namespace"`. The chosen
   names align with the existing producer dict's field names. Keep
   simple — agents already understand the library/resource
   distinction.

3. **`full_name` field**: included because `rf_native_context_manager.py:1654`
   already populates it. Alternative: drop and let agents
   concatenate `library` + `keyword_name` themselves. **Recommendation**:
   keep — it's free (already computed) and saves the agent string
   manipulation.

4. **Phase 2/3 timing**: should the deprecation warning + removal be
   scheduled (e.g., "Phase 2 in v0.34, Phase 3 in v0.36"), or
   driven by adoption (e.g., when external metrics show < 5% legacy
   reads)? **Recommendation**: schedule, because adoption metrics
   are hard to gather without telemetry. Suggested cadence: Phase 2
   in v0.34 (1-2 minor versions after OBS-23B), Phase 3 in v0.36 or
   later (after one major-version migration cycle).

---

## Out of scope (explicitly)

- Pattern strategy contract unification (filed as
  OBS-Future-pattern-unification follow-up).
- Catalog strategy contract unification (would require lifting
  `match_count` + `top_matches` + `results` under `result`; same
  migration template).
- Changes to `list_available_keywords` internal API (test_builder.py
  reader stays on the legacy lower-level interface).
- Glob support in session query (OBS-23A handles substring; deferred).
- Per-keyword doc retrieval in session strategy responses (agents
  use `get_keyword_info(mode="session")` for that — separate tool).

---

## Acceptance criteria recap (from the story)

- [x] **Investigation deliverable** — Survey of current readers
      (3 internal + 1 external + N historical records).
- [x] **Proposed unified shape** — `result.matches` schema +
      strategy-consistent outer envelope.
- [x] **Migration path** — Three-phase rollout with Phase 1 in
      scope for OBS-23B, Phases 2-3 documented for follow-up.
- [x] **Externalisation impact** — existing `field_path="result"`
      rule covers the unified shape; no new rule needed.

OBS-23B implementation story tasks (post-design):
1. Update `server.py` session branch to emit unified shape +
   preserve legacy fields.
2. Add `session_recommendations()` helper.
3. Wire OBS-23A's query+limit trim into the unified path.
4. Update `tests/e2e/test_openai_fastmcp.py:294` to read unified
   shape.
5. Add 6 new unit tests covering the unified shape + back-compat.
6. Update story doc + benchmark report with Phase 1 verification.
