# Session strategy schema redesign — OBS-23B

**Story**: OBS-23B (session strategy schema unification, design phase)
**Predecessor**: OBS-23 (single story, split per round-2 Codex review)
**Sibling**: OBS-23A (honour query+limit, backward-compat; ships
independently)
**Status**: **v2** — revised 2026-05-18 after Codex CLI + Claude
sub-agent adversarial review. v1 had: false externalisation claim
(legacy fields stayed top-level), incomplete reader survey,
dishonest `confidence=1.00`, unpinned Phase 2/3 versions, redundant
fields. All addressed below; see "Review findings + resolutions"
section at end.
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

## Current readers — survey (v2 — expanded after Codex + Claude review)

Pre-design v1 missed three active readers. v2 reader survey via
`grep -rn '"library_keywords"\|"resource_keywords"' src/ tests/ scripts/`:

| Path | Type | Reads via | v2 impact |
|---|---|---|---|
| `rf_native_context_manager.py:1641-1686` | **Producer** | n/a (writes) | Unchanged — lower-level API still emits legacy shape. |
| `src/robotmcp/components/test_builder.py:1855` | **Internal reader** | Calls `mgr.list_available_keywords(session_id)` DIRECTLY (NOT via MCP). Reads `library_keywords` to build keyword→library map for test suite generation. | Unchanged — bypasses the MCP wrapper, so OBS-23B's wrapper-only changes don't affect it. |
| `tests/e2e/test_openai_fastmcp.py:294` | **External reader** | MCP `find_keywords(strategy="session")` + `keywords.data.get("library_keywords", [])`. Production-facing pattern. | Updated in OBS-23B impl to read `result.matches`; legacy `library_keywords` still readable as a transitional check. |
| `tests/benchmarks/test_robustness_token_overhead.py:124` | **Active benchmark fixture** (Codex caught) | Benchmark harness that synthesises a response shape carrying `library_keywords`. Used in token-overhead regression tests. | Update: when OBS-23B impl lands, benchmark fixture is updated to mirror the unified shape. Old fixture archived as v1-shape for migration regression test. |
| `scripts/benchmark_discovery_tools.py` | **Live benchmark consumer** (Claude caught) | Reads any field present in `find_keywords` responses. Currently records `library_keywords` count for session strategy. | Update: counter logic uses `result.matches` first, falls back to legacy field for pre-OBS-23B baseline comparison. |
| `tests/e2e/metrics/autonomous/*.json` (8 files) | Historical records | Static JSON dumps; non-active | Unaffected. |

**Key insight**: only ONE external MCP-tool reader
(`test_openai_fastmcp.py:294`) consumes the legacy shape directly.
The other readers either bypass the MCP layer (`test_builder.py`)
or are tooling-side and can be updated in-place. The migration path:

1. Change the MCP wrapper (`find_keywords` session branch in
   `server.py`) to emit the unified shape with legacy fields still
   present (Phase 1).
2. Update `test_openai_fastmcp.py:294` + benchmark scripts to
   consume the unified shape.
3. Leave `test_builder.py`'s direct call to
   `list_available_keywords` unchanged — the lower-level method
   keeps its current return shape.

Phase 1 backward-compat is preserved through dual-emit. The
`test_robustness_token_overhead.py:124` benchmark fixture is the
one place that has to migrate immediately; the others can stay on
legacy reads through the deprecation window.

---

## Proposed unified shape (v2 — corrected after review)

```python
# After OBS-23B fully migrates (Phase 1 dual-emit shape):
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
        # v2 fix: match_type replaces the v1 dishonest
        # confidence=1.00 field. "exact_substring" tells the
        # agent how this match was found (no ranking; literal
        # name match against the query).
        "match_type": "exact_substring",  # "exact_substring" | "exact"
        "source_type": "library",  # "library" | "resource"
        "source": None,  # path-string when type=="resource"
      },
      {
        "keyword_name": "Click Element",
        "library": "SeleniumLibrary",
        "full_name": "SeleniumLibrary.Click Element",
        "match_type": "exact_substring",
        "source_type": "library",
        "source": None,
      },
      # ... up to limit (post-OBS-23A trim)
    ],
    # v2: dropped match_count (redundant with len(matches)).
    # library_count and resource_count document POST-TRIM values.
    "library_count": 2,   # distinct libraries in matches (post-trim)
    "resource_count": 0,  # distinct resources in matches (post-trim)
    # Pre-trim totals surface separately for diagnostic clarity.
    "total_before_trim": 47,  # what `list_available_keywords` returned
    "recommendations": [
      # NB: no fake confidence number — v1's "(confidence: 1.00)"
      # was misleading. Session strategy doesn't rank; the
      # recommendation prose says exact-match vs alternative.
      "Exact match: Click (Browser)",
      "Alternative options: Click Element, Click Button, Tap",
    ],
  },
  # Legacy fields — preserved during deprecation window (Phase 1).
  # These mirror result.matches split by source_type.
  "library_keywords": [...],   # mirror of matches where source_type=="library"
  "resource_keywords": [...],  # mirror of matches where source_type=="resource"
  "libraries_count": 2,        # legacy alias for library_count
}
```

### Why `recommendations` for session strategy

The session strategy doesn't compute semantic similarity — it's a
namespace listing. But callers benefit from a deterministic "best
match for this query" line. Decision: when `query` is non-empty,
`recommendations[0]` calls out the **exact name match** first if one
exists; otherwise the alphabetically-first match. The prose shape
matches semantic/pattern strategies (recommendation strings, no
ranking) but with honest wording about how the match was found.

Pseudocode (v2 — fixed):

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
    # v2 fix: honest prose, no fake confidence number.
    if exact:
        recs = [f"Exact match: {top['keyword_name']} ({top['library']})"]
    else:
        recs = [
            f"Substring match: {top['keyword_name']} "
            f"({top['library']}); no exact name match"
        ]
    # Required-arguments line is INTENTIONALLY OMITTED — session
    # strategy doesn't have per-keyword arg info from
    # list_available_keywords. Agents fetch full info via
    # get_keyword_info(mode="keyword", keyword_name=..., session_id=...)
    # if needed.
    if len(matches) > 1:
        alt_names = [m["keyword_name"] for m in matches[1:4]]
        recs.append(f"Alternative options: {', '.join(alt_names)}")
    return recs
```

### Externalisation impact (v2 — CORRECTED)

**v1 design was WRONG here.** It claimed the existing
`field_path="result"` rule would cover the unified shape, but Phase
1 also dual-emits legacy `library_keywords`/`resource_keywords`/
`libraries_count` at the TOP level — outside `result`. Those
top-level fields would leak inline even when `result` externalised.
Codex round-1 caught this; v2 fixes via explicit additional rules.

**v2 externalisation rules** (added to `DEFAULT_RULES` in
OBS-23B impl):

```python
# Existing — covers the unified shape's result.matches etc.
ExternalizationRule(tool_name="find_keywords", field_path="result"),

# OBS-23B Phase 1 — explicitly cover legacy mirror fields so the
# whole response can shrink during dual-emit. Drop when Phase 3
# removes the fields entirely.
ExternalizationRule(tool_name="find_keywords",
                    field_path="library_keywords"),
ExternalizationRule(tool_name="find_keywords",
                    field_path="resource_keywords"),
```

`libraries_count` is a tiny int — no rule needed.

With these rules, a Browser+BuiltIn session with ~350 keywords:
- `result.matches` (post-trim, post-OBS-23A) → externalises when
  cumulative size > 500 tokens.
- `library_keywords` (pre-trim or post-trim mirror — see Phase 1
  decision below) → externalises independently when > 500 tokens.
- `resource_keywords` → same.

OBS-27 (Externalise pattern `results` + session payload fields)
becomes partially redundant for the session half after OBS-23B
ships — the rules added here pre-empt the OBS-27 session-coverage.
OBS-27 still needs the pattern coverage (different code path).

### Phase 1 decision: legacy fields are pre-trim or post-trim?

**Decision: post-trim** (matches OBS-23A behaviour).

`list_available_keywords` after OBS-23A applies the `query` filter +
`limit` trim internally. Phase 1's dual-emit reuses whatever the
lower-level method returns. Result: `library_keywords` and
`resource_keywords` in the v2 unified-emit response are the
**filtered, trimmed** lists — same as before OBS-23B for any
caller that uses query+limit (the post-OBS-23A behaviour).

**Backward-compat note**: a pre-OBS-23A caller that read
`library_keywords` to get the FULL namespace would experience
behaviour change at OBS-23A (filter applied), not at OBS-23B (this
design). OBS-23A documents that change in its own AC.

`total_before_trim` in `result` surfaces the pre-trim count so the
agent can detect a filter-induced truncation.

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

### Phase 2 — deprecation warning (target: v0.34 release)

Emit the legacy fields PLUS a server-side WARNING log per session
when an agent reads them. Detection mechanism: instrument the
`_track_tool_result` hook (ADR-014) to record which response fields
the caller subsequently accesses; emit a one-time warning when
`library_keywords` / `resource_keywords` access is detected for a
session.

Operator-side opt-out: `ROBOTMCP_SESSION_LEGACY_FIELDS=false` env
var lets operators verify their agents work on the unified shape
only (legacy fields suppressed in response) before Phase 3 lands.

**Filed as**: OBS-34 (new follow-up story; placeholder for now).
**Target version**: v0.34 (1-2 minor versions after OBS-23B v1
ships in v0.33).

### Phase 3 — remove legacy fields (target: v0.36 release)

Drop `library_keywords`, `resource_keywords`, `libraries_count` from
the response unconditionally. Phase 2 warnings give operators a
window to migrate. Phase 3 is a breaking change behind a major-minor
boundary (v0.33 dual-emit → v0.34 warning → v0.36 removal).

**Filed as**: OBS-35 (new follow-up story; placeholder).
**Target version**: v0.36 (next major-minor after Phase 2).

### Phase rollout decision

OBS-23B implementation story implements **Phase 1 only**. Phases
2 + 3 are tracked as separate stories (OBS-34, OBS-35) per
**review feedback** — v1 said "filed as OBS-Future, not yet
numbered" which both Codex and Claude flagged as exactly the
pattern that leaves legacy fields in place forever.

**Phase versions are normative in this design**. If v0.33 ships
without OBS-23B Phase 1, the versions shift consistently (Phase 2
in the version after Phase 1 ships, etc.). The point is that the
phases are committed to specific releases, not "someday".

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
5. Update `tests/benchmarks/test_robustness_token_overhead.py:124`
   fixture (v2 addition — Codex caught this missed reader).
6. Update `scripts/benchmark_discovery_tools.py` counter logic
   (v2 addition — Claude caught this).
7. Add 8 new unit tests covering the unified shape + back-compat +
   externalisation of both unified and legacy fields.
8. Add the legacy-field externalisation rules to `DEFAULT_RULES`
   (v2 addition — fixes the AC #4 gap).
9. Update story doc + benchmark report with Phase 1 verification.

---

## Cross-design contract — shared with OBS-18A reranker

Both OBS-18A (reranker) and OBS-23B (session schema) touch the
`find_keywords` response shape. The outer envelope must be
consistent across all strategies for agents to write strategy-
agnostic consumers. v2 explicitly aligns the two designs:

**Common outer envelope** (all four strategies post-Wave 3):

```json
{
  "success": bool,
  "strategy": "semantic" | "pattern" | "catalog" | "session",
  "query": str,
  "result": {
    "matches": List[Dict],
    "recommendations": List[str],
    // Strategy-specific fields under `result`:
    // semantic: total_matches, filtered_count, action_type
    // session: library_count, resource_count, total_before_trim
    // catalog: top_matches, catalog_truncated, full_catalog_size
    // pattern: (still uses top-level results — see follow-up
    //          OBS-Future-pattern-unification)
  },
  // Diagnostic top-level fields (any strategy can emit):
  "library_filter": {...},        // when filter applied
  "excluded_alternatives": [...], // when filter applied
  "low_confidence_top_match": bool, // OBS-18A — semantic strategy only
  // Phase 1 legacy (session only, deprecated v0.34, removed v0.36):
  "library_keywords": [...], "resource_keywords": [...], "libraries_count": int,
}
```

**`low_confidence_top_match` ownership**: OBS-18A semantic strategy
only. Other strategies don't compute confidence (pattern is exact
substring; catalog is unfiltered; session is exact substring) so
this field is meaningless there.

**`recommendations` prose convention** (shared across strategies):
- First line names the top match (or no-matches prose).
- Subsequent lines may surface alternatives, required arguments
  (when available — semantic only), filter diagnostics.
- Tone is declarative, no fake confidence numbers (v2 fix —
  session strategy no longer claims "confidence: 1.00").

If OBS-18A and OBS-23B both ship before pattern is unified, the
"common envelope" claim is **3-of-4 strategies**, not all-four. The
section above is explicit about pattern remaining on its
top-level-`results` shape. Pattern unification is filed as
OBS-Future-pattern-unification follow-up — same migration template
as OBS-23B.

---

## Review findings + resolutions (Codex round-1 + Claude round-1)

Both reviewers ran adversarial review against v1 of this design.
Convergent findings → v2 fixes:

| Finding | Source | v2 resolution |
|---|---|---|
| Externalisation claim WRONG — Phase 1 keeps legacy fields at top level so the existing `result` rule doesn't cover them | BOTH | Added explicit `library_keywords` + `resource_keywords` rules to `DEFAULT_RULES`. Implementation task #8 added. |
| Reader survey missed `tests/benchmarks/test_robustness_token_overhead.py:124` | Codex | v2 survey table includes it; impl task #5 updates the fixture. |
| Reader survey missed `scripts/benchmark_discovery_tools.py` | Claude | v2 survey table includes it; impl task #6 updates the counter logic. |
| `confidence=1.00` is dishonest (session strategy doesn't rank) | BOTH | Replaced with `match_type: "exact_substring"` field. Recommendation prose uses "Exact match: X" / "Substring match: X" — no fake confidence. |
| `match_count` redundant with `len(matches)` | Codex | Dropped from the unified shape. |
| `library_count` / `resource_count` semantics ambiguous (pre-trim vs post-trim) | Codex | Documented as POST-TRIM. Added `total_before_trim` for pre-trim diagnostic. |
| Phase 2/3 hand-wavy ("OBS-Future, not numbered") | BOTH | Filed as OBS-34 (Phase 2, target v0.34) and OBS-35 (Phase 3, target v0.36) with normative version targets. |
| Cross-strategy claim overstated (pattern is still different) | BOTH | "Common outer envelope" section explicitly says 3-of-4 strategies; pattern is documented as follow-up. |
| OBS-23A interaction: legacy `library_keywords` field gets pre-filtered too | Claude | Documented as Phase 1 backward-compat note: "a pre-OBS-23A caller that read library_keywords to get the FULL namespace would experience behaviour change at OBS-23A (filter applied), not at OBS-23B". |
| Self-citing "round-2 Codex review" without source link | Both | Renamed to "Codex CLI round-2 review" (the transcript was the codex exec output captured in conversation; should link to a docs/reviews/ artifact in OBS-23B impl). |
| `low_confidence_top_match` (OBS-18A) ↔ session recommendations interaction undocumented | Claude | Cross-design contract section added; field is semantic-strategy-only. |

Non-convergent findings:
- Env var vs tool parameter for legacy-field suppression: Claude
  suggested `legacy_shape: bool` tool param instead of env var.
  **Decision: keep env var.** Tool parameter would require every
  caller to pass it; env var is the operator-side opt-out the
  design wants for Phase 2 verification.
- Bundle pattern unification into OBS-23B: Claude asked. **Decision:
  keep separate.** Pattern has more readers (used more often than
  session) and would more than double the OBS-23B scope.
