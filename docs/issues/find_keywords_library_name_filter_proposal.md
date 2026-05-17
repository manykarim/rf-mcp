# find_keywords ignores `library_name` for semantic/pattern strategies

**Status**: IMPLEMENTED on `feature/obstacle-course-followup` (2026-05-17) — see "Implementation outcome" at bottom of this file
**Defect reproduced**: 2026-05-17 against pre-fix code
**Affected tool**: `find_keywords` (MCP tool)
**Affected strategies**: `semantic` (alias: `intent`), `pattern` (alias: `search`)
**Unaffected strategies**: `catalog` (alias: `library`), `session`

## Summary

`find_keywords(strategy="semantic", library_name="Browser", ...)` returns
keywords from libraries other than Browser (notably SeleniumLibrary), and
silently — no `excluded_keywords` field, no `session_library` indicator,
no warning. The `library_name` parameter is documented as "Optional
library filter for catalog search" and is in fact only wired into the
catalog branch of `find_keywords`. For semantic and pattern strategies,
the parameter is **silently accepted and ignored**.

## Reproduction

Exact user payload:

```json
{
  "query": "select dropdown option by label or text",
  "strategy": "semantic",
  "context": "web",
  "library_name": "Browser",
  "limit": 10
}
```

Reproduced via in-process call (no MCP roundtrip needed):

```bash
uv run python -c "
import asyncio
from robotmcp.server import find_keywords
async def main():
    fn = getattr(find_keywords, 'fn', find_keywords)
    result = await fn(
        query='select dropdown option by label or text',
        strategy='semantic',
        context='web',
        library_name='Browser',
        limit=10,
    )
    matches = result.get('result', {}).get('matches', [])
    by_library = {}
    for m in matches:
        by_library.setdefault(m.get('library'), []).append(m.get('keyword_name'))
    for lib, kws in by_library.items():
        print(f'  {lib}: {kws}')
    print('excluded_keywords present:', 'excluded_keywords' in result)
asyncio.run(main())
"
```

Output:

```
SeleniumLibrary: ['Set Window Position', 'Click Button', 'Click Element',
                  'Click Link', 'Select From List By Label',
                  'Set Selenium Page Load Timeout', 'Set Browser Implicit Wait']
Browser:         ['Click', 'Tap', 'Select Options By']
excluded_keywords present: False
session_library present: False
```

7 of the top-10 matches are from SeleniumLibrary despite `library_name="Browser"`.
"Set Window Position" — a SeleniumLibrary keyword unrelated to dropdown
selection — appears as the top match (confidence 0.82).

## Root cause

`src/robotmcp/server.py:2107+` (`find_keywords` tool implementation).

The semantic branch (lines 2195-2238) and pattern branch (lines 2240-2288)
both ignore the `library_name` parameter. Filtering is applied *only* via
the session-preference path:

```python
# server.py:2186-2193
session_library_preference = None
if session_id:
    session = execution_engine.session_manager.get_session(session_id)
    if session:
        session_library_preference = getattr(
            session, "explicit_library_preference", None
        )
```

The filter (`_filter_keywords_by_session_library`, server.py:2037) is then
invoked only if BOTH `session_id` was passed AND the session has a
non-None `explicit_library_preference`. The `library_name` parameter is
never consulted.

Only the catalog branch honors `library_name` (server.py:2290-2292):

```python
if strategy_norm in {"catalog", "library"}:
    await _ensure_all_session_libraries_loaded()
    catalog = execution_engine.get_available_keywords(library_name)
```

The docstring confirms the gap explicitly:

```
library_name: Optional library filter for catalog search.
```

…but this is buried in the docstring under a parameter with a generic
name that strongly suggests a uniform filter.

## Why this is the wrong default

Three reasons it matters:

1. **Silent failure mode**. The parameter is accepted, the request returns
   200, the response carries no indication the filter was a no-op.
   Agents (especially small LLMs that don't read docstrings closely)
   reasonably interpret the response as "Browser library doesn't have a
   `Select Dropdown` keyword, so I'll use `Select From List By Label`
   from SeleniumLibrary." A later `execute_step` against a Browser-only
   session then fails with "Unknown keyword" — at runtime, not at
   discovery time.

2. **Top-match contamination**. "Set Window Position" — a SeleniumLibrary
   keyword with no semantic relation to "select dropdown option by label
   or text" — is the top match at confidence 0.82. This is a separate
   semantic-matcher quality issue, but the library filter would mask it
   entirely from a Browser-only session. The Browser-correct answer
   (`Select Options By`) is the 7th match.

3. **Inconsistent with the catalog branch**. The same parameter name on
   the same tool means two different things depending on `strategy`.
   This is the surface-area equivalent of P11's `NO_LOCATOR_KEYWORDS`
   inconsistency: a curated list that drifts out of alignment with the
   actual library semantics. Predictable API surface > clever overloads.

## Existing filter infrastructure (working, just under-used)

`_filter_keywords_by_session_library` already does the right thing for
the session-preference path:

- Consults `LibraryPluginManager.get_incompatible_libraries(preference)`.
  For `Browser`, this returns `["SeleniumLibrary"]`
  (browser_plugin.py:294-296). For `SeleniumLibrary`, it returns
  `["Browser"]` (selenium_plugin.py:183).
- Builds an `excluded_with_alternatives` list with helpful messages
  (alternative keyword + example) using `KEYWORD_ALTERNATIVES` from the
  plugin.
- Logs the filter action and emits the count.

The pieces are all present. What's missing is plumbing.

## Proposed fix

### Design

Honor `library_name` in **all four** strategies as an explicit per-call
filter, semantically equivalent to `explicit_library_preference` on the
session but scoped to the single call.

Precedence (highest wins):

1. **Explicit `library_name` parameter** (per-call override)
2. **Session `explicit_library_preference`** (session-wide default,
   already in place)
3. **No filter** (current behavior when neither is set)

When `library_name` is set:
- Apply the same `LibraryPluginManager.get_incompatible_libraries`
  exclusion table that the session-preference path uses.
- For `library_name="Browser"`, this excludes `SeleniumLibrary` and
  passes through `BuiltIn`, `Collections`, `String`, etc.
- Result includes `excluded_keywords` and `session_library` fields
  (renamed to `requested_library` for the explicit-param case, OR
  keep `session_library` populated with the effective preference and
  add a `library_filter_source: "library_name" | "session"` field for
  clarity).

### Implementation sketch

`server.py:2186-2193` becomes:

```python
# Resolve effective library preference: explicit param > session > none.
effective_library_preference: str | None = library_name
library_filter_source: str = "library_name" if library_name else ""

if not effective_library_preference and session_id:
    session = execution_engine.session_manager.get_session(session_id)
    if session:
        session_pref = getattr(session, "explicit_library_preference", None)
        if session_pref:
            effective_library_preference = session_pref
            library_filter_source = "session"
```

Then in each strategy branch, swap `session_library_preference` for
`effective_library_preference`:

```python
if effective_library_preference and discovery.get("matches"):
    # ... existing filter logic, parametrized on effective_library_preference
```

And in the response payload:

```python
if excluded:
    result["excluded_keywords"] = excluded
    result["library_filter"] = {
        "applied": effective_library_preference,
        "source": library_filter_source,  # "library_name" or "session"
    }
```

Catalog branch keeps its current behavior (already uses `library_name`
for the engine call, not for post-filter). Two-line cleanup: also feed
`library_name` through the session-preference filter so the catalog
result list is consistent with the catalog-engine result list. Low
priority — the catalog engine already scopes the search by library.

### Docstring update

```
library_name: Optional library filter applied to ALL strategies.
  When set, restricts results to the named library and its compatible
  siblings (e.g., library_name="Browser" excludes SeleniumLibrary but
  keeps BuiltIn, Collections, String). Takes precedence over the
  session's explicit_library_preference when both are present.
  Catalog strategy additionally scopes the underlying lookup to this
  library.
```

### Tests to add

| Layer | Test | Expected |
|---|---|---|
| Reproducer | semantic + `library_name="Browser"`, no session | SeleniumLibrary keywords absent; `excluded_keywords` populated; `library_filter.source="library_name"` |
| Symmetry | semantic + `library_name="SeleniumLibrary"`, no session | Browser keywords absent |
| Strategy parity | pattern + `library_name="Browser"` | same filter applied |
| Precedence | semantic + `library_name="Browser"` AND session preference `"SeleniumLibrary"` | `library_name` wins, `library_filter.source="library_name"` |
| Fall-through | semantic, no `library_name`, session preference `"Browser"` | session preference wins, `library_filter.source="session"` |
| Idempotency | semantic, no `library_name`, no session | unchanged behavior — no filter, no `library_filter` field |
| Unknown lib | semantic + `library_name="FakeLibrary"` | no exclusion (plugin returns empty incompatibility list), `excluded_keywords` empty/absent, `library_filter` may include the requested name for response-trace clarity |
| Compatible siblings preserved | semantic + `library_name="Browser"` matching "log message" | `BuiltIn.Log` still in results |

Existing tests likely affected (smoke-test these after the change):

- Anywhere semantic search is exercised against a Browser-only session
  without `library_name` being passed (should be unchanged).
- Tests that mock `_filter_keywords_by_session_library` directly (may
  need to mock the new resolver too — keep the resolver thin).

### Out of scope

- **Semantic-matcher confidence quality** for "select dropdown option by
  label or text" (the "Set Window Position" top-match issue). That's a
  separate finding worth a follow-up — the library filter masks it but
  doesn't fix it. The actionable scope here is filtering, not re-ranking.
- **`KeywordMatcher.discover_keywords()` library-aware search**: the
  current proposal filters post-discovery. A future optimization could
  pass the filter into the matcher so it doesn't compute scores for
  excluded keywords. Worth measuring before doing — the post-filter is
  O(n) on a small n.
- **`_extract_locator_from_args` skip-prefix scoping** (`link=` skipped
  for all libraries, not just SeleniumLibrary — found during OBS-15).
  Same root pattern (per-library config bleeding across libraries) but
  separate fix.

### Risk

Low. The filter function is well-isolated and already covered by tests
via the session-preference path. The resolver is a small precedence
function. The response shape gains optional fields (`library_filter`,
`excluded_keywords` when applicable); no field is removed or renamed.

Conservative migration: keep the existing `session_library` response
field populated whenever filtering applies (preserves backward
compat for callers that read it), and add `library_filter` as the
forward-looking shape.

### Estimated effort

S — single function rewrite (resolver) + four call-site touchups in
the two affected branches + ~8 unit tests + docstring update. Half-day.

## Related artifacts

- `src/robotmcp/server.py:2107` — `find_keywords` tool
- `src/robotmcp/server.py:2037` — `_filter_keywords_by_session_library`
- `src/robotmcp/components/keyword_matcher.py:254` — `discover_keywords`
  (no library_name knob today; out of scope)
- `src/robotmcp/plugins/builtin/browser_plugin.py:294` — Browser's
  `get_incompatible_libraries` → `["SeleniumLibrary"]`
- `src/robotmcp/plugins/builtin/selenium_plugin.py:183` — Selenium's
  `get_incompatible_libraries` → `["Browser"]`

## Implementation outcome (2026-05-17)

Implemented as described. Single resolver block at the top of
`find_keywords` (server.py:2186-2204) computes
`effective_library_preference` and `library_filter_source` with the
proposed precedence. Three branches (semantic / pattern / catalog) now
use the resolved value uniformly. The catalog branch still scopes its
underlying engine lookup by `library_name` (unchanged); the post-filter
is layered on top for response-shape consistency.

Response shape gains two optional fields when filtering applies:
- `library_filter`: `{"applied": "<lib>", "source": "library_name"|"session"}`
- `session_library`: `<lib>` (legacy field — preserved for callers that
  read it; populated with the same effective preference)

When no filter applies (neither parameter nor session preference set, or
unknown library_name with empty incompatibility list), no extra fields
are added — fully backward-compatible.

### Verification

Re-running the original repro with `library_name="Browser"` against the
fixed code:

```
=== After fix:
  Browser: ['Click', 'Tap', 'Select Options By']
excluded_keywords count: 7
library_filter: {'applied': 'Browser', 'source': 'library_name'}
session_library: Browser
```

All 7 SeleniumLibrary keywords are now excluded; `Select Options By`
moves into the top 3.

### Tests

`tests/unit/test_find_keywords_library_name_filter.py` — 14 tests
across 9 layers covering the matrix listed under "Tests to add":

- Reproducer (4 tests): defect-payload exclusion + `excluded_keywords`
  populated + `library_filter.source` + `session_library` backcompat
- Symmetry: SL filter excludes Browser
- Strategy parity: pattern + `library_name`
- Precedence: `library_name` overrides session preference
- Fall-through: session preference applies when `library_name` is None
- Idempotency: neither set → no filter, no extra fields
- Unknown library: graceful no-op (no exclusions, no error)
- Sibling preserved: BuiltIn keywords remain under Browser filter
- Docstring contract: signature + "ALL strategies" announcement

### Test totals

5906 unit tests passed, 1 skipped, 0 failures (+14 net).
