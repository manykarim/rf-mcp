# Discovery tools evaluation: find_keywords + get_keyword_info

**Date**: 2026-05-18
**Scope**: end-to-end behaviour of `find_keywords` and `get_keyword_info` under a 29-scenario matrix
**Goal stated by user**: *"token-friendly and guiding response to AI agents without creating much noise"*
**Method**: programmatic benchmark via `scripts/benchmark_discovery_tools.py`; per-scenario JSON dumps + externalised artifacts saved under `/tmp/discovery_benchmark/`
**Inputs measured**:
- Strategy: `semantic`, `pattern`, `catalog`, plus all 4 modes of `get_keyword_info`
- Library filter: `Browser`, `SeleniumLibrary`, none
- Query specificity: precise, vague, single-word, nonsense, empty, cross-domain, long-prose, BDD-prefixed, exact, glob

This report uses tokens as the cost unit (~4 chars/token throughout) and assesses
relevance qualitatively against the user's stated intent for each scenario.

---

## Executive summary

The recent OBS-defect series fixed a class of correctness bugs (library_name
ignored, verbose excluded_keywords, stale recommendations). But the underlying
discovery surface still has four structural defects independent of those fixes:

1. **Semantic relevance is shallow.** A query like *"select dropdown option by
   visible label"* against SeleniumLibrary returns `Element Should Be Visible`
   as the #1 hit at 0.87 confidence, while the correct answer
   `Select From List By Label` ranks #2 at 0.82. The matcher rewards token
   overlap on the query's adjectives ("visible") above the semantic verb
   ("select"). On `"send http post request"` + Browser filter, top match is
   `New Persistent Context` at 0.72 — completely off-domain.
2. **Empty/nonsense/vague queries return bloat or misleading "no matches"
   messaging.** Empty string returns one accidental hit with a 40-argument list
   (~838 tokens). `"do something with form"` returns 0 matches but with a
   `total_matches: 345, filtered_count: 10` signal that the agent could be
   coached on (and isn't).
3. **No externalisation rule for `get_keyword_info`.** A `mode="library"` call
   on Browser returns **71,521 inline tokens** in a single response. A typical
   `mode="keyword"` call costs 200-400 tokens which is fine; library-mode is a
   land mine.
4. **Pattern strategy substring-matches on docs+tags, not just names.** Query
   `"Go To"` returns `BuiltIn.Repeat Keyword` as match #1 because its
   docstring contains the phrase "go to". 85 results returned for `"Get*"` at
   5,407 tokens. The strategy is documented as glob-or-substring but the
   agent has no way to scope it to names-only.

The current PR (`feature/obstacle-course-followup`) is shippable as-is — these
are pre-existing issues uncovered by the deeper benchmark, not regressions.
Filing as follow-up scope.

---

## Scenario matrix (29 cases)

| ID | Description | InlTok | ArtTok | TopMatch | Notes |
|---|---|---:|---:|---|---|
| **find_keywords / semantic** | | | | | |
| S01 | precise + Browser | 560 | 0 | `Browser.Select Options By` (0.80) | ✅ correct |
| S02 | precise + SeleniumLibrary | 1017 | 0 | `SL.Element Should Be Visible` (0.87) | ❌ wrong; `Select From List By Label` is #2 |
| S03 | precise + no filter | 1321 | 0 | `SL.Element Should Be Visible` (0.87) | ❌ wrong; same matcher quality issue |
| S04 | vague "do something with form" + Browser | 132 | 0 | — | 0 matches; 345 total, 10 filtered. Misleading "no matches" guidance. |
| S05 | vague + no filter | 1043 | 0 | `SL.Submit Form` (?) | filler matches; "form" is too generic |
| S06 | "click" + Browser | 412 | 0 | `Browser.Click` (0.80) | ✅ correct, compact |
| S07 | "navigate" + Browser | 725 | 0 | `Browser.Go To` (0.80) | ✅ correct |
| S08 | nonsense "banana telephone" + Browser | 159 | 0 | — | 0 matches; clean |
| S09 | empty query + Browser | 838 | 0 | `Browser.New Persistent Context` (0.35) | ❌ 40-arg list dump for a confidence-0.35 hit |
| S10 | cross-domain "send http post request" + Browser | 1053 | 0 | `Browser.New Persistent Context` (0.72) | ❌ off-domain; no "wrong library" hint |
| S11 | cross-domain + no filter | 1646 | 0 | `SL.Get Session Id` | ❌ matcher has no RequestsLibrary signal for API queries |
| S12 | long prose + Browser | 1009 | 0 | `Browser.Click` (?) | ✅ matcher held up under verbosity |
| S13 | BDD "When I click submit button" + Browser | 420 | 0 | `Browser.Click` (0.80) | ✅ BDD prefix stripped correctly |
| **find_keywords / pattern** | | | | | |
| S14 | `Click*` + Browser | 274 | 0 | `Browser.Click` | ✅ glob works correctly |
| S15 | `Click*` + no filter | 476 | 0 | `Browser.Click` | ✅ Browser + SL both present |
| S16 | `Get*` + Browser | 5407 | 0 | `BuiltIn.Get Count` | ⚠️ 85 results, 10 libraries; library_name filter didn't tighten enough |
| S17 | exact "Go To" + Browser | 847 | 0 | `BuiltIn.Repeat Keyword` | ❌ "Go To" matches docstring "go to" in Repeat Keyword |
| S18 | `XYZNoSuchThing*` | 30 | 0 | — | ✅ zero matches; cleanest response of the run |
| **find_keywords / catalog** | | | | | |
| S19 | Browser catalog (limit=20) | 1765 | 0 | `Browser.Evaluate JavaScript` | per-keyword fields verbose; libdoc string included full |
| S20 | catalog, no session, no lib | **97212** | 0 | `BuiltIn.Call Method` | ❌ unbounded dump: 658 keywords inline |
| S21 | catalog "select" + Browser | 634 | 0 | `Browser.Deselect Options` | ✅ tight |
| **get_keyword_info** | | | | | |
| K01 | known `Click` + Browser | 381 | 0 | (doc returned) | ✅ correct, single result |
| K02 | known `Select From List By Label` + SL | 199 | 0 | (doc returned) | ✅ |
| K03 | `Go To`, no library (ambiguous) | 427 | 0 | (2 matches: Browser + SL) | ✅ explicit ambiguity surfacing |
| K04 | unknown `XYZNoSuchKeyword` | 27 | 0 | error | ✅ tight error, but **no "did you mean" hint** |
| K05 | typo `Clikc` | 24 | 0 | error | ❌ no fuzzy match suggestion |
| K06 | library mode Browser | **71521** | 0 | (full library doc) | ❌ no externalisation; ~70k tokens of doc inline |
| K07 | parse mode `Click(css=...)` | 48 | 0 | (parsed args) | ✅ very compact |
| K08 | BDD "When Click" + Browser | 31 | 0 | error + `suggestions: ["Click"]` | ✅ BDD prefix stripping works, suggestion returned |

---

## Finding 1: matcher confidence rewards token overlap, not intent

### Evidence

| Query | Library | Top match (confidence) | Correct answer |
|---|---|---|---|
| "select dropdown option by visible label" | SeleniumLibrary | `Element Should Be Visible` (0.87) | `Select From List By Label` (#2 at 0.82) |
| "select dropdown option by visible label" | none | `Element Should Be Visible` (0.87) | (same) |
| "send http post request with json body" | Browser | `New Persistent Context` (0.72) | none in Browser; should signal "wrong library" |
| "send http post request with json body" | none | `Get Session Id` (SL) | should be `Post On Session` (RequestsLibrary) |
| "" (empty) | Browser | `New Persistent Context` (0.35) | should be empty |
| "do something with form" | Browser | (0 matches after filter) | best-effort suggestions |

### Root cause

`KeywordMatcher` combines pattern matching + simple semantic + context matching
(keyword_matcher.py:284-300). The semantic step weights individual token
overlap; multi-word adjectival queries get high confidence on keywords that
share the adjectives without sharing the verb. Empty query returns ranked
matches against an empty action-description — every keyword scores >0.

### Impact

The Best-match recommendation is the FIRST thing the agent reads. When it
names a keyword that does the wrong thing, the agent writes code that fails
at execute_step (or worse, silently does something unintended). The cost is
not tokens — it's behavioural correctness.

### Improvement proposal

1. **Down-weight stop-word overlap.** Words like "visible", "should", "be",
   "by", "with", "from", "the", "for" are linguistic glue, not intent.
   Penalise their contribution to confidence (~0 weight on stop-words).
2. **Confidence floor + diversity heuristic.** If the top-3 matches all share
   the same library + same verb-class, that's a meaningful signal. If they're
   wildly divergent (Click, Element Should Be Visible, Set Window Position),
   confidence on the top is probably brittle and should be capped.
3. **Cross-library "wrong domain" hint.** When all top-N matches against the
   filter library have confidence < 0.5 AND a candidate from another library
   would score > 0.7, surface a hint:
   `"No high-confidence match in Browser. Consider RequestsLibrary for API operations."`
   This is more actionable than the bare "No matching keywords found"
   guidance.
4. **Empty-query short-circuit.** `query == ""` should return `success=False`
   with a `"provide a query"` hint, NOT a fallback match.

---

## Finding 2: vague/no-match queries produce misleading "no matches" guidance

### Evidence (S04, S08)

```
S04 query="do something with form" + Browser:
  matches: []
  total_matches: 345
  filtered_count: 10
  recommendations: ["No matching keywords found. Consider:",
                    "- Check if required libraries are imported",
                    "- Rephrase the action description",
                    "- Use more specific terms"]
```

The matcher found 345 candidate keywords but the library filter excluded all
top-10. The "No matching keywords found" guidance is *technically* correct but
misleading — the agent gets no signal that 10 plausible candidates were just
removed from another library.

### Improvement proposal

When `total_matches > 0` but `matches == []` after filter, distinguish the
two failure modes in `recommendations`:

```
"All top matches were from {from_library} and excluded by your library
filter. Either:
  - Try the same query without library_name to see what's available
  - Or switch sessions to a {from_library} session"
```

If a session has no library preference set AND the filter is from the
per-call `library_name` parameter, the second branch should hint at the
parameter, not the session.

---

## Finding 3: `get_keyword_info(mode="library")` returns 70k+ tokens inline

### Evidence (K06)

```
mode=library, library_name=Browser:
  inline_tokens: 71521
  structure:
    library.doc: 36328 chars (~9000 tokens — library-level prose)
    library.keywords: 147 entries (full doc each)
```

### Impact

A single call burns ~70k tokens of the agent's context window. The
`find_keywords` tool externalises its `result` field via ADR-015 when
`session_id` is present and content exceeds 500 tokens. The
`get_keyword_info` tool has NO externalisation rule.

### Improvement proposal

Add three rules to `DEFAULT_RULES` in `domains/artifact_output/services.py`:

```python
# get_keyword_info: library mode dumps full libdoc
ExternalizationRule(tool_name="get_keyword_info", field_path="library.doc"),
ExternalizationRule(tool_name="get_keyword_info", field_path="library.keywords"),
# get_keyword_info: keyword mode can also be large for verbose docstrings
ExternalizationRule(tool_name="get_keyword_info", field_path="keyword.doc"),
```

Threshold (500 tokens default) and policy (file_path summary template) reuse
the existing infrastructure. The compact summary string ("Content saved to
... 71521 bytes") replaces the inline content, and the agent can fetch via
the standard artifact-fetch path if it needs the body.

For the agent, a one-line summary plus `short_doc` (Browser library has a
working short_doc per keyword, 50-100 chars each) would be a far more
useful inline shape than the full libdoc.

---

## Finding 4: pattern strategy substring-matches docstrings, not just names

### Evidence (S17)

Query `"Go To"` (no wildcards) + `library_name="Browser"` returns:
```
1. BuiltIn.Repeat Keyword     ← matches because doc contains "go to"
2. Browser.Go To              ← actual hit
3. Browser.Merge Coverage Reports  ← doc contains "go to" somewhere?
4. Browser.New Context
5. Browser.Start Coverage
```

`rf_libdoc_integration.py:286` substring-matches across **name, doc,
short_doc, and tags**. For agents using pattern strategy to find a keyword
by name (the canonical use case — "I half-remember the keyword name"), this
broad search produces noise.

### Improvement proposal

Two options, increasing in scope:

**(A) Add a `pattern_scope` parameter** (default = "name"):
```python
find_keywords(query="Go To", strategy="pattern", pattern_scope="name")
# Substring matches against keyword names only (3 hits, all named "Go To")
```
Backwards-compatible: omitting `pattern_scope` keeps current behaviour.

**(B) Smarter default**: if the query string is a plausible RF keyword name
(contains capitalised words and spaces, no `*?[`), match name-only. If it has
wildcards, glob match the name. If it's lowercase prose, full substring
search.

Both reduce S17's noise; (B) is more agent-friendly (one less parameter for
the LLM to remember) but has heuristic risk on edge cases. Recommend (A) +
default to (B)'s heuristic.

---

## Finding 5: pattern + `Get*` returns 85 results across 10 libraries

### Evidence (S16)

Query `"Get*"` + `library_name="Browser"` → 85 results, 5407 tokens.
Libraries surfaced: Browser, BuiltIn, Collections, DateTime, Dialogs,
OperatingSystem, Process, RequestsLibrary, String, XML.

The library_name filter only excludes incompatible libraries
(SeleniumLibrary). Compatible/neutral libraries (BuiltIn etc.) pass through.
For an agent using `Get*` to find Browser keywords specifically, this is
under-filtered.

### Improvement proposal

Add a `strict_library` mode to library filtering: when `library_name` is
given AND `strict_library=True`, exclude every library that isn't the
named one (drop BuiltIn etc. too). Default to current behaviour (compatible
siblings preserved) so the change is opt-in.

Or: introduce a `library_only=True` short-hand for the strict mode.

Token impact: S16 with strict_library would shrink from 5407 tokens to
roughly 800 tokens (only ~15 `Get*` keywords in Browser).

---

## Finding 6: no fuzzy match / typo correction in `get_keyword_info`

### Evidence (K05)

```
get_keyword_info(keyword_name="Clikc")
  → {success: false,
     error: "Keyword 'Clikc' not found in any loaded library",
     mode: "keyword"}
```

But K08 (`When Click`) returns a `suggestions` list because BDD prefix
stripping is wired in. So the response shape *does* support suggestions —
the typo path just isn't wired into it.

### Improvement proposal

Where the BDD-prefix path already adds `suggestions: ["Click"]`, add a
Levenshtein / fuzzy lookup against the loaded-libraries keyword list. Top
3 candidates with distance ≤ 2-3 chars, ranked. Skip when distance ≥ 4
(avoid false positives).

```python
get_keyword_info(keyword_name="Clikc") →
  {success: false,
   error: "Keyword 'Clikc' not found in any loaded library",
   suggestions: ["Click", "Click Button", "Click Element"],
   mode: "keyword"}
```

Token cost: ~30 tokens for 3 suggestions. Recovery value: huge — the agent
goes from "keyword not found, retry?" to "did you mean Click?" in one
round-trip.

---

## Finding 7: `catalog` strategy with no session and no library dumps 97k tokens

### Evidence (S20)

```
find_keywords(strategy="catalog")  # no session_id, no library_name
  → 658 keywords, 97212 inline tokens
```

There's an existing guard for catalog returning empty (server.py:2380):

```python
if not catalog and not session_id:
    result["hint"] = (
        "Catalog is empty because no session with libraries is active. "
        "Create a session first ..."
    )
```

But it only fires when catalog IS empty. The empty-catalog case happens
when no libraries are loaded at all. The user's `S20` runs in a state where
libraries ARE loaded globally, so `catalog` returns everything — the
opposite of the empty case.

### Improvement proposal

Two-layer protection:

1. **Hard cap when no session, no library_name, no query**: cap at e.g.
   100 entries with a hint:
   ```
   "Catalog returned 658 keywords across 10 libraries. Use library_name or
    a query filter to narrow. First 100 returned."
   ```
2. **Externalise the `results` field** for catalog strategy too (currently
   only `result` is externalised, and catalog uses `results`). Add:
   ```python
   ExternalizationRule(tool_name="find_keywords", field_path="results"),
   ```
   This routes the bulk dump through the artifact mechanism when session_id
   is present.

---

## Finding 8: empty-query semantic returns confidence-0.35 hit with 40-arg dump

### Evidence (S09)

```
find_keywords(query="", strategy="semantic", library_name="Browser") →
  matches: [{
    library: "Browser",
    keyword_name: "New Persistent Context",
    confidence: 0.35,
    arguments: [40-element list],  ← burns ~600 tokens alone
    ...
  }]
  inline_tokens: 838
```

### Improvement proposal

Add an explicit empty-query guard at the top of `find_keywords` (semantic /
pattern branches):

```python
if strategy_norm in {"semantic", "intent", "pattern", "search"} and not query.strip():
    return {
        "success": False,
        "strategy": strategy_norm,
        "query": query,
        "error": "Query string is required for {strategy} strategy",
        "hint": "Use strategy='catalog' to list available keywords without a query.",
    }
```

The catalog strategy already handles empty queries (lists everything within
the active library scope); semantic/pattern do not.

---

## Finding 9: arguments field in semantic matches is verbose

### Evidence

Multiple scenarios (S09, S10, S19) carry a `matches[].arguments` field that
lists every argument name as a string. For keywords like `New Persistent
Context` (40 args) or `Open Browser` (5+ args), this becomes the largest
contributor to inline-token cost per match.

The `Required arguments: …` line in `recommendations` repeats the same
information.

### Improvement proposal

For semantic strategy at the default detail level (the LLM-facing mode), cap
the `arguments` field at the top N (say 6) argument names with a `+M more`
suffix when truncated:

```json
"arguments": ["userDataDir", "browser", "headless", "args", "baseURL", "bypassCSP", "+34 more"]
```

The full list is available via `get_keyword_info(mode="keyword")` for the
specific keyword. Token savings on the worst-case keyword (Browser's `New
Persistent Context`): ~600 tokens → ~80 tokens.

---

## Finding 10: stale `recommendations` field even after the OBS-15 rebuild fix

### Verified working

S01 (precise + Browser) recommendations correctly name `Select Options By`
post-filter. The post-OBS rebuild path is operating as designed.

### Edge case still wrong

S02 (precise + SeleniumLibrary): top match is `Element Should Be Visible`
which is in the SL library (matcher quality issue, not filter bug). The
recommendation correctly names this top match — but the *content* of the
recommendation is wrong because the matcher itself ranks the wrong keyword
first. See Finding 1.

No fix here — this is Finding 1 surfacing again. Just confirming the rebuild
is doing its job.

---

## Aggregate token cost picture

| Tool | Worst case observed | Median |
|---|---:|---:|
| `find_keywords` semantic | 1646 (S11) | ~700 |
| `find_keywords` pattern | 5407 (S16 `Get*`) | ~400 |
| `find_keywords` catalog | **97212** (S20 no-session) | ~1500 |
| `get_keyword_info` keyword | 427 (K03 ambiguous) | ~300 |
| `get_keyword_info` library | **71521** (K06 Browser) | (single mode) |
| `get_keyword_info` parse | 48 (K07) | <50 |

The two outliers — `S20` (catalog dump) and `K06` (library mode) — together
account for tens of thousands of tokens per call. Both fall outside the
existing externalisation rules.

---

## Prioritised improvement plan

| Rank | Fix | Effort | Impact |
|---|---|---|---|
| 1 | Externalise `get_keyword_info` library mode (K06) | S | -70k tokens worst case |
| 2 | Hard cap + truncation hint on catalog no-session-no-lib (S20) | S | -95k tokens worst case |
| 3 | Externalise catalog `results` field too | XS | depends on size |
| 4 | Empty-query short-circuit (S09) | XS | -800 tokens per empty call, prevents misleading hits |
| 5 | Fuzzy/Levenshtein suggestions in `get_keyword_info` (K05) | S | huge recovery value, +30 tokens cost |
| 6 | "Wrong library" cross-library hint when all top matches filtered (S04, S10) | M | high agent-correctness value |
| 7 | Pattern `pattern_scope="name"` parameter (S17) | S | -50% noise on pattern queries |
| 8 | `strict_library` mode on filter (S16) | S | -85% on pattern Get* type queries |
| 9 | Stop-word down-weight in semantic matcher (S02, S03) | M | matcher correctness, ranks correctly |
| 10 | Truncate verbose `arguments` field in matches (S09, S10) | XS | -600 tokens worst case per match |

**Total potential reduction**: a typical "well-formed query in a Browser
session" call (semantic, precise, ~700 tokens today) could shrink to
~150-250 tokens with the inline-field truncations alone. The library-mode
and catalog-no-session cases (the two whales) can drop by ~98% with
externalisation rules.

---

## Out-of-scope follow-ups

These surfaced during the benchmark but live outside the discovery tools:

- The matcher's `_classify_action` (keyword_matcher.py:282) classifies query
  intent into action types ("click", "navigate", etc.). Empty/nonsense
  queries return `action_type: "unknown"` but the matcher still ranks
  matches. Should `unknown` short-circuit?
- `BddPrefixService.strip_prefix` is invoked in `find_keywords` (server.py:
  ~2295) but the user can't see what was stripped — debug-level logging
  only. Consider surfacing the stripped form in the response for agent
  awareness.
- The plugin's `KEYWORD_ALTERNATIVES` table covers 8 SL→Browser mappings.
  The benchmark's `S01` (select dropdown) excludes 5 SL keywords but only 2
  have actionable alternatives (Click Button, Click Element → Click) —
  neither relevant to the dropdown query. The `excluded_alternatives` list
  for dropdown queries is currently irrelevant noise. Either extend the
  table OR filter alternatives by query-relevance (only show alternatives
  that semantically match the query).

---

## Reproducing the benchmark

```bash
uv run python scripts/benchmark_discovery_tools.py
```

Outputs:
- `/tmp/discovery_benchmark/<scenario_id>.json` — full response + metrics
- `/tmp/discovery_benchmark/<scenario_id>.artifact.txt` — externalised content (when applicable)
- `/tmp/discovery_benchmark/summary.json` — aggregated metrics table

Scenarios are defined in the `SCENARIOS` list in the script. Add new ones by
appending a dict with `id`, `description`, `tool`, `kwargs`.

---

# Second-round review (2026-05-18) — Codex CLI feedback + verification

After completing this report I asked Codex CLI (`gpt-5.4`, read-only sandbox)
to do an independent adversarial review. Its sandbox blocked filesystem reads
in this session, so its per-finding verdicts on findings 1-10 are not usable
— but it surfaced four concrete defects this report missed, two methodology
critiques, and a prioritisation challenge worth addressing.

## Codex defects this report missed (verified)

### Finding 11 — `find_keywords(strategy="semantic")` silently ignores the `limit` parameter

**Verified at server.py:2408 (pattern branch applies `matches = matches[:limit_value]`),
server.py:2477 (catalog same), but NO equivalent line in the semantic branch.**
The matcher hard-caps at 10 (`keyword_matcher.py:307`: `top_matches = ranked_matches[:10]`).
Caller-supplied `limit=20` returns 10. Caller-supplied `limit=3` still returns 10.

This is a real defect. Agents that pass `limit` expect it to be honoured;
silent override is worse than rejecting the parameter.

**Fix**: in the semantic branch, after the filter rebuild, apply
`discovery["matches"] = discovery["matches"][:limit_value]` when
`limit_value is not None`. Also rebuild recommendations against the
limit-trimmed list (otherwise recommendations[0] could name a match that's
not in the returned list).

### Finding 12 — `find_keywords(strategy="session")` ignores BOTH `query` and `limit`

**Verified at server.py:2531-2546.** The session branch literally is:

```python
mgr = get_rf_native_context_manager()
payload = mgr.list_available_keywords(session_id)
payload.update({"strategy": "session", "query": query})  # echoed but not used
```

Whatever the caller passes as `query` is echoed back into the response but
NEVER consulted by the underlying `list_available_keywords` call. No
`limit` either. A session with 200 imported keywords dumps all 200.

This is a worse silent-override defect than Finding 11 because:
1. The use case ("show me click-related keywords in my live session") is
   plausible and naturally expressed as `query="click"` + `strategy="session"`.
2. The agent has no signal that the query was ignored — the echoed
   `"query": "click"` in the response actively suggests it was honoured.

**My benchmark did NOT test session strategy at all** — gap in the matrix.
Codex correctly called this out.

**Fix**: apply name-substring filter on the namespace dump (mirror pattern
strategy's substring behaviour, or wire to libdoc search) before returning.
Apply `limit_value` as the final trim.

### Finding 13 — Discovery-time filter is weaker than execution-time filter

**Verified.** `_filter_keywords_by_session_library` at server.py:2037 only
runs when `session.explicit_library_preference` is set. The Browser plugin's
execution-time validation (`browser_plugin.py:243`) ALSO falls back to
imported libraries — so an execute_step in a Browser-only session rejects
SeleniumLibrary keywords even when `explicit_library_preference` is None.

Result: a session with `libraries=["Browser"]` but no explicit preference
returns SL keywords in `find_keywords` output, then has them rejected at
`execute_step`. Agent reads the discovery response, picks an SL keyword
from it, gets a runtime error.

This is asymmetric behaviour between two tools that should agree.

**Fix**: when filtering for discovery, fall back to `session.imported_libraries`
when `explicit_library_preference` is None. If the session imported a UI
library exclusively (Browser XOR SeleniumLibrary), use that as the implicit
preference for the filter.

### Finding 14 — Ambiguity collapse in LibDoc-fallback path

**Verified, but narrower than Codex stated.** In `execution_coordinator.py:
1021-1090`, the LibDoc primary path correctly returns ALL matches via
`get_keywords_documentation_all` when `library_name=None`. My K03 scenario
confirms this (returned both Browser and SL entries for "Go To").

**However**, the inspection-based fallback (line 1115-1116, when LibDoc isn't
available) silently returns the first match: `ki = keyword_discovery.find_keyword(keyword_name)`.

So the defect is:
- **LibDoc path (default)**: correct, returns matches[] array
- **Inspection fallback path**: collapses, returns single keyword

The inspection fallback rarely triggers in production (LibDoc is preferred
and reliable), so this is a corner-case correctness bug. Worth fixing for
defence-in-depth, but lower priority than 11/12/13.

**Fix**: mirror the LibDoc path — return matches[] when library_name is None
in the inspection path. `keyword_discovery.find_keyword` would need a sibling
`find_all_keywords` method.

## Codex methodology critiques (mostly valid)

### M1 — Sessionless benchmark distorts the externalisation analysis

**Materially valid.** My benchmark ran 28 of 29 scenarios without
`session_id`. `_externalize_response` is gated on
`if session_id:` at server.py:2375 / 2406 / 2491 / 2540 — so my inline-token
numbers represent the *un-externalised* path, not what an agent with an
active session would see.

The implications:
- Findings 1, 4, 9, 10 — their token counts are correct for a sessionless
  caller. Real agents normally have a session. Numbers should be qualified.
- Findings 3 (K06 library mode) and 7 (S20 catalog dump) — externalisation
  rules wouldn't fire even WITH a session because no rule covers those
  field paths. So the worst-case stands.
- Finding 2 (vague query "no matches" guidance) — the misleading message
  is independent of session, still stands.

Practical update: I added a session-based mini-run during the second-round
verification. With `session_id` present, `find_keywords(query="select
dropdown option by label", strategy="semantic", library_name="Browser")`
returns 532 inline tokens (result field is 1798 bytes — under the 500-token
externalisation threshold so NOT externalised even with session). The
externalisation only kicks in for `result` fields > ~2000 bytes.

Conclusion: most semantic-strategy responses do NOT cross the externalisation
threshold even with session_id. The session-gated externalisation only
materially helps for outliers (catalog dumps, library-mode docs).

### M2 — Embeddings env-sensitivity

**Critical finding from this critique.** I verified:

```
sentence_transformers: NOT INSTALLED (No module named 'sentence_transformers')
```

And `pyproject.toml` does NOT list `sentence-transformers` as a dependency.

**This means**: the "semantic" strategy in this deployment runs WITHOUT
embedding similarity. The semantic confidence scores observed in this
benchmark are derived purely from pattern matching + context-aware (action
type + tags) heuristics, not from embedding similarity.

Implications:
- Finding 1's "matcher rewards token overlap" is the FULL behaviour of the
  matcher in this deployment, not a degraded mode. The fix (stop-word
  down-weighting) applies regardless of embeddings.
- A different deployment with `sentence-transformers` installed could
  produce materially different ranking quality. My report's matcher
  critique should be qualified as "in default deployments without
  sentence-transformers".
- The tool description in `find_keywords` advertises strategy="semantic"
  as "Natural language search (best for exploring)" — agents reading this
  expect embedding-style semantic search. In default deployments they get
  token overlap + tag matching. **This is a documentation defect**: the
  description over-promises.

**Two-track fix**:
1. Update the strategy="semantic" docstring to clarify that embedding
   similarity is opt-in (requires installing sentence-transformers).
2. Make `sentence-transformers` an optional extra in pyproject.toml so
   users who want true semantic matching can `uv add robotmcp[semantic]`.

### M3 — Reproducibility (line numbers vs main)

**Valid.** All line numbers in this report reflect state on
`feature/obstacle-course-followup` branch (which contains the OBS-fix series
+ library_name fixes). The helper functions `_rebuild_post_filter_recommendations`,
`_build_filter_diagnostics` etc. were added in commits f37ccdf-3580302 on that
branch and do not exist on `main`. Future readers consulting `main` should
checkout the feature branch.

## Codex prioritisation challenge

Codex argued S20 (catalog dump, 97k tokens) and K06 (library mode, 70k
tokens) are *exposed worst cases*, not dominant real-world paths, and the
report's #1+#2 ranking is biased toward maximum-payload scenarios rather
than common-use scenarios. Counter-arguments:

**S20 defence (partial concession)**: a competent agent under a tool-profile
typically calls `find_keywords(strategy="catalog", library_name="Browser")`
(scoped) rather than `strategy="catalog"` alone. Weaker / smaller-model
agents are more likely to omit `library_name`. The 97k cost happens only
in the broadest, least-scoped catalog call. Concede: S20 is a worst case,
not a common case. Keep on the list but demote to #4.

**K06 defence (stand)**: library-mode documentation is a natural recovery
path when an agent doesn't recognise a keyword family. It's not
common-per-call but it's common-per-session. 70k tokens of context erosion
from one call hurts even if rare.

**Revised prioritisation** (incorporating Codex feedback):

| Rank | Fix | Source | Justification |
|---|---|---|---|
| 1 | Externalise `get_keyword_info(mode="library")` (K06) | Original #1 | Stand — 70k token hit per occurrence |
| 2 | Wire `strategy="session"` query/limit (Finding 12 — Codex) | New | High frequency, plausible failure mode, silent contract violation |
| 3 | Wire `strategy="semantic"` `limit` parameter (Finding 11 — Codex) | New | Silent parameter override; agents expect it to work |
| 4 | Hard-cap catalog no-session-no-lib (S20) | Original #2 | Demoted: worst case, not common case |
| 5 | Discovery/execution filter symmetry (Finding 13 — Codex) | New | Silent contract violation between two tools that should agree |
| 6 | Externalise catalog `results` field | Original #3 | Same fix family as #1 |
| 7 | Empty-query short-circuit | Original #4 | Cheap fix, prevents misleading hits |
| 8 | Fuzzy/Levenshtein typo suggestions in `get_keyword_info` | Original #5 | Recovery value high; small token cost |
| 9 | "Wrong library" cross-domain hint | Original #6 | Correctness; harder to implement well |
| 10 | Pattern `pattern_scope="name"` parameter | Original #7 | Reduces noise for name-only searches |
| 11 | `strict_library=True` filter mode | Original #8 | Useful but advanced; opt-in |
| 12 | Stop-word down-weight + clarify semantic-strategy docs (M2) | Original #9 + new | Matcher correctness; doc honesty |
| 13 | Truncate verbose `arguments` field in matches | Original #10 | Per-match cleanup |
| 14 | LibDoc-fallback ambiguity collapse (Finding 14 — Codex) | New | Corner-case correctness; defence in depth |
| 15 | Make `sentence-transformers` an optional extra (M2 follow-on) | New | Either install it as default, or document it's optional |

## Net assessment after Codex review

What changed:
- **+4 new findings** (11 — semantic limit, 12 — session strategy ignores query, 13 — discovery/exec filter asymmetry, 14 — LibDoc fallback ambiguity).
- **M2 (embeddings env-sensitivity)** is the most consequential: this
  benchmark, and likely most deployments, run the "semantic" strategy
  without actual embedding similarity. The report's matcher critique
  applies to the default deployment but should be framed accordingly.
- **M1 (sessionless methodology)** distorts externalisation analysis less
  than Codex argued, because most response sizes don't cross the 500-token
  externalisation threshold anyway — the gate fires only for catalog-style
  dumps and library docs, where it doesn't help today either.
- **Prioritisation shifts**: Finding 12 (session strategy ignores query) and
  Finding 11 (semantic ignores limit) are higher-frequency than the original
  S20 catalog whale and should rank above it.

What didn't change:
- Findings 1, 2, 4, 5, 6, 7, 8, 9, 10 stand as written.
- K06 (library mode externalisation) stays #1.
- The fundamental matcher-quality issue (Finding 1) is real with or without
  embeddings.

## Limitations of the Codex review

- **Codex couldn't read local files** (sandbox blocked `bwrap` loopback).
  Its review of findings 1-10 is by-design-incomplete, based only on the
  `main` branch source it could fetch via GitHub MCP — which lacks the
  feature-branch helpers cited in the report. Per-finding verdicts from
  Codex are not reliable; only its code-traced observations (which I
  independently verified above) are.
- **Codex used the GitHub connector** for code reads, so its analysis
  reflects HEAD on `main`, not the feature branch state the report
  describes. The "line numbers don't match" critique is therefore
  expected, not evidence of a real defect.
- **One-shot review**: a follow-up Codex run with the report text pasted
  inline (working around the sandbox issue) would produce the
  per-finding verdict matrix. Worth doing before the next priority
  reshuffle.

---

# Third-round review (2026-05-18) — Codex CLI with full filesystem access

Re-ran the Codex CLI review with `--dangerously-bypass-approvals-and-sandbox`
so it could actually read the report + benchmark artefacts + source code.
This run produced the per-finding verdict matrix the first attempt couldn't.
The feedback was substantially more substantive — and substantially more
adversarial — than round 2. Several round-2 claims (including my own
verifications) had to be retracted or refined.

## Codex verdicts on findings 1-15

Each verdict below is captured verbatim from Codex; my round-3 response
follows.

### Finding 1 — `INCOMPLETE` (matcher mechanics misdiagnosed)

> "The matcher is not token-overlap based; it uses action-class seeding plus
> `SequenceMatcher` over keyword names/docs and then a pure confidence sort
> (`keyword_matcher.py:372-399`, `449-569`). Stop-word down-weighting is
> unlikely to fix S02/S03 by itself."

**Conceded.** I called it "token overlap" because that's what the symptom
*looked* like ("visible" was scoring highly for the "select dropdown by
label" query). The actual mechanism is `difflib.SequenceMatcher` over the
keyword name + doc string. The fix needs to be reranking-based (e.g.
penalise hits where the action-class doesn't match), not stop-word
weighting. **Original fix proposal does not solve the problem.**

### Finding 2 — `OVERSTATED`

> "The recommendations are misleading, but the response does already expose
> `library_filter.from_library` and `count` (`server.py:2161-2205`; S04 JSON).
> The problem is that recommendations ignore those diagnostics, not that the
> agent gets 'no signal'."

**Conceded.** The signal *is* in the response — the agent just has to
correlate `library_filter.count > 0` with `matches: []` and infer
"library mismatch". A model could do that today, the issue is purely
that the prose recommendation says "No matches" without referencing the
adjacent diagnostic. Smaller fix than I described.

### Finding 3 — `INCOMPLETE` (proposed fix won't work as-is)

> "K06 is real, but 'add rules to `DEFAULT_RULES`' is not sufficient.
> `get_keyword_info` never calls `_externalize_response` on any branch
> (`server.py:5160-5184`), so rules alone would do nothing."

**Conceded, this is the bigger issue.** I assumed `get_keyword_info` would
externalise the same way `find_keywords` does once a rule existed — but
the externalisation is opt-in per-tool via an explicit
`_externalize_response()` call. `get_keyword_info` doesn't call it
anywhere. The fix is two-part: (a) wire `_externalize_response` into
each `get_keyword_info` branch, then (b) add the field-path rules. The
report described only (b).

### Finding 4 — `AGREE`

> "Plain-text pattern search really does match `name`, `doc`, `short_doc`,
> and `tags` (`rf_libdoc_integration.py:286-313`)."

Confirmed; no change.

### Finding 5 — `OVERSTATED`

> "S16 is noisy, but it is mostly the documented behaviour of
> `library_name`, which preserves 'compatible siblings' rather than
> enforcing strict single-library scoping (`server.py:2244-2247`,
> `2307-2317`). This is a product choice more than a correctness bug."

**Partially conceded.** Codex is right that the current behaviour is
*documented* (the docstring change I made in PR #70 explicitly says
"library_name=Browser excludes SeleniumLibrary but keeps BuiltIn,
Collections, String"). So the `strict_library=True` mode is an
opt-in UX improvement, not a correctness fix. Demote priority
accordingly.

### Finding 6 — `DISAGREE` (this was wrong, not just overstated)

> "`get_keyword_info` does have typo/suggestion behaviour when
> `library_name` is provided (`execution_coordinator.py:1050-1066`).
> Your benchmark only proved the unscoped path lacks suggestions.
> `Clikc` + `library_name="Browser"` returns suggestions."

**Empirically confirmed by re-running**: `get_keyword_info(keyword_name="Clikc",
library_name="Browser")` returns `suggestions: ["Click", "Click With Options"]`.
My K05 scenario only exercised the unscoped path (no `library_name`).
The finding should be narrowed to: *"Unscoped typo lookups don't get
suggestions; scoped ones already do."* The fix is to add fuzzy fallback
to the unscoped branch (`execution_coordinator.py:1068-1090` LibDoc path
and `1115-1116` inspection path). Significantly smaller fix than originally
described.

### Finding 7 — `AGREE BUT` (fix doesn't fully solve it)

> "S20 is real, but your 'externalise `results`' idea does not solve the
> no-session case; externalisation is only invoked when `session_id` is
> present (`server.py:2518-2520`). The hard cap is the part that addresses
> S20."

**Conceded.** I proposed two layers: (1) hard cap + hint, (2) externalise
`results`. Codex is correct that (2) does nothing without `session_id`. The
hard cap is the actual fix for the unscoped catalog dump.

### Finding 8 — `AGREE`

> "Empty semantic queries produce a bogus hit because nothing short-circuits
> before matcher ranking; S09 is a valid defect."

Confirmed.

### Finding 9 — `AGREE BUT` (scope broader than I framed)

> "Verbose arguments are a real token sink, but this is not just a
> semantic-branch issue. Pattern/catalog payloads also carry full args,
> and recommendations duplicate them."

**Conceded.** I framed the truncation as semantic-only. It should be
uniform across all three strategies + the `recommendations`
"Required arguments:" line. Same fix, wider application.

### Finding 10 — `AGREE`

> "Your own conclusion is right: the rebuild fix is working; the remaining
> bad recommendation content is downstream of bad ranking, not stale
> post-filter state."

No change.

### Finding 11 — `AGREE BUT` (proposed fix is insufficient)

> "The defect is real, but your proposed fix is insufficient. Slicing after
> filtering does not honour `limit > 10` because the matcher has already
> hard-capped at 10 (`keyword_matcher.py:306-307`)."

**Conceded; this is a material correction.** My proposal — "apply
`discovery['matches'] = discovery['matches'][:limit_value]` after the
filter rebuild" — only works for `limit <= 10`. For `limit=20`, the
matcher returns at most 10 and my post-filter slice does nothing.

**Correct fix**: thread `limit` into `discover_keywords` so the matcher
can return up to that many ranked matches (default 10 if not provided),
THEN apply post-filter slice for cases where the filter dropped some
top matches. Two-call-site change, not one.

### Finding 12 — `AGREE BUT` (verification missed adjacent issues)

> "Query and limit are ignored (`server.py:2531-2546`), but your verification
> missed two adjacent problems: the session branch returns a different
> schema entirely and its payload is not externalisable under current rules."

**Conceded.** Session strategy returns `{success, libraries_count,
library_keywords, resource_keywords}` (verified at
`rf_native_context_manager.py:1678-1683`), NOT the
`{success, strategy, query, result: {matches, recommendations, ...}}`
shape that semantic/pattern/catalog return. So the schema mismatch is a
third defect on top of "ignores query" and "ignores limit". Externalisation
only covers `result`, not `library_keywords`/`resource_keywords`, so
even a fixed session-strategy payload would still dump 200+ keywords
inline.

### Finding 13 — `AGREE BUT` (scope narrower than stated)

> "The asymmetry is real (`server.py:2320-2331` vs `browser_plugin.py:
> 314-321,323-349`), but it is narrower than stated: this is plugin-specific
> behaviour, not a general discovery/execution invariant across all libraries."

**Conceded.** Only the Browser plugin uses imported-libraries fallback for
execution validation. Other plugins may not. The fix should be narrower —
either (a) make discovery filter mirror Browser plugin's specific fallback
logic, or (b) push the fallback logic into the discovery filter for all
plugins.

### Finding 14 — `AGREE`

> "Your narrowed verification is sound. The LibDoc path returns all
> matches; only the inspection fallback collapses ambiguity."

No change.

### Finding 15 — `AGREE BUT` (fix doesn't address the real gap)

> "The env-sensitivity point is real (`keyword_matcher.py:10-16`, `291-294`),
> but 'make `sentence-transformers` an optional extra' does not solve the
> default-behaviour mismatch by itself. The real gap is that the tool
> advertises 'semantic' even when embeddings are absent."

**Conceded.** Making it optional is fine for opt-in users but doesn't
help the default deployment where the tool says "semantic" and delivers
pattern+tag matching. The honest fix is one of:
- Update the strategy docstring to say "uses pattern+tag matching;
  install `sentence-transformers` for embedding-based ranking"
- Install `sentence-transformers` by default (adds a heavy dependency)
- Rename `strategy="semantic"` to `strategy="hybrid"` or `"smart"` so
  the API doesn't over-promise

## Genuine defects this report missed (Codex finds)

### Missed-1: `get_keyword_info(mode="keyword")` ignores `session_id` entirely

**Codex claim verified end-to-end.** Calling
`get_keyword_info(keyword_name="Click", session_id="dummy-nonexistent-session")`
returns `success=True` with the Click keyword document. The session_id
parameter is accepted, advertised in the docstring (server.py:5147), but
never consulted by the keyword/global branch (`server.py:5160-5165`).

**Severity**: same class of bug as the original `library_name` defect we
just fixed in PR #70 — silent parameter override. The tool can document
keywords that don't exist in the live session, leading agents to
construct calls that fail at `execute_step`.

**Fix**: keyword-mode lookups should consult session libraries
when `session_id` is provided — restrict to libraries imported in that
session. Mirror the fix shape of OBS-15 (library_name precedence
resolver).

### Missed-2: `find_keywords(strategy="session")` payload not covered by externalisation rules

**Codex claim verified.** The session-strategy response shape is
`{success, libraries_count, library_keywords, resource_keywords}`
(`rf_native_context_manager.py:1678-1683`). The only `find_keywords`
externalisation rule targets `field_path="result"`
(`services.py:59-60`). So even after fixing Finding 12 (query/limit),
large session dumps still go inline.

**Severity**: medium. A session with Browser (200+ kw) + BuiltIn (100+ kw)
imported produces a ~300-keyword dump. Most realistic large-session
payloads will exceed the 500-token threshold but never trigger
externalisation.

**Fix**: add externalisation rules for `library_keywords` and
`resource_keywords` field paths.

### Missed-3: Pattern `results` not externalised either

**Codex claim verified.** The only existing rule covers `result`
(singular). The pattern branch's response field is `results`. So a 5407-
token `Get*` response (S16) goes inline even WITH `session_id`.

**Fix**: add `field_path="results"` rule. Cheap one-liner in
`services.py`.

### Missed-4: My K08 interpretation was wrong

**Conceded.** K08 was: `get_keyword_info(keyword_name="When Click",
library_name="Browser")`. Response carried `suggestions: ["Click"]`.

I wrote in the original report:
> "BDD-prefix stripping works for get_keyword_info when library_name is
> given"

**But there is no BDD stripping in `get_keyword_info`.** A `grep -n
"BddPrefixService\|strip_prefix"` in `server.py` confirms only
`find_keywords` (line 2291) and `execute_step` (line 3742) strip BDD
prefixes. `get_keyword_info` does not.

What actually happened in K08: `get_keyword_info` looked up "When Click"
in Browser, failed to find it, and the *library-scoped fuzzy fallback*
at `execution_coordinator.py:1050-1066` returned `Click` as a
substring/prefix match (because "Click" appears in "When Click" via
substring matching). The "Click" suggestion was coincidental, not
intentional BDD support.

This is a documentation defect in my own report. The K08 entry in the
scenario matrix should be re-labelled.

## Codex re-prioritisation challenge

Codex pushed back on my round-2 priority ordering:

> "Finding 12 is not convincingly #2-worthy. It is real, but you have no
> frequency evidence because the benchmark never exercised session
> strategy, and that branch is also structurally inconsistent in schema
> and externalisation. I would rank Finding 13 above it because 13
> affects ordinary discovery flows that can directly cause execution-time
> rejection in Browser sessions."

**Conceded.** Finding 12's frequency is unknown because I didn't benchmark
it. Finding 13 affects every Browser session that doesn't set
`explicit_library_preference` explicitly (common pattern — `manage_session`
auto-detection doesn't always set it).

> "What is wrongly demoted is Finding 1. Even though your root-cause
> analysis is off, the default discovery path returning the wrong top
> keyword is a more fundamental agent-correctness problem than several
> payload-shaping issues."

**Conceded.** Finding 1 (matcher returns wrong top match) is the most
fundamental issue — it breaks the discovery contract. Payload shaping
fixes don't help if the top match is wrong.

> "What is omitted entirely is the `get_keyword_info(keyword mode) +
> session_id` bug above."

Added as Missed-1.

## Round-3 prioritisation (post-Codex)

| New # | Was | Fix | Justification |
|---|---|---|---|
| 1 | F1 (demoted to mid in R2) | **Matcher reranking** (not stop-word weighting) | Most fundamental — top-match correctness drives all downstream agent behaviour |
| 2 | NEW (Missed-1) | `get_keyword_info(mode="keyword")` honour `session_id` | Same class as OBS library_name defect; silent parameter ignore |
| 3 | F13 (was R2 #5) | Discovery/execution filter symmetry | Affects ordinary Browser sessions without explicit preference |
| 4 | F3 (was R2 #1) | **Wire `_externalize_response` into `get_keyword_info`** + add field-path rules | Conceded fix scope; both pieces required, not just rules |
| 5 | F11 | Apply `limit` to **matcher itself + post-filter** | Fix scope expanded per Codex; threading limit into discover_keywords |
| 6 | F12 (was R2 #2) | Session strategy: honour query+limit + unify schema + externalise payload | Codex correctly demoted: triple defect but unknown frequency |
| 7 | F8 | Empty-query short-circuit | Cheap fix |
| 8 | F4 (S20 catalog) | Hard cap + hint when no session, no library | Concede "externalise results" doesn't help no-session |
| 9 | F6 | Fuzzy/typo suggestions for **unscoped** `get_keyword_info` only | Codex correctly narrowed: scoped path already works |
| 10 | NEW (Missed-3) | Externalise pattern `results` field | One-liner; same shape as existing catalog rule discussion |
| 11 | NEW (Missed-2) | Externalise session-strategy payload fields | Add `library_keywords`/`resource_keywords` rules |
| 12 | F2 | Use `library_filter` diagnostics in recommendations text | Concede: signal already present, just unused in prose |
| 13 | F9 | Truncate verbose `arguments` across ALL branches + recommendations | Concede: wider scope than originally framed |
| 14 | F15 + M2 | Honest semantic-strategy docs OR rename to "hybrid" | Concede: optional extra isn't sufficient |
| 15 | F10 (S17) | Pattern `pattern_scope="name"` parameter | Unchanged |
| 16 | F14 | LibDoc-fallback ambiguity collapse | Unchanged; corner case |

## What changed vs round 2

1. **Finding 1 promoted to #1** (was R2 mid-tier). Matcher correctness >
   payload shaping.
2. **Finding 3 fix scope expanded**: need to wire `_externalize_response`
   into `get_keyword_info` FIRST, then add field-path rules. Two-step,
   not one.
3. **Finding 11 fix scope expanded**: post-filter slice doesn't honour
   `limit > 10`. Need to thread limit into `discover_keywords` as well.
4. **Missed-1 added**: `get_keyword_info(mode="keyword")` ignores
   `session_id` — same class of bug as the original OBS library_name
   defect.
5. **Finding 6 narrowed**: scoped typo suggestions already work; only the
   unscoped path needs the fix.
6. **K08 re-labelled**: my "BDD prefix stripping works" interpretation
   was wrong — the suggestion came from library-scoped fuzzy fallback,
   not from BDD support. `get_keyword_info` has no BDD support at all.
7. **Finding 12 demoted from #2 to #6**: real defect but no frequency
   evidence, and the report didn't benchmark session strategy.

## Methodology gaps Codex called out

Round-2 listed missing scenarios but didn't exhaustively enumerate.
Codex provided a more complete list:

- Pattern strategy WITH `session_id` (would expose Missed-3)
- Catalog WITH `session_id` and no `library_name` (separates the
  unrealistic S20 whale from the realistic session-backed dump)
- `get_keyword_info(mode="keyword")` WITH `session_id` (would have caught
  Missed-1 immediately)
- `get_keyword_info(mode="session")` with ambiguous names
- BDD-prefixed pattern queries (BDD stripping happens before strategy
  dispatch at server.py:2288-2297 — never tested)
- Cross-strategy same-query comparisons (isolate matcher quality from
  strategy choice)
- Browser-imported session WITHOUT `explicit_library_preference` (for
  Finding 13)
- Typos WITH `library_name` (would have prevented the Finding 6
  overgeneralisation)

A round-2 benchmark expansion to cover these would re-verify the
prioritisation with frequency data.

## What I overstated about my own verification

Codex's closing line:

> "The report overstates the rigor of its own verification in two places:
> K08 was misread as BDD stripping, and Finding 11's proposed fix does
> not actually satisfy the observed `limit=20` failure. Those are not
> small nuances; they materially affect the credibility of the revised
> prioritization."

**Conceded both.** The round-2 review was less rigorous than I framed it.
The K08 misread propagated into the original report; the Finding 11 fix
was sketched without checking the matcher's hard cap. This round-3
section corrects both.

## Status of the underlying PR (#70 / feature/obstacle-course-followup)

None of the findings above (1-16 + 4 missed) are regressions caused by
the OBS-fix series in PR #70. They are pre-existing surface defects
uncovered by deeper benchmarking. PR #70 remains mergeable as-is. The
follow-up work is a separate scope.

