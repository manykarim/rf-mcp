# Discovery tools improvement user stories (OBS-18..OBS-32)

**Source**: `docs/benchmarks/2026-05-18_discovery_tools_evaluation.md` after
three review rounds (initial report → Codex CLI round 1 [sandbox-limited]
→ Codex CLI round 3 [full filesystem access, all verdicts verified]).
**Date**: 2026-05-18
**Status**: stories filed, not yet implemented
**Predecessors**: OBS-01..09 (PR #69), OBS-10..14 (proposed), OBS-15..17 (PR #70)

These stories cover the **15 prioritised improvements** to `find_keywords`
and `get_keyword_info` surfaced by the evaluation benchmark. Each story
captures the exact code citations, acceptance criteria a re-runnable
benchmark would test, and a task decomposition that should fit a single
PR per story (small enough to ship independently).

The round-3 prioritisation ordering is preserved in this file. Stories
appear top-to-bottom in implementation order.

## Story index

| ID | Title | Type | Priority | Effort | Source |
|---|---|---|---|---|---|
| OBS-18 | Matcher reranking — fix top-match quality | Bug | **High** | L | Finding 1 (corrected) |
| OBS-19 | `get_keyword_info(mode="keyword")` must honour `session_id` | Bug | **High** | S | Missed-1 |
| OBS-20 | Discovery filter falls back to imported libraries | Bug | **High** | M | Finding 13 |
| OBS-21 | Wire `_externalize_response` into `get_keyword_info` + rules | Bug | High | S | Finding 3 (corrected) |
| OBS-22 | Apply `limit` parameter in semantic strategy | Bug | High | S | Finding 11 (corrected) |
| OBS-23 | Session strategy: honour `query`+`limit`, unify schema, externalise | Bug | Med | M | Finding 12 |
| OBS-24 | Empty-query short-circuit for semantic + pattern | UX | Med | XS | Finding 8 |
| OBS-25 | Hard-cap unscoped catalog dump | UX | Med | S | Finding 7 (corrected) |
| OBS-26 | Unscoped fuzzy/typo suggestions in `get_keyword_info` | UX | Med | S | Finding 6 (narrowed) |
| OBS-27 | Externalise pattern `results` + session payload fields | UX | Med | XS | Missed-2 + Missed-3 |
| OBS-28 | Use `library_filter` diagnostics in `recommendations` prose | UX | Low | XS | Finding 2 (narrowed) |
| OBS-29 | Truncate verbose `arguments` field across all branches | UX | Low | S | Finding 9 (corrected) |
| OBS-30 | Honest `strategy="semantic"` docstring + optional extra | Docs | Low | S | Finding 15 + M2 |
| OBS-31 | Pattern `pattern_scope="name"` parameter | UX | Low | S | Finding 4 |
| OBS-32 | LibDoc-fallback ambiguity collapse fix | Bug | Low | XS | Finding 14 |
| OBS-33 | `strict_library=True` filter mode | UX | Med | S | Codex R2 scope gap |

**Round-2 changes** (Codex CLI story review, 2026-05-18):
- OBS-18 split into **OBS-18A** (design) + **OBS-18B** (implement).
- OBS-20 rewritten — original AC #5 was IMPL-INFEASIBLE; now mirrors
  the asymmetric Browser/SL execute-time rules.
- OBS-23 split into **OBS-23A** (honour query+limit, backward-compat)
  + **OBS-23B** (schema unification — needs design doc first).
- OBS-19 / OBS-26 / OBS-29 / OBS-31 tightened with explicit
  algorithm/field/glob-semantics decisions.
- OBS-33 added (was missing from the original story file —
  represents the evaluation report's `strict_library` proposal).
- Implementation order updated to reflect dependencies.

Complementary chore: re-run benchmark with the missing scenarios
identified by Codex (pattern+session, catalog+session+no-lib,
get_keyword_info+session, BDD-prefixed pattern, cross-strategy
same-query comparisons). Filed at the end of this doc as a
methodology task, not a separate story.

---

## OBS-18 — Matcher reranking: fix top-match quality

**Type**: Bug · **Priority**: High · **Effort**: L · **Status: SPLIT (see OBS-18A / OBS-18B below)**

> Round-2 Codex verdict: WRONG-SCOPE — bundles new action taxonomy +
> reranker + confidence calibration in one story; AC #2 permits two
> different outcomes. Split below.

## OBS-18A — Matcher reranking: design phase

**Type**: Investigation · **Priority**: High · **Effort**: M

### Background

Same defect data as OBS-18 (now-deprecated single story above). The
implementation has too many open design questions to be a single PR.
The design phase produces:

- A concrete action-class taxonomy (current `_classify_action` at
  `keyword_matcher.py:97` collapses `select`→`click`, `verify`→broad
  heuristics — need a richer set: `{click, fill, navigate, select,
  assert, wait, query, control, unknown}`).
- A keyword-side classifier — for each loaded keyword, derive its
  action class from tags + name patterns. Browser library tags
  (`Setter`, `Getter`, `PageContent`, `BrowserControl`,
  `Assertion`) give strong signal; SL doesn't expose the same tags
  uniformly so SL classification needs a fallback heuristic.
- The reranker formula (down-weight strength, when to apply, when to
  abstain).
- The confidence-cap rule for class-divergent top-3.

Deliverable: design doc at `docs/proposals/matcher_reranker_design.md`
covering taxonomy, classifier rules, reranker formula, confidence cap,
and rollback plan. No code changes in this story.

### Acceptance criteria

1. Action-class taxonomy table includes ≥ 8 classes with concrete
   keyword examples for each (Browser, SeleniumLibrary, BuiltIn).
2. Classification rules for keywords are reproducible: given a keyword
   name + tag list, the design specifies the resulting class
   deterministically.
3. Reranker formula is parameterised (down-weight factor configurable)
   so the implementation can be tuned post-merge.
4. The design names the exact S02 / S10 outcomes (single required
   value, not "A or B"):
   - S02 (`select dropdown option by visible label`, SL): top match
     MUST be `Select From List By Label`.
   - S10 (`send http post request with json body`, Browser): top
     match MUST have confidence ≤ 0.5 OR the response MUST signal
     "no high-confidence match" (define which).
5. Rollback plan: feature flag the reranker behind
   `ROBOTMCP_MATCHER_RERANK` (default off until OBS-18B lands).

### Tasks

1. Survey current keyword tag coverage in loaded libraries (one-time
   script).
2. Draft taxonomy + classifier rules + reranker formula.
3. Draft confidence-cap rule.
4. Land design doc.

## OBS-18B — Matcher reranking: implementation

**Type**: Bug · **Priority**: High · **Effort**: M · **Blocked by OBS-18A**

### Background

Implementation of the design from OBS-18A. Single AC: post-implementation,
S02 and S10 produce the outcomes specified in the design doc.

### Acceptance criteria

1. S02 top match is `Select From List By Label` (per OBS-18A AC #4a).
2. S10 satisfies the design's chosen behaviour (per OBS-18A AC #4b).
3. New unit tests `tests/unit/test_matcher_action_class_reranker.py`
   cover the reranker, classifier, and confidence cap in isolation.
4. Existing semantic benchmark scenarios that produce correct top
   matches today (S01, S06, S07, S12, S13) continue producing the
   same top matches.
5. Reranker is behind `ROBOTMCP_MATCHER_RERANK` flag, default ON
   after this story merges (flip the OBS-18A default).
6. Benchmark report regenerated; S02 and S10 entries updated.

### Tasks

1. Implement `_classify_keyword_action_class` per the design.
2. Implement `_apply_action_class_reranker` per the design.
3. Implement the confidence cap.
4. Wire into `discover_keywords` post-`_rank_matches`.
5. Add the env-var flag with sane default flip.
6. Unit + benchmark coverage.

### Out of scope (both OBS-18A and OBS-18B)

- Embedding-based reranking (separate; needs `sentence-transformers`
  per OBS-30).
- Cross-library "wrong domain" hints (separate story; needs domain
  classifier).
- OLD: Original "stop-word down-weighting" idea — Codex round 3
  showed it wouldn't solve the problem.

## OBS-18 (legacy single-story description — DEPRECATED, replaced by OBS-18A+OBS-18B above)

**Type**: Bug · **Priority**: High · **Effort**: L

### Background

The 2026-05-18 discovery evaluation surfaced systemic matcher quality
issues: against a SeleniumLibrary session, the query
*"select dropdown option by visible label"* ranks
`Element Should Be Visible` as #1 at confidence 0.87 while the correct
answer `Select From List By Label` lands at #2 at 0.82. Against a
Browser session with `"send http post request"`, top match is
`New Persistent Context` at 0.72 — completely off-domain.

Initial root-cause analysis (round 2) blamed "token overlap" and proposed
stop-word down-weighting. Codex round-3 review traced the actual mechanism:
`KeywordMatcher` uses `difflib.SequenceMatcher` over keyword name + doc
plus action-class seeding (`keyword_matcher.py:372-399`, `449-569`). The
issue is not stop-word weighting — it is that the action classifier is
coarse and there is no post-ranking reranker that penalises keywords
whose action-class doesn't match the query's intent.

Additionally: `sentence-transformers` is not installed in the default
deployment (and not in `pyproject.toml`), so the embedding similarity
branch (`keyword_matcher.py:411+`) is dead code in production. The
"semantic" strategy in default deployments is hybrid pattern+tag+
sequence-matching — not embedding-based.

### User story

> **As an** agent trying to discover the right keyword for a clearly-
> articulated task ("select dropdown option by label"),
> **I want** the top-ranked match to be the keyword that performs that
> action,
> **so that** I don't have to scan the full top-10 list and infer which
> result is actually relevant. Wrong top matches mislead agents into
> writing code that fails at execute_step or silently does the wrong
> thing.

### Acceptance criteria

1. The `S02` scenario benchmark
   (`query="select dropdown option by visible label"`, library=SL)
   returns `Select From List By Label` as the #1 match. Recommendation
   "Best match: ..." names this keyword.
2. The `S10` scenario benchmark
   (`query="send http post request with json body"`, library=Browser)
   produces either: (a) the correct cross-library answer if "wrong
   library" hint is implemented (OBS to follow), OR (b) an explicit
   "no high-confidence Browser match" signal — NOT a 0.72-confidence
   off-domain hit.
3. New unit tests pin the action-class reranker: a contrived
   keyword fixture where the SequenceMatcher score favours an
   irrelevant keyword but the action-class match favours the correct
   one. Test asserts the correct one wins.
4. Existing semantic benchmark scenarios that produce correct top
   matches today (S01, S06, S07, S12, S13) continue producing the
   same top matches after the change (no regression).

### Implementation notes

- Reranker logic should live in `keyword_matcher.py`, after
  `_rank_matches` (line ~530-569). New method
  `_apply_action_class_reranker(ranked, normalized_action, action_type)`
  that down-weights matches whose action-class is incompatible with the
  classified intent (e.g., query classified as `select`, but match is
  classified as `assertion` → down-weight 0.4×).
- `_classify_action` (line ~282) returns the intent class for the
  query. Need a sibling method that classifies the keyword itself
  by inspecting its `tags` and name patterns. Browser library
  tags like `Setter`, `Getter`, `PageContent` give strong signal.
- Confidence cap: when top-3 matches span ≥ 3 distinct action
  classes, cap top-match confidence at 0.75 (the matcher is uncertain
  and should signal that to the agent).
- DO NOT touch the embedding path (`_semantic_matching`,
  line 411-440) since it's dead code without `sentence-transformers`.

### Tasks

1. Add `_classify_keyword_action_class(keyword_info)` returning one of
   `{click, fill, navigate, select, assert, wait, query, control,
   unknown}`. Use Browser tag conventions where present.
2. Add `_apply_action_class_reranker(matches, query_action_type)` that
   down-weights mismatched matches.
3. Wire reranker into `discover_keywords` after `_rank_matches`.
4. Add confidence cap when top-3 matches are class-divergent.
5. Add unit tests `tests/unit/test_matcher_action_class_reranker.py`
   pinning S02 and S10 outcomes.
6. Re-run `scripts/benchmark_discovery_tools.py`, diff against pre-fix
   summary, assert no regression on S01/S06/S07/S12/S13.

### Out of scope

- Embedding-based reranking (OBS-30 covers the docstring honesty around
  embeddings).
- Cross-library "wrong domain" hints (filed as a separate concern; needs
  domain classifier).
- The original "stop-word down-weighting" idea — Codex showed it
  wouldn't solve the problem.

---

## OBS-19 — `get_keyword_info(mode="keyword")` must honour `session_id`

**Type**: Bug · **Priority**: High · **Effort**: S

> Round-2 Codex verdict: DEPENDENCIES — AC #2 requires Browser-plugin
> alternative-hint output that currently only exists in execution
> validation (`browser_plugin.py:302+`). The task list adds
> `allowed_libraries` but doesn't explain plugin hint reuse.
>
> Resolution: AC #2 reworded to make plugin hint reuse explicit, and
> a new task added to extract `validate_keyword_for_session`'s hint
> shape into a shared helper that both execute_step and
> get_keyword_info can call.

### Background

Direct repro:
```python
get_keyword_info(mode="keyword", keyword_name="Click",
                 session_id="dummy-nonexistent-session")
# → success: True, returns Browser.Click documentation
```

The `session_id` parameter is accepted at the API surface
(`server.py:5139`), advertised in the docstring (`server.py:5148`),
but never consulted by the keyword/global branch
(`server.py:5160-5165`). The tool happily documents keywords that are
not loaded in the live session, leading agents to construct calls that
fail at `execute_step` with "Keyword not found".

This is the same class of silent-parameter-ignore defect as the
original OBS library_name bug we just fixed in PR #70.

### User story

> **As an** agent inside a Browser-only session,
> **I want** `get_keyword_info(keyword_name="Click Element")` to return
> "not found" (or a suggestion to use Browser's `Click` instead),
> **so that** my session-aware discovery path doesn't mislead me into
> writing SeleniumLibrary calls that will fail at execute time.

### Acceptance criteria

1. `get_keyword_info(mode="keyword", keyword_name="Click", session_id=<browser-session>)`
   returns Browser's `Click` doc (success path unchanged).
2. `get_keyword_info(mode="keyword", keyword_name="Click Element", session_id=<browser-session>)`
   where Browser-only session does NOT have SL imported, returns
   `success: False` with an SL→Browser alternative hint. The hint
   shape is generated by a NEW shared helper
   `LibraryPluginManager.format_keyword_mismatch_hint(plugin, session,
   keyword_name, source_library)` that wraps the existing
   `browser_plugin.py:323-360` logic and can be called from both
   `execute_step` and `get_keyword_info` paths. (Replaces ad-hoc
   duplication.)
2b. Ambiguous keyword names within the allowed session libraries
   (e.g., `Go To` exists in both Browser and SL, both imported)
   return the `matches[]` array (mirroring the LibDoc path at
   `execution_coordinator.py:1067-1090`), NOT a collapsed single
   result. Pin the array length and library labels.
3. `get_keyword_info(mode="keyword", keyword_name="Click",
   session_id="nonexistent-session-id")` returns `success: False` with
   error `"Session 'nonexistent-session-id' not found"`. No silent fall
   through to global lookup.
4. When `session_id` is omitted (current API surface), behaviour is
   unchanged — global lookup against all loaded libraries.
5. The docstring at `server.py:5148` clearly states that
   `session_id` restricts the search to libraries imported in that
   session.

### Implementation notes

- The session library list lives on `ExecutionSession.imported_libraries`.
- Build a precedence resolver mirroring OBS-15's
  `effective_library_preference` resolver: if `session_id` is set,
  derive `allowed_libraries` from the session; pass into
  `_get_keyword_documentation_payload` as a filter.
- `execution_coordinator.get_keyword_documentation` (line 1021) needs
  to accept an optional `allowed_libraries: List[str]` parameter and
  filter `get_keywords_documentation_all` results by it.
- Symmetry: `mode="parse"` and `mode="session"` should already consult
  session state — verify those paths and document any gaps as
  follow-up.

### Tasks

1. Add `allowed_libraries: Optional[List[str]]` parameter to
   `execution_coordinator.get_keyword_documentation`.
2. Resolve `allowed_libraries` from session in `_get_keyword_documentation_payload`
   when `session_id` is provided.
3. Update the `get_keyword_info` docstring to reflect session scoping
   semantics.
4. Unit tests `tests/unit/test_get_keyword_info_session_scoping.py`:
   - session_id + keyword in session → returns the doc
   - session_id + keyword NOT in session → returns not-found + hint
   - session_id + nonexistent session → returns clear error
   - no session_id → unchanged global lookup
5. Integration test: real Browser session, lookup SL-only keyword,
   assert not-found.

### Out of scope

- BDD-prefix stripping in `get_keyword_info` (separate issue — see
  Missed-4 in the evaluation report).
- `mode="library"` library_name filtering (already strict per
  `server.py:5168`).

---

## OBS-20 — Discovery filter falls back to imported libraries (symmetry with execute)

**Type**: Bug · **Priority**: High · **Effort**: M · **Status: REWRITTEN**

> Round-2 Codex verdict: IMPL-INFEASIBLE — the original AC #5
> ("both imported → no filter") conflicts with the actual Browser
> execute-time rule at `browser_plugin.py:314-321`. Re-read the code:
> Browser plugin rejects SL keywords whenever Browser is in
> imported_libraries (regardless of whether SL is also imported).
> SeleniumLibrary plugin at `selenium_plugin.py:203-205` is
> asymmetric: it only rejects Browser when
> `explicit_library_preference.startswith("selenium")` — does NOT
> use the imported-libraries fallback.
>
> The rewritten story below mirrors the asymmetric execute-time
> behaviour rather than inventing a clean XOR rule that doesn't match
> production.

### Background

`_filter_keywords_by_session_library` (server.py:2037) only applies the
plugin's incompatibility filter when `session.explicit_library_preference`
is non-None. Meanwhile, the Browser plugin's execution-time validation
(`browser_plugin.py:314-321`) ALSO falls back to imported libraries when
no explicit preference exists:

```python
# browser_plugin.py:314-321
pref = (getattr(session, "explicit_library_preference", "") or "").lower()
if pref and pref != "browser":
    return None
if not pref:
    imported = getattr(session, "imported_libraries", []) or []
    if "Browser" not in imported:
        return None
```

Result: an agent in a session created with `libraries=["Browser"]` but
no explicit preference set will see SeleniumLibrary keywords in
`find_keywords` results, pick one, and have it rejected at
`execute_step`. Same tools, asymmetric filtering rules.

Codex correctly narrowed this to a plugin-specific defect — only the
Browser plugin has the imported-fallback. The fix should either (a)
mirror plugin-specific fallback logic in the discovery filter, or (b)
ask each plugin to declare its own fallback policy and consult it.

### User story

> **As an** agent in a Browser-only session that was initialised
> without an explicit library preference,
> **I want** `find_keywords` to surface only Browser-compatible
> keywords (matching what `execute_step` will accept),
> **so that** I don't pick a keyword from discovery only to have it
> rejected at execution.

### Acceptance criteria

1. Create a session with `libraries=["Browser"]`, no
   `explicit_library_preference` set. Run
   `find_keywords(strategy="semantic", session_id=<that session>,
   query="click button")`. Response does NOT include
   `SeleniumLibrary.Click Button` in matches.
2. Same session, run `execute_step(keyword="Click Element", ...)`.
   Response is the existing "Keyword is from SeleniumLibrary..."
   rejection. (Pinning that execute behaviour is unchanged.)
3. The discovery filter and execute-time filter agree on the set of
   allowed keywords for any session — pinned by a paired
   integration test that exercises both tools on the same session and
   asserts the symmetric difference is empty.
4. **Mixed-import case** (Browser + SeleniumLibrary BOTH imported, no
   explicit preference): per the actual rule at
   `browser_plugin.py:312-329`, SL keywords are STILL rejected at
   execute time when Browser is imported. The discovery filter must
   mirror this — SL keywords excluded from discovery output. (This
   replaces the original wrong-but-clean AC #5.)
5. **Asymmetric case** (SL imported, Browser not, no explicit
   preference): per `selenium_plugin.py:203-205`, SL plugin does NOT
   trigger the imported-libraries fallback — Browser keywords are
   NOT rejected at execute time. Discovery must mirror this — Browser
   keywords stay visible.
6. When `explicit_library_preference` IS set, behaviour is unchanged
   (no regression on the existing path).
7. BuiltIn / Collections / String / DateTime / OperatingSystem /
   Process keywords (the "neutral" set used at
   `keyword_discovery.py:264`) are NEVER excluded by this filter —
   they remain visible alongside the active UI library.

### Implementation notes

The asymmetry is real and intentional in the codebase. The fix must
preserve it. Two approaches:

**Approach A (mirror per-plugin)**: discovery filter consults each
plugin's `validate_keyword_for_session` rule. Slow (calls the validator
for every keyword) but trivially consistent. Used for the
integration-test path; might be too slow for catalog-scale discovery.

**Approach B (resolver hardcoded asymmetric)**: build a resolver
helper that mirrors the actual rules:

```python
def _resolve_session_effective_library(session) -> Optional[str]:
    pref = (getattr(session, "explicit_library_preference", "") or "").lower()
    if pref:
        return pref  # explicit wins
    imported = set(getattr(session, "imported_libraries", []) or [])
    # Browser is the only plugin with imported-fallback (browser_plugin.py:312-321)
    if "Browser" in imported:
        return "Browser"
    # SL plugin has no imported-fallback (selenium_plugin.py:203-205)
    return None
```

Add a TODO in the resolver pointing at the asymmetry so future
maintainers know it intentionally diverges. If the SL plugin ever
adopts an imported-fallback, the resolver gets a symmetric branch.

Plugin-API refactor (each plugin declares its own
`get_implicit_preference_rule()`) is a larger follow-up. Keep
hardcoded for OBS-20.

### Tasks

1. Add `_resolve_session_effective_library(session)` helper in
   server.py near the existing resolver.
2. Wire it into `find_keywords` so discovery uses the same resolved
   preference as execution.
3. (Optional, larger): plumb a plugin API for declaring fallback rules.
   Keep behind feature flag.
4. Unit tests `tests/unit/test_discovery_execution_filter_symmetry.py`:
   - Browser imported, no preference → SL excluded from discovery
   - SL imported, no preference → Browser excluded
   - Both imported → no filter
   - Neither imported → no filter
5. Integration test in `tests/integration/`: real Browser session
   without explicit preference, lookup SL keyword via `find_keywords`,
   assert absent.

### Out of scope

- Generalizing to all plugins (AppiumLibrary, RequestsLibrary, etc.).
  Browser/SL XOR covers the documented incompatibility table.
- Changing execution-time validation. The fix is to make discovery
  consistent with execution, not the other way around.

---

## OBS-21 — Wire `_externalize_response` into `get_keyword_info` + add rules

**Type**: Bug · **Priority**: High · **Effort**: S

> Round-2 Codex verdict: READY with one small inconsistency — tasks
> mention the session branch but the proposed rules only cover
> `library.*` and `keyword.doc`. Session-mode (`mode="session"`)
> payloads are small and don't need externalisation. Resolution:
> remove the misleading "session" mention from the tasks; session
> mode stays inline. The session externalisation we DO need is the
> `find_keywords(strategy="session")` payload, which is in OBS-23B.

### Background

`get_keyword_info(mode="library", library_name="Browser")` returns
71,521 inline tokens (full Browser libdoc dump: 36k-char doc string +
147 keywords with full per-keyword docs). The benchmark identified
this as the worst single-call token cost in the matrix.

Initial fix proposal (round 2): add field-path rules to `DEFAULT_RULES`
in `artifact_output/services.py`. Codex round 3 traced the actual code
and found this is insufficient: **`get_keyword_info` never calls
`_externalize_response` on any branch** (`server.py:5160-5184`).
Externalisation is opt-in per call site; adding rules without wiring
the call site is a no-op.

Two-part fix required:
1. Wire `_externalize_response("get_keyword_info", session_id, result)`
   into each `get_keyword_info` branch that returns large payloads.
2. Add field-path rules.

### User story

> **As an** agent that wants library-level documentation as a
> recovery path (e.g., after a keyword-not-found error),
> **I want** the response to be a compact summary with a fetch path
> for the full doc,
> **so that** one library-mode call doesn't consume ~70k tokens of my
> context budget for content I may only need to skim.

### Acceptance criteria

1. `get_keyword_info(mode="library", library_name="Browser",
   session_id=<any session>)` returns an inline payload <2000 tokens.
   The full library doc + keyword list lives in an artifact file
   referenced by a `result: "Content saved to ..."` string.
2. When `session_id` is omitted, externalisation does not fire
   (consistent with current `find_keywords` behaviour). Document this
   in the docstring.
3. `get_keyword_info(mode="keyword", keyword_name=<verbose-docstring-keyword>,
   session_id=<...>)` externalises when `keyword.doc` exceeds the
   threshold. Most keyword-mode responses (<500 tokens) do NOT
   externalise — preserves the common case.
4. `mode="parse"` and `mode="session"` payloads continue inline (small
   responses).
5. The artifact summary uses the existing
   `FILE_PATH_SUMMARY_TEMPLATE` shape so it's parseable by callers
   already aware of the `find_keywords` externalisation pattern.

### Implementation notes

- Add `_externalize_response("get_keyword_info", session_id, result)`
  call at end of each `get_keyword_info` branch when `session_id` is
  available. Mirror the conditional gate from `find_keywords`.
- Add three rules to `DEFAULT_RULES` in
  `artifact_output/services.py`:
  ```python
  ExternalizationRule(tool_name="get_keyword_info", field_path="library.doc"),
  ExternalizationRule(tool_name="get_keyword_info", field_path="library.keywords"),
  ExternalizationRule(tool_name="get_keyword_info", field_path="keyword.doc"),
  ```
- Inline summary fields that should remain visible even when doc is
  externalised: `name`, `library`, `short_doc`, `args`, `tags`,
  `is_deprecated`. The 50-100-char `short_doc` is far more useful
  inline than the full `doc`.

### Tasks

1. Wire `_externalize_response` call into `get_keyword_info` library /
   keyword / session branches (conditional on session_id).
2. Add the three field-path rules to `DEFAULT_RULES`.
3. Update the `get_keyword_info` docstring to mention externalisation.
4. Unit test in `tests/unit/test_get_keyword_info_externalization.py`:
   - mode="library" + session_id → result is an artifact link
   - mode="library" + no session_id → result is the full doc (current
     behaviour, kept for backwards compat / debugging)
   - mode="keyword" with small doc → inline
   - mode="keyword" with large doc → externalised
5. Benchmark re-run: K06 inline_tokens should drop from 71521 to
   <2000.

### Out of scope

- Adding externalisation for keyword-mode small payloads (already
  cheap).
- Changing the `FILE_PATH_SUMMARY_TEMPLATE` (uniform across tools).

---

## OBS-22 — Apply `limit` parameter in semantic strategy

**Type**: Bug · **Priority**: High · **Effort**: S

### Background

`find_keywords(strategy="semantic", limit=20, ...)` silently caps at
10 because:
- `server.py:2300-2305` parses `limit` into `limit_value`
- `server.py:2408` (pattern) and `server.py:2477` (catalog) apply
  `matches = matches[:limit_value]`
- The semantic branch (server.py:~2356) has **no equivalent line**
- `keyword_matcher.discover_keywords` hard-caps internally at 10
  (`keyword_matcher.py:307`: `top_matches = ranked_matches[:10]`)

Initial fix proposal (round 2): apply `discovery["matches"] =
discovery["matches"][:limit_value]` post-filter. Codex round 3 verified
this doesn't honour `limit > 10` because the matcher already capped.

Correct fix is two-part: (a) thread limit into `discover_keywords` so
the matcher can return up to that many, (b) apply post-filter slice
so recommendations align with the returned list.

### User story

> **As an** agent that wants more than 10 semantic matches (e.g.,
> exploring all possible keywords related to "click"),
> **I want** `limit=20` to actually return up to 20 ranked matches,
> **so that** I don't silently lose the bottom half of the ranking
> below the hard cap.

### Acceptance criteria

1. `find_keywords(strategy="semantic", query="click", limit=20)` returns
   up to 20 matches (if 20 exist above the relevance floor).
2. `find_keywords(strategy="semantic", query="click", limit=3)` returns
   exactly 3 matches.
3. Omitting `limit` keeps the current default of 10.
4. `recommendations[0]` ("Best match: ...") names the top match of the
   *returned* (post-limit, post-filter) list — not a match that was
   sliced off.
5. The limit applies BEFORE the filter pass — so if `limit=20` and 5
   are filtered out, the agent gets 15 (not 20), matching the
   contract "up to N matches survived filtering".

### Implementation notes

- `KeywordMatcher.discover_keywords` (line 254) takes an action
  description, context, current_state. Add `limit: Optional[int] = None`
  parameter. Default falls back to current `top_matches = ranked_matches[:10]`;
  when set, use `ranked_matches[:limit]`.
- In `find_keywords` semantic branch (server.py:~2356), pass
  `limit=limit_value` into `discover_keywords` call.
- Apply post-filter trim after `_filter_keywords_by_session_library`
  to keep recommendations in sync.
- Rebuild recommendations against the trimmed list (already done by
  the OBS-15 rebuild path, just verify it picks up the new shape).

### Tasks

1. Add `limit` parameter to `KeywordMatcher.discover_keywords`.
2. Thread `limit_value` through `find_keywords` semantic branch.
3. Apply post-filter trim if `limit_value < len(discovery["matches"])`.
4. Unit test in `tests/unit/test_find_keywords_limit.py`:
   - `limit=3` returns 3 matches
   - `limit=20` returns up to 20 matches
   - `limit` omitted returns up to 10 (default unchanged)
   - filter excludes some → final count ≤ limit
   - recommendations[0] matches matches[0]
5. Re-run benchmark; pin scenario S01 with explicit `limit=3` to ≤ 3.

### Out of scope

- `limit` for pattern/catalog (already works).
- `limit` for session strategy (covered by OBS-23).

---

## OBS-23 — Session strategy: honour `query` + `limit`, unify schema, externalise

**Type**: Bug · **Priority**: Med · **Effort**: M · **Status: SPLIT**

> Round-2 Codex verdict: WRONG-SCOPE — bundles four changes (query,
> limit, schema unification, externalisation). AC #3 left the
> backward-compat decision ambiguous. Split into OBS-23A (cheap,
> backward-compat) and OBS-23B (schema redesign, requires its own
> design doc).

## OBS-23A — Session strategy: honour `query` + `limit`

**Type**: Bug · **Priority**: Med · **Effort**: S

### Acceptance criteria (backward-compat preserving)

1. `find_keywords(strategy="session", session_id=<...>, query="click",
   limit=5)` returns at most 5 keywords whose name contains "click"
   (case-insensitive substring on `name`, not `full_name`).
2. Empty `query` returns the full namespace, trimmed to `limit`
   (default unchanged: no limit when omitted, matching current
   behaviour where the namespace is unbounded).
3. Response shape is UNCHANGED from current
   (`{success, libraries_count, library_keywords, resource_keywords}`).
   Schema unification deferred to OBS-23B.
4. `library_keywords` AND `resource_keywords` both honour the filter
   (substring + limit applied to the union, then split back into the
   two arrays).

### Implementation notes

- Add `name_filter: Optional[str] = None` and
  `limit: Optional[int] = None` to
  `RobotFrameworkNativeContextManager.list_available_keywords`
  (`rf_native_context_manager.py:1641`).
- Filter in the manager BEFORE building the
  `library_keywords` / `resource_keywords` split.
- `find_keywords` session branch passes `query` and `limit_value`.

### Tasks

1. Add params + filter logic to `list_available_keywords`.
2. Pass through from `find_keywords` session branch.
3. Unit tests: query filter applied; limit applied; default behaviour
   unchanged.
4. Document in docstring that `query` is name-only substring.

## OBS-23B — Session strategy: schema unification (requires design)

**Type**: Investigation + Bug · **Priority**: Low · **Effort**: M · **Blocked by OBS-23A**

> Round-2 Codex verdict: pre-decide the backward-compat contract
> before writing code. The proposal "merge library_keywords +
> resource_keywords under result.matches" is API-breaking. Need a
> design doc covering deprecation path.

### Investigation deliverable

`docs/proposals/session_strategy_schema.md` covering:

1. Survey of who currently reads `library_keywords` /
   `resource_keywords` (grep across rf-mcp + any known consumer).
2. Proposed unified shape: `{success, strategy, query, result:
   {matches: [...], recommendations: [...]}, library_count,
   resource_count}` — same shape as semantic/pattern.
3. Migration path: dual-emit (both shapes for one version) →
   deprecate old shape → remove. Pin the rollout windows.
4. Externalisation impact: under unified shape, the existing
   `field_path="result"` rule covers session strategy too (eliminates
   the need for separate session-field externalisation rules — see
   round-2 Codex note about OBS-27 overlap).

### Implementation (after design accepted)

1. Implement dual-emit shape in `find_keywords` session branch.
2. Add deprecation warning when callers explicitly request the old
   shape via env var or header (TBD by design).
3. Cross-strategy contract tests pin all four strategies return the
   same outer shape (`success`, `strategy`, `query`, `result`).
4. Once design is accepted, OBS-27's "session payload externalisation"
   becomes redundant (the existing `result` rule covers it).

### Original OBS-23 description (DEPRECATED — replaced by OBS-23A + OBS-23B above)

### Background

`find_keywords(strategy="session")` has three independent defects:

1. **`query` is decorative** (`server.py:2539`). Echoed in response but
   never consulted. `list_available_keywords(session_id)` takes only
   the session_id.
2. **`limit` ignored** for the same reason — no post-trim.
3. **Schema mismatch**: returns `{success, libraries_count,
   library_keywords, resource_keywords}` (`rf_native_context_manager.py:
   1678-1683`), NOT the
   `{success, strategy, query, result: {matches, recommendations, ...}}`
   shape that semantic / pattern / catalog return. Agents reading
   `result.matches` get nothing.
4. **Externalisation never triggers**: only the `result` field path
   has a rule (`services.py:60`). Session strategy's
   `library_keywords` / `resource_keywords` fields are not covered, so
   even with `session_id` the payload goes inline.

A session with Browser (200+ kw) + BuiltIn (100+ kw) imported produces
~300 keyword entries inline.

### User story

> **As an** agent that wants to browse only the keywords actually
> available in my live session (post-import, post-resource-loading),
> **I want** `find_keywords(strategy="session", query="click",
> limit=5)` to filter and trim the namespace dump consistently with
> the other strategies,
> **so that** I get the same compact, filterable shape regardless of
> which strategy I picked.

### Acceptance criteria

1. `find_keywords(strategy="session", session_id=<...>, query="click",
   limit=5)` returns ≤ 5 keywords whose name contains "click"
   (case-insensitive).
2. Empty `query` returns the full namespace, trimmed to `limit`
   (default unchanged behaviour for backwards compat — but with limit
   now applied).
3. Response shape unifies with the other strategies: a top-level
   `result` field containing `{matches: [...], recommendations: [...]}`.
   `library_keywords` / `resource_keywords` may stay as siblings for
   backwards compat OR be merged into `matches` — recommend merging
   under one key.
4. Large session dumps externalise via `_externalize_response` using
   field paths that cover the new shape.
5. `recommendations[0]` names the top match from the trimmed list, or
   a "no matches" message when the filter excludes everything.

### Implementation notes

- Modify `RobotFrameworkNativeContextManager.list_available_keywords`
  (`rf_native_context_manager.py:1641`) to accept optional
  `name_filter: Optional[str]` and `limit: Optional[int]` parameters.
  Apply substring matching on `kw.name` like the pattern path does.
- In `find_keywords` session branch (`server.py:2531-2546`), pass
  `query` as `name_filter` and `limit_value` as `limit`.
- Wrap the response into the unified shape:
  ```python
  payload = {
      "success": result.get("success", True),
      "strategy": "session",
      "query": query,
      "result": {
          "matches": [
              {"keyword_name": kw["name"], "library": kw.get("library"), ...}
              for kw in (result.get("library_keywords") or []) +
                        (result.get("resource_keywords") or [])
          ],
      },
  }
  ```
- Add field-path rules covering the new shape:
  ```python
  ExternalizationRule(tool_name="find_keywords", field_path="result.matches"),
  ```
  (or keep `result` as the catch-all, which already exists).

### Tasks

1. Add `name_filter` + `limit` params to `list_available_keywords`.
2. Apply substring filter + slice in the manager.
3. Wrap response into unified shape in `find_keywords` session branch.
4. (Optional) Generate `recommendations` from the trimmed matches via
   the existing rebuild helper.
5. Add externalisation field-path rule for new shape (if not already
   covered by the existing `result` rule).
6. Unit tests `tests/unit/test_find_keywords_session_strategy.py`:
   - query="click" + limit=5 → ≤ 5 matches all containing "click"
   - empty query → full namespace, trimmed by limit
   - schema matches semantic/pattern shape
   - large session triggers externalisation when session_id present
7. Integration test: real session with Browser + BuiltIn, exercise
   filtering.

### Out of scope

- Glob support in the session filter (pattern strategy already handles
  globs against the full catalog).
- Tag-based filtering (separate enhancement).

---

## OBS-24 — Empty-query short-circuit for semantic + pattern

**Type**: UX · **Priority**: Med · **Effort**: XS

### Background

`find_keywords(query="", strategy="semantic", library_name="Browser")`
returns 1 match: `Browser.New Persistent Context` at confidence 0.35
with 40 arguments in the response (`~838 tokens` per the benchmark).
The matcher scores every keyword against an empty action description
and ranks them; with no signal, even a 0.35-confidence match passes
the floor.

`pattern` strategy with empty query also misbehaves: the substring
match against `""` succeeds for every keyword, so an unfiltered call
returns the full catalog.

`catalog` strategy correctly handles empty queries (intent matches).

### User story

> **As an** agent that accidentally constructed an empty `query`
> (typo, template error, programmatic miscalculation),
> **I want** the tool to reject the call with a clear error rather
> than return a bogus low-confidence match,
> **so that** I don't write code based on a hallucinated "best match"
> that has no actual relevance.

### Acceptance criteria

1. `find_keywords(query="", strategy="semantic", ...)` returns
   `success: False` with error `"Query string is required for
   strategy='semantic'"` and hint
   `"Use strategy='catalog' to list available keywords without a
   query."`
2. `find_keywords(query="", strategy="pattern", ...)` returns the
   same error shape.
3. `find_keywords(query="   ", strategy="semantic", ...)` (whitespace
   only) treated as empty.
4. `find_keywords(query="", strategy="catalog", ...)` continues
   working as today (catalog supports empty queries by design).
5. `find_keywords(query="", strategy="session", ...)` returns the
   full namespace (after OBS-23 limit is applied) — consistent with
   "show me everything in scope".

### Implementation notes

- Add the guard near the top of `find_keywords` after the BDD-prefix
  strip (server.py:~2298):
  ```python
  if strategy_norm in {"semantic", "intent", "pattern", "search"}:
      if not query or not query.strip():
          return {
              "success": False,
              "strategy": strategy_norm,
              "query": query,
              "error": f"Query string is required for strategy='{strategy_norm}'",
              "hint": "Use strategy='catalog' to list available keywords without a query.",
          }
  ```

### Tasks

1. Add the guard.
2. Unit tests:
   - empty query + semantic → error
   - empty query + pattern → error
   - whitespace-only query → error
   - empty query + catalog → existing behaviour preserved
3. Re-run benchmark scenario S09; assert success=False.

### Out of scope

- Validating non-empty queries (separate concern; queries like "x" are
  technically non-empty).

---

## OBS-25 — Hard-cap unscoped catalog dump

**Type**: UX · **Priority**: Med · **Effort**: S

### Background

`find_keywords(strategy="catalog")` with no `session_id` AND no
`library_name` returns 658 keywords across 10 libraries = 97,212
inline tokens. The existing empty-catalog hint at server.py:~2376
fires only when catalog is actually empty (no libraries loaded), not
when libraries are loaded and the user just hasn't scoped the query.

Initial fix (round 2): externalise `results` + cap. Codex round 3
verified externalisation is gated on `session_id`, so the cap is the
only fix that works in the no-session case.

### User story

> **As an** agent that's exploring available keywords for the first
> time (no session yet),
> **I want** `find_keywords(strategy="catalog")` to either tell me
> how to narrow my query or cap the output at a reasonable size,
> **so that** my first discovery call doesn't burn 90k tokens.

### Acceptance criteria

1. `find_keywords(strategy="catalog")` with no `session_id` AND no
   `library_name` AND empty/no `query` returns ≤ 100 keywords with a
   hint:
   ```
   "Catalog returned NNN keywords across M libraries. Use library_name
    or a query filter to narrow. First 100 returned (use
    strategy='catalog', library_name='Browser' for Browser-only)."
   ```
2. `library_name` OR `query` OR `session_id` provided → existing
   behaviour, no cap.
3. The cap value (100) is configurable via
   `ROBOTMCP_CATALOG_HARD_CAP` env var, default 100.

### Implementation notes

- Add the cap logic in the catalog branch (server.py:~2461) after the
  query filter and library filter pass, BEFORE the limit slice:
  ```python
  if not session_id and not library_name and not query:
      hard_cap = int(os.getenv("ROBOTMCP_CATALOG_HARD_CAP", "100"))
      if len(catalog) > hard_cap:
          truncated = len(catalog)
          libraries = sorted({c["library"] for c in catalog})
          catalog = catalog[:hard_cap]
          result["hint"] = (
              f"Catalog returned {truncated} keywords across "
              f"{len(libraries)} libraries. Use library_name or "
              f"a query filter to narrow. First {hard_cap} returned."
          )
          result["catalog_truncated"] = True
          result["full_catalog_size"] = truncated
  ```

### Tasks

1. Add the cap logic in the catalog branch.
2. Add env var support.
3. Unit tests:
   - unscoped catalog + many keywords → capped + hint
   - scoped catalog → no cap
   - empty catalog (no libraries) → existing empty hint
4. Benchmark scenario S20: inline_tokens < 15000.

### Out of scope

- Externalising the catalog results when session_id IS provided —
  covered by OBS-27.

---

## OBS-26 — Unscoped fuzzy/typo suggestions in `get_keyword_info`

**Type**: UX · **Priority**: Med · **Effort**: S

> Round-2 Codex verdict: TIGHTEN — AC originally said "Levenshtein
> distance" while implementation notes used `difflib.get_close_matches`.
> Those produce different candidate sets. Resolution: pick ONE
> algorithm and threshold. Going with `difflib.get_close_matches`
> because it's stdlib (no new dependency) and the scoped path
> already uses the same pattern at `execution_coordinator.py:1056-1058`.
> AC and implementation now both say `difflib` with `cutoff=0.6`
> (matches stdlib default; tweak only if benchmark shows poor
> recall).

### Background

`get_keyword_info(keyword_name="Clikc", library_name="Browser")`
returns `suggestions: ["Click", "Click With Options"]` thanks to the
library-scoped fuzzy fallback at `execution_coordinator.py:1050-1066`.

But `get_keyword_info(keyword_name="Clikc")` (no library_name)
returns a bare:
```json
{"success": false, "error": "Keyword 'Clikc' not found in any loaded library"}
```
No suggestions. Codex round 3 narrowed the original Finding 6 — the
scoped path already works; only the unscoped path needs fuzzy
fallback.

### User story

> **As an** agent that typo'd a keyword name and doesn't know which
> library it belongs to,
> **I want** the unscoped `get_keyword_info(keyword_name="Clikc")`
> call to suggest "Click" (and similar), with library labels,
> **so that** I can correct the typo without a separate
> `find_keywords` round-trip.

### Acceptance criteria

1. `get_keyword_info(keyword_name="Clikc")` returns
   `success: False` with `suggestions: [{name: "Click", library:
   "Browser"}, {name: "Click With Options", library: "Browser"}, ...]`.
2. Suggestions ranked by `difflib.get_close_matches` with `n=5`,
   `cutoff=0.6`. (Matches the scoped path's algorithm at
   `execution_coordinator.py:1056-1058`. Levenshtein was the wrong
   word in the original draft.)
3. Unknown gibberish (e.g., "XYZNoSuchThing") returns
   no suggestions (distance threshold filters everything).
4. The scoped path's suggestion behaviour is unchanged (regression
   test).

### Implementation notes

- Use `difflib.get_close_matches` against the union of all loaded
  libraries' keyword names. Faster than full Levenshtein and good
  enough for typo recovery.
- Add the fallback in `execution_coordinator.get_keyword_documentation`
  (line ~1090, in the global path "No matches anywhere" branch).
  Mirror the suggestion format the scoped path uses but include
  the library label per entry.
- Suggestion entry shape:
  ```python
  {"name": "Click", "library": "Browser"}
  ```

### Tasks

1. Add `_global_fuzzy_suggestions(keyword_name)` helper in
   `execution_coordinator.py`.
2. Wire into the global-lookup-failed branch.
3. Unit tests:
   - "Clikc" → suggests Click/Click With Options/Click Element with
     library labels
   - "XYZNoSuchThing" → no suggestions
   - "click element" (exact, lowercase) → already routes through
     normal path; no suggestions needed
4. Benchmark scenario K05: ensure suggestions field is populated.

### Out of scope

- Suggestion ranking by library priority (the scoped path doesn't
  do this either — keep it simple).

---

## OBS-27 — Externalise pattern `results` + session payload fields

**Type**: UX · **Priority**: Med · **Effort**: XS

### Background

The only externalisation rule for `find_keywords` covers
`field_path="result"` (singular). But:
- Pattern strategy puts results under `results` (plural) — not covered.
- Session strategy puts results under `library_keywords` and
  `resource_keywords` — not covered.

So even with `session_id` and a large pattern/session response, the
payload goes inline. Pattern `Get*` scenario (S16) shows 5407 inline
tokens for a session-backed call.

### User story

> **As an** agent that runs a pattern search like `"Get*"` in an
> active session,
> **I want** the response to externalise when it exceeds the threshold,
> **so that** I get the same compact inline + artifact link shape
> as `result` externalisation already provides for semantic.

### Acceptance criteria

1. `find_keywords(strategy="pattern", query="Get*", session_id=<...>)`
   when the results exceed 500 tokens → response is the externalised
   form (`results: "Content saved to ..."`). The top-level
   `match_count` + `top_matches` summary fields stay inline (the
   compact agent-facing summary).
2. `find_keywords(strategy="session", session_id=<...>)` with a large
   namespace → `library_keywords` / `resource_keywords` externalised.
3. Small responses stay inline (no regression).

### Implementation notes

- Add three field-path rules to `DEFAULT_RULES`:
  ```python
  ExternalizationRule(tool_name="find_keywords", field_path="results"),
  ExternalizationRule(tool_name="find_keywords",
                      field_path="library_keywords"),
  ExternalizationRule(tool_name="find_keywords",
                      field_path="resource_keywords"),
  ```
- The `_externalize_response` machinery already exists; this is just
  adding rules.

### Tasks

1. Add the three rules.
2. Unit tests pin externalisation triggers for each:
   - pattern + session_id + large results → externalised
   - session strategy + session_id + large namespace → externalised
   - small payloads → unchanged
3. Benchmark scenarios S14 + S16 with session_id added.

### Out of scope

- Externalising the legacy non-session paths (no session_id =
  externalisation disabled by design).

---

## OBS-28 — Use `library_filter` diagnostics in `recommendations` prose

**Type**: UX · **Priority**: Low · **Effort**: XS

### Background

When the library filter excludes all top matches (vague-query
scenario S04: "do something with form" + Browser → 0 surviving
matches), the response contains:
- `library_filter: {count: 10, from_library: "SeleniumLibrary"}`
  (correct diagnostic)
- `recommendations: ["No matching keywords found. Consider: ...
  rephrase, use more specific terms"]` (generic, ignores the
  diagnostic)

Codex round 3 narrowed this: the signal IS in the response; the
issue is purely that the recommendations text doesn't read the
adjacent `library_filter` diagnostic.

### User story

> **As an** agent that got 0 matches after applying a library filter,
> **I want** the `recommendations` prose to tell me that N matches
> were filtered out because they're from another library,
> **so that** I know to either widen my filter or change session
> libraries.

### Acceptance criteria

1. Scenario S04 (`query="do something with form"`, library=Browser,
   matches=0 after filter, library_filter.count=10) returns
   recommendations like:
   ```
   ["10 top matches were excluded by library_name=Browser
     (from SeleniumLibrary).",
    "Either drop library_name to see those matches,
     or switch to a SeleniumLibrary session."]
   ```
2. Scenario with no filter applied (no library_name, matches=0) keeps
   the existing "No matches found" guidance.
3. Filter applied + non-zero surviving matches keeps the
   "Best match: ..." prose (the existing rebuild path).

### Implementation notes

- In `_rebuild_post_filter_recommendations` (server.py:~2098), add a
  branch: when matches is empty AND `library_filter.count > 0`, emit
  the diagnostic-aware prose instead of the generic "no matches found".
- The helper currently signature: `(matches)`. Needs to also see the
  filter context. Either:
  - Pass `library_filter` as a second arg, or
  - Move the call site to after `_build_filter_diagnostics` runs so the
    helper can inspect the resulting `library_filter` dict.

### Tasks

1. Extend the rebuild helper to accept filter context.
2. Add the diagnostic-aware branch.
3. Unit tests:
   - S04 shape → diagnostic prose
   - No-filter + zero-matches → generic prose (unchanged)
   - Filter + some-matches → existing rebuild path
4. Re-run benchmark; visually verify S04 recommendations text.

### Out of scope

- Auto-switching library context (would need session mutation; agent's
  responsibility to act on the suggestion).

---

## OBS-29 — Truncate verbose `arguments` / `args` field across all branches

**Type**: UX · **Priority**: Low · **Effort**: S

> Round-2 Codex verdict: TIGHTEN — story said "all branches" but
> field names differ: semantic uses `arguments` (at
> `keyword_matcher.py:318`), pattern/catalog use `args` (at
> `execution_coordinator.py:888`). Resolution: AC + tasks now
> enumerate the actual field per branch. `arg_types` (parallel field
> in pattern/catalog) is NOT truncated — typically much shorter than
> arg names, and useful for inferring contract.

### Background

`matches[].arguments` contains the full argument-name list for each
keyword. For `Browser.New Persistent Context` (40+ args), this is
~600 tokens per match. The `recommendations: ["Required arguments:
... 40+ names ..."]` line duplicates it. Codex round 3 verified the
issue affects all strategies (semantic, pattern, catalog), not just
semantic.

### User story

> **As an** agent reading a response,
> **I want** the `arguments` field truncated to the top N (default 6)
> with a `+M more` suffix,
> **so that** one verbose keyword doesn't dominate the response token
> budget. The full arg list is one `get_keyword_info` call away.

### Acceptance criteria

1. Semantic strategy: `matches[].arguments` (the list at
   `keyword_matcher.py:318`) capped at 6 entries with a 7th entry
   `"+M more"` when the underlying list is longer.
2. Pattern strategy: `results[].args` (the list at
   `execution_coordinator.py:1001`) capped identically.
3. Catalog strategy: `results[].args` capped identically.
4. `recommendations: "Required arguments: ..."` line mirrors the
   semantic truncation.
5. Keywords with ≤ 6 args show the full list (no `+M more` suffix).
6. A new optional field `argument_count: <int>` surfaces the total
   count so the agent knows the truncation happened (added to each
   match/result dict).
7. `arg_types` field (parallel in pattern/catalog) is NOT truncated —
   it carries type-signature information that's load-bearing for the
   agent and typically short.

### Backward compatibility

This is a response-shape change. Existing callers that read the
full `arguments` / `args` list may break. Document in the changelog;
also surface the full list via `get_keyword_info(mode="keyword")`
unchanged so agents have a deterministic recovery path.

### Implementation notes

- Add a `_truncate_arguments(args: List[str], cap: int = 6) -> Tuple[List[str], int]`
  helper in server.py.
- Apply post-filter in each strategy branch (semantic / pattern /
  catalog), before the response is returned.
- For semantic strategy, also apply to the `recommendations`
  "Required arguments:" line — easier if the rebuild helper does it
  uniformly.
- Make the cap configurable via env var
  `ROBOTMCP_ARGUMENT_CAP`, default 6.

### Tasks

1. Add the truncation helper.
2. Apply it in each strategy branch.
3. Wire into the recommendations rebuild path.
4. Unit tests:
   - Keyword with 40 args → 6 + "+34 more"
   - Keyword with 3 args → full list, no suffix
   - argument_count field populated
   - Recommendations line truncated too
5. Re-run benchmark; assert S09 inline_tokens drops by ~600.

### Out of scope

- Argument type list truncation (different field, similar issue —
  can be a follow-up).

---

## OBS-30 — Honest `strategy="semantic"` docstring + optional extra

**Type**: Docs · **Priority**: Low · **Effort**: S

### Background

`strategy="semantic"` is advertised in the docstring as "Natural
language search (best for exploring)". In default deployments without
`sentence-transformers` installed (and it's not in `pyproject.toml`),
the matcher uses pattern matching + `SequenceMatcher` over keyword
docs + action-class tags — no actual embedding similarity.

Two-track fix per Codex:
- Update the strategy docstring to clarify when embedding similarity
  is active vs hybrid mode.
- Offer `sentence-transformers` as an optional extra so users who want
  true semantic matching can opt in via `uv add robotmcp[semantic]` or
  similar.

### User story

> **As an** agent (or human) reading the `find_keywords` docstring,
> **I want** an honest description of what `strategy="semantic"`
> actually does in default deployments,
> **so that** I don't over-rely on its ranking quality and know that
> for true semantic search I need to install an optional extra.

### Acceptance criteria

1. The `find_keywords` docstring section for `strategy="semantic"`
   reads (approximate):
   ```
   "semantic": Hybrid keyword search combining name/doc pattern
              matching, tag/action-class classification, and (when
              installed) sentence-transformers embedding similarity.
              For best results, install the optional 'semantic'
              extra: `uv add robotmcp[semantic]`.
   ```
2. `pyproject.toml` declares a `semantic` optional dependency group
   listing `sentence-transformers`.
3. Installation docs (README.md or docs/installation.md if it exists)
   mention the optional group.
4. Runtime detection of `sentence-transformers` availability is
   logged at INFO level during keyword matcher initialisation so
   users can see whether embedding mode is active.

### Implementation notes

- The matcher already handles the optional import gracefully
  (`keyword_matcher.py:10-16`); just need to log clearly.
- Optional extras in `pyproject.toml` use the
  `[project.optional-dependencies]` table.

### Tasks

1. Add `sentence-transformers` as `[project.optional-dependencies].semantic`.
2. Update the `find_keywords` docstring.
3. Add INFO log line in matcher init reporting whether embeddings are
   active.
4. Unit test: matcher initialisation log captured + asserted.

### Out of scope

- Renaming the strategy to "hybrid" (would be an API-breaking change;
  filed only as the most aggressive option in the report).
- Bundling `sentence-transformers` as a required dependency (heavy ~2GB
  install due to torch).

---

## OBS-31 — Pattern `pattern_scope="name"` parameter

**Type**: UX · **Priority**: Low · **Effort**: S

> Round-2 Codex verdict: TIGHTEN — AC didn't define what glob means
> for `pattern_scope="docs"` and `pattern_scope="all"` given that
> the current glob logic at `rf_libdoc_integration.py:298` only
> applies to names. Resolution: define glob semantics per scope.

### Background

Pattern strategy substring-matches the query against keyword
**name + doc + short_doc + tags** (`rf_libdoc_integration.py:286-313`).
Query `"Go To"` returns `BuiltIn.Repeat Keyword` as #1 because its
doc contains the phrase "go to". For agents that half-remember a
keyword name and just want to find it, the broad match is noise.

### User story

> **As an** agent that's half-remembering a keyword name,
> **I want** `find_keywords(strategy="pattern", query="Go To",
> pattern_scope="name")` to substring-match the name only,
> **so that** I don't get unrelated keywords whose docs happen to
> contain the phrase.

### Acceptance criteria

1. `find_keywords(strategy="pattern", query="Go To",
   pattern_scope="name")` returns only keywords whose name contains
   "Go To" (case-insensitive substring on `name`).
2. `pattern_scope="all"` (default) preserves current behaviour
   (name + doc + short_doc + tags substring match for plain text;
   name-only glob for wildcard queries — same as
   `rf_libdoc_integration.py:298-311`).
3. `pattern_scope="docs"` matches `doc` + `short_doc` only —
   substring for plain text; glob queries return zero matches with
   a clear error "Glob patterns only supported on names; use
   pattern_scope='name' or 'all' with a glob".
4. `pattern_scope="name"` with a glob query: glob-match against
   names only (existing behaviour at line 304).
5. Invalid `pattern_scope` value returns
   `{success: False, error: "Invalid pattern_scope; expected one of
   'name', 'docs', 'all'"}`.

### Backward compatibility

Default unchanged (`pattern_scope="all"`). Callers that don't pass
the parameter see no behaviour change. Document in the
`find_keywords` docstring.

### Implementation notes

- Add `pattern_scope: Literal["name", "docs", "all"] = "all"` parameter
  to `find_keywords` tool surface.
- Extend `rf_libdoc_integration.search_keywords` to accept a scope
  parameter.
- Default behaviour unchanged (backwards compat).

### Tasks

1. Add the parameter to `find_keywords` + `search_keywords`.
2. Plumb through to `rf_libdoc_integration.search_keywords`.
3. Unit tests:
   - `"Go To"` + scope=name → no Repeat Keyword
   - `"Go To"` + scope=all → current behaviour
   - `"docs mention go to"` + scope=docs → Repeat Keyword surfaces
   - Glob + scope=name → only name globbed
4. Docstring update on `find_keywords` strategy="pattern".

### Out of scope

- Tag-only scope (rarely useful).
- Regex pattern scope (overkill; users can already shape their query).

---

## OBS-32 — LibDoc-fallback ambiguity collapse fix

**Type**: Bug · **Priority**: Low · **Effort**: XS

### Background

`execution_coordinator.get_keyword_documentation` (line 1021) has two
paths:
- **LibDoc primary path** (line 1029-1090): when `library_name=None`,
  correctly returns `matches[]` array via
  `get_keywords_documentation_all`.
- **Inspection fallback path** (line 1115-1116): when LibDoc isn't
  available, silently returns the FIRST match via
  `keyword_discovery.find_keyword(keyword_name)`.

The inspection fallback rarely triggers in production (LibDoc is
preferred). But when it does, the fallback's collapse to a single
arbitrary match contradicts the documented contract for the unscoped
lookup.

### User story

> **As an** agent calling `get_keyword_info(keyword_name="Go To")` in
> a degraded environment where LibDoc isn't available,
> **I want** the same `matches[]` array shape as the LibDoc path,
> **so that** my code doesn't break when one of the two paths is
> active.

### Acceptance criteria

1. In a setup where `rf_doc_storage.is_available()` returns False,
   `get_keyword_documentation("Go To")` (no library_name) returns
   `matches: [...]` shape matching the LibDoc path's shape.
2. When only one library has the keyword, `matches[]` has one entry
   (no special-casing).
3. Existing LibDoc path is unchanged.

### Implementation notes

- Need a sibling method `keyword_discovery.find_all_keywords(keyword_name)`
  that returns all matches across loaded libraries.
- Inspection fallback path (line 1114-1116) calls the new method
  instead of `find_keyword`, and wraps the result in the same
  `matches[]` shape as the LibDoc branch.

### Tasks

1. Add `find_all_keywords` method to `keyword_discovery.py`.
2. Update the inspection fallback in
   `execution_coordinator.get_keyword_documentation`.
3. Unit test: mock `rf_doc_storage.is_available()=False`, verify the
   fallback returns matches[] for an ambiguous keyword.

### Out of scope

- Reworking the inspection fallback to use LibDoc-equivalent shape
  beyond this specific method (large scope).

---

## OBS-33 — `strict_library=True` filter mode

**Type**: UX · **Priority**: Med · **Effort**: S · **Status: NEW (added in round-2 review)**

> Round-2 Codex verdict: scope gap — the evaluation report's
> `strict_library` idea for the S16 problem (`Get*` + Browser
> returns 85 results across 10 compatible libraries because the
> filter only excludes "incompatible" siblings) was missing from
> the story file. Added now.

### Background

`find_keywords(strategy="pattern", query="Get*", library_name="Browser")`
returns 85 results across `Browser, BuiltIn, Collections, DateTime,
Dialogs, OperatingSystem, Process, RequestsLibrary, String, XML`
(per S16 in the benchmark). The `library_name` filter excludes only
`SeleniumLibrary` (the documented "incompatible sibling"), but
"compatible" libraries like BuiltIn / Collections / etc. stay
visible. For an agent that wants Browser keywords specifically,
this is over-inclusive.

OBS-31's `pattern_scope="name"` fixes the noisy-doc-match aspect
but not the cross-library flood. They're complementary fixes.

### User story

> **As an** agent searching for keywords in a specific library
> (e.g., "Browser only, no helpers from BuiltIn or Collections"),
> **I want** to opt into a strict mode that excludes ALL libraries
> except the named one,
> **so that** I get a tight, library-scoped result instead of
> wading through 85 hits across 10 libraries.

### Acceptance criteria

1. `find_keywords(strategy="pattern", query="Get*",
   library_name="Browser", strict_library=True)` returns ONLY
   Browser keywords. BuiltIn / Collections / etc. excluded.
2. Default (`strict_library=False` or omitted) preserves current
   behaviour: incompatible libraries excluded via plugin table,
   "compatible siblings" remain.
3. Works across all strategies (semantic, pattern, catalog,
   session) symmetric to how `library_name` already works.
4. When `library_name` is None: `strict_library=True` is ignored
   (no library to be strict about). Pin in a unit test.
5. Token impact: re-running S16 with `strict_library=True` reduces
   `inline_tokens` from 5407 to <1000.

### Implementation notes

- Add `strict_library: bool = False` parameter to `find_keywords`
  tool surface.
- In `_filter_keywords_by_session_library` (server.py:2037),
  extend the filter: when `strict_library=True` AND a library
  preference is set, exclude EVERY library that doesn't match
  the preference (not just the plugin's `get_incompatible_libraries`
  list).
- Surface in `library_filter.mode = "strict" | "compatible"` for
  agent visibility.

### Tasks

1. Add `strict_library` parameter to `find_keywords` + tool docstring.
2. Extend filter helper to apply strict mode.
3. Surface `library_filter.mode` field.
4. Unit tests:
   - `strict_library=True` + Browser → only Browser keywords
   - `strict_library=False` (default) → existing behaviour
   - `strict_library=True` + no `library_name` → ignored
5. Benchmark re-run: S16 with strict mode shrinks to <1000 tokens.

### Backward compatibility

Default unchanged. Callers don't see any behaviour change unless
they opt in.

---

## Methodology task — benchmark expansion

Not a story (no implementation work), but tracked as a follow-up
chore to close the round-3 methodology gaps:

Add to `scripts/benchmark_discovery_tools.py`:

- Pattern strategy WITH `session_id` (would expose Missed-3 firing).
- Catalog WITH `session_id` and no `library_name` (separates the
  unrealistic S20 whale from realistic session-backed cases).
- `get_keyword_info(mode="keyword")` WITH `session_id` (would have
  caught Missed-1 immediately).
- `get_keyword_info(mode="session")` with ambiguous names.
- BDD-prefixed pattern queries (BDD stripping happens before strategy
  dispatch at server.py:2288-2297 — never tested).
- Cross-strategy same-query comparisons (isolate matcher quality from
  strategy choice).
- Browser-imported session WITHOUT `explicit_library_preference` (for
  OBS-20 verification).
- Typos WITH `library_name` (would have prevented the Finding 6
  overgeneralisation).

After OBS-18..32 land, the expanded benchmark provides empirical
data for the next priority shuffle.

---

## Implementation order suggestion (post-round-2 update)

Cross-cutting infrastructure that other stories rely on, then per-story:

- **Wave 1 (small, independent, ready-now)**:
  OBS-19 (session_id honoured), OBS-21 (externalisation wired),
  OBS-22 (limit honoured), OBS-24 (empty-query guard), OBS-25
  (catalog hard cap), OBS-30 (docs), OBS-32 (LibDoc fallback).
- **Wave 2 (design first, then ship)**:
  OBS-18A (matcher reranker design), OBS-23B (session strategy
  schema design). These produce design docs reviewed by stakeholders
  before implementation starts.
- **Wave 3 (implementation after Wave 1 + design)**:
  OBS-18B (reranker impl, after OBS-18A), OBS-20 (filter symmetry —
  now mirrors actual plugin rules per round-2 fix), OBS-23A
  (session query/limit, no schema change), OBS-33 (strict_library
  mode).
- **Wave 4 (UX polish)**:
  OBS-26 (fuzzy suggestions), OBS-27 (externalise pattern results;
  session-payload externalisation deferred — covered by OBS-23B),
  OBS-28 (filter-aware recommendations), OBS-29 (truncate args),
  OBS-31 (pattern_scope).

**Key changes from the round-1 implementation order**:
- OBS-18 (now OBS-18A+18B) moved from "Wave 4 last" to Waves 2+3 — the
  matcher quality issue is too fundamental to defer.
- OBS-20 moved from Wave 2 to Wave 3 — needs design clarification
  (the now-rewritten asymmetric rules).
- OBS-23 split with 23B requiring design before implementation.
- OBS-30 promoted to Wave 1 (docs change is independent and quick).
- OBS-33 added as a Wave 3 story.

Total estimated effort (revised): ~5-6 PR-weeks if Wave 2 design
docs are reviewed thoroughly before Wave 3 starts. Wave 4 can run
in parallel with Wave 3 (different code paths, different test files).

---

## Round-2 review acknowledgement (Codex CLI, 2026-05-18)

After drafting OBS-18..32, I ran another adversarial review with
`codex --dangerously-bypass-approvals-and-sandbox`. Codex assessed
each story against the actual source code and surfaced concrete
defects in the story specifications themselves. Summary of the
verdicts (full Codex output preserved in conversation history):

| Story | Codex verdict | Round-2 resolution |
|---|---|---|
| OBS-18 | WRONG-SCOPE | Split into OBS-18A (design) + OBS-18B (impl) |
| OBS-19 | DEPENDENCIES | Added AC + task for plugin hint shared helper |
| OBS-20 | IMPL-INFEASIBLE | Rewritten to mirror asymmetric execute rules |
| OBS-21 | READY (minor) | Removed misleading "session branch" mention |
| OBS-22 | READY | No change |
| OBS-23 | WRONG-SCOPE | Split into OBS-23A (query/limit) + OBS-23B (schema, design first) |
| OBS-24 | READY | No change |
| OBS-25 | READY | No change |
| OBS-26 | TIGHTEN | Picked one algorithm (`difflib.get_close_matches`) |
| OBS-27 | DEPENDENCIES | Noted overlap with OBS-23B; session externalisation deferred |
| OBS-28 | READY | No change |
| OBS-29 | TIGHTEN | Fixed field names per-branch (`arguments` vs `args`) |
| OBS-30 | READY | No change |
| OBS-31 | TIGHTEN | Defined glob semantics per scope |
| OBS-32 | READY | No change |
| OBS-33 | NEW | Added (Codex caught the scope gap re: strict_library) |

**Codex's identified worst-quality stories** (before fixes): OBS-20,
OBS-23, OBS-18. All three needed structural changes, not just
parameter tweaks. Codex's identified best stories (positive
exemplars): OBS-21, OBS-22, OBS-24 — precise defects, exact code
points, minimal surface area. The pattern to repeat in future
stories.

**Cross-story risks Codex flagged**:
- Four distinct `find_keywords` payload shapes already live at
  `server.py:2370`, `2423`, `2492`, `2537`. Multiple stories touch
  this fault line — pin contract tests early.
- Backward compatibility was understated in OBS-23 / OBS-29 /
  OBS-31. All three are API-contract stories; not just UX polish.
- The benchmark-expansion methodology task is a real blocker for
  prioritisation confidence — without empirical frequency data, the
  prioritisation between e.g. OBS-12 (session strategy) and
  OBS-20 (filter symmetry) is based on intuition.

**Status of the story doc after round 2**: 15 stories → 17 effective
stories (after splits), each with one of {READY, TIGHTEN-RESOLVED,
WRONG-SCOPE-RESOLVED, DEPENDENCIES-RESOLVED, IMPL-INFEASIBLE-RESOLVED,
NEW} status. No stories remain in an unresolved problematic state.
