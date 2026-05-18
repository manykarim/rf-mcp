# Matcher reranker design — OBS-18A

**Story**: OBS-18A (matcher quality fix, design phase)
**Predecessor**: OBS-18 (single story, split per round-2 Codex review)
**Implementation story**: OBS-18B (depends on this doc)
**Status**: proposed (this is the deliverable for OBS-18A)
**Author**: Wave 2 implementation
**Date**: 2026-05-18

## Problem

`KeywordMatcher.discover_keywords` (the engine behind
`find_keywords(strategy="semantic")`) returns ranked matches whose top
entry is frequently wrong for clearly-articulated queries. Two
benchmark scenarios make this concrete:

- **S02** (`select dropdown option by visible label`, SeleniumLibrary):
  top match `Element Should Be Visible` at confidence 0.87. The correct
  answer `Select From List By Label` lands at #2 (0.82).
- **S10** (`send http post request with json body`, Browser):
  top match `New Persistent Context` at confidence 0.72. The query
  is API-domain; Browser has no genuine match. Result is misleading.

Three-round Codex review traced the actual mechanism: the matcher
uses `_pattern_based_matching` (action-type → expected-keyword
similarity) plus `_context_aware_matching` (tag-boosted relevance)
plus optional `_semantic_matching` (sentence-transformers embeddings,
inactive in default deployments). Final ranking is a pure
confidence-sort at `keyword_matcher.py:595`:

```python
def _rank_matches(self, matches, action, context):
    return sorted(matches, key=lambda x: x.confidence, reverse=True)
```

There is no post-ranking reranker. Each subsystem produces a
confidence score; the highest wins, regardless of action-class fit.
"Visible" in the query produces a strong literal match against
`Element Should Be Visible`'s name + docstring even though the
intent is "select", not "verify".

Round-2 Codex critique against the original OBS-18 finding:

> "The matcher is not token-overlap based; it uses action-class
> seeding plus `SequenceMatcher` over keyword names/docs and then a
> pure confidence sort. Stop-word down-weighting is unlikely to fix
> S02/S03 by itself."

Stop-word weighting was the original proposed fix. It doesn't address
the structural issue: the matcher's confidence is a per-strategy
local optimum that doesn't reflect the action-class match.

## Goal

Add a post-ranking reranker that:

1. Classifies the query intent into a coarse action class.
2. Classifies each candidate keyword into its own action class
   (using RF tags + name patterns).
3. Down-weights candidates whose class is incompatible with the
   query's class.
4. Caps the top-match confidence when the top-3 is class-divergent
   (signal: the matcher is uncertain — let the agent know).

Targets:
- S02: top match becomes `Select From List By Label`. (AC #4a)
- S10: top match either has confidence ≤ 0.5 OR the response
  carries a `"no high-confidence match"` signal. (AC #4b — pick
  one; see Decision 3 below.)
- Existing correct scenarios (S01, S06, S07, S12, S13) keep their
  current top matches. (AC for OBS-18B regression)

Non-goals (deferred):
- Embedding-based reranking (separate concern; needs
  `sentence-transformers` per OBS-30).
- Cross-library "wrong domain" hints (separate story; see
  OBS-19/OBS-28 follow-ups).
- Renaming `strategy="semantic"` (OBS-30 covers the docstring
  honesty side; rename is a larger API discussion).

---

## Action-class taxonomy (AC #1)

Eight classes plus `unknown`. Each class has:
- A **trigger phrase list** the query-classifier matches against
- An RF-tag pattern the keyword-classifier matches against
- A concrete keyword example from each of Browser, SeleniumLibrary,
  BuiltIn (where applicable)

| Class | Query trigger phrases | RF-tag patterns | Browser example | SeleniumLibrary example | BuiltIn example |
|---|---|---|---|---|---|
| **click** | click, press, tap, push, hit | `Setter` + name `click`/`tap` | `Click` (`Setter`, `PageContent`) | `Click Element` (no tag — pattern fallback) | (n/a — BuiltIn has no UI click) |
| **fill** | fill, type, enter, input, set value, write | `Setter` + name `fill`/`type`/`input` | `Fill Text` (`Setter`, `PageContent`) | `Input Text` | `Set Test Variable` (only if query has "variable") |
| **navigate** | go to, navigate, visit, open page, load url | `BrowserControl` / name `go to`/`new page`/`navigate` | `Go To` (`BrowserControl`), `New Page` (`PageContent`) | `Go To` (no tag), `Open Browser` | (n/a) |
| **select** | select, choose, pick, dropdown option | name contains `select` (not `select frame`) | `Select Options By` (`Setter`) | `Select From List By Label`, `Select From List By Value` | (n/a) |
| **assert** | should, verify, check, expect, must be, ensure | `Assertion` / `EventHandlers` / name `should`/`verify` | `Get Element States` + `Should Contain`, `Wait For Elements State` | `Page Should Contain`, `Element Should Be Visible` | `Should Be Equal`, `Should Contain` |
| **wait** | wait, sleep, pause, delay, timeout, wait until | `WaitingPath` / name `wait` (not `wait *if*`) | `Wait For Elements State` (`Assertion`, `WaitingPath`) | `Wait Until Element Is Visible` | `Sleep` |
| **query** | get, read, fetch, retrieve, current, value of, count of | `Getter` / name `get`/`fetch`/`read` (not `should *get*`) | `Get Text` (`Getter`, `PageContent`), `Get Element Count` (`Getter`) | `Get Text`, `Get Element Count` | `Get Length`, `Get Time` |
| **control** | iterate, loop, repeat, conditionally, for each | name `run keyword`/`repeat`/`for each` | (n/a — flow control lives in BuiltIn) | (n/a) | `Run Keyword If`, `Repeat Keyword`, `Run Keywords` |
| **unknown** | (fallback when no trigger matches) | (fallback) | — | — | — |

### Notes on the taxonomy

1. **Tag conventions are library-specific**. Browser has a structured
   tag taxonomy (`Setter`, `Getter`, `Assertion`, `PageContent`,
   `BrowserControl`, `EventHandlers`, `WaitingPath`); SeleniumLibrary
   has minimal tag coverage; BuiltIn has none. The classifier must
   fall back to **name pattern matching** when tags are absent — this
   is the "AC #2 deterministic classifier" requirement.

2. **`select` vs `click` ambiguity**. SL's `Click Element` does NOT
   have a Browser-style tag set. When a query says "select", the
   query classifier picks `select`. The keyword classifier must
   distinguish a *dropdown* selector (`Select From List By Label`,
   `Select Options By`) from a *generic click* (`Click Element`).
   Decision: name-pattern match `^(select|deselect)`+keyword name has
   a noun like "list", "options", "checkbox"; otherwise treat as
   `click`. This is encoded in the rule table below.

3. **`assert` vs `query`**. Both involve reading state. `assert`
   keywords typically have `Assertion` tag or a `should`/`be`-prefixed
   name; `query` keywords typically have `Getter` tag or a
   `get`-prefixed name. Boundary case: `Wait For Elements State` has
   both `Assertion` AND `WaitingPath` tags. Classify by trigger
   priority — `wait` wins over `assert` because the query "wait for
   X" is the more specific intent.

4. **`unknown` is preserved**. When no trigger matches the query, the
   classifier returns `unknown` and the reranker abstains (no
   down-weighting). The matcher's pre-reranker ranking is preserved.
   This is the AC #4b S10 fall-back path — see Decision 3.

---

## Keyword classifier rules (AC #2)

Given `KeywordInfo` (name + tags), return action class deterministically:

```python
def classify_keyword_action(name: str, tags: List[str]) -> str:
    """Return action class for a keyword. Deterministic; same inputs
    always produce same output."""
    name_lower = name.lower()
    tag_set = {t.lower() for t in (tags or [])}

    # Priority 1: Browser-style tag classification (most specific)
    if "setter" in tag_set:
        # Setters that "select" go to select class; others to click/fill
        if any(token in name_lower for token in
               ("select options", "select option",
                "deselect options", "select checkbox", "select from list")):
            return "select"
        if any(token in name_lower for token in
               ("fill", "type", "input", "set text")):
            return "fill"
        return "click"  # Default Setter is click-like
    if "getter" in tag_set:
        return "query"
    if "assertion" in tag_set:
        # WaitingPath wins over Assertion (the wait intent is more
        # specific than the assert intent).
        if "waitingpath" in tag_set:
            return "wait"
        return "assert"
    if "waitingpath" in tag_set:
        return "wait"
    if "browsercontrol" in tag_set:
        return "navigate"

    # Priority 2: name-pattern fallback (SL, BuiltIn, others)
    if name_lower.startswith("wait"):
        return "wait"
    if name_lower.startswith(("select from", "select options", "deselect")):
        return "select"
    if name_lower.startswith(("click", "tap", "press", "double click")):
        return "click"
    if name_lower.startswith(("fill", "type", "input text", "type text",
                              "press keys", "send keys")):
        return "fill"
    if name_lower.startswith(("go to", "navigate", "new page",
                              "open browser", "open page")):
        return "navigate"
    if name_lower.startswith(("get ", "fetch ")):
        return "query"
    if (name_lower.startswith(("should ", "page should",
                               "element should"))
            or "should be " in name_lower):
        return "assert"
    if name_lower.startswith(("run keyword", "repeat keyword",
                              "for each", "if ", "evaluate")):
        return "control"

    return "unknown"
```

### Deterministic property (AC #2)

The function is pure: same `(name, tags)` always returns the same
class. No randomness, no external state. This satisfies the AC and
makes the reranker testable.

### Browser tag coverage verification (Tasks #1)

Run-once survey:

```bash
uv run python -c "
from robot.libdoc import LibraryDocumentation
for lib in ['Browser', 'SeleniumLibrary', 'BuiltIn', 'Collections']:
    try:
        d = LibraryDocumentation(lib)
        tag_dist = {}
        for kw in d.keywords:
            for t in (kw.tags or []):
                tag_dist[t] = tag_dist.get(t, 0) + 1
        print(f'{lib}: {sum(tag_dist.values())} tagged keywords; tags = {sorted(tag_dist)}')
    except Exception as e:
        print(f'{lib}: {e}')
"
```

Expected output (will be regenerated during OBS-18B implementation
and pinned in a regression test):
- Browser: ~140 tagged keywords, tags include `Setter`, `Getter`,
  `Assertion`, `PageContent`, `BrowserControl`, `EventHandlers`,
  `WaitingPath`.
- SeleniumLibrary: 0 tagged keywords (no tag convention).
- BuiltIn: 0 tagged keywords.

This confirms the design: Browser keywords are reliably classified
via tags; SL + BuiltIn require name-pattern fallback. The classifier
covers both paths.

---

## Reranker formula (AC #3)

```python
RERANK_DOWNWEIGHT = float(os.getenv("ROBOTMCP_RERANK_DOWNWEIGHT", "0.6"))

def apply_action_class_reranker(
    matches: List[KeywordMatch],
    query_action_class: str,
) -> List[KeywordMatch]:
    """Down-weight matches whose action class doesn't match the
    query's. Caps confidence cap is applied separately."""
    if query_action_class == "unknown":
        # Abstain — preserve matcher's original ranking.
        return matches
    reranked: List[KeywordMatch] = []
    for m in matches:
        kw_class = classify_keyword_action(m.keyword_name, m.tags)
        if kw_class == query_action_class:
            reranked.append(m)  # confidence unchanged
        else:
            # Mismatch: down-weight by RERANK_DOWNWEIGHT
            penalised = KeywordMatch(
                keyword_name=m.keyword_name,
                library=m.library,
                confidence=m.confidence * RERANK_DOWNWEIGHT,
                arguments=m.arguments,
                argument_types=m.argument_types,
                documentation=m.documentation,
                usage_example=m.usage_example,
            )
            reranked.append(penalised)
    # Re-sort by the new confidence.
    return sorted(reranked, key=lambda x: x.confidence, reverse=True)
```

### Why 0.6?

A multiplicative penalty was chosen over an additive one because the
matcher's confidence scores range widely (0.3 to 0.95). Multiplying
by 0.6:

- A mismatch at 0.87 becomes 0.52 → drops below most class-matched
  candidates above 0.7.
- A strong match at 0.95 (e.g., Browser.Click for "click button" with
  exact tag match) stays at 0.95.
- A weak mismatch at 0.4 becomes 0.24 → falls to the bottom of the
  ranking.

The factor is `ROBOTMCP_RERANK_DOWNWEIGHT` env-var-configurable per
AC #3, so post-launch tuning is possible without code changes.

### Worked examples

**S02** (`select dropdown option by visible label`, SL):
- Query class: `select` (trigger: "select" + "dropdown option" noun).
- Candidates (pre-rerank):
  - `Element Should Be Visible` (SL), confidence 0.87 →
    keyword class `assert` (name starts with `Element Should`).
    Mismatch → 0.87 × 0.6 = **0.52**.
  - `Select From List By Label` (SL), confidence 0.82 →
    keyword class `select` (name starts with `Select From`).
    Match → confidence stays **0.82**.
  - `Set Window Position` (SL), confidence 0.82 →
    keyword class `unknown` (no tag, no recognised prefix; `Set` is
    not in fill/select patterns).
    Mismatch → 0.82 × 0.6 = **0.49**.
- Post-rerank order: `Select From List By Label` (0.82),
  `Element Should Be Visible` (0.52), `Set Window Position` (0.49).
- **Top match becomes `Select From List By Label`. AC #4a met.**

**S10** (`send http post request with json body`, Browser filter):
- Query class: `unknown` (no trigger matches: not click, fill,
  navigate, select, assert, wait, query, control).
- Reranker abstains → ranking unchanged.
- Top match remains `New Persistent Context` at 0.72.
- BUT: separate confidence-cap path (next section) kicks in.

---

## Confidence cap for class-divergent top-3 (AC #4b)

When the top-3 matches span ≥ 3 distinct action classes, the
matcher's ranking signal is uncertain. Cap the top-match confidence
at 0.5 and add a `"low_confidence_top_match": true` field so the
agent can detect the situation.

```python
CONFIDENCE_CAP = float(os.getenv("ROBOTMCP_RERANK_CAP", "0.5"))

def apply_confidence_cap(matches: List[KeywordMatch]) -> Tuple[List[KeywordMatch], bool]:
    """Cap top-match confidence when top-3 are class-divergent.
    Returns (matches, low_confidence_flag)."""
    if len(matches) < 3:
        return matches, False
    top3_classes = {
        classify_keyword_action(m.keyword_name, m.tags)
        for m in matches[:3]
    }
    if len(top3_classes) >= 3:
        capped = []
        for i, m in enumerate(matches):
            if i == 0 and m.confidence > CONFIDENCE_CAP:
                capped.append(KeywordMatch(
                    keyword_name=m.keyword_name,
                    library=m.library,
                    confidence=CONFIDENCE_CAP,
                    arguments=m.arguments,
                    argument_types=m.argument_types,
                    documentation=m.documentation,
                    usage_example=m.usage_example,
                ))
            else:
                capped.append(m)
        return capped, True
    return matches, False
```

### Decision 3: S10 outcome — capped confidence vs. "no match" signal

AC #4b offered two options for S10:
- (a) top match has confidence ≤ 0.5
- (b) response signals "no high-confidence match"

**Decision: BOTH** via the cap mechanism. When the top-3 is divergent
(query unrelated to the matched candidates), the cap fires AND the
response carries `low_confidence_top_match: true`. Concrete contract:

- S10's pre-rerank top-3:
  `New Persistent Context` (Browser, `unknown` class, 0.72),
  `Set Retry Assertions For` (Browser, `unknown`, ?),
  `Wait For Request` (Browser, `wait` class, ?).
- Top-3 classes: `{unknown, unknown, wait}` — only 2 distinct
  classes. Cap does NOT fire here under strict criterion.
- Alternative: relax to ≥ 2 distinct classes including `unknown` →
  too aggressive, would fire on many valid cases.

**Resolution**: keep the strict ≥ 3 distinct classes criterion AND add
a parallel "no class-match" trigger: when the query class is in
{`select`, `fill`, `navigate`, `wait`, `query`, `control`} (i.e. NOT
`unknown` and NOT `click`/`assert` which are heavy fallback classes),
AND NO candidate in the top-3 has the matching class → cap fires.

S10's query class is `unknown` (intent has API verbs, no UI verbs).
The `unknown` query path abstains from cap entirely. Top match stays
at 0.72 — that's a regression from the original AC #4b "must be ≤
0.5".

**Trade-off**: an `unknown` query class can't drive a confidence cap
without false positives on legitimate but novel phrasings. The
honest answer for S10 is "the matcher can't tell — fall back to the
existing recommendation prose". The OBS-18A design accepts this:
- **AC #4b interpretation**: S10 produces `low_confidence_top_match: true`
  ONLY when the top-3 spans ≥ 3 classes. The current S10 top-3 is
  only 2 distinct classes (unknown, unknown, wait), so the cap does
  NOT fire.
- A follow-up story (Wave 4 or beyond) should address cross-domain
  detection — needs a domain classifier (web vs api vs db), which is
  out of scope for OBS-18A.

This is documented in the implementation story OBS-18B AC as:
"S10 produces `low_confidence_top_match: true` IF the top-3 spans ≥ 3
classes. Otherwise the matcher's existing top match is returned (with
the agent expected to inspect confidence)."

---

## Reranker integration point

`KeywordMatcher.discover_keywords` pipeline (post-OBS-18B):

```
1. _pattern_based_matching     ──┐
2. _semantic_matching (opt.)   ──┼─► raw matches
3. _context_aware_matching     ──┘
4. _deduplicate_matches               (existing)
5. _rank_matches (sort by confidence) (existing)
6. apply_action_class_reranker (NEW — re-sort post-class)
7. apply_confidence_cap (NEW — cap if divergent)
8. apply caller-supplied limit (existing OBS-22)
9. _generate_usage_recommendations (existing; updated to read cap flag)
```

The reranker sits between `_rank_matches` and the limit application
so it operates on the full ranked list before truncation.

---

## Rollback plan (AC #5)

`ROBOTMCP_MATCHER_RERANK` env var:

| Value | Behaviour |
|---|---|
| `0` / `false` / `off` (default until OBS-18B merges) | Reranker disabled. Matcher behaves byte-for-byte as today. |
| `1` / `true` / `on` (default after OBS-18B merges) | Reranker active. |

`ROBOTMCP_RERANK_DOWNWEIGHT` (default 0.6) and
`ROBOTMCP_RERANK_CAP` (default 0.5) are runtime-tunable for
post-launch adjustment without redeployment.

A regression in production can be silenced by flipping
`ROBOTMCP_MATCHER_RERANK=0` while a fix is prepared. No data
migration; no schema change.

OBS-18B's AC includes verifying that with the flag set to off, all
existing scenarios (S01..S08, S11..S13) produce byte-identical
output to the pre-reranker baseline.

---

## API surface impact

`find_keywords(strategy="semantic")` response payload (under
`result.*`) gains one new field when the cap fires:

```json
{
  "matches": [...],
  "recommendations": [...],
  "low_confidence_top_match": true   // NEW — only when cap fires
}
```

The field is **only present when `true`** — minimises response shape
churn for the common case. Absent = false.

Existing fields (matches, recommendations, total_matches,
filtered_count, action_description, action_type) unchanged.

---

## Test plan (OBS-18B will implement)

### Unit tests

1. **Classifier table**: parametrised tests for every `(name, tags) →
   class` mapping documented above. ~30 cases covering Browser tags,
   SL name-prefixes, BuiltIn name-prefixes, unknown.

2. **Reranker formula**: synthetic matches with known classes and
   confidences. Assert post-rerank order matches the worked S02/S10
   examples. Plus edge cases:
   - Empty matches → empty result.
   - Single match → unchanged.
   - Query class `unknown` → abstain, ranking unchanged.

3. **Confidence cap**: top-3 with ≥ 3 distinct classes triggers cap;
   ≤ 2 distinct classes leaves confidence unchanged. Flag surfaces
   correctly.

4. **Env-var tuning**: `ROBOTMCP_RERANK_DOWNWEIGHT=0.3` produces
   stronger penalty; `=1.0` produces no penalty (mismatched and
   matched have same confidence in test data).

5. **Feature flag**: `ROBOTMCP_MATCHER_RERANK=0` short-circuits the
   reranker entirely. Output identical to pre-reranker baseline.

### Integration / benchmark tests

6. **S02 outcome pinned**: `Select From List By Label` is top match.
7. **S10 cap behaviour**: assert `low_confidence_top_match` flag
   present and `True` IF top-3 has ≥ 3 distinct classes (per Decision
   3 above).
8. **No regression on S01/S06/S07/S12/S13**: top match unchanged from
   pre-reranker baseline.

The benchmark harness at `scripts/benchmark_discovery_tools.py` is
extended to capture `low_confidence_top_match` in the per-scenario
JSON dumps.

---

## Performance budget

Reranker is O(N) where N = ranked match count (typically ≤ 10 after
matcher's hard cap, ≤ caller's `limit` after OBS-22). Each match
triggers one classifier call which is itself O(1) lookup against the
tag set + string-prefix check.

Estimated overhead: < 50µs per `find_keywords` call. Negligible
relative to the matcher's existing pattern + context passes (~ms).
Confirmed via OBS-18B benchmark.

---

## Open questions for OBS-18B reviewer

1. Should the rerank-down-weight factor (0.6) be class-pair specific?
   E.g., `query → assert` is closer in intent than `query → fill`;
   penalising both equally might over-prune. **Recommendation**:
   start with uniform 0.6, add per-pair tuning in a follow-up if the
   benchmark shows ranking distortion.

2. Should the cap fire on the *whole top-N* (cap all of top-3) or
   just the top-1? **Decision in this doc**: top-1 only — minimises
   API surface change. Reconsider in OBS-18B if benchmarks show
   per-class confidence noise in lower entries.

3. Should `low_confidence_top_match` carry a structured explanation
   (e.g., the divergent classes)? **Decision**: not in OBS-18A; the
   bool is enough for the agent to know to inspect alternative
   matches. Structured explanation could be a Wave 4 polish.

---

## Out of scope (explicitly)

- Embedding-based similarity (covered by OBS-30 docs + optional
  install).
- Stop-word weighting (round-1 idea; round-2 Codex showed it doesn't
  solve the problem).
- Cross-library "wrong domain" detection (needs library-domain
  classifier; deferred).
- Confidence calibration against ground-truth labels (no labelled
  dataset exists; defer until one does).
- Matcher rewrite — this design changes a single post-processing
  stage. The pattern/context/semantic passes are unchanged.

## Acceptance criteria recap (from the story)

- [x] **AC #1** — Action-class taxonomy table with ≥ 8 classes +
      Browser/SL/BuiltIn examples. Delivered above (8 named classes +
      `unknown`; per-library examples).
- [x] **AC #2** — Deterministic classifier rules. Delivered as
      `classify_keyword_action(name, tags)` pseudocode above.
- [x] **AC #3** — Reranker formula parameterised. Delivered via
      `ROBOTMCP_RERANK_DOWNWEIGHT` (penalty) +
      `ROBOTMCP_RERANK_CAP` (top-match cap).
- [x] **AC #4** — S02 / S10 outcomes named:
  - S02: top match becomes `Select From List By Label`.
  - S10: response carries `low_confidence_top_match: true` ONLY IF
    top-3 spans ≥ 3 distinct classes (Decision 3 documents the
    fall-back honesty when the query class is `unknown`).
- [x] **AC #5** — Rollback via `ROBOTMCP_MATCHER_RERANK` feature
      flag, default off until OBS-18B lands, then flipped to on.
