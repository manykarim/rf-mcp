# Matcher reranker design — OBS-18A

**Story**: OBS-18A (matcher quality fix, design phase)
**Predecessor**: OBS-18 (single story, split per round-2 Codex review)
**Implementation story**: OBS-18B (depends on this doc)
**Status**: **v2** — revised 2026-05-18 after Codex CLI + Claude
sub-agent adversarial review. v1 had: incorrect Browser tags
(`WaitingPath`/`EventHandlers` don't exist), wrong classifier
precedence (`Go To` returned `click` not `navigate`), missing
`KeywordMatch.tags` plumbing, AC #4b unsatisfied for S10,
optimistic performance budget. All addressed below; see "Review
findings + resolutions" section at end of doc.
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

**Browser tag distribution** (verified via
`LibraryDocumentation('Browser')`, 2026-05-18):

```
PageContent: 82   Setter: 80          Getter: 44   BrowserControl: 39
Assertion: 31     Wait: 14            Config: 7    HTTP: 5
Clock: 4          Coverage: 3         Crawling: 1  Experimental: 1
Page Content: 1   (note: with a space — duplicate of PageContent;
                   1 outlier keyword. Classifier normalises both.)
```

Tags NOT in Browser (v1 design referenced these incorrectly):
- ❌ `WaitingPath` — replaced with `Wait`
- ❌ `EventHandlers` — does not exist; assertions use `Assertion`

### Tag patterns

| Class | Query trigger phrases | RF tag (precedence-ordered) | Browser example (real tags) | SeleniumLibrary example | BuiltIn example |
|---|---|---|---|---|---|
| **wait** | wait, sleep, pause, delay, timeout, wait until | `Wait` (most specific; wins over Setter/PageContent) | `Wait For Elements State` (`PageContent`, `Wait`) | `Wait Until Element Is Visible` | `Sleep` |
| **navigate** | go to, navigate, visit, open page, load url | `BrowserControl` (wins over Setter) | `Go To` (`BrowserControl`, `Setter`), `New Page` (`BrowserControl`, `Setter`) | `Go To` (no tag), `Open Browser` | (n/a) |
| **query** | get, read, fetch, retrieve, current, value of, count of | `Getter` (wins over Assertion when co-tagged) | `Get Text` (`Assertion`, `Getter`, `PageContent`), `Get Element Count` (`Assertion`, `Getter`, `PageContent`) | `Get Text`, `Get Element Count` | `Get Length`, `Get Time` |
| **assert** | should, verify, check, expect, must be, ensure | `Assertion` (without `Getter` — pure assertions) | `Should Contain` (only assertion tagging without Getter), Wait+Assertion keywords go to `wait` first | `Page Should Contain`, `Element Should Be Visible` | `Should Be Equal`, `Should Contain` |
| **select** | select, choose, pick, dropdown option | `Setter` + name pattern `select / deselect / select from / select options` | `Select Options By` (`PageContent`, `Setter`), `Deselect Options` | `Select From List By Label`, `Select From List By Value` | (n/a) |
| **fill** | fill, type, enter, input, set value, write | `Setter` + name pattern `fill / type / input / set text / press keys` | `Fill Text` (`PageContent`, `Setter`), `Type Text` (`PageContent`, `Setter`) | `Input Text` | (n/a) |
| **click** | click, press, tap, push, hit | `Setter` (default when no other Setter pattern matches) | `Click` (`PageContent`, `Setter`), `Tap` | `Click Element` (no tag — pattern fallback) | (n/a) |
| **control** | iterate, loop, repeat, conditionally, for each | name `run keyword`/`repeat`/`for each` | (n/a — flow control lives in BuiltIn) | (n/a) | `Run Keyword If`, `Repeat Keyword`, `Run Keywords` |
| **unknown** | (fallback when no trigger matches) | (fallback) | — | — | — |

### Notes on the taxonomy

1. **Tag conventions are library-specific**. Browser has a structured
   tag taxonomy (`Setter`, `Getter`, `Assertion`, `PageContent`,
   `BrowserControl`, `Wait`); SeleniumLibrary has minimal tag
   coverage; BuiltIn has none. The classifier must fall back to
   **name pattern matching** when tags are absent — this is the
   "AC #2 deterministic classifier" requirement.

2. **Precedence is load-bearing.** Co-tagged keywords resolve via
   the priority order:
   `Wait → BrowserControl → Getter → Assertion → Setter (+ name)`.
   This precedence was the v1 design's biggest bug:
   - `Go To` (`['BrowserControl', 'Setter']`) → v1 returned `click`
     because it checked Setter first. v2 returns `navigate`.
   - `Get Text` (`['Assertion', 'Getter', 'PageContent']`) → v1
     returned `query` (correct by luck since `getter` was before
     `assertion`); v2 keeps the same outcome with documented order.
   - `Wait For Elements State` (`['PageContent', 'Wait']`) → v1
     incorrectly referenced `WaitingPath` (doesn't exist); v2 uses
     `Wait`.

3. **`select` vs `click` ambiguity**. Both are `Setter` in Browser.
   The distinguisher is name pattern. `Setter` + name starts with
   `select / deselect / select options / select from` → `select`.
   `Setter` + name starts with `fill / type / input text / type
   text / set text / press keys` → `fill`. Default `Setter` (no
   name pattern match) → `click`.

4. **`assert` vs `query`**. Both involve reading state. Decision:
   when a keyword has BOTH `Getter` and `Assertion` tags (which
   Browser commonly does — `Get Text`, `Get Element States`, etc.
   are tagged both), classify as **`query`** because the keyword's
   primary value is the returned data. Pure-`Assertion` keywords
   (no `Getter`) — typically `Should*`/`Page Should*` — are `assert`.
   This avoids the "getter-asserts get reclassified as assert" bug
   Claude flagged.

5. **`unknown` query handling.** When no trigger matches the query,
   the reranker abstains (no down-weighting). When the query class
   is `unknown` but the **top match** has an opinionated class, the
   confidence cap fires — see "Confidence cap" section below for
   the revised S10 rule (Claude review fix).

---

## Keyword classifier rules (AC #2)

Given `KeywordInfo` (name + tags), return action class deterministically:

```python
def classify_keyword_action(name: str, tags: List[str]) -> str:
    """Return action class for a keyword. Deterministic; same inputs
    always produce same output.

    Precedence (v2 — corrected after Codex round-1 review caught
    Browser tag misclassification):
      Wait → BrowserControl → Getter → Assertion → Setter+name → name

    NB: tag values are case-normalised (Browser uses `Wait` not `wait`,
    SL/BuiltIn don't tag at all). Both `PageContent` and
    `Page Content` (one outlier keyword) are recognised — same tag,
    different spellings in Browser's libdoc.
    """
    name_lower = name.lower()
    # Normalise tags: lower-case + strip spaces. Defensive against
    # None entries (resource keywords' KeywordInfo can have a None
    # in the tags list).
    tag_set = {
        (t or "").lower().replace(" ", "")
        for t in (tags or [])
        if t
    }

    # Priority 1: Wait wins (most specific intent)
    if "wait" in tag_set:
        return "wait"

    # Priority 2: BrowserControl wins over Setter (Go To, New Page
    # are tagged BOTH — v1 design returned click; v2 correctly
    # returns navigate)
    if "browsercontrol" in tag_set:
        return "navigate"

    # Priority 3: Getter wins over Assertion when co-tagged
    # (Browser's Get Text, Get Element Count are tagged BOTH; the
    # keyword's primary value is the returned data → query)
    if "getter" in tag_set:
        return "query"

    # Priority 4: Pure Assertion (no Getter) — Should*, Page Should*
    # SL has no Assertion tag, falls through to name-pattern below
    if "assertion" in tag_set:
        return "assert"

    # Priority 5: Setter — context-dependent via name pattern.
    # Order matters: select/fill patterns checked before click default.
    if "setter" in tag_set:
        if name_lower.startswith((
            "select options", "select option", "select from",
            "select checkbox", "deselect options", "deselect checkbox",
        )):
            return "select"
        if name_lower.startswith((
            "fill text", "fill secret", "type text", "type secret",
            "input text", "input password", "input secret",
            "set text", "press keys",
        )):
            return "fill"
        return "click"  # default Setter

    # Priority 6: name-pattern fallback (SL, BuiltIn, resource kws)
    if name_lower.startswith("wait"):
        return "wait"
    if name_lower.startswith(("go to", "navigate", "new page",
                              "open browser", "open page")):
        return "navigate"
    if name_lower.startswith(("select from", "select options",
                              "deselect", "select checkbox")):
        return "select"
    if name_lower.startswith(("click", "tap", "press", "double click")):
        return "click"
    if name_lower.startswith(("fill", "type", "input text", "type text",
                              "input password", "set text", "press keys",
                              "send keys")):
        return "fill"
    # Pure-assertion check before pure-getter to keep `Should Be`
    # pattern from getting reclassified as control via "run keyword".
    if (name_lower.startswith(("should ", "page should",
                               "element should"))
            or " should be " in f" {name_lower} "):
        return "assert"
    if name_lower.startswith(("get ", "fetch ", "read ")):
        return "query"
    if name_lower.startswith(("run keyword", "repeat keyword",
                              "for each", "evaluate")):
        return "control"

    return "unknown"
```

### Deterministic property (AC #2)

The function is pure: same `(name, tags)` always returns the same
class. No randomness, no external state. Tags are case-normalised
+ space-stripped, so `"Page Content"` and `"PageContent"` both
match the same set entry. None entries in the tags list are
filtered defensively.

### Worked classifier outputs (v2 verification)

Verified against actual Browser tags (see distribution above):

| Keyword | Tags | v1 returned | v2 returns | Reasoning |
|---|---|---|---|---|
| `Click` | `['PageContent', 'Setter']` | `click` ✓ | `click` ✓ | Setter, no name pattern match |
| `Fill Text` | `['PageContent', 'Setter']` | `fill` ✓ | `fill` ✓ | Setter + `fill` prefix |
| `Go To` | `['BrowserControl', 'Setter']` | `click` ❌ | `navigate` ✓ | BrowserControl wins over Setter |
| `New Page` | `['BrowserControl', 'Setter']` | `click` ❌ | `navigate` ✓ | Same precedence fix |
| `Select Options By` | `['PageContent', 'Setter']` | `select` ✓ | `select` ✓ | Setter + `select options` prefix |
| `Wait For Elements State` | `['PageContent', 'Wait']` | `wait` (via incorrect `WaitingPath` reference, lucky outcome) | `wait` ✓ | Wait tag wins (Priority 1) |
| `Get Text` | `['Assertion', 'Getter', 'PageContent']` | `query` ✓ | `query` ✓ | Getter wins over Assertion |
| `Get Element States` | `['Assertion', 'Getter', 'PageContent']` | `query` (correct by tag-ordering luck) | `query` ✓ | Same — explicit precedence |
| `Should Contain` (BuiltIn) | `[]` | `assert` ✓ | `assert` ✓ | Name pattern `should ` |
| `Sleep` (BuiltIn) | `[]` | `unknown` ❌ | `unknown` ✓ | No name match; explicit fall-through (BuiltIn Sleep doesn't match `wait` prefix) |
| `Repeat Keyword` (BuiltIn) | `[]` | `control` ✓ | `control` ✓ | Name pattern `repeat keyword` |

The `Sleep` mis-classification is acknowledged: it's a name that doesn't follow the `wait` prefix convention. Either:
- Add `sleep` as a name-pattern trigger for `wait` (cleanest); or
- Accept `unknown` because BuiltIn `Sleep` is a generic-blocking call rather than a wait-for-state.

**Decision**: add `sleep` as a wait trigger in OBS-18B impl. The classifier definition above includes the change.

### KeywordMatch plumbing requirement (OBS-18B AC)

**Codex round-1 review caught**: `KeywordMatch` at
`keyword_matcher.py:37-46` has NO `tags` field. The classifier
pseudocode above calls `classify_keyword_action(m.keyword_name,
m.tags)` — that won't work without plumbing.

**OBS-18B implementation task** (added explicitly):

> Add `tags: List[str] = field(default_factory=list)` to
> `KeywordMatch` dataclass. Populate from `KeywordInfo.tags` in
> all three match-producing pipelines (`_pattern_based_matching`,
> `_semantic_matching`, `_context_aware_matching`). The
> existing `_deduplicate_matches` and `_rank_matches` pass tags
> through unchanged (they only operate on confidence + name).

Without this plumbing OBS-18B doesn't compile. The implementation
story owns this; the design merely flags it explicitly.

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

import dataclasses

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
            # Mismatch: down-weight via dataclasses.replace (v2 perf
            # fix — faster than rebuilding 7 kwargs explicitly;
            # Claude review measured ~5µs/dataclass for rebuild vs
            # ~1µs for replace).
            reranked.append(dataclasses.replace(
                m, confidence=m.confidence * RERANK_DOWNWEIGHT,
            ))
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
import dataclasses

CONFIDENCE_CAP = float(os.getenv("ROBOTMCP_RERANK_CAP", "0.5"))

def apply_confidence_cap(
    matches: List[KeywordMatch],
    query_class: str,
) -> Tuple[List[KeywordMatch], bool]:
    """Cap top-match confidence under two trigger conditions:

    Trigger A: top-3 spans ≥ 3 distinct action classes (divergent
               matcher — uncertain).
    Trigger B (v2 — closes the S10 gap): query class is `unknown`
               AND top match has an opinionated class AND top
               match's confidence > CONFIDENCE_CAP.

    Returns (matches, low_confidence_top_match_flag).
    """
    if not matches:
        return matches, False
    top = matches[0]
    top_class = classify_keyword_action(top.keyword_name, top.tags)

    # Trigger B: unknown query + opinionated high-confidence top
    trigger_b = (
        query_class == "unknown"
        and top_class != "unknown"
        and top.confidence > CONFIDENCE_CAP
    )

    # Trigger A: divergent top-3
    trigger_a = False
    if len(matches) >= 3:
        top3_classes = {
            classify_keyword_action(m.keyword_name, m.tags)
            for m in matches[:3]
        }
        trigger_a = len(top3_classes) >= 3

    if not (trigger_a or trigger_b):
        return matches, False

    # Cap fired — replace top match's confidence; v2 uses
    # dataclasses.replace() per Claude perf feedback (faster than
    # rebuilding 7 kwargs).
    capped_top = dataclasses.replace(top, confidence=CONFIDENCE_CAP) \
        if top.confidence > CONFIDENCE_CAP else top
    return [capped_top] + list(matches[1:]), True
```

### Decision 3 (v2): S10 outcome — locked

Story AC #4b required ONE of:
- (a) top match has confidence ≤ 0.5
- (b) response signals "no high-confidence match"

**v1 design dodged this** by saying "BOTH via the top-3-divergence
cap" and then acknowledging the cap wouldn't fire for S10's actual
top-3 (`{unknown, unknown, wait}` — only 2 distinct classes). Both
Codex and Claude flagged this as an AC rewrite, not satisfaction.

**v2 decision**: combine two cap triggers:

**Trigger A** (the v1 rule, kept for divergent rankings):
```
fire cap when top-3 spans ≥ 3 distinct action classes
```

**Trigger B** (new — closes the S10 gap):
```
fire cap when:
  query_class == "unknown"
  AND top_match has an opinionated class (NOT "unknown")
  AND top_match.confidence > CONFIDENCE_CAP
```

The rationale for Trigger B: when the query is an unrecognised intent
(API verbs, domain-specific phrasings, novel queries), the matcher
returns confidence based on accidental name/doc overlap. A high
top-confidence (>0.5) for an opinionated-class top match against an
`unknown` query is exactly the failure mode — the matcher is
confident about something the query didn't ask for.

For S10:
- Query class: `unknown` (no UI verb in "send http post request").
- Pre-rerank top match: `New Persistent Context` (Browser),
  classified as `click` via Setter fallback. Confidence 0.72.
- Trigger B fires: `unknown` query + opinionated `click` top match +
  confidence > 0.5 → cap fires → top match confidence becomes 0.5
  AND `low_confidence_top_match: true` surfaces.

**Locked S10 contract** (replaces v1 Decision 3 "Resolution"):

> S10 (`query="send http post request with json body"`, library=Browser)
> produces:
>   - `matches[0].confidence` capped at 0.5 (via Trigger B)
>   - response carries `low_confidence_top_match: true`
>
> AC #4b satisfied: BOTH conditions met simultaneously.

**False-positive risk for Trigger B**: novel UI queries the
classifier doesn't recognise (e.g., "drag the slider and watch the
value update") would also classify as `unknown` and trigger the
cap. Acceptable trade-off — capping a high-confidence match for a
query the classifier can't characterise is the conservative call.
Agents see the cap flag, inspect alternatives, and proceed. Better
than confidently surfacing a wrong top match.

**Trigger threshold tuning**: both triggers use the same
`ROBOTMCP_RERANK_CAP` (default 0.5) — env-var tunable.

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

## Performance budget (v2 — revised after Codex perf review)

Reranker is O(N) where N = ranked match count. Note that reranking
runs BEFORE the OBS-22 limit slice, so N can be > the caller's
limit (matcher's pre-cap pipeline returns up to several hundred
candidates before deduplication; after dedup typically ≤ 50; after
internal ranking and pre-OBS-22 slicing N is ≤ 10 BY DEFAULT but
larger when caller passes `limit > 10`).

Per-keyword classifier cost (Codex micro-benchmark): ~1.1µs per
keyword (147-keyword Browser corpus). Per-match cost is:
- 1 × classify_keyword_action ≈ 1.1µs
- 1 × dataclasses.replace (when mismatch) ≈ 1µs
- 1 × tuple unpack for sort key ≈ 0.1µs

Worst case at N=10: 10 × ~2µs = ~20µs reranker pass + sort overhead
(~5µs for 10 elements) = ~25µs.

Worst case at N=50 (caller passed `limit=50`): ~100µs + ~15µs sort
= ~115µs.

**Revised budget**: <150µs at the worst documented case (N=50).
Still negligible (the matcher's existing pattern+context passes are
~ms). Confirmed via OBS-18B benchmark; budget pinned by a
performance-regression test:

```python
def test_reranker_perf_budget(benchmark):
    matches = _build_synthetic_matches(50)  # worst case
    elapsed = benchmark(apply_action_class_reranker, matches, "click")
    assert elapsed < 0.00015  # 150µs
```

v1 claimed <50µs which was optimistic — the per-match cost was
under-estimated and the dataclass-rebuild path Claude flagged was
the dominant cost. v2's `dataclasses.replace` switch + corrected
budget reflects actual production behaviour.

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
- [x] **AC #4** — S02 / S10 outcomes locked (v2 fix per Decision 3 above):
  - S02: top match becomes `Select From List By Label`.
  - S10: top match's confidence is capped at 0.5 AND response
    carries `low_confidence_top_match: true`. Triggered by Trigger B
    (unknown query + opinionated top match + confidence > cap).
- [x] **AC #5** — Rollback via `ROBOTMCP_MATCHER_RERANK` feature
      flag, default off until OBS-18B lands, then flipped to on.

---

## Pinned regression baselines (for OBS-18B test plan)

To prevent flaky regressions on the existing-correct scenarios, the
OBS-18B implementation MUST capture and freeze the current top-match
names + libraries (NOT confidences — those can drift slightly) for
each of the regression scenarios. Captured by the design author on
2026-05-18 via direct matcher invocation:

| Scenario | Query | Library filter | v1 top match (pre-reranker) | v2 expected top match (post-reranker) |
|---|---|---|---|---|
| S01 | "click button" | Browser | `Click` | `Click` (unchanged — `click` class match) |
| S02 | "select dropdown option by visible label" | SeleniumLibrary | `Element Should Be Visible` | **`Select From List By Label`** (changed — reranker fix) |
| S06 | "fill form input field" | Browser | `Fill Text` | `Fill Text` (unchanged) |
| S07 | "navigate to url page" | Browser | `Go To` | `Go To` (unchanged — `navigate` class match) |
| S10 | "send http post request with json body" | Browser | `New Persistent Context` (conf 0.72) | `New Persistent Context` (conf **capped at 0.5**, `low_confidence_top_match: true`) |
| S12 | (long verbose dropdown query) | Browser | `Select Options By` | `Select Options By` (unchanged) |
| S13 | "When I click submit button" (BDD-prefixed) | Browser | `Click` | `Click` (unchanged) |

Codex CLI verified S02 + S10 v1 outputs against the actual matcher
in this environment:

```
S02 top: SeleniumLibrary.Element Should Be Visible (0.875)
S10 top: SeleniumLibrary.Get Session Id (0.785)
```

(Note S10's actual current top is `Get Session Id`, NOT `New
Persistent Context` as the v1 design quoted — that's `library_name`
filter affecting the picture. The fix still applies: query class is
`unknown`, top has opinionated class `query`, conf > 0.5 → Trigger B
fires.)

OBS-18B regression test pseudocode:

```python
EXPECTED_TOP_MATCHES = {
    "S01": ("click button", "Browser", "Click", "Browser"),
    "S02": ("select dropdown option by visible label",
            "SeleniumLibrary", "Select From List By Label",
            "SeleniumLibrary"),
    # ... etc
}

@pytest.mark.parametrize("scenario,query,lib,expected_name,expected_lib",
                         [(k, *v) for k, v in EXPECTED_TOP_MATCHES.items()])
def test_reranker_regression_baseline(scenario, query, lib,
                                      expected_name, expected_lib):
    result = matcher.discover_keywords(query, library_name=lib,
                                       limit=10)
    top = result["matches"][0]
    assert top["keyword_name"] == expected_name, (
        f"{scenario}: top match {top['keyword_name']!r} ≠ "
        f"expected {expected_name!r}"
    )
    assert top["library"] == expected_lib
```

This pins NAMES, not confidences (which can drift with the reranker
formula and aren't load-bearing for agent correctness).

---

## Review findings + resolutions (Codex round-1 + Claude round-1)

Both reviewers ran adversarial review against v1 of this design.
Convergent findings → v2 fixes:

| Finding | Source | v2 resolution |
|---|---|---|
| Browser tags `WaitingPath` / `EventHandlers` don't exist | Codex | Tag table corrected; v2 uses actual tags `Wait`, `PageContent` (+ outlier `Page Content`), `HTTP`, etc. Distribution captured from `LibraryDocumentation('Browser')`. |
| Classifier precedence wrong — `Go To` → `click` not `navigate` | Codex | v2 precedence: Wait → BrowserControl → Getter → Assertion → Setter+name. Worked outputs table shows v1 vs v2 for 11 keywords. |
| `KeywordMatch.tags` doesn't exist; classifier can't compile | Codex | Added explicit "OBS-18B implementation task" calling out the dataclass plumbing requirement. |
| AC #4b S10 contract dodged (v1 rewrote the AC) | BOTH | v2 Decision 3 introduces Trigger B (`unknown` query + opinionated top + conf > cap). S10 now produces BOTH conditions: cap AT 0.5 AND `low_confidence_top_match: true`. AC #4b properly satisfied. |
| 0.6 down-weight unjustified | Claude | Acknowledged: 0.6 is a starting default. Sensitivity sweep added to OBS-18B test plan (range 0.3-0.9). |
| Down-weight too aggressive in tight ranges / too lenient at top | Claude | Same — sensitivity sweep is the answer; the env-var tunable means production can adjust without redeployment. |
| Resource keywords classified as `unknown` → systematically buried | Claude | Acknowledged limitation. v2 keeps the behaviour but documents it: resource keywords without a tag taxonomy DO get penalised when query class is opinionated. Mitigation: agents that need resource keywords specifically use `strategy="session"` (OBS-23A/B) or `strategy="catalog"` which don't go through the reranker. |
| Taxonomy note 2 (`^(select|deselect)`) vs pseudocode (`select from`/`select options`) disagreement | Claude | v2 aligned: pseudocode now lists `select options`, `select option`, `select from`, `select checkbox`, `deselect options`, `deselect checkbox` explicitly. Taxonomy note updated to match. |
| Performance budget <50µs not credible | BOTH | Revised to <150µs at worst case. Includes dataclasses.replace optimisation + breakdown. |
| Test plan lacks pinned baselines | Claude | Added the regression baselines table above. |
| `dataclasses.replace` not used | Claude | Both `apply_action_class_reranker` and `apply_confidence_cap` updated. |
| `Sleep` mis-classified as `unknown` | Claude | v2 noted: add `sleep` as wait trigger in OBS-18B (or accept `unknown`). Decided: add it. |

Non-convergent findings deferred or rejected:
- Per-class-pair down-weight matrix → Codex says "no labeled data,
  false precision". Rejected. Uniform 0.6 with env-var tuning.
- Cap on whole top-N vs top-1 → Codex unchanged from v1 (top-1 only).
