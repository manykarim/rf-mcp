# PRD — `analyze_scenario` Explicit Library Preference Detection

**Date**: 2026-05-29
**Status**: **IMPLEMENTED v7 — production-ready** (per Codex round-7 review 2026-06-05). Source in `src/robotmcp/utils/library_detection.py`, `nlp_processor.py`, `session_models.py`, `server.py`. Tests in `tests/unit/test_explicit_library_detection_fix.py` — **83 targeted tests; full unit suite 6249 passing**.
**Author**: rf-mcp maintainer
**Reporter**: User report on 2026-05-29
**Related**: ADR-024 (companion architecture decision), DDD library_preference bounded context, solution proposal `docs/proposals/explicit_library_detection_fix_proposal.md`

## Revision history

**v7 (2026-06-05)** — Codex round-6 review verified all 4 v6 source fixes but flagged 3 new issues (1 paragraph-boundary regression introduced by v6, 1 nlp_processor helper still had the abstain-fallback bug, 1 pre-existing sampling-coherence issue). v7 addresses 2:
- **C1 — Paragraph boundary**: `_SENTENCE_DELIMITERS` changed from `r"[.;!?,]+"` to `r"[.;!?,]+|\n\s*\n+"`. Single `\n` stays non-boundary (preserves v6 D-newline-negation fix); `\n\s*\n+` (paragraph break) IS a boundary (fixes the v6 paragraph regression).
- **C3 — NLP helper abstain-fallback removed**: `nlp_processor._detect_explicit_library_preference` no longer falls back to broad substring on deliberate None. Mirrors the v6 session_models fix.
- **C2 deferred**: sampling Site B session-coherence is a pre-existing architectural issue, scoped to its own follow-up.
- **Tests**: 14 new v7 tests in `TestV7ParagraphBoundary` (7), `TestV7ParagraphBoundaryEndToEnd` (1), `TestV7NlpProcessorAbstainNoFallback` (6). **Total 83 in the v7 file; full unit suite 6249 passing.**

**v6 (2026-06-05)** — Codex round-5 review verified v5 fixed the analysis-path defect but flagged 4 source bugs + propagation gaps in PRD/ADR. v6 fixes all 4 source bugs (3 critical) and aligns docs:
- **D-session-fallback fixed**: `session_models.py:detect_explicit_library_preference` no longer falls back to the broad substring heuristic when v5 returns None. Session aggregate now correctly reports None when the v5 detector abstains. Tested at `TestV6SessionFallbackBug` (3 cases).
- **D-newline-negation fixed**: `\n` dropped from `_SENTENCE_DELIMITERS`. Tested at `TestV6NewlineNegation` (4 parametrised cases + preference-verb-newline regression).
- **D-repeated-token fixed**: `_subtract_sentence_score` counts occurrences via `findall`. Tested at `TestV6RepeatedTokenDeduction` (4 cases).
- **D-multiword-migration fixed**: destination lookahead allows multi-word capture. `Requests library` / `Database library` / `SSH library` / `XML library` all resolve correctly. Tested at `TestV6MultiwordMigration` (6 cases).
- **§FR-5 evidence field corrected**: evidence entries use singular `pattern` (not `patterns_matched`). `library_preference_conflicts[]` entries DO use `patterns_matched` (list of pattern source strings).
- **§AC-7 RequestsLibrary example corrected**: documented pattern now matches source (no bare `requests` in preference verb).
- **Total tests**: 69 in `test_explicit_library_detection_fix.py` (51 v5 + 18 v6). Pre-existing tests unchanged: 6166 passing.

**v5 (2026-06-05)** — Codex round-4 review (verdict v4: ADR/proposal REWRITE, PRD/DDD TIGHTEN) identified 5 propagation gaps + 4 new v4 regressions. v5 implements the fix in source and addresses all of them:
- **§FR-1 bare-noun preference verbs propagated**: removed `(requests|database|ssh|xml)` from preference-verb alternations. Brand names (`selenium`, `playwright`, `appium`) retained. Aligned with proposal v4 §3.1 Step 2 (and verified in `library_detection.py:LIBRARY_RULES_DEFAULT`).
- **§FR-6 negation algorithm propagated**: replaced the stale `\b(not|don't|do\s+not|...)\s+(?:using\s+)?(\w+)` capture-based regex prose with the **single-alternation `_NEGATION_REGEX` + `_first_library_token_in(remaining_text)` two-step approach** that actually shipped. Sentence delimiter is `[.;!?,\n]+` (comma + newline both included). `skip` restored to standalone alternation.
- **§FR-7 sampling override coherence**: `primary_library` fallback dropped per Codex finding — only `library_preference` writes to `explicit_library_preference` (the field is "user-stated", `primary_library` is "LLM-recommended"; different semantics). Both override sites (`server.py:1860-1881` Site A + `server.py:1961-1972` Site B) implemented.
- **§AC-5 numerics re-verified**: under the v5 pattern table, "Test both selenium and playwright sites and compare" → SL=6, Browser=9, diff=3 ≤ 4 → conflict fires. AC holds. **Tested via `TestConflicts::test_both_selenium_and_playwright_returns_none_with_conflict`.**
- **§9 stakeholder table aligned**: "No existing tests should break" was always wrong — reconciled to "3-5 existing tests need updating + 18 new tests added" (already in v4; confirmed in v5 implementation: 51 new tests added in `test_explicit_library_detection_fix.py`).
- **Implementation status promoted to IMPLEMENTED**: source code now reflects the spec. The PRD/ADR/DDD/proposal are documentation of what the implemented system does, not a forward design.

**v4 (2026-05-29)** — Third-round independent review (verdict: TIGHTEN for PRD). v4 addresses:
- **Test count reconciled**: FR-7 and AC-8 already say "18 distinct functions" — no change. Proposal §9 Validation Plan updated to match (was "~30 + ~3"; now "18 + 3-5"). PRD itself unchanged for this item.
- **`session.preference_source` field declaration referenced**: §2 downstream-consumer table can now reference the v4 field declaration in proposal §3.3 — no direct PRD change needed (the field is a proposal-level mechanism, not a PRD-level requirement).
- **No other PRD-level changes required**: the round-3 review's PRD finding was "no issues, just propagate proposal fixes". Round-3 report verified PRD is correct as-is at v3.

**v3 (2026-05-29)** — Codex CLI second-round critical review (verdict: TIGHTEN for PRD). v3 addresses:
- **§9 vs AC-8 contradiction resolved**: stakeholder table no longer claims "No existing tests should break". v3 honestly says "3-5 existing tests need updating; ~18 new tests added".
- **Evidence shape standardised (§FR-5)**: flat list of `{library, pattern, weight, text_span}` entries — matches ADR §3.5, DDD §4.1.6 PatternMatch, proposal Step 4. v2 had a nested per-library shape that diverged from the other docs.
- **AC-7 verbatim recomputation**: v3 verified pattern-table arithmetic still produces correct results under the v3 pattern table changes (bare `requests` removed from preference verb).
- **Field name `patterns_matched` (conflicts) and per-evidence `pattern` (evidence)**: confirmed consistent across all 4 docs.
- **Sampling override coherence (§FR-7)**: now references BOTH override sites at `server.py:1860-1881` AND `1961-1972` per v3 proposal §3.2.1.
- **Architecture correction reinforced**: §13 already correctly notes `_determine_capabilities` is separate; no change needed here.
- **Test count updated to 18** (matches proposal §5 honest count).

**v2 (2026-05-29)** — Codex CLI critical review identified 15 substantive issues with v1. Key fixes:
- **Scope expansion**: bug affects ALL 7 libraries, not just Selenium. Updated requirements + ACs accordingly.
- **AC-4 fixed**: negation pattern was wrong (current regex requires contiguous `migrate from`; missed real cases). Replaced with sentence-scoped source-vs-destination resolution.
- **AC-5 fixed**: conflict-vs-threshold ordering swapped. Conflict check now runs on RAW scores BEFORE threshold filter, so `"both selenium and playwright"` actually returns None+conflict as documented.
- **AC-7 fixed**: `status code` is mention-only per design; replaced AC-7's example with a pattern that's actually in the explicit set.
- **AC-8 reconciled**: PRD no longer claims "all existing tests pass" — explicitly acknowledges 3-5 existing test updates needed and lists them.
- **Architecture claim corrected**: `suggested_libraries` does NOT come from `LibraryDetector` — it's a separate substring heuristic (`_determine_capabilities`). Removed wrong cross-reference.
- **Field name standardised**: `patterns_matched` (was inconsistent across docs).
- **Sampling override coherence defined**: when sampling overrides preference, evidence is cleared and `preference_source: "sampling"` is set.
- **Downstream consumers expanded**: library_recommender, adapter_factory, keyword_executor, browser_plugin, selenium_plugin added to the impact analysis.
- **Performance claim removed**: replaced unsupported "≤60µs" with "no measurable regression vs baseline".

**v1 (2026-05-29)** — initial draft. Identified the reported defect, proposed mention-vs-preference split.

---

## 1. Problem statement

`analyze_scenario` reports `explicit_library_preference: "SeleniumLibrary"` for scenarios that never explicitly request SeleniumLibrary. Reproduced with the exact user-supplied input:

```json
{
  "scenario": "Test e-commerce website https://demoshop.makrocode.de: open browser, add items to shopping cart, verify items, complete checkout, and close browser",
  "context": "web",
  "session_id": "demoshop_test"
}
```

Response includes:
```json
"analysis": {
  "explicit_library_preference": "SeleniumLibrary",
  "suggested_libraries": ["Browser", "SeleniumLibrary", "RequestsLibrary"],
  ...
}
```

The user wrote "open browser" as a generic English verb for "launch a web browser session". They did NOT mention `selenium`, `webdriver`, `seleniumlibrary`, or any other Selenium-explicit phrase. The system nevertheless classified this as an *explicit* preference.

Root cause traced to `src/robotmcp/utils/library_detection.py:33` — the pattern `\bopen\s+browser\b` is weighted 6 against SeleniumLibrary (because `Open Browser` is the SL keyword name) and the default min_score is 5. The phrase fires the threshold by itself with no other Selenium signal in the text.

**Scope: same bug class affects ALL seven libraries**, not just SeleniumLibrary. Verified false-positive triggers (raw scores against the current pattern table):

| Library | Triggering NL phrase | Score | Threshold | Fires |
|---|---|---|---|---|
| SeleniumLibrary | "open browser" | 6 | 5 | YES |
| Browser | "Open new page in browser context" | 8 | 5 | YES |
| AppiumLibrary | "Open application on the device" | 8 | 5 | YES |
| SSHLibrary | "Execute command on the remote server" | 11 | 5 | YES |
| Database | "Check if exists and verify row count" | 14 | 5 | YES |
| RequestsLibrary | "Verify status code returned by service" | 5 | 5 | YES |
| XML | "Parse the XML response from the API" | 7 | 5 | YES |

The fix must address the entire pattern table, not just the Selenium rows. v1 framed this as a Selenium-only problem; v2 broadens scope.

## 2. Why this matters

Downstream consequences of a false `explicit_library_preference`. Each row identifies a call-site that reads the field and changes behaviour based on it.

| # | Consumer (file:lines) | What it does with the field | Failure mode when value is wrong |
|---|---|---|---|
| 1 | `server.py:2037` `_filter_keywords_by_session_library` | Drops the OTHER library's keywords from `find_keywords` results | Agents see SL keywords only; write SL code; fail at `execute_step` against a Browser session (or vice versa) |
| 2 | `session_models.py:787` `ExecutionSession.configure_from_scenario` | Eagerly loads the detected library at session init | Loads SeleniumLibrary → WebDriver init + system check; 1-3s wasted; misleading errors |
| 3 | `components/library_recommender.py:111-166` | Adds the explicit preference to the front of the recommendation list with high confidence | `recommend_libraries` returns SL as primary recommendation when the user wanted Browser |
| 4 | `components/library_recommender.py:310-321` | Excludes the other library from candidates | Browser library not even surfaced in the alternatives list |
| 5 | `adapters/adapter_factory.py:131-140` | Selects the platform adapter (Browser vs SL) | Adapter picks SL-specific paths (e.g., WebDriver wait policies); Browser-loaded session gets SL semantics |
| 6 | `components/execution/keyword_executor.py:1901-1914` | Disambiguates same-named keywords (e.g., `Take Screenshot`) by preferring the explicit library | `Take Screenshot` resolves to SL when Browser is also loaded — fails because no SL session exists |
| 7 | `plugins/builtin/browser_plugin.py:314-321,379-390` | Suppresses Browser auto-init when SL is preferred | Browser session never initialized; `execute_step` errors with "No Browser session" on a Browser-intent scenario |
| 8 | `plugins/builtin/selenium_plugin.py:202-208` | Eagerly initializes SL when it's the explicit preference | SL session pre-created and tied to the wrong driver type; conflicts with later Browser intent |

Plus the **contract violation**: the field is called `explicit_library_preference`. Users and agents reasonably interpret "explicit" as "the user said so explicitly". The current implementation conflates "the user used English phrases that happen to overlap with the library's keyword names" with "the user explicitly chose that library".

## 3. Goals

- The `explicit_library_preference` field must fire **only** when the user has explicitly mentioned a specific RF library. Implicit signals (e.g., "open browser", "click element", "input text") must NOT trigger explicit preference, regardless of which library's keyword names they happen to match.
- When the scenario is ambiguous between libraries (Browser vs SeleniumLibrary for web work), the field must return `None` rather than picking one arbitrarily by pattern-table ordering.
- When the scenario has signals for multiple libraries (e.g., both `playwright` and `selenium` mentioned), the field must surface the conflict rather than silently picking one.
- The fix must not break existing tests that assert explicit detection on truly-explicit scenarios (e.g., "use Selenium to test ...").

## 4. Non-goals

- We do NOT need to extend `_determine_capabilities` (the `suggested_libraries` field) — it's allowed to be generous and suggest both Browser and SeleniumLibrary for a generic "test website" scenario. Only the *explicit* path needs tightening.
- We do NOT need to replace pattern matching with an LLM call. The sampling path (`sample_analyze_scenario` in server.py:1863) already exists and can override the rule-based detection when the operator opts in.
- We do NOT need to detect library preference from action verbs (click/fill/input) at all. Those signals belong to the capability/session-type detection, not the explicit-preference path.

## 5. Functional requirements

### FR-1 — Explicit phrases only (all 7 libraries)

The detector must annotate every pattern with an `explicit: bool` flag and only patterns where `explicit=True` contribute to the `explicit_library_preference` decision. All patterns remain in the table for the **mention** layer (used by callers that want a broad "what topics did the user mention" signal — see §5 FR-7 below for scope of consumers).

**Patterns marked `explicit=True` (KEEP, contribute to preference):**

| Class | Examples | Rationale |
|---|---|---|
| Library name verbatim | `\bseleniumlibrary\b`, `\bplaywright\b`, `\brequestslibrary\b`, `\bwebdriver\b`, `\brfbrowser\b`, `\brobotframework[- ]browser\b`, `\bappium(?:library)?\b`, `\bsshlibrary\b`, `\bdatabaselibrary\b`, `\bxmllibrary\b` | The library is named by its actual identifier |
| Preference verb + library token | `\b(use\|using\|with\|via\|through\|prefer)[^\S\n]+(selenium\|playwright\|appium\|seleniumlibrary\|browserlibrary\|requestslibrary\|appiumlibrary\|databaselibrary\|sshlibrary\|xmllibrary)\b` | v5: brand names + verbatim library names only. Bare generic nouns (`browser`, `database`, `ssh`, `requests`, `xml`) deliberately NOT in this alternation — they're domain words, not library identifiers. `[^\S\n]+` (not `\s+`) so newlines act as sentence boundaries. |
| Library-specific tech terms | `\bchromedriver\b`, `\bgeckodriver\b`, `\bchromium\b`, `\bwebkit\b`, `\bplaywright-?core\b`, `\biwebdriver\b` | Unambiguously names the library's runtime |

**Patterns marked `explicit=False` (KEEP in mention layer, EXCLUDE from preference):**

These are the noise patterns that triggered the reported defect. They overlap with keyword names or generic English domain prose but DO NOT indicate an explicit library choice. In v1 some of these were proposed for removal; v2 keeps them in the mention table (so the capability-suggestion path can still see "the user is talking about a browser") but removes their contribution to `explicit_library_preference`.

- Generic action verbs that overlap with keyword names: `\bopen\s+browser\b`, `\binput\s+text\b`, `\bclick\s+element\b`, `\bpage\s+should\s+contain\b`, `\bnew\s+(browser|page|context)\b`
- Generic technique terms: `\b(implicit|explicit)\s+wait\b`, `\b(SPA|single\s+page\s+app(lication)?)\b`, `\b(e2e|end.to.end)\s+(test|automat)\b`, `\bcross[- ]browser\s+testing\b`
- Domain markers (Codex flagged these as wrongly annotated `explicit=True` in v1): `\brest\s+api\s+testing\b`, `\bmobile\s+(automation|testing)\b`, `\b(android|ios)\s+(testing|test)\b`, `\b(SELECT|INSERT|UPDATE|DELETE)\s+(\*|FROM|INTO)\b`, `\b(postgres(ql)?|mysql|sqlite|oracle|mariadb|mssql|sqlserver)\b`, `\bstatus\s+code\b`, `\bxpath\b`, `\bxsd\b`
- SSH-domain noise: `\bexecute\s+command\b`, `\bremote\s+(server|host|machine)\b`, `\bssh\s+(into|to)\b`
- Mobile-domain noise: `\bopen\s+application\b`, `\bdevice\b` (alone)
- DB-domain noise: `\b(check|verify)\s+if\s+exists\b`, `\brow\s+count\b`, `\bquery\s+(the|a)?\s*database\b`

These patterns can still inform `suggested_libraries` indirectly via `_determine_capabilities` (a SEPARATE substring heuristic at `nlp_processor.py:517-544`, not driven by `LibraryDetector` — see Architecture Correction in §13).

**Non-English limitation (deliberate)**: all patterns are English-only. Scenarios written in other languages will yield `None` for `explicit_library_preference`. This is acceptable because (a) the existing baseline is also English-only and (b) the field is opt-in semantics — `None` is the safe default. Future work could add a per-locale pattern set; out of scope here.

### FR-2 — Confidence threshold

The detector uses a single `min_score` threshold. After FR-1 removes noise contributions, a weight-6 pattern firing means the user wrote either a library name verbatim (`\bselenium\b` at weight 6) or used a tech-specific term — both legitimate signals. The threshold can stay at 5 for libraries with no competitor in a conflict group (Database, SSH, XML), but rises to **8** for libraries inside a conflict group (Browser, SeleniumLibrary).

Configurable via env vars (defaults shown):
- `ROBOTMCP_LIBRARY_DETECTION_MIN_SCORE=5` (out-of-group libraries)
- `ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD=8` (within-group libraries)
- `ROBOTMCP_LIBRARY_DETECTION_AMBIGUITY_WINDOW=4` (see FR-3)

### FR-3 — Abstain on ambiguity (algorithm ordering)

**Critical**: conflict detection runs on **raw explicit scores BEFORE threshold filtering**. This is the v1→v2 fix that makes AC-5 actually work. Pseudocode:

```
raw_scores = {lib: sum(weights of matched explicit=True patterns)}
apply_negation_and_migration(raw_scores)   # see FR-6

for group in CONFLICT_GROUPS:
    libs_with_signal = [lib for lib in group if raw_scores[lib] > 0]
    if len(libs_with_signal) >= 2:
        top2 = sorted(libs_with_signal, key=raw_scores.get, reverse=True)[:2]
        if raw_scores[top2[0]] - raw_scores[top2[1]] <= ambiguity_window:
            return ConflictResult(None, conflicts={group: scores_for_libs(top2)})

# No conflict — apply threshold and pick winner
effective_threshold = conflict_threshold if lib_in_group else min_score
candidates = {lib: s for lib, s in raw_scores.items() if s >= effective_threshold}
if not candidates:
    return None
return max(candidates, key=candidates.get)
```

Worked examples (with `min_score=5`, `conflict_threshold=8`, `ambiguity_window=4`):

| Scenario | SL raw | Browser raw | Group? | Diff | Pre-threshold conflict? | Post-threshold result |
|---|---|---|---|---|---|---|
| "use playwright" | 0 | 10 (`use ... playwright`) + 9 (`playwright`) = 19 | Yes | 19 | no (one lib only) | Browser ≥ 8 → Browser |
| "use Selenium 4" | 10 + 6 + 8 = 24 | 0 | Yes | 24 | no | SL ≥ 8 → SeleniumLibrary |
| "Run a Selenium test" | 6 | 0 | Yes | 6 | no | SL 6 < 8 → None |
| "both selenium and playwright" | 6 | 9 | Yes | 3 ≤ 4 | **YES → conflict** | None + conflicts={web_automation: [SL=6, Browser=9]} |
| "Test e-commerce site, open browser, add to cart" (the bug) | 0 (after FR-1) | 0 (after FR-1) | n/a | 0 | no | None |
| "Parse XML using xmllibrary" | 0 | 0 | n/a | n/a | n/a | XML 10+8 = 18 ≥ 5 → XML |

The pre-threshold conflict check is what makes the `"both selenium and playwright"` example work. Under v1's design (conflict-after-threshold), SL=6 would be filtered out by threshold=8 before the conflict check ran, leaving Browser=9 as a single candidate → no conflict → wrong answer.

### FR-4 — Conflict surfacing in the response

When the conflict check fires, the `analysis` block must include a `library_preference_conflicts` field listing the conflicting candidates with their scores and the patterns that matched.

Example (after fix):
```json
"analysis": {
  "explicit_library_preference": null,
  "preference_source": "rule",
  "library_preference_conflicts": {
    "web_automation": [
      {"library": "SeleniumLibrary", "score": 6, "patterns_matched": ["\\bselenium\\b"]},
      {"library": "Browser", "score": 9, "patterns_matched": ["\\bplaywright\\b"]}
    ]
  },
  "suggested_libraries": ["Browser", "SeleniumLibrary"]
}
```

Field-name standardisation (v6 final): `patterns_matched` is used in **`library_preference_conflicts[]` entries** (list of pattern source strings). **Evidence entries use singular `pattern`** (single source string per match) — verified in `src/robotmcp/utils/library_detection.py:PatternMatch.to_dict()` and `tests/unit/test_explicit_library_detection_fix.py::TestEvidenceShape`. v5 PRD wrongly conflated the two; v6 separates them.

### FR-5 — Verbatim provenance (v3 canonical shape)

When `explicit_library_preference` is set by the rule-based detector, the response must include `explicit_library_evidence` — a **flat list** of pattern-match entries with the canonical shape used across all four docs:

```json
"explicit_library_preference": "SeleniumLibrary",
"preference_source": "rule",
"explicit_library_evidence": [
  {
    "library": "SeleniumLibrary",
    "pattern": "\\b(use|using|with|via|through|prefer)\\s+(selenium|seleniumlibrary|selenium\\s*library)\\b",
    "weight": 10,
    "text_span": "use selenium"
  },
  {
    "library": "SeleniumLibrary",
    "pattern": "\\bselenium\\b",
    "weight": 6,
    "text_span": "selenium"
  }
]
```

Each entry: `{library: str, pattern: str (regex source), weight: int, text_span: str (matched substring)}`. The list is FLAT (one entry per matching pattern); the `library` field per entry lets consumers filter or group. This matches DDD §4.1.2 `PatternMatch` and ADR §3.5.

### FR-6 — Negation and migration (sentence-scoped, source-vs-destination aware)

The v1 design proposed an 80-character forward window. Codex flagged that this breaks `"do not use Selenium, instead use Playwright"` (the window covers both library mentions, zeroing both). v2 replaces this with a **sentence-scoped algorithm that distinguishes source from destination**:

```python
# v7 IMPLEMENTED in src/robotmcp/utils/library_detection.py

# v7: sentence punctuation OR paragraph break (2+ newlines).
# Single newlines are NOT boundaries — they keep negation contiguous with its
# target ("do not use\nPlaywright" → None). Paragraph breaks ARE boundaries
# ("Do not use this approach\n\nUse Playwright" → Browser).
# Lineage: v5 had \n in delim (orphaned negation), v6 dropped \n entirely
# (broke paragraphs), v7 uses paragraph-aware alternation.
_SENTENCE_DELIMITERS = r"[.;!?,]+|\n\s*\n+"

# v5: SINGLE alternation regex, longest-first. Regex engine guarantees one
# match per span — no double-deduction. `skip` restored as standalone.
_NEGATION_REGEX = re.compile(
    r"\b(?:"
    r"do\s+not\s+use|don't\s+use|"
    r"not\s+using|without\s+using|stop\s+using|avoid\s+using|"
    r"skip\s+using|exclude\s+using|"
    r"do\s+not|don't|"
    r"without|stop|avoid|skip|exclude"
    r")\b",
    re.IGNORECASE,
)

# v5: source-vs-destination resolution via _first_library_token_in() on the
# captured spans (NOT raw \w+ capture, which broke multi-word library names).
MIGRATION_PATTERNS = [
    r"\bmigrat(?:e|ion|ing)\b.+?\bfrom\b\s+(?P<src>.+?)\s+\bto\b\s+(?P<dst>.+?)(?=[\s.,;!?]|$)",
    r"\bswitch(?:ing)?\s+from\b\s+(?P<src>.+?)\s+\bto\b\s+(?P<dst>.+?)(?=[\s.,;!?]|$)",
    r"\binstead\s+of\b\s+(?P<src>.+?)\s+\b(?:use|with|via)\b\s+(?P<dst>.+?)(?=[\s.,;!?]|$)",
    r"\breplace\b\s+(?P<src>.+?)\s+\b(?:with|by|for)\b\s+(?P<dst>.+?)(?=[\s.,;!?]|$)",
]

for sentence_index, sentence_span in enumerate(_build_sentence_spans(text)):
    sent = text[sentence_span.start:sentence_span.end]

    # Step A — migration (source/destination resolution)
    for pattern in MIGRATION_PATTERNS:
        for m in re.finditer(pattern, sent, re.IGNORECASE):
            src_lib = _first_library_token_in(m.group("src"))
            dst_lib = _first_library_token_in(m.group("dst"))
            if src_lib:
                _subtract_sentence_score(raw_scores, matches_by_lib,
                                         sent, sentence_index, src_lib)
            if dst_lib:
                raw_scores[dst_lib] += 5  # destination bonus

    # Step B — single-library negation (SINGLE alternation regex)
    for m in _NEGATION_REGEX.finditer(sent):
        remaining = sent[m.end():]
        target_lib = _first_library_token_in(remaining)
        if target_lib:
            _subtract_sentence_score(raw_scores, matches_by_lib,
                                     sent, sentence_index, target_lib)
```

This is the actually-shipped v5 implementation. v4's documented `\w+` capture pattern was the round-3-flagged D1 bug; v5 uses phrase-list + token-resolution to avoid capturing the wrong word as the negation target.

Worked examples:

| Sentence | Migration match | Negation match | Source effect | Dest effect | Net |
|---|---|---|---|---|---|
| "Migrate the test suite from Selenium to Playwright" | from=Selenium, to=Playwright | — | SL: -all-in-sentence | Browser: +5 | Browser wins |
| "do not use Selenium, instead use Playwright" | (split on comma) S1: instead-of variant doesn't match (no "of"); S2: bare "use Playwright" | S1: matches "not use Selenium" | SL: -all-in-S1 | Browser: +10 in S2 | Browser wins |
| "without selenium" | — | matches "without selenium" | SL: -all | — | SL = 0 |
| "Use Playwright for the e2e test" | — | — | — | Browser: +10 + +9 | Browser wins (no negation) |

The negation handling **must preserve** the AC-4 outcome (Browser wins on migration), which v1's regex couldn't deliver because it required contiguous "migrate from".

### FR-7 — Sampling override coherence (v3 — TWO sites)

LLM sampling overrides the rule-based detector at **two independent sites** in `server.py` (v2 documented only the first; Codex v2-round-2 review surfaced the second):

| Site | File:lines | Effect | Coherence protocol |
|---|---|---|---|
| A | `server.py:1860-1881` | Modifies `analysis["explicit_library_preference"]` in the analyze_scenario response | Set `analysis.preference_source="sampling"`, clear `explicit_library_evidence` + `library_preference_conflicts`, populate `sampling_evidence` |
| B | `server.py:1961-1972` | Modifies `session.explicit_library_preference` on the ExecutionSession aggregate | Set `session.preference_source="sampling"` |

Both sites are gated by `is_sampling_enabled()` reading the `ROBOTMCP_USE_SAMPLING` env var (verified at `sampling.py:23`).

When sampling does NOT override (or is disabled):
- `analysis.preference_source` ← `"rule"`
- `session.preference_source` ← `"rule"`
- Evidence/conflicts populated by the rule-based detector as in FR-4/FR-5

This contract makes it unambiguous to all 8 downstream consumers (listed in §2) which signal produced the preference and where to look for the reasoning.

### FR-8 — Mention layer scope clarification

v1 conflated two systems. v2 makes the boundary explicit:

| System | File:lines | Input | Output | Used by |
|---|---|---|---|---|
| Preference layer (this PRD) | `library_detection.py` + `nlp_processor._detect_explicit_library_preference` | scenario text | `analysis.explicit_library_preference` + evidence + conflicts | All 8 downstream consumers from §2 |
| Capability layer (untouched) | `nlp_processor._determine_capabilities` lines 517-544 | scenario text | `analysis.suggested_libraries` (broad advisory list) | Session import hints only |

The capability layer is a **separate substring heuristic** that does NOT call `LibraryDetector`. v1's docs implied a shared codepath; that was wrong. The capability layer can remain generous (suggest both Browser and SL for a vague web scenario) because it's advisory; only the preference layer needs the tighter explicit/mention distinction this PRD specifies.

## 6. Non-functional requirements

- **NFR-1 Backward compatibility**: most tests calling `_detect_explicit_library_preference` for truly-explicit scenarios ("use Selenium", "with playwright") continue passing unchanged. A small number of existing tests assert false-positive behaviour and must be updated — see AC-8 + §13 for the list. Response shape adds three optional fields (`library_preference_conflicts`, `explicit_library_evidence`, `preference_source`); does not remove or rename existing fields. The `_compiled_patterns` internal attribute remains (with annotated entries) because integration tests inspect it directly (`tests/integration/test_nlp_improvements.py:103-104,571-576`).
- **NFR-2 Performance**: pattern matching is O(N × P) where N is text length and P is patterns. The fix adds the `explicit: bool` flag check per match (constant per pattern) and adds sentence splitting + per-sentence negation/migration application. No microbenchmark target — the requirement is **no measurable regression** vs the current baseline (`tests/benchmarks/test_library_detection_bench.py` continues to pass within existing tolerances).
- **NFR-3 Determinism**: same input → same output. No randomness, no clock dependency.
- **NFR-4 Observability**: detection logs at INFO level when a preference is set, at DEBUG when conflicts arise. New log line at DEBUG when a migration pattern fires (`Migration detected: source=X destination=Y`).
- **NFR-5 Test surface compatibility**: existing tests inspect `LibraryDetector._compiled_patterns` directly. The fix annotates entries (adds the `explicit` flag) without removing the attribute or changing its key types. Existing assertions on pattern presence continue to work.

## 7. Acceptance criteria

The fix is accepted when ALL of these hold:

### AC-1 — Reported scenario

`analyze_scenario(scenario="Test e-commerce website https://demoshop.makrocode.de: open browser, add items to shopping cart, verify items, complete checkout, and close browser", context="web")` returns:
- `analysis.explicit_library_preference` == `None`
- `analysis.suggested_libraries` may contain Browser, SeleniumLibrary (advisory list — unchanged)
- No false positive for SeleniumLibrary

### AC-2 — Browser-explicit scenario

`analyze_scenario(scenario="Use playwright to test the checkout flow on demoshop.makrocode.de", context="web")` returns:
- `analysis.explicit_library_preference` == `"Browser"`
- `analysis.explicit_library_evidence` lists the `playwright` pattern hit

### AC-3 — Selenium-explicit scenario

`analyze_scenario(scenario="Use Selenium to test the login form against the e-commerce site", context="web")` returns:
- `analysis.explicit_library_preference` == `"SeleniumLibrary"`
- `analysis.explicit_library_evidence` lists the `use selenium` pattern hit

### AC-4 — Conflict scenario

`analyze_scenario(scenario="Migrate the test suite from Selenium to Playwright for the demoshop site")` returns:
- `analysis.explicit_library_preference` == `"Browser"` (the destination of the migration; "migrate from selenium" subtracts Selenium score per negation patterns, leaving Playwright alone above threshold)
- `analysis.library_preference_conflicts` is absent (Selenium was negated out, no conflict)

### AC-5 — Genuine conflict (rare but pinned)

`analyze_scenario(scenario="Test both selenium and playwright sites and compare")` returns:
- `analysis.explicit_library_preference` == `None`
- `analysis.preference_source` == `"rule"`
- `analysis.library_preference_conflicts.web_automation` lists both Browser and SeleniumLibrary with their respective scores

Verification under the algorithm in FR-3 (raw-score conflict check BEFORE threshold filter):
- SL raw = 6 (`\bselenium\b`)
- Browser raw = 9 (`\bplaywright\b`)
- Both > 0, |9 − 6| = 3 ≤ 4 (ambiguity_window) → conflict fires → None + conflicts populated

This AC was impossible under v1's algorithm ordering (SL=6 would be filtered before conflict check). v2's ordering makes it work.

### AC-6 — Pure NL phrase scenarios (no library mention)

For each of:
- "click element by id submit"
- "Page should contain Welcome text"
- "Input text into the username field"
- "Open new page in browser context"
- "Execute command on the remote server" (was SSH false positive)
- "Open application on the device" (was Appium false positive)
- "Check if exists and verify row count" (was Database false positive)
- "Verify status code returned by service" (was Requests false positive)
- "Parse the XML response from the API" (was XML false positive)

The detector returns `explicit_library_preference: None`. (Currently all nine return a false library per the §1 bug-class table.)

### AC-7 — No regression for explicit library mentions

`analyze_scenario(scenario="Use requestslibrary to POST a JSON body to /api/users")` returns:
- `analysis.explicit_library_preference` == `"RequestsLibrary"` (matches the `\b(use|using|with|via|through|prefer)[^\S\n]+(requestslibrary|requests\s*library)\b` preference-verb pattern. v6 NOTE: bare `requests` is NOT a valid preference-verb target — see `library_detection.py:LIBRARY_RULES_DEFAULT['RequestsLibrary']`)

`analyze_scenario(scenario="Parse XML response using xmllibrary and validate xpath /root/element")` returns:
- `analysis.explicit_library_preference` == `"XML"` (matches `\bxmllibrary\b` verbatim)

`analyze_scenario(scenario="Use AppiumLibrary to test the iOS app")` returns:
- `analysis.explicit_library_preference` == `"AppiumLibrary"` (matches `\bappium(?:library)?\b` verbatim + preference-verb pattern)

`analyze_scenario(scenario="SSH into the build server using SSHLibrary")` returns:
- `analysis.explicit_library_preference` == `"SSHLibrary"` (matches `\bsshlibrary\b` verbatim)

v1 used `"verify status code"` which is mention-only per FR-1; v2 swaps to scenarios that actually mention the library.

### AC-8 — Test suite passes; existing false-positive assertions updated

**Tests that pass unchanged** (the "use X" / "with Y" style scenarios — happy path):
- `tests/unit/test_library_detection.py` — most cases
- `tests/integration/test_nlp_improvements.py:103-104,571-576` — `_compiled_patterns` inspection (entries keep their pattern strings; only the `explicit` flag is added)

**Tests that need updating** (estimated 3-5; will be enumerated in solution proposal §3.4):
- Any test asserting that "open browser" alone returns `SeleniumLibrary` → must update to expect `None`
- Any test asserting that "click element" alone returns `SeleniumLibrary` → must update to expect `None`
- Any test in `test_session_models_library_preference.py` exercising session auto-config from these noise phrases

**New tests added** (**~18 distinct test functions** across 9 test classes, ~37 invocations after parametrisation expansion — see proposal §5 Test Matrix). v1's "5-8" and "25-30" estimates and v2's "10" claim are reconciled to **18 in v3** matching the honest matrix count.

## 8. Out of scope

- LLM-based detection (`sample_analyze_scenario` path) — already exists, opt-in via env, unchanged by this PRD.
- `_determine_capabilities` / `suggested_libraries` tightening — those fields are advisory and acceptable as-is.
- Auto-import behaviour in `configure_from_scenario` — fixed indirectly by the more-accurate `explicit_library_preference`. No code change to the importer.
- Renaming the field — the name `explicit_library_preference` is accurate after the fix; no renaming required.

## 9. Stakeholders & impact

| Stakeholder | Impact |
|---|---|
| **End users** writing scenarios | Stop seeing SeleniumLibrary autoload when they meant Browser. Stop seeing Browser keywords filtered out during discovery. Bug fix; positive. |
| **Agents** consuming `analyze_scenario` output | Correct `explicit_library_preference` value. New optional fields available (`library_preference_conflicts`, `explicit_library_evidence`) for richer reasoning. |
| **rf-mcp maintainers** | Smaller, more accurate detection rule set. Easier to extend with new libraries — only need to add explicit-mention patterns. |
| **Existing test suite** | ~18 new tests added (see proposal §5). 3-5 existing tests asserting false-positive behaviour need updating (see AC-8 + proposal §3.4). v2 wrongly said "no existing tests should break" — v3 reconciles with AC-8's explicit list. |

## 10. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Removing the `open browser` pattern misses some genuinely-Selenium scenarios | Low | The pattern was already overzealous. Users intending Selenium typically say "use Selenium" or mention "webdriver" / "chromedriver". The `selenium` standalone-mention pattern at weight 6 still catches "Selenium test" / "run Selenium". |
| Bumping the conflict-group threshold to 8 misses some Selenium-only scenarios that only had weight-6 hits | Medium | Document the new threshold. Operators can lower via env var if their corpus needs it (see ADR-024 §6). |
| The `library_preference_conflicts` field surfaces noisy data | Low | Only emitted when two libraries in the SAME conflict group both score > 0. Web-automation is the only registered group. |
| Sampling path silently overrides the rule-based detector | Already exists | The PRD doesn't change sampling behaviour. Sampling opt-in stays opt-in. Documented in ADR-024 §7. |

## 11. Success metrics

Before/after the fix, run the following on the project's existing scenario test corpus (if any) and on a synthetic set of 20 scenarios (10 truly explicit, 10 pure-NL):

| Metric | Pre-fix | Post-fix target |
|---|---|---|
| Precision on "scenarios with no library mention" returning None | ~30% (estimated) | ≥95% |
| Recall on "use playwright" / "use selenium" returning correct library | ~100% | 100% |
| False positives via single weight-6 NL pattern | Common | Eliminated |
| Test suite duration impact | n/a | < 50ms added |

## 12. Open questions

1. **Should `\bwebdriver\b` stay at weight 6?** It IS a Selenium-explicit term but appears in error messages and unrelated docstrings sometimes. Recommended: keep at 6 but require a context check (the conflict-group threshold of 8 will block lone-`webdriver` from triggering anyway).
2. **Should we expose `min_score` as a tool parameter?** Probably not — adds API surface for an internal knob. Keep as env var only (`ROBOTMCP_LIBRARY_DETECTION_MIN_SCORE`, default 5; `ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD`, default 8).
3. **Should the "ambiguity window" be configurable?** Default 4 covers the common case; let operators override via `ROBOTMCP_LIBRARY_DETECTION_AMBIGUITY_WINDOW`.
4. **What about resource keywords with overlapping names?** Out of scope; resource keywords don't have a libdoc tag taxonomy and shouldn't drive explicit-preference detection.

## 13. References

### Source files

- `src/robotmcp/utils/library_detection.py:21-126` — LIBRARY_PATTERNS table
- `src/robotmcp/utils/library_detection.py:132-135` — NEGATION_PATTERNS (current; v2 replaces with sentence-scoped algorithm in FR-6)
- `src/robotmcp/utils/library_detection.py:155-213` — `detect()` and `get_scores()` methods
- `src/robotmcp/utils/library_detection.py:240-242` — CONFLICT_GROUPS (web_automation: Browser + SeleniumLibrary)
- `src/robotmcp/components/nlp_processor.py:31` — class is `NaturalLanguageProcessor` (v1 wrongly called it `NLPProcessor`)
- `src/robotmcp/components/nlp_processor.py:203-280` — `analyze_scenario` builds response dict
- `src/robotmcp/components/nlp_processor.py:517-544` — `_determine_capabilities` (the separate substring heuristic — see FR-8)
- `src/robotmcp/components/nlp_processor.py:652-704` — `_detect_explicit_library_preference` + fallback
- `src/robotmcp/models/session_models.py:563-590` — `detect_explicit_library_preference` aggregate method
- `src/robotmcp/server.py:1811-1985` — `analyze_scenario` MCP tool wrapper
- `src/robotmcp/server.py:1866-1880` — sampling override path (FR-7)
- `src/robotmcp/server.py:1944-1956` — analyze_scenario auto-configure path
- `src/robotmcp/server.py:2037` — `_filter_keywords_by_session_library` (downstream consumer 1)

### Downstream consumer files (§2 table)

- `src/robotmcp/components/library_recommender.py:111-166,310-321`
- `src/robotmcp/adapters/adapter_factory.py:131-140`
- `src/robotmcp/components/execution/keyword_executor.py:1901-1914`
- `src/robotmcp/plugins/builtin/browser_plugin.py:314-321,379-390`
- `src/robotmcp/plugins/builtin/selenium_plugin.py:202-208`

### Architecture correction (v1→v2)

v1 implied that `suggested_libraries` was backed by `LibraryDetector.get_scores`. **This was wrong.** `suggested_libraries` is computed by `_determine_capabilities` (nlp_processor.py:517-544), which is a **separate substring/keyword heuristic** that does NOT call `LibraryDetector`. The mention layer described in the DDD bounded context (`get_scores`) is currently consumed only by `_detect_explicit_library_preference` itself — there is no current capability path through it. Future work could route capabilities through the mention layer for consistency; that's out of scope here.

### Companions (also revised to v2)

- `docs/adr/ADR-024-explicit-library-detection-confidence.md`
- `docs/ddd/library_preference_bounded_context.md`
- `docs/proposals/explicit_library_detection_fix_proposal.md`
