# ADR-024: Explicit Library Detection — Confidence Model + Conflict Surfacing

**Status:** **ACCEPTED v5** (implemented in `src/robotmcp/utils/library_detection.py`)
**Date:** 2026-06-05
**Author:** rf-mcp maintainer
**Domain:** Scenario Analysis, Library Selection, Session Auto-Configuration
**Supersedes:** None
**Relates to:** ADR-006 (Tool Profile Bounded Context — explicit_library_preference consumed by sessions), ADR-014 (Persistent Semantic Memory — sampling-based library override path), PRD `docs/prd/analyze_scenario_explicit_library_prd.md`

---

## Revision history

**v7 (2026-06-05)** — Codex round-6 review verified v6 source-code fixes but found a paragraph-boundary regression from v6's `\n`-drop. v7 update:
- **§3.2 P3 sentence-split regex updated**: `r"[.;!?\n]+"` (stale) → `r"[.;!?,]+|\n\s*\n+"` (paragraph-aware). Single newline stays non-boundary; paragraph break (2+ newlines) IS a boundary. Documentation now matches the v7 source at `library_detection.py:_SENTENCE_DELIMITERS`.

**v6 (2026-06-05)** — Codex round-5 review marked ADR v5 still REWRITE (P2 + RequestsLibrary table row still allowed bare `requests`/`browser` contradicting source). v6 closes:
- **§3.2 P2 brand-only rule documented**: preference-verb tokens restricted to brand names (`selenium`, `playwright`, `appium`) + verbatim library names. Generic English/domain nouns (`browser`, `database`, `ssh`, `requests`, `xml`) excluded.
- **§6 pattern table aligned with source**: RequestsLibrary / DatabaseLibrary / SSHLibrary / XML preference-verb rows now declare BRAND-ONLY restriction. Matches the v6 `library_detection.py:LIBRARY_RULES_DEFAULT`.
- **Architecture decisions confirmed by implementation**: v6 source addresses 4 round-5 critical/high source bugs (session-fallback, newline-negation, repeated-token, multi-word migration). See proposal v6 for source-level details.

**v5 (2026-06-05)** — Codex round-4 review marked ADR v4 REWRITE due to stale algorithm code blocks. v5 implements the architecture and updates all references:
- **§3.2 P3 code block replaced**: the v4 ADR still showed `\b(?:not|don't|do\s+not|...)\s+(?:using\s+)?(?P<target>\w+)` — Codex flagged this is the same `target='use'` bug from v2 that the proposal had fixed but the ADR hadn't propagated. v5 replaces with the actually-shipped `_NEGATION_REGEX` alternation + `_first_library_token_in()` two-step.
- **§3.3 / §3.4 / §3.5 dict-shaped conflicts/evidence**: v4 ADR still had tuple-shaped `conflicts={group: [(l, score, patterns) ...]}` and `evidence=[(p, w, span) ...]`. v5 aligns to the canonical dict shape used in PRD/DDD/proposal AND in the implementation (`PreferenceResolution.conflicts: Dict[str, List[Dict[str, Any]]]`, evidence is `List[Dict[str, Any]]`).
- **§11 sampling override coherence — Site A semantic fix**: v4 still mapped `sampling_result.get("primary_library")` into `explicit_library_preference`. Codex flagged `primary_library` is the LLM's "main detected" (a recommendation), not user-stated preference. v5 drops that fallback — only `sampling_result["library_preference"]` writes to the explicit field. **Implemented at `server.py:1860-1881`.**
- **`_NEGATION_REGEX` ordering documented**: longest-first alternation guarantees one match per span. `skip` restored to standalone alternation (round-4 D-skip regression fixed).
- **Newline boundary fix**: preference-verb patterns use `[^\S\n]+` instead of `\s+` so multi-line scenarios like `"do not use\nSelenium"` don't match `use selenium` across the line break.

**v4 (2026-05-29)** — Third-round independent review (verdict: TIGHTEN for ADR). v4 addresses:
- **§3.2 P3 negation algorithm description updated**: replaced "phrase-list iteration" description with "single regex alternation, longest-first" to match the v4 proposal §3.1 Step 5 fix. The proposal's algorithm was changed (D1 fix); this ADR section now reflects the new design.
- **§3.4 conflicts shape annotation**: emphasises that `library_preference_conflicts` values are LIST-OF-DICTS `[{library, score, patterns_matched}]`, NOT list-of-tuples. v3 DDD §4.1.6 had a stray tuple type that diverged; v4 DDD now aligned and ADR cross-references the canonical shape.
- **No other ADR-level architecture changes required**: round-3 review found ADR is materially correct; the algorithm fix lives in the proposal.

**v3 (2026-05-29)** — Codex CLI second-round critical review (verdict: TIGHTEN). v3 addresses:
- **§3.3 CONFLICT_GROUPS location corrected**: v2 said "already defined at `library_detection.py:240-242`". Source verification shows it's a LOCAL dict INSIDE `get_conflicting_detections()`, NOT a module-level constant. v3 documents this and notes the implementation promotes it to module-level.
- **§3.5 evidence shape standardised**: now matches the canonical flat-list shape `{library, pattern, weight, text_span}` per entry — same as PRD §FR-5, DDD §4.1.2 `PatternMatch`, proposal Step 4. v2 had three different shapes across the 4 docs.
- **§3.2 P3 algorithm rewritten**: v2's regex `\b(not|...)\s+(?:using\s+)?(?P<target>\w+)` captured `target="use"`. v3 uses **phrase-list + `_first_library_token_in()`** approach (see proposal §3.1 Step 5). Sentence delimiter now `[.;!?,\n]+` (includes comma) so "do not use Selenium, instead use Playwright" splits correctly.
- **§6 pattern-table tightening**: bare-noun preference verbs (`use browser`, `use database`, `use ssh`, `use requests`, `use xml`) dropped from `explicit=True` per Codex round-2 finding. Brand names (`use selenium`, `use playwright`, `use appium`) retained.
- **§11 sampling override coherence — TWO sites documented**: v2 mentioned only `server.py:1860-1881` (analysis-response override). Source verification shows a second override at `server.py:1961-1972` modifying `session.explicit_library_preference`. v3 covers both. Correct env var `ROBOTMCP_USE_SAMPLING` (v2 mistakenly wrote `ROBOTMCP_USE_SAMPLING_FOR_NLP`).
- **§12 `_compiled_patterns` test contract corrected**: v2 claimed compatibility via `__iter__` yielding 4 values. Real test contract at `test_nlp_improvements.py:571-576` is `for p, _ in entries: p.findall(...)` — `p` must be a compiled `re.Pattern`. v3 keeps `_compiled_patterns` as `Dict[str, List[Tuple[Pattern, int]]]` and stores rich annotations in a parallel `_rules_metadata` attribute.

**v2 (2026-05-29)** — Codex CLI critical review identified architecture and algorithm errors. Key changes:
- **Architecture correction (§3.1)**: removed false claim that `LibraryDetector.get_scores` backs `suggested_libraries`. The capability layer (`_determine_capabilities` at `nlp_processor.py:517-544`) is a separate substring heuristic that does NOT call `LibraryDetector`. The mention layer is currently unused by the capability path; this PRD does not change that.
- **Algorithm ordering correction (§3.3/§3.4)**: conflict detection now runs on **raw explicit scores BEFORE threshold filtering**. v1's "filter then detect conflicts" ordering made the AC-5 example impossible (`"both selenium and playwright"` would be filtered to a single candidate before the conflict check ran). Pseudocode now explicit.
- **Negation redesigned (§3.2)**: replaced v1's 80-character forward window with a **sentence-scoped source-vs-destination resolver** (migration patterns + per-sentence negation). The 80-char window broke `"do not use Selenium, instead use Playwright"` by zeroing both libraries.
- **Pattern annotations corrected (§6)**: dropped wrongly-annotated `explicit=True` for `rest api testing`, `mobile automation`, `android/ios testing`, SQL fragments, DB engine names. These are domain/topic markers, not library identifiers.
- **Downstream consumers expanded (§1.3)**: from 3 to 8 (the additional 5 surfaced by Codex's grep across library_recommender, adapter_factory, keyword_executor, browser_plugin, selenium_plugin).
- **Alternatives section expanded (§4)**: added the 3 alternatives Codex flagged as missing — strict allow-list of library tokens, decision-tree/grammar parser, span-based source-vs-destination NLP.
- **Performance claim removed (§9)**: v1's "≤60µs" / "<100µs aggregate" was unsupported. Replaced with "no regression vs current benchmark".
- **Sampling override coherence (§11, NEW)**: defines what happens to evidence/conflicts when sampling overrides preference. v1 was silent on this.
- **Test-surface compatibility (§12, NEW)**: `_compiled_patterns` introspection is preserved.

**v1 (2026-05-29)** — initial draft.

---

## 1. Context and Problem Statement

### 1.1 Reported defect

On 2026-05-29 a user reported the `analyze_scenario` MCP tool returning `explicit_library_preference: "SeleniumLibrary"` for a scenario that never explicitly mentioned SeleniumLibrary:

```
Scenario: "Test e-commerce website https://demoshop.makrocode.de:
           open browser, add items to shopping cart, verify items,
           complete checkout, and close browser"
```

The user neither typed `selenium`, `seleniumlibrary`, `webdriver`, nor any other Selenium-specific term. They used "open browser" as a generic English verb. The system nevertheless declared an *explicit* preference for SeleniumLibrary.

### 1.2 Root cause trace

`src/robotmcp/utils/library_detection.py:33` contains the offending pattern:

```python
'SeleniumLibrary': [
    ...
    (r'\bopen\s+browser\b', 6),         # Weight 6
    (r'\b(input\s+text|click\s+element|page\s+should\s+contain)\b', 6),
    (r'\b(implicit|explicit)\s+wait\b', 6),
    ...
],
```

These patterns target SL keyword **names** (`Open Browser`, `Input Text`, `Click Element`, `Page Should Contain`). The weights are tuned at 6 — single-pattern-above-threshold, since `DEFAULT_MIN_SCORE = 5`.

The user's text matches `\bopen\s+browser\b` exactly once → SeleniumLibrary score = 6 ≥ 5 → returned. Browser library scores 0 because the scenario uses neither `playwright`, `browser library`, nor `new browser/page/context`.

The same overzealous detection fires on these other natural-language inputs (empirically verified on 2026-05-29):

| Scenario | Detected (wrongly) | Should detect |
|---|---|---|
| "click element by id submit" | SeleniumLibrary | None |
| "Page should contain Welcome text" | SeleniumLibrary | None |
| "Input text into the username field" | SeleniumLibrary | None |
| "Open new page in browser context" | Browser | None (also keyword name) |
| "Test e-commerce site: open browser..." (the report) | SeleniumLibrary | None |

### 1.3 Why a single overzealous pattern matters

`explicit_library_preference` is consumed in **eight** load-bearing places downstream (v1 listed three; Codex review surfaced five more):

| # | File:lines | Effect |
|---|---|---|
| 1 | `session_models.py:787` (`configure_from_scenario`) | Auto-imports the detected library at session init |
| 2 | `server.py:2037` (`_filter_keywords_by_session_library`) | Suppresses keywords from incompatible libraries in `find_keywords` discovery |
| 3 | `server.py` (`find_keywords` filter precedence) | Treats explicit preference as second priority (after `library_name`) |
| 4 | `components/library_recommender.py:111-166` | Adds the preference to the front of the recommendation list with high confidence |
| 5 | `components/library_recommender.py:310-321` | Excludes the other library from candidates |
| 6 | `adapters/adapter_factory.py:131-140` | Selects the platform adapter (Browser vs SL) |
| 7 | `components/execution/keyword_executor.py:1901-1914` | Disambiguates same-named keywords (e.g., `Take Screenshot`) |
| 8 | `plugins/builtin/browser_plugin.py:314-321,379-390` + `plugins/builtin/selenium_plugin.py:202-208` | Suppress / pre-init the respective session |

False detection cascades through all eight call-sites. A wrong `explicit_library_preference` is not a cosmetic bug; it determines which library loads, which keywords surface, which adapter runs, and which session pre-boots.

The defect violates the principle of least surprise: "explicit" should mean "the user said so". The current implementation conflates "user used English phrases that overlap with SL keyword names" with "user explicitly chose SL".

### 1.4 Constraints we cannot change

- **Backward compatibility of the response shape**: `analysis.explicit_library_preference` is the documented field; existing agents read it. We can ADD fields but cannot remove or rename.
- **Existing capability detection** (`_determine_capabilities` → `suggested_libraries`) is intentionally broad and is acceptable to keep — it's an advisory list, not a single declarative choice.
- **Sampling-based override** (`sample_analyze_scenario` at `server.py:1863`) already exists as an opt-in LLM path; the rule-based detector must continue to provide a deterministic default.

---

## 2. Decision Drivers

| Driver | Weight | Rationale |
|---|---|---|
| **Field-name accuracy** | Critical | `explicit_library_preference` must mean what it says — explicit user choice. The patterns table currently violates this contract. |
| **Backward compatibility** | High | Existing callers (test suites, downstream tooling, ADR-006-aware sessions) read the field. Cannot rename. Can add advisory fields. |
| **Determinism** | High | Same input → same output. Rules out runtime ML scoring. Rules must be transparent and auditable. |
| **Pattern auditability** | High | Every pattern in the table must have a clear "this is library X mention" justification. Patterns that match generic English are out of bounds. |
| **Conflict transparency** | Medium | When the scenario could mean either Browser or SeleniumLibrary, the user benefits from being told so rather than silently arbitrating. |
| **Performance** | Low | The detector runs once per `analyze_scenario` call. P95 cost is ~50µs. Not on a hot path. |

---

## 3. Architecture Decision

### 3.1 Distinguish "mention" from "preference"

Introduce a conceptual split with **honest scope limits** (v1 overstated the mention layer's role; v2 corrects this):

- **Mention** — the scenario text contains a token that COULD identify the library. Exposed via `LibraryDetector.get_scores(text)`. Allowed to be over-inclusive. **Current consumers: none in production code.** v1 implied this layer backed `suggested_libraries`; that was wrong. `suggested_libraries` comes from `_determine_capabilities` (nlp_processor.py:517-544), a SEPARATE substring heuristic that does not call `LibraryDetector`. The mention layer is preserved for diagnostics and potential future use, not for active consumption by this ADR's flow.
- **Preference** — the user has explicitly chosen the library. Exposed via the new `LibraryDetector.detect_explicit_preference(text)`. Used for the load-bearing `explicit_library_preference` field that all eight consumers in §1.3 read. Must be conservative.

A pattern can belong to one, both, or neither layer. Each `LIBRARY_PATTERNS` entry gains an `explicit: bool` flag (see §6 for the full table). `get_scores` ignores the flag (computes mention scores); `detect_explicit_preference` consults only `explicit=True` entries.

### 3.2 Preference rules

Three rule categories, evaluated in order:

**P1 — Verbatim library identifier (decisive)**:
- The scenario contains a library identifier token verbatim:
  `seleniumlibrary`, `playwright`, `browserlibrary` (Browser), `requestslibrary`, `appium(?:library)?`, `databaselibrary`, `sshlibrary`, `xml library`/`xmllibrary`, `webdriver`, `chromedriver`/`geckodriver`/`edgedriver`/`safaridriver`, `chromium`/`webkit` (Playwright-side kernels), `rfbrowser`, `robotframework-browser`, `selenium2library`.
- Weight: 9 or 10 depending on specificity.

**P2 — Preference verb + library token (decisive)**:
- `\b(use|using|with|via|through|prefer)[^\S\n]+(<library token>)\b` where `<library token>` is any verbatim P1 term OR a library-specific brand short name. **v6 BRAND-ONLY RULE**: only brand-name short forms (`selenium`, `playwright`, `appium`) qualify. Generic English/domain nouns (`browser`, `database`, `ssh`, `requests`, `xml`) are NOT valid preference-verb targets — they're too ambiguous (could mean web browser app, any DB tool, ssh protocol, Python `requests` package, XML the file format). To prefer those libraries, users must spell the verbatim library name (`requestslibrary`, `database library`, `ssh library`, `xml library`).
- Weight: 10.
- v6 newline boundary: `[^\S\n]+` instead of `\s+` so multi-line scenarios like `"use\nplaywright"` don't trigger the preference verb across the line break.

**P3 — Negation and migration (sentence-scoped)**:

v1 documented this as "the existing `NEGATION_PATTERNS` plus an 80-char forward window". Codex showed the 80-char window breaks `"do not use Selenium, instead use Playwright"` (it covers both library mentions, zeroing both). v2 replaced this with a sentence-scoped algorithm but used a PHRASE-LIST iteration, which caused round-3 D1: both `\bdo\s+not\s+use\b` and `\bdo\s+not\b` fired on the same span. **v4 uses a SINGLE regex with alternation, longest-first**, so each negation span fires exactly once (regex engine's left-to-right alternation guarantees longest match wins). Sentence-scoped algorithm with source/destination resolution:

```
# v7: split on sentence punctuation OR paragraph break (\n\s*\n+).
# Single newlines are NOT boundaries (so "do not use\nPlaywright" stays one
# sentence and negation finds the target). Paragraph breaks ARE boundaries
# (so "Do not use this approach\n\nUse Playwright" splits correctly).
sentences = split(scenario, regex=r"[.;!?,]+|\n\s*\n+")

MIGRATION_PATTERNS = [
  r"\bmigrat(?:e|ion|ing)\b.*?\bfrom\s+(?P<src>\w+).*?\bto\s+(?P<dst>\w+)",
  r"\bswitch(?:ing)?\s+from\s+(?P<src>\w+).*?\bto\s+(?P<dst>\w+)",
  r"\binstead\s+of\s+(?P<src>\w+).*?\b(?:use|with|via)\s+(?P<dst>\w+)",
  r"\breplace\s+(?P<src>\w+)\s+(?:with|by|for)\s+(?P<dst>\w+)",
]

# v5 — SINGLE regex alternation, longest-first. Each negation span fires
# exactly ONCE (regex engine guarantees longest match at each position).
# `skip` restored to standalone alternation (round-4 D-skip regression).
# v4 used a phrase LIST iterated in Python — that caused double-deduction
# when both `\bdo\s+not\s+use\b` and `\bdo\s+not\b` matched the same span.
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

for sentence_index, sent in enumerate(_build_sentence_spans(text)):
    # Migration (source / destination resolution)
    for migration in MIGRATION_PATTERNS:
        for m in re.finditer(migration, sent, re.IGNORECASE):
            src_lib = _first_library_token_in(m.group("src"))
            dst_lib = _first_library_token_in(m.group("dst"))
            if src_lib:
                _subtract_sentence_score(raw_scores, matches_by_lib,
                                         sent, sentence_index, src_lib)
            if dst_lib:
                raw_scores[dst_lib] += 5  # destination bonus
    # Negation — single regex, then resolve target via remaining-text lookup
    for m in _NEGATION_REGEX.finditer(sent):
        remaining = sent[m.end():]
        target_lib = _first_library_token_in(remaining)
        if target_lib:
            _subtract_sentence_score(raw_scores, matches_by_lib,
                                     sent, sentence_index, target_lib)
```

Per-sentence scoping ensures the second clause ("instead use Playwright") gets to apply its positive contribution independently of the first clause's negation. The migration patterns explicitly distinguish source from destination, avoiding the v1 forward-window pitfall. `_first_library_token_in()` does the canonical library-name resolution (handles multi-word names like "Browser Library", brand names like "Selenium", and deliberately rejects bare generic nouns like `browser`/`database`/`ssh`).

**P4 — Ambiguity check**:
- See §3.3 — runs on RAW scores BEFORE threshold filtering. When top-2 candidates inside a conflict group are within `ambiguity_window` of each other → return None + populate conflicts.

**REMOVED from preference layer (kept in mention layer with `explicit=False`)**:
- Generic action verbs that overlap with keyword names: `\bopen\s+browser\b`, `\b(input\s+text|click\s+element|page\s+should\s+contain)\b`, `\bnew\s+(browser|page|context)\b`, `\b(fill\s+text|fill\s+secret)\b`
- Generic technique terms: `\b(implicit|explicit)\s+wait\b`, `\b(SPA|single\s+page\s+app(lication)?)\b`, `\b(e2e|end.to.end)\s+(test|automat)\b`, `\bcross[- ]browser\s+testing\b`, `\bheadless\b`, `\bshadow\s+dom\b`, `\bweb\s+components?\b`, `\bmodern\s+web\s+testing\b`, `\bmodern\s+browser\s+automation\b`
- Domain markers (v1 wrongly kept these as `explicit=True`; Codex flagged): `\brest\s+api\s+testing\b`, `\bmobile\s+(automation|testing)\b`, `\b(android|ios)\s+(testing|test)\b`, `\b(SELECT|INSERT|UPDATE|DELETE)\s+(?:\*|FROM|INTO|SET)\b`, `\b(postgres(?:ql)?|mysql|sqlite|oracle|mariadb|mssql|sqlserver)\b`, `\bstatus\s+code\b`, `\bxpath\b`, `\bxsd\b`
- SSH-domain noise: `\bexecute\s+command\b`, `\bremote\s+(server|host|machine)\b`, `\bssh\s+(into|to)\b`
- Mobile-domain noise: `\bopen\s+application\b`, `\b(tap|swipe|long\s+press)\b`, `\bdevice\b`
- DB-domain noise: `\b(check|verify)\s+if\s+exists\b`, `\brow\s+count\b`

### 3.3 Conflict groups + thresholds + algorithm ordering

**Conflict groups** — v3 correction: source verification at `library_detection.py:240-242` shows `CONFLICT_GROUPS` is currently a LOCAL variable inside `get_conflicting_detections()`, NOT a module-level constant. The implementation must **promote it to module-level** so the new `detect_explicit_preference` (and tests) can reference the same definition:

```python
# v3 — module-level constant in library_detection.py (PROMOTED from the local
# dict currently scoped inside get_conflicting_detections):
CONFLICT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "web_automation": ("Browser", "SeleniumLibrary"),
    # Future: "mobile_native" (UIAutomator2 vs XCUITest AppiumLibrary configs)
    # Future: "api_client" (RequestsLibrary vs hypothetical alt)
}
```

After promotion, the existing `get_conflicting_detections()` method also reads the module-level constant (no behaviour change). This is part of the §5 implementation checklist.

**Thresholds**:
- `default_min_score = 5` — libraries OUTSIDE a conflict group (XML, Database, SSH, Requests, Appium).
- `conflict_min_score = 8` — libraries INSIDE a conflict group (Browser, SeleniumLibrary).
- `ambiguity_window = 4` — if two in-group libraries' raw scores differ by ≤ this value, declare a conflict.
- Env-tunable: `ROBOTMCP_LIBRARY_DETECTION_MIN_SCORE`, `ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD`, `ROBOTMCP_LIBRARY_DETECTION_AMBIGUITY_WINDOW`.

**Algorithm ordering — critical correction from v1**:

```
# Step 1: compute raw scores using only explicit=True patterns
raw_scores = {lib: sum_weights_of_matched_explicit_patterns(lib, text) for lib in libraries}

# Step 2: apply negation + migration (per §3.2 P3) — mutates raw_scores
apply_sentence_scoped_negation_and_migration(raw_scores, text)

# Step 3: CONFLICT CHECK ON RAW SCORES (v2 fix — was post-threshold in v1)
for group, members in CONFLICT_GROUPS.items():
    libs_with_signal = [lib for lib in members if raw_scores.get(lib, 0) > 0]
    if len(libs_with_signal) >= 2:
        ranked = sorted(libs_with_signal, key=lambda l: raw_scores[l], reverse=True)
        top1, top2 = ranked[0], ranked[1]
        if raw_scores[top1] - raw_scores[top2] <= ambiguity_window:
            # v5: dict-shaped conflict entries (matches PRD §FR-4, DDD §4.1.6,
            # proposal §3.1 Step 6, and the implemented PreferenceResolution).
            return PreferenceResolution(
                library=None,
                source="rule",
                conflicts={group: [
                    {"library": l,
                     "score": raw_scores[l],
                     "patterns_matched": [pm.pattern for pm in matches_by_lib[l]]}
                    for l in ranked
                ]},
                evidence=[],
            )

# Step 4: threshold filter (post-conflict)
candidates = {}
for lib, score in raw_scores.items():
    threshold = conflict_min_score if lib_in_any_conflict_group(lib) else default_min_score
    if score >= threshold:
        candidates[lib] = score

if not candidates:
    return PreferenceResolution(library=None, source="rule", conflicts={}, evidence=[])

winner = max(candidates, key=candidates.get)
return PreferenceResolution(
    library=winner,
    source="rule",
    conflicts={},
    evidence=[(p, w, span) for p, w, span in matches_for(winner)],
)
```

Why this ordering matters: v1's "threshold then conflict" sequence made the AC-5 example impossible. With `conflict_min_score=8`, `"both selenium and playwright"` yields SL=6 (filtered out before conflict check) + Browser=9 (sole survivor) → no conflict, Browser declared winner — which is NOT the documented expected outcome. v2's "conflict on raw THEN threshold" sequence catches the SL=6/Browser=9 pair at the raw stage (diff 3 ≤ 4) and returns None+conflicts, matching the AC.

### 3.4 Conflict surfacing

When the algorithm in §3.3 step 3 returns conflicts, the wrapping `analyze_scenario` response includes:

```json
"library_preference_conflicts": {
  "web_automation": [
    {"library": "Browser", "score": 9, "patterns_matched": ["\\bplaywright\\b"]},
    {"library": "SeleniumLibrary", "score": 6, "patterns_matched": ["\\bselenium\\b"]}
  ]
}
```

This is purely additive: when no conflict exists, the field is absent. Field name standardised to `patterns_matched` (was `matched_patterns` in v1 inconsistently).

### 3.5 Evidence provenance (v3 canonical shape)

When `detect_explicit_preference` returns a library name, the response gains an `explicit_library_evidence` field — a **flat list** of pattern-match entries, each with the canonical shape used across all 4 docs:

```json
"explicit_library_evidence": [
  {
    "library": "SeleniumLibrary",
    "pattern": "<regex source>",
    "weight": 10,
    "text_span": "<matched substring>"
  }
]
```

Each entry: `{library: str, pattern: str, weight: int, text_span: str}`. The list is FLAT (one entry per matching pattern, NOT grouped by library); the `library` field per entry lets downstream consumers filter or group as needed. Matches DDD §4.1.2 `PatternMatch` exactly and proposal §3.1 Step 4. Absent when no preference is set.

---

## 4. Alternatives Considered

### 4.1 Alternative A — Raise `DEFAULT_MIN_SCORE` to 10 across the board

Reject. Would break `(use|using|with) selenium` (weight 10) — that's the prototypical explicit case and must trigger. Would also miss "use playwright" (weight 10) etc. Too blunt.

### 4.2 Alternative B — Remove SL keyword-name patterns ENTIRELY (including from the mentions table)

Reject. The capability suggestion (`suggested_libraries`) benefits from knowing "this scenario discusses element clicks" → "needs UI library". Removing means losing that signal. Better: filter patterns OUT only for the preference layer; keep them in the mention layer.

### 4.3 Alternative C — Replace pattern matching with an LLM call

Reject as required behaviour. The sampling path already exists for operators who want LLM-backed detection. The rule-based path must remain deterministic so the default behaviour is reproducible and offline-capable.

### 4.4 Alternative D — Demand a structured `library_hint` parameter in `analyze_scenario`

Reject. Adds API surface. Most users won't pass it. The fix should make the existing detection conservative; explicit overrides via `library_name` on subsequent tools already exist.

### 4.5 Alternative E — Strict allow-list of library tokens (no scoring)

Reject. Hard-coded list of library identifiers; if the scenario contains an exact token, declare preference; otherwise None. Conceptually clean and trivial to implement. Downsides:
- Loses the "use Selenium 4" / "with Playwright" preference-verb idiom unless we either bake the verbs into the allow-list (degenerates back into pattern matching) or accept that `"use Selenium"` returns None unless `selenium` is also tokenised.
- Can't represent partial signals or conflict (it's binary).
- Doesn't compose with the mention layer — the mention layer needs weighted scoring for capability hints.

The chosen Alternative G keeps the allow-list spirit (verbatim P1 patterns) but layers it on the existing weighted scoring so we get preference-verb support and conflict signalling for free.

### 4.6 Alternative F — Decision tree / grammar parser for "use X" / "from X to Y" / "prefer X"

Reject. Build a tiny grammar over scenario sentences:
```
PrefStatement := "use" Lib | "with" Lib | "via" Lib | "prefer" Lib
MigrationStmt := "migrate from" Lib "to" Lib | "switch from" Lib "to" Lib
NegationStmt  := "do not use" Lib | "without" Lib | ...
```
Run it via an actual parser (Lark / PEG / etc.). Pros: stronger guarantees on negation correctness; handles "from X to Y" cleanly; resists adversarial inputs. Cons:
- Heavy dependency (parser library) for a relatively small problem.
- Schema changes every time we add a library or idiom; harder to env-tune than thresholds.
- The chosen Alternative G's sentence-scoped algorithm captures the same source-vs-destination semantics with simple regex per sentence at materially less code/maintenance cost.

We borrow the source-vs-destination concept from this alternative; we adopt it via MIGRATION_PATTERNS (§3.2 P3) rather than a full grammar.

### 4.7 Alternative F' — Span-based source-vs-destination NLP

Reject. Detect the spans (positions) of library tokens, then classify each span as source / destination / standalone using syntactic features (preposition, verb, position). Pros: language-model-grade quality without LLM cost. Cons:
- Requires syntactic features we don't have (need spaCy or similar). Adds a heavy dependency.
- Overkill for the cases this PRD must handle (migration, instead-of, replacement). The simpler `MIGRATION_PATTERNS` in Alternative G covers 95% of the expected idioms with zero dependencies.

We acknowledge this as a future option if migration/negation handling grows in complexity. For now, the sentence-scoped regex approach is the right complexity match.

### 4.8 Alternative G (chosen) — Two-tier "mention vs preference" model with conflict groups + thresholds + sentence-scoped negation

Adopted (was Alternative E in v1; renumbered to G in v2 with the new alternatives inserted above it). Smallest API surface change. Preserves the mention layer for diagnostics. Makes the preference layer match its name. Conflict surfacing is purely additive — agents that ignore the new field see no change. Sentence-scoped negation/migration handles the source-vs-destination cases without requiring a grammar parser or NLP toolchain.

---

## 5. Decision

**Adopt Alternative G** as detailed in §3:

1. Annotate each entry in `LIBRARY_PATTERNS` with `explicit: bool` (table at §6).
2. Add `LibraryDetector.detect_explicit_preference(text)` that:
   - Computes raw scores using only `explicit=True` patterns.
   - Applies sentence-scoped negation + migration (P3 in §3.2).
   - Runs the conflict check on RAW scores BEFORE threshold filter (§3.3 ordering).
   - Applies `default_min_score` or `conflict_min_score` per library.
   - Returns a `PreferenceResolution(library, source, conflicts, evidence)`.
3. `get_scores(text)` keeps its current behaviour (mention layer, ignores `explicit` flag).
4. Update `nlp_processor._detect_explicit_library_preference` to call `detect_explicit_preference`.
5. Update `session_models.detect_explicit_library_preference` to call `detect_explicit_preference` for single source of truth.
6. Surface `library_preference_conflicts` + `explicit_library_evidence` + `preference_source` in `analyze_scenario` response.
7. Env-tunable: `ROBOTMCP_LIBRARY_DETECTION_MIN_SCORE` (5), `ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD` (8), `ROBOTMCP_LIBRARY_DETECTION_AMBIGUITY_WINDOW` (4).
8. Sampling override (server.py:1866-1880) clears evidence + conflicts and sets `preference_source: "sampling"` per §11.

---

## 6. Pattern Annotation Schema

Replace the current `List[Tuple[str, int]]` pattern format with a richer dataclass:

```python
@dataclass(frozen=True)
class LibraryPattern:
    pattern: str       # regex source
    weight: int        # 1-10
    explicit: bool     # if True, contributes to preference scoring
    rationale: str     # one-line justification (audit trail)
```

Migration path: keep tuples backward compatible for now; new patterns use the dataclass; gradual migration in a follow-up.

The classification of every existing pattern as `explicit: True` or `explicit: False`. v2 corrects v1's miscategorisation of domain markers (`rest api testing`, `mobile automation`, `android/ios testing`, SQL fragments, DB engine names) — those are TOPICS the user is working in, not library identifiers, so they belong to the mention layer only.

**Decision rule for `explicit: True`**: the pattern must be either (a) a verbatim library/runtime/driver identifier, or (b) a preference-verb idiom binding the verb to a library token. Patterns that describe the user's TASK DOMAIN (REST testing, mobile testing, SQL queries, SSH commands) do not qualify — they are ambient context.

| Library | Pattern | Weight | `explicit` | Rationale |
|---|---|---:|---|---|
| SeleniumLibrary | `(use\|using\|with\|via\|through\|prefer)\s+(selenium\|...)` | 10 | **True** | Preference verb + library token |
| SeleniumLibrary | `seleniumlibrary` | 9 | **True** | Verbatim library token |
| SeleniumLibrary | `\bselenium\b` (standalone) | 6 | **True** | Library name; conflict threshold 8 prevents lone trigger |
| SeleniumLibrary | `\bwebdriver\b` | 6 | **True** | Selenium-specific runtime |
| SeleniumLibrary | `selenium grid`/`standalone`/`selenium 4`/etc. | 7-8 | **True** | Selenium-specific configurations |
| SeleniumLibrary | `chromedriver`/`geckodriver`/`edgedriver`/`safaridriver` | 7 | **True** | Selenium drivers |
| SeleniumLibrary | `(desired\|driver) capabilities` | 7 | **True** | Selenium Capabilities API |
| SeleniumLibrary | `create webdriver`/`get webelement` | 8 | **True** | Selenium-specific kw |
| SeleniumLibrary | `\bopen\s+browser\b` | 6 | **False** | Generic NL; overlaps with SL keyword name |
| SeleniumLibrary | `(input text\|click element\|page should contain)` | 6 | **False** | Keyword names; generic NL |
| SeleniumLibrary | `(implicit\|explicit) wait` | 6 | **False** | Generic concept |
| Browser | `(use\|using\|with\|via\|prefer)\s+(playwright\|browser\s+library)` | 10 | **True** | Preference verb |
| Browser | `\bplaywright\b` | 9 | **True** | Verbatim |
| Browser | `\bbrowser\s+library\b` | 9 | **True** | Verbatim |
| Browser | `\bchromium\b`/`\bwebkit\b` | 9 | **True** | Playwright kernels (v2: raised from 7 to 9; as specific as `playwright`) |
| Browser | `rfbrowser`/`robotframework-browser` | 9 | **True** | Verbatim |
| Browser | `playwright-?core` | 9 | **True** | Playwright runtime |
| Browser | `new (browser\|page\|context)` | 8 | **False** | Keyword names + generic NL |
| Browser | `fill (text\|secret)` | 7 | **False** | Keyword name |
| Browser | `modern web testing`/`modern browser automation` | 7 | **False** | Marketing copy |
| Browser | `cross-browser testing` | 6 | **False** | Generic test type |
| Browser | `SPA\|single page app(lication)?` | 5 | **False** | Describes app, not library |
| Browser | `e2e\|end.to.end` | 5 | **False** | Generic test type |
| Browser | `headless` | 6 | **False** | Generic (SL also has headless mode) |
| Browser | `shadow dom\|web components?` | 6 | **False** | Browser feature, not library identifier |
| RequestsLibrary | `\b(use\|using\|with\|via\|through\|prefer)[^\S\n]+(requestslibrary\|requests\s*library)\b` | 10 | **True** | v6: BRAND-ONLY — bare `requests` excluded (too generic; could mean Python `requests` package, REST API requests, etc.). Library must be spelled as `requestslibrary` or `requests library`. |
| RequestsLibrary | `requestslibrary` | 9 | **True** | Verbatim |
| RequestsLibrary | `(create\|get on\|post on) session` | 8 | **True** | RL keyword names with RL-specific session noun |
| RequestsLibrary | `rest api testing` | 7 | **False** | **v2 correction (was True in v1)**: REST API testing is a DOMAIN, not a library identifier |
| RequestsLibrary | `\bhttp\s+requests?\b` | 5 | **False** | Generic NL |
| RequestsLibrary | `\bstatus\s+code\b` | 5 | **False** | Domain marker, not library identifier |
| RequestsLibrary | `(GET\|POST\|PUT\|DELETE\|PATCH)\s+(request\|on session)` | 7 | **True (only when `on session` form matches)** | The `on session` qualifier is RL-keyword-specific; bare HTTP verbs are domain-generic so drop those from explicit |
| AppiumLibrary | `(use\|using\|with\|via\|prefer)\s+appium(?:library)?` | 10 | **True** | Preference verb |
| AppiumLibrary | `appium(?:library)?` | 9 | **True** | Verbatim |
| AppiumLibrary | `(UIAutomator2?\|XCUITest\|Espresso)` | 7 | **True** | Mobile-specific runtime |
| AppiumLibrary | `mobile (automation\|testing)` | 7 | **False** | **v2 correction (was True in v1)**: domain, not library identifier — could be Appium, Detox, Maestro, etc. |
| AppiumLibrary | `(android\|ios) (testing\|test)` | 6 | **False** | **v2 correction**: platform, not library identifier |
| AppiumLibrary | `(open\|close) application` | 8 | **False** | Keyword names |
| AppiumLibrary | `(tap\|swipe\|long press\|scroll\|pinch)` | 6 | **False** | Generic action verbs |
| AppiumLibrary | `\bdevice\b` (alone) | 5 | **False** | Too generic |
| DatabaseLibrary | `\b(use\|using\|with\|via\|through\|prefer)[^\S\n]+(databaselibrary\|database\s*library)\b` | 10 | **True** | v6: BRAND-ONLY — bare `database` excluded (generic noun; could mean any DB tool). |
| DatabaseLibrary | `databaselibrary` | 9 | **True** | Verbatim |
| DatabaseLibrary | `(SELECT\|INSERT\|UPDATE\|DELETE)\s+(FROM\|INTO\|SET\|\*)` | 5 | **False** | **v2 correction**: SQL is the user's domain, not a library identifier; could be DatabaseLibrary, raw psycopg2 via Python wrapper, etc. |
| DatabaseLibrary | `(postgres(ql)?\|mysql\|sqlite\|oracle\|mariadb\|mssql\|sqlserver)` | 5 | **False** | **v2 correction**: DB engine name, not RF library identifier |
| DatabaseLibrary | `(check\|verify)\s+if\s+exists`/`row count` | 5-6 | **False** | Keyword names; generic NL |
| SSHLibrary | `\b(use\|using\|with\|via\|through\|prefer)[^\S\n]+(sshlibrary\|ssh\s*library)\b` | 10 | **True** | v6: BRAND-ONLY — bare `ssh` excluded (protocol noun; could mean ssh CLI / paramiko). |
| SSHLibrary | `sshlibrary` | 9 | **True** | Verbatim |
| SSHLibrary | `execute command` | 6 | **False** | Keyword name; generic NL |
| SSHLibrary | `remote (server\|host\|machine)` | 5 | **False** | Domain, not library identifier |
| SSHLibrary | `ssh into`/`ssh to` | 6 | **False** | Action verb, not library identifier (could be sshpass, paramiko-script, etc.) |
| XML | `\b(use\|using\|with\|via\|through\|prefer)[^\S\n]+(xmllibrary\|xml\s*library)\b` | 10 | **True** | v6: bare `xml` already excluded (XML is a file format). Library must be spelled `xmllibrary` or `xml library`. |
| XML | `xmllibrary` | 9 | **True** | Verbatim |
| XML | `\bxpath\b`/`\bxsd\b`/`\bxslt\b` | 5 | **False** | **v2 correction**: XML technology, not RF library identifier (XPath is also used by SeleniumLibrary locators) |
| XML | `\bxml\b` (alone) | 4 | **False** | Too generic; could mean XML the file format |
| XML | `parse xml` | 5 | **False** | Generic NL |

**Patterns removed from `explicit=True` since v1**: `rest api testing`, `mobile automation`, `android testing`, `ios testing`, SQL fragments, DB engine names, XPath/XSD/XSLT (XML), bare `xml` keyword. All retained in `LIBRARY_PATTERNS` with `explicit=False` so `get_scores` still surfaces them in the mention layer for diagnostic use.

---

## 7. Consequences

### 7.1 Positive

- **`explicit_library_preference` becomes trustworthy.** Agents/users can act on it without worrying about false positives.
- **Reproducer scenario fixed.** "Open browser" no longer cascades into SL session config.
- **Conflict awareness.** Users with genuinely ambiguous scenarios get told and can clarify.
- **Audit trail.** `explicit_library_evidence` lets users see why detection fired.
- **Defensible threshold knob.** Conflict-group threshold (8) is principled — must clear two weight-6 patterns OR one weight-9 pattern.

### 7.2 Negative

- **Some scenarios that previously fired detection won't anymore.** The user's session won't be auto-configured for Selenium when they typed only "open browser" — they'll need to pass `library_name="SeleniumLibrary"` explicitly OR use a phrase like "use Selenium". This is the correct outcome (the previous behaviour was the bug) but may surprise users who relied on the false-positive auto-config.
- **Mitigation**: the `suggested_libraries` field still recommends both Browser and SeleniumLibrary for generic web work — agents can pick one explicitly.

### 7.3 Neutral

- **New env vars**: 3 added (`ROBOTMCP_LIBRARY_DETECTION_*`). Documented; default values mean operators don't need to touch them.
- **Response shape**: 2 new optional fields. Additive. No breakage for callers reading only existing fields.

---

## 8. Implementation Anchors (for the solution proposal)

- `src/robotmcp/utils/library_detection.py:21-126` — `LIBRARY_PATTERNS` table; add `_EXPLICIT_PATTERN_FILTER` or per-pattern `explicit` flag.
- `src/robotmcp/utils/library_detection.py:155-180` — `detect()`; replace single-threshold logic with `detect_explicit_preference()` that uses the explicit subset + conflict thresholds.
- `src/robotmcp/utils/library_detection.py:231-252` — `get_conflicting_detections()`; extend to return scores + matched patterns.
- `src/robotmcp/components/nlp_processor.py:652-704` — `_detect_explicit_library_preference`; switch to the new method; populate `explicit_library_evidence` in response.
- `src/robotmcp/components/nlp_processor.py:253-280` — `analyze_scenario` return shape; add `library_preference_conflicts` and `explicit_library_evidence` to `analysis` block.
- `src/robotmcp/models/session_models.py:563-590` — `detect_explicit_library_preference`; switch to the new method to keep single source of truth.
- `src/robotmcp/server.py:1944-1985` — `analyze_scenario` MCP wrapper; ensure the new fields surface in the response unchanged.

---

## 9. Acceptance & Validation Plan

(Detailed acceptance criteria live in the PRD §7. ADR-level validation:)

| Validation | Method |
|---|---|
| **Reproducer fixed** | Empirical: re-run the reported scenario; assert `explicit_library_preference: None`. |
| **No regression on truly-explicit cases** | Unit tests for "use playwright", "use selenium", "POST to /api/users", "parse XML and xpath". |
| **Conflict detection** | Unit test for "Test both Selenium and Playwright" → `library_preference_conflicts.web_automation` populated. |
| **Negation preserved** | Existing negation tests (if any) plus new test for "migrate from selenium to playwright" → Browser wins after negation. |
| **Threshold tunability** | Unit test: `ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD=4` allows weight-6 lone hits; default `8` blocks them. |
| **Evidence provenance** | Unit test: explicit detection includes `explicit_library_evidence` with the matched pattern. |
| **Performance** | No regression vs baseline. `tests/benchmarks/test_library_detection_bench.py` continues to pass within existing tolerances. v1's "<100µs aggregate" claim was unmeasured; v2 drops it. |

---

## 10. Open Questions

| Question | Recommended Default |
|---|---|
| `webdriver` weight stays at 6? | Yes — conflict-threshold 8 means it can't fire alone. |
| Move `chromium`/`webkit` to weight 9 (verbatim browser kernels)? | Yes — they're as specific as `playwright`. Folded into §6 in v2. |
| Should `library_preference_conflicts` also fire for non-conflict-group cases (e.g., DB + Web both mentioned)? | No — different domains; both are valid signals. Only fire WITHIN a group. |
| Should `explicit_library_evidence` include text spans? | Yes — pattern + the actual matched substring, capped at first match per pattern. |
| Sampling-based override: does it bypass these rules? | Yes (existing behaviour). Sampling's output is authoritative when enabled. v2 §11 defines the evidence/conflicts coherence. |
| Should non-English scenarios route through the rule-based detector? | Yes, returning None is the safe outcome. Future locale-specific pattern sets are out of scope. |

---

## 11. Sampling override coherence (v3 — TWO override sites)

LLM sampling overrides the rule-based detector at **two independent sites** in `server.py`. v2 documented only the first; v3 covers both:

**Site A — analyze_scenario response override** (`server.py:1860-1881`):
Overrides `analysis["explicit_library_preference"]` in the MCP response.

**Site B — session aggregate override** (`server.py:1961-1972`):
Overrides `session.explicit_library_preference` on the ExecutionSession.

Both sites are gated by `is_sampling_enabled()` which reads the **`ROBOTMCP_USE_SAMPLING`** env var (verified at `sampling.py:23`). v2 wrongly wrote `ROBOTMCP_USE_SAMPLING_FOR_NLP`; that env var doesn't exist.

**Coherence protocol (Site A — analysis response)**:

| Override state | `explicit_library_preference` | `preference_source` | `explicit_library_evidence` | `library_preference_conflicts` | `sampling_evidence` |
|---|---|---|---|---|---|
| No sampling (default) | rule-based winner or None | `"rule"` | rule-based matches | rule-based conflicts | absent |
| Sampling enabled, override applied | sampling value | `"sampling"` | **cleared** | **cleared** | model rationale |
| Sampling enabled, no override returned | rule-based winner or None | `"rule"` | rule-based matches | rule-based conflicts | absent |

**Coherence protocol (Site B — session aggregate)**:

| Override state | `session.explicit_library_preference` | `session.preference_source` |
|---|---|---|
| No sampling (default) | rule-based winner or None | `"rule"` |
| Sampling enabled, override applied | sampling value | `"sampling"` (NEW attribute) |
| Sampling enabled, no override | rule-based winner or None | `"rule"` |

Rationale: presenting rule-based evidence next to a sampling-derived preference would mislead consumers into thinking the patterns justified the choice. Clearing the fields and setting `preference_source` makes the source-of-truth unambiguous at both sites.

---

## 12. Test-surface compatibility (v3 — real contract)

v2's claim of `_compiled_patterns` compatibility was wrong. Source verification at `tests/integration/test_nlp_improvements.py:569-576` shows:

```python
raw_browser = any(
    p.findall("migrate from selenium to browser library for modern testing")
    for p, _ in library_detector._compiled_patterns.get("Browser", [])
)
```

The test:
1. Destructures with `for p, _ in ...` — requires **exactly 2 items per entry**. Python tuple unpacking enforces exact length; an `__iter__` yielding 4 raises `ValueError: too many values to unpack`.
2. Calls `p.findall(...)` — `p` must be a compiled `re.Pattern`, NOT a `PatternRule`.

v3 honours both requirements with a **two-store** design (per proposal §3.1 Step 7):

```python
class LibraryDetector:
    LIBRARY_RULES: Dict[str, List[PatternRule]]   # source of truth (rich)

    # v3: legacy store — Pattern tuples for test-fixture compat. UNCHANGED shape:
    _compiled_patterns: Dict[str, List[Tuple[re.Pattern, int]]]

    # v3: rich metadata store for the new detect_explicit_preference() path:
    _rules_metadata: Dict[str, List[PatternRule]]
```

`_compiled_patterns` keeps `(Pattern, int)` 2-tuples exactly as today. The `for p, _ in ...` test contract continues to work. The new code paths read `LIBRARY_RULES` or `_rules_metadata` directly to access the `explicit` and `rationale` annotations.

This preserves the existing test surface without changes to test code (other than the 3-5 false-positive-asserting tests enumerated in PRD §AC-8). `PatternRule` does NOT need `__iter__` — legacy tests never see `PatternRule` objects.
