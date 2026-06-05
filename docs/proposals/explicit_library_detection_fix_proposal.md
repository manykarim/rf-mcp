# Solution Proposal — Explicit Library Detection Fix

**Date**: 2026-05-29
**Status**: **IMPLEMENTED v7 — production-ready** (Codex round-7 verdict 2026-06-05). Source in `src/robotmcp/utils/library_detection.py`, `nlp_processor.py`, `session_models.py`, `server.py`. **83 targeted tests in `tests/unit/test_explicit_library_detection_fix.py`; full unit suite: 6249 passing.**
**Related**: PRD `docs/prd/analyze_scenario_explicit_library_prd.md`, ADR-024, DDD library_preference bounded context

---

## Revision history

**v7 (2026-06-05) — IMPLEMENTED** — Codex round-6 review (verdict: NOT production-ready) verified the 4 v6 fixes BUT found 3 new issues. v7 addresses 2 of them now (C1 + C3); C2 is deferred to a separate follow-up since it's a pre-existing architectural concern (sampling Site B not session-coherent — exists since long before v5):
- **C1 — Paragraph boundary restored (HIGH — fixed)**: v6's `_SENTENCE_DELIMITERS = r"[.;!?,]+"` dropped `\n` entirely to fix D-newline-negation, but then paragraph-separated text leaked negation across paragraphs. `"Do not use this approach\n\nUse Playwright for the test."` returned `None` (the negation reached into the second paragraph and zeroed Browser). v7 uses `r"[.;!?,]+|\n\s*\n+"` — single newline is NOT a boundary (preserves v6 fix); paragraph break (2+ newlines) IS a boundary (restores paragraph scope). Tested at `TestV7ParagraphBoundary` (3 paragraph + 4 single-newline cases).
- **C3 — `nlp_processor.py:663` abstain-fallback removed (MEDIUM — fixed)**: the same bug v6 fixed in `session_models.py` was still present in `nlp_processor._detect_explicit_library_preference`. The main `analyze_scenario` path bypasses this helper (uses `_resolve_explicit_library_preference` directly), so production wasn't affected — but any caller of the helper would re-introduce the bug. v7 mirrors the v6 fix: fall back ONLY on Exception, never on deliberate None. Tested at `TestV7NlpProcessorAbstainNoFallback` (5 cases + 2 genuine-positive regressions).
- **Doc propagation**: ADR §3.2 P3 sentence-split regex updated; PRD §FR-6 + Proposal Step 6 code blocks updated to show the v7 paragraph-aware delimiter; PRD/Proposal status promoted to `IMPLEMENTED v7`.
- **C2 deferred (separate PR)**: sampling Site B at `server.py:1964` overrides `session.explicit_library_preference` but doesn't re-run `configure_from_scenario`, so `session.loaded_libraries`/`search_order` stay on the rule-based pick. This is a session-coherence problem that existed before v5; needs its own ADR + change to either re-run configure after override OR move Site B before initial configuration. Out of scope for the analyze_scenario library-detection PRD.
- **Tests**: 14 new v7 tests (7 paragraph + 1 end-to-end via analyze_scenario + 6 abstain) — total `test_explicit_library_detection_fix.py` is now **83 tests, all green**. Full unit suite: **6249 passing, 1 skipped**.
- **Codex round-7 verdict (2026-06-05): PRODUCTION-READY YES.** All v7 fixes verified end-to-end through the public `analyze_scenario()` path including CRLF/mixed-whitespace edge cases. C2 (sampling Site B session-coherence) remains scoped to a separate follow-up.

**v6 (2026-06-05) — superseded by v7** — Codex round-5 review (PRD/DDD TIGHTEN; ADR/Proposal REWRITE) found that v5 fixed the analysis-path defect but the live implementation was NOT end-to-end coherent. 4 source bugs (3 critical) + doc propagation gaps. v6 fixes them:
- **D-session-fallback (CRITICAL — fixed)**: `session_models.py:597` was falling back to `_fallback_detect_library` (broad substring heuristic) when v5 returned None, re-introducing false-positive explicit preferences at the session aggregate level. Example: `analyze_scenario("Parse the XML response from the API")` correctly returned `analysis.explicit_library_preference: None`, but `session.explicit_library_preference` was set to `"XML"` by the fallback. Downstream consumers reading the session saw the wrong value. **v6 removes the fallback on None — only ImportError triggers it.**
- **D-newline-negation (CRITICAL — fixed)**: v5's `_SENTENCE_DELIMITERS = r"[.;!?,\n]+"` made `\n` a sentence boundary. `"do not use\nPlaywright"` split into two sentences; sentence 0 had the negation phrase but no target. **v6 removes `\n` from the delimiter** (now `[.;!?,]+`); newline-boundary at the explicit-pattern level is already enforced by `[^\S\n]+` so the newline-crossing preference verb protection is preserved.
- **D-repeated-token (CRITICAL — fixed)**: `_subtract_sentence_score` used `rule.compiled.search(sentence)` (returns once per rule regardless of occurrence count). `"do not use Playwright Playwright"` scored Browser=28 via 3 occurrences but only deducted 19 → Browser=9 wins despite negation. **v6 uses `len(rule.compiled.findall(sentence))` to count occurrences** and multiply by weight.
- **D-multiword-migration (HIGH — fixed)**: v5 migration destination lookahead `(?=[\s.,;!?]|$)` stopped at first whitespace, so `"Migrate from Selenium to Requests library"` captured `dst='Requests'` (bare `requests` excluded from `_LIBRARY_TOKENS`). Requests/Database/SSH/XML migrations all silently failed. **v6 lookahead is `(?=[.,;!?\n]|$)`** — destination captures up to next sentence punctuation; `_first_library_token_in()` then resolves canonical (matches `requests\s*library`, `database\s*library`, etc.).
- **Doc propagation**: ADR §3.2 P2 now declares the brand-only rule explicitly (`browser`/`requests`/`database`/`ssh`/`xml` excluded from preference-verb alternation); §6 pattern table updated for all 4 brand-only rows. PRD evidence-field name corrected (`pattern` singular, not `patterns_matched`). Proposal `_last_resolution` removed from sample code.
- **Tests**: 18 new tests added across 4 classes (`TestV6SessionFallbackBug`, `TestV6NewlineNegation`, `TestV6RepeatedTokenDeduction`, `TestV6MultiwordMigration`) plus `test_preference_verb_does_not_match_across_newline`. Total: 69 tests in the v6 file, all green. Full unit suite: **6235 passed** (1 skipped).

**v5 (2026-06-05) — superseded by v6** — Codex round-4 review marked v4 REWRITE; identified 1 critical algorithm bug (`primary_library` fallback semantic), 1 race condition (`_last_resolution` on singleton), 1 regression (standalone `skip` dropped), 1 cross-sentence false-positive gap (`\s+` matching newline), plus several dict-vs-tuple shape discrepancies. v5 addresses all:
- **D-skip restored**: `_NEGATION_REGEX` alternation now ends with `...|stop|avoid|skip|exclude` (standalone `skip` was missing in v4).
- **Newline boundary**: preference-verb patterns use `[^\S\n]+` instead of `\s+`. Newlines act as sentence delimiters; `\b(use|with|...)\b\s+\bselenium\b` no longer matches `use\nselenium` across a line break.
- **Race-free invocation**: removed `self._last_resolution` cache on `NaturalLanguageProcessor` singleton. The new `_resolve_explicit_library_preference()` returns the resolution as a local variable; `_build_analysis_block()` consumes it inline. Tested at `tests/unit/test_explicit_library_detection_fix.py:TestNoSharedState`.
- **Sampling semantic fix**: dropped `primary_library` fallback at server.py:1860-1881 (Site A). Only `library_preference` writes to `explicit_library_preference`. `preference_source` flips to "sampling" + evidence/conflicts cleared.
- **§3.2 emit-bug fixed**: v4 had `for lib, score, patterns in conflict_list` (tuple unpacking) but `resolution.conflicts` returns dicts. v5 passes dict entries through unchanged.
- **§3.4 test count reconciled**: §3.4 (was "10 new tests"), §5 (18), §9 (18) — all now say 18. Actual count: 51 tests in the implemented test file (more granular than the original 18-distinct-function design; parametrisation expanded).
- **ALL DOC PROPAGATION CLOSED**: ADR §3.2 code block, PRD §FR-1 / §FR-6 bare-noun list and algorithm code, DDD §2 overview shape strings — all aligned with the actually-shipped implementation.

**SOURCE/TEST EVIDENCE**: `src/robotmcp/utils/library_detection.py` (full v5 implementation), `tests/unit/test_explicit_library_detection_fix.py` (51 tests covering 8 marquee scenarios + AC-1..AC-7 + sampling coherence + race-free invocation + sentence_index evidence filtering). Pre-existing test count was 6166; v5 adds 51 → 6217 passing.

**v4 (2026-05-29)** — Third-round independent reviewer found one algorithmic bug + 7 consistency/specification gaps in v3. v4 addresses:
- **D1 — Double-deduction eliminated**: v3 iterated `NEGATION_PHRASES` as a Python list, so both `\bdo\s+not\s+use\b` and `\bdo\s+not\b` fired on the same span for `"do not use Selenium"` and called `_subtract_sentence_score` twice. v3 worked by luck (`max(0, …)` clamps), but for `"do not use Selenium and stop the SeleniumLibrary"` the double-application risked corrupting partial-signal libraries. v4 replaces the list with a **single regex with alternation** ordered longest-first. The regex engine's left-to-right alternation guarantees longest match wins; each negation span fires exactly once. **Verified empirically against 8 scenarios including v3 edge cases.**
- **D3 — Test unpacking fixed (proposal §3.4)**: v3 `TestConflicts.test_both_selenium_and_playwright_returns_none_with_conflict` extracts `entry[0]` (tuple style) but algorithm Step 6 returns dicts. v4 fixes the test to use `entry["library"]`.
- **D3 — DDD §4.1.6 conflicts type updated**: DDD `Dict[str, List[Tuple[str, int, List[str]]]]` (tuple) → `Dict[str, List[Dict[str, Any]]]` (dict-of-dicts) to match PRD §FR-4, ADR §3.4, proposal Step 6.
- **D4 — `ExecutionSession.preference_source` field declared**: §3.3 v4 now adds the field to the ExecutionSession dataclass as `preference_source: Optional[str] = "rule"`. v3 wrote to the attribute without specifying where it came from.
- **D2 — INV-4 strengthened**: replaced trivial `max(0, x-d)` idempotence with **"deduction-sum equality"** — the property that the sum of deductions across all negation-phrase matches in a sentence equals the deduction from a single canonical phrase. This DOES catch D1 double-deduction.
- **D5 — Test count reconciled to 18**: §9 Validation Plan dropped the stale "~30 new + ~3 modified" wording. Matrix and validation now both say "18 distinct functions, ~37 invocations after parametrisation".
- **D6 — Multi-word migration limitation documented**: §3.1 Step 5 v4 notes that `"to Browser library"` migrations silently no-op because `_LIBRARY_TOKENS` excludes bare `browser`. Acknowledged as intentional consequence of the bare-noun fix.
- **D7 — Evidence filtering via `sentence_index` field**: v3's `pm.text_span.lower() not in sentence.lower()` substring check could corrupt evidence across multi-sentence overlap. v4 adds `sentence_index: int = -1` to `PatternMatch` and filters by exact sentence membership, not substring inclusion. `sentence_index` is internal — not surfaced in `to_dict()`.
- **D8 — `Literal` import added**: §3.1 Step 1 imports updated to include `Literal` from typing for `PreferenceResolution.source` annotation.
- **F5 — `LIBRARY_RULES_DEFAULT` declared**: §3.1 Step 7 now explicitly defines `LIBRARY_RULES_DEFAULT` (referenced but not declared in v3) as the module-level constant carrying the v4 pattern table.

**v3 (2026-05-29)** — Codex CLI second-round critical review identified 11 hard failures in v2 (verdict: REWRITE). v3 addresses them:
- **Negation algorithm replaced (§3.1 Step 5)**: v2's single-regex `\b(not|don't|...)\s+(?:using\s+)?(?P<target>\w+)` captured `target="use"` (not the library) for "do not use Selenium". Codex ran the Python and proved v2 left SL=16, Browser=19 (raw conflict), not the documented Browser. v3 replaces with **phrase-list + find_first_library_token(remaining_text)** approach. Sentence delimiter now `[.;!?,\n]+` (includes comma) so "do not use Selenium, instead use Playwright" splits into 2 sentences.
- **`_compiled_patterns` compat actually preserved (§3.1 Step 7)**: real tests do `for p, _ in _compiled_patterns; p.findall(...)`. v2's `PatternRule.__iter__` yielding 4 values would raise `ValueError`, and `PatternRule` has no `.findall`. v3 keeps `_compiled_patterns: Dict[str, List[Tuple[Pattern, int]]]` EXACTLY as today and introduces a parallel `_rules_metadata` dict for the rich annotations.
- **Bare-token preference verbs dropped (§3.1 Step 2)**: `use browser`, `use database`, `use ssh`, `use requests`, `use xml` removed from explicit-preference patterns. These are generic English/domain nouns. Brand names `selenium`, `playwright`, `appium` retained as explicit because they're library-specific identifiers, not generic English.
- **`PatternMatch` defined (§3.1 Step 4)**: v2 used `PatternMatch(...)` without declaring it. v3 defines it explicitly with the `library` field per DDD §4.1.2.
- **`PatternRule.compiled` defined (§3.1 Step 1)**: v2 used `rule.compiled.finditer(...)` without declaring `compiled`. v3 adds a `compiled` property via `__post_init__` setting.
- **`PreferenceResolution` fields aligned (§3.1 Step 4)**: v2 declared 4 fields then constructed with 5. v3 canonical shape: `(library, source, evidence, conflicts, all_scores, sampling_evidence)` — 6 fields.
- **Sampling env var corrected (§3.2.1 + §3.4)**: `ROBOTMCP_USE_SAMPLING` (not `ROBOTMCP_USE_SAMPLING_FOR_NLP` as v2 mistakenly wrote — verified at `sampling.py:23`).
- **Both sampling override sites covered (§3.2.1)**: v2 documented only `server.py:1860-1881` (the analysis-response override). Real code also has `server.py:1961-1972` modifying `session.explicit_library_preference`. v3 patches both.
- **Scope table corrected (§3)**: 4 files (server.py added).
- **Test matrix totals fixed (§5)**: matrix totals are 16 distinct functions across 9 classes; v3 honestly counts to match.
- **`keyword_executor` consumer description corrected (§12)**: real code at `keyword_executor.py:1901-1914` hardcodes the `Open Browser` keyword check based on `pref.startswith("selenium" | "browser")`; v2's "library_name= fallback" claim was inaccurate.
- **chromium/webkit weight aligned to 9 (§3.1 Step 2)**: matches ADR §6 "as specific as playwright" rationale.

**v2 (2026-05-29)** — Codex CLI critical review identified 12 substantive issues in the proposal. Key changes:
- **Pattern table corrected (§3.1)**: dropped wrongly-annotated `explicit=True` for `rest api testing`, `mobile automation`, `android testing`, `ios testing`, SQL fragments, DB engine names, XPath/XSD/XSLT, bare `xml`. v1 annotated these as `explicit=True` despite them being domain markers, not library identifiers.
- **Negation algorithm replaced (§3.1 Step 5)**: v1's 80-character forward window broke `"do not use Selenium, instead use Playwright"` by zeroing both libraries. v2 uses sentence-scoped algorithm with separate `NEGATION_PATTERNS` (single-library) and `MIGRATION_PATTERNS` (source-vs-destination).
- **Algorithm ordering corrected (§3.1 Step 5)**: conflict check now runs on RAW explicit scores BEFORE threshold filter. v1's "filter then check" sequence made the AC-5 example impossible.
- **Class name fix (§3.2/§3.4)**: `NLPProcessor` → `NaturalLanguageProcessor` (actual class name at `nlp_processor.py:31`). v1 wrote bogus imports.
- **Test count reconciled (§3.4/§5)**: 10 new tests per PRD v2 AC-8, not the inconsistent 25 / 30 v1 had.
- **Impossible AC-5 test fixed (§3.4 TestConflicts)**: rewrote the "both selenium and playwright" test to match v2's raw-score conflict-detection algorithm.
- **Existing-test updates listed (§3.4)**: 3-5 tests asserting false-positive behaviour are now enumerated with file:line locations.
- **`_compiled_patterns` compatibility added (§3.1)**: implementation must preserve this internal attribute since `tests/integration/test_nlp_improvements.py:103-104,571-576` inspects it directly.
- **Performance claim removed (§6)**: replaced unsupported "≤60µs per call" with "no measurable regression vs baseline benchmark".
- **Sampling override coherence implemented (§3.2)**: when `server.py:1866-1880` overrides, clear evidence/conflicts and set `preference_source: "sampling"`.
- **Downstream consumer compatibility (§12, NEW)**: explicit section covering library_recommender, adapter_factory, keyword_executor, browser_plugin, selenium_plugin — five consumers v1 missed.
- **Migration patterns added (§3.1 Step 5b)**: new `MIGRATION_PATTERNS` constant with `from X to Y` / `switch X to Y` / `instead of X use Y` / `replace X with Y` idioms.

**v1 (2026-05-29)** — initial draft.

---

## 1. Summary

`analyze_scenario` reports `explicit_library_preference: "SeleniumLibrary"` for the scenario *"Test e-commerce website https://demoshop.makrocode.de: open browser, add items to shopping cart, ..."* even though the user never mentioned SeleniumLibrary. Root cause: a single overzealous regex pattern (`\bopen\s+browser\b` weight 6) above the min_score threshold (5) at `src/robotmcp/utils/library_detection.py:33`.

The fix introduces a clean split between **library mention** (broad, advisory) and **library preference** (narrow, decisive), removes keyword-name patterns from the preference scoring, adds a conflict-group threshold (8 for Browser-vs-SL conflicts), and surfaces ambiguity to the caller via two new optional response fields.

This document is the concrete implementation guide that follows the PRD/ADR/DDD design. It includes step-by-step code edits, test fixtures, env-var tunables, and a migration path.

---

## 2. Reproducer

```python
# CURRENT (broken) behaviour — verified 2026-05-29
from robotmcp.utils.library_detection import get_library_detector
detector = get_library_detector()
scenario = (
    "Test e-commerce website https://demoshop.makrocode.de: "
    "open browser, add items to shopping cart, verify items, "
    "complete checkout, and close browser"
)
print(detector.detect(scenario))  # → "SeleniumLibrary" (WRONG)
print(detector.get_scores(scenario))  # → {"SeleniumLibrary": 6}
```

The single match comes from `\bopen\s+browser\b` (weight 6, ≥ default min_score 5).

---

## 3. Proposed Code Changes

The changes are scoped to **four files** for the core fix and **one** for the new tests (v3 correction — v2's "three files" missed `server.py`):

| File | Change type | LOC delta (rough) |
|---|---|---|
| `src/robotmcp/utils/library_detection.py` | Refactor + new entry point | +150 / −20 |
| `src/robotmcp/components/nlp_processor.py` | Wire new entry point + surface evidence/conflicts | +30 |
| `src/robotmcp/models/session_models.py` | Route through new entry point | +5 / −10 |
| `src/robotmcp/server.py` | Sampling override coherence (both sites: lines 1866-1880 + 1961-1972) | +15 |
| `tests/unit/test_explicit_library_detection_fix.py` | New test file | +200 |

### 3.1 `library_detection.py` — refactor + annotate

#### Step 1: introduce `PatternRule` and `PatternMatch` dataclasses

`PatternRule` is the new annotated source-of-truth pattern entry. `PatternMatch` is the evidence record returned by the resolver. Both `compiled` and the validation logic are defined upfront (v2 omitted these and downstream code referenced them as if they existed).

```python
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Pattern, Tuple  # v4: Literal added


@dataclass(frozen=True)
class PatternRule:
    """A single regex pattern that scores toward a library identification."""
    pattern: str          # raw regex source
    weight: int           # 1-10 — strength of the signal
    explicit: bool        # True → contributes to preference; False → mention only
    rationale: str        # one-line audit comment

    # `compiled` is set in __post_init__ via object.__setattr__ because the
    # dataclass is frozen. Consumers use rule.compiled.finditer(text), etc.
    compiled: Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not (1 <= self.weight <= 10):
            raise ValueError(f"weight must be 1-10, got {self.weight}")
        # Fail fast on bad regex AND cache the compiled object
        object.__setattr__(self, "compiled", re.compile(self.pattern, re.IGNORECASE))


@dataclass(frozen=True)
class PatternMatch:
    """A single (pattern, span, library) record used as evidence."""
    library: str          # canonical library name (v3 — per DDD §4.1.2)
    pattern: str          # regex source (for audit)
    weight: int
    text_span: str        # the substring that matched (first occurrence)
    sentence_index: int = -1  # v4: which sentence the match came from
                              # (internal — used by _subtract_sentence_score
                              # for exact-sentence evidence filtering; NOT
                              # included in to_dict() output)

    def to_dict(self) -> Dict[str, Any]:
        # v4: sentence_index INTENTIONALLY omitted — internal-only
        return {
            "library": self.library,
            "pattern": self.pattern,
            "weight": self.weight,
            "text_span": self.text_span,
        }
```

#### Step 2: replace `LIBRARY_PATTERNS` table

Convert each entry to a `PatternRule`. Annotate `explicit` per the ADR-024 §6 table.

Concrete decisions for the patterns currently identified as causing false positives:

v3 NOTE on bare-token preference verbs: Codex correctly flagged that `use browser`, `use database`, `use ssh`, `use requests`, `use xml` are generic English nouns and should NOT be `explicit=True`. v3 keeps brand names (`selenium`, `playwright`, `appium`) inside preference-verb patterns because those tokens are library-specific identifiers, but removes the generic nouns. The bare nouns remain in the mention layer at lower weight.

```python
'SeleniumLibrary': [
    # P1 — verbatim library identifiers (explicit)
    # NOTE: 'selenium' is a brand/library name (not a generic English noun) so it
    # stays inside the preference-verb pattern.
    PatternRule(r'\b(use|using|with|via|through|prefer)\s+(selenium|seleniumlibrary|selenium\s*library)\b',
                10, explicit=True, rationale="preference verb + selenium brand token"),
    PatternRule(r'\bseleniumlibrary\b', 9, explicit=True, rationale="verbatim library name"),
    PatternRule(r'\bselenium\b', 6, explicit=True, rationale="standalone selenium mention"),
    PatternRule(r'\bwebdriver\b', 6, explicit=True, rationale="selenium-specific WebDriver term"),
    PatternRule(r'\b(chromedriver|geckodriver|edgedriver|safaridriver)\b',
                7, explicit=True, rationale="selenium drivers"),
    PatternRule(r'\bselenium\s+grid\b', 8, explicit=True, rationale="selenium-specific tech"),
    PatternRule(r'\bselenium\s+standalone\b', 7, explicit=True, rationale="selenium-specific tech"),
    PatternRule(r'\bclassic\s+selenium\b', 7, explicit=True, rationale="selenium-specific phrasing"),
    PatternRule(r'\bselenium\s+automation\b', 7, explicit=True, rationale="selenium-domain phrasing"),
    PatternRule(r'\b(selenium\s+(2|3|4)|selenium2library)\b', 8, explicit=True, rationale="selenium version mention"),
    PatternRule(r'\b(desired\s+capabilities|driver\s+capabilities)\b',
                7, explicit=True, rationale="selenium Capabilities API"),
    PatternRule(r'\b(create\s+webdriver|get\s+webelement)\b',
                8, explicit=True, rationale="selenium keyword names that aren't generic NL"),
    PatternRule(r'\btest\s+automation\s+with\s+selenium\b',
                8, explicit=True, rationale="explicit phrasing"),
    # Mention-only (NOT explicit) — kept for capability suggestion
    PatternRule(r'\bopen\s+browser\b', 6, explicit=False,
                rationale="REMOVED from explicit — generic NL verb that overlaps with SL keyword"),
    PatternRule(r'\b(input\s+text|click\s+element|page\s+should\s+contain)\b',
                6, explicit=False,
                rationale="REMOVED from explicit — SL keyword names but also generic NL"),
    PatternRule(r'\b(implicit|explicit)\s+wait\b',
                6, explicit=False,
                rationale="REMOVED from explicit — generic concept across web libraries"),
],
'Browser': [
    # P1 — verbatim. v3: bare 'browser' DROPPED from preference verb (too generic
    # — could mean a web browser app, not the RF Browser library). Only
    # library-specific tokens remain.
    PatternRule(r'\b(use|using|with|via|through|prefer)\s+(playwright|browserlibrary|browser\s*library|rfbrowser|robotframework[- ]browser|playwright[- ]core)\b',
                10, explicit=True, rationale="preference verb + Browser-specific token (NO bare 'browser')"),
    PatternRule(r'\bbrowser\s*library\b', 9, explicit=True, rationale="verbatim library name"),
    PatternRule(r'\bplaywright\b', 9, explicit=True, rationale="verbatim Browser library underlying tech"),
    PatternRule(r'\b(rfbrowser|robotframework[- ]browser|playwright[- ]core)\b',
                9, explicit=True, rationale="verbatim package names"),
    PatternRule(r'\bchromium\b', 9, explicit=True, rationale="v3: weight 9 (Playwright kernel — as specific as 'playwright', aligned with ADR §6)"),
    PatternRule(r'\bwebkit\b', 9, explicit=True, rationale="v3: weight 9 (Playwright kernel — as specific as 'playwright')"),
    # Bare 'browser' alone is mention-only (could be SL OR Browser OR generic NL)
    PatternRule(r'\bbrowser\b', 4, explicit=False,
                rationale="v3: bare 'browser' is too generic for explicit; kept in mention layer"),
    # Mention-only — REMOVED from explicit
    PatternRule(r'\bmodern\s+web\s+testing\b', 7, explicit=False, rationale="marketing copy"),
    PatternRule(r'\bmodern\s+browser\s+automation\b', 8, explicit=False, rationale="marketing copy"),
    PatternRule(r'\bcross[- ]browser\s+testing\b', 6, explicit=False, rationale="generic test type"),
    PatternRule(r'\bnew\s+(browser|page|context)\b', 8, explicit=False,
                rationale="REMOVED from explicit — Browser keyword names but also generic NL"),
    PatternRule(r'\bfill\s+(text|secret)\b', 7, explicit=False,
                rationale="REMOVED from explicit — Browser keyword name but also generic NL"),
    PatternRule(r'\b(headless\s+browser|headless\s+chromium)\b', 6, explicit=False,
                rationale="generic test config"),
    PatternRule(r'\b(shadow\s+dom|web\s+components?)\b', 6, explicit=False,
                rationale="modern web concept; not Browser-exclusive"),
    PatternRule(r'\b(SPA|single\s+page\s+app(lication)?)\b', 5, explicit=False,
                rationale="describes app architecture"),
    PatternRule(r'\b(e2e|end.to.end)\s+(test|automat)', 5, explicit=False,
                rationale="generic test type"),
],
'RequestsLibrary': [
    # P1 — verbatim + preference-verb. v3: bare 'requests' DROPPED (Python has a
    # 'requests' package; users often mean "make HTTP requests" not RequestsLibrary).
    PatternRule(r'\b(use|using|with|via|through|prefer)\s+(requestslibrary|requests\s*library)\b',
                10, explicit=True, rationale="preference verb + Requests-specific token (NO bare 'requests')"),
    PatternRule(r'\brequestslibrary\b', 9, explicit=True, rationale="verbatim library name"),
    # Bare 'requests' alone is mention-only
    PatternRule(r'\brequests\b', 4, explicit=False,
                rationale="v3: bare 'requests' could mean Python requests package or generic HTTP; kept in mention layer"),
    # RL-specific keyword names with RL-specific session noun
    PatternRule(r'\b(create\s+session|get\s+on\s+session|post\s+on\s+session)\b',
                8, explicit=True, rationale="RL keyword names with 'session' qualifier"),
    PatternRule(r'\b(status\s+should\s+be|request\s+should\s+be)\b',
                7, explicit=True, rationale="RL-specific keyword phrasing"),
    PatternRule(r'\b(GET|POST|PUT|DELETE|PATCH)\s+on\s+session\b',
                7, explicit=True, rationale="HTTP-method + RL 'on session' qualifier — the qualifier makes it RL-specific"),
    # Mention-only (v2: rest api testing + status code + bare HTTP methods moved from explicit=True)
    PatternRule(r'\brest\s+api\s+testing\b', 7, explicit=False,
                rationale="v2: REMOVED from explicit — domain marker, not library identifier"),
    PatternRule(r'\bapi\s+automation\b', 6, explicit=False,
                rationale="v2: REMOVED from explicit — domain marker"),
    PatternRule(r'\bstatus\s+code\b', 5, explicit=False,
                rationale="v2: REMOVED from explicit — generic HTTP term"),
    PatternRule(r'\b(GET|POST|PUT|DELETE|PATCH)\s+request\b', 5, explicit=False,
                rationale="v2: bare HTTP verb + request is domain-generic; only the 'on session' form is RL-specific"),
    PatternRule(r'\bhttp\s+requests?\b', 5, explicit=False, rationale="generic HTTP term"),
    PatternRule(r'\b(webservice|web\s+service)\b', 5, explicit=False, rationale="generic"),
    PatternRule(r'\bmicroservice\b', 5, explicit=False, rationale="generic"),
    PatternRule(r'\b(bearer\s+token|JWT|OAuth2?)\b', 5, explicit=False, rationale="auth concepts"),
    PatternRule(r'\b(swagger|openapi)\b', 5, explicit=False, rationale="api-doc tooling"),
    PatternRule(r'\b(graphql|gRPC|SOAP)\b', 5, explicit=False, rationale="API protocols"),
    PatternRule(r'\b(webhook|callback\s+url)\b', 5, explicit=False, rationale="generic"),
],
'AppiumLibrary': [
    # P1 — verbatim + preference-verb
    PatternRule(r'\b(use|using|with|via|through|prefer)\s+(appium|appiumlibrary|appium\s*library)\b',
                10, explicit=True, rationale="preference verb + Appium token"),
    PatternRule(r'\bappium(?:library)?\b', 9, explicit=True, rationale="verbatim library / runtime name"),
    PatternRule(r'\b(UIAutomator2?|XCUITest|Espresso)\b', 7, explicit=True,
                rationale="Appium-specific runtime (UIAutomator/XCUITest/Espresso are Appium drivers)"),
    # Mention-only (v2: mobile/android/ios domain markers + app/action verbs moved from explicit=True)
    PatternRule(r'\bmobile\s+automation\b', 7, explicit=False,
                rationale="v2: REMOVED from explicit — mobile automation is a domain, could be Appium, Detox, Maestro, etc."),
    PatternRule(r'\bmobile\s+app\s+testing\b', 7, explicit=False,
                rationale="v2: REMOVED from explicit — domain marker"),
    PatternRule(r'\bandroid\s+testing\b', 6, explicit=False,
                rationale="v2: REMOVED from explicit — platform, not library identifier"),
    PatternRule(r'\bios\s+testing\b', 6, explicit=False,
                rationale="v2: REMOVED from explicit — platform, not library identifier"),
    PatternRule(r'\b(open\s+application|close\s+application)\b', 8, explicit=False,
                rationale="Appium keyword names but generic NL"),
    PatternRule(r'\b(tap|swipe|long\s+press|double\s+tap|flick|scroll|pinch)\b', 6, explicit=False,
                rationale="mobile action verbs"),
    PatternRule(r'\bdevice\b', 5, explicit=False, rationale="too generic"),
    PatternRule(r'\b(emulator|simulator)\b', 5, explicit=False, rationale="generic"),
    PatternRule(r'\b(native\s+app|hybrid\s+app|webview)\b', 6, explicit=False, rationale="generic mobile"),
    PatternRule(r'\b(APK|IPA|bundle\s+id|package\s+name)\b', 6, explicit=False, rationale="mobile artifacts"),
    PatternRule(r'\b(device\s+farm|BrowserStack|Sauce\s+Labs)\b', 5, explicit=False, rationale="cloud services"),
    PatternRule(r'\b(iphone|ipad|tablet|smartphone)\b', 5, explicit=False, rationale="device names"),
],
'DatabaseLibrary': [
    # P1 — verbatim + preference-verb. v3: bare 'database' DROPPED (generic domain noun).
    PatternRule(r'\b(use|using|with|via|through|prefer)\s+(databaselibrary|database\s*library)\b',
                10, explicit=True, rationale="preference verb + Database-specific token (NO bare 'database')"),
    PatternRule(r'\bdatabaselibrary\b', 9, explicit=True, rationale="verbatim library name"),
    PatternRule(r'\b(connect\s+to\s+database|execute\s+sql|call\s+stored\s+procedure)\b',
                8, explicit=True, rationale="DatabaseLibrary keyword names"),
    # Mention-only (v2: SQL fragments + DB engine names moved from explicit=True)
    PatternRule(r'\b(SELECT|INSERT|UPDATE|DELETE)\s+(FROM|INTO|SET|\*)\b', 5, explicit=False,
                rationale="v2: REMOVED from explicit — SQL is the user's domain; could be DatabaseLibrary or raw psycopg2-via-Python wrapper"),
    PatternRule(r'\b(postgres(?:ql)?|mysql|mariadb|sqlite)\b', 5, explicit=False,
                rationale="v2: REMOVED from explicit — DB engine name, not RF library identifier"),
    PatternRule(r'\b(oracle|sql\s+server|mssql|mongodb)\b', 5, explicit=False,
                rationale="v2: REMOVED from explicit — DB engine name, not RF library identifier"),
    PatternRule(r'\bsql\s+testing\b', 6, explicit=False, rationale="generic SQL test type"),
    PatternRule(r'\bdatabase\s+validation\b', 6, explicit=False, rationale="generic"),
    PatternRule(r'\b(row\s+count|check\s+if\s+exists)\b', 7, explicit=False,
                rationale="DB-keyword names but generic NL"),
    PatternRule(r'\b(connection\s+string|DSN|ODBC)\b', 5, explicit=False, rationale="generic DB config"),
    PatternRule(r'\bstored\s+procedure\b', 6, explicit=False, rationale="generic SQL concept"),
    PatternRule(r'\b(CRUD|schema\s+migration)\b', 5, explicit=False, rationale="generic"),
],
'SSHLibrary': [
    # P1 — verbatim + preference-verb. v3: bare 'ssh' DROPPED (protocol noun;
    # 'use ssh to deploy' could mean ssh CLI or paramiko).
    PatternRule(r'\b(use|using|with|via|through|prefer)\s+(sshlibrary|ssh\s*library)\b',
                10, explicit=True, rationale="preference verb + SSH-specific token (NO bare 'ssh')"),
    PatternRule(r'\bsshlibrary\b', 9, explicit=True, rationale="verbatim library name"),
    # SSH-specific keyword names + protocols
    PatternRule(r'\blogin\s+with\s+public\s+key\b',
                7, explicit=True, rationale="SSHLibrary-specific keyword name"),
    PatternRule(r'\b(sftp|scp)\b', 6, explicit=True, rationale="SSH-specific protocols"),
    # Mention-only (v2: execute command, remote server, ssh into — all moved from True or remain False)
    PatternRule(r'\bopen\s+connection\b', 7, explicit=False,
                rationale="v2: REMOVED from explicit — overlaps with Telnet, Database, Browser keyword names"),
    PatternRule(r'\b(execute\s+command|start\s+command)\b', 6, explicit=False,
                rationale="generic command-execution terms"),
    PatternRule(r'\b(get\s+file|put\s+file|get\s+directory|put\s+directory)\b',
                6, explicit=False, rationale="generic file ops"),
    PatternRule(r'\bssh\s+(into|to)\b', 6, explicit=False,
                rationale="v2: action verb, not library identifier — could be sshpass, paramiko-script, etc."),
    PatternRule(r'\bremote\s+server\s+commands?\b', 5, explicit=False, rationale="generic"),
    PatternRule(r'\b(remote\s+(server|execution|machine|host))\b', 5, explicit=False, rationale="generic"),
    PatternRule(r'\b(linux|unix)\s+(server|machine|system)\b', 5, explicit=False, rationale="OS-generic"),
],
'XML': [
    # P1 — verbatim + preference-verb. Bare 'xml' already kept out of preference
    # (XML is a file format, not a library identifier).
    PatternRule(r'\b(use|using|with|via|through|prefer)\s+(xmllibrary|xml\s*library)\b',
                10, explicit=True, rationale="preference verb + XML library-specific token (NO bare 'xml')"),
    PatternRule(r'\bxmllibrary\b', 9, explicit=True, rationale="verbatim library name"),
    # Mention-only (v2: parse xml, xml parsing/validation, XSD/XSLT/XPath all moved from True)
    PatternRule(r'\bxml\s+parsing\b', 6, explicit=False,
                rationale="v2: REMOVED from explicit — XML parsing is the user's task, not a library choice"),
    PatternRule(r'\bxml\s+validation\b', 6, explicit=False, rationale="v2: same as above"),
    PatternRule(r'\b(parse\s+xml|save\s+xml|log\s+element)\b', 7, explicit=False,
                rationale="v2: XMLLibrary keyword names but also generic NL"),
    PatternRule(r'\b(xslt|dtd|xsd)\b', 6, explicit=False,
                rationale="v2: REMOVED from explicit — XML standards, not RF library identifier"),
    PatternRule(r'\bxpath\s+(expression|query|selector)\b', 6, explicit=False,
                rationale="v2: REMOVED from explicit — XPath is also used by SeleniumLibrary locators"),
    PatternRule(r'\b(get\s+element\s+text|get\s+element\s+attribute)\b',
                6, explicit=False, rationale="overlaps with other libs' keyword names"),
    PatternRule(r'\b(namespace|element\s+tree|lxml)\b', 5, explicit=False, rationale="generic XML lib terms"),
    PatternRule(r'\bxml\s+(file|document|response|config)\b', 5, explicit=False, rationale="generic XML mention"),
    PatternRule(r'\b\bxml\b', 4, explicit=False, rationale="too generic; could mean XML file format"),
],
```

#### Step 3: introduce `DetectionPolicy`

```python
@dataclass(frozen=True)
class DetectionPolicy:
    default_min_score: int = 5
    conflict_min_score: int = 8
    ambiguity_window: int = 4

    @classmethod
    def from_env(cls) -> "DetectionPolicy":
        return cls(
            default_min_score=int(os.getenv("ROBOTMCP_LIBRARY_DETECTION_MIN_SCORE", "5")),
            conflict_min_score=int(os.getenv("ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD", "8")),
            ambiguity_window=int(os.getenv("ROBOTMCP_LIBRARY_DETECTION_AMBIGUITY_WINDOW", "4")),
        )
```

#### Step 4: introduce `PreferenceResolution` (v3 — 6 fields, canonical shape)

v2 declared 4 fields (`library`, `evidence`, `conflicts`, `all_scores`) but Step 6's constructor passed 5 positional args plus `source="rule"` named arg — internally inconsistent. v3 uses the canonical 6-field shape from DDD §4.1.6:

```python
from typing import Literal

@dataclass(frozen=True)
class PreferenceResolution:
    """Public output of detect_explicit_preference. v3 canonical shape:
    6 fields used consistently across all 4 docs."""
    library: Optional[str]                                # the chosen library, or None
    source: Literal["rule", "sampling"]                   # provenance
    evidence: List[Dict[str, Any]]                        # [{library, pattern, weight, text_span}]
    conflicts: Dict[str, List[Dict[str, Any]]]            # group_name -> [{library, score, patterns_matched}]
    all_scores: Dict[str, int]                            # full score map (diagnostics)
    sampling_evidence: Optional[str] = None               # LLM rationale when source=sampling

    @property
    def is_decisive(self) -> bool:
        return self.library is not None

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)
```

Canonical evidence-entry shape (v3 — used consistently in PRD/ADR/DDD/proposal): each evidence entry is `{library: str, pattern: str, weight: int, text_span: str}`. The list is flat (NOT grouped by library); the `library` field on each entry lets consumers filter or group as needed. This shape matches `PatternMatch.to_dict()` from Step 1.

Canonical conflicts-entry shape: each conflict-group value is a list of `{library: str, score: int, patterns_matched: List[str]}`.

#### Step 5: define `NEGATION_PHRASES`, `MIGRATION_PATTERNS`, `_LIBRARY_TOKENS` (v3 — fixed algorithm)

v2 used `\b(?:not|...)\s+(?:using\s+)?(?P<target>\w+)`. Codex ran the Python and showed `target="use"` (not the library) for `"do not use Selenium"`. v3 abandons that approach in favour of a **two-step phrase-then-token-lookup**:

```python
# v4 — SINGLE regex with alternation, ordered longest-first. The regex engine's
# left-to-right alternation guarantees the longest negation phrase wins at each
# position. Each span fires exactly ONCE — no double-deduction.
#
# v3 used a Python list of phrases iterated with `for phrase in NEGATION_PHRASES`,
# which fired both `\bdo\s+not\s+use\b` AND `\bdo\s+not\b` on the same span and
# called `_subtract_sentence_score` twice. v3 worked by luck because `max(0, …)`
# clamps at zero; v4 closes that hole.
_NEGATION_REGEX: Pattern = re.compile(
    r"\b(?:"
    # Compound phrases (longest first — regex engine picks the longest match)
    r"do\s+not\s+use|don't\s+use|"
    r"not\s+using|without\s+using|stop\s+using|avoid\s+using|"
    r"skip\s+using|exclude\s+using|"
    # Simple phrases
    r"do\s+not|don't|"
    r"without|stop|avoid|exclude"
    r")\b",
    re.IGNORECASE,
)

# v3 — migration patterns return SOURCE and DESTINATION text spans. Token
# resolution happens via `_first_library_token_in()` so multi-word forms
# like "Browser Library" or hyphenated "robotframework-browser" are handled
# correctly. v2's `\w+` capture broke on hyphens and multi-word names.
MIGRATION_PATTERNS: List[str] = [
    r"\bmigrat(?:e|ion|ing)\b.+?\bfrom\b\s+(?P<src>.+?)\s+\bto\b\s+(?P<dst>.+?)(?=[\s.,;!?]|$)",
    r"\bswitch(?:ing)?\s+from\b\s+(?P<src>.+?)\s+\bto\b\s+(?P<dst>.+?)(?=[\s.,;!?]|$)",
    r"\binstead\s+of\b\s+(?P<src>.+?)\s+\b(?:use|with|via)\b\s+(?P<dst>.+?)(?=[\s.,;!?]|$)",
    r"\breplace\b\s+(?P<src>.+?)\s+\b(?:with|by|for)\b\s+(?P<dst>.+?)(?=[\s.,;!?]|$)",
]

# v3 — Library tokens for negation/migration target resolution.
# v3 NOTE: this is the canonical map used by `_first_library_token_in()`.
# Entries deliberately EXCLUDE generic English nouns ('browser', 'database',
# 'ssh', 'requests', 'xml') because users writing "do not use ssh" usually
# mean "do not use the SSH protocol" not "do not use SSHLibrary".
_LIBRARY_TOKENS: Dict[str, List[str]] = {
    "SeleniumLibrary": [
        "selenium", "seleniumlibrary", r"selenium\s*library",
        "selenium2library", "selenium3library",
        "webdriver", "chromedriver", "geckodriver", "edgedriver", "safaridriver",
    ],
    "Browser": [
        "playwright", "browserlibrary", r"browser\s*library",
        "rfbrowser", r"robotframework[- ]browser", r"playwright[- ]core",
        "chromium", "webkit",
    ],
    "RequestsLibrary": [
        "requestslibrary", r"requests\s*library",
    ],
    "AppiumLibrary": [
        "appium", "appiumlibrary", r"appium\s*library",
    ],
    "DatabaseLibrary": [
        "databaselibrary", r"database\s*library",
    ],
    "SSHLibrary": [
        "sshlibrary", r"ssh\s*library",
    ],
    "XML": [
        "xmllibrary", r"xml\s*library",
    ],
}


def _first_library_token_in(text: str) -> Optional[str]:
    """Return the canonical library name of the FIRST library token found in
    `text` (by character position). Returns None if no library token matches.
    """
    earliest: Optional[Tuple[str, int]] = None
    for lib, tokens in _LIBRARY_TOKENS.items():
        for token in tokens:
            for m in re.finditer(rf"\b{token}\b", text, re.IGNORECASE):
                if earliest is None or m.start() < earliest[1]:
                    earliest = (lib, m.start())
                break  # first match per token is enough
    return earliest[0] if earliest else None


def _build_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """v4: split `text` on `_SENTENCE_DELIMITERS` while preserving (start, end)
    character positions so PatternMatch entries can record the sentence_index
    they came from. Used by Step 1 + Step 2 of `detect_explicit_preference`.
    """
    spans: List[Tuple[int, int]] = []
    pos = 0
    delim_re = re.compile(_SENTENCE_DELIMITERS)
    for m in delim_re.finditer(text):
        if pos < m.start():
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans
```

#### Step 6: add `LibraryDetector.detect_explicit_preference` (v3 algorithm)

```python
from collections import defaultdict

CONFLICT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "web_automation": ("Browser", "SeleniumLibrary"),
}

# v7 (IMPLEMENTED) — sentence punctuation OR paragraph break (2+ newlines).
# Lineage:
#   v3-v5: r"[.;!?,\n]+"      — single newline as boundary; orphaned negation
#                               when target was on the next line
#   v6:    r"[.;!?,]+"        — dropped \n; broke paragraph-separated text
#                               (negation leaked across paragraphs)
#   v7:    r"[.;!?,]+|\n\s*\n+" — single newline NOT a boundary (negation
#                                  reaches across-line target); paragraph
#                                  break IS a boundary (separates ideas)
_SENTENCE_DELIMITERS = r"[.;!?,]+|\n\s*\n+"


def detect_explicit_preference(
    self,
    text: str,
    policy: Optional[DetectionPolicy] = None,
) -> PreferenceResolution:
    """Decisive library preference. v3 algorithm:
      Step 1: compute raw explicit scores + collect matches
      Step 2: sentence-scoped negation + migration (v3 phrase-based)
      Step 3: conflict check on RAW scores BEFORE threshold filter
      Step 4: threshold filter (post-conflict)
      Step 5: pick winner
    """
    policy = policy or DetectionPolicy.from_env()
    if not text:
        return PreferenceResolution(
            library=None, source="rule", evidence=[], conflicts={},
            all_scores={}, sampling_evidence=None,
        )

    # === Step 1: compute raw explicit scores + collect matches ===
    # v4: build sentence spans FIRST so each PatternMatch can record its
    # source sentence_index for D7-safe evidence filtering.
    sentence_spans = _build_sentence_spans(text)  # List[Tuple[int, int]]

    def _sentence_index_for(pos: int) -> int:
        for i, (start, end) in enumerate(sentence_spans):
            if start <= pos < end:
                return i
        return -1

    raw_scores: Dict[str, int] = defaultdict(int)
    matches_by_lib: Dict[str, List[PatternMatch]] = defaultdict(list)
    for lib, rules in self.LIBRARY_RULES.items():
        for rule in rules:
            if not rule.explicit:
                continue
            for m in rule.compiled.finditer(text):
                raw_scores[lib] += rule.weight
                if len(matches_by_lib[lib]) < 8:  # cap evidence
                    matches_by_lib[lib].append(PatternMatch(
                        library=lib,
                        pattern=rule.pattern,
                        weight=rule.weight,
                        text_span=m.group(0),
                        sentence_index=_sentence_index_for(m.start()),
                    ))

    # === Step 2: sentence-scoped negation + migration (v4) ===
    # v4: reuse sentence_spans from Step 1 so the sentence_index threaded
    # through _subtract_sentence_score matches the index recorded on each
    # PatternMatch. Skip empty/whitespace-only sentences.
    for sentence_index, (start, end) in enumerate(sentence_spans):
        sent = text[start:end]
        if not sent.strip():
            continue

        # 2a. Migration patterns (source-vs-destination)
        # v3/v4: source and destination identified via _first_library_token_in()
        # on the matched text spans, not by raw \w+ capture. Handles multi-word
        # names like "Browser Library" and hyphenated forms.
        #
        # v4 KNOWN LIMITATION: "Migrate from X to Browser library" silently
        # no-ops on the destination because _LIBRARY_TOKENS deliberately
        # excludes bare `browser`. This is an intentional consequence of the
        # bare-noun fix from v3 — `browser` alone is too generic for explicit
        # detection. Workaround: phrase as "to playwright" or "to browserlibrary".
        for pattern_src in MIGRATION_PATTERNS:
            for m in re.finditer(pattern_src, sent, re.IGNORECASE):
                src_lib = _first_library_token_in(m.group("src"))
                dst_lib = _first_library_token_in(m.group("dst"))
                if src_lib:
                    self._subtract_sentence_score(
                        raw_scores, matches_by_lib, sent, sentence_index, src_lib
                    )
                if dst_lib:
                    raw_scores[dst_lib] += 5  # destination bonus

        # 2b. Single-library negation — SINGLE ALTERNATION REGEX (v4)
        # v4 algorithm:
        #   1. Single regex `_NEGATION_REGEX` matches negation phrases
        #      (longest-first alternation guarantees each span fires once)
        #   2. For each non-overlapping match, look in the rest of the sentence
        #   3. Identify the first library token in that suffix
        #   4. Subtract that library's contribution from raw_scores
        for m in _NEGATION_REGEX.finditer(sent):
            remaining = sent[m.end():]
            target_lib = _first_library_token_in(remaining)
            if target_lib:
                self._subtract_sentence_score(
                    raw_scores, matches_by_lib, sent, sentence_index, target_lib
                )

    # === Step 3: CONFLICT CHECK ON RAW SCORES (BEFORE threshold filter) ===
    conflicts: Dict[str, List[Dict[str, Any]]] = {}
    for group_name, members in CONFLICT_GROUPS.items():
        libs_with_signal = [lib for lib in members if raw_scores.get(lib, 0) > 0]
        if len(libs_with_signal) < 2:
            continue
        ranked = sorted(libs_with_signal, key=lambda l: raw_scores[l], reverse=True)
        if raw_scores[ranked[0]] - raw_scores[ranked[1]] <= policy.ambiguity_window:
            conflicts[group_name] = [
                {
                    "library": lib,
                    "score": raw_scores[lib],
                    "patterns_matched": [pm.pattern for pm in matches_by_lib[lib]],
                }
                for lib in ranked
            ]

    if conflicts:
        return PreferenceResolution(
            library=None, source="rule", evidence=[], conflicts=conflicts,
            all_scores=dict(raw_scores), sampling_evidence=None,
        )

    # === Step 4: threshold filter (post-conflict) ===
    candidates: Dict[str, int] = {}
    for lib, score in raw_scores.items():
        threshold = (policy.conflict_min_score
                     if self._in_conflict_group(lib)
                     else policy.default_min_score)
        if score >= threshold:
            candidates[lib] = score

    if not candidates:
        return PreferenceResolution(
            library=None, source="rule", evidence=[], conflicts={},
            all_scores=dict(raw_scores), sampling_evidence=None,
        )

    # === Step 5: pick winner ===
    winner = max(candidates, key=candidates.get)
    return PreferenceResolution(
        library=winner, source="rule",
        evidence=[pm.to_dict() for pm in matches_by_lib[winner]],
        conflicts={}, all_scores=dict(raw_scores), sampling_evidence=None,
    )


def _in_conflict_group(self, library: str) -> bool:
    return any(library in members for members in CONFLICT_GROUPS.values())


def _subtract_sentence_score(
    self,
    raw_scores: Dict[str, int],
    matches_by_lib: Dict[str, List[PatternMatch]],
    sentence: str,
    sentence_index: int,
    library: str,
) -> None:
    """Subtract all explicit-pattern contributions for `library` that match
    `sentence`. Also strip evidence that came FROM the negated sentence by
    matching on `sentence_index`, not substring inclusion.

    v4 (D7 fix): v3 used `pm.text_span.lower() not in sentence.lower()`
    which could corrupt evidence from sibling sentences containing the same
    library token. v4 filters by exact sentence membership via the
    `sentence_index` field set on PatternMatch at scoring time.
    """
    deduction = 0
    for rule in self.LIBRARY_RULES.get(library, []):
        if not rule.explicit:
            continue
        if rule.compiled.search(sentence):
            deduction += rule.weight
    if deduction:
        raw_scores[library] = max(0, raw_scores[library] - deduction)
        # v4: exact-sentence filtering — only drops evidence from THIS sentence
        matches_by_lib[library] = [
            pm for pm in matches_by_lib[library]
            if pm.sentence_index != sentence_index
        ]
```

**v3 walkthrough: `"do not use Selenium, instead use Playwright"`** (the scenario v2 broke):

Step 1 — raw scores across full text:
- `\b(use|using|...)\s+selenium\b` → SL +10
- `\bselenium\b` → SL +6
- `\b(use|using|...)\s+playwright\b` → Browser +10
- `\bplaywright\b` → Browser +9
- Result: `raw_scores = {SL: 16, Browser: 19}`

Step 2 — sentences after splitting on `[.;!?,\n]+`:
- Sentence 1: `"do not use Selenium"`
- Sentence 2: `" instead use Playwright"`

Sentence 1 processing:
- 2a Migration: no migration phrase fires
- 2b Negation: `\bdo\s+not\s+use\b` matches at position 0-11
  - `remaining = " Selenium"`
  - `_first_library_token_in(" Selenium")` → `SeleniumLibrary`
  - `_subtract_sentence_score(SL, sentence="do not use Selenium")`
  - SL patterns matching this sentence: `\b(use...)\s+selenium\b` (+10) and `\bselenium\b` (+6) → deduction = 16
  - SL: 16 − 16 = 0

Sentence 2 processing:
- 2a Migration: no phrase fires (no "from X to Y" / "instead of X" / "switch from X" / "replace X with")
  - Note: "instead use" is NOT "instead OF X use Y" — migration regex correctly doesn't match.
- 2b Negation: no negation phrase fires (sentence 2 has no negation words)
- Browser score unchanged at 19

Step 3 — conflict check:
- `libs_with_signal = ["Browser"]` (SL=0 excluded)
- Only one library → no conflict

Step 4 — threshold filter:
- Browser=19 ≥ conflict_min_score=8 → candidate

Step 5 — winner: **Browser** ✓ (matches documented expected outcome)

**v3 walkthrough: `"Migrate the test suite from Selenium to Playwright"`**:

Step 1 — raw scores:
- `\bselenium\b` → SL +6
- `\bplaywright\b` → Browser +9
- Result: `raw_scores = {SL: 6, Browser: 9}`

Step 2 — single sentence (no comma/period):
- 2a Migration: `\bmigrat...from\s+(?P<src>.+?)\s+to\s+(?P<dst>.+?)(?=[\s.,;!?]|$)` matches
  - `src = "Selenium"`, `dst = "Playwright"`
  - `_first_library_token_in("Selenium")` → `SeleniumLibrary`
  - `_first_library_token_in("Playwright")` → `Browser`
  - Subtract SL contributions in sentence (deduction = 6) → SL: 0
  - Browser bonus +5 → Browser: 14
- 2b Negation: no negation phrase

Step 3 — only Browser has signal → no conflict
Step 4 — Browser=14 ≥ 8 → candidate
Step 5 — winner: **Browser** ✓

**v3 walkthrough: `"Test both selenium and playwright sites and compare"`** (the AC-5 conflict case):

Step 1 — raw scores: SL=6, Browser=9
Step 2 — no migration or negation fires
Step 3 — conflict check: both > 0, diff=3 ≤ 4 → **conflict fires** → returns None + conflicts ✓

**v3 algorithm verified empirically (2026-05-29)**: a Python simulation of the v3 algorithm (NEGATION_PHRASES + MIGRATION_PATTERNS + `_first_library_token_in` + sentence delimiter `[.;!?,\n]+`) was run against all six PRD acceptance scenarios and produced the documented outcomes:

```
"do not use Selenium, instead use Playwright"      → Browser            ✓
"Migrate the test suite from Selenium to Playwright" → Browser           ✓
"Test both selenium and playwright sites and compare" → None + conflict ✓
"Use playwright to test demoshop"                  → Browser            ✓
"Test e-commerce site: open browser, add items..." → None               ✓
"Run a Selenium test"                              → None (SL=6 < 8)    ✓
```

Initial raw scores for the first case: `{SL: 16, Browser: 19}` (the v2 problem); after v3 sentence-scoped negation the SL contributions in the first clause are subtracted: `{SL: 0, Browser: 19}` → Browser wins. This is the specific scenario v2 broke and v3 fixes.

#### Step 7: preserve `_compiled_patterns` test-surface compatibility (v3 — real contract)

**v2 was wrong.** v2 proposed `__iter__` yielding 4 values; Codex showed the real tests at `tests/integration/test_nlp_improvements.py:569-576` do:

```python
raw_browser = any(
    p.findall("migrate from selenium to browser library for modern testing")
    for p, _ in library_detector._compiled_patterns.get("Browser", [])
)
```

The test:
1. Destructures with `for p, _ in ...` — requires **exactly 2 items per entry**. `__iter__` yielding 4 would raise `ValueError: too many values to unpack`.
2. Calls `p.findall(...)` — `p` MUST be a compiled regex (`re.Pattern`), not a `PatternRule`.

v3 honours both requirements by keeping `_compiled_patterns` EXACTLY as today (`Dict[str, List[Tuple[Pattern, int]]]`) and storing the rich annotations in a parallel `_rules_metadata` attribute:

```python
# v4: LIBRARY_RULES_DEFAULT is the module-level constant carrying the v4
# pattern table from Step 2. Declared here so LibraryDetector.__init__ can
# reference it. v3 used the name without declaring it.
LIBRARY_RULES_DEFAULT: Dict[str, List[PatternRule]] = {
    "SeleniumLibrary": [...],   # the Step 2 table
    "Browser":         [...],
    "RequestsLibrary": [...],
    "AppiumLibrary":   [...],
    "DatabaseLibrary": [...],
    "SSHLibrary":      [...],
    "XML":             [...],
}


class LibraryDetector:
    """Singleton. Holds two parallel pattern stores:
      - _compiled_patterns: legacy (Pattern, int) tuples for test-fixture compat
      - _rules_metadata:    rich PatternRule entries used by the new
                             detect_explicit_preference() path
    """

    LIBRARY_RULES: Dict[str, List[PatternRule]]   # source of truth
    # v4: NEGATION_PHRASES replaced by _NEGATION_REGEX (single alternation)

    def __init__(
        self,
        rules: Optional[Dict[str, List[PatternRule]]] = None,
        policy: Optional[DetectionPolicy] = None,
    ):
        self.LIBRARY_RULES = rules if rules is not None else LIBRARY_RULES_DEFAULT
        self.policy = policy or DetectionPolicy.from_env()

        # v3: TWO parallel stores.
        # _compiled_patterns: legacy 2-tuple shape — preserves the test
        # contract verified at test_nlp_improvements.py:569-576 where the
        # fixture does `for p, _ in entries: p.findall(...)`. p MUST be a
        # compiled re.Pattern (NOT PatternRule).
        self._compiled_patterns: Dict[str, List[Tuple[Pattern, int]]] = {
            lib: [(rule.compiled, rule.weight) for rule in rules_list]
            for lib, rules_list in self.LIBRARY_RULES.items()
        }
        # _rules_metadata: rich PatternRule entries used by the new explicit-
        # preference path. Keyed by library, same ordering as _compiled_patterns
        # so the i-th tuple in _compiled_patterns corresponds to the i-th
        # PatternRule in _rules_metadata. New code uses this; existing tests
        # use _compiled_patterns.
        self._rules_metadata: Dict[str, List[PatternRule]] = {
            lib: list(rules_list)
            for lib, rules_list in self.LIBRARY_RULES.items()
        }
```

Critically, `PatternRule` does NOT need an `__iter__` — the legacy tests never see `PatternRule` objects. They only ever read `_compiled_patterns`, which is plain `(Pattern, int)` tuples. The new code paths (`detect_explicit_preference`, `_subtract_sentence_score`, etc.) read `LIBRARY_RULES` or `_rules_metadata` directly to get the `explicit` and `rationale` fields.

This is the minimal, contract-faithful approach. No backward-compat shims, no surprising iteration behaviour.

#### Step 6: keep back-compat `detect()` method

```python
def detect(self, text: str, min_score: int = None) -> Optional[str]:
    """Backwards-compatible wrapper. Returns the detected library or None.

    Prefer ``detect_explicit_preference`` for new code — it returns the
    full PreferenceResolution including evidence and conflicts.
    """
    policy = DetectionPolicy.from_env()
    if min_score is not None:
        policy = dataclasses.replace(policy, default_min_score=min_score)
    return self.detect_explicit_preference(text, policy).library
```

### 3.2 `nlp_processor.py` — wire new entry point

**Class name fix**: the actual class at `nlp_processor.py:31` is `NaturalLanguageProcessor`, NOT `NLPProcessor` (v1 used the wrong name).

Update `NaturalLanguageProcessor._detect_explicit_library_preference` and `analyze_scenario`:

```python
# v5 IMPLEMENTED (race-free — no self._last_resolution stash on the
# module-level NaturalLanguageProcessor singleton). Caller passes the
# resolution through as a local variable.
def _resolve_explicit_library_preference(
    self, scenario_text: str
):
    """Return the full PreferenceResolution (library + evidence + conflicts
    + provenance). Returns None on ImportError fallback."""
    if not scenario_text:
        return None
    try:
        from robotmcp.utils.library_detection import get_library_detector
        return get_library_detector().detect_explicit_preference(scenario_text)
    except ImportError:
        return None


def _detect_explicit_library_preference(
    self, scenario_text: str
) -> Optional[str]:
    """Back-compat shim used by existing tests; returns just the library name."""
    resolution = self._resolve_explicit_library_preference(scenario_text)
    if resolution and resolution.library:
        return resolution.library
    return self._fallback_detect_library_preference(scenario_text)
```

And in `analyze_scenario` (in the analysis-block builder), the resolution is
threaded through as a local — NOT stashed on `self`:

```python
# v5: race-free — resolution is a local variable scoped to one analyze_scenario
# call. Stashing `self._last_resolution` on the module-level singleton would
# corrupt under concurrent requests; v5 deliberately avoids that pattern.
resolution = self._resolve_explicit_library_preference(normalized_scenario)
explicit_library_preference = resolution.library if resolution else None

analysis_block = self._build_analysis_block(
    actions=actions,
    required_capabilities=required_capabilities,
    explicit_library_preference=explicit_library_preference,
    resolution=resolution,
    session_type=session_type,
)
```

Inside `_build_analysis_block`:

```python
analysis_block = {
    "action_count": len(actions),
    "complexity": self._assess_complexity(actions),
    "estimated_steps": len(actions) * 2,
    "suggested_libraries": required_capabilities,  # NOTE: from _determine_capabilities, NOT LibraryDetector
    "explicit_library_preference": explicit_library_preference,
    "preference_source": "rule",  # v2: provenance field; sampling override may overwrite to "sampling"
    "detected_session_type": session_type,
}
# Add evidence / conflicts when present (additive — empty fields omitted)
if resolution is not None:
    if resolution.evidence:
        analysis_block["explicit_library_evidence"] = list(resolution.evidence)
    if resolution.conflicts:
        # v5 (D3 fix): PreferenceResolution.conflicts entries are ALREADY
        # dicts with {library, score, patterns_matched} keys. v4 had a
        # tuple-unpacking emit `for lib, score, patterns in conflict_list`
        # which was incompatible with the dict-shaped resolution. v5 passes
        # the dict entries through unchanged.
        analysis_block["library_preference_conflicts"] = {
            group: list(entries) for group, entries in resolution.conflicts.items()
        }
```

#### 3.2.1 Sampling override coherence (v3 — both override sites + correct env var)

v2 mentioned only `server.py:1866-1880`. v3 covers BOTH override sites (Codex found the second one):

**Site A — analyze_scenario response override** (`server.py:1860-1881`):
Modifies `analysis["explicit_library_preference"]` returned to the MCP caller.

**Site B — session aggregate override** (`server.py:1961-1972`):
Modifies `session.explicit_library_preference` on the `ExecutionSession` instance.

Both are gated by `is_sampling_enabled()` which reads the **`ROBOTMCP_USE_SAMPLING`** env var at `sampling.py:23` (v2 wrongly wrote `ROBOTMCP_USE_SAMPLING_FOR_NLP`).

```python
# Site A — server.py:1860-1881 patch (v5 IMPLEMENTED)
# v5 (Codex round-4 finding): drop the `primary_library` fallback.
# `primary_library` is the LLM's "main detected library" (a recommendation,
# not a user-stated preference). Only `library_preference` semantically maps
# to `explicit_library_preference`. Mixing the two corrupted the meaning of
# the field under sampling.
if sampling_result.get("library_preference"):
    analysis["explicit_library_preference"] = sampling_result["library_preference"]
    # provenance + clear rule-based evidence/conflicts (per ADR-024 §11)
    analysis["preference_source"] = "sampling"
    analysis.pop("explicit_library_evidence", None)
    analysis.pop("library_preference_conflicts", None)
    if "rationale" in sampling_result:
        analysis["sampling_evidence"] = sampling_result["rationale"]
# else: preference_source stays "rule" from the analysis_block builder above
```

```python
# Site B — server.py:1961-1972 patch (NEW in v3)
llm_lib_pref = await sample_detect_library_preference(ctx, scenario)
if llm_lib_pref:
    session.explicit_library_preference = llm_lib_pref
    # v3: provenance marker on the session so downstream consumers (e.g.,
    # _filter_keywords_by_session_library) can tell rule-based from sampling.
    session.preference_source = "sampling"   # NEW attribute on ExecutionSession
    logger.info(f"LLM sampling detected library preference: {llm_lib_pref}")
```

Session-aggregate evidence is NOT populated under sampling (the sampling result returns a string, not a structured evidence list). Consumers that need evidence should read it from the analyze_scenario response (Site A) not the session aggregate.

### 3.3 `session_models.py` — single source of truth (v4: + `preference_source` field)

**v4 (D4 fix)**: v3 wrote `session.preference_source = "sampling"` in the sampling override block (§3.2.1 Site B) but never declared the attribute on `ExecutionSession`. v4 adds the field explicitly:

```python
# In session_models.py — ExecutionSession dataclass (existing class, add field):
@dataclass
class ExecutionSession:
    # ... existing fields ...
    explicit_library_preference: Optional[str] = None
    preference_source: Optional[str] = "rule"  # v4: NEW — "rule" | "sampling"
    # ... existing fields ...
```

Default `"rule"` because the rule-based detector is always the initial path; sampling overrides set this to `"sampling"` per §3.2.1 Site B. Downstream consumers (the 8 listed in PRD §2) can read `session.preference_source` to distinguish rule-based vs LLM-driven preferences when needed.

Update `ExecutionSession.detect_explicit_library_preference`:

```python
def detect_explicit_library_preference(self, scenario_text: str) -> Optional[str]:
    """Detect explicit library preference using LibraryDetector (single source of truth).
    Sets `self.preference_source = "rule"` (default) — sampling override path
    is handled separately in server.py:1961-1972 per §3.2.1 Site B.
    """
    if not scenario_text:
        return None
    try:
        from robotmcp.utils.library_detection import get_library_detector
        resolution = get_library_detector().detect_explicit_preference(scenario_text)
        self.preference_source = "rule"  # v4: explicit provenance
        return resolution.library
    except ImportError:
        return self._fallback_detect_library(scenario_text)
```

The `_fallback_detect_library` private method stays as the import-failure backup but is no longer the primary path.

### 3.4 New test file

`tests/unit/test_explicit_library_detection_fix.py` covers PRD AC-1 through AC-7 + negation + migration + sampling override coherence = **10 new tests** (v2 reconciled the v1 count from 25/30 down to the PRD AC-8 target). Tests are parametrised where it reduces duplication; the count is of distinct ACs covered, with parametrisation expanding to ~30 individual test invocations.

```python
import os
from unittest.mock import patch

import pytest

from robotmcp.utils.library_detection import (
    DetectionPolicy,
    LibraryDetector,
    PreferenceResolution,
    get_library_detector,
)


class TestReportedReproducer:
    """The exact scenario from the user report."""

    def test_demoshop_open_browser_no_explicit(self):
        d = get_library_detector()
        scenario = (
            "Test e-commerce website https://demoshop.makrocode.de: "
            "open browser, add items to shopping cart, verify items, "
            "complete checkout, and close browser"
        )
        r = d.detect_explicit_preference(scenario)
        assert r.library is None, (
            f"'open browser' is generic NL and must NOT trigger explicit "
            f"SL preference. Got {r.library} from {r.evidence}"
        )

    def test_demoshop_evidence_empty(self):
        d = get_library_detector()
        scenario = (
            "Test e-commerce website https://demoshop.makrocode.de: "
            "open browser, add items to shopping cart, verify items"
        )
        r = d.detect_explicit_preference(scenario)
        assert r.evidence == []


class TestGenericNLPhrasesDoNotTrigger:
    """Pure NL phrases that overlap with keyword names must NOT
    fire explicit preference."""

    @pytest.mark.parametrize("scenario", [
        "click element by id submit",
        "Page should contain Welcome text",
        "Input text into the username field",
        "Open new page in browser context",
        "Test the checkout flow on an SPA",
        "Tap the menu button on the home screen",
    ])
    def test_generic_nl_no_explicit(self, scenario):
        d = get_library_detector()
        r = d.detect_explicit_preference(scenario)
        assert r.library is None, (
            f"NL phrase {scenario!r} should NOT trigger explicit detection; "
            f"got {r.library}"
        )


class TestTrulyExplicitDetected:
    @pytest.mark.parametrize("scenario,expected", [
        ("Use playwright to test the checkout flow", "Browser"),
        ("Use Selenium to test the login form", "SeleniumLibrary"),
        ("with SeleniumLibrary, open the page", "SeleniumLibrary"),
        ("Test using browserlibrary against demoshop", "Browser"),
        ("Use requestslibrary to call /api/users", "RequestsLibrary"),
        ("With AppiumLibrary, open the mobile app", "AppiumLibrary"),
        ("Use database library to validate the inventory table", "DatabaseLibrary"),
        ("Use ssh library to deploy", "SSHLibrary"),
        ("Use XML library to validate the response", "XML"),
        ("Run Selenium 4 against the site", "SeleniumLibrary"),
        ("chromedriver and the chromedriver path", "SeleniumLibrary"),
        ("Test using chromium and webkit", "Browser"),
    ])
    def test_explicit_detected(self, scenario, expected):
        d = get_library_detector()
        r = d.detect_explicit_preference(scenario)
        assert r.library == expected, (
            f"{scenario!r}: expected {expected}, got {r.library}"
        )

    def test_evidence_populated_when_detected(self):
        d = get_library_detector()
        r = d.detect_explicit_preference("Use playwright to test")
        assert r.evidence, "evidence must be populated when library detected"
        assert any("playwright" in e["pattern"] for e in r.evidence)


class TestConflicts:
    """Within-conflict-group ambiguity returns None + surfaces conflict.
    v2: tests verify the RAW-SCORE-BEFORE-THRESHOLD ordering (without it
    the SL=6 / Browser=9 pair would be filtered before the conflict check)."""

    def test_both_selenium_and_playwright_returns_none_with_conflict(self):
        d = get_library_detector()
        # Selenium raw = 6 (\bselenium\b), Browser raw = 9 (\bplaywright\b).
        # Both > 0, |9 - 6| = 3 <= ambiguity_window (4) → conflict.
        # v1 algorithm: SL=6 filtered before conflict check (threshold 8) →
        # Browser sole survivor → no conflict → Browser wrongly returned.
        # v2 algorithm: conflict check on RAW scores → None + conflicts.
        r = d.detect_explicit_preference(
            "Test both selenium and playwright sites and compare"
        )
        assert r.library is None, (
            f"Conflict-on-raw-scores must detect this case; got {r.library}"
        )
        assert "web_automation" in r.conflicts
        # v4 (D3 fix): conflicts entries are dicts, not tuples — extract via key
        libs = {entry["library"] for entry in r.conflicts["web_automation"]}
        assert libs == {"Browser", "SeleniumLibrary"}

    def test_migration_resolves_to_destination(self):
        """v2: PRD AC-4 — migration patterns subtract source, add to destination."""
        d = get_library_detector()
        r = d.detect_explicit_preference(
            "Migrate the test suite from Selenium to Playwright"
        )
        assert r.library == "Browser", (
            "'migrate ... from Selenium to Playwright' should resolve to "
            f"Browser; got {r.library}"
        )
        # No conflict — Selenium was negated; only Browser above threshold
        assert "web_automation" not in r.conflicts

    def test_negation_across_sentences(self):
        """v2: sentence-scoped negation should NOT zero both libs in
        'do not use Selenium, instead use Playwright'."""
        d = get_library_detector()
        r = d.detect_explicit_preference(
            "do not use Selenium, instead use Playwright"
        )
        # Sentence 1: negation hits Selenium. Sentence 2: positive Playwright.
        # Result: Browser wins.
        assert r.library == "Browser", (
            f"Sentence-scoped negation should leave Browser intact; got {r.library}"
        )


class TestThresholdsTunable:
    def test_default_min_score_5_for_non_conflict(self):
        d = get_library_detector()
        # A library outside the web_automation conflict group with a
        # single weight-6 hit should still trigger (XML doesn't conflict
        # with any other library in current setup)
        r = d.detect_explicit_preference("xml parsing of the response")
        assert r.library == "XML"

    def test_conflict_threshold_8_blocks_single_weight_6(self):
        d = get_library_detector()
        # Just "selenium" alone is weight 6 → in conflict group → blocked
        r = d.detect_explicit_preference("Run a Selenium test")
        # Weight 6 < conflict_min_score 8 → None
        assert r.library is None

    def test_conflict_threshold_8_passes_weight_9(self):
        d = get_library_detector()
        # "playwright" is weight 9 → passes conflict threshold
        r = d.detect_explicit_preference("Run a playwright test")
        assert r.library == "Browser"

    def test_env_var_lowers_threshold(self, monkeypatch):
        monkeypatch.setenv("ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD", "5")
        # Need a fresh policy / detector since policy is read at call time
        d = LibraryDetector()
        r = d.detect_explicit_preference("Run a Selenium test")
        assert r.library == "SeleniumLibrary"


class TestBackwardsCompatibility:
    """Existing detect() / detect_all() entry points keep working."""

    def test_detect_returns_string_or_none(self):
        d = get_library_detector()
        assert d.detect("use playwright") == "Browser"
        assert d.detect("xyz") is None

    def test_get_scores_unchanged_for_mention_layer(self):
        d = get_library_detector()
        scores = d.get_scores("open browser")
        # Mention layer keeps the 'open browser' pattern at weight 6
        # for SeleniumLibrary (used for suggested_libraries advisory)
        assert scores.get("SeleniumLibrary", 0) >= 6


class TestEvidenceShape:
    def test_evidence_includes_pattern_and_text_span(self):
        d = get_library_detector()
        r = d.detect_explicit_preference("Use playwright for the demoshop test")
        assert r.evidence
        first = r.evidence[0]
        assert "pattern" in first
        assert "text_span" in first
        assert "weight" in first
        assert "playwright" in first["text_span"].lower()


class TestAnalyzeScenarioResponse:
    """End-to-end via the MCP wrapper — fields surface correctly.
    v2: class name fix — NaturalLanguageProcessor (not NLPProcessor)."""

    @pytest.mark.asyncio
    async def test_response_omits_evidence_when_no_detection(self):
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor
        nlp = NaturalLanguageProcessor()
        result = await nlp.analyze_scenario(
            "Test e-commerce website https://demoshop.makrocode.de: "
            "open browser, add items, complete checkout",
            context="web",
        )
        analysis = result["analysis"]
        assert analysis["explicit_library_preference"] is None
        assert analysis["preference_source"] == "rule"  # v2 field
        # Evidence/conflicts ABSENT (not present-but-empty)
        assert "explicit_library_evidence" not in analysis
        assert "library_preference_conflicts" not in analysis

    @pytest.mark.asyncio
    async def test_response_includes_evidence_on_explicit_detection(self):
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor
        nlp = NaturalLanguageProcessor()
        result = await nlp.analyze_scenario(
            "Use playwright to test the demoshop checkout flow",
            context="web",
        )
        analysis = result["analysis"]
        assert analysis["explicit_library_preference"] == "Browser"
        assert analysis["preference_source"] == "rule"  # v2 field
        assert "explicit_library_evidence" in analysis
        assert any("playwright" in e["text_span"].lower()
                   for e in analysis["explicit_library_evidence"])


class TestSamplingOverrideCoherence:
    """v3: when sampling overrides preference, evidence/conflicts are cleared
    and preference_source is 'sampling'. Uses correct env var ROBOTMCP_USE_SAMPLING."""

    @pytest.mark.asyncio
    async def test_sampling_clears_rule_evidence(self, monkeypatch):
        # Mock the sampling path to return a library decision
        async def fake_sampling(*args, **kwargs):
            return {
                "library_preference": "SeleniumLibrary",
                "rationale": "Detected via LLM",
            }
        # v3: patch at the real import path (robotmcp.utils.sampling)
        monkeypatch.setattr(
            "robotmcp.utils.sampling.sample_analyze_scenario", fake_sampling
        )
        # v3: correct env var name (sampling.py:23 reads ROBOTMCP_USE_SAMPLING)
        monkeypatch.setenv("ROBOTMCP_USE_SAMPLING", "1")
        # Invoke analyze_scenario via the MCP server wrapper
        from robotmcp.server import analyze_scenario
        result = await analyze_scenario(
            "Use playwright to test demoshop",
            context="web",
        )
        analysis = result["analysis"]
        assert analysis["explicit_library_preference"] == "SeleniumLibrary"
        assert analysis["preference_source"] == "sampling"
        assert "explicit_library_evidence" not in analysis
        assert "library_preference_conflicts" not in analysis
        assert analysis.get("sampling_evidence") == "Detected via LLM"
```

#### Existing tests requiring updates (PRD §AC-8)

The following existing tests assert false-positive behaviour and need updating to match v2:

| Test file:line (estimate) | What it asserts (old) | New expectation |
|---|---|---|
| `tests/unit/test_library_detection.py` | "open browser" → SeleniumLibrary | "open browser" alone → None |
| `tests/unit/test_library_detection.py` | "click element" → SeleniumLibrary | "click element" alone → None |
| `tests/unit/test_session_models_library_preference.py` | session auto-configures SL on "open browser" | session does NOT auto-configure (None preference) |
| `tests/unit/test_nlp_processor.py` (if such assertion exists) | "input text" triggers explicit SL | "input text" alone → None |
| `tests/integration/test_nlp_improvements.py:103-104,571-576` | inspects `_compiled_patterns` directly | NO UPDATE NEEDED — `_compiled_patterns` preserved per Step 7 |

Estimated: 3-5 existing test assertions updated (exact count after grep against v2 implementation). All updates align with the v1→v2 semantic change ("explicit" now means "user said so").

---

## 4. Migration Plan

### Phase 1 — Internal refactor (this proposal)

1. Add `PatternRule` dataclass with `explicit` annotation.
2. Convert `LIBRARY_PATTERNS` table entries one by one — keep behaviour byte-for-byte by setting `explicit=True` for every existing pattern initially.
3. Add `detect_explicit_preference` method.
4. Verify all existing tests pass (no behaviour change yet).
5. Then re-annotate the keyword-name patterns to `explicit=False` per the ADR-024 §6 table.
6. Run existing tests + new test file. Expected: new tests pass; some existing tests that asserted "click element triggers SL" need updating to assert mention-only behaviour.

### Phase 2 — Surface evidence/conflicts in response

7. Update `nlp_processor.analyze_scenario` to add `explicit_library_evidence` + `library_preference_conflicts` fields.
8. Update `session_models.detect_explicit_library_preference` to call the new method.

### Phase 3 — Documentation

9. Update `analyze_scenario` MCP tool docstring to mention the new optional fields.
10. Release notes entry.

### Phase 4 — Cleanup (separate follow-up PR)

11. Remove `_fallback_detect_library_preference` after one release if no import-failure regression is reported.
12. Migrate any remaining tuple-form patterns to the dataclass form.

---

## 5. Test Matrix (v3 — honest count)

v2 claimed "10 distinct test functions" but the matrix summed to 18. v3 honestly counts both.

| Test class | PRD AC coverage | Distinct test functions | Parametrised invocations |
|---|---|---|---|
| `TestReportedReproducer` | AC-1 | 2 | 2 |
| `TestGenericNLPhrasesDoNotTrigger` | AC-6 (9 phrases) | 1 | 9 |
| `TestTrulyExplicitDetected` | AC-2, AC-3, AC-7 | 2 (one parametrised + evidence check) | 13 |
| `TestConflicts` | AC-4, AC-5 + sentence-scoped negation | 3 | 3 |
| `TestThresholdsTunable` | FR-2, FR-3 + env-var contract | 4 | 4 |
| `TestBackwardsCompatibility` | NFR-1, NFR-5 (`_compiled_patterns`) | 2 | 2 |
| `TestEvidenceShape` | FR-5 | 1 | 1 |
| `TestAnalyzeScenarioResponse` | end-to-end AC-1, AC-2 | 2 | 2 |
| `TestSamplingOverrideCoherence` | FR-7 (v3) | 1 | 1 |

**Test counts (v3 honest)**:
- **18 distinct test functions** across 9 classes.
- **37 invocations** after parametrisation expansion.
- **3-5 existing test assertions updated** (see §3.4 table).

PRD AC-8 v3 (updated) refers to "approximately 18 new tests" matching this matrix.

---

## 6. Performance Impact

The detector runs once per `analyze_scenario` call. v2 dropped v1's unsupported "≤60µs per call" claim. The requirement is:

- **No measurable regression** vs the current baseline in `tests/benchmarks/test_library_detection_bench.py`.
- The detector keeps the existing per-call complexity envelope. Added work per call:
  - Conditional check on `rule.explicit` (constant per pattern).
  - Sentence splitting via single `re.split` per call.
  - Sentence-scoped negation + migration scan: O(S × (N_negation + N_migration × M_patterns)) where S = sentences, N_* = pattern counts (small constants), M_patterns = library pattern count. Bounded; sentences typically ≤ 5 for scenarios.
- Negation/migration replace the v1 single forward-window sweep — comparable cost, more correct semantics.

Validation: run `pytest tests/benchmarks/test_library_detection_bench.py` before and after; the benchmark assertion windows are not changed.

---

## 7. Risks + Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Some users RELIED on the false-positive auto-config for Selenium | Medium | Document the behaviour change in release notes; provide migration steps (use `library_name="SeleniumLibrary"` on tool calls). |
| Conflict threshold of 8 is too strict | Low-Medium | `ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD` env var lets operators tune down to 5 or less. |
| Some tests assume the old behaviour and would break | High (necessary breakage) | Update those tests — they were pinning the bug. Document each updated test in the PR commit message. |
| `PatternRule` adoption breaks downstream consumers reading the table | Low | The table format isn't public API. Internal-only consumption via `get_library_detector()`. |

---

## 8. Out of Scope

- LLM-based detection — exists as opt-in (`sample_analyze_scenario`); unchanged.
- Auto-import policy in `ExecutionSession.configure_from_scenario` — fixed indirectly; no code change to the importer.
- Field rename of `explicit_library_preference` — the name is correct after the fix.
- `_determine_capabilities` tightening — its broader inclusion is acceptable.

---

## 9. Validation Plan (v4 — reconciled to §5 matrix)

1. **Reproducer**: re-run the user's exact `analyze_scenario` call. Expected: `explicit_library_preference: None`.
2. **Test suite**: full unit suite (`uv run pytest tests/unit/`) — expect **18 new distinct test functions** (~37 invocations after parametrisation) + **3-5 modified tests** asserting false-positive behaviour. Pass count up by ~15-18 net. (v3 wording "~30 new + ~3 modified" was stale; v4 matches the §5 honest matrix count.)
3. **Synthetic corpus**: run on 20 scenarios (10 truly explicit, 10 pure NL). Compute precision/recall. Expected: precision on NL → None ≥ 95%; recall on explicit ≥ 100%.
4. **Manual smoke**: try 5 user-style scenarios live via the MCP server. Spot-check behaviour.
5. **v4 algorithm verification**: confirm the empirical Python verification (proposal §3.1 Step 6 walkthroughs) passes against the implemented code — run the 8 marquee + edge-case scenarios as integration tests.

---

## 10. Effort Estimate

- Code: half-day (~3-4 hours of focused work).
- Tests: half-day.
- Documentation update + release notes: 1 hour.
- Review cycles (Codex/Claude subagent): 2-3 hours including iteration.
- **Total**: ~1.5 engineering days for a single-engineer single-PR delivery.

---

## 11. Related Documents

- PRD: `docs/prd/analyze_scenario_explicit_library_prd.md`
- ADR-024: `docs/adr/ADR-024-explicit-library-detection-confidence.md`
- DDD: `docs/ddd/library_preference_bounded_context.md`
- Implementation will land in a single PR named: `fix(nlp): explicit library preference — preference vs mention split (ADR-024)`

---

## 12. Downstream Consumer Compatibility (NEW in v2)

PRD §2 enumerates 8 places that read `explicit_library_preference`. v1 documented only 3. v2 verifies each consumer continues to work with the v2 response shape:

| Consumer (file:lines) | Reads | v2 status |
|---|---|---|
| `server.py:2037` `_filter_keywords_by_session_library` | `analysis.explicit_library_preference` (str or None) | **Compatible** — same key, same type; v2 returns None in more cases (intended) |
| `session_models.py:787` `ExecutionSession.configure_from_scenario` | Same field | **Compatible** — receives None for noise scenarios; skips auto-import (correct behaviour) |
| `library_recommender.py:111-166` | Same field; uses as primary recommendation hint | **Compatible** — when None, falls back to capability-based recommendations |
| `library_recommender.py:310-321` | Same field; excludes the "other library" | **Compatible** — when None, no exclusion (intended) |
| `adapter_factory.py:131-140` | Same field; selects adapter (Browser vs SL) | **Compatible** — when None, default adapter selection |
| `keyword_executor.py:1901-1914` | Hardcodes `Open Browser` keyword check: when `pref.startswith("selenium")` → SL; when `pref.startswith("browser")` → Browser; when neither → falls through to `_get_library_for_keyword(keyword)`. v3 verified: this is the ONLY explicit-preference branch in keyword_executor — not the general "library_name= fallback" v2 wrongly described. | **Compatible** — when None, neither branch fires, normal library resolution proceeds. |
| `browser_plugin.py:314-321,379-390` | Same field; suppresses Browser auto-init when SL is preferred | **Compatible** — when None, Browser plugin behaves normally |
| `selenium_plugin.py:202-208` | Same field; eagerly initializes SL | **Compatible** — when None, no eager init |

**New optional response fields** (`explicit_library_evidence`, `library_preference_conflicts`, `preference_source`, `sampling_evidence`) are ignored by all 8 consumers — they read only `explicit_library_preference`. No consumer needs updating.

**Behavioural change**: scenarios that previously triggered false-positive auto-config now produce None. Users who depended on the false positive will need to:
- Pass `library_name="SeleniumLibrary"` (or whichever) explicitly on subsequent tool calls, OR
- Rephrase the scenario to use a preference verb (`"Use Selenium to test..."`).

This is documented in PRD §10 (Risks) and the implementation PR will include a release-notes entry under `docs/RELEASE_NOTES.md`.

---

## 15. Round-3 review findings + resolutions (v4)

| # | Round-3 finding (v3) | Resolution in v4 |
|---|---|---|
| D1 | NEGATION_PHRASES iterated as Python list — both `\bdo\s+not\s+use\b` and `\bdo\s+not\b` fired on same span, double-deducting | §3.1 Step 5: replaced list iteration with single `_NEGATION_REGEX` alternation, longest-first. Regex engine guarantees one match per span. **Verified empirically against 9 scenarios including D1 edge cases.** |
| D2 | INV-4 was trivial `max(0, x-d)` idempotence — wouldn't catch D1 | DDD §7 INV-4 replaced with deduction-sum equality (catches D1) |
| D3 | DDD §4.1.6 conflicts type was tuple-of-three; PRD/ADR/proposal used dict-of-dicts | DDD §4.1.6 updated to `Dict[str, List[Dict[str, Any]]]`; proposal TestConflicts updated to unpack `entry["library"]` not `entry[0]` |
| D4 | `session.preference_source` written but never declared on `ExecutionSession` | §3.3: field added explicitly as `preference_source: Optional[str] = "rule"` on the dataclass |
| D5 | Test count §9 said "~30 + ~3" but §5 said "18 distinct" | §9 updated to match §5: "18 new + 3-5 modified" |
| D6 | Multi-word migration `"to Browser library"` silently no-ops | §3.1 Step 5: documented as KNOWN LIMITATION (intentional consequence of bare-noun fix) |
| D7 | `_subtract_sentence_score` used fragile substring check `text_span.lower() not in sentence.lower()` | §3.1 Step 1: `PatternMatch.sentence_index: int = -1` field added; `_subtract_sentence_score` filters by `pm.sentence_index != sentence_index` (exact membership, not substring) |
| D8 | `Literal["rule", "sampling"]` declared but not imported | §3.1 Step 1: `from typing import ..., Literal, ...` added to imports |
| F5 | `LIBRARY_RULES_DEFAULT` referenced in Step 7 but never declared | §3.1 Step 7: explicit module-level declaration added before `LibraryDetector` class |

---

## 14. Round-2 review findings + resolutions (v3)

| # | Codex finding (v2 round 2) | Resolution in v3 |
|---|---|---|
| B1 | v2 negation regex captured `target="use"` not the library; "do not use Selenium, instead use Playwright" left both libs intact | §3.1 Step 5: phrase-list `NEGATION_PHRASES` + `_first_library_token_in()` two-step approach; sentence delimiter now `[.;!?,\n]+` (includes comma) |
| B2 | v2's `PatternRule.__iter__` yielding 4 values breaks `for p, _ in _compiled_patterns; p.findall(...)` at `test_nlp_improvements.py:571-576` | §3.1 Step 7: dual stores — `_compiled_patterns` keeps `(Pattern, int)` tuples; new code reads `_rules_metadata` (parallel `PatternRule` list) |
| B3 | `use browser`, `use database`, `use ssh`, `use requests` still flagged `explicit=True` (generic English nouns) | §3.1 Step 2: bare-noun tokens dropped from preference-verb patterns; only library-specific tokens (`playwright`, `seleniumlibrary`, `appium`, etc.) remain |
| B4 | DDD prose says mention layer diagnostic-only but §5 diagram still shows `MentionScorer` → capability list | DDD §5 diagram fixed (separate doc edit) |
| C1 | `PreferenceResolution` declared with 4 fields but constructed with 5 + named `source="rule"` | §3.1 Step 4: canonical 6-field shape `(library, source, evidence, conflicts, all_scores, sampling_evidence)` |
| C2 | `rule.compiled.finditer()` used but `compiled` never declared on `PatternRule` | §3.1 Step 1: `compiled: Pattern = field(init=False)` set in `__post_init__` |
| C3 | `_TOKEN_TO_LIBRARY` incomplete; migration regex `\w+` breaks hyphens & multi-word | §3.1 Step 5: `_LIBRARY_TOKENS` map + `_first_library_token_in()`; migration regex uses `.+?` capture with `_first_library_token_in()` resolution |
| C5 | Sampling test patches wrong target and uses non-existent `ROBOTMCP_USE_SAMPLING_FOR_NLP` env var | §3.4: patches `robotmcp.utils.sampling.sample_analyze_scenario`; uses `ROBOTMCP_USE_SAMPLING` (verified at `sampling.py:23`) |
| C6 | Scope table says 3 files but §3.2.1 patches `server.py` | §3 scope table: 4 files |
| C7 | Test matrix totals 18 but text says 10 distinct | §5: honestly 18 distinct functions, 37 invocations |
| E1 | ADR §3.3 claims `CONFLICT_GROUPS` module-level at `library_detection.py:240-242` (actually local in `get_conflicting_detections()`) | ADR §3.3 corrected (separate doc edit) |
| E3 | Sampling override has TWO sites (`server.py:1860-1881` AND `1961-1972`); v2 missed second | §3.2.1: both sites documented; Site B patches `session.explicit_library_preference` + adds `session.preference_source` |
| E4 | `keyword_executor.py:1901-1914` actually checks `Open Browser` keyword + `pref.startswith("selenium" \| "browser")` — NOT library_name= fallback | §12: corrected description |

---

## 13. Round-1 review findings + resolutions (v2)

| # | Codex finding | Resolution in v2 |
|---|---|---|
| 1 | Pattern table wrongly annotated `explicit=True` for domain markers | §3.1: `rest api testing`, `mobile automation`, `android/ios testing`, SQL fragments, DB engine names, XPath/XSD/XSLT, bare `xml` all moved to `explicit=False` |
| 2 | 80-char forward window for negation breaks `"do not use Selenium, instead use Playwright"` | §3.1 Step 5: replaced with sentence-scoped `NEGATION_PATTERNS` + `MIGRATION_PATTERNS` |
| 3 | Conflict-vs-threshold ordering makes AC-5 impossible | §3.1 Step 6: conflict check now runs on RAW scores BEFORE threshold filter |
| 4 | Class name `NLPProcessor` wrong | §3.2, §3.4: corrected to `NaturalLanguageProcessor` |
| 5 | Test count inconsistent (25 vs 30) | §3.4 + §5: reconciled to 10 distinct tests, ~37 parametrised invocations |
| 6 | AC-5 test logic was unreachable under v1 algorithm | §3.4 `TestConflicts`: rewrote to assert v2 algorithm |
| 7 | Existing-test updates not listed | §3.4: 3-5 tests enumerated with file:line locations |
| 8 | `_compiled_patterns` test surface compat not handled | §3.1 Step 7: dedicated section + `__iter__` for tuple-unpacking compat |
| 9 | Performance claim "≤60µs" unsupported | §6: replaced with "no measurable regression vs benchmark" |
| 10 | Sampling override coherence undefined | §3.2.1: clear evidence + conflicts, set `preference_source: "sampling"` |
| 11 | 5 of 8 downstream consumers not documented | §12 NEW: full compatibility table for all 8 consumers |
| 12 | Migration patterns absent from v1 | §3.1 Step 5: `MIGRATION_PATTERNS` constant + sentence-scoped application |
