# DDD — Library Preference Bounded Context

**Date**: 2026-05-29
**Status**: **IMPLEMENTED v5** (source in `src/robotmcp/utils/library_detection.py`)
**Scope**: scenario analysis → library identification
**Related**: PRD `docs/prd/analyze_scenario_explicit_library_prd.md`, ADR-024 `docs/adr/ADR-024-explicit-library-detection-confidence.md`, solution proposal `docs/proposals/explicit_library_detection_fix_proposal.md`

---

## Revision history

**v6 (2026-06-05)** — Codex round-5 review confirmed DDD v5 TIGHTEN; v6 adds notes on the 4 critical source bugs that v6 fixed (architecture is unchanged, implementation was tightened):
- **D-session-fallback note added to §5 Boundaries**: `ExecutionSession.detect_explicit_library_preference` is a DELEGATE to `LibraryDetector.detect_explicit_preference` — it MUST NOT fall back to broad mention-layer heuristics when v6 detector deliberately returns None (that would undo the bounded context's conservative preference contract).
- **Implementation status confirmed coherent end-to-end**: analysis-path AND session-aggregate paths both return None for the reported scenario. 18 new v6 unit tests cover the previously-orphaned cases.

**v5 (2026-06-05)** — implementation landed; Codex round-4 review confirmed propagation gaps in v4 ADR/proposal that are closed in v5. Key DDD updates:
- **§2 overview shape strings updated**: `List[PatternMatch]` → `List[Dict[str, Any]]` with the canonical `{library, pattern, weight, text_span}` entry shape. `List[(lib, score, patterns)]` → `Dict[str, List[Dict[str, Any]]]`. Now matches `PreferenceResolution` in the actual implementation.
- **§4.1.6 `PreferenceResolution.conflicts` type confirmed dict-of-dicts**: same shape used by PRD/ADR/proposal AND by `src/robotmcp/utils/library_detection.py:PreferenceResolution`. v4 already had this; v5 propagation completed.
- **Race-free invocation noted**: `NaturalLanguageProcessor._resolve_explicit_library_preference()` returns the resolution as a local — no `self._last_resolution` stash. Tested via `TestNoSharedState::test_interleaved_analyses_no_state_bleed`. The shared `_detector` module-level singleton remains, but it's only read after construction so there's no mutation race.

**v4 (2026-05-29)** — Third-round independent review (verdict: TIGHTEN for DDD). v4 addresses:
- **§4.1.6 `PreferenceResolution.conflicts` type fixed**: v3 declared `Dict[str, List[Tuple[str, int, List[str]]]]` (tuple) but PRD/ADR/proposal use dict-of-dicts shape `[{library, score, patterns_matched}]`. v3 DDD was the outlier and the proposal's `TestConflicts` was broken because it unpacked `entry[0]` (tuple style) while the algorithm returned dicts. v4 aligns DDD to the dict shape; proposal test fixed in parallel.
- **§4.1.6 `evidence` type fixed**: `List[PatternMatch]` → `List[Dict[str, Any]]` to reflect the actual to_dict-flattened shape that surfaces in the JSON response.
- **§7 INV-4 strengthened**: v3's negation-idempotence was trivially true (`max(0, x-d)` always idempotent). v4 replaces with **deduction-sum equality** — sum of deductions across all phrase matches equals single-canonical-phrase deduction. This DOES catch the round-3 D1 double-deduction bug. v3's INV-4 wouldn't have.

**v3 (2026-05-29)** — Codex CLI second-round critical review (verdict: TIGHTEN). v3 addresses:
- **§5 boundary diagram fixed**: v2 prose claimed mention layer is diagnostic-only but the diagram still showed `MentionScorer` "Used for: capability list (suggested_libs)". v3 redraws the diagram to match the prose — mention layer marked "diagnostic only, no current production consumer".
- **INV-4 replaced (was vacuous)**: v2's "When `score(text)[lib] > 0` for some library and `lib` has only `explicit=False` patterns..." was vacuous because every library in the v3 pattern table has at least one `explicit=True` pattern. v3 replaces it with a verifiable invariant about sentence-scoped negation idempotence.
- **§6.2 `_compiled_patterns` description corrected**: v2 implied the attribute structure could change. v3 documents the **two-store design** where `_compiled_patterns` keeps its exact current shape `Dict[str, List[Tuple[Pattern, int]]]` and rich annotations live in a parallel `_rules_metadata` store.
- **§4.1.2 `PatternMatch.to_dict()` field order standardised**: matches the canonical evidence shape `{library, pattern, weight, text_span}` used in PRD §FR-5, ADR §3.5, proposal Step 4.
- **§11 round-1 findings table preserved** + new round-2 table added at §12.

**v2 (2026-05-29)** — Codex CLI critical review identified over-modeling and incorrect domain claims. Key changes:
- **Removed over-modeled value objects**: `DetectionScore`, `ConflictGroup`, and `MentionScorer` are no longer separate value objects. Codex correctly identified them as "window dressing" — `DetectionScore` is `(library_name, int_score, [PatternMatch])` which can be a tuple/dict; `ConflictGroup` is a literal `{name: tuple_of_libs}` map already declared as `CONFLICT_GROUPS`; `MentionScorer` is `LibraryDetector.get_scores` renamed. v2 keeps just `PatternRule`, `DetectionPolicy`, `PatternMatch`, and `PreferenceResolution` as the actual value objects.
- **"Stateless" claim retracted (§4.2)**: `LibraryDetector` is NOT stateless — it holds a compiled-pattern cache and is consumed as a process-global singleton from `nlp_processor` and `session_models`. v1's claim was wrong. v2 documents the reality.
- **INV-3 (purity) reframed**: `PreferenceResolver.resolve(text)` is pure *within a given DetectionPolicy snapshot*. Because the policy reads env vars at construction, two resolvers built with different env states differ; v1 implied the resolver was globally pure.
- **INV-4 (mention ≥ preference) dropped**: v1 claimed mention scores ≥ preference scores as an invariant. After v2's architecture correction (capability suggestion does NOT consume mention scores in the current codebase), this invariant is unverifiable in production. Kept as an algorithmic property of `score()` but not a context invariant.
- **NEGATION_PATTERNS + MIGRATION_PATTERNS added to §5 Boundaries**: v1 silently owned negation but never listed it. v2 explicitly names the patterns this context owns vs delegates.
- **Architecture correction**: §2 + §5 corrected to reflect that `suggested_libraries` is computed by `_determine_capabilities` (separate substring heuristic) NOT by this context's mention layer. Mention layer is preserved for diagnostics/future use only.
- **`_compiled_patterns` test-surface compatibility added to §6.2**: existing tests inspect this internal attribute; v2 implementation preserves it.

**v1 (2026-05-29)** — initial draft.

---

## 1. Why a bounded context?

The current code conflates two related-but-distinct domain concerns into one detector:

1. **Library mention** — "does the scenario text discuss things this library could automate?" (advisory, broad)
2. **Library preference** — "did the user choose this library?" (decisive, narrow)

Both reasonably need pattern matching against scenario text, but they have different value semantics, different consumers, and different acceptable false-positive rates. Treating them as one rule set has produced the reported defect (`open browser` triggering explicit SL preference).

This document defines the bounded context that separates the two, names the value objects + aggregates + services, and pins their interactions. The implementation lands per ADR-024 §3 and the solution proposal.

---

## 2. Bounded Context Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Library Preference Bounded Context                                 │
│                                                                     │
│  Inputs:                                                            │
│    - scenario text (string)                                         │
│    - DetectionPolicy (env-resolved at construction time)            │
│                                                                     │
│  Outputs (via PreferenceResolution):                                │
│    - explicit_library_preference: Optional[str]                     │
│    - explicit_library_evidence: List[Dict[str, Any]]                │
│        each: {library, pattern, weight, text_span}                  │
│    - library_preference_conflicts: Dict[str, List[Dict[str, Any]]]  │
│        each: {library, score, patterns_matched}                     │
│    - preference_source: Literal["rule", "sampling"]                 │
│                                                                     │
│  Internal model (v2 — minimised after Codex review):                │
│    - PatternRule (frozen value object — pattern, weight, explicit)  │
│    - PatternMatch (frozen value object — pattern, weight, span)     │
│    - DetectionPolicy (frozen value object — env-resolved thresholds)│
│    - PreferenceResolution (frozen value object — public output)     │
│    - LibraryDetector (stateful service — holds compiled-pattern     │
│        cache as a process-global singleton)                         │
│    - PreferenceResolver (stateless service — delegates to detector) │
│                                                                     │
│  Consumers (downstream — the 8 from PRD §2):                        │
│    - session_models.ExecutionSession.configure_from_scenario        │
│    - server._filter_keywords_by_session_library                     │
│    - components.library_recommender (lines 111-166, 310-321)        │
│    - adapters.adapter_factory (lines 131-140)                       │
│    - execution.keyword_executor (lines 1901-1914)                   │
│    - plugins.browser_plugin (lines 314-321, 379-390)                │
│    - plugins.selenium_plugin (lines 202-208)                        │
└─────────────────────────────────────────────────────────────────────┘
```

The context is intentionally narrow: it does NOT decide library *capabilities* (`_determine_capabilities` at `nlp_processor.py:517-544` is a SEPARATE substring heuristic that does not call this context), it does NOT auto-import libraries (that stays in `ExecutionSession`), it does NOT consume MCP tool params. Its single responsibility is "given scenario text, what's the user's explicit library intent?"

**Mention layer status (v2 correction)**: `LibraryDetector.get_scores(text)` (the mention API) is preserved for diagnostics and potential future consumption by capability suggestion. **It is not consumed by any production code path today** — v1 implied it backed `suggested_libraries`; that was wrong. Keeping it documented here makes future capability-suggestion refactors easier without committing to a near-term consumer.

---

## 3. Ubiquitous Language

| Term | Definition |
|---|---|
| **Library** | A Robot Framework library identified by its canonical name (`Browser`, `SeleniumLibrary`, `RequestsLibrary`, `AppiumLibrary`, `DatabaseLibrary`, `SSHLibrary`, `XML`). |
| **Mention** | Any scenario-text occurrence that COULD identify a library, including keyword names and generic action verbs. Scores accumulate; allowed to be inclusive. |
| **Preference** | A decisive identification: the user has unambiguously chosen this library. Requires verbatim library identifiers OR preference verbs. |
| **Evidence** | The pattern (and matched text span) that contributed to a preference decision. Used for audit trails. |
| **Conflict group** | A set of libraries that compete for the same job (e.g., `{Browser, SeleniumLibrary}` for web automation). When multiple group members are detected, the user has likely been ambiguous. |
| **Ambiguity window** | The score difference threshold below which two scores in the same conflict group are considered tied. Default 4. |
| **Pattern weight** | An integer 1-10 reflecting how strongly a pattern signals a library. 10 = verbatim preference verb; 9 = verbatim library name; 7-8 = library-specific concept; 5-6 = weaker hint. |
| **Explicit pattern** | A pattern that contributes to preference scoring. Annotated `explicit: True`. |
| **Mention pattern** | A pattern that contributes only to mention scoring. Annotated `explicit: False`. |
| **Conflict threshold** | Min score to declare preference for a library inside a conflict group. Default 8 (higher than the general min_score of 5). |
| **Negation** | A textual cue ("not", "without", "instead of", "migrate from") that subtracts score from the preceding library mention. |

---

## 4. Domain Model

### 4.1 Value Objects (immutable, identity by value)

v2 minimised the value object set after Codex flagged three of v1's objects as over-modeled. The objects below are the ones that genuinely deserve dataclass status (they're passed between modules, returned by public methods, have invariants worth checking). The dropped objects (`DetectionScore`, `ConflictGroup`, `MentionScorer`) live on as plain dicts/tuples and a renamed method.

#### 4.1.1 `PatternRule`

```python
@dataclass(frozen=True)
class PatternRule:
    """A single regex pattern that matches a library identifier or
    library-related concept."""
    pattern: str          # raw regex source
    weight: int           # 1-10 — strength of the signal
    explicit: bool        # True → contributes to preference; False → mention only
    rationale: str        # one-line audit comment: why this pattern is in the table

    def compile(self) -> re.Pattern:
        return re.compile(self.pattern, re.IGNORECASE)
```

Invariants:
- `weight in range(1, 11)` (1-10 inclusive)
- `pattern` compiles as a valid regex
- `rationale` non-empty

#### 4.1.2 `PatternMatch`

```python
@dataclass(frozen=True)
class PatternMatch:
    """A single (library, pattern, weight, span) match record."""
    library: str        # canonical library name (v3: now first field)
    pattern: str        # pattern source (for audit)
    weight: int
    text_span: str      # the substring that matched (first occurrence)

    def to_dict(self) -> Dict:
        return {
            "library": self.library,
            "pattern": self.pattern,
            "weight": self.weight,
            "text_span": self.text_span,
        }
```

v3 field-order standardised: `library` first to match the canonical evidence-entry shape used in PRD §FR-5, ADR §3.5, proposal §3.1 Step 4. v2 had `library` last; v3 puts it first for consistency. The `to_dict()` ordering also matches the JSON shape downstream consumers read.

#### 4.1.3 Conflict groups — plain map, not a value object

v1 introduced `ConflictGroup` as a frozen dataclass with `.includes()`. Codex flagged this as over-modeling — the data is a static `{group_name: tuple_of_libraries}` mapping that lives at module top level:

```python
CONFLICT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "web_automation": ("Browser", "SeleniumLibrary"),
    # Future: "mobile_native": ("AppiumLibrary_iOS", "AppiumLibrary_Android")
    # Future: "api_client": ("RequestsLibrary", ...)
}
```

The `includes()` check is `library in CONFLICT_GROUPS[group]` — no abstraction warranted. The mapping is `frozen` by virtue of being a module-level constant (no mutator API).

#### 4.1.4 `DetectionScore` — plain dict, not a value object

v1 wrapped per-library `(score, matches)` in a `DetectionScore` dataclass. Codex flagged this as window-dressing. v2 uses a plain dict directly:

```python
# Internal type alias used by LibraryDetector.score() return value
ScoresByLibrary = Dict[str, int]                # mention layer return
ExplicitScoresByLibrary = Dict[str, Tuple[int, List[PatternMatch]]]   # preference layer return
```

The `PreferenceResolver` consumes these dict types directly and emits a single `PreferenceResolution` (4.1.6) — no intermediate VO needed.

#### 4.1.5 `DetectionPolicy`

```python
@dataclass(frozen=True)
class DetectionPolicy:
    """Operator-tunable thresholds for the preference resolver."""
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

#### 4.1.6 `PreferenceResolution`

```python
@dataclass(frozen=True)
class PreferenceResolution:
    """The outcome of running the preference resolver on a scenario.
    Constitutes the public output of this bounded context."""
    library: Optional[str]                                # the chosen library, or None
    source: Literal["rule", "sampling"]                   # provenance (v2)
    evidence: List[Dict[str, Any]]                        # v4: list of PatternMatch.to_dict() — flat shape {library, pattern, weight, text_span}
    conflicts: Dict[str, List[Dict[str, Any]]]            # v4: dict-of-dicts to match PRD/ADR/proposal; each entry {library, score, patterns_matched}
    all_scores: Dict[str, int]                            # full score map (for diagnostics)
    sampling_evidence: Optional[str] = None               # raw LLM rationale when source=sampling (v2)

    @property
    def is_decisive(self) -> bool:
        return self.library is not None

    @property
    def has_conflicts(self) -> bool:
        return any(len(c) >= 2 for c in self.conflicts.values())
```

**v4 type correction (D3 fix)**: v3 declared `conflicts: Dict[str, List[Tuple[str, int, List[str]]]]` (tuple-of-three shape). PRD §FR-4, ADR §3.4, and proposal Step 6 all use a dict-of-dicts shape `[{library, score, patterns_matched}]`. v3 was the outlier and the proposal's `TestConflicts` test in §3.4 unpacked `entry[0]` (tuple style) but the algorithm returned dicts — broken as written. v4 aligns DDD to the dict shape used by the other three docs. `evidence` similarly changed from `List[PatternMatch]` to `List[Dict[str, Any]]` to reflect the actual to_dict-flattened shape that surfaces in the JSON response.

### 4.2 Aggregates and statefulness (v2 correction)

v1 claimed "the detector is stateless". That was wrong. `LibraryDetector` holds a compiled-pattern cache (computed once at `__init__`, used per call) and is consumed as a **process-global singleton** from `nlp_processor` and `session_models`. The singleton design predates this PRD and is not being changed.

What the v2 design genuinely guarantees:
- `LibraryDetector.score(text)` is **idempotent**: same text + same policy + same pattern table → same scores.
- `PreferenceResolver.resolve(text)` is **idempotent within a constructed policy**: see INV-3 in §7 for the precise statement.
- No state mutates after construction. The compiled-pattern cache is read-only at runtime; the policy is captured at construction. There is no add-pattern API at runtime.

The detector is therefore "effectively immutable" but not "stateless". The distinction matters for testing and for understanding the singleton lifecycle.

### 4.3 Domain Services

#### 4.3.1 `LibraryDetector` (existing, augmented)

Sole responsibility: turn a scenario text + pattern table into per-library scores.

```python
class LibraryDetector:
    """Process-global singleton. Holds compiled-pattern cache.
    Idempotent: same input + same policy + same pattern table -> same output."""

    LIBRARY_RULES: Dict[str, List[PatternRule]]   # source-of-truth pattern table
    NEGATION_PATTERNS: List[str]                   # subtractive patterns (v2: owned here)
    MIGRATION_PATTERNS: List[str]                   # source-vs-dest patterns (v2: new, owned here)

    def __init__(self, rules: Dict[str, List[PatternRule]], policy: DetectionPolicy): ...

    def score(self, text: str) -> Dict[str, int]:
        """Compute MENTION scores (uses all patterns; ignores explicit flag).
        Backward-compatible with existing `get_scores` callers (alias)."""
        ...

    def explicit_score(self, text: str) -> Dict[str, Tuple[int, List[PatternMatch]]]:
        """Compute PREFERENCE scores (uses only explicit=True patterns).
        Applies sentence-scoped negation + migration internally."""
        ...

    # Backward-compatibility shim — keeps test fixtures working:
    @property
    def _compiled_patterns(self) -> Dict[str, List[Tuple]]:
        """Existing tests inspect this directly. v2 preserves the attribute
        and ensures entries are tuple-compatible (extra trailing fields ignored
        by destructuring at index 0/1)."""
```

`get_scores(text)` is preserved as an alias for `score(text)` so existing test code keeps working (see PRD §NFR-5).

#### 4.3.2 `PreferenceResolver` (new, stateless)

Sole responsibility: combine detector output + policy + conflict groups into a `PreferenceResolution`.

```python
class PreferenceResolver:
    """Stateless. Composes detector + policy."""

    def __init__(self, detector: LibraryDetector, policy: DetectionPolicy,
                 conflict_groups: Dict[str, Tuple[str, ...]] = CONFLICT_GROUPS): ...

    def resolve(self, text: str) -> PreferenceResolution:
        """Return PreferenceResolution per the algorithm in ADR-024 §3.3.

        Steps (matches ADR-024 §3.3 exactly):
          1. detector.explicit_score(text) → raw_scores + matches
             (this call internally applies sentence-scoped negation + migration)
          2. CONFLICT CHECK on raw_scores (BEFORE threshold filter):
             for each group, if 2+ members have score > 0 and top-2 diff <= ambiguity_window
             → return PreferenceResolution(library=None, conflicts=...)
          3. Threshold filter:
             effective_threshold = conflict_min_score if lib in any group else default_min_score
             candidates = {lib: score for lib, score in raw_scores if score >= effective_threshold}
          4. If candidates is empty → return PreferenceResolution(library=None, ...)
          5. Else → return PreferenceResolution(library=max(candidates), evidence=matches[winner], ...)
        """
        ...
```

Note: `MentionScorer` from v1 is dropped. It was `LibraryDetector.get_scores` with a wrapper class. The wrapper added no behaviour; the alias suffices. v2 preserves `get_scores` as a method on `LibraryDetector` directly.

### 4.4 Domain Events (optional, for observability)

```python
@dataclass(frozen=True)
class PreferenceDetected:
    library: str
    score: int
    evidence_count: int
    text_hash: str  # not the raw text — keep observability privacy-safe

    def to_dict(self) -> Dict: ...

@dataclass(frozen=True)
class ConflictDetected:
    group: str
    libraries: Tuple[str, ...]
    scores: Dict[str, int]
    text_hash: str

    def to_dict(self) -> Dict: ...

@dataclass(frozen=True)
class NoPreferenceDetected:
    """Emitted when no library scored above threshold."""
    top_score: int                # highest score seen
    top_library: Optional[str]    # if any
    text_hash: str
```

These are optional. They land if rf-mcp wires an observability sink for the bounded context (instruction-learning hooks already capture tool results — events are NOT required for the bug fix).

---

## 5. Boundaries and Interactions

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  nlp_processor.analyze_scenario  (application service)           │
   │                                                                  │
   │  Inputs: scenario_text, context                                  │
   │  Outputs: structured TestScenario + analysis dict                │
   └─────────────┬─────────────────────────────┬─────────────────────┘
                 │                             │
                 │ (this PRD's path)           │ (SEPARATE path — outside this context)
                 ▼                             ▼
   ┌──────────────────────────┐    ┌─────────────────────────────────┐
   │  PreferenceResolver      │    │  _determine_capabilities         │
   │  ──────────────────      │    │  (substring heuristic)           │
   │  Used for:               │    │  ─────────────────────           │
   │    explicit_library_     │    │  Used for:                       │
   │      preference          │    │    suggested_libraries           │
   │    explicit_library_     │    │    (advisory list)               │
   │      evidence            │    │                                  │
   │    library_preference_   │    │  Does NOT call LibraryDetector;  │
   │      conflicts           │    │  uses its own substring matching │
   │    preference_source     │    │  at nlp_processor.py:517-544.    │
   │                          │    │                                  │
   │  Behaviour: conservative │    │  OUT OF SCOPE for this bounded   │
   │  evidence-based          │    │  context. Documented for clarity │
   │                          │    │  only.                           │
   └──────────────────────────┘    └─────────────────────────────────┘
                 │
                 ▼
   ┌──────────────────────────────────────────────────────┐
   │   LibraryDetector  (process-global singleton)        │
   │   ─────────────────                                  │
   │   detect_explicit_preference(text) → PreferenceResolution
   │   get_scores(text) → Dict[str, int]                  │
   │     (mention layer — DIAGNOSTIC ONLY in v3;          │
   │      no current production consumer)                 │
   └──────────────────────────────────────────────────────┘
```

v3 diagram correction: v2's diagram showed `MentionScorer` feeding "capability list (suggested_libs)" — that arrow was wrong. `suggested_libraries` is populated by `_determine_capabilities` (a separate substring heuristic at `nlp_processor.py:517-544`) which does NOT call `LibraryDetector`. The mention layer is preserved on `LibraryDetector` for diagnostics and potential future use, but has no production consumer in the current codebase.

**What this context OWNS** (v2 — added negation/migration to make ownership explicit):

| Asset | Owner | Notes |
|---|---|---|
| `LIBRARY_RULES` (per-library patterns) | `library_detection.py` | Single source of truth; module-level constant. |
| `NEGATION_PATTERNS` (subtractive) | `library_detection.py` | v2: explicitly owned here. Applied per-sentence inside `explicit_score`. v1 silently owned this but never documented. |
| `MIGRATION_PATTERNS` (source-vs-destination) | `library_detection.py` | v2: new; explicitly owned here. Source/dest tokens resolved to libraries via the same name-resolution as `LIBRARY_RULES`. |
| `CONFLICT_GROUPS` map | `library_detection.py` | Static module-level dict. |
| `DetectionPolicy` defaults + env names | `library_detection.py` | All `ROBOTMCP_LIBRARY_DETECTION_*` env vars resolved here. |
| `PreferenceResolver` algorithm | `library_detection.py` | Implements ADR-024 §3.3 exactly. |

**What this context DOES NOT OWN**:

| Asset | Owner | Notes |
|---|---|---|
| `_determine_capabilities` substring heuristic | `nlp_processor.py:517-544` | Separate code path; does not call this context. v1 implied a coupling that does not exist. |
| `suggested_libraries` response field | `nlp_processor.analyze_scenario` | Populated from `_determine_capabilities`, not from this context. |
| Sampling-based preference (`sample_analyze_scenario`) | `server.py:1863-1880` | External override; replaces `preference_source: "sampling"`. See ADR-024 §11. |
| Session library auto-import | `session_models.ExecutionSession.configure_from_scenario` | Downstream consumer of `PreferenceResolution.library`. |
| `find_keywords` filter precedence | `server.py:2037` | Downstream consumer. |

Anti-patterns the bounded context refuses:

- ❌ Direct mutation of `LIBRARY_RULES` / `NEGATION_PATTERNS` / `MIGRATION_PATTERNS` / `CONFLICT_GROUPS` from another module. Patterns are module-level constants and pattern entries are frozen dataclasses.
- ❌ Caching `PreferenceResolution` across scenarios — it's keyed on text + policy; different inputs must run the resolver fresh.
- ❌ Inferring preference from session library imports. Session config is downstream; detection is upstream.
- ❌ Calling `PreferenceResolver.resolve` with a per-call `DetectionPolicy` override at runtime. Policy is read at resolver construction; runtime overrides would break INV-3.

---

## 6. Migration from Current Code

### 6.1 What changes

| Current location | New location |
|---|---|
| `LIBRARY_PATTERNS: Dict[str, List[Tuple[str, int]]]` | `LIBRARY_RULES: Dict[str, List[PatternRule]]` (dataclass per rule with `explicit` + `rationale`) |
| `LibraryDetector.get_scores(text)` | Stays (alias for `score(text)`); now the documented mention API |
| `LibraryDetector.detect(text, min_score=...)` | Becomes thin wrapper around `PreferenceResolver.resolve` for backward compat |
| `LibraryDetector.get_conflicting_detections(text)` | Becomes a thin wrapper around `PreferenceResolver.resolve(text).conflicts` |
| `NEGATION_PATTERNS` (current location at lines 132-135) | Stays in `library_detection.py`; algorithm shifts from "single forward window across whole text" to "per-sentence application inside `explicit_score`" |
| (new) `MIGRATION_PATTERNS` | Added to `library_detection.py`; applied per-sentence with source/destination distinction |
| `nlp_processor._detect_explicit_library_preference` | Calls `PreferenceResolver.resolve` and returns `(resolution.library, resolution.evidence, resolution.conflicts)` |
| `session_models.detect_explicit_library_preference` | Same — single source of truth |
| `server.py:1866-1880` sampling override | Sets `preference_source="sampling"`, clears `evidence` and `conflicts`; per ADR-024 §11 |

### 6.2 What stays the same (back-compat surfaces)

- `LIBRARY_PATTERNS` source-of-truth (renamed `LIBRARY_RULES` in v3) — type changes from `Dict[str, List[Tuple[str, int]]]` to `Dict[str, List[PatternRule]]`. NOT directly inspected by tests; only used internally.
- **`LibraryDetector._compiled_patterns` — v3 keeps the EXACT current shape `Dict[str, List[Tuple[re.Pattern, int]]]`**. v2 implied a refactor; v3 holds the line. Real test contract at `tests/integration/test_nlp_improvements.py:571` is `for p, _ in entries: p.findall(...)` — `p` must be a compiled `re.Pattern`, NOT a `PatternRule`. The v3 implementation builds `_compiled_patterns` as 2-tuples and stores rich annotations in a parallel `_rules_metadata: Dict[str, List[PatternRule]]` attribute. New code paths read `_rules_metadata` or `LIBRARY_RULES`; legacy tests read `_compiled_patterns`. No `__iter__` shim on `PatternRule` is needed.
- Negation IDENTITY — the existing patterns ("not", "without", "instead of", "migrate from") keep working in spirit. The IMPLEMENTATION switches from a single forward-window regex to a **phrase-list + token-resolution** approach (v3 ADR §3.2 P3 + proposal §3.1 Step 5). The improved algorithm correctly handles `"do not use Selenium, instead use Playwright"`, which v2's regex broke.
- All public method names on `LibraryDetector` continue to exist (`get_scores`, `detect`, `get_conflicting_detections`).
- The `LibraryDetector` singleton pattern. No change to construction or to how `nlp_processor` and `session_models` obtain the instance.

### 6.3 What's added

- `PatternRule.explicit: bool` annotation. `True` only for verbatim library identifiers + preference-verb idioms; `False` for keyword-name overlaps + domain markers (per ADR-024 §6 v2 table).
- `PatternRule.rationale: str` audit comment.
- `MIGRATION_PATTERNS` constant (v2 new) — list of regex sources for "from X to Y" idioms.
- `DetectionPolicy.from_env()` for env-driven threshold overrides.
- `PreferenceResolver.resolve()` as the documented preference entry point.
- `PreferenceResolution.source` field (`"rule"` | `"sampling"`).
- `PreferenceResolution.sampling_evidence` field (optional LLM rationale).
- Response shape additions on `analyze_scenario`: `explicit_library_evidence`, `library_preference_conflicts`, `preference_source`.

### 6.4 What's NOT added (rejected scope creep)

- No new MCP tool (rejected per ADR-024 §4.4).
- No grammar parser / spaCy / parser dependency (rejected per ADR-024 §4.6/§4.7).
- No allow-list-only detector (rejected per ADR-024 §4.5).
- No public capability suggestion through this context. The mention layer stays diagnostic-only.

---

## 7. Invariants

| # | Invariant | Verification |
|---|---|---|
| INV-1 | `PatternRule.weight in range(1, 11)` | dataclass `__post_init__` |
| INV-2 | `PatternRule.pattern` compiles successfully | dataclass `__post_init__` |
| INV-3 | `PreferenceResolver.resolve(text)` is idempotent **for a given resolver instance**: a resolver constructed with policy P returns identical output for identical text input across repeated calls. (v2: NOT globally pure — policy is captured at construction from env vars; two resolvers built with different env states differ.) | parametrised property test against a fixed resolver |
| INV-4 | **(v4 replacement — catches D1 double-deduction)** Negation deduction-sum equality: for any sentence S containing one or more negation phrases all targeting library L, the total `raw_scores[L]` deduction across all phrase matches in S equals the deduction from a single canonical phrase match in S. Equivalently: `len(_NEGATION_REGEX.findall(S))` may be > 1 only when the matches target DISTINCT libraries; matches that resolve to the SAME library do not double-deduct. v3's INV-4 (`max(0, x-d)` idempotence) was trivially true and did NOT catch the round-3-flagged D1 bug where compound + simple phrases both fired on the same span. v4's invariant DOES catch it. | parametrised unit test: `"do not use Selenium"` → assert deduction = single-phrase deduction; `"do not use Selenium and stop the SeleniumLibrary"` → assert total deduction = 2× per-phrase (both target SL, but two distinct negation spans, not double-fire on one span) |
| INV-5 | Preference within a conflict group requires `score ≥ conflict_min_score` AFTER the raw-score conflict check has cleared. | unit test |
| INV-6 | When `PreferenceResolution.conflicts` is non-empty, `PreferenceResolution.library is None` and `source == "rule"`. | unit test |
| INV-7 | When `PreferenceResolution.library is not None` and `source == "rule"`, `evidence` is non-empty. When `source == "sampling"`, `evidence` is empty and `sampling_evidence` may be set. | unit test |
| INV-8 | Sentence-scoped negation reduces score only for the targeted library and only for contributions within the matching sentence. Other sentences' contributions for the same library remain. | unit test (multi-sentence scenarios) |
| INV-9 | Migration pattern with source=A and destination=B: score for A is reduced by all explicit contributions in the matching sentence, score for B is increased by `destination_bonus` (default 5). | unit test |
| INV-10 | All env-tunable knobs have defaults matching the documentation in ADR-024 §3.3 (`MIN_SCORE=5`, `CONFLICT_THRESHOLD=8`, `AMBIGUITY_WINDOW=4`). | unit test loading `DetectionPolicy.from_env()` with empty env |
| INV-11 | `_compiled_patterns` attribute exists on `LibraryDetector` and its values support `(pattern, weight)` destructuring at indices 0/1. | integration test (`tests/integration/test_nlp_improvements.py`) |

---

## 8. Bounded Context Boundaries (summary)

This context does NOT own (full table in §5):

- **Capability suggestion** (`_determine_capabilities` at nlp_processor.py:517-544) — separate substring heuristic that does NOT call this context. v1 implied a coupling; corrected in v2.
- **Session auto-import** (`ExecutionSession.configure_from_scenario`) — downstream consumer.
- **Find_keywords filter precedence** — downstream consumer.
- **Sampling-based override** (`sample_analyze_scenario`) — replaces this context's output when enabled. v2 defines the override coherence in ADR-024 §11: when sampling overrides, `evidence` and `conflicts` are cleared and `source` is set to `"sampling"`.
- **MCP tool surface** — `analyze_scenario` MCP tool is the application service that orchestrates this context with others.

The context owns the question "given scenario text and a `DetectionPolicy`, what library did the user explicitly intend?" and emits a `PreferenceResolution`. That is its only responsibility.

---

## 9. Open Questions for Implementation

1. **PatternRule construction syntax**: dataclass-by-keyword or builder? Recommended: keyword-only dataclass with a `__iter__` that yields `(pattern, weight)` first for tuple-unpacking compat.
2. **Should `explicit_library_evidence` include the regex source verbatim?** Recommended: yes — both `pattern` and `text_span` so users can grep their own text.
3. **How to handle the `_fallback_detect_library_preference` paths in `nlp_processor.py:665` and `session_models.py:592`?** Both predate the centralised detector. v2 deprecates them and routes through `PreferenceResolver` for single source of truth.
4. **What about future libraries (CryptoLibrary, ImageLibrary, etc.)?** Each gets its own pattern set + explicit/mention annotations. Adding a library is a localized change to the patterns table; the resolver doesn't need to know.
5. **Should the mention layer ever feed `_determine_capabilities`?** Out of scope for this PRD. A future refactor could route capabilities through `get_scores` for consistency; v2 leaves the path open by preserving the mention API but does not commit to a consumer.

---

## 13. Round-3 review findings + resolutions (v4)

| # | Round-3 finding (v3) | Resolution in v4 |
|---|---|---|
| D3 | §4.1.6 `conflicts` declared `Dict[str, List[Tuple[str, int, List[str]]]]` (tuple) but PRD/ADR/proposal use dict-of-dicts | §4.1.6: `Dict[str, List[Dict[str, Any]]]` — each entry `{library, score, patterns_matched}` |
| D3 | §4.1.6 `evidence` declared `List[PatternMatch]` but JSON shape is flat dicts | §4.1.6: `List[Dict[str, Any]]` reflecting `PatternMatch.to_dict()` shape |
| D2 | INV-4 `max(0, x-d)` idempotence trivially true; wouldn't catch the D1 double-deduction bug | §7 INV-4: replaced with deduction-sum equality across multiple negation-phrase matches; CATCHES D1 |

---

## 12. Round-2 review findings + resolutions (v3)

| # | Codex finding (v2 round 2) | Resolution in v3 |
|---|---|---|
| B4 | §5 diagram showed `MentionScorer` → "capability list (suggested_libs)" contradicting v2 prose | §5 diagram redrawn: `_determine_capabilities` shown as SEPARATE path outside the bounded context; mention layer marked "diagnostic only" |
| C4 | INV-4 was vacuous (every library has at least one `explicit=True` pattern) | §7 INV-4 replaced with negation-idempotence: `_subtract_sentence_score` applied twice equals applied once |
| B2 | `_compiled_patterns` test contract description ambiguous | §6.2 documents the two-store design: `_compiled_patterns` keeps `(Pattern, int)` 2-tuples; `_rules_metadata` holds rich `PatternRule` entries |
| D1 | Evidence shape differed across 4 docs | §4.1.2 `PatternMatch.to_dict()` standardised to `{library, pattern, weight, text_span}` field order; matches PRD §FR-5, ADR §3.5, proposal Step 4 |
| E3 | Sampling override path is singular in docs; source has TWO (server.py:1860-1881 AND :1961-1972) | §8 + §11 reference both override sites |

---

## 11. Round-1 review findings + resolutions (v2)

| # | Codex finding | Resolution in v2 |
|---|---|---|
| 1 | `DetectionScore`/`ConflictGroup`/`MentionScorer` are window-dressing | Demoted to plain dict/constant/method-alias in §4.1.3, §4.1.4, §4.3.2 |
| 2 | "LibraryDetector becomes stateless" — false | Retracted in §4.2; "effectively immutable but not stateless" is the corrected term |
| 3 | INV-3 (purity) unprovable with env-driven policy | Reworded to "idempotent for a given resolver instance" |
| 4 | INV-4 (mention ≥ preference) unprovable since capability suggestion doesn't consume mention scores | Replaced with INV-4 about explicit-only filtering |
| 5 | NEGATION_PATTERNS ownership omitted from §5 Boundaries | Explicit ownership table added to §5 with negation + migration listed |
| 6 | Architecture claim wrong (mention layer backs `suggested_libraries`) | §2 + §5 corrected; mention layer is diagnostic-only |
| 7 | `_compiled_patterns` test surface compat ignored | INV-11 added; §6.2 documents the back-compat shim |
| 8 | Sampling override coherence undefined | §4.1.6 adds `source` + `sampling_evidence` fields; §8 + ADR-024 §11 define behaviour |

---

## 10. Cross-references

- **PRD**: `docs/prd/analyze_scenario_explicit_library_prd.md` — user-facing requirements + acceptance criteria.
- **ADR-024**: `docs/adr/ADR-024-explicit-library-detection-confidence.md` — architecture decisions + alternatives.
- **Solution proposal**: `docs/proposals/explicit_library_detection_fix_proposal.md` — concrete code changes + migration plan.
