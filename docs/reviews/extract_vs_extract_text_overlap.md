# Analysis: `intent="extract"` vs `intent="extract_text"` overlap

**Date**: 2026-05-17
**Branch**: `feature/obstacle-course-improvements` @ `4ba0a1e`
**Question**: After OBS-06 (PR #69), does `intent_action` expose both `"extract"`
and `"extract_text"` as parallel verbs? Are they functionally overlapping?
Could they be unified into a single `extract` with `mode=`? Will having both
confuse agents?

**Verdict at a glance**:

| Question | Answer |
|---|---|
| Are both exposed? | Yes — `IntentVerb.EXTRACT_TEXT = "extract_text"` (since ADR-006/007/008, ~6 months ago) and `IntentVerb.EXTRACT = "extract"` (since OBS-06, this PR). |
| Functional overlap? | **Yes — strict superset.** `extract(mode="text")` produces byte-identical RF dispatch to `extract_text`, plus exposes `extracted_value` at top level. |
| Could they be unified? | Yes — `extract(mode="text")` covers 100% of `extract_text` semantics. |
| Will it confuse agents? | **Yes — material risk.** Empirical evidence below. |

---

## 1. Empirical comparison

Resolved via the real `IntentActionAdapter` with an identity-normaliser and a
mocked session lookup (Browser library):

```
extract_text(id=foo):
  keyword='Get Text'  args=['id=foo']  mode=None

extract(id=foo):
  keyword='Get Text'  args=['id=foo']  mode='text'

extract(id=foo, mode=text):
  keyword='Get Text'  args=['id=foo']  mode='text'
```

The RF keyword + arguments are **identical** for text extraction. The only
difference in the resolution dict: `extract` carries `extract_mode="text"`,
which the server uses to:

1. Surface `result["extracted_value"]` at the top level (OBS-06).
2. Optionally bypass pre-validation for `mode="count"` (irrelevant for text).

For text extraction specifically, `extract(mode="text")` is `extract_text` plus
an `extracted_value` convenience field. **There is no scenario where
`extract_text` does something `extract(mode="text")` can't.**

## 2. Registration coverage

Each library has TWO separate `IntentMapping` entries for what's effectively
the same operation:

| Library | EXTRACT_TEXT mapping | EXTRACT (mode=text) mapping |
|---|---|---|
| Browser | `intent_verb=EXTRACT_TEXT, keyword="Get Text", requires_target=True` | `intent_verb=EXTRACT, keyword="Get Text", requires_target=False, transformer=_extract_browser_transformer` |
| SeleniumLibrary | `intent_verb=EXTRACT_TEXT, keyword="Get Text", requires_target=True` | `intent_verb=EXTRACT, keyword="Get Text", requires_target=False, transformer=_extract_selenium_transformer` |
| AppiumLibrary | `intent_verb=EXTRACT_TEXT, keyword="Get Text", requires_target=True` | `intent_verb=EXTRACT, keyword="Get Text", requires_target=False, transformer=_extract_selenium_transformer` |

The mappings differ only in:
- `requires_target=True` (EXTRACT_TEXT) vs `False` (EXTRACT, since modes
  `url`/`title` don't need a target — the per-mode transformer validates)
- `argument_transformer=None` (EXTRACT_TEXT, default positional) vs
  `_extract_*_transformer` (EXTRACT, mode-aware shape)

Both produce the same `Get Text <locator>` dispatch for text extraction.

## 3. User-facing surfaces where both names appear

### 3a. `intent_action` docstring (server.py:6361-6363) — CONFUSING

```
Valid intents: navigate, click, fill, hover, select, assert_visible,
extract_text, wait_for, extract.
```

Both `extract_text` AND `extract` are listed as valid intents in a flat
8+1-item list. No annotation that one supersedes the other, no cross-reference,
no hint that they overlap. An LLM scanning this list sees two parallel verbs
for state extraction and has to guess which to use.

### 3b. Error-hint "valid intents:" message (server.py:6571) — STALE BUG

```python
"valid intents: navigate, click, fill, hover, select, "
"assert_visible, extract_text, wait_for"
```

This error message — shown when intent resolution fails — lists 8 intents and
**omits "extract" entirely**. An agent that fails resolution and reads this
hint would conclude `extract` is invalid. Bug-grade inconsistency between
the success-path docstring and the failure-path hint.

### 3c. `IntentVerb` Literal alias (kernel.py:433-440)

```python
IntentVerb = Annotated[
    Literal[
        "navigate", "click", "fill", "hover",
        "select", "assert_visible", "extract_text", "wait_for",
        "extract",
    ],
    BeforeValidator(_normalize_str),
]
```

Both validate cleanly; the schema reaches downstream consumers (OpenAPI /
ADR-009 schemas / agent tool definitions) — both appear in every consumer.

### 3d. Cookbook (discovery_first instructions) — neither named

Section 8 ("WHEN PRE-VALIDATION REJECTS YOUR LOCATOR") cross-references
`force=True` and `commit=True` but mentions neither `extract` nor
`extract_text`. So agents don't learn about either from the default
instructions; they have to fall back to the `intent_action` docstring's
"Valid intents:" list — which is where 3a's confusion bites.

## 4. Internal test coverage of `extract_text`

14 test references across the unit + integration suite. Categorised:

| Category | Count | Examples |
|---|---|---|
| List-of-all-verbs assertions | 6 | `test_adr009_type_aliases.py`, `test_intent_value_objects.py::test_has_exactly_9_values`, `test_intent_registry.py::test_appium_7` |
| Dedicated `extract_text` mapping/resolution tests | 3 | `test_intent_registry.py::test_extract_text_browser`, `test_intent_resolver.py::test_extract_text_browser` |
| `extract_text` as a string literal in negative parametrise (`test_action_keyword_recorded`) | 1 | `test_record_gate_fn12.py` |
| Coincidental string match in doc comments | 4 | Various |

So **9 of 14** references are genuine production-test pins of the verb's
existence + behaviour. If `extract_text` is removed, all 9 need migration.

## 5. Confusion scenarios — concrete agent failure modes

The OBS-* series wasn't tested with both verbs available because OBS-06
shipped them simultaneously. But by direct analogy from the 2026-05-17
Tricentis benchmark behaviour I've seen:

| # | Scenario | Likely agent behaviour | Cost |
|---|---|---|---|
| 1 | "I want the text of the cart badge to assert against" | Picks `extract_text` (most verbose, "looks safer"). Misses the `extracted_value` top-level field, has to dig through `result["result"]`. | 1-2 extra calls to find the value, OR an `assign_to` capture that the user already does anyway. Minor. |
| 2 | "I want the count of `.item` elements" | Knows `extract_text` won't work for count, reads docstring for `extract`. Picks `extract(mode="count")`. Successful. | Cognitive overhead: had to learn TWO names for related operations. |
| 3 | Generated test suite reviewer | Suite from agent A uses `Get Text` (via `extract_text`). Suite from agent B uses `Get Text` (via `extract(mode="text")`). Diffs look inconsistent in PR review even though they do the same thing. | Inconsistent suite output across runs / agents. |
| 4 | Agent reads the error-hint list (3b above) and concludes `extract` is invalid | Falls back to `extract_text` even when it wanted `mode="count"`. | Worst case — agent reaches for `Evaluate JavaScript` workaround. The OBS-06 work is partially defeated. |
| 5 | New rf-mcp user reading the docstring's "Valid intents" | Asks "what's the difference?" and either reads the source or just picks arbitrarily. | Documentation debt. |

**Worst-case (#4)** is the only one that's directly user-impactful. The
stale error-hint at server.py:6571 is a real bug regardless of how this
analysis resolves.

## 6. Options + recommendation

### Option A — Deprecate `extract_text` in favour of `extract(mode="text")` ★ RECOMMENDED

**What changes:**
- `IntentVerb.EXTRACT_TEXT` stays for backward compat; the IntentRegistry still
  resolves it; existing user code keeps working.
- `intent_action` docstring marks `extract_text` as `[DEPRECATED — prefer
  intent="extract" with mode="text"]` in the "Valid intents:" listing.
- The error-hint at server.py:6571 is updated to list `extract` and omit
  `extract_text` (or marks `extract_text` deprecated).
- A `DeprecationWarning` fires when the resolver receives `intent_verb=EXTRACT_TEXT`.
- Internally, the EXTRACT_TEXT mapping's resolution is kept as-is; no behaviour
  change for callers.
- After 1-2 minor releases (or 1 major), the mapping + verb can be removed.

**Cost**:
- 1 docstring update + 1 hint-text fix + 1 DeprecationWarning emit.
- 9 internal tests for `extract_text` keep passing (deprecation doesn't break
  them).
- No new agent learning required — agents discover `extract` from the docstring.

**Benefit**:
- LLM sees one canonical verb (`extract`) + a deprecation note guiding to it.
- Generated suites converge on `Get Text` via `extract(mode="text")` (clean,
  consistent).
- The new mode-aware verb gets the agent-attention it earned.

### Option B — Remove `extract_text` entirely (breaking change)

Functional unification. Requires migrating 9 internal tests and any external
consumer code. Not recommended without a major-version bump.

### Option C — Keep both, fix only the error-hint inconsistency

Lowest-effort, lowest-payoff. Cures the stale-hint bug (3b) but doesn't
reduce verb confusion (3a). Defers the unification debt.

### Option D — Make `extract_text` an internal alias of `extract(mode="text")`

In the adapter, translate `intent="extract_text"` → `intent="extract",
mode="text"` before resolution. Removes the duplicate IntentMapping per
library (3 mappings deleted). Both names still appear in the public Literal
alias and docstring.

Less code, same agent-confusion surface. Halfway house.

## 7. Recommendation

**Option A**, executed as a follow-up commit on the current branch (or a
follow-up PR after #69 merges):

1. Fix the bug at `server.py:6571` — the error-hint must list `extract` (and
   optionally call out `extract_text` as deprecated).
2. Mark `extract_text` deprecated in the `intent_action` docstring's "Valid
   intents:" line with a single `[deprecated, use intent="extract" with
   mode="text"]` annotation.
3. Add a `DeprecationWarning` at the resolver entry point when
   `intent_verb == IntentVerb.EXTRACT_TEXT`.
4. Add a one-line note to the IntentVerb enum docstring + the kernel.py
   Literal comment.
5. Keep all internal tests passing; no breaking change.

**Effort**: ~30 lines across 4 files; ~5 tests pinning the deprecation
behaviour. Same scope as OBS-07's docstring reframe.

**Defer**: removal of `EXTRACT_TEXT` enum value + Literal entry + mappings.
Mark as a v0.34 cleanup if the deprecation warning generates no friction
during this release cycle.

## Appendix A — Files where both verbs touch user-visible behaviour

```
src/robotmcp/domains/intent/value_objects.py
  Line 27   IntentVerb.EXTRACT_TEXT = "extract_text"
  Line 32   IntentVerb.EXTRACT = "extract"

src/robotmcp/domains/intent/aggregates.py
  Lines 517, 525   Browser mappings (EXTRACT_TEXT + EXTRACT)
  Lines 612, 620   Selenium mappings
  Lines 684, 692   Appium mappings

src/robotmcp/domains/shared/kernel.py
  Lines 433-440   IntentVerb Literal alias — both listed

src/robotmcp/server.py
  Line 6362   intent_action docstring "Valid intents:" — both listed
  Line 6571   error-hint message — extract_text listed, extract OMITTED (bug)
```

## Appendix B — Reproduction script

```python
from robotmcp.domains.intent.adapters.mcp_tool import IntentActionAdapter
from robotmcp.domains.intent.aggregates import IntentRegistry
from robotmcp.domains.intent.services import IntentResolver
from robotmcp.domains.intent.value_objects import NormalizedLocator


class FakeSessionLookup:
    def get_active_library(self, sid): return "Browser"
    def get_active_web_library(self, sid): return "Browser"
    def get_imported_libraries(self, sid): return ["Browser"]


class IdentityNormalizer:
    def normalize(self, target, library):
        return NormalizedLocator(
            value=target.locator, source_locator=target.locator,
            target_library=library, strategy_applied="auto",
            was_transformed=False,
        )


reg = IntentRegistry.with_builtins()
adapter = IntentActionAdapter(
    resolver=IntentResolver(
        registry=reg,
        session_lookup=FakeSessionLookup(),
        normalizer=IdentityNormalizer(),
    ),
)

for label, kwargs in [
    ("extract_text(id=foo)", {"intent": "extract_text", "target": "id=foo"}),
    ("extract(id=foo)",       {"intent": "extract",      "target": "id=foo"}),
    ("extract(id=foo, mode=text)",
                              {"intent": "extract", "target": "id=foo", "mode": "text"}),
]:
    r = adapter.resolve_intent(**kwargs)
    print(label)
    print(f"  keyword={r['keyword']!r}  args={r['arguments']!r}  mode={r.get('extract_mode')!r}")
```

Run with: `uv run python repro.py`. Output confirms identical RF dispatch
for the text-extraction case.
