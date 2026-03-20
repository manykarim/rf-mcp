# ADR-019: BDD Style, Embedded Arguments & Data-Driven Testing -- DDD Architecture

**Status:** Proposed
**Date:** 2026-03-07
**Author:** System Architect (Swarm)
**Domain:** rf-mcp Test Style & Data-Driven Support
**Research:** `docs/research/bdd-style-support.md`, `docs/research/embedded-arguments-support.md`, `docs/research/data-driven-style-support.md`, `docs/research/datadriver-integration.md`
**Depends on:** ADR-001 (DDD Architecture), ADR-005 (Multi-Test Sessions), ADR-007 (Intent Layer), ADR-011 (Batch Execution)

---

## 1. Context and Problem Statement

rf-mcp has **zero support** for three widely-used Robot Framework test authoring patterns: BDD style (Given/When/Then), embedded arguments, and data-driven testing (`[Template]` / DataDriver). These three features are deeply interrelated in RF -- BDD-style tests commonly use embedded argument keywords, and data-driven tests often combine `[Template]` with embedded arguments for readable, parameterized specifications.

### 1.1 BDD Prefix Stripping -- 10+ Broken Locations

RF natively strips BDD prefixes (`Given`, `When`, `Then`, `And`, `But`) before keyword resolution via `Namespace._get_bdd_style_runner()`. However, rf-mcp performs its own keyword resolution and processing in **10+ locations before** the keyword reaches RF's namespace:

| Component | BDD Handling | Consequence |
|-----------|-------------|-------------|
| `KeywordDiscovery.find_keyword()` | None | `"Given Log"` returns None; cache keys are plain names |
| `DynamicKeywordDiscovery.find_keyword()` | None | Session-aware lookup fails |
| `_ensure_library_registration()` | None | Required library not auto-loaded |
| Pre-validation (`_requires_pre_validation()`) | None | Element actionability checks skipped |
| Timeout injection (`_inject_timeout()`) | None | Smart timeouts not applied |
| Plugin overrides | None | e.g., Open Browser guidance skipped |
| Context routing lists | None | Wrong execution path chosen |
| `find_keywords` MCP tool | None | Discovery results wrong |
| `build_test_suite` | None | No BDD-style `.robot` output |
| `intent_action` | None | IntentVerb has no BDD concept |

### 1.2 Embedded Arguments -- KeywordDiscovery Cannot Match

RF's `EmbeddedArguments` class compiles keyword name templates (e.g., `Select ${animal} from list`) into regex patterns and matches concrete calls (e.g., `Select dog from list`) via `fullmatch()`. rf-mcp's `KeywordDiscovery` stores keywords in a flat dict keyed by lowercase name. When an LLM passes `"Select dog from list"`, no match is found because the cache only contains the template key `"select ${animal} from list"`. RF's own `KeywordCache` separates `normal` (dict lookup) from `embedded` (linear regex scan) -- rf-mcp has no equivalent split.

### 1.3 Data-Driven Testing -- No `[Template]` Support

rf-mcp's test builder (`_generate_rf_text()`) produces keyword-driven `.robot` files exclusively. There is no `template` field on `TestInfo` or `GeneratedTestCase`, no `[Template]` emission in generated output, and no `manage_session` parameter to declare a test as data-driven. The entire data model treats every step as an explicit keyword call.

### 1.4 DataDriver Library -- Companion File Path Breakage

`SuiteExecutionService` copies `.robot` content to a temporary directory but does not copy companion data files (CSV, Excel, JSON). DataDriver resolves data file paths relative to the suite source, so execution in temp directories fails with `FileNotFoundError`. Additionally, DataDriver provides zero keywords (it is a listener-based test case generator), so it does not fit the `LibraryPlugin` protocol.

## 2. Decision Drivers

| Driver | Weight | Rationale |
|--------|--------|-----------|
| **Pattern consistency** | Critical | Must follow the exact DDD conventions from ADR-001/006/007/008/011 (frozen value objects, mutable aggregates, Protocol-based services, frozen events with `to_dict()`) |
| **BDD prefix stripping order** | Critical | Must happen BEFORE embedded argument matching -- RF strips prefixes first, then matches against embedded patterns. Reversing this order breaks `Given user selects dog` (Experiment 4 in embedded-arguments research) |
| **RF native context compatibility** | Critical | RF's `BuiltIn.run_keyword()` already handles BDD + embedded args correctly. rf-mcp's preprocessing layers must align, not conflict, with RF's behavior |
| **Minimal API surface** | High | Prefer extending existing MCP tools (`execute_step`, `find_keywords`, `build_test_suite`, `manage_session`) over adding new tools. Only `load_test_data` is genuinely new |
| **LLM guidance quality** | High | Embedded argument keywords must be surfaced with usage examples (e.g., `"Select dog from list"` not just `"Select ${animal} from list"`) so LLMs understand the calling convention |
| **Backward compatibility** | High | All existing keyword-style tests must continue to work unchanged. BDD/embedded/template features are additive |
| **DataDriver path resolution** | High | Suite execution must handle companion files; this blocks all DataDriver workflows |
| **Bounded context isolation** | High | Keyword resolution (BDD strip + embedded match) is a different concern from Intent (LLM verb simplification) and batch execution (sequential steps). Clean separation follows ADR-001 |

## 3. Experiment Results (Validated)

### 3.1 BDD Prefix Experiments

**Experiment 1: RF BDD Prefix Regex**
```python
from robot.conf.languages import Languages
lang = Languages()
lang.bdd_prefixes  # -> {'When', 'But', 'Given', 'And', 'Then'}
lang.bdd_prefix_regexp.pattern  # -> '(given|when|then|but|and)\s'
```
Case-insensitive, requires trailing space. `GivenLog` does NOT match.

**Experiment 2: KeywordDiscovery Without BDD Stripping**
```
find_keyword("Log")                   -> Log       (works)
find_keyword("Given Log")             -> None      (FAILS)
find_keyword("When Log")              -> None      (FAILS)
find_keyword("Then Should Be Equal")  -> Should Be Equal  (accidental fuzzy match >= 0.6)
find_keyword("And Log")               -> None      (FAILS)
```
Short keywords with BDD prefixes fail completely. Longer keywords may accidentally match wrong keywords via fuzzy scoring.

**Experiment 3: RF Execution Chain**
`BuiltIn.run_keyword("Given Log", "hello")` correctly strips the prefix and resolves `Log`. The gap is entirely in rf-mcp's preprocessing layers.

### 3.2 Embedded Argument Experiments

**Experiment 4: EmbeddedArguments API**
```python
from robot.running.arguments.embedded import EmbeddedArguments
ea = EmbeddedArguments.from_name('Select ${animal} from list')
ea.args                                # -> ('animal',)
ea.name.pattern                        # -> 'Select\\s(.*?)\\sfrom\\slist'
ea.matches('Select dog from list')     # -> True
ea.matches('Click button')            # -> False
ea.parse_args('Select dog from list')  # -> ('dog',)
```

**Experiment 5: Custom Regex Patterns**
```python
ea = EmbeddedArguments.from_name('Select ${count:\\d+} items')
ea.custom_patterns  # -> {'count': '\\d+'}
ea.matches('Select 5 items')    # -> True
ea.matches('Select abc items')  # -> False
```

**Experiment 6: BDD Prefix + Embedded Args Interaction**
```python
ea = EmbeddedArguments.from_name('user selects ${item}')
ea.matches('Given user selects apple')  # -> False
ea.matches('user selects apple')        # -> True
```
**Critical finding**: BDD prefix stripping MUST happen before embedded matching. RF does this in `_get_bdd_style_runner()`.

**Experiment 7: No Standard RF Libraries Use Embedded Arguments**
Checked BuiltIn -- none of its keywords use embedded arguments. The feature is primarily used in user-defined keywords and custom test libraries.

### 3.3 Data-Driven Experiments

**Experiment 8: RF TestCase.template Attribute**
```python
from robot.running.model import TestCase
tc = TestCase(name='Test')
tc.template  # -> None (default)
tc.template = 'Log'
tc.template  # -> 'Log'
```
`TestCase.template` is a first-class `str | None` attribute in RF 7.3.2.

**Experiment 9: Continue-on-Failure Behavior**
Template tests with `[Template]` use continue-on-failure semantics. All data rows execute regardless of individual failures. The test status is FAIL only after all rows complete, with the first failure's message.

**Experiment 10: Embedded Arguments in Templates**
With `[Template]    Login With ${user} And ${password} Should ${result}`, RF substitutes data values into the keyword name. Body keywords get `name='Login With admin And secret Should Succeed'` with empty `args`.

### 3.4 DataDriver Experiments

**Experiment 11: run_test_suite with DataDriver**
- Relative data path: FAILS -- `FileNotFoundError`, DataDriver falls back to running only the template test (1 test instead of 3).
- Absolute data path: WORKS -- 3 tests, 3 passed.

**Experiment 12: Programmatic Reader Access**
```python
from DataDriver.ReaderConfig import ReaderConfig
from DataDriver.csv_reader import csv_reader
config = ReaderConfig(file='test_data.csv', dialect='Excel-EU', delimiter=';')
reader = csv_reader(config)
data = reader.get_data_from_source()  # Returns List[TestCaseData]
```
Works standalone without RF context. Returns structured test case data.

## 4. Architecture Decision

Implement a **new bounded context** `keyword_resolution` plus targeted extensions to existing domains (intent, batch_execution) and infrastructure components (test_builder, suite_execution_service).

### 4.1 Why a New Bounded Context

The `keyword_resolution` domain is distinct from the existing `intent` domain:

| Concern | Intent Domain (ADR-007) | Keyword Resolution Domain (this ADR) |
|---------|------------------------|--------------------------------------|
| Purpose | Map abstract LLM verbs to concrete keyword names | Transform concrete keyword names through BDD/embedded/template pipelines |
| Direction | Abstract -> Concrete (`CLICK` -> `Click Element`) | Concrete -> Resolved (`Given Select dog from list` -> `Select ${animal} from list` with args `('dog',)`) |
| Consumer | Small LLMs that cannot pick correct library keywords | All LLMs, regardless of capability |
| RF dependency | None (pure mapping tables) | Uses RF's `Languages`, `EmbeddedArguments` classes directly |
| Lifecycle | Static mappings, rarely change | Dynamic -- depends on loaded libraries' embedded keywords |

Extending the Intent domain would conflate two different concerns and violate the single-responsibility principle established in ADR-001.

### 4.2 Context Map

```
                    +------------------------------------------------------------+
                    |                       server.py                             |
                    |  +---------------+ +---------------+ +------------------+  |
                    |  | execute_step  | | find_keywords  | | build_test_suite |  |
                    |  | (BDD strip    | | (BDD strip    | | (style param,   |  |
                    |  |  at entry)    | |  + embedded   | |  template emit) |  |
                    |  +-------+-------+ |  metadata)    | +--------+---------+  |
                    |          |         +-------+-------+          |            |
                    |  +-------+-------+         |         +--------+---------+  |
                    |  | manage_session|         |         | load_test_data   |  |
                    |  | (template     |         |         | (DataDriver      |  |
                    |  |  param)       |         |         |  readers)        |  |
                    |  +-------+-------+         |         +--------+---------+  |
                    +----------+-----------------+------------------+------------+
                               |                 |                  |
             +-----------------+-----------------+------------------+----------+
             | keyword_resolution domain                                       |
             |                                                                 |
             |  +-------------------+    +------------------+                  |
             |  | KeywordResolver   |    | DataDrivenSuite  |                  |
             |  | (aggregate)       |    | (aggregate)      |                  |
             |  | 1. strip BDD      |    | template +       |                  |
             |  | 2. exact match    |    | data rows        |                  |
             |  | 3. embedded match |    +------------------+                  |
             |  +--------+----------+                                          |
             |           | uses                                                |
             |  +--------+----------+    +------------------+                  |
             |  | BddPrefixService  |    | EmbeddedMatcher  |                  |
             |  | (strip prefix)    |    | (regex match)    |                  |
             |  +-------------------+    +------------------+                  |
             |                                                                 |
             |  +-------------------+    +------------------+                  |
             |  | TemplateRenderer  |    | DataSourceLoader |                  |
             |  | ([Template] emit) |    | (DataDriver API) |                  |
             |  +-------------------+    +------------------+                  |
             +--+-------------------+------------------+-------+---------------+
                |                   |                  |       |
     +----------+---+   +----------+------+   +-------+--+  +-+------------------+
     | KeywordDis-  |   | keyword_executor|   | test_    |  | suite_execution_   |
     | covery       |   | (use resolved   |   | builder  |  | service            |
     | (+embedded   |   |  keyword for    |   | (template|  | (companion file    |
     |  cache)      |   |  pre-val/       |   |  render) |  |  handling)         |
     +-------+------+   |  timeout)       |   +----------+  +--------------------+
             |          +--------+--------+
             |                   |
     +-------+-------------------+-----+
     | existing domains                |
     | +----------+ +---------------+  |
     | | intent/  | | batch_exec/   |  |
     | | (no dep) | | (template-    |  |
     | |          | |  aware exec)  |  |
     | +----------+ +---------------+  |
     +---------------------------------+
```

### 4.3 Bounded Context: `keyword_resolution`

Domain: `src/robotmcp/domains/keyword_resolution/`

```
keyword_resolution/
    __init__.py            # Public API exports
    value_objects.py       # BddPrefix, EmbeddedPattern, EmbeddedMatch,
                           # KeywordForm, TemplateSpec, DataSource,
                           # DataRow, DataFormat, BddPrefixType, TestStyle
    entities.py            # ResolvedKeyword
    aggregates.py          # KeywordResolver, DataDrivenSuite
    services.py            # BddPrefixService, EmbeddedMatcher,
                           # TemplateRenderer, DataSourceLoader
                           # + Protocol definitions
    events.py              # BddPrefixStripped, EmbeddedArgMatched,
                           # TemplateApplied, DataSourceLoaded,
                           # KeywordResolutionFailed
```

### 4.4 Value Objects

All value objects follow the established `@dataclass(frozen=True)` + `__post_init__` validation + `ClassVar` constants pattern (per timeout/intent/batch_execution domains).

#### `BddPrefixType` -- Prefix Classification

```python
class BddPrefixType(str, Enum):
    GIVEN = "Given"
    WHEN = "When"
    THEN = "Then"
    AND = "And"
    BUT = "But"
```

#### `BddPrefix` -- Stripped Prefix Info

```python
@dataclass(frozen=True)
class BddPrefix:
    original_name: str          # "Given Open Browser"
    stripped_name: str           # "Open Browser"
    prefix: str                 # "Given"
    prefix_type: BddPrefixType  # BddPrefixType.GIVEN

    def __post_init__(self):
        if not self.original_name:
            raise ValueError("original_name must not be empty")
        if not self.stripped_name:
            raise ValueError("stripped_name must not be empty")
        if not self.original_name.lower().startswith(self.prefix.lower()):
            raise ValueError(f"original_name must start with prefix '{self.prefix}'")
```

**Example**: `BddPrefix("Given Open Browser", "Open Browser", "Given", BddPrefixType.GIVEN)`

#### `EmbeddedPattern` -- Compiled Embedded Arg Template

```python
@dataclass(frozen=True)
class EmbeddedPattern:
    template_name: str          # "Select ${animal} from list"
    args: tuple[str, ...]       # ("animal",)
    regex: re.Pattern           # compiled from RF's EmbeddedArguments
    custom_patterns: dict       # {"count": "\\d+"} for constrained args
    _embedded: object           # RF EmbeddedArguments instance (private, for matching)

    @classmethod
    def from_rf(cls, ea: "EmbeddedArguments", template_name: str) -> "EmbeddedPattern":
        """Create from RF's EmbeddedArguments object."""
        return cls(
            template_name=template_name,
            args=tuple(ea.args),
            regex=ea.name,  # compiled regex
            custom_patterns=dict(getattr(ea, 'custom_patterns', {})),
            _embedded=ea,
        )

    def matches(self, name: str) -> bool:
        """Check if a concrete keyword name matches this pattern."""
        return self._embedded.matches(name)

    def parse_args(self, name: str) -> tuple[str, ...]:
        """Extract argument values from a concrete keyword name."""
        return self._embedded.parse_args(name)
```

**Invariant**: `template_name` must contain `${`. The regex is compiled at construction time via RF's `EmbeddedArguments.from_name()`.

#### `EmbeddedMatch` -- Result of Embedded Arg Matching

```python
@dataclass(frozen=True)
class EmbeddedMatch:
    pattern: EmbeddedPattern          # the matched pattern
    extracted_args: tuple[str, ...]   # ("dog",) -- concrete values
    keyword_info: object              # KeywordInfo from discovery cache

    @property
    def template_name(self) -> str:
        return self.pattern.template_name

    @property
    def arg_names(self) -> tuple[str, ...]:
        return self.pattern.args
```

**Example**: Calling `Select dog from list` matches pattern `Select ${animal} from list`, extracting `("dog",)`.

#### `KeywordForm` -- Normalized Keyword Ready for Resolution

```python
@dataclass(frozen=True)
class KeywordForm:
    raw_name: str                           # original from LLM: "Given Select dog from list"
    normalized_name: str                    # after BDD strip: "Select dog from list"
    bdd_prefix: Optional[BddPrefix]        # BddPrefix or None
    embedded_match: Optional[EmbeddedMatch] # EmbeddedMatch or None (set after matching)

    @classmethod
    def from_raw(cls, raw_name: str, bdd_prefix: Optional[BddPrefix] = None) -> "KeywordForm":
        return cls(
            raw_name=raw_name,
            normalized_name=bdd_prefix.stripped_name if bdd_prefix else raw_name,
            bdd_prefix=bdd_prefix,
            embedded_match=None,
        )

    def with_embedded_match(self, match: EmbeddedMatch) -> "KeywordForm":
        """Return new KeywordForm with embedded match set (immutable update)."""
        return KeywordForm(
            raw_name=self.raw_name,
            normalized_name=self.normalized_name,
            bdd_prefix=self.bdd_prefix,
            embedded_match=match,
        )
```

#### `TestStyle` -- Test Authoring Style

```python
class TestStyle(str, Enum):
    KEYWORD = "keyword"         # Standard keyword-driven
    BDD = "bdd"                 # Given/When/Then prefixed
    DATA_DRIVEN = "data_driven" # [Template] with data rows
```

#### `TemplateSpec` -- Test Template Configuration

```python
@dataclass(frozen=True)
class TemplateSpec:
    keyword: str                    # template keyword name
    continue_on_failure: bool = True  # RF default for templates

    def __post_init__(self):
        if not self.keyword or not self.keyword.strip():
            raise ValueError("template keyword must not be empty")
```

#### `DataFormat` -- Data Source Format

```python
class DataFormat(str, Enum):
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"
    JSON = "json"
    PICT = "pict"
    GLOB = "glob"
```

#### `DataSource` -- External Data File Reference

```python
@dataclass(frozen=True)
class DataSource:
    file_path: str                    # absolute path to data file
    format: DataFormat                # CSV, XLSX, JSON, etc.
    reader_options: dict              # dialect, encoding, delimiter, sheet_name, etc.

    def __post_init__(self):
        if not self.file_path:
            raise ValueError("file_path must not be empty")

    @classmethod
    def from_path(cls, file_path: str, **options) -> "DataSource":
        """Detect format from file extension."""
        ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
        fmt_map = {'csv': DataFormat.CSV, 'xlsx': DataFormat.XLSX,
                   'xls': DataFormat.XLS, 'json': DataFormat.JSON,
                   'pict': DataFormat.PICT}
        return cls(file_path=file_path, format=fmt_map.get(ext, DataFormat.CSV),
                   reader_options=options)
```

#### `DataRow` -- Single Row of Test Data

```python
@dataclass(frozen=True)
class DataRow:
    test_name: str                     # "Login with admin and secret"
    arguments: dict[str, str]          # {"${username}": "admin", "${password}": "secret"}
    tags: tuple[str, ...]              # ("smoke", "regression")
    documentation: str = ""

    def __post_init__(self):
        if not self.test_name:
            raise ValueError("test_name must not be empty")
```

### 4.5 Entities

#### `ResolvedKeyword` -- Full Resolution Result

```python
@dataclass
class ResolvedKeyword:
    keyword_form: KeywordForm           # the normalized form
    keyword_info: object                # KeywordInfo from discovery (or None)
    library_name: Optional[str]         # resolved library name
    resolution_path: str                # "exact" | "embedded" | "fuzzy" | "rf_native"
    embedded_args: Optional[tuple]      # extracted embedded arg values, if any

    def to_dict(self) -> dict:
        return {
            "raw_name": self.keyword_form.raw_name,
            "resolved_name": self.keyword_form.normalized_name,
            "bdd_prefix": self.keyword_form.bdd_prefix.prefix if self.keyword_form.bdd_prefix else None,
            "library": self.library_name,
            "resolution_path": self.resolution_path,
            "embedded_args": list(self.embedded_args) if self.embedded_args else None,
        }
```

### 4.6 Aggregates

#### `KeywordResolver` -- Resolution Pipeline Orchestrator (Aggregate Root)

Orchestrates the three-stage resolution pipeline: BDD strip -> exact match -> embedded match.

```python
@dataclass
class KeywordResolver:
    bdd_service: BddPrefixService
    embedded_matcher: EmbeddedMatcher
    keyword_finder: KeywordFinderProtocol   # Protocol wrapping KeywordDiscovery

    def resolve(self, raw_name: str) -> ResolvedKeyword:
        """Resolve a raw keyword name through the full pipeline.

        Pipeline:
        1. Strip BDD prefix (Given/When/Then/And/But)
        2. Try exact keyword match on stripped name
        3. Try embedded argument regex match on stripped name
        4. Return ResolvedKeyword with full context
        """
        # Stage 1: BDD prefix stripping
        bdd_prefix = self.bdd_service.strip_prefix(raw_name)
        form = KeywordForm.from_raw(raw_name, bdd_prefix)

        # Stage 2: Exact match
        kw_info = self.keyword_finder.find(form.normalized_name)
        if kw_info is not None:
            return ResolvedKeyword(
                keyword_form=form,
                keyword_info=kw_info,
                library_name=getattr(kw_info, 'library', None),
                resolution_path="exact",
                embedded_args=None,
            )

        # Stage 3: Embedded argument match
        match = self.embedded_matcher.match(form.normalized_name)
        if match is not None:
            form = form.with_embedded_match(match)
            return ResolvedKeyword(
                keyword_form=form,
                keyword_info=match.keyword_info,
                library_name=getattr(match.keyword_info, 'library', None),
                resolution_path="embedded",
                embedded_args=match.extracted_args,
            )

        # Stage 4: No match found -- return partial result for RF native fallback
        return ResolvedKeyword(
            keyword_form=form,
            keyword_info=None,
            library_name=None,
            resolution_path="rf_native",
            embedded_args=None,
        )
```

**Invariant**: BDD stripping always runs first. Embedded matching only runs if exact match fails. This matches RF's own resolution order.

#### `DataDrivenSuite` -- Template + Data for Suite Generation

```python
@dataclass
class DataDrivenSuite:
    template: TemplateSpec
    data_rows: list[DataRow]
    data_source: Optional[DataSource]   # None if data was provided inline

    def add_row(self, row: DataRow) -> None:
        self.data_rows.append(row)

    @property
    def row_count(self) -> int:
        return len(self.data_rows)

    def to_dict(self) -> dict:
        return {
            "template_keyword": self.template.keyword,
            "continue_on_failure": self.template.continue_on_failure,
            "row_count": self.row_count,
            "data_source": self.data_source.file_path if self.data_source else None,
        }
```

### 4.7 Domain Services

#### `BddPrefixService` -- Strip BDD Prefixes

```python
class BddPrefixService:
    """Strips BDD prefixes using RF's Languages class.

    English-only initially. Extensible to localized prefixes via
    Languages(language_code) in future phases.
    """

    ENGLISH_PREFIXES: ClassVar[frozenset] = frozenset({
        "Given", "When", "Then", "And", "But"
    })

    _PREFIX_MAP: ClassVar[dict] = {
        "given": BddPrefixType.GIVEN,
        "when": BddPrefixType.WHEN,
        "then": BddPrefixType.THEN,
        "and": BddPrefixType.AND,
        "but": BddPrefixType.BUT,
    }

    def __init__(self):
        from robot.conf.languages import Languages
        lang = Languages()
        self._regex = lang.bdd_prefix_regexp

    def strip_prefix(self, name: str) -> Optional[BddPrefix]:
        """Strip BDD prefix from keyword name.

        Returns BddPrefix if a prefix was found, None otherwise.
        Uses RF's own compiled regex for consistency.
        """
        match = self._regex.match(name)
        if not match:
            return None
        prefix_text = name[:match.end()].strip()
        stripped = name[match.end():]
        prefix_type = self._PREFIX_MAP.get(prefix_text.lower())
        if prefix_type is None:
            return None
        return BddPrefix(
            original_name=name,
            stripped_name=stripped,
            prefix=prefix_text,
            prefix_type=prefix_type,
        )
```

**Key design decision**: Uses RF's `Languages` class for the prefix regex rather than hardcoding. This ensures consistency with RF's own stripping behavior and enables future localization support.

#### `EmbeddedMatcher` -- Regex-Based Embedded Argument Matching

```python
class EmbeddedMatcher:
    """Matches concrete keyword calls against embedded argument patterns.

    Mirrors RF's KeywordCache split: normal keywords use dict lookup,
    embedded keywords use linear regex scan.
    """

    def __init__(self):
        self._patterns: list[tuple[EmbeddedPattern, object]] = []  # (pattern, KeywordInfo)

    def register(self, template_name: str, keyword_info: object) -> None:
        """Register an embedded argument keyword pattern.

        Called during keyword discovery when a keyword name contains '${'.
        Uses RF's EmbeddedArguments.from_name() for parsing.
        """
        from robot.running.arguments.embedded import EmbeddedArguments
        ea = EmbeddedArguments.from_name(template_name)
        if ea is None:
            return
        pattern = EmbeddedPattern.from_rf(ea, template_name)
        self._patterns.append((pattern, keyword_info))

    def match(self, name: str) -> Optional[EmbeddedMatch]:
        """Match a concrete keyword name against all registered patterns.

        Returns the first matching EmbeddedMatch, or None.
        O(n) linear scan -- same as RF's own approach.
        """
        for pattern, kw_info in self._patterns:
            if pattern.matches(name):
                extracted = pattern.parse_args(name)
                return EmbeddedMatch(
                    pattern=pattern,
                    extracted_args=extracted,
                    keyword_info=kw_info,
                )
        return None

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    def clear(self) -> None:
        self._patterns.clear()
```

**Key design decision**: Uses RF's `EmbeddedArguments` class directly -- does not reimplement regex parsing. The class is stable, well-tested, and handles edge cases (custom patterns, type hints, whitespace flexibility).

#### `TemplateRenderer` -- Generate `[Template]` Output

```python
class TemplateRenderer:
    """Renders data-driven test cases with [Template] setting."""

    def render_template_test(
        self,
        test_name: str,
        template: TemplateSpec,
        steps: list,  # list of ExecutionStep or similar
        tags: tuple[str, ...] = (),
        setup: Optional[dict] = None,
        teardown: Optional[dict] = None,
    ) -> str:
        """Generate RF text for a data-driven test case.

        Returns .robot text fragment with [Template] and data rows.
        """
        lines = [test_name]
        if tags:
            lines.append(f"    [Tags]    {'    '.join(tags)}")
        if setup:
            lines.append(f"    [Setup]    {setup['keyword']}    {'    '.join(setup.get('arguments', []))}")
        lines.append(f"    [Template]    {template.keyword}")
        for step in steps:
            # Data-driven: render only arguments, no keyword name
            escaped_args = [self._escape_rf_arg(a) for a in step.arguments]
            lines.append(f"    {'    '.join(escaped_args)}")
        if teardown:
            lines.append(f"    [Teardown]    {teardown['keyword']}    {'    '.join(teardown.get('arguments', []))}")
        return '\n'.join(lines)

    def render_embedded_template_test(
        self,
        test_name: str,
        template: TemplateSpec,
        data_rows: list[DataRow],
    ) -> str:
        """Generate RF text for embedded-argument template tests.

        For templates like 'Login With ${user} And ${password}',
        data values are substituted into the keyword name.
        """
        lines = [test_name]
        lines.append(f"    [Template]    {template.keyword}")
        for row in data_rows:
            # Extract positional arg values (strip ${} wrappers from keys)
            values = list(row.arguments.values())
            lines.append(f"    {'    '.join(values)}")
        return '\n'.join(lines)

    @staticmethod
    def _escape_rf_arg(value: str) -> str:
        """Escape special RF characters in argument values."""
        if not value:
            return "${EMPTY}"
        return value
```

#### `DataSourceLoader` -- Load External Test Data

```python
class DataSourceLoaderProtocol(Protocol):
    def load(self, source: DataSource) -> list[DataRow]: ...

class DataSourceLoader:
    """Loads test data from external files using DataDriver's reader classes.

    Gracefully degrades if DataDriver is not installed.
    """

    def load(self, source: DataSource) -> list[DataRow]:
        """Load test data from a DataSource.

        Returns list of DataRow value objects.
        Raises ImportError if DataDriver is not installed.
        """
        try:
            from DataDriver.ReaderConfig import ReaderConfig
        except ImportError:
            raise ImportError(
                "DataDriver is not installed. Install with: pip install robotframework-datadriver"
            )

        reader_map = {
            DataFormat.CSV: 'DataDriver.csv_reader.csv_reader',
            DataFormat.JSON: 'DataDriver.json_reader.json_reader',
            DataFormat.XLSX: 'DataDriver.xlsx_reader.xlsx_reader',
            DataFormat.XLS: 'DataDriver.xls_reader.xls_reader',
        }

        reader_path = reader_map.get(source.format)
        if reader_path is None:
            raise ValueError(f"Unsupported data format: {source.format}")

        config = ReaderConfig(file=source.file_path, **source.reader_options)

        module_path, class_name = reader_path.rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_path)
        reader_cls = getattr(module, class_name)
        reader = reader_cls(config)

        raw_data = reader.get_data_from_source()
        return [
            DataRow(
                test_name=tc.test_case_name,
                arguments=dict(tc.arguments),
                tags=tuple(tc.tags) if tc.tags else (),
                documentation=tc.documentation or "",
            )
            for tc in raw_data
        ]

    @staticmethod
    def is_available() -> bool:
        """Check if DataDriver is installed."""
        try:
            import DataDriver
            return True
        except ImportError:
            return False
```

#### Protocol Definitions

```python
class KeywordFinderProtocol(Protocol):
    """Protocol for keyword lookup -- wraps KeywordDiscovery."""
    def find(self, name: str) -> Optional[object]: ...

class EventPublisher(Protocol):
    """Protocol for domain event publication."""
    def publish(self, event: object) -> None: ...
```

### 4.8 Domain Events

All events follow the `@dataclass(frozen=True)` + `to_dict()` with `"event_type"` key pattern (per intent/batch_execution/recovery domains).

```python
@dataclass(frozen=True)
class BddPrefixStripped:
    original_name: str
    stripped_name: str
    prefix: str
    prefix_type: str  # BddPrefixType value

    def to_dict(self) -> dict:
        return {
            "event_type": "BddPrefixStripped",
            "original_name": self.original_name,
            "stripped_name": self.stripped_name,
            "prefix": self.prefix,
            "prefix_type": self.prefix_type,
        }

@dataclass(frozen=True)
class EmbeddedArgMatched:
    call_name: str          # "Select dog from list"
    template_name: str      # "Select ${animal} from list"
    extracted_args: tuple    # ("dog",)
    library: str             # "MyLibrary"

    def to_dict(self) -> dict:
        return {
            "event_type": "EmbeddedArgMatched",
            "call_name": self.call_name,
            "template_name": self.template_name,
            "extracted_args": list(self.extracted_args),
            "library": self.library,
        }

@dataclass(frozen=True)
class TemplateApplied:
    test_name: str
    template_keyword: str
    row_count: int

    def to_dict(self) -> dict:
        return {
            "event_type": "TemplateApplied",
            "test_name": self.test_name,
            "template_keyword": self.template_keyword,
            "row_count": self.row_count,
        }

@dataclass(frozen=True)
class DataSourceLoaded:
    file_path: str
    format: str             # DataFormat value
    row_count: int

    def to_dict(self) -> dict:
        return {
            "event_type": "DataSourceLoaded",
            "file_path": self.file_path,
            "format": self.format,
            "row_count": self.row_count,
        }

@dataclass(frozen=True)
class KeywordResolutionFailed:
    raw_name: str
    normalized_name: str
    attempted_strategies: tuple  # ("exact", "embedded")
    bdd_prefix_stripped: bool

    def to_dict(self) -> dict:
        return {
            "event_type": "KeywordResolutionFailed",
            "raw_name": self.raw_name,
            "normalized_name": self.normalized_name,
            "attempted_strategies": list(self.attempted_strategies),
            "bdd_prefix_stripped": self.bdd_prefix_stripped,
        }
```

---

## 5. Integration Points

### 5.1 server.py Changes

#### `execute_step` -- BDD Prefix Stripping at Entry

Strip BDD prefix at the MCP tool boundary. All downstream code sees the plain keyword name:

```python
# At execute_step entry (before any keyword resolution):
from robotmcp.domains.keyword_resolution import BddPrefixService

bdd_service = BddPrefixService()
bdd_prefix = bdd_service.strip_prefix(keyword)
if bdd_prefix:
    keyword = bdd_prefix.stripped_name
    # Preserve prefix info in response
```

The BDD prefix is included in the response metadata:
```python
result["bdd_prefix"] = bdd_prefix.prefix if bdd_prefix else None
```

#### `find_keywords` -- BDD Stripping + Embedded Metadata

Strip BDD prefix from the query before searching. Annotate embedded argument keywords in results:

```python
# In find_keywords:
query, _ = bdd_service.strip_prefix(query) or (query, None)

# In result formatting, for keywords with embedded args:
kw_result["embedded_args"] = True
kw_result["embedded_pattern"] = "Select <animal> from list"
kw_result["usage_example"] = "Select dog from list"
```

#### `get_keyword_info` -- Support Lookup by Concrete Embedded Call

When `get_keyword_info(keyword="Select dog from list")` is called and exact lookup fails, try embedded matching and return the template keyword's info with extracted arg values.

#### `manage_session(start_test)` -- Add `template` Parameter

```python
# New optional parameter on start_test action:
manage_session(
    action="start_test",
    session_id="s1",
    test_name="Login Validation",
    template="Should Be Equal",  # NEW: declares data-driven test
)
```

#### `build_test_suite` -- Add `style` Parameter

```python
build_test_suite(
    test_name="My Suite",
    session_id="s1",
    style="bdd",         # NEW: "keyword" | "bdd" | "data_driven"
)
```

- `"keyword"` (default): Current behavior, explicit keyword calls.
- `"bdd"`: Preserve BDD prefixes recorded during execution in output.
- `"data_driven"`: Emit `[Template]` + data rows for template-marked tests.

#### `run_test_suite` -- Companion File Handling

Fix `SuiteExecutionService` to copy companion files alongside `.robot` to temp directory, or execute in the original directory.

#### New Tool: `load_test_data` (disabled by default)

```python
@mcp.tool(**DISABLED_TOOL_KWARGS)
async def load_test_data(
    file_path: str,
    session_id: str = "",
    encoding: str = "utf-8",
    dialect: str = "Excel-EU",
    delimiter: str = ";",
    sheet_name: str = "0",
) -> dict:
    """Load test data from external file (CSV, Excel, JSON) for data-driven testing.

    Uses DataDriver's reader classes. Returns structured test case data.
    """
```

### 5.2 KeywordDiscovery Changes

Add `embedded_keywords` list mirroring RF's `KeywordCache` split:

```python
class KeywordDiscovery:
    def __init__(self):
        self._cache = {}                 # existing: lowercase name -> KeywordInfo
        self._embedded_keywords = []     # NEW: list of (EmbeddedPattern, KeywordInfo)

    def add_keywords_to_cache(self, keywords):
        for kw_info in keywords:
            if '${' in kw_info.name:
                # Register as embedded pattern
                from robot.running.arguments.embedded import EmbeddedArguments
                ea = EmbeddedArguments.from_name(kw_info.name)
                if ea is not None:
                    pattern = EmbeddedPattern.from_rf(ea, kw_info.name)
                    self._embedded_keywords.append((pattern, kw_info))
            # Also add to normal cache (for template name lookup)
            self._cache[kw_info.name.lower()] = kw_info

    def find_keyword(self, name):
        # Stage 1: Exact match (existing)
        result = self._cache.get(name.lower())
        if result:
            return result

        # Stage 2: Embedded match (NEW)
        for pattern, kw_info in self._embedded_keywords:
            if pattern.matches(name):
                return kw_info  # Caller can get extracted args separately

        # Stage 3: Fuzzy match (existing fallback)
        ...
```

The resolution chain becomes: exact -> embedded -> fuzzy.

### 5.3 keyword_executor Changes

Use `KeywordResolver.resolve()` instead of direct `find_keyword()`:

```python
# In _execute_keyword_internal:
resolved = keyword_resolver.resolve(keyword)

# Use normalized name for pre-validation, timeout, plugin lookups:
effective_keyword = resolved.keyword_form.normalized_name

# Pass extracted embedded args to run_keyword:
if resolved.embedded_args:
    # Prepend embedded args to any additional arguments (RF convention)
    all_args = list(resolved.embedded_args) + list(arguments)
else:
    all_args = list(arguments)
```

### 5.4 test_builder Changes

```python
@dataclass
class GeneratedTestCase:
    name: str
    steps: list
    template: Optional[str] = None  # NEW: template keyword for data-driven
    style: TestStyle = TestStyle.KEYWORD  # NEW: rendering style
    # ... existing fields ...
```

In `_generate_rf_text()`:

```python
if test_case.template:
    lines.append(f"    [Template]    {test_case.template}")
    for step in test_case.steps:
        # Data-driven: render only arguments, no keyword name
        escaped_args = [self._escape_robot_argument(arg) for arg in step.arguments]
        lines.append(f"    {'    '.join(escaped_args)}")
elif test_case.style == TestStyle.BDD:
    for step in test_case.steps:
        # BDD: preserve prefix if recorded
        prefix = getattr(step, 'bdd_prefix', None) or ""
        kw_name = f"{prefix} {step.keyword}" if prefix else step.keyword
        line = await self._render_step_with_name(kw_name, step)
        lines.append(line)
else:
    # Existing keyword-style rendering
    ...
```

In `_create_rf_suite()`:

```python
if test_case.template:
    rf_test.template = test_case.template
```

### 5.5 suite_execution_service Changes

Fix companion file handling for DataDriver:

```python
def _create_temp_suite_file(self, suite_content, source_path=None):
    # ... existing temp dir creation ...

    # NEW: Copy companion data files if source_path is provided
    if source_path:
        source_dir = os.path.dirname(source_path)
        data_extensions = {'.csv', '.xlsx', '.xls', '.json', '.pict'}
        for file in os.listdir(source_dir):
            if any(file.lower().endswith(ext) for ext in data_extensions):
                src = os.path.join(source_dir, file)
                dst = os.path.join(temp_dir, file)
                shutil.copy2(src, dst)
```

Alternative (preferred for security): execute in the original directory with output sandboxing.

### 5.6 MCP Instructions Update

Add BDD style guidance, data-driven testing workflow, and embedded argument examples to the MCP server instructions (appended when relevant features are detected):

```
## BDD Style Keywords
You can use BDD prefixes (Given/When/Then/And/But) with any keyword.
The prefix is stripped before keyword resolution.
Example: execute_step(keyword="Given Open Browser", arguments=["https://example.com", "chromium"])

## Embedded Argument Keywords
Some keywords accept arguments embedded in the keyword name.
Example: execute_step(keyword="Select dog from list")
This calls the keyword "Select ${animal} from list" with animal="dog".

## Data-Driven Testing
1. Declare a data-driven test: manage_session(action="start_test", test_name="Login Tests", template="Verify Login")
2. Execute data rows: execute_step(keyword="Verify Login", arguments=["admin", "secret"])
3. Generate suite: build_test_suite(test_name="Suite", style="data_driven")
```

---

## 6. Phased Implementation Plan

### Phase 1: BDD Prefix Stripping (~80 lines production code)

**Deliverables**:
- `BddPrefixService` + `BddPrefix` + `BddPrefixType` value objects
- Entry-point stripping in `execute_step` and `find_keywords` in server.py
- `BddPrefixStripped` event
- BDD prefix info in response metadata

**Estimated tests**: ~30
- Value object invariants (empty name, prefix mismatch)
- Stripping all 5 English prefixes (case-insensitive)
- No-match cases (no prefix, no space, non-BDD word)
- Integration with `execute_step` entry point
- Integration with `find_keywords` entry point

### Phase 2: Embedded Argument Matching (~300 lines)

**Deliverables**:
- `EmbeddedPattern`, `EmbeddedMatch` value objects
- `EmbeddedMatcher` service using RF's `EmbeddedArguments`
- `KeywordDiscovery` embedded cache + matching in `find_keyword()`
- Enhanced `find_keywords` and `get_keyword_info` responses with embedded metadata
- `EmbeddedArgMatched` event
- `KeywordResolutionFailed` event

**Estimated tests**: ~50
- EmbeddedPattern creation from RF's EmbeddedArguments
- Match/no-match cases (standard, custom regex, multiple patterns)
- Argument extraction correctness
- KeywordDiscovery embedded cache population
- find_keyword resolution chain (exact -> embedded -> fuzzy)
- find_keywords metadata enrichment
- BDD prefix + embedded arg combined resolution

### Phase 3: Data-Driven Templates (~150 lines)

**Deliverables**:
- `TemplateSpec`, `TestStyle` value objects
- `template` field on `TestInfo` and `GeneratedTestCase`
- `TemplateRenderer` service
- `_generate_rf_text()` template rendering
- `manage_session(start_test)` template parameter
- `_create_rf_suite()` template support
- `TemplateApplied` event

**Estimated tests**: ~40
- TemplateSpec invariants (empty keyword)
- Template rendering with data rows
- Template rendering with embedded arguments
- Mixed suite (some template, some keyword-style)
- RF API suite template attribute
- manage_session template parameter flow

### Phase 4: DataDriver Integration (~200 lines)

**Deliverables**:
- `DataSource`, `DataRow`, `DataFormat` value objects
- `DataSourceLoader` service
- Suite execution companion file fix
- `load_test_data` MCP tool (disabled by default)
- `DataSourceLoaded` event

**Estimated tests**: ~40
- DataSource format detection from extension
- DataRow invariants (empty test_name)
- DataSourceLoader with CSV (requires DataDriver installed)
- DataSourceLoader with JSON
- DataSourceLoader graceful degradation (DataDriver not installed)
- Suite execution with companion files
- load_test_data tool end-to-end

### Phase 5: Full Pipeline (~100 lines)

**Deliverables**:
- `KeywordResolver` aggregate (orchestrates BDD + embedded + exact)
- `KeywordForm` value object
- `ResolvedKeyword` entity
- `build_test_suite` `style` parameter
- BDD prefix preservation in output (`style="bdd"`)
- `DataDrivenSuite` aggregate
- MCP instruction updates
- Container registration (lazy singletons for BddPrefixService, EmbeddedMatcher)

**Estimated tests**: ~30
- KeywordResolver pipeline (BDD strip -> exact -> embedded -> rf_native)
- KeywordForm immutable updates
- ResolvedKeyword serialization
- build_test_suite with style="bdd"
- build_test_suite with style="data_driven"
- DataDrivenSuite lifecycle

**Total estimated**: ~830 lines production code, ~190 tests

---

## 7. Test Strategy

| Level | Scope | Test Count | Approach |
|-------|-------|------------|----------|
| Unit | Value objects (frozen, invariants, `__post_init__`) | ~40 | Pytest, no mocks |
| Unit | BddPrefixService (all prefixes, edge cases) | ~15 | Pytest, RF Languages mock optional |
| Unit | EmbeddedMatcher (match/no-match, extraction) | ~25 | Pytest, uses real RF EmbeddedArguments |
| Unit | TemplateRenderer (RF text output) | ~15 | Pytest, string comparison |
| Unit | DataSourceLoader (CSV, JSON, missing DataDriver) | ~20 | Pytest, temp files |
| Unit | KeywordResolver aggregate (pipeline order) | ~15 | Pytest, mock KeywordFinderProtocol |
| Integration | execute_step with BDD/embedded keywords | ~20 | Pytest-asyncio, mock RF context |
| Integration | build_test_suite with template/BDD output | ~15 | Pytest-asyncio, verify RF text |
| Integration | KeywordDiscovery embedded cache | ~15 | Pytest, real KeywordDiscovery |
| E2E | DataDriver CSV -> run_test_suite | ~5 | Pytest, requires DataDriver installed |
| E2E | BDD + embedded args end-to-end | ~5 | Pytest, requires RF context |

**Test file layout**:
```
tests/unit/domains/keyword_resolution/
    test_value_objects.py        # ~40 tests
    test_services.py             # ~50 tests (BddPrefixService, EmbeddedMatcher, etc.)
    test_aggregates.py           # ~20 tests (KeywordResolver, DataDrivenSuite)
    test_events.py               # ~10 tests
    test_template_renderer.py    # ~15 tests
    test_data_source_loader.py   # ~20 tests

tests/integration/
    test_bdd_embedded_integration.py   # ~25 tests
    test_data_driven_integration.py    # ~10 tests
```

---

## 8. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| BDD prefix stripping at entry point misses a code path | Medium | High | Phase 1 validates the 10+ locations. If entry-point stripping is insufficient for edge cases, Phase 5 adds deeper integration points (Approach A from BDD research) |
| Embedded arg regex matching is O(n) per lookup | Low | Low | Same as RF's own approach. Typical libraries have <10 embedded keywords. Profile if >100 patterns observed |
| DataDriver not installed in user environment | Medium | Medium | `DataSourceLoader.is_available()` check + graceful error message. `load_test_data` tool disabled by default |
| RF `EmbeddedArguments` API changes in RF 8.x | Low | Medium | Pin to RF 7.x API. `EmbeddedArguments.from_name()` has been stable since RF 4.x |
| Template keyword name normalization (qualified vs unqualified) | Medium | Low | Use existing `remove_library_prefixes` utility in test_builder. Normalize before comparison |
| Companion file copying exposes sensitive files | Low | Medium | Only copy files with known data extensions (`.csv`, `.xlsx`, `.xls`, `.json`, `.pict`). Alternative: execute in original directory with output sandboxing |
| BDD prefix conflicts with keyword names starting with "And", "But", etc. | Low | Low | RF's own regex requires trailing space: `"And"` alone does not match. `"Android"` does not match. Verified in Experiment 2 |
| Large data files overwhelm LLM context in `load_test_data` response | Medium | Medium | Add `limit` parameter (default 100 rows). Paginate or summarize for larger datasets |

---

## 9. Alternatives Considered

### 9.1 Extend the Intent Domain (Rejected)

**Considered**: Adding BDD prefix handling and embedded argument matching to the existing Intent domain (ADR-007).

**Rejected because**:
- The Intent domain maps abstract verbs to concrete keywords (CLICK -> Click Element). Keyword resolution transforms concrete names through BDD/embedded pipelines. These are different concerns operating in different directions.
- The Intent domain is static (mapping tables); keyword resolution is dynamic (depends on loaded libraries' embedded keywords).
- Mixing both would violate the single-responsibility principle and make the Intent domain harder to reason about.

### 9.2 Monolithic Approach -- All Changes in server.py (Rejected)

**Considered**: Adding BDD stripping, embedded matching, and template support directly in server.py functions without a domain layer.

**Rejected because**:
- server.py is already 4000+ lines. Adding 800+ lines of resolution logic would worsen maintainability.
- No reuse across tools -- the same BDD stripping logic would be duplicated in `execute_step`, `find_keywords`, `get_keyword_info`, etc.
- No testability -- domain logic buried in MCP tool handlers is hard to unit test.
- Inconsistent with established DDD patterns (ADR-001).

### 9.3 New MCP Tool per Feature (Rejected)

**Considered**: Adding `execute_bdd_step`, `execute_embedded_step`, `execute_template_test` as separate tools.

**Rejected because**:
- Increases API surface area (already 20+ tools).
- Forces the LLM to choose the "right" tool variant, which is exactly the kind of decision small LLMs struggle with.
- BDD/embedded/template features should be transparent -- the LLM passes `"Given Select dog from list"` to `execute_step` and it just works.

### 9.4 Full BDD-Aware Pipeline from Day One (Deferred)

**Considered**: Making every internal component (pre-validation, timeout injection, plugin overrides, etc.) BDD-aware simultaneously.

**Deferred because**:
- Entry-point stripping (Phase 1) covers 95% of use cases with 2-3 lines of code per tool.
- Deeper integration can be added incrementally if edge cases are discovered.
- Aligns with the research recommendation of Approach B + selective elements of C.

---

## 10. File Layout

```
src/robotmcp/domains/
    keyword_resolution/
        __init__.py              # Public API (~20 exports)
        value_objects.py         # BddPrefixType, BddPrefix, EmbeddedPattern,
                                 # EmbeddedMatch, KeywordForm, TestStyle,
                                 # TemplateSpec, DataFormat, DataSource, DataRow
        entities.py              # ResolvedKeyword
        aggregates.py            # KeywordResolver (root), DataDrivenSuite
        services.py              # BddPrefixService, EmbeddedMatcher,
                                 # TemplateRenderer, DataSourceLoader
                                 # + Protocols: KeywordFinderProtocol, EventPublisher
        events.py                # 5 events: BddPrefixStripped, EmbeddedArgMatched,
                                 # TemplateApplied, DataSourceLoaded,
                                 # KeywordResolutionFailed

Tests:
tests/unit/domains/keyword_resolution/
    test_value_objects.py        # ~40 tests: BddPrefix invariants, EmbeddedPattern
                                 #            creation, KeywordForm, TemplateSpec,
                                 #            DataSource format detection, DataRow
    test_services.py             # ~50 tests: BddPrefixService stripping,
                                 #            EmbeddedMatcher match/parse,
                                 #            DataSourceLoader CSV/JSON
    test_aggregates.py           # ~20 tests: KeywordResolver pipeline,
                                 #            DataDrivenSuite lifecycle
    test_events.py               # ~10 tests: Event serialization, to_dict()
    test_template_renderer.py    # ~15 tests: RF text output, embedded templates
    test_data_source_loader.py   # ~20 tests: CSV/JSON/graceful degradation

tests/integration/
    test_bdd_embedded_integration.py   # ~25 tests: execute_step BDD, find_keywords
                                       #            embedded, KeywordDiscovery cache
    test_data_driven_integration.py    # ~10 tests: build_test_suite template output,
                                       #            run_test_suite companion files

Estimated: ~190 new tests
```

---

## 11. Related ADRs

| ADR | Relationship |
|-----|-------------|
| **ADR-001** (DDD Architecture) | Foundational patterns: frozen value objects, Protocol services, event conventions |
| **ADR-005** (Multi-Test Sessions) | `TestRegistry` and `TestInfo` are prerequisites for per-test template support |
| **ADR-007** (Intent Layer) | Orthogonal -- Intent maps abstract verbs to keywords; keyword_resolution transforms concrete names. No dependency |
| **ADR-011** (Batch Execution) | `execute_batch` can be used for data-driven row execution. Template-aware batch execution is a Phase 3 extension |
| **ADR-012** (PlatynUI Desktop) | Reference for plugin integration pattern. PlatynUI keywords do not use embedded arguments |
| **ADR-015** (Artifact Externalization) | `load_test_data` responses for large datasets should use externalization |
| **ADR-016** (Slim Profiles) | `load_test_data` tool visibility controlled by profile selection |
