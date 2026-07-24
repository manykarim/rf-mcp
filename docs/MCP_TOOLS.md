# rf-mcp MCP Tool Reference

rf-mcp (RobotMCP) is an MCP server that lets an AI coding agent *drive* Robot
Framework: discover keywords, run steps live against Browser, SeleniumLibrary,
AppiumLibrary, RequestsLibrary, DatabaseLibrary, or PlatynUI, inspect the DOM and
session state, and build real `.robot` suites from what actually worked.

One thing to keep straight: **rf-mcp is not a Robot Framework library.** It ships
no keywords of its own. Its "keywords" are the MCP tools below — the verbs your
agent calls to make Robot Framework do the work. Think of this page as the libdoc
for that toolset.

The tools sort into nine groups. The happy path runs left to right: analyze a
scenario, pick libraries, open a session, discover the right keywords, execute
step by step, inspect state when something breaks, then build and run a suite.

- [Tool index](#tool-index)
- [Conventions](#conventions)
- [Planning & Analysis](#planning--analysis)
- [Session & Execution](#session--execution)
- [Discovery & Documentation](#discovery--documentation)
- [Locators & Guidance](#locators--guidance)
- [State & Observability](#state--observability)
- [Suite Lifecycle](#suite-lifecycle)
- [Artifacts](#artifacts)
- [Library & Plugin Management](#library--plugin-management)
- [Memory](#memory-optional)

---

## Tool index

| Tool | One-liner |
|------|-----------|
| **Planning & Analysis** | |
| [`analyze_scenario`](#analyze_scenario) | Turn a natural-language scenario into structured intent and a session. First call, always. |
| [`recommend_libraries`](#recommend_libraries) | Suggest the Robot Framework libraries a scenario needs. |
| [`check_library_availability`](#check_library_availability) | Verify named libraries can be imported, with install guidance. |
| [`validate_scenario`](#validate_scenario) | Sanity-check a parsed scenario for feasibility before executing. |
| [`suggest_next_step`](#suggest_next_step) | Ask for an AI-suggested next test step given current state. |
| **Session & Execution** | |
| [`manage_session`](#manage_session) | The session Swiss Army knife: init, import, variables, multi-test, profiles. |
| [`execute_step`](#execute_step) | Run a single keyword (or an Evaluate expression) in a session. |
| [`execute_batch`](#execute_batch) | Run many keywords in one call with variable chaining and recovery. |
| [`resume_batch`](#resume_batch) | Restart a failed batch from the failure point, optionally with fixes. |
| [`execute_flow`](#execute_flow) | Run an `if` / `for` / `try` control structure inside a session. |
| [`intent_action`](#intent_action) | Library-agnostic verbs (click, fill, extract…) that resolve to the right keyword. |
| [`evaluate_expression`](#evaluate_expression) | Evaluate a Python expression in the RF context (`BuiltIn.Evaluate`). |
| [`set_variables`](#set_variables) | Set several RF variables at once. |
| [`set_library_search_order`](#set_library_search_order) | Set keyword-resolution precedence for a session. |
| [`initialize_context`](#initialize_context) | Seed a session with libraries and variables. |
| `execute_if` | Legacy single-purpose `if` flow (superseded by `execute_flow`). |
| `execute_for_each` | Legacy single-purpose `for` flow (superseded by `execute_flow`). |
| `execute_try_except` | Legacy single-purpose `try` flow (superseded by `execute_flow`). |
| `import_resource` | Import a `.resource` file into the session RF namespace (attach-aware). |
| `import_custom_library` | Import a custom library by module name or file path (attach-aware). |
| **Discovery & Documentation** | |
| [`find_keywords`](#find_keywords) | Discover keywords by semantic / pattern / catalog / session strategy. |
| [`discover_keywords`](#discover_keywords) | Find keywords matching an action description. |
| [`search_keywords`](#search_keywords) | Native libdoc search across names, docs, and tags. |
| [`get_keyword_info`](#get_keyword_info) | One endpoint for keyword/library docs and signature parsing. |
| [`get_keyword_documentation`](#get_keyword_documentation) | Full libdoc for a single keyword. |
| [`get_library_documentation`](#get_library_documentation) | Full libdoc for a library and all its keywords. |
| [`get_available_keywords`](#get_available_keywords) | List keywords with minimal metadata. |
| [`get_loaded_libraries`](#get_loaded_libraries) | Status of every loaded library. |
| [`get_library_status`](#get_library_status) | Install status for one library. |
| `list_available_keywords` | List keywords from the session's RF namespace (attach-aware). |
| `get_session_keyword_documentation` | Docs for a keyword available in the session RF namespace. |
| `debug_parse_keyword_arguments` | Parse an argument list against a keyword's signature. |
| **Locators & Guidance** | |
| [`get_locator_guidance`](#get_locator_guidance) | Consolidated locator/API/visual cookbook for the target library. |
| [`get_browser_locator_guidance`](#get_browser_locator_guidance) | Browser Library (Playwright) selector guidance. |
| [`get_selenium_locator_guidance`](#get_selenium_locator_guidance) | SeleniumLibrary locator strategy guidance. |
| [`get_appium_locator_guidance`](#get_appium_locator_guidance) | AppiumLibrary locator strategy guidance. |
| **State & Observability** | |
| [`get_session_state`](#get_session_state) | Aggregated session insight: DOM, variables, validation, libraries, UI tree. |
| [`get_session_info`](#get_session_info) | Configuration and state summary for a session. |
| [`get_session_validation_status`](#get_session_validation_status) | Per-step validation status for a session. |
| [`get_application_state`](#get_application_state) | Current app state (DOM / API / database). |
| [`get_page_source`](#get_page_source) | Page source / DOM for a browser session, optionally filtered. |
| [`get_context_variables`](#get_context_variables) | All variables from a session. |
| [`validate_test_readiness`](#validate_test_readiness) | Check whether a session is ready to build a suite. |
| [`visual_check`](#visual_check) | Capture a screenshot for visual validation (path by default, image on opt-in). |
| `diagnose_rf_context` | Inspect RF context state: libraries, search order, variable count. |
| **Suite Lifecycle** | |
| [`build_test_suite`](#build_test_suite) | Generate a `.robot` suite from executed steps. |
| [`run_test_suite`](#run_test_suite) | Validate (dry run) or execute a suite. |
| [`run_test_suite_dry`](#run_test_suite_dry) | Dry-run validation of a suite's syntax and structure. |
| [`load_test_data`](#load_test_data) | Load external CSV/Excel/JSON data for data-driven tests. |
| **Artifacts** | |
| [`fetch_artifact`](#fetch_artifact) | Retrieve externalized large output by artifact ID. |
| **Library & Plugin Management** | |
| [`manage_library_plugins`](#manage_library_plugins) | List / reload / diagnose library plugins from one endpoint. |
| [`list_library_plugins`](#list_library_plugins) | Summary of every loaded plugin. |
| [`diagnose_library_plugin`](#diagnose_library_plugin) | Detailed info on one plugin. |
| [`reload_library_plugins`](#reload_library_plugins) | Reload plugins, optionally from manifest paths. |
| [`manage_attach`](#manage_attach) | Inspect or control the debug attach bridge. |
| `attach_status` | Report attach-mode configuration and bridge health. |
| `attach_stop_bridge` | Send a stop command to the external attach bridge. |
| **Memory** *(optional — `rf-mcp[memory]`)* | |
| [`recall_step`](#recall_step) | Recall proven step sequences for a scenario. |
| [`recall_fix`](#recall_fix) | Recall known fixes for an error. |
| [`recall_locator`](#recall_locator) | Recall working locators for an element. |
| [`store_knowledge`](#store_knowledge) | Store reusable domain knowledge. |
| [`get_memory_status`](#get_memory_status) | Check memory availability and stats. |

---

## Conventions

A few facts hold across the whole toolset. Learn them once.

- **`session_id` is the thread that ties a run together.** `analyze_scenario`
  (or `manage_session(action="init")`) hands you one; pass it to every subsequent
  call. Lose it and you lose the browser, the variables, and the recorded steps.
- **Discover before you execute.** Do not guess keyword names. `find_keywords`
  and `get_keyword_info` exist precisely so `execute_step` doesn't fail on a
  keyword that never existed (`Press Button`, `Verify`, `Validate Json` — none
  are real).
- **Type-constrained parameters.** `action`, `mode`, `strategy`, `context`,
  `intent`, and similar arguments are `Literal` enums. Values are
  case-insensitive; anything off the list is rejected rather than guessed.
- **Automatic coercion.** JSON-stringified arrays (`"[\"Browser\"]"`) and
  comma-separated strings (`"Browser,BuiltIn"`) are parsed into lists
  server-side, so a small model that fumbles list syntax still gets through.
- **`detail_level` / `mode` trim tokens.** Where a tool offers
  `detail_level="minimal"|"standard"|"full"`, `minimal` is the terse default and
  `full` is the whole story. Large payloads may be externalized — see
  [`fetch_artifact`](#fetch_artifact).
- **Most responses carry `success: bool`,** plus `error` and (usually)
  `guidance` when something goes wrong.

---

## Planning & Analysis

Where a run begins. Parse the intent, choose the libraries, confirm they exist.

### `analyze_scenario`

Analyze a natural-language scenario into structured intent and create a session.
**This should be your first tool call for any scenario.**

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `scenario` | `str` | *(required)* | Human-language description of the task to automate. |
| `context` | `"web" \| "mobile" \| "api" \| "desktop" \| "generic" \| "database"` | `"web"` | Application context. `context="desktop"` **deterministically** forces a native PlatynUI session regardless of wording — use it for Linux/GNOME desktop GUI work. |
| `session_id` | `str` | `None` | Existing session id to reuse; a new one is created if omitted. |

**Returns:** `session_id` (save it), `session_info` (auto-configured libraries,
search order, next-step guidance), and parsed `intent` / `requirements` / `risk`.

**When to use:** first, always. It creates the session the rest of the workflow
depends on and picks a sensible starting library set from the scenario text.

### `recommend_libraries`

Recommend the libraries a scenario needs — no more guessing which import provides
the keyword you want.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `scenario` | `str` | *(required)* | Natural-language task description. |
| `context` | `"web" \| "mobile" \| "api" \| "desktop" \| "generic" \| "database"` | `"web"` | Scenario context. |
| `session_id` | `str` | `None` | Align recommendations with an existing session. |
| `max_recommendations` | `int` | `5` | Max libraries returned (direct mode). |
| `check_availability` | `bool` | `True` | Also check that suggested libraries are installable/present. |
| `apply_search_order` | `bool` | `True` | Apply the recommended order to the session. |
| `mode` | `"direct" \| "sampling_prompt" \| "sampling" \| "merge_samples" \| "merge"` | `"direct"` | Direct recommendation, an LLM sampling prompt, or a merge of sampled results. |
| `samples` | `list[dict]` | `None` | Sampled recommendations to merge (`mode="merge_samples"`). |
| `k` | `int` | `None` | Number of samples to request (`mode="sampling_prompt"`; defaults to 4). |
| `available_libraries` | `list[dict]` | `None` | Pre-fetched library metadata to use instead of registry defaults. |
| `include_keywords` | `bool` | `True` | Include a compact keyword-name list for the top recommendation. |
| `use_llm_refinement` | `bool` | `False` | Refine recommendations via `ctx.sample()`. |

**Returns:** `recommendations` (or a `sampling_prompt` / merged result depending
on `mode`), with the `session_id` preserved. Browser and SeleniumLibrary are
never recommended together, and negation ("not using Selenium") is honored.

**When to use:** immediately after `analyze_scenario`, before `execute_step` —
or any time a "No keyword with name" error tells you a library is missing.

### `check_library_availability`

Verify that named Robot Framework libraries can be imported or installed.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `libraries` | `list[str]` | *(required)* | Library names to verify — preferably the ones from `recommend_libraries`. |

**Returns:** `results`, a per-library availability and install-guidance map.

**When to use:** step three of planning — after `analyze_scenario` and
`recommend_libraries` — using the recommended names so you don't check libraries
you'll never import.

### `validate_scenario`

Pre-execution feasibility check for a parsed scenario.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `parsed_scenario` | `dict` | *(required)* | The parsed scenario from `analyze_scenario`. |
| `available_libraries` | `list[str]` | `None` | Libraries known to be available. |

**When to use:** an optional gate between analysis and execution when you want to
confirm the plan is achievable with the libraries at hand.

### `suggest_next_step`

AI-driven suggestion for the next test step, given where you are and where you're
headed.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `current_state` | `dict` | *(required)* | Current application state. |
| `test_objective` | `str` | *(required)* | Overall test objective. |
| `executed_steps` | `list[dict]` | `None` | Steps executed so far. |
| `session_id` | `str` | `"default"` | Session identifier. |

**When to use:** when you're mid-run and unsure what to do next — feed it the
state and the goal and it proposes a move.

---

## Session & Execution

The heart of the loop. A session holds the live Robot Framework context — its
libraries, variables, search order, and the steps you've recorded for the suite.

### `manage_session`

Manage the whole session lifecycle from one endpoint: initialize, import
libraries/resources, set variables, organize multi-test suites, and switch tool
profiles. `session_id` is always required.

**Actions** (`action`, a `Literal` enum) and their parameters:

| Action (aliases) | What it does | Key params |
|------------------|--------------|-----------|
| `init` (`initialize`, `bootstrap`) | Create a session and load libraries. | `libraries`, `variables` |
| `import_library` (`library`) | Add a library to an existing session. | `library_name`, `args`, `alias` |
| `import_resource` (`resource`) | Import a `.resource` file. | `resource_path`, `args` |
| `set_variables` (`variables`) | Set session variables. | `variables`, `scope` (`test`/`suite`/`global`, default `suite`) |
| `import_variables` (`load_variables`) | Load variables from a Python variable file. | `variable_file_path`, `args` |
| `start_test` (`start_task`) | Begin a named test (enables multi-test mode; local mode only). | `test_name` (required), `test_documentation`, `test_tags`, `test_setup`, `test_teardown`, `template` |
| `end_test` (`end_task`) | End the current test. | `test_status` (`pass`/`fail`), `test_message` |
| `add_data_row` (`data_row`) | Add a data row to the active data-driven (template) test. | `args` (values matching the template keyword's `[Arguments]`) |
| `list_tests` | List all tests with status and step counts. | — |
| `set_suite_setup` | Set a suite-level setup keyword. | `keyword`, `args` |
| `set_suite_teardown` | Set a suite-level teardown keyword. | `keyword`, `args` |
| `set_tool_profile` (`tool_profile`) | Switch which tools are visible to the model. | `tool_profile` / `profile`, `model_tier`, `model_name`, `scenario` |

**Selected parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `libraries` | `list[str]` | `None` | Library names to load (`init`). |
| `variables` | `dict \| list[str]` | `None` | `{"NAME": "value"}` or `["NAME=value"]`. |
| `scope` | `"test" \| "suite" \| "global"` | `"suite"` | Variable lifetime for `set_variables`. |
| `tool_profile` / `profile` | `"browser_exec" \| "api_exec" \| "discovery" \| "minimal_exec" \| "desktop_exec" \| "slim_exec" \| "full"` | `None` | Visible-tool profile — smaller models see fewer, leaner tools. |

**Returns:** `success`, `session_id`, and action-specific details.

**When to use:** any time you need to configure or reshape a session beyond what
`analyze_scenario` set up — import an extra library, seed variables, start a
second test, or slim the tool surface for a small model. The canonical
multi-test loop is `init → set_suite_setup → start_test → execute_step… →
end_test → start_test… → build_test_suite`.

> **Note.** `test_status` / `test_message` on `end_test` are session-tracking
> metadata only. They do **not** change the `.robot` file that
> `build_test_suite` generates.

### `execute_step`

Execute a single Robot Framework keyword (or an `Evaluate` expression) within a
session. This is the workhorse.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `keyword` | `str` | *(required)* | Keyword name; `Library.Keyword` form supported. Discover it first — don't invent it. |
| `arguments` | `list[str]` | `None` | Positional and named (`name=value`) arguments. |
| `session_id` | `str` | `"default"` | Session to execute in. |
| `raise_on_failure` | `bool` | `True` | If `True`, raise on failure; otherwise return the error in the payload. |
| `detail_level` | `"minimal" \| "standard" \| "full"` | `"minimal"` | Response verbosity. |
| `scenario_hint` | `str` | `None` | Scenario text to auto-configure libraries on the first call. |
| `assign_to` | `str \| list[str]` | `None` | Variable name(s) to capture the result — e.g. `assign_to="response"` → `${response}`. |
| `use_context` | `bool` | `None` | Run inside the RF native context; defaults via config/attach. |
| `mode` | `"keyword" \| "evaluate"` | `"keyword"` | `"evaluate"` runs `BuiltIn.Evaluate`. |
| `expression` | `str` | `None` | Expression for `mode="evaluate"`. |
| `timeout_ms` | `int` | `None` | Per-keyword timeout. Smart defaults apply by keyword type (clicks ~5000ms, navigation ~60000ms, reads ~2000ms, API ~30000ms). `0`/negative disables it. |
| `bdd_group` | `str` | `""` | Group name; steps sharing it cluster into one behavioral keyword when `build_test_suite(bdd_style=True)` runs. |
| `bdd_intent` | `str` | `""` | `given` / `when` / `then` / `and` / `but` prefix for the group. |
| `record` | `bool` | `None` | Override the record gate. `None` auto-classifies (read-only inspection keywords aren't recorded; `assign_to` or an open named test always records). `True`/`False` force it. |
| `pre_validate_timeout_ms` | `int` | `None` | Override the ~500ms pre-validation gate for this call. A positive int extends it (slow pages); `0`/negative skips pre-validation entirely (last resort — also disables the keyword timeout). |

**Returns:** `success`, `result`/`output`, `assigned_variables` /
`session_variables` where applicable, and `recorded: bool` (was the step kept for
the suite). Failure payloads carry `error`, `guidance`, and a
`pre_validate_timeout_hint` when the gate tripped.

**When to use:** to run one keyword at a time — the interactive, inspect-between-
steps rhythm rf-mcp is built for. Reach for `execute_batch` when you already know
the next several steps and want to save round-trips.

### `execute_batch`

Execute multiple keywords in one call, with variable chaining and tiered
recovery. Turns N MCP round-trips into 1.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | *(required)* | Session to execute within (created if absent). |
| `steps` | `list[dict]` | *(required)* | Ordered step dicts (see below). |
| `on_failure` | `"stop" \| "retry" \| "recover"` | `"recover"` | Failure policy. `recover` attempts tiered recovery before giving up. |
| `max_recovery_attempts` | `int` | `2` | Recovery retries per failed step (1–10). |
| `timeout_ms` | `int` | `120000` | Total batch budget in ms (1000–600000). |

Each **step** dict:

- `keyword` (`str`, required) — the keyword name. A step missing this returns an
  actionable validation error.
- `arguments` (`list[str]`, optional) — positional args; may contain `${STEP_N}`
  references to earlier results. Both 0-based (`${STEP_0}` = first step) and
  1-based (`${STEP_1}` = first step) indexing are accepted; 1-based wins when
  ambiguous. Legacy alias `args` is accepted, but supplying both differently is
  an error.
- `label` (`str`, optional) — human-readable label.
- `timeout` (`str`, optional) — per-step RF timeout (e.g. `"10s"`).
- `assign_to` (`str`, optional) — capture the return value into a variable.

**Returns:** `status` (`PASS` / `FAIL` / `RECOVERED` / `TIMEOUT`), `summary`,
`total_time_ms`, `steps_executed` / `steps_total`, a per-step `steps` array, and
on `FAIL` a `failure` diagnostic plus a `batch_id` for [`resume_batch`](#resume_batch).

**When to use:** when the next several steps are known and independent of
mid-flight inspection. On desktop (PlatynUI) sessions retries are deliberately
conservative — only element-not-found failures re-fire, so a stray click or
keystroke is never blindly repeated.

> **Note.** Batch steps do **not** support `bdd_group` / `bdd_intent`. For BDD
> grouping, use per-step `execute_step(bdd_group=…, bdd_intent=…)`.

### `resume_batch`

Resume a failed batch from its failure point, optionally injecting fixes first.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `batch_id` | `str` | *(required)* | The `batch_id` from a failed `execute_batch`. |
| `fix_steps` | `list[dict]` | `None` | Steps to run before retrying the failed step (same format as `execute_batch` steps). |
| `timeout_ms` | `int` | `None` | Override the remaining budget (uses the original if omitted). |

**Returns:** same shape as `execute_batch`.

**When to use:** right after `execute_batch` returns `FAIL` — patch the cause
(re-locate, wait, dismiss an overlay) and pick up where the batch stopped instead
of replaying it from the top.

### `execute_flow`

Execute a structured control flow — `if`, `for`, or `try` — inside a session.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `structure` | `"if" \| "for" \| "try"` | *(required)* | Flow type. |
| `session_id` | `str` | *(required)* | Session to run the flow in. |
| `condition` | `str` | `None` | Expression for `if`. |
| `then_steps` | `list[dict]` | `None` | Main branch / loop body / try block. |
| `else_steps` | `list[dict]` | `None` | Else branch (`if`). |
| `items` | `list` | `None` | Items to iterate (`for`). |
| `item_var` | `str` | `"item"` | Loop variable name bound to each item. |
| `stop_on_failure` | `bool` | `True` | Stop the loop/branch on first failure. |
| `max_iterations` | `int` | `1000` | Iteration cap for `for` loops. |
| `try_steps` | `list[dict]` | `None` | Try block (`try`). |
| `except_patterns` | `list[str]` | `None` | Error patterns to match for `except`. |
| `except_steps` | `list[dict]` | `None` | Except block. |
| `finally_steps` | `list[dict]` | `None` | Finally block. |
| `rethrow` | `bool` | `False` | Re-raise after except/finally. |

**Returns:** `success`, the `structure` executed, the echoed `session_id`, and
per-branch results/errors.

**When to use:** when the test genuinely needs branching or iteration executed in
the live context — conditional cleanup, looping over data, or guarded actions
with a fallback.

### `intent_action`

Execute a high-level intent that auto-resolves to the correct library keyword and
locator format for the session's active library (Browser / Selenium / Appium). No
need to remember library-specific keyword names.

**Valid intents:** `navigate`, `click`, `fill`, `hover`, `select`,
`assert_visible`, `extract`, `wait_for`. (`extract_text` is accepted but
deprecated — use `extract` with `mode="text"`.)

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `intent` | `Literal` (see above) | *(required)* | The action verb. |
| `target` | `str` | `None` | Locator or URL (`"#submit"`, `"text=Login"`, a URL). Optional for `extract` with `mode="url"`/`"title"`. |
| `value` | `str` | `None` | Value for `fill` / `select`. |
| `session_id` | `str` | `None` | Session to execute against (default if omitted). |
| `options` | `dict[str,str]` | `None` | Extra options, e.g. `{"timeout": "10s"}`. |
| `assign_to` | `str` | `None` | Capture the result — especially useful with `extract`. |
| `detail_level` | `"minimal" \| "standard" \| "full"` | `"standard"` | Response detail. |
| `force` | `bool` | `False` | For a Browser click blocked by an overlay/sticky banner: swaps `Click` for `Click With Options force=True`, skipping actionability checks. Do **not** use it to drive genuinely hidden elements. |
| `match` | `"label" \| "value" \| "index" \| "text" \| "auto"` | `"label"` | Select-match strategy for the `select` intent. `auto` is an opt-in heuristic (numeric → value). |
| `nth` | `int` | `None` | Zero-based nth-match index to disambiguate multiple matches. |
| `commit` | `bool` | `False` | After a Browser `fill`, dispatch a real DOM `change` event (for Vue/React/Angular/jQuery-validate forms that gate on it). Best-effort; never escalates a good fill into a failure. |
| `mode` | `"text" \| "attribute" \| "count" \| "value" \| "url" \| "title"` | `"text"` | For `extract` only — what to read. `count` also skips pre-validation (zero/multiple matches is the expected outcome). |
| `attribute_name` | `str` | `None` | Required when `intent="extract"` and `mode="attribute"` (e.g. `"href"`). |

**Returns:** the action result; for `extract`, the read value is surfaced at
`result["extracted_value"]` and assigned to `assign_to` if given.

**When to use:** the token-light path for common web/mobile actions, ideal for
smaller models — express *what* you want and let rf-mcp map it to the *how*. When
`intent="navigate"` fails with no browser open, the server opens one and retries
(`fallback_applied: true`), saving a couple of calls.

### `evaluate_expression`

Evaluate a Python expression in the RF context (`BuiltIn.Evaluate`).

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | *(required)* | Session whose variables are in scope. |
| `expression` | `str` | *(required)* | Python expression; `${var}` references supported. |
| `assign_to` | `str` | `None` | Assign the result to a variable (test scope). |

**When to use:** for quick computation or assertion glue against live session
variables without hunting for a dedicated keyword.

### `set_variables`

Set several variables at once in the RF session variable store.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | *(required)* | Target session. |
| `variables` | `dict \| list[str]` | *(required)* | `{name: value}` or `["name=value"]`. |
| `scope` | `"test" \| "suite" \| "global"` | `"test"` | Variable lifetime. |

**When to use:** to seed test data or configuration mid-run. (For init-time
variables, `manage_session(action="set_variables")` does the same with `suite`
scope by default.)

### `set_library_search_order`

Set the explicit library search order that decides which library wins when two
provide a same-named keyword.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `libraries` | `list[str]` | *(required)* | Library names, highest priority first. |
| `session_id` | `str` | `"default"` | Session to apply the order to. |

**Returns:** `old_search_order` / `new_search_order`, plus `warnings` for any
invalid or missing libraries.

**When to use:** when keyword resolution goes to the wrong library — for example
Browser vs SeleniumLibrary both offering `Click`, or a plugin colliding with a
core keyword.

### `initialize_context`

Seed a session with libraries and variables.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | *(required)* | Session identifier. |
| `libraries` | `list[str]` | `None` | Libraries to import. |
| `variables` | `dict` | `None` | Initial variables to set. |

**When to use:** a lightweight session-seed. Note that full RF native-context
mode is not yet implemented here — for most flows `manage_session(action="init")`
is the richer, preferred entry point.

---

## Discovery & Documentation

Don't guess keyword names — look them up. These tools read the live namespace and
Robot Framework's native libdoc.

### `find_keywords`

Discover keywords using one of four strategies. **Call this before
`execute_step` with any unfamiliar keyword** — it's the cure for "No keyword with
name 'X' found".

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `query` | `str` | *(required)* | Search text or intent (`"click a button"`, `"validate json"`, `"Get*"`). |
| `strategy` | `"semantic" \| "pattern" \| "catalog" \| "session"` | `"semantic"` | See below. |
| `context` | `"web" \| "mobile" \| "api" \| "desktop" \| "generic" \| "database"` | `"web"` | Context used by semantic discovery. |
| `session_id` | `str` | `None` | Required for `strategy="session"` to search the live namespace. |
| `library_name` | `str` | `None` | Restrict results to this library and its compatible siblings; applies to all strategies and scopes the catalog lookup. |
| `current_state` | `dict` | `None` | State payload to sharpen semantic matching. |
| `limit` | `int` | `None` | Max results. |
| `strict_library` | `bool` | `False` | When `True` with a library preference set, exclude *every* other library — including BuiltIn/Collections/String helpers. |

**Strategies:** `semantic` (hybrid name/doc/tag matching, plus embedding
similarity when the `semantic` extra or `ROBOTMCP_SEMANTIC_KEYWORDS` backend is
enabled), `pattern` (glob/regex on names), `catalog` (a **literal substring**
filter — a multi-word natural query returns nothing; pass `library_name` to list
a library), and `session` (keywords from the session's loaded libraries).

**Returns:** `strategy`, echoed `query`, and a strategy-specific `result` /
`results` payload.

**When to use:** any time you're unsure of a keyword's exact name or want to see
what a library offers. This is the discovery tool the workflow guide points to.

### `discover_keywords`

Find keywords matching an action description.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `action_description` | `str` | *(required)* | The action to perform. |
| `context` | `str` | `"web"` | Current context (web, mobile, API…). |
| `current_state` | `dict` | `None` | Current application state. |

**When to use:** a simpler, description-first discovery call. For most work
`find_keywords` (with its strategy and library filters) is the sharper tool.

### `search_keywords`

Search keywords by pattern using Robot Framework's native libdoc — across names,
documentation, `short_doc`, and tags. Ensures session libraries are loaded first.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `pattern` | `str` | *(required)* | Pattern to match against names, docs, or tags. |

**Returns:** a list of matching keywords with libdoc metadata (`short_doc`,
argument types, deprecation status, tags).

**When to use:** when you want libdoc-grade search results with full metadata
rather than intent-ranked suggestions.

### `get_keyword_info`

One endpoint for keyword/library documentation and signature parsing.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `mode` | `"keyword" \| "library" \| "session" \| "parse"` | `"keyword"` | What to retrieve. |
| `keyword_name` | `str` | `None` | Required for `keyword` / `session` / `parse`. |
| `library_name` | `str` | `None` | Required for `library`; optional per-call scope for `keyword` mode (takes precedence over session scope). |
| `session_id` | `str` | `None` | Scopes `keyword` lookups to a session's libraries (a mismatch returns a library-hint instead of the doc), addresses the live namespace for `session` mode, and **enables artifact externalization** for large payloads. |
| `arguments` | `list[str]` | `None` | Arguments to parse when `mode="parse"`. |

**Returns:** `mode`, the doc/signature data (or a library-mismatch `hint`), with
large fields possibly replaced by an artifact summary when `session_id` is set.

**When to use:** the consolidated documentation tool — check a keyword's
signature before calling it, or parse an argument list to get the shape right.

### `get_keyword_documentation`

Full libdoc for a single keyword via Robot Framework's native
`LibraryDocumentation` / `KeywordDoc`.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `keyword_name` | `str` | *(required)* | Keyword to document. |
| `library_name` | `str` | `None` | Narrow the search to a library. |

**Returns:** `keyword` with `name`, `library`, `args`, `arg_types`, full `doc`,
`short_doc`, `tags`, `is_deprecated`, `source`, and `lineno`.

**When to use:** when you want the complete, authoritative docstring — arguments,
types, source location — for exactly one keyword.

### `get_library_documentation`

Full libdoc for a library and every keyword it exposes.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `library_name` | `str` | *(required)* | Library to document. |

**Returns:** `library` with `name`, `doc`, `version`, `type`, `scope`, `source`,
`keyword_count`, `data_source` (`libdoc` or `inspection`), and a `keywords` list
with per-keyword details.

**When to use:** to survey an unfamiliar library in full before deciding which
keywords to use.

### `get_available_keywords`

List keywords with minimal metadata — one compact entry per keyword.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `library_name` | `str` | `None` | Filter to one library (loaded on demand if needed). |

**Returns:** per keyword: `name`, `library`, `args`, `arg_types`, and
`short_doc` — no full docstrings.

**When to use:** a fast, low-token inventory when you just need names and
argument shapes.

### `get_loaded_libraries`

Status of all currently loaded libraries, via both libdoc and inspection.

**Parameters:** none.

**When to use:** to see what's actually imported and ready before executing.

### `get_library_status`

Detailed installation status for one library.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `library_name` | `str` | *(required)* | Library to check (e.g. `"Browser"`, `"SeleniumLibrary"`). |

**Returns:** detailed status plus installation information.

**When to use:** to diagnose why a specific library isn't available and how to
install it.

---

## Locators & Guidance

Locators are where UI tests live or die. These cookbooks teach each library's
selector syntax so you write real locators, not lucky guesses.

### `get_locator_guidance`

The consolidated guidance endpoint. Covers Browser, SeleniumLibrary,
AppiumLibrary, and PlatynUI.BareMetal locators — and doubles as an API and visual
cookbook.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `library` | `str` | `"browser"` | Target: `"Browser"`, `"SeleniumLibrary"`, `"AppiumLibrary"`, `"PlatynUI.BareMetal"`, `"requests"`/`"api"`, or `"visual"`/`"screenshot"`. Case-insensitive. |
| `error_message` | `str` | `None` | Error text to tailor the guidance. |
| `keyword_name` | `str` | `None` | Keyword name for context-specific hints. |

**Returns:** `library` (resolved), plus `tips` / `warnings` / `examples`.

**When to use:** before writing locators for any UI library. For `library="requests"`
you get a RequestsLibrary request/response cookbook — session setup, response-field
access (`${resp.json()["field"]}`), the `$resp`-in-`Evaluate` rule, `Status Should
Be`, `expected_status=` — **before** you reach for `Evaluate`-based assertions. For
`library="visual"` you get the vision cookbook: *when* a screenshot beats the DOM
(canvas/image text, layout/overlap, obscured elements, color, charts) and the
dual read-back pattern.

### `get_browser_locator_guidance`

Browser Library (Playwright) selector guidance — CSS, xpath, text, id,
data-testid; cascaded selectors, iframe piercing, shadow DOM, strict mode; and
the implicit detection rules (plain → CSS, `//` → xpath, quoted → text).

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `error_message` | `str` | `None` | Error to analyze for specific guidance. |
| `keyword_name` | `str` | `None` | Failed keyword for context-specific tips. |

**When to use:** Browser-specific detail. `get_locator_guidance(library="browser")`
is the newer consolidated front door.

### `get_selenium_locator_guidance`

SeleniumLibrary locator strategies — `id`, `name`, `identifier`, `class`, `tag`,
`xpath`, `css`, `dom`, `link`, `partial link`, `data`, `jquery`, `default` — with
examples and error-specific advice.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `error_message` | `str` | `None` | Error to analyze. |
| `keyword_name` | `str` | `None` | Failed keyword for context. |

**When to use:** when driving SeleniumLibrary and a locator won't resolve.

### `get_appium_locator_guidance`

AppiumLibrary locator strategies — `id`, `xpath`, `accessibility_id`, `class`,
platform-specific `android`/`ios`/`predicate`/`chain`, and webview `css` — plus
WebElement usage and error-specific tips.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `error_message` | `str` | `None` | Error to analyze. |
| `keyword_name` | `str` | `None` | Failed keyword for context. |

**When to use:** for mobile element location, native or webview.

---

## State & Observability

When a step surprises you, look. These tools expose the DOM, the variables, the
validation ledger, and the rendered pixels.

### `get_session_state`

The primary insight tool: aggregated session state for debugging and visibility —
DOM/ARIA snapshots, RF variables and search order, validation summaries, library
lists, attach status, and (on desktop) the accessibility UI tree.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | *(required)* | Session to inspect. |
| `sections` | `list[str]` | `None` | Blocks to include (`summary`, `page_source`, `variables`, `application_state`, `ui_tree`, …). |
| `state_type` | `str` | `"all"` | App-state type for `application_state` (`dom` / `api` / `database` / `all`). |
| `elements_of_interest` | `list[str]` | `None` | Element identifiers to focus (also expands `ui_tree` application subtrees). |
| `page_source_filtered` | `bool` | `False` | Return sanitized/filtered DOM instead of full source. |
| `page_source_filtering_level` | `"standard" \| "aggressive"` | `"standard"` | Filtering aggressiveness. |
| `include_reduced_dom` | `bool` | `True` | Include the lightweight ARIA snapshot. |
| `include_dom_stream` | `bool` | `False` | Chunk large page source into `page_source_stream` entries. |
| `dom_chunk_size` | `int` | `65536` | Chunk size when streaming (min 1024). |
| `mode` | `"full" \| "delta" \| "auto" \| "none"` | `"auto"` | `delta` returns only sections changed since `since_version`; `auto` deltas automatically once a prior version exists. |
| `since_version` | `int` | `None` | Baseline version for `delta`. |

**Returns:** `requested` (the section names asked for), `sections` — a map of section name → content (variables, page
source/ARIA snapshots, validation, libraries, application state).

**When to use:** the go-to move on any "element not found" — pull a fresh ARIA
snapshot and read the real ids, roles, and text. `mode="delta"` keeps multi-step
inspection cheap by returning only what changed.

### `get_session_info`

Comprehensive information about a session's configuration and state.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | `"default"` | Session to inspect. |

**When to use:** a quick read of how a session is set up — libraries, mode,
configuration — without pulling the full DOM.

### `get_session_validation_status`

Validation status of every step in a session, with intelligent session
resolution.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | `""` | Session (auto-resolves if empty/invalid). |

**When to use:** to see which recorded steps have passed validation before you
build a suite.

### `get_application_state`

Retrieve the current application state.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `state_type` | `str` | `"all"` | `dom` / `api` / `database` / `all`. |
| `elements_of_interest` | `list[str]` | `None` | Elements to focus on. |
| `session_id` | `str` | `"default"` | Session identifier. |

**When to use:** for plugin-provided application insight. For most inspection
`get_session_state` is the richer aggregate.

### `get_page_source`

Page source and context for a browser session, with optional DOM filtering.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | `"default"` | Browser session. |
| `full_source` | `bool` | `False` | Return the full, unreduced source. |
| `filtered` | `bool` | `False` | Return sanitized/filtered DOM. |
| `filtering_level` | `str` | `"standard"` | Filtering aggressiveness. |
| `include_reduced_dom` | `bool` | `True` | Include the ARIA snapshot. |

**When to use:** when you want the raw or filtered DOM directly. Within a broader
inspection, `get_session_state(sections=["page_source"])` wraps the same data.

### `get_context_variables`

Get all variables from a session.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | *(required)* | Session to read. |

**When to use:** to dump the current variable store — useful when chaining values
across steps and confirming an `assign_to` landed.

### `validate_test_readiness`

Check whether a session is ready for suite generation. Enforces the stepwise
workflow by verifying that all steps have been validated.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | `"default"` | Session to validate. |

**Returns:** readiness status with guidance on next actions.

**When to use:** right before `build_test_suite`, as a quality gate.

### `visual_check`

Capture a screenshot of the current UI for visual validation. Token-cheap by
default — saves the image to disk and returns the **path** as text; a multimodal
agent with file access reads it on demand for checks the DOM can't do
(canvas/image text, layout/overlap, obscured elements, color, charts).

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | *(required)* | Session to screenshot. |
| `return_image` | `bool` | `False` | Return an actual image content block. Honored **only** when the model can't read the saved file *and* `ROBOTMCP_SCREENSHOT_MODE` allows images (`image`/`auto`). |
| `filename` | `str` | `None` | Optional output path. Honored **only** when it is an absolute path; otherwise ignored and the file is written to `ROBOTMCP_SCREENSHOT_DIR` (or the temp dir) as `visual_check_<session_id>.png`. |

**Returns:** by default `{success, screenshot_path, size_bytes, mode, visual_hint}` (text only); an image
block when `return_image=true` is honored. Works across
Browser/Selenium/Appium/PlatynUI and degrades cleanly if capture fails.

**When to use:** for the vision-only cases — call
`get_locator_guidance(library="visual")` first to learn when a screenshot
genuinely beats `Get Text`. Text-only deployments (`ROBOTMCP_SCREENSHOT_MODE=file`,
the default) always get just the path, so a text-only model is never sent an
image it can't read.

---

## Suite Lifecycle

The payoff: turn a session's successful steps into a real, runnable `.robot`
suite — and validate it before you trust it.

### `build_test_suite`

Generate a Robot Framework suite from the steps executed in a session.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `test_name` | `str` | *(required)* | Name for the generated test case. |
| `session_id` | `str` | `""` | Session with executed steps (auto-resolves if empty/invalid). |
| `tags` | `list[str]` | `None` | Test tags. |
| `documentation` | `str` | `""` | Test case documentation. |
| `remove_library_prefixes` | `bool` | `True` | Strip library prefixes from keyword names. |
| `bdd_style` | `bool` | `False` | Generate a BDD suite: steps grouped into Given/When/Then behavioral keywords with an appended `*** Keywords ***` section. |
| `data_driven_mode` | `"auto" \| "per_test" \| "suite_template"` | `"auto"` | How to render template tests. `auto` picks `suite_template` for named rows, else `per_test`. |
| `include_pre_start` | `bool` | `False` | Adopt exploratory steps executed *before* `start_test` into the test body. Default excludes them (response reports `excluded_pre_start_count`). |
| `output_path` | `str` | `""` | Absolute path to persist the `.robot` file (UTF-8, parents created). **Always use this to save a suite.** |

**Returns:** `suite` metadata, `rf_text` (the generated content),
`statistics` / `optimization_applied`, and `output_path` / `output_bytes` when
written to disk.

**When to use:** at the end of a run, once steps are validated. Set
`bdd_style=True` for Given/When/Then output.

> **⚠ Never write `rf_text` yourself via the `Create File` keyword.** Robot
> Framework resolves `${variables}` and expands `\n` / `\t` inside the argument,
> silently corrupting the suite. `output_path` writes the bytes through plain
> file I/O, preserving the content exactly.

### `run_test_suite`

Validate or execute a Robot Framework suite.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | `""` | Session whose steps build the suite (optional if `suite_file_path` given). |
| `suite_file_path` | `str` | `None` | Path to an existing `.robot` file to validate/execute. |
| `mode` | `"dry" \| "validate" \| "full"` | `"full"` | `dry`/`validate` for a dry run; `full` to execute. |
| `validation_level` | `"minimal" \| "standard" \| "strict"` | `"standard"` | Dry-run depth. |
| `include_warnings` | `bool` | `True` | Include warnings in validation output. |
| `execution_options` | `dict` | `None` | RF options (variables, tags, loglevel, `test`/`tests`, `pythonpath`, `dry_run_timeout`, …). |
| `output_level` | `str` | `"standard"` | Response verbosity (`minimal`/`standard`/`detailed`). |
| `capture_screenshots` | `bool` | `False` | Capture screenshots on failure, where supported. |

**Returns:** `mode`, plus `statistics` / `execution_details` / `output_files`
when executed, or `validation_results` on a dry run.

**When to use:** to actually run the suite — or to validate it first with
`mode="dry"`.

### `run_test_suite_dry`

Validate a suite using Robot Framework's dry-run mode. The dedicated validation
step between building and executing.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `session_id` | `str` | `""` | Session with executed steps (auto-resolves if empty/invalid). |
| `suite_file_path` | `str` | `None` | Direct path to a `.robot` file (overrides session). |
| `validation_level` | `"minimal" \| "standard" \| "strict"` | `"standard"` | `minimal` = syntax only; `standard` = + keyword/imports; `strict` = + argument/structure checks. |
| `include_warnings` | `bool` | `True` | Include warnings. |
| `execution_options` | `dict` | `None` | RF CLI-oriented options when validating a file path. |

**Returns:** structured validation results — issues, warnings, suggestions.

**When to use:** the recommended step after `build_test_suite` and before
`run_test_suite` — catch syntax and keyword errors without a full execution.

### `load_test_data`

Load external data (CSV, Excel, JSON) for data-driven testing. Works with
DataDriver-compatible formats; falls back to built-in CSV/JSON parsing if
DataDriver isn't installed.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `file_path` | `str` | *(required)* | Absolute path to the data file. |
| `encoding` | `str` | `"utf-8"` | File encoding. |
| `dialect` | `str` | `"Excel-EU"` | CSV dialect (`Excel-EU`, `excel`, `unix`). |
| `delimiter` | `str` | `";"` | Column delimiter for CSV. |
| `sheet_name` | `str` | `"0"` | Excel sheet index or name. |
| `limit` | `int` | `100` | Max rows to return. |

**Returns:** `test_cases`, `count`, `format`, and `column_names`.

**When to use:** to inspect an external data file before building a data-driven
suite — pair it with `manage_session(action="add_data_row")`.

---

## Artifacts

### `fetch_artifact`

Retrieve externalized artifact content by ID. **This tool is hidden by default** — it is only registered as visible when `ROBOTMCP_FETCH_ARTIFACT=true` *and* `ROBOTMCP_OUTPUT_MODE` is not `inline`. When a response would be large
(HTML page source, logs, stack traces), rf-mcp externalizes it and returns a
summary with an `artifact_id`; this tool fetches the full content, paginated.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `artifact_id` | `str` | *(required)* | The identifier, format `art_{12 hex}`. |
| `offset` | `int` | `0` | Starting character offset. |
| `limit` | `int` | `4000` | Max characters to return. |

**Returns:** `content`, `offset`, `total_size`, `has_more`.

**When to use:** whenever a response hands you an `artifact_id` and you need the
full payload — page through it with `offset` / `limit`.

---

## Library & Plugin Management

Extend and inspect rf-mcp's library plugins, and manage the debug attach bridge.

### `manage_library_plugins`

Inspect or reload library plugins from one endpoint.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `action` | `"list" \| "reload" \| "diagnose"` | `"list"` | Operation to perform. |
| `plugin_name` | `str` | `None` | Plugin name when `action="diagnose"`. |

**Returns:** `action` echoed plus `plugins` / `plugin` / `reload_result`.

**When to use:** the consolidated front door for plugin management — list what's
loaded, reload after editing a plugin, or diagnose one that misbehaves.

### `list_library_plugins`

Return a summary of every loaded library plugin.

**Parameters:** none.

**When to use:** a quick roster of active plugins. `manage_library_plugins(action="list")`
does the same.

### `diagnose_library_plugin`

Return detailed information about one library plugin.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `plugin_name` | `str` | *(required)* | Plugin to diagnose. |

**When to use:** when a plugin isn't behaving and you need its detail.
`manage_library_plugins(action="diagnose")` is the consolidated equivalent.

### `reload_library_plugins`

Reload library plugins and return the resulting library list.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `manifest_paths` | `list[str]` | `None` | Specific manifest files to reload from. |

**When to use:** after adding or editing a workspace plugin manifest under
`.robotmcp/plugins/`, to pick up the change without restarting the server.

### `manage_attach`

Inspect or control the debug attach bridge — the localhost HTTP link that lets
rf-mcp reuse a live Robot Framework debug session's context (variables, imports,
search order) instead of spinning up its own.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `action` | `"status" \| "info" \| "stop" \| "shutdown" \| "cleanup" \| "clean" \| "reset" \| "reconnect" \| "disconnect_all" \| "terminate" \| "force_stop"` | `"status"` | Bridge operation. `status` reports config/health; `cleanup` clears expired sessions; `reset`/`reconnect` stops the bridge and cleans local sessions; `disconnect_all`/`terminate`/`force_stop` force-stops everything. |

**Returns:** `action` echoed, plus `configured` / `reachable` / `default_mode` /
`strict` and, per action, `sessions_cleaned`, `bridge_stopped`, `recovery_hint`,
or `diagnostics`.

**When to use:** to check whether the attach bridge is reachable, clean up stale
sessions, or tear the bridge down. The bridge is enabled by setting
`ROBOTMCP_ATTACH_HOST` (see the README's Debug Attach Bridge section).

---

## Memory *(optional)*

These five tools appear **only** when persistent semantic memory is enabled —
install `rf-mcp[memory]` (sqlite-vec + model2vec) and set
`ROBOTMCP_MEMORY_ENABLED=true`. With memory on, rf-mcp learns across sessions:
storing successful step sequences, working locators, and error→fix mappings, and
injecting recalled hints back into ordinary tool responses. Every lookup is
timeout-bounded so it never slows the run down.

### `recall_step`

Recall previously successful step sequences for a scenario.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `scenario` | `str` | *(required)* | Scenario to recall step sequences for. |
| `top_k` | `int` | `5` | Accepted for schema compatibility but currently **ignored** — the query always requests up to 10 results. |

**Returns:** ranked `results`, a `count`, and a `suggestion`. When a match scores
similarity > 0.3, prefer the recalled steps over discovering new ones.

**When to use:** **before** building new steps for a scenario you may have
automated before — reuse what worked.

### `recall_fix`

Recall known fixes for an error.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `error_text` | `str` | *(required)* | The error message to look up. |

**Returns:** ranked `results` (previously successful recovery strategies),
`count`, and a `suggestion`.

**When to use:** **immediately** when `execute_step` fails, before retrying —
check whether this exact error has a known cure.

### `recall_locator`

Recall working locators for a UI element.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `element_description` | `str` | *(required)* | Description of the element. |

**Returns:** ranked `results` with success/failure history, `count`, and a
`suggestion` (high-confidence when the top match scores > 0.5).

**When to use:** **before** DOM inspection for an element you've interacted with
before — start from a proven locator instead of re-deriving one.

### `store_knowledge`

Store reusable domain knowledge (site structure, auth flows) for future recall.

**Parameters**

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `content` | `str` | *(required)* | The knowledge to store. |
| `knowledge_type` | `"documentation" \| "domain_knowledge"` | `"domain_knowledge"` | Category — anything else is rejected. |
| `tags` | `list[str]` | `[]` | Tags for retrieval. |

**Returns:** `stored: bool` and a `record_id` on success, or a `reason` on
failure.

**When to use:** when you discover reusable facts about the system under test
that will pay off in a later run.

### `get_memory_status`

Check memory availability and collection statistics.

**Parameters:** none.

**Returns:** availability/stats plus the embedding `backend` info.

**When to use:** at session start, to see whether memory is on and what
historical data exists to draw on.

---

*rf-mcp is open source, Apache-2.0. Built for the Robot Framework and AI
automation community — [github.com/manykarim/rf-mcp](https://github.com/manykarim/rf-mcp).*
