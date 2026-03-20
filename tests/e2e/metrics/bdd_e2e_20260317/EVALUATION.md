# E2E BDD Test Evaluation — 2026-03-17

## Scenario
BDD Purchase Flow on https://demoshop.makrocode.de/
- Open demoshop, add 2 products to cart, verify cart, checkout, verify success
- Generate BDD .robot file using `build_test_suite(bdd_style=True)`

## Tool Comparison

| Metric | Copilot CLI (GPT-5.4) | Codex CLI | Claude CLI (Opus 4.6) |
|--------|----------------------|-----------|----------------------|
| **Status** | SUCCESS | FAILED (billing) | SUCCESS |
| **Turns** | 39 | 0 | 38 |
| **execute_step calls** | 53 (40 passed) | 0 | 17 (all passed) |
| **get_session_state calls** | 16 | 0 | ~12 |
| **find_keywords calls** | 4 | 0 | 0 |
| **build_test_suite calls** | 1 | 0 | 1 |
| **Total MCP tool calls** | 75 | 0 | ~32 |
| **Duration** | ~8min | 0 | ~5min (314s) |
| **Cost** | N/A (GitHub sub) | N/A | $1.70 |
| **Checkout completed** | Yes (ORD-6ADE8C07) | No | Yes (ORD-7E670FF1) |
| **BDD .robot generated** | Yes | No | Yes |
| **Permission issues** | Run 1: 33 denials (--autopilot insufficient), Run 2: 0 (--allow-all) | Usage limit | None |

## Detailed Analysis

### Copilot CLI (GitHub Copilot, GPT-5.4)
- **Run 1**: `--autopilot` flag was insufficient for MCP tool permissions — 33 "Permission denied" errors. Model tried `web_fetch` and `bash` instead of MCP tools. Zero `execute_step` calls.
- **Run 2** (`--allow-all`): Successfully completed the full flow. 53 execute_step calls, all 40 unique steps passed. Model adapted well to the actual site content (Flowline Supply, not Sauce Labs). Generated BDD .robot with `*** Keywords ***` section.
- **BDD Quality**: Keyword names are generic ("the user clicks the element", "the expected results are verified"). The `then` keywords contain excessive `Get Property` calls rather than targeted assertions. Embedded arg detection produced `the user clicks ${value} t` (truncated name).
- **Key strengths**: Thorough DOM inspection (16 get_session_state calls), adapted to real site content.
- **Key weaknesses**: Verbose execution (53 calls for ~15 logical steps), generic BDD names, noisy assertion keywords.

### Codex CLI (OpenAI)
- **Status**: MCP server started and connected successfully, but execution failed immediately due to billing: "You've hit your usage limit."
- **Note**: Not a tool capability issue — purely a billing/quota problem.

### Claude CLI (Anthropic, Opus 4.6)
- **Run 1** (50 turns): Hit max_turns while still actively working. 50 turns of tool use were not enough for the full scenario.
- **Run 2** (80 turns): Completed successfully in 38 turns. 17 execute_step calls, all passed. Model was efficient — fewer calls than copilot for the same outcome.
- **BDD Quality**: Keyword names are more natural ("the demoshop is open", "the user fills in the form") but still generic at the assertion level ("the expected results are verified"). The heuristic grouper compressed 17 steps into 3 keywords — too aggressive, losing checkout and order verification granularity.
- **Key strengths**: Efficient (17 steps vs copilot's 53), clean ARIA locators (role=link, role=button), proper session management.
- **Key weaknesses**: Over-aggressive heuristic grouping (3 keywords for 17 steps), first run needed more than 50 turns.

## Generated .robot Files

### Copilot Generated Suite
- 14 optimized steps from 40 original (65% optimization)
- Has `*** Keywords ***` section with BDD keywords
- Uses XPath locators (more fragile than ARIA)
- `the expected results are verified` keyword has 14 lines of Get Property calls

### Claude Generated Suite
- 17 steps grouped into 3 BDD keywords
- Has `*** Keywords ***` section
- Uses ARIA role-based locators (more robust)
- Keywords are too coarse-grained (all checkout steps merged into one)

## Recommendations

1. **Copilot**: Requires `--allow-all` flag for MCP tools. The `--autopilot` flag only auto-approves built-in tools.
2. **Codex**: Needs active billing/quota to test.
3. **Claude**: 50 turns is insufficient for full demoshop scenario; 80 turns with streamlined prompt works.
4. **BDD quality**: Both tools produce functional but mediocre BDD — the heuristic grouper needs refinement for multi-phase scenarios (setup → browse → cart → checkout → verify).
5. **Locator strategy**: Claude's ARIA locators (`role=button[name="..."]`) are more maintainable than Copilot's XPath locators.
