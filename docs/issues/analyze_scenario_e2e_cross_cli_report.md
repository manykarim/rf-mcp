# `analyze_scenario` Defect — Cross-CLI E2E Reproduction + Post-Fix Verification

**Date**: 2026-06-05
**Defect**: `explicit_library_preference: "SeleniumLibrary"` returned for scenarios that never mention SeleniumLibrary
**Status**: **FIXED in v5** (`src/robotmcp/utils/library_detection.py`)
**Related**: PRD `docs/prd/analyze_scenario_explicit_library_prd.md`, ADR-024, DDD library_preference, solution proposal (all at v5 IMPLEMENTED)
**Purpose**: confirm the defect (a) reproduces consistently across multiple LLM clients consuming rf-mcp via MCP, AND (b) is resolved by the v5 implementation across the same three clients.

---

## Scenario under test (verbatim user report)

```json
{
  "scenario": "Test e-commerce website https://demoshop.makrocode.de: open browser, add items to shopping cart, verify items, complete checkout, and close browser",
  "context": "web",
  "session_id": "demoshop_<cli>_e2e"
}
```

Expected behaviour after v4 fix: `explicit_library_preference: null` (the user never mentions SeleniumLibrary, `webdriver`, `selenium`, etc. — `open browser` is generic English).

Current (pre-fix) behaviour: `explicit_library_preference: "SeleniumLibrary"`.

---

## Method

Three independent LLM CLIs consumed the rf-mcp MCP server's `analyze_scenario` tool:

| CLI | Version | Model | rf-mcp transport |
|---|---|---|---|
| Claude Code (Anthropic) | this session | claude-opus-4-7[1m] | Direct Python subprocess (`uv run python -c ...`) — rf-mcp not loaded in session |
| Codex CLI (OpenAI) | 0.133.0 | gpt-5.4 (high reasoning) | MCP via ephemeral `-c 'mcp_servers.robotmcp.*'` config |
| opencode | 1.2.15 | github-copilot/claude-sonnet-4.6 | MCP via pre-registered `opencode mcp add` (status: connected) |

Each CLI was instructed to call `analyze_scenario` with the exact reported arguments and report only the values of `analysis.explicit_library_preference` and `analysis.suggested_libraries`.

---

## Results

### Claude Code (direct Python invocation)

`/tmp/claude_e2e_output` (via inline subprocess):

```json
{
  "action_count": 1,
  "complexity": "simple",
  "estimated_steps": 2,
  "suggested_libraries": ["Browser", "RequestsLibrary", "SeleniumLibrary"],
  "explicit_library_preference": "SeleniumLibrary",
  "detected_session_type": "web_automation"
}
```

### Codex CLI

`/tmp/codex_e2e_output.txt`:

```
mcp: robotmcp/analyze_scenario started
mcp: robotmcp/analyze_scenario (completed)
codex
"SeleniumLibrary"
["Browser","RequestsLibrary","SeleniumLibrary"]
tokens used 13,643
```

### opencode

`/tmp/opencode_e2e_output.txt`:

```json
"explicit_library_preference": "SeleniumLibrary"
"suggested_libraries": ["RequestsLibrary", "Browser", "SeleniumLibrary"]
```

Plus an interesting downstream observation captured in the opencode JSON event stream:

```json
"libraries_loaded": ["SeleniumLibrary", "BuiltIn"]
```

The session aggregate auto-loaded SeleniumLibrary as a direct consequence of the false `explicit_library_preference`. This is the cascade documented in PRD §2 consumer #2 (`session_models.py:787` `configure_from_scenario`) and PRD §2 consumer #8 (`selenium_plugin.py:202-208` eager init).

---

## Cross-CLI comparison

| Field | Claude | Codex | opencode |
|---|---|---|---|
| `explicit_library_preference` | `"SeleniumLibrary"` | `"SeleniumLibrary"` | `"SeleniumLibrary"` |
| `suggested_libraries` set | `{Browser, RequestsLibrary, SeleniumLibrary}` | same | same |
| Session library auto-loaded | (not observed — no session created at this step) | (not observed — only analyze called) | `SeleniumLibrary` + `BuiltIn` ← downstream cascade visible |

Three CLIs, one defect, consistent result. The bug is in the rf-mcp server's rule-based detector — NOT in any LLM client's interpretation. Every client faithfully reports what the tool returned.

---

## What this confirms

1. **Defect is real, deterministic, and tool-side.** No prompt-engineering, model-choice, or CLI-specific behaviour explains it. The string-pattern regex `\bopen\s+browser\b` at `library_detection.py:33` (weight 6, ≥ min_score 5) fires by itself on `"open browser"` and is sufficient to flag SeleniumLibrary as explicit.

2. **The downstream cascade is real.** opencode's `libraries_loaded` snapshot shows the session aggregate loaded SeleniumLibrary in response to the false preference — exactly the failure path the v4 fix targets in PRD §2 consumer table (8 sites).

3. **Three independent agents all faithfully relay the bug.** Agents do not second-guess the tool. If the tool says `SeleniumLibrary`, agents will write SL-style code. Without the fix, agents working through any of these CLIs against rf-mcp will:
   - Get SL keywords from `find_keywords` (consumer #1)
   - See an SL-loaded session (consumer #2)
   - Have SL pinned as primary recommendation (consumer #3)

4. **The fix lands at the rf-mcp pattern-detection layer.** No client-side work is needed. The v4 solution proposal `docs/proposals/explicit_library_detection_fix_proposal.md` describes the exact code change.

---

## Post-v6 verification — end-to-end coherence (2026-06-05)

v5 fixed the analysis path but Codex round-5 review found the session aggregate was still broken via a fallback substring heuristic. v6 closes that gap + 3 other source bugs. Re-ran the e2e protocol on the reported scenario AND the round-5-flagged regression case:

| CLI | Reported scenario `analysis.explicit_library_preference` | `Parse the XML response from the API` `analysis.explicit_library_preference` | `libraries_loaded` (downstream cascade) |
|---|---|---|---|
| Claude (direct Python) | `null` | `null` | (no session) |
| Codex CLI (via MCP) | `null` | `null` | `["BuiltIn"]` (no false SL or XML auto-load) |
| opencode (via MCP) | `null` | `null` | `["BuiltIn"]` (no false SL or XML auto-load) |

**Session-level coherence verified**: `ExecutionSession.detect_explicit_library_preference("Parse the XML response from the API")` returns `None` (was `"XML"` in v5 via the fallback heuristic, contradicting `analyze_scenario`'s `null` response).

**Multi-word migration verified** via Claude direct: `"Migrate from Selenium to Requests library"` → `RequestsLibrary` (was `None` in v5).

**Repeated-token negation verified**: `"do not use Playwright Playwright"` → `None` (was `Browser` in v5 due to under-subtraction).

**Newline negation verified**: `"do not use\nPlaywright"` → `None` (was `Browser` in v5 due to sentence split orphaning the target).

## Post-v5 verification — defect FIXED across all three CLIs (2026-06-05)

Re-ran the same protocol immediately after v5 implementation landed. Results:

| CLI | `explicit_library_preference` | `preference_source` | `libraries_loaded` (downstream) |
|---|---|---|---|
| Claude (direct Python) | `null` | `"rule"` | n/a (no session) |
| Codex CLI (via MCP) | `null` | `"rule"` | n/a (analyze only) |
| opencode (via MCP) | `null` | `"rule"` | `["BuiltIn"]` ← **SL no longer auto-loaded** |

**The downstream cascade is also fixed.** opencode's snapshot pre-fix showed `libraries_loaded: ["SeleniumLibrary", "BuiltIn"]` (SL eagerly auto-loaded because of the false preference, per PRD §2 consumer #2). Post-v5: `libraries_loaded: ["BuiltIn"]` — SL is not loaded because the explicit preference is correctly `null`.

Side-by-side comparison:

| Metric | Pre-v5 (defect) | Post-v5 (fix) |
|---|---|---|
| Claude → `explicit_library_preference` | `"SeleniumLibrary"` | `null` |
| Codex → `explicit_library_preference` | `"SeleniumLibrary"` | `null` |
| opencode → `explicit_library_preference` | `"SeleniumLibrary"` | `null` |
| opencode → `libraries_loaded` | `["SeleniumLibrary", "BuiltIn"]` | `["BuiltIn"]` |
| `preference_source` field exposed | N/A (not in API) | `"rule"` (v5 new field) |

Truly-explicit cases (verified in unit tests, not re-run via all three CLIs):
- `"Use playwright to test demoshop"` → Browser (with `explicit_library_evidence` populated)
- `"do not use Selenium, instead use Playwright"` → Browser (sentence-scoped negation working)
- `"Test both selenium and playwright sites and compare"` → null + `library_preference_conflicts.web_automation`

## Verification plan (post-fix) — historical

When the v4 fix was implemented, re-run this exact protocol. Expected outcome:

| Field | Pre-fix (current) | Post-fix (target) |
|---|---|---|
| `explicit_library_preference` | `"SeleniumLibrary"` | `null` |
| `suggested_libraries` | unchanged (advisory list) | unchanged (`_determine_capabilities` is out of scope per PRD §FR-8) |
| `preference_source` (new v4 field) | absent | `"rule"` |
| Session auto-load | `SeleniumLibrary` eagerly loaded | not loaded (None preference → no auto-import per PRD §2 consumer #2) |
| `library_preference_conflicts` | absent | absent (no conflict in this scenario — neither Browser nor SL explicitly mentioned) |

For the opposite case (`scenario = "Use playwright to test demoshop"`), v4 must return `explicit_library_preference: "Browser"` + populated `explicit_library_evidence`.

For the v3 round-3-flagged edge case `"do not use Selenium, instead use Playwright"`, v4 must return `"Browser"` (empirically verified against the v4 algorithm in the proposal §3.1 Step 6 walkthroughs).

---

## Reproduction commands

For future verification — exact commands used to produce this report:

```bash
# Claude (direct Python)
uv run python -c "
import asyncio
from robotmcp.components.nlp_processor import NaturalLanguageProcessor
async def main():
    nlp = NaturalLanguageProcessor()
    r = await nlp.analyze_scenario(
        'Test e-commerce website https://demoshop.makrocode.de: open browser, '
        'add items to shopping cart, verify items, complete checkout, and close browser',
        context='web')
    import json; print(json.dumps(r['analysis'], indent=2))
asyncio.run(main())"

# Codex (ephemeral MCP config)
codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
  -c 'mcp_servers.robotmcp.command="uv"' \
  -c 'mcp_servers.robotmcp.args=["run","--directory","/home/many/workspace/rf-mcp","-m","robotmcp.server"]' \
  'Call analyze_scenario with scenario="Test e-commerce website ..." context="web" session_id="demoshop_codex_e2e" and report explicit_library_preference + suggested_libraries.'

# opencode (pre-registered MCP)
opencode run --format json -m github-copilot/claude-sonnet-4.6 \
  'Call analyze_scenario with scenario="Test e-commerce website ..." context="web" session_id="demoshop_opencode_e2e" and report explicit_library_preference + suggested_libraries.'
```
