# Instruction Templates Guide

## Overview

rf-mcp uses **MCP server instructions** (introduced in v0.29.0, ADR-002) to guide LLMs toward a "discover-then-act" workflow when interacting with Robot Framework. Without instructions, smaller or faster LLMs tend to guess keyword names and element locators, leading to failed tool calls, wasted tokens, and poor test quality.

Instructions are delivered through the MCP protocol's `instructions` field in the `initialize` response. Every MCP client receives them automatically — no prompt engineering required on the client side.

## How It Works

When rf-mcp starts, the server reads three environment variables, resolves the appropriate instruction text, and passes it to FastMCP:

```
Environment Variables
        │
        ▼
FastMCPInstructionAdapter.create_config_from_env()
        │
        ▼
InstructionResolver.resolve(config, context)
        │
        ▼
InstructionValidator.validate(content)
        │
        ▼
FastMCP("Robot Framework MCP Server", instructions=<resolved text>)
        │
        ▼
MCP initialize response → LLM client receives instructions
```

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `ROBOTMCP_INSTRUCTIONS` | `off`, `default`, `custom` | `default` | Master switch — controls whether instructions are active |
| `ROBOTMCP_INSTRUCTIONS_TEMPLATE` | `minimal`, `standard`, `detailed`, `browser-focused`, `api-focused` | `standard` | Selects the built-in template (only used when mode is `default`) |
| `ROBOTMCP_INSTRUCTIONS_FILE` | File path (`.txt`, `.md`, `.instruction`, `.instructions`) | *(none)* | Path to custom instructions file (only used when mode is `custom`) |

### Precedence Rules

1. `ROBOTMCP_INSTRUCTIONS=off` always wins — even if a template or file is specified, instructions are disabled.
2. `ROBOTMCP_INSTRUCTIONS=custom` requires `ROBOTMCP_INSTRUCTIONS_FILE` to be set. If the file is missing or invalid, the server falls back to `default` mode.
3. `ROBOTMCP_INSTRUCTIONS=default` ignores `ROBOTMCP_INSTRUCTIONS_FILE` entirely.
4. If `ROBOTMCP_INSTRUCTIONS` is empty or unset, mode defaults to `default`.

## Built-in Templates

### `minimal` (~200 tokens, ~300 chars)

Best for: **Capable LLMs** (Claude Opus, GPT-4, Gemini Pro)

These models already follow instructions well and need only a brief reminder. The minimal template reduces token overhead to almost nothing.

```
Use discovery tools (find_keywords, get_keyword_info) before executing keywords.
Verify keywords exist via discovery, never guess keyword names or arguments.
For DOM inspection: get_session_state(sections=["page_source"], include_reduced_dom=True).
```

### `standard` (~400 tokens, ~1200 chars) — **Default**

Best for: **Mid-range LLMs** (Claude Sonnet, GPT-4o, Gemini Flash)

A balanced template with a structured workflow guide covering discovery, locator guidance, session management, DOM inspection, error recovery, and multi-test suites.

Key sections:
1. DISCOVER before EXECUTE
2. GET LOCATOR GUIDANCE before INTERACT
3. SESSION MANAGEMENT & DOM INSPECTION
4. ERROR RECOVERY
5. MULTI-TEST SUITES

Contains the `{available_tools}` placeholder that is automatically filled with the list of discovery tools.

### `detailed` (~600 tokens, ~1800 chars)

Best for: **Smaller LLMs** (Claude Haiku, GPT-4o-mini, Gemini Flash 8B)

A step-by-step guide that walks the LLM through the entire workflow with concrete examples. Each step includes the exact tool call and arguments to use.

Key sections:
- STEP 1: START SESSION — with `manage_session` example
- STEP 2: DISCOVER KEYWORDS — with `find_keywords` and `get_keyword_info` examples
- STEP 3: INSPECT THE PAGE — with `get_session_state` ARIA snapshot usage
- STEP 4: EXECUTE STEPS — with correct vs. incorrect locator examples
- STEP 5: HANDLE ERRORS — recovery strategies

### `browser-focused` (~350 tokens, ~1000 chars)

Best for: **Web automation only** scenarios

A streamlined template that focuses entirely on the Browser Library workflow: opening pages, finding elements via ARIA snapshots, and clicking/filling. Lists common Browser Library keywords and the 3-step pattern: `manage_session → get_session_state → execute_step`.

### `api-focused` (~300 tokens, ~800 chars)

Best for: **API testing only** scenarios

Focuses on the RequestsLibrary workflow: creating sessions, making HTTP requests, and asserting responses. Omits all DOM/locator guidance since it's irrelevant for API testing.

## Configuration Examples

### Example 1: Default Setup (Most Common)

Use the standard template — works well for most LLMs.

**.mcp.json (Claude Code / VS Code):**
```json
{
  "mcpServers": {
    "robotmcp": {
      "command": "uv",
      "args": ["run", "-m", "robotmcp.server"],
      "env": {
        "ROBOTMCP_INSTRUCTIONS": "default",
        "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "standard"
      }
    }
  }
}
```

**Shell:**
```bash
export ROBOTMCP_INSTRUCTIONS=default
export ROBOTMCP_INSTRUCTIONS_TEMPLATE=standard
uv run -m robotmcp.server
```

### Example 2: Minimal for Capable Models

Reduce token overhead when using Claude Opus or GPT-4.

```json
{
  "mcpServers": {
    "robotmcp": {
      "command": "uv",
      "args": ["run", "-m", "robotmcp.server"],
      "env": {
        "ROBOTMCP_INSTRUCTIONS": "default",
        "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "minimal"
      }
    }
  }
}
```

### Example 3: Detailed for Small Models

Maximize guidance for Claude Haiku or GPT-4o-mini.

```json
{
  "mcpServers": {
    "robotmcp": {
      "command": "uv",
      "args": ["run", "-m", "robotmcp.server"],
      "env": {
        "ROBOTMCP_INSTRUCTIONS": "default",
        "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "detailed"
      }
    }
  }
}
```

### Example 4: Browser-Only Testing

Use the browser-focused template for dedicated web test automation.

```json
{
  "mcpServers": {
    "robotmcp": {
      "command": "uv",
      "args": ["run", "-m", "robotmcp.server"],
      "env": {
        "ROBOTMCP_INSTRUCTIONS": "default",
        "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "browser-focused"
      }
    }
  }
}
```

### Example 5: API-Only Testing

Use the api-focused template for REST API test automation.

```json
{
  "mcpServers": {
    "robotmcp": {
      "command": "uv",
      "args": ["run", "-m", "robotmcp.server"],
      "env": {
        "ROBOTMCP_INSTRUCTIONS": "default",
        "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "api-focused"
      }
    }
  }
}
```

### Example 6: Custom Instructions from File

Provide your own instruction text.

**my_instructions.txt:**
```text
COMPANY TESTING STANDARDS:

1. Always initialize sessions with libraries=["Browser", "BuiltIn", "String"]
2. Use get_session_state with include_reduced_dom=True before any interaction
3. Log all test steps with BuiltIn.Log keyword
4. Use {available_tools} for keyword discovery
5. Never hardcode URLs — use environment variables via Get Variable Value
6. Always close browser sessions after tests complete
```

**.mcp.json:**
```json
{
  "mcpServers": {
    "robotmcp": {
      "command": "uv",
      "args": ["run", "-m", "robotmcp.server"],
      "env": {
        "ROBOTMCP_INSTRUCTIONS": "custom",
        "ROBOTMCP_INSTRUCTIONS_FILE": "./my_instructions.txt"
      }
    }
  }
}
```

The `{available_tools}` placeholder in custom files is automatically substituted with the list of discovery tools.

### Example 7: Disable Instructions Entirely

Turn off instructions for benchmarking or when the client provides its own system prompt.

```json
{
  "mcpServers": {
    "robotmcp": {
      "command": "uv",
      "args": ["run", "-m", "robotmcp.server"],
      "env": {
        "ROBOTMCP_INSTRUCTIONS": "off"
      }
    }
  }
}
```

## Custom Instruction Files

### Allowed File Extensions

Only these extensions are accepted (security measure):
- `.txt`
- `.md`
- `.instruction`
- `.instructions`

Other extensions (`.py`, `.json`, `.yaml`, `.sh`, `.exe`, etc.) are rejected with an error.

### Path Security

- **No directory traversal**: Paths containing `..` are rejected.
- **Resolved paths**: Relative paths are resolved against the working directory and checked to ensure they don't escape the base directory.

### Content Constraints

| Constraint | Value | Description |
|------------|-------|-------------|
| Minimum length | 10 chars | Prevents empty or trivially short instructions |
| Maximum length | 50,000 chars | Prevents excessively large instructions |
| Token budget | 1,000 tokens (default) | Warns if instructions are too long for small context models |

### Placeholders in Custom Files

Custom files support `{placeholder}` substitution. The following placeholders are filled automatically:

| Placeholder | Value |
|-------------|-------|
| `{available_tools}` | `find_keywords, get_keyword_info, get_session_state, get_locator_guidance, analyze_scenario, recommend_libraries, check_library_availability` |

Example custom file using placeholders:

```text
Use these tools before executing any keyword: {available_tools}

Never guess locators. Always inspect the page first with get_session_state.
```

### Validation

Custom instruction content is validated for:
- **Dangerous patterns**: `<script>`, `javascript:`, `eval()`, `exec()`, `__import__()` are rejected.
- **Recommended keywords**: Warnings (not errors) if the content doesn't mention `discovery`, `snapshot`, `element`, or `locator`.
- **Token budget**: Error if content exceeds the configured token budget.

If a custom file fails validation, the server automatically falls back to the default `standard` template and logs a warning.

## Fallback Behavior

The instruction system is designed to never prevent server startup:

| Failure | Behavior |
|---------|----------|
| Custom file not found | Falls back to default template, logs warning |
| Custom file fails validation | Falls back to default template, logs warning |
| Invalid file extension in env var | Falls back to default template, logs error |
| Invalid template name in env var | Falls back to `standard` template, logs warning |
| Any unexpected exception | Server starts without instructions, logs warning |

## Template Comparison

| Template | Tokens | Best For | Key Feature |
|----------|--------|----------|-------------|
| `minimal` | ~40 | Opus, GPT-4 | 3-line reminder only |
| `standard` | ~400 | Sonnet, GPT-4o | Structured 5-section workflow |
| `detailed` | ~600 | Haiku, GPT-4o-mini | Step-by-step with examples |
| `browser-focused` | ~350 | Web-only testing | Browser keywords + ARIA workflow |
| `api-focused` | ~300 | API-only testing | RequestsLibrary + HTTP methods |

## How Instructions Reach the LLM

The instructions are set on the FastMCP server at startup:

```python
# In server.py
instructions = _resolve_server_instructions()
mcp = FastMCP("Robot Framework MCP Server", instructions=instructions)
```

When an MCP client connects and sends `initialize`, the server responds with:

```json
{
  "protocolVersion": "2024-11-05",
  "capabilities": { ... },
  "serverInfo": { "name": "Robot Framework MCP Server" },
  "instructions": "rf-mcp WORKFLOW GUIDE:\n\n1. DISCOVER before EXECUTE..."
}
```

The client (Claude, GPT, Gemini, etc.) receives the `instructions` field and treats it as server-level guidance for how to use the tools.

## Programmatic Usage

The instruction system can also be used programmatically for advanced integrations:

```python
from robotmcp.domains.instruction import (
    FastMCPInstructionAdapter,
    InstructionConfig,
    InstructionTemplate,
    InstructionResolver,
    InstructionRenderer,
)

# 1. Load config from environment (standard usage)
adapter = FastMCPInstructionAdapter()
config = adapter.create_config_from_env()
instructions = adapter.get_server_instructions(config)

# 2. Force a specific template programmatically
adapter = FastMCPInstructionAdapter(template_name="detailed")
config = InstructionConfig.create_default()
instructions = adapter.get_server_instructions(
    config,
    context={"available_tools": "find_keywords, get_keyword_info"},
)

# 3. Render a template directly
template = InstructionTemplate.get_by_name("browser-focused")
content = template.render({"available_tools": "find_keywords, get_keyword_info"})
print(content.value)
print(f"~{content.token_estimate} tokens")

# 4. Render for a specific LLM format
renderer = InstructionRenderer()
claude_fmt = renderer.render(content, target=InstructionRenderer.TargetFormat.CLAUDE)
# Wraps in <instructions>...</instructions>

openai_fmt = renderer.render(content, target=InstructionRenderer.TargetFormat.OPENAI)
# Wraps in # System Instructions ... ---

# 5. Create a fully custom template
custom = InstructionTemplate(
    template_id="my_company",
    content="Rules for {team}: always use {available_tools} before acting.",
    description="Company-specific template",
    placeholders=("team", "available_tools"),
)
result = custom.render({"team": "QA", "available_tools": "find_keywords"})
```

## Additional Built-in Templates

Beyond the five environment-selectable templates, the code also includes:

| Template | ID | Purpose |
|----------|----|---------|
| `discovery_first` | `discovery_first` | Comprehensive 6-section discovery-focused guide (used internally by the resolver as the default when `ROBOTMCP_INSTRUCTIONS_TEMPLATE` is not set to one of the five named templates) |
| `locator_prevention` | `locator_prevention` | Strict "MUST NOT / MUST" rules focusing exclusively on preventing keyword and locator guessing |

These can be accessed programmatically via `InstructionTemplate.discovery_first()`, `InstructionTemplate.locator_prevention()`, or `InstructionTemplate.get_by_name("discovery_first")`.

## Troubleshooting

**Instructions not appearing in client:**
- Check that `ROBOTMCP_INSTRUCTIONS` is not set to `off`.
- Verify the MCP client supports the `instructions` field in the `initialize` response (MCP protocol 2024-11-05+).

**Custom file not loading:**
- Check that the file extension is `.txt`, `.md`, `.instruction`, or `.instructions`.
- Verify the file path does not contain `..` (directory traversal is blocked).
- Check server logs for `"Custom instruction file not found"` or `"Invalid instruction path"` messages.

**Template not changing:**
- Ensure `ROBOTMCP_INSTRUCTIONS=default` is set (template selection only applies in default mode).
- Verify the template name exactly matches one of: `minimal`, `standard`, `detailed`, `browser-focused`, `api-focused`.
- Template names are case-insensitive and trimmed.

**Token budget warning:**
- The default token budget is 1,000 tokens. Custom instructions exceeding this will log a validation error and fall back to the default template.
- For longer custom instructions, this is a warning only — the instructions will still be used if under the 50,000 character hard limit.
