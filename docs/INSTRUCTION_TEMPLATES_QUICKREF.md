# Instruction Templates Quick Reference

rf-mcp sends server-level instructions to LLMs via the MCP `initialize` response, guiding them to discover keywords before executing them. Configure with three environment variables:

| Variable | Values | Default |
|----------|--------|---------|
| `ROBOTMCP_INSTRUCTIONS` | `off` / `default` / `custom` | `default` |
| `ROBOTMCP_INSTRUCTIONS_TEMPLATE` | `minimal` / `standard` / `detailed` / `browser-focused` / `api-focused` | `standard` |
| `ROBOTMCP_INSTRUCTIONS_FILE` | Path to `.txt` or `.md` file | *(none, required when mode=custom)* |

## Templates at a Glance

| Template | Tokens | Use When |
|----------|--------|----------|
| `minimal` | ~40 | Opus, GPT-4 — needs only a brief reminder |
| `standard` | ~400 | Sonnet, GPT-4o — balanced workflow guide |
| `detailed` | ~600 | Haiku, GPT-4o-mini — step-by-step with examples |
| `browser-focused` | ~350 | Web-only testing scenarios |
| `api-focused` | ~300 | API-only testing scenarios |

## Example .mcp.json

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

Custom files support `{available_tools}` placeholder substitution. If a custom file is missing or fails validation, the server falls back to the `standard` template automatically.

See [INSTRUCTION_TEMPLATES_GUIDE.md](./INSTRUCTION_TEMPLATES_GUIDE.md) for full details.
