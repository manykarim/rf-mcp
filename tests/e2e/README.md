# E2E AI Agent Testing

This directory contains end-to-end tests for validating AI agent tool discovery and usage patterns.

## Quick Start

```bash
# Run all E2E tests
uv run pytest tests/e2e/ -v

# Run with real LLM (requires OPENAI_API_KEY)
USE_REAL_LLM=true uv run pytest tests/e2e/ -v

# Run specific test
uv run pytest tests/e2e/test_agent_tool_discovery.py::TestAgentToolDiscovery::test_mcp_tools_discoverable -v
```

## Directory Structure

```
tests/e2e/
├── README.md                    # This file
├── models.py                    # Pydantic models
├── metrics_collector.py         # Metrics tracking
├── fixtures.py                  # Pytest fixtures
├── test_agent_tool_discovery.py # Tests
├── scenarios/                   # YAML scenario definitions
│   ├── todomvc_browser.yaml
│   ├── todomvc_selenium.yaml
│   ├── restful_booker_api.yaml
│   └── xml_testing.yaml
└── metrics/                     # Generated JSON metrics
```

## What's Tested

- ✅ MCP tool discoverability
- ✅ Tool call tracking and metrics
- ✅ Tool hit rate calculation
- ✅ Realistic scenario execution (4 scenarios)

## Key Metrics

**Tool Hit Rate**: Percentage of expected tools called correctly

```
Tool Hit Rate = (Expected Tools Met) / (Total Expected Tools)
```

## Adding Scenarios

Create a new YAML file in `scenarios/`:

```yaml
id: my_scenario
name: My Test Scenario
description: What this tests
context: web
prompt: |
  The full prompt given to the AI agent

  For web UI scenarios, ALWAYS include headless mode:
  - Browser Library: Please use headless=True (New Browser with headless=True)
  - Selenium Library: Please use headlesschrome or headlessfirefox
expected_tools:
  - tool_name: analyze_scenario
    min_calls: 1
    max_calls: 1
expected_outcome: What should happen
min_tool_hit_rate: 0.8
tags: [web, my-feature]
```

**Important**: For web UI scenarios, always specify headless mode in the prompt to ensure tests can run in CI/CD environments without a display server.

## Documentation

See [docs/e2e_testing_implementation.md](../../docs/e2e_testing_implementation.md) for complete documentation.
