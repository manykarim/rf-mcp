# AILibrary Examples

This directory contains Robot Framework test examples demonstrating AILibrary usage.

## Prerequisites

1. Install rf-mcp with the lib extra:
   ```bash
   pip install rf-mcp[lib]
   ```

2. Set up your API key:
   ```bash
   export ANTHROPIC_API_KEY=your-key-here
   # or for OpenAI:
   export OPENAI_API_KEY=your-key-here
   ```

3. Install Playwright browsers (for Browser library examples):
   ```bash
   rfbrowser init
   ```

## Examples

| File | Description |
|------|-------------|
| `01_basic_usage.robot` | Core keywords: Do, Check, Ask |
| `02_recording_export.robot` | Recording and exporting test suites |
| `03_retry_configuration.robot` | Handling dynamic/slow content |
| `04_provider_examples.robot` | Using different AI providers |
| `05_mixed_keywords.robot` | Combining AI with traditional RF keywords |
| `06_form_testing.robot` | Form validation and input testing |
| `ai_config.yaml` | Example YAML configuration file |

## Running Examples

Run a single example:
```bash
robot examples/ailibrary/01_basic_usage.robot
```

Run all examples:
```bash
robot examples/ailibrary/
```

Run with specific provider:
```bash
# OpenAI
robot --variable PROVIDER:openai examples/ailibrary/01_basic_usage.robot

# Ollama (local)
robot --variable PROVIDER:ollama examples/ailibrary/01_basic_usage.robot
```

## Output

Test results are saved to:
- `output.xml` - Full test results
- `log.html` - Detailed log
- `report.html` - Summary report
- `generated/` - Exported test files (from recording examples)

## Customization

Modify `ai_config.yaml` to change default settings, or pass configuration directly in the test:

```robot
Library    AILibrary
...    provider=anthropic
...    model=claude-sonnet-4-20250514
...    retries=5
...    retry_delay=2s
```
