# Library-aware skill retrieval: evaluation

This harness measures whether the skill pointers of `library-aware-skill-retrieval` (`project_skills` on the first
result, `skill` links on hints) help small models write working Robot Framework tests. It covers design D8 and
tasks 8.1–8.2 of `openspec/changes/library-aware-skill-retrieval`.

A model on OpenRouter works through tool calls against an in-memory rf-mcp server. The server's instructions are
the system prompt, and the run ends when the model stops calling tools. Each run gets a fresh project whose
`.robotframework/rf-mcp.toml` points at the evaluation interpreter, so rf-mcp sees the project's real libraries.

| arm | skills | pointers |
| --- | --- | --- |
| A | rf-mcp's built-in skills | on (built-in catalogue) |
| B | official skills from a robotframework-plugins checkout via `--skills-dir`, then built-in | on |
| C | as B | off (`skill_pointers=False`) |

The tasks run against the cookbook fixtures in `tests/fixtures`:

- **browser**: Browser library on the widget lab page: a payment form in an iframe, and a coupon form in an iframe
  inside a shadow root.
- **requests**: RequestsLibrary against the Ledger API: a bearer token, a nested payload, money arithmetic, and a
  negative test.

**Grading.** The harness grades through the same server's tools:
- **pass** means `rf_analyze` is clean, `rf_run` passes, and the suite contains every required test name.
- **bug caught** means the passing suite also fails against a broken copy of the app: the payment page reports the
  wrong digits, and the ledger computes 16% VAT.

## Setup

```bash
# evaluation interpreter (outside the repo)
uv venv --python 3.12 <scratch>/evalenv
uv pip install --python <scratch>/evalenv/Scripts/python.exe "robotframework>=7.3" "robotcode[all]==<rf-mcp's robotcode>" \
    "robotframework-browser>=20,<21" "robotframework-requests>=0.9.7" fastapi uvicorn python-multipart strawberry-graphql
<scratch>/evalenv/Scripts/rfbrowser init chromium

# official skills at a pinned commit
git clone https://github.com/MarketSquare/robotframework-plugins <scratch>/official
git -C <scratch>/official checkout <commit>
```

`OPENROUTER_API_KEY` (and optionally `OPENROUTER_BASE_URL`) come from the environment or the repository's `.env`.
The harness never prints them or writes them to its output.

## Run

```bash
uv run python experiments/library-aware-skills-eval/harness.py \
    --python <scratch>/evalenv/Scripts/python.exe --official <scratch>/official \
    --models deepseek/deepseek-v4.1-flash-1017 openai/gpt-6-luna minimax/minimax-m3.1-flash \
    --reps 5 --jobs 3 --workdir <scratch>/eval-work --out <scratch>/eval-work/results.jsonl
uv run python experiments/library-aware-skills-eval/report.py <scratch>/eval-work/results.jsonl
```

Results are appended one JSON line per run. A second invocation skips the runs already in the file, so an
interrupted run can be continued. Each run's project, suite and transcript stay under `<workdir>/runs/`.
