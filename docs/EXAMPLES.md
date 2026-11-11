# Examples

Typical tool sequences an agent follows with rf-mcp. The arguments are what the agent sends; the agent chooses them
from the server instructions and the `rf-robotcode` skill, so a user only asks in plain language. Parameters are
listed in [MCP_TOOLS.md](MCP_TOOLS.md).

## Find out why a test failed

> "Why did *Checkout Total* fail in the last run?"

1. Read the result before any source file:

   ```json
   rf_results {"action": "log", "by_longname": ["Tests.Checkout.Checkout Total"], "failed_only": true}
   ```

   The log shows the failing keyword, its arguments, the message and the hints attached to it.
2. Open only the keyword or resource the log points to, fix it, and rerun that one test:

   ```json
   rf_run {"by_longname": ["Tests.Checkout.Checkout Total"]}
   ```

3. The task is done when the response starts with `outcome: PASS`.

`rf_results` with `action='stats'` and `by=['tag']` shows which areas fail; `action='diff'` with `baseline` lists
new failures compared with an earlier result file.

## Run a selection

```json
rf_discover {"action": "tags"}
rf_run {"include_tags": ["smoke"], "exclude_tags": ["wip"]}
rf_run {"by_longname": ["Tests.Login.Valid Login", "Tests.Login.Locked User"], "profile": "ci"}
```

Select with longnames or tags rather than a single file in `paths`: a file passed alone becomes the top suite, and
the setups in parent `__init__.robot` files do not run.

## Rerun failures and merge

```json
rf_run {"output": "full.xml"}
rf_run {"rerunfailed": "results/full.xml", "output": "rerun.xml"}
rf_results {"action": "merge", "merge_inputs": ["results/full.xml", "results/rerun.xml"], "output_path": "results/merged.xml"}
```

The paths depend on the output directory set in `robot.toml`.

## Write a new test

> "Write a test that searches for *robot* and checks the first result."

1. Look up the keywords instead of guessing them, and check the project conventions:

   ```json
   rf_doc {"action": "keywords", "library": "Browser", "pattern": ["fill", "type", "text"]}
   read_resource {"uri": "skill://shop-conventions/SKILL.md"}
   ```

2. Try the steps live. One session keeps imports, variables and the open browser:

   ```json
   rf_repl {"action": "start", "source": "resources/common.resource"}
   rf_repl {"action": "send", "session_id": "...", "statement": "New Page    https://shop.example.com\nGet Aria Snapshot    main"}
   rf_repl {"action": "send", "session_id": "...", "statement": "Fill Text    role=searchbox    robot\nKeyboard Key    press    Enter\nGet Text    css=.result >> nth=0    contains    Robot"}
   ```

   Several lines in one send run in order and stop at the first failure.
3. Save the passing steps as a test, then close the session:

   ```json
   rf_repl {"action": "save", "session_id": "...", "save_path": "tests/search.robot", "test_name": "Search Finds Robot"}
   rf_repl {"action": "stop", "session_id": "..."}
   ```

4. Check and run the new test:

   ```json
   rf_analyze {"paths": ["tests/search.robot"]}
   rf_run {"by_longname": ["Tests.Search.Search Finds Robot"]}
   ```

   Analysis is fast and finds wrong keyword names and arguments before a run; it executes nothing.

## Debug a failing test in place

> "*Valid Login* fails on CI but not locally. Find out why."

```json
rf_debug {"action": "start", "by_longname": ["Tests.Login.Valid Login"]}
rf_debug {"action": "send", "session_id": "...", "command": ".where\n.vars --user\nGet Url"}
rf_debug {"action": "send", "session_id": "...", "command": "Get Element Count    css=.error-banner"}
rf_debug {"action": "stop", "session_id": "..."}
```

The test pauses at the first failure (or at `breakpoints` such as `tests/login.robot:42`) with the real suite setup
and variables. Debugger commands start with a dot; other lines run as keywords in the paused frame. The debugger
does not overwrite `output.xml`.

## Check a project without running it

```json
rf_analyze {"severity": "warning"}
rf_analyze {"collect_unused": true}
rf_analyze {"code": "KeywordNotFound"}
```

`severity` reports that severity and worse. A clean result says so and reminds the agent that nothing ran.

## Remote test runner

The server runs in a container with the browsers (see [DOCKER.md](DOCKER.md)); the agent edits files on its own
machine and sends them along with the call:

```json
rf_analyze {"files": {"tests/search.robot": "*** Test Cases ***\nSearch Finds Robot\n    ..."}, "paths": ["tests/search.robot"]}
rf_run {"files": {"tests/search.robot": "..."}, "by_longname": ["Tests.Search.Search Finds Robot"]}
```

## Client mode

With `--mode client` (or `target='client'` on one call) tools return the command instead of running it:

```json
rf_run {"include_tags": ["smoke"], "target": "client", "shell": "bash"}
```

```json
{"mode": "client",
 "command": ["robotcode", "-r", "/home/dev/shop", "robot", "-i", "smoke"],
 "shell_command": "ROBOTCODE_FORCE_AI_AGENT=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 robotcode -r /home/dev/shop robot -i smoke",
 "instruction": "Execute `shell_command` locally on the client machine using your terminal/shell tool. ..."}
```

The agent runs `shell_command` in its shell. If it fails, the agent passes the output to `rf_diagnose`:

```json
rf_diagnose {"error_text": "TimeoutError: locator.click: Timeout 10000ms exceeded. waiting for locator('#login')", "keyword": "Click", "libraries": ["Browser"]}
```

```json
{"keyword": "Click", "hints_count": 1,
 "hints": [{"id": "browser-locator-timeout", "title": "Browser locator timed out waiting for element",
            "message": "The selector did not match a visible/enabled element within the timeout. Capture the current page structure with `Get Aria Snapshot    css=body` or `Get Url` ...",
            "fix_examples": ["Get Aria Snapshot    css=body", "Get Url", "Wait For Elements State    <selector>    visible    timeout=5s"],
            "relevance": 88}]}
```
