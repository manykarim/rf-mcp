## 1. Upgrade to FastMCP 3.x (gated step)

- [x] 1.1 Pin `fastmcp>=3.0` in `pyproject.toml` (from `>=2.8.0`); run `uv lock` + `uv sync`.
- [x] 1.2 Verify `fastmcp.__version__` starts with `3.` and `ToolError(msg, log_level=logging.WARNING)` constructs (log_level == 30).
- [x] 1.3 Confirm `requires-python` stays `>=3.10` and the CI matrix (3.10/3.11/3.12) is unchanged — fastmcp 3.4.4 / starlette 1.3.1 / mcp all require `>=3.10` (verified). Only adjust if `uv lock` surfaces a higher floor.

## 2. Test-harness migration for 3.x

- [x] 2.1 Reuse the existing `robotmcp.compat.fastmcp_compat.get_tool_fn(tool)` (already 2.x/3.x-aware: `getattr(tool_obj, "fn", tool_obj)`) — no new helper needed.
- [x] 2.2 Apply `tool_fn(...)` in the test files that fail with `'function' object has no attribute 'fn'` (~9 files: test_desktop_mcp_workflow_correctness, test_test_scoping_integrity, test_batch_resume_args, test_explicit_library_detection_fix, test_get_session_state_bridge_improvements, test_debug_bridge_unification, test_builtin_and_attach_improvements, and any others surfaced).
- [x] 2.3 Update `test_fastmcp_compat.py` assertions for FastMCP 3.x (version detection, disabled-tool kwargs, get-tool-fn).

## 3. Expected step failures → ToolError(WARNING)

- [x] 3.1 Add the compat helper `tool_error(message, level=logging.WARNING)` (in `robotmcp/compat` / `fastmcp_compat`): return `ToolError(message, log_level=level)`, falling back to `ToolError(message)` on `TypeError` (2.x). 
- [x] 3.2 Convert `server.py:4572` (failed RF step) from `raise Exception(detailed_error)` to `raise tool_error(detailed_error)`.
- [x] 3.3 Convert `server.py:4448` (attach-bridge connectivity, strict/force) to `raise tool_error(<connectivity message + guidance>)`.
- [x] 3.4 Keep `_CollapseFrameworkTracebackFilter` as-is (defensive net for non-converted paths / accidental 2.x).

## 4. Tests

- [x] 4.1 Payload-preservation regression: a failing `execute_step` returns `isError=True` AND the text content still contains (a) the RF error string, (b) the suggested-keyword / a hint, (c) the `step_id`. This is the tripwire for the 2.x TypeError-swallow and any masking-default flip.
- [x] 4.2 Constructor/version guard: `tool_error(msg)` yields a `ToolError` whose `log_level == logging.WARNING` on the resolved runtime (or falls back cleanly).
- [x] 4.3 Attach path (4448): a failing attach returns `isError=True` with the connectivity message + guidance intact.
- [x] 4.4 Existing `pytest.raises(Exception)` / `(Exception, ToolError)` tests remain valid (ToolError subclasses Exception; messages preserved with masking off) — no weakening.

## 5. Verify (upgrade gate)

- [x] 5.1 Full unit suite green on 3.x (baseline ~7000+); MCP handshake still answers (no import/timing regression from starlette/mcp bump).
- [x] 5.2 Clean-room docker probe: a failed step logs one WARNING line (no traceback) on stderr; the tool result still carries the RF error + hint + step_id; 0 misrouted JSON-RPC.
- [x] 5.3 Record before/after (ERROR+traceback → WARNING one-liner) in the archive notes.
