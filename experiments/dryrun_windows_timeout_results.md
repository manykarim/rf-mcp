# Windows dry-run timeout — experiment results

Diagnostics from `dryrun_windows_timeout_diagnostics.md`, run on the Windows client
in the same environment rf-mcp uses. **Bottom line: the hang is a Windows-specific
inherited-stdin-pipe deadlock in the dry-run subprocess — NOT a PlatynUI problem.**

---

## TL;DR — root cause (proven)

The MCP server (`robotmcp.server`) is a long-lived stdio process that keeps a
**pending synchronous blocking read on its stdin pipe** (waiting for the next MCP
request). Its dry-run helper runs:

```python
# robotmcp/components/execution/suite_execution_service.py  (~L324)
subprocess.run([sys.executable, "-m", "robot", *rf_options],
               capture_output=True, text=True, encoding="utf-8",
               errors="replace", timeout=timeout_s)
```

`capture_output=True` redirects **stdout/stderr but not stdin**, so the child
`robot` process **inherits the server's stdin pipe handle**. On Windows, a
*synchronous* file object serializes all operations, so the child's C-runtime
`fflush → _lseeki64 → SetFilePointerEx → NtQueryInformationFile` on that shared
handle **blocks behind the parent's pending read** — a deadlock that only ends when
rf-mcp's 180 s timeout fires. On Linux the same inheritance is harmless (no such
serialization), which is why Linux finishes in ~0.9 s.

- **Not hypothesis A or B** (PlatynUI import deadlock / background thread): import,
  libdoc, and instantiation are all fast and exit cleanly; the hung stack contains
  **no PlatynUI/robot/import frames**.
- **A variant of hypothesis C** — but the pipe that deadlocks is the inherited
  **stdin**, not a leaked helper holding the capture pipe.
- **Suite-agnostic**: a trivial `Log hi` suite hangs too (both standalone repro and
  end-to-end through the real MCP server).

### Fix (validated standalone)

Give the dry-run child a stdin that is **not** the inherited pipe:

```python
subprocess.run([...], capture_output=True, text=True, encoding="utf-8",
               errors="replace", timeout=timeout_s,
               stdin=subprocess.DEVNULL)            # <-- the fix
```

With `stdin=subprocess.DEVNULL` the deadlock disappears even while a pending read
exists (trial 3 below). Secondary hardening: a real process-tree kill on timeout
(§8), and mirroring the normal-run path's `--console none` for dry-run.

---

## §0 Setup / environment

### 0.1 Same Python as MCP server
- `sys.version`: 3.13.14 (main, Jun 11 2026) [MSC v.1944 64 bit (AMD64)]
- `sys.executable`: `C:\workspace\ai-in-qa-demo\.venv\Scripts\python.exe`
- Invoked via `uv run python` (uv-managed workspace).
- Versions: `robotframework` 7.4.2, `robotframework-platynui` 0.13.0.dev2,
  `platynui-native` 0.13.0.dev2, `platynui-cli` 0.13.0.dev2.
- py-spy 0.4.2 installed via `uv pip install py-spy`.

### 0.2 Launch context (E0)
- User `dsv\many.kasiriha`, **interactive console** session.
- OS: Windows 11 Enterprise, build 10.0.26200.
- MCP launch: `uv run python -m robotmcp.server --with-frontend`, cwd `c:/workspace/ai-in-qa-demo`.
- **Process topology:** `.venv\Scripts\python.exe` is a **uv trampoline launcher**, not a
  real interpreter. Every invocation is a two-process chain:
  `.venv\Scripts\python.exe` (shim) → `…\uv\python\cpython-3.13…\python.exe` (real).
  py-spy on the shim returns *"Failed to find python version"*; the real hang lives in
  the grandchild.

## §1 E1 — does `platynui_native` import and exit? (A vs B)
| command | result |
|---|---|
| `import platynui_native` (timed) | **imported in 0.02s**, returned to prompt |
| `import platynui_native; os._exit(0)` | printed `ok`, exited immediately |

→ Bare import neither blocks (not A) nor keeps the process alive (not B).

## §2–3 E2 / E3 — libdoc & instantiation
| step | result |
|---|---|
| `LibraryDocumentation('PlatynUI.BareMetal')` | **0.46s**, 30 keywords |
| `BareMetal(); get_keyword_names()` | **0.00s**, 30 keywords |

→ All PlatynUI paths are fast and exit cleanly. PlatynUI is exonerated.

## §4 E4 — standalone `python -m robot --dryrun`
`Measure-Command { python -m robot --dryrun … windows_calculator.robot }` →
**1.02 s**, returns normally. **No hang standalone.**

## §5 E5 — pipe vs file capture
| variant | result |
|---|---|
| (a) `capture_output=True` (pipe, like rf-mcp) | **PIPE returned 0.90s rc=0** |
| (b) redirect stdout/stderr to files | **FILE returned 0.73s rc=0** |

→ Plain pipe capture from a normal parent does **not** reproduce it.

## §6 E6 — stack dump of the hung subprocess (THE smoking gun)
Reproduced through the **real MCP server** (`run_test_suite` mode=dry → `timed out after
180s`) while a watcher py-spy-dumped the child.

- Two-process chain (shim → real interpreter, per §0.2). The real interpreter hangs.
- `py-spy dump` (python frames): **`Thread … (idle)` with NO Python frames.**
- `py-spy dump --native` leaf frames:
  ```
  NtQueryInformationFile (ntdll.dll)
  SetFilePointerEx        (…)
  fflush_nolock           (ucrtbase.dll)
  lseeki64                (ucrtbase.dll)
  … python313.dll frames (nearest-export, unreliable) …
  Py_InitializeFromConfig / Py_Main
  ```

→ Child **blocked in the C runtime doing a stdio flush/seek on a std handle**, with
**no PlatynUI/robot/import frames**. Rules out A and B; std-handle/pipe issue.

## §7 Isolation — conditions that did NOT trigger the hang (~0.8–1.2 s each)
| # | condition | result |
|---|---|---|
| A | stdin inherited (console), capture stdout/err | OK 0.89s |
| B | `stdin=DEVNULL`, capture | OK 0.97s |
| C | `stdin=PIPE` (open, not written), `communicate` | OK 0.86s |
| D | `stdin=PIPE` closed (EOF) | OK 0.85s |
| E | capture + `CREATE_NO_WINDOW` | OK 0.89s |
| F | capture + `DETACHED_PROCESS` | OK 1.22s |
| G | capture + `CREATE_NO_WINDOW` + `stdin=DEVNULL` | OK 0.91s |
| — | console-less **detached parent** with pipe stdio runs the capture dry-run | OK 0.77s |

→ The missing ingredient is a **pending synchronous blocking read** on the inherited
stdin pipe.

## §8 Deterministic repro + fix validation (`repro_deadlock*.py`)
A parent thread holds a **pending blocking `os.read`** on a synchronous pipe; the child
inherits that pipe as **stdin** and runs `robot --dryrun` with `capture_output`.

| trial | condition | result |
|---|---|---|
| 1 | pending-read pipe inherited as child **stdin** | **HANG — timed out (reproduced)** |
| 2 | pipe stdin but **no** pending read (control) | OK 0.77s |
| 3 | pending read exists, child `stdin=DEVNULL` (**fix**) | OK 0.75s |

**Secondary finding (timeout cleanup):** after trial 1's timeout, `subprocess.run`'s
post-kill `communicate()` itself hung, because killing the **uv trampoline** orphans the
**real interpreter**, which keeps the capture pipes open. On Windows the current
`subprocess.run(timeout=…)` cannot cleanly reap a hung dry-run — a real **process-tree
kill** (`taskkill /F /T /PID`) is needed on timeout.

## §9 E9 — suite bisection (suite-agnostic)
| suite | result |
|---|---|
| `_diag_builtin_only.robot` (just `Log hi`) | **HANG — timed out** (standalone + via MCP) |
| `_diag_platynui_min.robot` (`Library PlatynUI.BareMetal` + `Log hi`) | **HANG — timed out** |

→ Suite-agnostic. PlatynUI is coincidental; any dry-run through the MCP server on
Windows deadlocks.

## Recommended fix (in `suite_execution_service.py`, `run_robot_dry`)
1. Pass `stdin=subprocess.DEVNULL` to the dry-run `subprocess.run(...)` (primary fix).
2. On timeout, kill the **whole process tree** (`taskkill /F /T /PID` Windows;
   `killpg` POSIX) since a plain `kill()` orphans the uv-trampoline grandchild.
3. Optional: `--console none` for dry-run (as already done for the normal-run path).
