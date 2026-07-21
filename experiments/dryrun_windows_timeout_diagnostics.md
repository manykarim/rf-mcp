# Windows dry-run timeout — diagnostics playbook

**Symptom.** `run_test_suite(mode="dry", suite_file_path="c:\...\windows_calculator.robot")`
returns `Dry run execution failed: Dry run execution timed out after 180s`. The suite is a
PlatynUI (`PlatynUI.BareMetal`) Windows Calculator test.

**What we already know (from Linux investigation).**
- The dry-run runs a **subprocess**: `python -m robot --dryrun <suite>` with a 180 s timeout,
  stdout/stderr captured via a **pipe** (`subprocess.run(capture_output=True, timeout=180)`).
- RF `--dryrun` **imports every `Library`** (to validate keyword signatures) but does **not**
  execute keywords or Suite Setup.
- PlatynUI's platform runtime is **lazy** (created only on the first keyword), so dry-run
  should not create it. On **Linux the same call finishes in ~0.9 s**.
- So the Windows hang is almost certainly the **cold `import PlatynUI.BareMetal`** in that
  throwaway subprocess doing something Windows-specific. Three candidates:

  | Hyp | Mechanism | Signature |
  |-----|-----------|-----------|
  | **A** | COM/UIAutomation init **deadlocks at import** (STA thread, no message pump) | the *import call itself* never returns |
  | **B** | import returns, but a native/background thread (COM pump, tokio) is started → the **process can't exit** | import prints OK, but the process/`robot` never terminates |
  | **C** | a **helper process** is spawned that inherits & holds the stdout/stderr pipe | `robot` finishes, but `subprocess.run` blocks draining the pipe |

The experiments below pin down **which** of A/B/C it is. Run them **on the Windows client**,
in the **same Python environment that rf-mcp uses**. Record every result (a template is at the
end) and send it back.

---

## 0. Setup

### 0.1 Use the SAME Python as the MCP server
The dry-run subprocess uses `sys.executable` — the interpreter running rf-mcp. Use that exact
environment (venv activate, `uv run`, or the `uv tool` python). Verify:

```bat
python -c "import sys, robot, platynui_native; print(sys.version); print(sys.executable)"
python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['robotframework','robotframework-platynui','platynui-native','platynui-cli']})"
```
Both must succeed. If `import platynui_native` here already hangs → that alone is a headline
result (hypothesis A/B) — note it and continue.

### 0.2 Install py-spy (for live stack dumps — the most important tool)
```bat
python -m pip install py-spy
```
`py-spy` dumps the Python **and native** call stacks of a running/hung process without
modifying it. This is what turns "it hangs" into "it hangs *here*".

### 0.3 Point a variable at the suite (adjust to your path)
```bat
set SUITE=c:\workspace\ai-in-qa-demo\tests\windows_calculator.robot
```

---

## 1. E0 — Environment & launch context
Record how rf-mcp is launched on this box (this decides whether the subprocess runs in an
interactive window station — relevant to COM/UIAutomation):

```powershell
# How is the MCP server started? (Claude Desktop / VS Code / CLI / a service?)
# Paste the mcp config command + args for robotmcp.

# Is the host session interactive?
query session            # or: qwinsta   -> is there an interactive console/RDP session?
whoami                    # is it running as a service account / SYSTEM?
```
Record: OS build (`winver`), interactive vs service, the MCP launch command.

---

## 2. E1 — Does the process IMPORT and EXIT? (distinguishes A vs B)
This is the pivotal test. Run it **from a plain terminal** (not inside the MCP host):

```bat
python -c "import time,sys; t=time.time(); import platynui_native; print('imported in %.2fs' % (time.time()-t)); sys.stdout.flush()"
```
Interpretation — watch **two** things: does it print, and does it **return to the prompt**?
- **Never prints "imported"** → **(A)** the import blocks. Stop, capture a stack dump (§6).
- **Prints "imported" then hangs before the prompt returns** → **(B)** the process can't exit
  (a background thread keeps it alive). Capture a stack dump (§6).
- **Prints and returns to the prompt immediately** → neither A nor B on the bare import; the
  trigger is later (in `robot --dryrun` / the pipe). Go to E4/E5.

Repeat once more forcing a clean exit to be sure about (B):
```bat
python -c "import platynui_native, os; print('ok'); os._exit(0)"
```
If `os._exit(0)` is needed to get the prompt back, that strongly implies **(B)**.

---

## 3. E2 / E3 — libdoc and instantiation (narrow the trigger)
```bat
python -c "import time; t=time.time(); from robot.libdocpkg import LibraryDocumentation as L; d=L('PlatynUI.BareMetal'); print('libdoc %.2fs %d kw' % (time.time()-t, len(d.keywords)))"

python -c "import time; from PlatynUI.BareMetal import BareMetal; t=time.time(); lib=BareMetal(); ns=lib.get_keyword_names(); print('instantiate+keywords %.2fs %d kw' % (time.time()-t, len(ns)))"
```
Record each: fast / slow / hangs / (and whether the process exits afterward). On Linux both are
< 0.5 s and exit cleanly.

---

## 4. E4 — Reproduce the exact failing subprocess (standalone)
This is literally what rf-mcp runs (minus the outputdir), but visible:

```bat
python -m robot --dryrun --output NONE --report NONE --log NONE "%SUITE%"
```
- **Prints the suite result then hangs before returning** → **(B)** or **(C)** (robot did its
  work but the process/pipe won't finish). Go to §5 and §6.
- **Hangs with no output** → **(A)** likely. Go to §6.
- **Returns fast** → the hang is specific to how rf-mcp runs it (pipe/context). Go to §5 and E9.

Also time the "no timeout kill" path — let it run and note whether it EVER returns (it may
return at ~180 s only because rf-mcp kills it; standalone it may hang forever):
```powershell
Measure-Command { python -m robot --dryrun --output NONE --report NONE --log NONE $env:SUITE }
```

---

## 5. E5 — Pipe vs file (tests C directly, and validates the fix)
rf-mcp captures via a **pipe**; a leaked pipe-holding helper (C) makes that block even after
`robot` exits. Redirect to **files** instead and compare:

```bat
:: (a) like rf-mcp: capture via a pipe
python -c "import subprocess,sys,time; t=time.time(); p=subprocess.run([sys.executable,'-m','robot','--dryrun','--output','NONE','--report','NONE','--log','NONE',r'%SUITE%'],capture_output=True,timeout=60); print('PIPE returned %.2fs rc=%d'%(time.time()-t,p.returncode))"

:: (b) redirect to files (no pipe to hold)
python -c "import subprocess,sys,time; t=time.time(); o=open('out.txt','wb'); e=open('err.txt','wb'); rc=subprocess.run([sys.executable,'-m','robot','--dryrun','--output','NONE','--report','NONE','--log','NONE',r'%SUITE%'],stdout=o,stderr=e,timeout=60).returncode; print('FILE returned %.2fs rc=%d'%(time.time()-t,rc))"
```
- **(a) times out but (b) returns fast** → **(C) confirmed** — a helper holds the pipe; the
  file-redirect fix works.
- **Both hang** → not the pipe; it's **(A)** or **(B)** — the `robot` process itself never
  exits. Confirm with the stack dump (§6).

---

## 6. E6 — Stack dump of the hung process (THE smoking gun)
Whenever something hangs above, capture *where*. Two ways:

### 6a. py-spy on the live hung process (best)
Start the hang in terminal #1 (e.g. E4). In terminal #2 find the PID and dump it:
```powershell
# find the hung python running robot
Get-CimInstance Win32_Process | ? { $_.CommandLine -match 'robot' -and $_.Name -eq 'python.exe' } | Select ProcessId, CommandLine
py-spy dump --pid <PID>          # add --native for the Rust/UIAutomation frames
```
Save the full output. The top frames tell us everything:
- stuck in `import` / `_bootstrap` / a `platynui_native` frame → **A**
- main thread done but a thread parked in a COM/UIA/`WaitForSingleObject`/pump → **B**
- blocked in `_winapi.WaitForSingleObject` / `communicate` / reading a pipe → **C**

### 6b. faulthandler self-dump (fallback, no py-spy)
Save this as `win_dryrun_probe.py` and run it — it dumps ALL thread stacks every 20 s if a
stage hangs, and times each stage:
```python
import faulthandler, time, sys, subprocess, os
faulthandler.dump_traceback_later(20, repeat=True)   # every 20s, print all thread stacks to stderr
SUITE = sys.argv[1] if len(sys.argv) > 1 else r"c:\workspace\ai-in-qa-demo\tests\windows_calculator.robot"

def stage(label, fn):
    t = time.time()
    try:
        fn(); print(f"OK   {label}: {time.time()-t:.2f}s", flush=True)
    except Exception as e:
        print(f"ERR  {label}: {type(e).__name__}: {e}", flush=True)

stage("1 import platynui_native", lambda: __import__("platynui_native"))
stage("2 libdoc BareMetal",       lambda: __import__("robot.libdocpkg", fromlist=["LibraryDocumentation"]).LibraryDocumentation("PlatynUI.BareMetal"))
stage("3 BareMetal() + names",    lambda: __import__("PlatynUI.BareMetal", fromlist=["BareMetal"]).BareMetal().get_keyword_names())
stage("4 python -m robot --dryrun", lambda: subprocess.run([sys.executable,"-m","robot","--dryrun","--output","NONE","--report","NONE","--log","NONE",SUITE], timeout=200, capture_output=True))
print("all stages returned; does THIS process now exit?", flush=True)
```
```bat
python win_dryrun_probe.py "%SUITE%" 1> probe_out.txt 2> probe_stacks.txt
```
`probe_out.txt` = per-stage timing; `probe_stacks.txt` = the stack of wherever it hangs. Send
both.

---

## 7. E7 — Child processes & held handles (confirms C)
While a run is hung, inspect what it spawned and what holds the pipe:

```powershell
# child processes of the hung python running robot (PID from E6a)
Get-CimInstance Win32_Process | ? ParentProcessId -eq <PID> | Select ProcessId, Name, CommandLine
```
Look for a leftover **platynui** / **cli** / helper process. If one exists and outlives
`robot`, that's the pipe-holder (**C**).

Optional, precise: with Sysinternals `handle.exe` (run as admin), list handles held on the
pipe by any surviving child:
```bat
handle.exe -a -p <child_PID>
```

---

## 8. E8 — Context sensitivity (interactive vs MCP host)
UIAutomation/COM can behave differently under a non-interactive window station. Compare:

1. Run E4 in a **normal interactive terminal** (you logged in, desktop visible).
2. Run the **same** through rf-mcp (the failing tool call) — the MCP host may run it in a
   different session/window station.

If E4 is **fast standalone but hangs only via the MCP host** → it's a **window-station / COM
apartment / handle-inheritance** issue of how the host spawns the subprocess (not PlatynUI
itself). Note the difference.

---

## 9. E9 — Suite bisection (confirm PlatynUI is the trigger, isolate the line)
Create trimmed suites next to the original and dry-run each (via `python -m robot --dryrun`):

```robotframework
# builtin_only.robot  — expect FAST (proves the harness itself is fine)
*** Test Cases ***
T
    Log    hi
```
```robotframework
# platynui_min.robot  — ONLY the PlatynUI import, no Process/OperatingSystem, no Suite Setup
*** Settings ***
Library    PlatynUI.BareMetal
*** Test Cases ***
T
    Log    hi
```
```robotframework
# platynui_kw.robot  — PlatynUI import + one PlatynUI keyword referenced (still not executed in dryrun)
*** Settings ***
Library    PlatynUI.BareMetal
*** Test Cases ***
T
    Get Attribute    //control:Text    Name
```
Expected map:
- `builtin_only` fast, `platynui_min` **hangs** → the hang is the **PlatynUI import** (A/B), not
  Process/OperatingSystem/Suite Setup/keyword resolution.
- `platynui_min` fast but `platynui_kw` hangs → keyword *spec resolution* touches the runtime on
  Windows (unlike Linux). (Less likely, but record it.)

---

## 10. Decision tree (map your results → root cause)

```
E1 import platynui_native ─┬─ never prints ............................. A (import deadlocks)
                           ├─ prints, but process won't exit ........... B (background thread)
                           └─ prints & exits fast ─┬─ E5(a) pipe hangs, E5(b) file OK ... C (pipe-held by helper; see E7)
                                                   ├─ E4 hangs after printing result .... B (robot can't exit)
                                                   └─ E4 fast standalone, hangs via host . window-station/COM context (E8)
E9 platynui_min hangs  → trigger is the PlatynUI IMPORT (rules out Process/OS/SuiteSetup/keywords)
E6 stack dump           → the authoritative answer (frame names: import→A, thread/pump→B, WaitForSingleObject/pipe→C)
```

## 11. What to send back
- E0: OS build, interactive-vs-service, MCP launch command.
- E1–E4: exact console output (did it print? did it return?) + timings.
- **E5**: PIPE vs FILE return/hang + timings.
- **E6**: `py-spy dump` output (or `probe_stacks.txt`) — the single most useful artifact.
- E7: child process list while hung.
- E9: the bisection results (which trimmed suite hangs).
- Versions from §0.1.

With E1 + E5 + one E6 stack dump we can name A/B/C with confidence and scope the fix
(in-process/tiered validation, file-redirect, real process-tree kill on timeout, and/or an
upstream PlatynUI-Windows lazy-init issue).

> Safety: these only *validate* a desktop suite (no keywords execute). E4/E5 may spawn/leak a
> helper or leave a hung `python.exe` — close stray `python.exe` / PlatynUI processes between
> runs (`taskkill /IM python.exe /F` only if you have no other Python work running).

