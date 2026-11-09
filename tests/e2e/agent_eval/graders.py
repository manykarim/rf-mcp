"""Graders for agent runs.

Process graders are pure functions over a Claude Code stream-json transcript (`load_transcript`). They flag
the anti-patterns of task 13.2:

- `output_xml_reads`: reading or grepping output.xml (Read/Grep/Glob/Bash, or REPL/debug statements parsing it)
- `single_file_selections`: selecting tests by one .robot file path instead of a longname
- `runs_before_analyze`: running written/edited suites before rf_analyze covered them
- `unannounced_open_sessions`: REPL/debug sessions left open without saying so
- `results_after_project_reads`: for "why did X fail?" tasks, reading project files before rf_results

plus the pitfall counters of the later experiments (`pitfall_counts`, `static_counts`).

Outcome graders (`grade_*`) run robotcode in the project environment and need the suite's fixtures.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------------------- transcript


@dataclass
class Call:
    index: int
    id: str
    full_name: str
    input: dict
    result: str = ""
    is_error: bool = False
    has_result: bool = False

    @property
    def name(self) -> str:
        """Short tool name: `mcp__rf-mcp__rf_run` -> `rf_run`."""
        return self.full_name.split("__")[-1]

    @property
    def action(self) -> str:
        default = "start" if self.name in ("rf_repl", "rf_debug") else ""
        return str(self.input.get("action") or default).lower()

    @property
    def command(self) -> str:
        """Shell command of a Bash/PowerShell call ('' otherwise)."""
        return str(self.input.get("command") or "") if self.name in SHELL_TOOLS else ""

    @property
    def statement(self) -> str:
        """REPL statement or debugger command ('' otherwise)."""
        return str(self.input.get("statement") or self.input.get("command") or "") if self.name in SESSION_TOOLS else ""


@dataclass
class Transcript:
    calls: list[Call] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    cwd: str = ""

    @property
    def final(self) -> str:
        if self.result.get("result"):
            return str(self.result["result"])
        return self.texts[-1] if self.texts else ""

    def named(self, *names: str) -> list[Call]:
        return [c for c in self.calls if c.name in names]

    def mentions(self, *needles: str) -> bool:
        blob = json.dumps([c.input for c in self.calls])
        return any(n in blob for n in needles)


SHELL_TOOLS = {"Bash", "PowerShell"}
SESSION_TOOLS = {"rf_repl", "rf_debug"}
FILE_READ_TOOLS = {"Read", "Grep", "Glob", "NotebookRead"}
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}


def _flatten(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return "" if content is None else str(content)


def _events(source: str | Path | Iterable[str]) -> Iterable[dict]:
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and source.endswith(".jsonl")):
        lines: Iterable[str] = Path(source).read_text(encoding="utf-8", errors="replace").splitlines()
    elif isinstance(source, str):
        lines = source.splitlines()
    else:
        lines = source
    for line in lines:
        if isinstance(line, dict):
            yield line
            continue
        try:
            ev = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(ev, dict):
            yield ev


_SHELL_WRAPPER = re.compile(r'^\s*"?(?P<exe>[^"]*?(?P<shell>pwsh|powershell|cmd|bash|sh)(?:\.exe)?)"?\s+'
                            r"(?:-NoProfile\s+)?(?:-Command|/c|-lc|-c)\s+(?P<body>.*)$", re.S | re.I)


def _unwrap_shell(command: str) -> tuple[str, str]:
    """Codex logs `"...pwsh.exe" -Command "<cmd>"`; return (tool name, inner command)."""
    m = _SHELL_WRAPPER.match(command)
    if not m:
        return "Bash", command
    body = m.group("body").strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in "\"'":
        body = body[1:-1].replace('\\"', '"')
    body = body.replace("\\\\", "\\")  # codex doubles backslashes inside the logged command
    return ("PowerShell" if m.group("shell").lower() in ("pwsh", "powershell") else "Bash"), body


def _load_codex(events: Iterable[dict], t: Transcript) -> Transcript:
    """Map `codex exec --json` items onto Claude-style calls, so every grader works on both agents.

    mcp_tool_call -> `mcp__<server>__<tool>`; command_execution -> Bash/PowerShell with the shell wrapper removed;
    file_change -> Write (add) / Edit (update, delete); web_search -> WebSearch. Turns are approximated as tool calls
    + 1 (codex exec reports no model-request count); tokens come from `turn.completed`.
    """
    usage: Counter = Counter()
    for ev in events:
        kind = ev.get("type")
        if kind == "turn.completed":
            usage.update({k: v for k, v in (ev.get("usage") or {}).items() if isinstance(v, int)})
            continue
        if kind == "turn.failed" or kind == "error":
            t.result.setdefault("errors", []).append(str(ev.get("error") or ev.get("message"))[:500])
            continue
        if kind != "item.completed":
            continue
        item = ev.get("item") or {}
        itype = item.get("type")
        if itype == "agent_message":
            t.texts.append(item.get("text", ""))
        elif itype == "mcp_tool_call":
            res = item.get("result") or {}
            text = _flatten(res.get("content")) if isinstance(res, dict) else str(res)
            if res.get("structured_content") and not text:
                text = json.dumps(res["structured_content"])
            err = item.get("error")
            t.calls.append(Call(len(t.calls), item.get("id", ""), f"mcp__{item.get('server', '')}__{item.get('tool', '')}",
                                item.get("arguments") or {}, text or (str(err) if err else ""),
                                bool(err) or bool(res.get("is_error") or res.get("isError")) or item.get("status") == "failed",
                                True))
        elif itype == "command_execution":
            name, command = _unwrap_shell(item.get("command", ""))
            code = item.get("exit_code")
            t.calls.append(Call(len(t.calls), item.get("id", ""), name, {"command": command},
                                item.get("aggregated_output", ""), code not in (0, None) or item.get("status") == "failed", True))
        elif itype == "file_change":
            for ch in item.get("changes") or []:
                tool = "Write" if ch.get("kind") == "add" else "Edit"
                t.calls.append(Call(len(t.calls), item.get("id", ""), tool, {"file_path": ch.get("path", "")}, "",
                                    item.get("status") == "failed", True))
        elif itype == "web_search":
            t.calls.append(Call(len(t.calls), item.get("id", ""), "WebSearch", {"query": item.get("query", "")}, "", False, True))
    t.result.update(num_turns=len(t.calls) + 1, result=t.texts[-1] if t.texts else "",
                    usage={"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                           "cached_input_tokens": usage["cached_input_tokens"]})
    return t


def load_transcript(source: str | Path | Iterable[str] | Iterable[dict]) -> Transcript:
    """Parse a stream-json transcript (path, text or iterable of lines/events); `codex exec --json` too."""
    t = Transcript()
    by_id: dict[str, Call] = {}
    events = list(_events(source))  # type: ignore[arg-type]
    if any(ev.get("type") == "thread.started" for ev in events[:3]):
        return _load_codex(events, t)
    for ev in events:
        kind = ev.get("type")
        if kind == "system" and ev.get("subtype") == "init":
            t.cwd = ev.get("cwd") or t.cwd
        elif kind == "result":
            t.result = ev
        msg = ev.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                call = Call(len(t.calls), block.get("id", ""), block.get("name", ""), block.get("input") or {})
                t.calls.append(call)
                by_id[call.id] = call
            elif block.get("type") == "tool_result":
                call = by_id.get(block.get("tool_use_id", ""))
                if call:
                    call.result = _flatten(block.get("content"))
                    call.is_error = bool(block.get("is_error"))
                    call.has_result = True
            elif block.get("type") == "text" and kind == "assistant":
                t.texts.append(block.get("text", ""))
    return t


def run_metrics(t: Transcript) -> dict:
    """Turns, cost, tokens and tool counts from the transcript (the `result` event)."""
    usage = t.result.get("usage") or {}
    return dict(
        num_turns=t.result.get("num_turns"),
        cost=t.result.get("total_cost_usd"),
        in_tokens=sum(usage.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")),
        out_tokens=usage.get("output_tokens") or 0,
        tools=dict(Counter(c.full_name for c in t.calls)),
        tool_errors=sum(c.is_error for c in t.calls),
        final=t.final,
    )


# --------------------------------------------------------------------------------------- shell helpers

_SEGMENT_SPLIT = re.compile(r"\|\||&&|[|;\n]")
_ENV_PREFIX = re.compile(r"^\s*(?:(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S*)\s+)+")


def _segments(command: str) -> list[str]:
    return [s.strip() for s in _SEGMENT_SPLIT.split(command) if s.strip()]


def _tokens(segment: str) -> list[str]:
    segment = segment.replace("\\", "/")  # Windows paths: keep separators instead of shell-unescaping them
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _program(segment: str) -> tuple[str, list[str]]:
    """(program basename, remaining tokens) of a shell segment, skipping env assignments, `uv run`, `&`."""
    toks = _tokens(_ENV_PREFIX.sub("", segment))
    while toks and toks[0] in ("&", "call", "sudo", "time"):
        toks = toks[1:]
    if len(toks) >= 2 and Path(toks[0]).name.lower() in ("uv", "uv.exe") and toks[1] == "run":
        toks = toks[2:]
        while toks and toks[0].startswith("-"):
            toks = toks[2:] if toks[0] in ("--directory", "--project", "--with", "--python") else toks[1:]
    if not toks:
        return "", []
    prog = Path(toks[0].replace("\\", "/")).name.lower()
    if prog.endswith(".exe"):
        prog = prog[:-4]
    if prog.startswith("python") and len(toks) >= 3 and toks[1] == "-m":
        return toks[2].lower(), toks[3:]
    return prog, toks[1:]


_RUN_SUBCOMMANDS = ("robot", "debug", "robot-debug", "run")


def _shell_run_args(segment: str) -> list[str] | None:
    """Arguments of a test run (`robotcode robot|debug ...`, `robot ...`, `python -m robot ...`), else None."""
    prog, args = _program(segment)
    if prog == "robot":
        return args
    if prog != "robotcode":
        return None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("session", "results", "analyze", "discover", "repl", "doc", "profiles", "config"):
            if a == "session" and "start" in args[i:] and "debug" in args[i:]:
                return args[args.index("debug", i) + 1:]
            return None
        if a in _RUN_SUBCOMMANDS:
            return args[i + 1:]
        i += 2 if a in ("-r", "--root", "-p", "--profile", "-f", "--format", "-d", "--dir") else 1
    return None


def _shell_robotcode_sub(segment: str, sub: str) -> list[str] | None:
    """Arguments after `robotcode ... <sub>` (e.g. 'results', 'analyze'), else None."""
    prog, args = _program(segment)
    if prog != "robotcode" or sub not in args:
        return None
    return args[args.index(sub) + 1:]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                return [str(v) for v in json.loads(text)]
            except ValueError:
                pass
        return [p.strip() for p in text.split(",") if p.strip()]
    if isinstance(value, dict):
        return [str(k) for k in value]
    return [str(v) for v in value]


def _rel(path: str, root: str) -> str:
    p = path.strip().strip("\"'").replace("\\", "/")
    r = root.replace("\\", "/").rstrip("/")
    if r and p.lower().startswith(r.lower() + "/"):
        p = p[len(r) + 1:]
    while p.startswith("./"):
        p = p[2:]
    return p.rstrip("/").lower()


# --------------------------------------------------------------------------------------- (a) output.xml reads

OUTPUT_XML = re.compile(r"output\.xml|results[\\/]+[^\s\"']*\.xml", re.I)
_GREP_HIT = re.compile(r"(?m)^\S*output\.xml[:-]\d+[:-]")
_NON_READERS = {"robotcode", "robot", "rebot", "robot.rebot", "ls", "dir", "test", "test-path", "rm", "del", "remove-item",
                "mkdir", "cp", "copy", "copy-item", "mv", "move", "move-item", "stat", "echo", "cd"}


def output_xml_reads(t: Transcript) -> list[str]:
    """Evidence of reading or grepping a result XML (output.xml) instead of using rf_results."""
    hits = []
    for c in t.calls:
        if c.name in FILE_READ_TOOLS:
            if OUTPUT_XML.search(json.dumps(c.input)):
                hits.append(f"#{c.index} {c.name} {json.dumps(c.input)[:160]}")
            elif c.name == "Grep" and c.input.get("output_mode") == "content" and _GREP_HIT.search(c.result):
                hits.append(f"#{c.index} Grep matched lines of output.xml ({c.input.get('pattern')!r})")
        elif c.name in SHELL_TOOLS:
            for seg in _segments(c.command):
                if OUTPUT_XML.search(seg) and _program(seg)[0] not in _NON_READERS:
                    hits.append(f"#{c.index} {c.name} {seg[:160]}")
                    break
        elif c.name in SESSION_TOOLS and OUTPUT_XML.search(c.statement):
            hits.append(f"#{c.index} {c.name} {c.statement[:160]}")
    return hits


# --------------------------------------------------------------------------------------- (b) file selection


def single_file_selections(t: Transcript) -> list[str]:
    """Runs/debug sessions that select tests by one .robot file (parent __init__.robot setups are skipped)."""
    hits = []
    for c in t.calls:
        if c.name == "rf_run" or (c.name == "rf_debug" and c.action == "start"):
            paths = _as_list(c.input.get("paths"))
            if len(paths) == 1 and paths[0].strip().lower().endswith(".robot"):
                hits.append(f"#{c.index} {c.name} paths={paths}")
        elif c.name in SHELL_TOOLS:
            for seg in _segments(c.command):
                args = _shell_run_args(seg)
                if args is not None and len([a for a in args if a.lower().endswith(".robot")]) == 1:
                    hits.append(f"#{c.index} {c.name} {seg[:160]}")
                    break
    return hits


# --------------------------------------------------------------------------------------- (c) run before analyze


def _norm_part(part: str) -> str:
    part = re.sub(r"\.(robot|resource|txt)$", "", part, flags=re.I)
    part = re.sub(r"^\d+__", "", part)
    return re.sub(r"[\s_]+", "", part).lower()


def _file_parts(rel: str) -> list[str]:
    return [_norm_part(p) for p in rel.split("/") if p]


def _longname_selects(longname: str, rel_file: str) -> bool:
    ln = [_norm_part(p) for p in longname.split(".") if p]
    fp = _file_parts(rel_file)
    if not ln or not fp:
        return False
    for k in range(len(fp)):  # tolerate longnames without the top suite ('Calc.Cart Contains Items')
        tail = fp[k:]
        n = min(len(tail), len(ln))
        if tail[:n] == ln[:n] and (k == 0 or len(ln) >= len(tail)):
            return True
    return False


def _path_covers(path: str, rel_file: str) -> bool:
    return rel_file == path or rel_file.startswith(path + "/") or path in ("", ".")


def _written_files(c: Call, root: str) -> list[str]:
    files = []
    if c.name in WRITE_TOOLS:
        files.append(_rel(str(c.input.get("file_path") or c.input.get("notebook_path") or ""), root))
    if c.name == "rf_repl" and c.action == "save" and c.input.get("save_path"):
        files.append(_rel(str(c.input["save_path"]), root))
    if c.name in ("rf_run", "rf_repl", "rf_debug", "rf_analyze") and isinstance(c.input.get("files"), dict):
        files += [_rel(k, root) for k in c.input["files"]]
    return [f for f in files if f.endswith((".robot", ".resource"))]


def _analyzed_paths(c: Call, root: str) -> list[str] | None:
    """Paths an analysis covers ([''] = whole project); None if `c` is no analysis."""
    if c.name == "rf_analyze":
        if c.is_error:
            return None
        paths = [_rel(p, root) for p in _as_list(c.input.get("paths"))]
        return paths or [""]
    for seg in _segments(c.command):
        args = _shell_robotcode_sub(seg, "analyze")
        if args is not None:
            paths = [_rel(a, root) for a in args if not a.startswith("-") and a != "code"]
            return paths or [""]
    return None


def _run_selection(c: Call, root: str) -> dict | None:
    """What a run selects: {'longnames': [...], 'paths': [...]} (empty = everything); None if `c` is no run."""
    if c.name == "rf_run" or (c.name == "rf_debug" and c.action == "start"):
        return {"longnames": _as_list(c.input.get("by_longname")),
                "paths": [_rel(p, root) for p in _as_list(c.input.get("paths"))]}
    for seg in _segments(c.command):
        args = _shell_run_args(seg)
        if args is not None:
            longnames = [args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-bl", "--by-longname")]
            paths = [_rel(a, root) for a in args if a.lower().endswith(".robot")]
            return {"longnames": longnames, "paths": paths}
    return None


def _selected(sel: dict, rel_file: str) -> bool:
    if rel_file.endswith(".resource"):
        return True
    if sel["longnames"] and not any(_longname_selects(ln, rel_file) for ln in sel["longnames"]):
        return False
    if sel["paths"] and not any(_path_covers(p, rel_file) for p in sel["paths"]):
        return False
    return True


def runs_before_analyze(t: Transcript) -> list[str]:
    """Runs of suites/resources that were written or edited after the last rf_analyze covering them."""
    pending: dict[str, int] = {}
    hits = []
    for c in t.calls:
        analyzed = _analyzed_paths(c, t.cwd)
        if analyzed is not None:
            for f in [f for f in pending if any(_path_covers(p, f) for p in analyzed)]:
                del pending[f]
            continue
        sel = _run_selection(c, t.cwd)
        if sel is not None and not c.is_error:
            stale = sorted(f for f in pending if _selected(sel, f))
            if stale:
                hits.append(f"#{c.index} {c.name} ran unanalyzed {', '.join(stale)}")
        for f in _written_files(c, t.cwd):
            pending[f] = c.index
    return hits


_SHELL_WRITE = re.compile(r"Set-Content|Out-File|Add-Content|WriteAllText|\btee\b|>\s*\S*\.(?:robot|resource)\b", re.I)


def _writes_suite(c: Call, root: str) -> bool:
    if c.name == "rf_analyze":
        return False  # weak models pass `files` with the unchanged sources to analysis (s3: GLM, Qwen)
    if _written_files(c, root):
        return True
    return bool(c.command) and bool(re.search(r"\.(?:robot|resource)\b", c.command)) and bool(_SHELL_WRITE.search(c.command))


def _run_passed(c: Call, root: str) -> bool:
    if c.name == "rf_run":
        return c.has_result and not c.input.get("dryrun") and bool(re.search(r'"exit_code"\s*:\s*0\b', c.result))
    return c.has_result and not c.is_error and _run_selection(c, root) is not None and c.name in SHELL_TOOLS


def unverified_finish(t: Transcript) -> bool:
    """A .robot/.resource file was written and no test run passed after the last such write.

    2026-11-09 OpenRouter sweep: Qwen3-Coder ended authoring runs with "I have successfully created..." after a failed
    rf_run or a clean rf_analyze only. The final text is no reliable signal (it lists "timeout" settings, quotes `Fail`).
    """
    writes = [c.index for c in t.calls if _writes_suite(c, t.cwd)]
    return bool(writes) and not any(_run_passed(c, t.cwd) for c in t.calls[writes[-1] + 1:])


def last_analyze_has_errors(t: Transcript) -> bool | None:
    """has_errors of the last rf_analyze result, or None if the agent never analyzed."""
    last = None
    for c in t.named("rf_analyze"):
        if c.has_result and not c.is_error:
            m = re.search(r'"has_errors"\s*:\s*(true|false)', c.result)
            if m:
                last = m.group(1) == "true"
    return last


# --------------------------------------------------------------------------------------- (d) open sessions

_SESSION_ID = re.compile(r'"session_id"\s*:\s*"([^"]+)"|\b((?:repl|debug)-[0-9a-f]{6,})\b')
_ENDED = re.compile(r'"status"\s*:\s*"(stopped|ended|finished|terminated|exited|closed)"')
_ANNOUNCED = re.compile(
    r"\bsessions?\b[^.\n]{0,80}\b(open|running|active|alive)\b|\b(left|kept|still)\b[^.\n]{0,40}\b(open|running)\b",
    re.I,
)


def open_sessions(t: Transcript) -> list[str]:
    """Ids of REPL/debug sessions started and never stopped."""
    opened: dict[str, int] = {}
    for c in t.calls:
        ids = [a or b for a, b in _SESSION_ID.findall(c.result)] if c.has_result else []
        if c.name in SESSION_TOOLS:
            if c.action == "start" and not c.is_error and ids:
                opened[ids[0]] = c.index
            elif c.action == "stop":
                scope = str(c.input.get("stop_scope") or "id").lower()
                if scope in ("project", "all"):
                    opened.clear()
                else:
                    opened.pop(str(c.input.get("session_id") or ""), None)
                    for i in ids:
                        opened.pop(i, None)
            elif _ENDED.search(c.result) and c.input.get("session_id"):
                opened.pop(str(c.input["session_id"]), None)
        for seg in _segments(c.command):
            args = _shell_robotcode_sub(seg, "session")
            if args is None or not args:
                continue
            if args[0] == "start" and ids:
                opened[ids[0]] = c.index
            elif args[0] == "stop":
                named = [a for a in args[1:] if not a.startswith("-")]
                if named and "--project" not in args and "--all" not in args:
                    for a in named:
                        opened.pop(a, None)
                else:
                    opened.clear()
    return list(opened)


def unannounced_open_sessions(t: Transcript) -> list[str]:
    """Sessions left open while the final answer does not mention an open session."""
    left = open_sessions(t)
    final = t.final
    if not left or _ANNOUNCED.search(final) or any(s in final for s in left):
        return []
    return left


# --------------------------------------------------------------------------------------- (e) results first

_READ_PROGRAMS = {"cat", "type", "get-content", "gc", "sed", "head", "tail", "less", "more", "grep", "rg", "findstr",
                  "select-string", "awk", "nl"}


def _pipeline_heads(command: str) -> list[str]:
    """First command of every pipeline (`x | head` filters x's output, it reads no file)."""
    return [s.split("|", 1)[0].strip() for s in re.split(r"\|\||&&|[;\n]", command) if s.strip()]


def _is_project_read(c: Call) -> bool:
    if c.name in ("Read", "Grep", "NotebookRead"):
        target = str(c.input.get("file_path") or c.input.get("path") or "").replace("\\", "/")
        return "/.claude/" not in target and not target.startswith(".claude/")
    return any(_program(seg)[0] in _READ_PROGRAMS and ".claude/" not in seg.replace("\\", "/")
               for seg in _pipeline_heads(c.command))


def _is_results_call(c: Call) -> bool:
    if c.name == "rf_results":
        return True
    return any(_shell_robotcode_sub(seg, "results") is not None for seg in _segments(c.command))


def results_after_project_reads(t: Transcript) -> dict:
    """For "why did X fail?" tasks: did the agent read project files before asking rf_results?"""
    first_read = next((c.index for c in t.calls if _is_project_read(c)), None)
    first_results = next((c.index for c in t.calls if _is_results_call(c)), None)
    flagged = first_read is not None and (first_results is None or first_read < first_results)
    return {"first_project_read": first_read, "first_results": first_results, "flagged": flagged}


# --------------------------------------------------------------------------------------- pitfall counters

PITFALLS = {
    "json_or_headers_as_text": r"'str' object has no attribute 'items'|valid dictionary or object",
    "inline_dict_syntax": r"Evaluating expression .* failed: SyntaxError",
    "not_imported": r"No keyword with name '(Create Session|GET On Session|POST On Session|New Browser|New Page|Open Browser)",
    "int_vs_string": r"\(integer\) != .* \(string\)|\(string\) != .* \(integer\)",
    "arith_not_evaluated": r"cannot be converted to a floating point number",
    "frame_or_shadow_not_found": r"waiting for locator|Element .* not found|did not match any elements|Unable to locate element",
    "strict_mode": r"strict mode violation",
    "http_4xx_raised": r"HTTPError: 4\d\d",
    "type_rejected_422": r"Input should be a valid (integer|number|boolean)",
    "json_decode": r"JSONDecodeError|Expecting value: line 1",
    "files_arg": r"files.*(TypeError|ValueError|AttributeError)|expected str, bytes or os.PathLike",
    "missing_key": r"has no key",
    "multiple_xml_matches": r"Multiple elements \(\d+\) matching",
    "xml_no_match": r"No element matching",
    "xpath_prefix": r"prefix .* not found|SyntaxError: prefix",
    "http_5xx_raised": r"HTTPError: 5\d\d",
    "graphql_errors": r'"errors"|Cannot query field',
}
FAILED = re.compile(r'"status":\s*"failed"|\[ FAIL \]|\| FAIL \|')


def pitfall_counts(t: Transcript) -> dict:
    """REPL sends, failed sends, pitfalls seen in failed rf_repl/rf_run results, skill reads, probes."""
    sends = multi = failed = 0
    pit: Counter = Counter()
    for c in t.calls:
        if c.name == "rf_repl" and c.action == "send":
            sends += 1
            multi += (c.statement.strip().count("\n") > 0)
        if c.name in ("rf_repl", "rf_run") and c.has_result and FAILED.search(c.result):
            failed += c.name == "rf_repl"
            for key, rx in PITFALLS.items():
                if re.search(rx, c.result):
                    pit[key] += 1
    statements = "\n".join(c.statement for c in t.named("rf_repl"))
    return dict(
        sends=sends,
        multi_sends=multi,
        failed_sends=failed,
        pitfalls=dict(pit),
        skill_reads=[str(c.input.get("uri")) for c in t.named("read_resource", "ReadMcpResourceTool")],
        expected_status_any=len(re.findall(r"expected_status=any", statements)),
        token_as_json=len(re.findall(r"/token\s+json=", statements)),
        json_text_in_repl=len(re.findall(r"json=\s*[{\[]", statements)),
    )


def static_counts(src: str) -> dict:
    """Pitfalls in a produced suite's source."""
    return {
        "json_text_body": len(re.findall(r"json=\s*[{\[]", src)),
        "headers_text": len(re.findall(r"headers=\s*\{", src)),
        "data_form": len(re.findall(r"data=", src)),
        "files_arg": len(re.findall(r"files=", src)),
        "json_loads": len(re.findall(r"json\.loads|Convert String To Json|To Json", src)),
        "sleep": len(re.findall(r"\bSleep\b", src)),
        "force": len(re.findall(r"force=True", src)),
        "blocked_tag": len(re.findall(r"\bblocked\b", src)),
        "skip": len(re.findall(r"\bSkip\b", src)),
        "expected_status": src.count("expected_status"),
        "expected_status_any": src.count("expected_status=any"),
        "precision": "precision=" in src,
        "suite_variable_chaining": len(re.findall(r"Set (Suite|Global) Variable", src)),
        "str_numbers_in_create_dict": len(re.findall(r"Create Dictionary[^\n]*\b(qty|quantity|unit_price|totalprice)=\d", src)),
        "xpath": len(re.findall(r"xpath[=:]|\s//", src)),
        "shadow_root": bool(re.search(r"shadow_?[rR]oot", src)),
        "attachShadow_patch": "attachShadow" in src,
    }


def process_grades(t: Transcript, why_failed: bool = False) -> dict:
    """All process graders for one transcript; `flags` lists the anti-patterns that fired."""
    g = dict(
        output_xml_reads=output_xml_reads(t),
        single_file_selections=single_file_selections(t),
        runs_before_analyze=runs_before_analyze(t),
        open_sessions=open_sessions(t),
        unannounced_open_sessions=unannounced_open_sessions(t),
        last_analyze_has_errors=last_analyze_has_errors(t),
        unverified_finish=unverified_finish(t),
        **pitfall_counts(t),
    )
    if why_failed:
        g["results_first"] = results_after_project_reads(t)
    flags = [name for name, key in (("output_xml", "output_xml_reads"), ("file_selection", "single_file_selections"),
                                    ("run_before_analyze", "runs_before_analyze"),
                                    ("open_session", "unannounced_open_sessions"), ("unverified", "unverified_finish"))
             if g[key]]
    if why_failed and g["results_first"]["flagged"]:
        flags.append("results_late")
    g["flags"] = flags
    return g


# --------------------------------------------------------------------------------------- trimming


_KEEP = [re.compile(p) for p in (
    r'"status":\s*"\w+"', r"\[ FAIL \]|\| FAIL \|", r'"session_id"\s*:\s*"[^"]+"', r'"has_errors"\s*:\s*(?:true|false)',
    r"(?m)^\S*output\.xml[:-]\d+[:-]", *PITFALLS.values())]


def trim_transcript(source: str | Path | Iterable[str]) -> list[dict]:
    """Keep only the events and fields the graders read, so recorded transcripts stay small."""
    out: list[dict] = []
    for ev in _events(source):  # type: ignore[arg-type]
        kind = ev.get("type")
        if kind == "system" and ev.get("subtype") == "init":
            out.append({"type": "system", "subtype": "init", "cwd": ev.get("cwd"), "model": ev.get("model")})
        elif kind == "result":
            usage = ev.get("usage") or {}
            out.append({"type": "result", "subtype": ev.get("subtype"), "num_turns": ev.get("num_turns"),
                        "total_cost_usd": ev.get("total_cost_usd"), "result": str(ev.get("result") or "")[:2000],
                        "usage": {k: usage.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                            "cache_creation_input_tokens")}})
        elif kind in ("assistant", "user") and isinstance((ev.get("message") or {}).get("content"), list):
            blocks = []
            for b in ev["message"]["content"]:
                if b.get("type") == "tool_use":
                    inp = {k: (v[:200] + "...") if k in ("content", "old_string", "new_string") and isinstance(v, str)
                           and len(v) > 200 else v for k, v in (b.get("input") or {}).items()}
                    blocks.append({"type": "tool_use", "id": b.get("id"), "name": b.get("name"), "input": inp})
                elif b.get("type") == "tool_result":
                    text = _flatten(b.get("content"))
                    found = [m.group(0) for rx in _KEEP for m in rx.finditer(text)]
                    # long `.*` matches keep both ends, so the pattern still matches the trimmed text
                    keep = [text[:80]] + [f if len(f) <= 120 else f"{f[:60]} ~ {f[-60:]}" for f in found]
                    blocks.append({"type": "tool_result", "tool_use_id": b.get("tool_use_id"),
                                   "is_error": bool(b.get("is_error")), "content": " | ".join(dict.fromkeys(keep))})
                elif b.get("type") == "text" and kind == "assistant":
                    blocks.append({"type": "text", "text": b.get("text", "")[:200]})
            if blocks:
                out.append({"type": kind, "message": {"content": blocks}})
    return out


# --------------------------------------------------------------------------------------- outcome helpers


def tree_hash(root: Path) -> dict:
    h = {}
    skip = (".venv", "results", ".claude", ".codex", "__pycache__", ".robotcode_cache", ".git")
    for p in root.rglob("*"):
        if p.is_file() and not any(x in p.parts for x in skip):
            h[str(p.relative_to(root)).replace("\\", "/")] = hashlib.md5(p.read_bytes()).hexdigest()
    return h


def _json_tail(text: str) -> Any:
    return json.loads(text[text.index("{"):])


def analyze_errors(env, root: Path, *targets: str) -> int | None:
    """Error count of `robotcode analyze code <targets>` (None if it could not be determined)."""
    if not targets:
        return 0
    _, out = env.rc(root, "-f", "json", "analyze", "code", *targets)
    try:
        return int(_json_tail(out)["summary"]["errors"])
    except (ValueError, KeyError, TypeError):
        return None


def written_suite_files(t: Transcript, root: Path) -> list[str]:
    files = []
    for c in t.calls:
        for f in _written_files(c, str(root)):
            if f not in files and (root / f).is_file():
                files.append(f)
    return files


def run_suite(env, root: Path, *selection: str, key: str = "name", extra_env: dict | None = None) -> dict:
    """Run `robotcode robot <selection>` into a scratch output; return {test name|fullName: status}."""
    out = root / "results" / f"_grade_{uuid.uuid4().hex[:6]}.xml"
    try:
        env.rc(root, "robot", "--output", str(out), "--log", "NONE", "--report", "NONE", *selection, **(extra_env or {}))
        _, js = env.rc(root, "-f", "json", "results", "show", "-o", str(out))
        data = _json_tail(js)
        return {t[key]: t["status"] for t in data.get("tests", [])}
    except Exception as e:  # noqa: BLE001 - the grade records the failure
        return {"_error": repr(e)}
    finally:
        if out.exists():
            out.unlink()


def statuses(st: dict) -> dict:
    return {k: v for k, v in st.items() if not k.startswith("_")}


def all_pass(st: dict) -> bool:
    s = statuses(st)
    return bool(s) and "_error" not in st and all(v == "PASS" for v in s.values())


# --------------------------------------------------------------------------------------- baseline outcome


def _ran_by_tag(t: Transcript, tag: str) -> bool:
    for c in t.named("rf_run"):
        if tag in _as_list(c.input.get("include_tags")):
            return True
    return any(re.search(rf"(-i|--include)\s+{tag}\b", c.command) for c in t.calls)


def _ran_exactly(t: Transcript, longnames: set[str]) -> bool:
    """A run selected exactly these tests by longname (e.g. the ones discover listed for a tag)."""
    for c in t.calls:
        sel = _run_selection(c, t.cwd)
        if sel and not sel["paths"] and set(sel["longnames"]) == longnames:
            return True
    return False


SMOKE_LONGNAMES = {"Tests.Api.Booking.Ping Service", "Tests.Calc.Calc.Add Two Numbers",
                   "Tests.Calc.Calc.Format Money In Euro"}


def _used_shell(t: Transcript, *needles: str) -> bool:
    return any(n in c.command for c in t.calls for n in needles)


def _any_run(t: Transcript) -> bool:
    return any(_run_selection(c, t.cwd) is not None for c in t.calls)


def grade_baseline(scenario, root: Path, t: Transcript, env, before: dict) -> dict:
    g: dict[str, Any] = {}
    after = tree_hash(root)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    g["changed_files"] = changed
    final = t.final
    name = scenario.name
    if name == "s1_triage_fix":
        code, _ = env.rc(root, "robot", "--output", "NONE", "--log", "NONE", "--report", "NONE",
                         "-bl", "Tests.Calc.Calc.Ten Percent Discount")
        g["test_passes"] = code == 0
        g["expectation_untouched"] = "200    10    180" in (root / "tests/calc/calc.robot").read_text(encoding="utf-8")
        g["fixed_lib"] = any("CalcLib.py" in c for c in changed)
        g["used_results"] = bool(t.named("rf_results")) or _used_shell(t, "results show", "results log", "results summary")
        g["pass"] = g["test_passes"] and g["expectation_untouched"] and g["fixed_lib"]
    elif name == "s2_smoke":
        g["ran_by_tag"] = _ran_by_tag(t, "smoke")
        g["mentions_3"] = any(s in final for s in ("3 tests", "three", "3/3", "3 passed", "All 3", "all 3"))
        g["names"] = all(n in final for n in ("Ping Service", "Add Two Numbers", "Format Money In Euro"))
        g["ran_smoke_by_longname"] = _ran_exactly(t, SMOKE_LONGNAMES)
        g["pass"] = (g["ran_by_tag"] or g["ran_smoke_by_longname"]) and g["names"]
    elif name == "s3_analyze":
        g["no_run"] = not _any_run(t)
        g["found_kw"] = "Add Numbrs" in final
        g["found_args"] = "Format Money" in final
        g["found_runid"] = "RUN_ID" in final
        g["used_analyze"] = bool(t.named("rf_analyze")) or _used_shell(t, "analyze code")
        g["pass"] = g["no_run"] and g["found_kw"] and g["found_args"]
    elif scenario.file:
        f = root / scenario.file
        g["suite_exists"] = f.is_file()
        if f.is_file():
            code, out = env.rc(root, "robot", "--output", "NONE", "--log", "NONE", "--report", "NONE",
                               "-bl", scenario.longname, timeout=900)
            g["suite_passes"] = code == 0
            g["run_tail"] = out[-400:]
            g["analyze_errors"] = analyze_errors(env, root, scenario.file)
        g["stepwise"] = bool(t.named("rf_repl")) or _used_shell(t, "session start repl", "robotcode repl", "session send")
        g["used_save"] = any(c.action == "save" for c in t.named("rf_repl")) or t.mentions(".save")
        g["pass"] = bool(g.get("suite_exists") and g.get("suite_passes"))
    elif name == "s7_debug_inspect":
        g["no_changes"] = not changed
        g["answer"] = "apple" in final and "pear" in final
        g["used_debugger"] = bool(t.named("rf_debug")) or _used_shell(t, "session start debug", "robot-debug", "robotcode debug")
        g["used_repl_wrongly"] = bool(t.named("rf_repl")) or _used_shell(t, "session start repl", "robotcode repl")
        g["pass"] = g["no_changes"] and g["answer"] and g["used_debugger"]
    return g


# --------------------------------------------------------------------------------------- demoshop outcome

CO = "Tests.Ui.Checkout."
CA = "Tests.Ui.Catalogue."


def _full_reruns(t: Transcript) -> int:
    n = 0
    for c in t.named("rf_run"):
        if not any(c.input.get(k) for k in ("by_longname", "include_tags", "rerunfailed")):
            n += 1
    for c in t.calls:
        for seg in _segments(c.command):
            args = _shell_run_args(seg)
            if args is not None and not any(a in ("-bl", "--by-longname", "-t", "--test", "-i", "--include", "--rerunfailed",
                                                   "-s", "--suite") or a.endswith(".robot") for a in args):
                n += 1
                break
    return n


def _reports_t3_defects(final: str) -> bool:
    """t3: the final answer names both shop defects, the wrong card price and the missing tax.

    A report may name a defect by its test id instead of the word (2026-11-07: "WEB-002_AC-1: API says ... `$79.00`;
    UI renders `$90.85`" was graded as not reporting the price defect).
    """
    final = final.lower()
    flagged = "defect" in final or "bug" in final
    price = "price" in final or (flagged and "web-002_ac-1" in final)
    tax = "tax" in final or (flagged and "web-006_ac-1" in final)
    return price and tax


def grade_demoshop(scenario, root: Path, t: Transcript, env, shop) -> dict:
    g: dict[str, Any] = {}
    g["analyze_errors"] = analyze_errors(env, root, "tests", "resources")
    p = env.run_python(root, "hooks/no_inline_locators.py", "tests/")
    g["inline_locators"] = len([ln for ln in p.stdout.splitlines() if ln.strip()]) if p.returncode else 0
    g["read_output_xml"] = bool(output_xml_reads(t))
    g["used_results"] = bool(t.named("rf_results")) or _used_shell(t, "results show", "results log", "results summary")
    g["used_debugger"] = bool(t.named("rf_debug")) or _used_shell(t, "session start debug", "robot-debug")
    g["used_analyze"] = bool(t.named("rf_analyze")) or _used_shell(t, "analyze code")
    final = t.final.lower()

    def status(preset, select, tag):
        return shop.run_status(env, root, preset, select, tag=tag)

    name = scenario.name
    if name == "t1_checkout":
        f = root / "tests/ui/checkout.robot"
        g["suite_exists"] = f.is_file()
        if not f.is_file():
            return dict(g, **{"pass": False})
        txt = f.read_text(encoding="utf-8")
        g["acs_present"] = [ac for ac in ("AC-1", "AC-7", "AC-8", "AC-10", "AC-11", "AC-12") if f"WEB-006_{ac} " in txt]
        clean, stage4, buggy = (status(p, ["Tests.Ui.Checkout"], tag) for p, tag in
                                (("clean", "t1c"), ("stage4", "t1s"), ("buggy", "t1b")))
        g["clean"], g["stage4"], g["buggy"] = clean, stage4, buggy
        g["clean_all_pass"] = all_pass(clean)
        g["stage4_all_pass"] = all_pass(stage4)
        ac1 = [k for k in buggy if "AC-1 " in k]
        g["ac1_catches_tax_bug"] = bool(ac1) and all(buggy[k] == "FAIL" for k in ac1)
        g["pass"] = g["clean_all_pass"] and len(g["acs_present"]) == 6
        g["quality"] = sum([g["stage4_all_pass"], g["ac1_catches_tax_bug"], g["inline_locators"] == 0,
                            g["analyze_errors"] == 0])
    elif name == "t2_hybrid":
        f = root / "tests/ui/cart_hybrid.robot"
        g["suite_exists"] = f.is_file()
        if not f.is_file():
            return dict(g, **{"pass": False})
        src = f.read_text(encoding="utf-8")
        allsrc = src + "".join(p.read_text(encoding="utf-8") for p in (root / "resources").glob("*.resource"))
        g["uses_session_header"] = "X-Session-ID" in allsrc
        g["ui_add_click"] = "Add Product To Cart From Its Page" in src
        clean = status("clean", ["Tests.Ui.Cart Hybrid"], "t2c")
        clean2 = status("clean", ["Tests.Ui.Cart Hybrid"], "t2d")
        g["clean"] = clean
        g["passes_twice"] = all_pass(clean) and all_pass(clean2)
        g["pass"] = g["passes_twice"] and g["uses_session_header"] and not g["ui_add_click"]
        g["quality"] = sum([g["inline_locators"] == 0, g["analyze_errors"] == 0])
    elif name == "t3_drift":
        stage4 = status("stage4", ["Tests.Ui"], "t3s")
        dab = status("drift_and_bug", ["Tests.Ui"], "t3d")
        clean = status("clean", ["Tests.Ui"], "t3c")
        g["stage4"], g["drift_and_bug"], g["clean"] = stage4, dab, clean
        drift = [CA + "WEB-002_AC-7 Audio Filter Shows Only Audio", CO + "WEB-006_AC-7 Successful Order",
                 CA + "WEB-002_AC-1 Card Prices Are The Product Prices", CO + "WEB-006_AC-1 Order Total Adds Up"]
        defects = [CA + "WEB-002_AC-1 Card Prices Are The Product Prices", CO + "WEB-006_AC-1 Order Total Adds Up"]
        g["drift_repaired"] = sum(stage4.get(x) == "PASS" for x in drift)
        g["defects_still_fail"] = all(dab.get(x) == "FAIL" for x in defects)
        g["clean_ok"] = all(clean.get(x) == "PASS" for x in drift)
        g["reports_defects"] = _reports_t3_defects(final)
        g["pass"] = g["drift_repaired"] == 4 and g["defects_still_fail"] and g["clean_ok"] and g["reports_defects"]
        g["quality"] = sum([not g["read_output_xml"], g["used_results"], g["inline_locators"] == 0])
    elif name == "t6_triage":
        tri = root / "results/triage.md"
        g["triage_exists"] = tri.is_file()
        text = tri.read_text(encoding="utf-8").lower() if tri.is_file() else ""
        g["triage_mentions"] = {k: k in text for k in ("tax", "price", "button", "its empty", "confirm", "audios")}
        clean = status("clean", [CO + "WEB-006_AC-11 Validation Errors Next To Fields", CO + "WEB-006_AC-12 Cart Cleared After Order",
                                 CA + "WEB-002_AC-7 Audio Filter Shows Only Audio"], "t6c")
        buggy = status("buggy", [CA + "WEB-002_AC-1 Every Card Offers Add To Cart", CA + "WEB-002_AC-1 Card Prices Are The Product Prices",
                                 CO + "WEB-006_AC-1 Order Total Adds Up"], "t6b")
        g["clean"], g["buggy"] = clean, buggy
        g["test_bugs_fixed"] = sum(v == "PASS" for v in statuses(clean).values())
        g["defects_still_fail"] = len(buggy) == 3 and all(v == "FAIL" for v in buggy.values())
        g["full_reruns"] = _full_reruns(t)
        g["pass"] = g["triage_exists"] and g["test_bugs_fixed"] == 3 and g["defects_still_fail"] and not g["read_output_xml"]
        g["quality"] = sum(g["triage_mentions"].values())
    return g


# --------------------------------------------------------------------------------------- fixtures / api outcome

EXPECTED_WIDGET_TEXTS = ["Welcome, alice", "deep ok", "Paid with card ending 1111", "Confirmed", "Accepted",
                         "Coupon SAVE10 applied", "card 9999 verified"]


def _tests_grade(g: dict, st: dict) -> dict:
    tests = statuses(st)
    g["status"] = st
    g["passed"], g["total"] = sum(v == "PASS" for v in tests.values()), len(tests)
    g["pass"] = g["total"] > 0 and g["passed"] == g["total"]
    return g


def grade_fixtures(scenario, root: Path, t: Transcript, env, ctx) -> dict:
    f = root / scenario.file
    g: dict[str, Any] = {"file_exists": f.exists()}
    if not f.exists():
        return dict(g, passed=0, total=0, **{"pass": False})
    src = f.read_text(encoding="utf-8", errors="replace")
    g["static"] = static_counts(src)
    if scenario.name == "fx_api":
        space = f"ev-fxg-{uuid.uuid4().hex[:8]}"
        st = run_suite(env, root, "-v", f"SHOP_SPACE:{space}", scenario.file)
        st2 = run_suite(env, root, "-v", f"SHOP_SPACE:{space}", scenario.file)
        ctx.shop.reset(space)
        g["rerun_same_space_all_pass"] = all_pass(st2)
    else:
        st = run_suite(env, root, scenario.file)
        g["expected_texts"] = sum(x in src for x in EXPECTED_WIDGET_TEXTS)
    return _tests_grade(g, st)


def isolation(env, root: Path, rel: str, names: Iterable[str]) -> int:
    """14.9: run each test alone (`-t`) with a fresh ledger user; count the tests that pass."""
    ok = 0
    for n in names:
        st = run_suite(env, root, "-v", f"LEDGER_USER:iso-{uuid.uuid4().hex[:6]}", "-t", n, rel)
        ok += st.get(n) == "PASS"
    return ok


def grade_api(scenario, root: Path, t: Transcript, env, ctx) -> dict:
    rel = scenario.file
    f = root / rel
    g: dict[str, Any] = {"file_exists": f.exists()}
    if not f.exists():
        return dict(g, passed=0, total=0, **{"pass": False})
    g["static"] = static_counts(f.read_text(encoding="utf-8", errors="replace"))
    if scenario.bugs:
        user = f"grader-{uuid.uuid4().hex[:8]}"
        st = run_suite(env, root, "-v", f"LEDGER_USER:{user}", rel)
        st2 = run_suite(env, root, "-v", f"LEDGER_USER:{user}", rel)
        g["rerun_same_user_all_pass"] = all_pass(st2)
        caught = {}
        for bug in scenario.bugs:
            sb = run_suite(env, root, "-v", f"LEDGER_USER:m-{uuid.uuid4().hex[:6]}", "-v",
                           f"LEDGER_URL:{ctx.ledger.bug_urls[bug]}", rel)
            caught[bug] = any(v != "PASS" for v in statuses(sb).values())
        g["bugs_caught"] = caught
        g["bugs_caught_n"] = sum(caught.values())
    else:
        st = run_suite(env, root, rel)
        st2 = run_suite(env, root, rel)
        g["rerun_all_pass"] = all_pass(st2)
    _tests_grade(g, st)
    g["isolated_pass"] = isolation(env, root, rel, list(statuses(st)))
    return g
