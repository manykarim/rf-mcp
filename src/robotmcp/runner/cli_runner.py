"""Stateless RobotCode CLI runner and command synthesizer (`server` and `client` modes)."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

import tomlkit

from robotmcp.config import ResolvedMode, ServerConfig, ShellKind
from robotmcp.hints import Hint, HintContext, HintEngine
from robotmcp.runner.process import run_process
from robotmcp.runner.shell_escape import build_agent_env, format_cli_command

DiscoverAction = Literal[
    "tests",
    "tasks",
    "suites",
    "tags",
    "files",
    "metadata",
    "info",
    "all",
    "config",
    "profiles",
    "profile_show",
]
ResultsAction = Literal["summary", "show", "log", "stats", "diff", "merge"]
DocAction = Literal["lib", "keywords", "keyword"]

ROBOTCODE_INSTALL_HINT = 'uv add --dev --prerelease=allow "robotcode[all]>=2.9.0.dev115"'


def resolve_robotcode_executable() -> list[str]:
    """Resolve the robotcode CLI executable in the current environment."""
    which_rc = shutil.which("robotcode")
    if which_rc:
        return [which_rc]
    venv_dir = Path(sys.executable).parent
    for candidate_name in ("robotcode.exe", "robotcode"):
        candidate = venv_dir / candidate_name
        if candidate.is_file():
            return [str(candidate)]
    return ["robotcode"]


def _load_project_config(root: Path) -> dict[str, Any]:
    """Return the robotcode settings of robot.toml / .robot.toml / pyproject.toml in `root`."""
    for name in ("robot.toml", ".robot.toml"):
        cfg_file = root / name
        if cfg_file.is_file():
            try:
                return dict(tomlkit.parse(cfg_file.read_text(encoding="utf-8")).unwrap())
            except Exception:
                return {}
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            doc = tomlkit.parse(pyproject.read_text(encoding="utf-8")).unwrap()
            rc_cfg = doc.get("tool", {}).get("robotcode", {})
            return rc_cfg if isinstance(rc_cfg, dict) else {}
        except Exception:
            return {}
    return {}


def configured_paths(root: Path) -> list[str]:
    """Return the `paths` configured for robotcode in `root` (empty when none)."""
    paths = _load_project_config(root).get("paths") or []
    if isinstance(paths, str):
        return [paths]
    return [str(p) for p in paths]


def _has_configured_paths(root: Path) -> bool:
    """Return True if robot.toml, .robot.toml, or pyproject.toml in `root` defines robotcode `paths`."""
    return bool(configured_paths(root))


def _longname_flag(ln: str) -> str:
    """Return '-bl' for exact dotted longnames ('Suite.Test') and '-t' for short test names or wildcards ('*Test*')."""
    if "." in ln and "*" not in ln and "?" not in ln:
        return "-bl"
    return "-t"


_CONSOLE_STATUS_RE = re.compile(r"^(?P<name>.+?)\s+\|\s+(?P<status>FAIL|PASS|SKIP)\s+\|\s*$")
_CONSOLE_COUNTS_RE = re.compile(r"^\d+ (tests?|tasks?)\b")


def parse_console_failures(console: str) -> list[str]:
    """Extract `<test>: <message>` for each failing test from `robot` console output."""
    lines = console.splitlines()
    failures: list[str] = []
    for i, line in enumerate(lines):
        m = _CONSOLE_STATUS_RE.match(line.rstrip())
        if not m or m.group("status") != "FAIL":
            continue
        message: list[str] = []
        for follow in lines[i + 1 :]:
            if follow.startswith(("---", "===")):
                break
            message.append(follow.strip())
        if message and _CONSOLE_COUNTS_RE.match(message[-1]):
            continue  # suite status line, not a test
        failures.append(f"{m.group('name').strip()}: {' '.join(message).strip()}")
    return failures


_CONSOLE_TOTALS_RE = re.compile(r"^\d+ (?:tests?|tasks?), \d+ passed, \d+ failed(?:, \d+ skipped)?$")


def run_outcome(result: CommandResult, *, dryrun: bool = False) -> str | None:
    """One line verdict for rf_run, shown first: `FAIL: 6 tests, 2 passed, 4 failed. ...`.

    2026-11-09 OpenRouter sweep: Qwen3-Coder ended four authoring runs with "I have successfully created..." after an
    rf_run with exit_code 1 and a `failures` list - both sat behind the echoed command. None for timeouts and client mode.
    """
    if result.mode != "server" or result.status is not None or result.exit_code is None:
        return None
    totals = [line.strip() for line in result.stdout.splitlines() if _CONSOLE_TOTALS_RE.match(line.strip())]
    counts = totals[-1] if totals else ""
    if result.exit_code == 0 and not result.failures:
        if dryrun:
            return f"PASS (dry run: keywords and arguments resolved, nothing executed): {counts or 'no failures'}"
        return f"PASS: {counts}" if counts else "PASS"
    if not counts:
        return (f"FAIL: robot exited with code {result.exit_code} without running tests (see stderr). "
                "Nothing passed yet.")
    return f"FAIL: {counts}. The tests do not pass yet: fix them and rerun before you report the task as done."


ANALYZE_CLEAN_NOTE = ("Clean analysis checks keyword names and arguments only; it does not run anything. After "
                      "writing or changing tests, run them with rf_run before you report that they work.")


_SEVERITIES = ("error", "warning", "info", "hint")
_SEVERITY_ALIASES = {"warn": "warning", "information": "info"}


def severity_or_worse(severity: str) -> str:
    """`info` -> `error,warning,info`: `analyze code --severity` reports only the listed severities, while
    rf_analyze promises "this severity or worse" (agents passing `info` to see everything got no errors at all).
    A list such as `error,hint` is passed through."""
    value = severity.strip().lower()
    level = _SEVERITY_ALIASES.get(value, value)
    if level not in _SEVERITIES:
        return severity
    return ",".join(_SEVERITIES[: _SEVERITIES.index(level) + 1])


def decode_analyze_exit_code(code: int | None) -> dict[str, bool]:
    """Decode the `analyze code` exit-code bitmask (1 error, 2 warning, 4 info, 8 hint)."""
    value = code or 0
    return {
        "has_errors": bool(value & 1),
        "has_warnings": bool(value & 2),
        "has_infos": bool(value & 4),
        "has_hints": bool(value & 8),
    }


def _project_discover_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for item in items:
        source = item.get("relSource") or item.get("source") or ""
        if item.get("lineno"):
            source = f"{source}:{item['lineno']}"
        entry: dict[str, Any] = {"longname": item.get("longname"), "source": source}
        if item.get("tags"):
            entry["tags"] = item["tags"]
        projected.append(entry)
    return projected


def project_discover_data(action: str, data: Any) -> Any:
    """Compact projection of `discover` JSON: longname, relSource:lineno and tags only."""
    if not isinstance(data, dict):
        return data
    if action in ("tests", "tasks", "suites") and isinstance(data.get("items"), list):
        return {"items": _project_discover_items(data["items"])}
    if action == "tags" and isinstance(data.get("tags"), dict):
        return {
            "tags": {
                tag: [i.get("longname") for i in items if isinstance(i, dict)]
                for tag, items in data["tags"].items()
            }
        }
    return data


_LOG_NOISE = ("startTime", "timestamp", "elapsedSeconds")


def _compact_log_node(node: Any) -> Any:
    if isinstance(node, list):
        return [_compact_log_node(n) for n in node if not (isinstance(n, dict) and n.get("status") == "NOT RUN")]
    if isinstance(node, dict):
        return {k: _compact_log_node(v) for k, v in node.items()
                if k not in _LOG_NOISE and not (k == "isHtml" and v is False)
                and not (k == "source" and "relSource" in node)}
    return node


def compact_result_log(data: Any) -> Any:
    """Shrink `results log` JSON: drop per-node timestamps, `isHtml: false` and NOT RUN branches.

    Measured on the 2026-11-07 Codex demoshop runs: 12 failed-test logs went from 156k to 104k characters, so a
    log fits a small context window (and the 20k output cap) with a third more of its keywords. Each test keeps
    its own `elapsedSeconds`.
    """
    if not isinstance(data, dict) or not isinstance(data.get("tests"), list):
        return data
    tests = []
    for test in data["tests"]:
        compact = _compact_log_node(test)
        if isinstance(test, dict) and "elapsedSeconds" in test:
            compact["elapsedSeconds"] = test["elapsedSeconds"]
        tests.append(compact)
    return {**data, "tests": tests}


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _longest_list(value: Any, path: str = "") -> tuple[str, list | None]:
    best: tuple[str, list | None] = ("", None)
    if isinstance(value, list):
        best = (path or "data", value)
        children = enumerate(value)
    elif isinstance(value, dict):
        children = value.items()
    else:
        return best
    for key, child in children:
        cand = _longest_list(child, f"{path}.{key}" if path else str(key))
        if cand[1] is not None and (best[1] is None or len(cand[1]) > len(best[1])):
            best = cand
    return best


def truncate_data(data: Any, limit: int) -> tuple[Any, str | None]:
    """Shorten `data` until it serializes within `limit` characters: cut a string that alone takes half the budget
    (e.g. `doc lib` markdown, 83k characters for the XML library), then halve the longest lists, then cut strings."""
    if data is None or _json_size(data) <= limit:
        return data, None
    trimmed = json.loads(json.dumps(data, default=str))
    notes: dict[str, int] = {}
    cut: list[str] = []

    def cut_strings(min_len: int) -> None:
        for _ in range(10):
            excess = _json_size(trimmed) - limit
            if excess <= 0:
                return
            path, holder, key = _longest_string(trimmed)
            if holder is None or len(holder[key]) <= min_len:
                return
            text = holder[key]
            holder[key] = truncate_text(text, max(500, len(text) - excess - 200))[0]
            cut.append(path)

    cut_strings(limit // 2)
    for _ in range(50):
        if _json_size(trimmed) <= limit:
            break
        path, lst = _longest_list(trimmed)
        if lst is None or len(lst) <= 1:
            break
        notes.setdefault(path, len(lst))
        del lst[max(1, len(lst) // 2) :]
    cut_strings(500)
    parts = []
    if notes:
        parts.append("shortened lists: " + ", ".join(f"{p} (of {n})" for p, n in notes.items()))
    if cut:
        parts.append("cut text: " + ", ".join(dict.fromkeys(cut)))
    return trimmed, f"Output exceeded {limit} characters; {'; '.join(parts)}. Narrow the query."


def _longest_string(value: Any, path: str = "") -> tuple[str, Any, Any]:
    """(path, container, key) of the longest string in `value`; container is None when there is none."""
    best: tuple[str, Any, Any] = ("", None, None)
    best_len = -1
    children = enumerate(value) if isinstance(value, list) else value.items() if isinstance(value, dict) else ()
    for key, child in children:
        child_path = f"{path}.{key}" if path else str(key)
        if isinstance(child, str):
            cand = (child_path, value, key)
        elif isinstance(child, (list, dict)):
            cand = _longest_string(child, child_path)
        else:
            continue
        if cand[1] is not None and len(cand[1][cand[2]]) > best_len:
            best, best_len = cand, len(cand[1][cand[2]])
    return best


def truncate_text(text: str, limit: int) -> tuple[str, str | None]:
    """Cut `text` to `limit` characters, keeping head and tail."""
    if len(text) <= limit:
        return text, None
    head = text[: limit * 2 // 3]
    tail = text[-(limit // 3) :]
    return (
        f"{head}\n... [{len(text) - limit} characters cut] ...\n{tail}",
        f"Output exceeded {limit} characters and was cut. Narrow the query.",
    )


class CommandResult(BaseModel):
    """Structured result of a `robotcode` CLI invocation or `client`-mode synthesis."""

    mode: ResolvedMode
    command: list[str]
    shell_command: str | None = None
    status: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    data: Any = None
    failures: list[str] | None = None
    hints: list[Hint] = Field(default_factory=list)
    warning: str | None = None
    truncated: str | None = None
    instruction: str | None = None

    def to_response(self) -> dict[str, Any]:
        """Return the tool response with every payload once and no empty fields.

        A successful server-mode call leaves out `command`: the agent does not run it, and the echoed argv (with the
        absolute root) was most of a short response. Failures keep it for reproduction; client mode always has it.
        """
        out = self.model_dump(exclude_none=True)
        for key in ("stdout", "stderr", "hints"):
            if not out.get(key):
                out.pop(key, None)
        if self.mode == "server" and self.exit_code == 0:
            out.pop("command", None)
        return out


class CliRunner:
    """Executes stateless `robotcode` CLI subcommands (`server` mode) or synthesizes shell-safe commands (`client` mode)."""

    def __init__(
        self,
        config: ServerConfig,
        hint_engine: HintEngine | None = None,
    ) -> None:
        self.config = config
        self.hint_engine = hint_engine or HintEngine(
            hints_file=config.resolve_hints_file()
        )

    def _mode(self, mode: str | None) -> ResolvedMode:
        return self.config.resolved_mode(mode)

    def _effective_paths(
        self,
        root: str | Path | None,
        paths: Sequence[str] | None,
        *,
        mode: str | None = None,
    ) -> list[str]:
        if paths:
            return list(paths)
        if self._mode(mode) == "client":
            return []
        resolved_root = self.config.resolve_root(root)
        if _has_configured_paths(resolved_root):
            return []
        return ["."]

    def build_global_args(
        self,
        *,
        root: str | Path | None = None,
        profile: str | None = None,
        json_format: bool = True,
        mode: str | None = None,
    ) -> list[str]:
        """Build `robotcode [-r <root>] [-p <profile>] [-f json]` global prefix.

        In `client` mode the agent-supplied root is passed verbatim (it is a path on the
        client machine) and `-r` is omitted when none is given.
        """
        args: list[str] = ["robotcode"]
        if self._mode(mode) == "client":
            if root is not None and str(root):
                args.extend(["-r", str(root)])
        else:
            args.extend(["-r", str(self.config.resolve_root(root))])
        effective_profile = profile if profile is not None else self.config.profile
        if effective_profile:
            args.extend(["-p", effective_profile])
        if json_format:
            args.extend(["-f", "json"])
        return args

    def build_discover_args(
        self,
        action: DiscoverAction = "tests",
        *,
        root: str | Path | None = None,
        profile: str | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        by_longname: Sequence[str] | None = None,
        by_test_metadata: Sequence[str] | None = None,
        exclude_by_test_metadata: Sequence[str] | None = None,
        include_values: bool = True,
        search: str | None = None,
        search_regex: str | None = None,
        paths: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> list[str]:
        """Build CLI arguments for `rf_discover` (`discover`, `config`, `profiles`)."""
        base = self.build_global_args(root=root, profile=profile, json_format=True, mode=mode)
        if action == "config":
            base.extend(["config", "show"])
            if paths:
                base.extend(paths)
            return base
        if action == "profiles":
            base.extend(["profiles", "list"])
            return base
        if action == "profile_show":
            base.extend(["profiles", "show"])
            return base

        base.extend(["discover", action])
        if action in ("info", "files"):
            if action == "files":
                base.extend(self._effective_paths(root, paths, mode=mode))
            return base
        if action == "metadata" and include_values:
            base.append("--values")
        if search and action in ("tests", "tasks", "suites", "all"):
            base.extend(["--search", search])
        if search_regex and action in ("tests", "tasks", "suites", "all"):
            base.extend(["--search-regex", search_regex])
        for tag in include_tags or []:
            base.extend(["-i", tag])
        for tag in exclude_tags or []:
            base.extend(["-e", tag])
        for ln in by_longname or []:
            flag = _longname_flag(ln) if action in ("tests", "tasks", "all") else "-bl"
            base.extend([flag, ln])
        for btm in by_test_metadata or []:
            base.extend(["-btm", btm])
        for ebtm in exclude_by_test_metadata or []:
            base.extend(["-ebtm", ebtm])
        base.extend(self._effective_paths(root, paths, mode=mode))
        return base

    def build_analyze_args(
        self,
        *,
        root: str | Path | None = None,
        profile: str | None = None,
        severity: str | None = None,
        code: str | None = None,
        filter_glob: str | None = None,
        collect_unused: bool = False,
        paths: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> list[str]:
        """Build CLI arguments for `rf_analyze` (`robotcode -r <root> -f json analyze code`)."""
        args = self.build_global_args(root=root, profile=profile, json_format=True, mode=mode)
        args.extend(["analyze", "code"])
        if severity:
            args.extend(["--severity", severity_or_worse(severity)])
        if code:
            args.extend(["--code", code])
        if filter_glob:
            args.extend(["--filter", filter_glob])
        if collect_unused:
            args.append("--collect-unused")
        if paths:
            args.extend(paths)
        return args

    def build_doc_args(
        self,
        action: DocAction = "keyword",
        *,
        library: str | None = None,
        pattern: str | Sequence[str] | None = None,
        root: str | Path | None = None,
        profile: str | None = None,
        mode: str | None = None,
    ) -> list[str]:
        """Build CLI arguments for `rf_doc` (`robotcode -r <root> -f json doc {lib|keywords|keyword}`)."""
        args = self.build_global_args(root=root, profile=profile, json_format=True, mode=mode)
        args.extend(["doc", action])
        if isinstance(pattern, str):
            raw = pattern.split("|") if "|" in pattern else [pattern]
        else:
            raw = list(pattern or [])
        patterns = [p.strip() for p in raw if p and p.strip()]
        if library:
            args.append(library)
        if action in ("keywords", "keyword"):
            args.extend(patterns)
        return args

    def build_run_args(
        self,
        *,
        root: str | Path | None = None,
        profile: str | None = None,
        by_longname: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        by_test_metadata: Sequence[str] | None = None,
        exclude_by_test_metadata: Sequence[str] | None = None,
        variables: Sequence[str] | None = None,
        rpa: bool = False,
        wrapper: str | None = None,
        outputdir: str | None = None,
        output: str | None = None,
        rerunfailed: str | None = None,
        dryrun: bool = False,
        paths: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> list[str]:
        """Build CLI arguments for `rf_run` (`robotcode -r <root> [-p <prof>] robot ...`)."""
        args = self.build_global_args(root=root, profile=profile, json_format=False, mode=mode)
        args.append("robot")
        if wrapper:
            args.extend(["--wrapper", wrapper])
        if rpa:
            args.append("--rpa")
        if dryrun:
            args.append("--dryrun")
        if outputdir:
            args.extend(["--outputdir", outputdir])
        if output:
            args.extend(["--output", output])
        if rerunfailed:
            args.extend(["--rerunfailed", rerunfailed])
        for ln in by_longname or []:
            args.extend([_longname_flag(ln), ln])
        for tag in include_tags or []:
            args.extend(["-i", tag])
        for tag in exclude_tags or []:
            args.extend(["-e", tag])
        for btm in by_test_metadata or []:
            args.extend(["-btm", btm])
        for ebtm in exclude_by_test_metadata or []:
            args.extend(["-ebtm", ebtm])
        for var in variables or []:
            args.extend(["-v", var])
        args.extend(self._effective_paths(root, paths, mode=mode))
        return args

    def build_results_args(
        self,
        action: ResultsAction = "summary",
        *,
        root: str | Path | None = None,
        profile: str | None = None,
        failed_only: bool = False,
        include_tags: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        by_longname: Sequence[str] | None = None,
        by_test_metadata: Sequence[str] | None = None,
        exclude_by_test_metadata: Sequence[str] | None = None,
        search: str | None = None,
        search_regex: str | None = None,
        sort: str | None = None,
        level: str | None = None,
        max_depth: int | None = None,
        message_chars: int | None = None,
        by: Sequence[str] | None = None,
        top: int | None = None,
        extract_dir: str | None = None,
        output_path: str | None = None,
        baseline: str | None = None,
        only: Sequence[str] | None = None,
        merge_inputs: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> list[str]:
        """Build CLI arguments for `rf_results` (`robotcode -f json results ...` or `rebot --merge`)."""
        if action == "merge":
            if not merge_inputs or len(merge_inputs) < 2:
                raise ValueError("action='merge' needs `merge_inputs` with at least two result files (first the full run, then the reruns).")
            args = self.build_global_args(root=root, profile=profile, json_format=False, mode=mode)
            # `--outputdir .` keeps output_path relative to the root, like every other output_path
            args.extend(
                [
                    "rebot",
                    "--merge",
                    "--outputdir",
                    ".",
                    "--output",
                    output_path or self.default_merge_output(root, mode=mode),
                    "--log",
                    "NONE",
                    "--report",
                    "NONE",
                ]
            )
            args.extend(merge_inputs)
            return args

        args = self.build_global_args(root=root, profile=profile, json_format=True, mode=mode)
        args.extend(["results", action])
        if action == "diff":
            if not baseline:
                raise ValueError("action='diff' needs `baseline` (the older result file).")
            args.append(baseline)
            if output_path:
                args.append(output_path)
            for value in only or []:
                args.extend(["--only", value])
        elif output_path:
            args.extend(["-o", output_path])
        if failed_only and action in ("summary", "show", "log", "stats"):
            args.append("--failed")
        if search:
            args.extend(["--search", search])
        if search_regex:
            args.extend(["--search-regex", search_regex])
        if sort and action in ("show", "stats"):
            args.extend(["--sort", sort])
        if top is not None and action in ("show", "stats"):
            args.extend(["--top", str(top)])
        if message_chars is not None and action in ("show", "diff"):
            args.extend(["--message-chars", str(message_chars)])
        if action == "log":
            if level:
                args.extend(["--level", level.lower()])
            if max_depth is not None:
                args.extend(["--max-depth", str(max_depth)])
            if extract_dir:
                args.extend(["--extract", extract_dir])
        if action == "stats":
            for value in by or []:
                args.extend(["--by", value])
        for tag in include_tags or []:
            args.extend(["-i", tag])
        for tag in exclude_tags or []:
            args.extend(["-e", tag])
        for ln in by_longname or []:
            args.extend(["-bl", ln])
        for btm in by_test_metadata or []:
            args.extend(["-btm", btm])
        for ebtm in exclude_by_test_metadata or []:
            args.extend(["-ebtm", ebtm])
        return args

    def default_merge_output(self, root: str | Path | None, *, mode: str | None = None) -> str:
        """`<configured output-dir>/merged.xml`, or `merged.xml` when none is configured."""
        if self._mode(mode) == "client":
            return "merged.xml"
        output_dir = _load_project_config(self.config.resolve_root(root)).get("output-dir")
        return f"{output_dir}/merged.xml" if output_dir else "merged.xml"

    def selection_warning(
        self,
        *,
        root: str | Path | None,
        paths: Sequence[str] | None,
        by_longname: Sequence[str] | None,
        mode: str | None = None,
    ) -> str | None:
        """Warn when a single `.robot` file inside the configured `paths` is used as the run target."""
        files = [p for p in paths or [] if p.lower().endswith(".robot")]
        if len(files) != 1 or self._mode(mode) == "client":
            return None
        resolved_root = self.config.resolve_root(root)
        cfg_paths = configured_paths(resolved_root)
        if not cfg_paths:
            return None
        target = (resolved_root / files[0]).resolve()
        inside = any(
            target == (resolved_root / p).resolve() or (resolved_root / p).resolve() in target.parents
            for p in cfg_paths
        )
        if not inside:
            return None
        rel = target.relative_to(resolved_root).as_posix() if resolved_root in target.parents else files[0]
        suggestion = self._longnames_for_file(resolved_root, rel)
        combined = " together with `by_longname`" if by_longname else ""
        return (
            f"`paths` names the single file '{files[0]}'{combined}: it becomes the top suite, so parent "
            "`__init__.robot` setups and variables are skipped and longnames lose their parent suite prefix. "
            "Drop `paths` and select with `by_longname`"
            + (f", for example by_longname={suggestion}." if suggestion else " taken from `rf_discover`.")
        )

    def _longnames_for_file(self, root: Path, rel: str) -> list[str]:
        args = self.build_global_args(root=root, json_format=True, mode="server") + ["discover", "tests"]
        outcome = run_process(
            resolve_robotcode_executable() + args[1:],
            cwd=str(root),
            env=build_agent_env(),
            timeout_seconds=60,
        )
        try:
            items = json.loads(outcome.stdout).get("items", [])
        except (json.JSONDecodeError, AttributeError):
            return []
        names = [i["longname"] for i in items if i.get("relSource") == rel]
        if len(names) > 3:
            suite = names[0].rsplit(".", 1)[0]
            return [suite]
        return names

    def execute(
        self,
        args: Sequence[str],
        *,
        root: str | Path | None = None,
        mode_override: str | None = None,
        shell: ShellKind | None = None,
        timeout_seconds: int = 120,
        stdin_lines: Sequence[str] | None = None,
        hint_keyword: str = "",
        hint_kind: str = "runtime",
    ) -> CommandResult:
        """Execute `args` in `server` mode or synthesize a shell-safe command in `client` mode."""
        mode = self.config.resolved_mode(mode_override)
        cmd_list = list(args)

        if mode == "client":
            effective_shell = shell or self.config.default_shell
            return CommandResult(
                mode="client",
                command=cmd_list,
                shell_command=format_cli_command(
                    cmd_list,
                    shell=effective_shell,
                    stdin_lines=stdin_lines,
                ),
                instruction=(
                    "Execute `shell_command` locally on the client machine using your terminal/shell tool. "
                    "If 'robotcode' is installed in a local uv virtualenv and not on global PATH, prefix with 'uv run ' "
                    "(e.g., 'uv run robotcode ...'). "
                    "If the command fails or returns an error, pass the error output to `rf_diagnose`."
                ),
            )

        resolved_root = self.config.resolve_root(root)
        exec_cmd = list(cmd_list)
        if exec_cmd and exec_cmd[0] == "robotcode":
            exec_cmd = resolve_robotcode_executable() + exec_cmd[1:]

        outcome = run_process(
            exec_cmd,
            cwd=str(resolved_root),
            env=build_agent_env(),
            timeout_seconds=timeout_seconds,
            input_text="\n".join(stdin_lines) + "\n" if stdin_lines is not None else None,
        )
        if outcome.status == "error":
            return CommandResult(
                mode="server",
                command=cmd_list,
                status="error",
                stderr=outcome.error or "",
                instruction=(
                    "`robotcode` was not found on the MCP server host. Install it in the project environment "
                    f"(`{ROBOTCODE_INSTALL_HINT}`) and start rf-mcp from that environment; run `rf-mcp doctor` to check."
                ),
            )
        stdout = outcome.stdout.lstrip("﻿")
        stderr = outcome.stderr.lstrip("﻿")
        if outcome.status == "timeout":
            return CommandResult(
                mode="server",
                command=cmd_list,
                status="timeout",
                stdout=truncate_text(stdout, self.config.max_output_chars)[0],
                stderr=stderr,
                instruction=(
                    f"`robotcode` did not finish within {timeout_seconds} s and was stopped with all its child "
                    "processes. Raise `timeout_seconds`, or narrow the selection (`by_longname`, `include_tags`)."
                ),
            )

        parsed_data: Any = None
        if "-f" in cmd_list and "json" in cmd_list and stdout.strip():
            try:
                parsed_data = json.loads(stdout)
            except json.JSONDecodeError:
                parsed_data = None
            if "results" in cmd_list and "log" in cmd_list:
                parsed_data = compact_result_log(parsed_data)

        hints: list[Hint] = []
        failures: list[str] | None = None
        combined_output = f"{stdout}\n{stderr}".strip()
        if hint_kind == "runtime":
            failures = parse_console_failures(stdout) or None
        failed = outcome.returncode != 0 or bool(failures) or "[ ERROR ]" in combined_output
        if hint_kind == "static":
            failed = bool(parsed_data and (parsed_data.get("summary") or {}).get("errors")) or (
                parsed_data is None and outcome.returncode != 0
            )
        if failed:
            hints = self.hint_engine.diagnose(
                HintContext(
                    keyword=hint_keyword or " ".join(cmd_list[:4]),
                    error_text="\n".join(failures) if failures else combined_output,
                ),
                kind="static" if hint_kind == "static" else "runtime",
            )

        result = CommandResult(
            mode="server",
            command=cmd_list,
            exit_code=outcome.returncode,
            stdout=stdout if parsed_data is None else "",
            stderr=stderr,
            data=parsed_data,
            failures=failures,
            hints=hints,
        )
        self._truncate(result)
        return result

    def _truncate(self, result: CommandResult) -> None:
        limit = self.config.max_output_chars
        notes = []
        if result.stdout:
            result.stdout, note = truncate_text(result.stdout, limit)
            notes.append(note)
        if result.data is not None:
            result.data, note = truncate_data(result.data, limit)
            notes.append(note)
        notes = [n for n in notes if n]
        if notes:
            result.truncated = " ".join(notes)
