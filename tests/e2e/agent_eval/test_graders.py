"""Offline tests of the agent-eval graders, report and dry-run seeding (no network, claude, browsers or fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import graders as G
from .__main__ import main
from .report import render_markdown, write_report
from .runner import make_jobs, parse_conditions
from .scenarios import SUITES

RECORDED = Path(__file__).resolve().parent / "recorded"
ROOT = "C:/lab/run"


def tx(*calls, final: str = "Done.", cwd: str = ROOT) -> G.Transcript:
    """Synthetic stream-json transcript: calls are (tool name, input[, result text[, is_error]])."""
    events = [{"type": "system", "subtype": "init", "cwd": cwd}]
    for i, call in enumerate(calls):
        name, inp, result, err = (*call, "", False)[:4] if len(call) == 2 else (*call, False)[:4]
        name = name if name[0].isupper() else f"mcp__rf-mcp__{name}"
        events.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}]}})
        events.append({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "is_error": err, "content": [{"type": "text", "text": result}]}]}})
    events.append({"type": "assistant", "message": {"content": [{"type": "text", "text": final}]}})
    events.append({"type": "result", "subtype": "success", "num_turns": len(calls) + 1, "total_cost_usd": 0.01, "result": final,
                   "usage": {"input_tokens": 10, "output_tokens": 5}})
    return G.load_transcript([json.dumps(e) for e in events])


# ------------------------------------------------------------------------------------- (a) output.xml


@pytest.mark.parametrize("call", [
    ("Read", {"file_path": f"{ROOT}/results/output.xml"}),
    ("Grep", {"pattern": "FAIL", "path": "results\\output.xml", "output_mode": "content"}),
    ("Grep", {"pattern": "Ten Percent", "output_mode": "content"}, 'results\\output.xml:152:<test name="Ten Percent Discount">'),
    ("Glob", {"pattern": "results/output.xml"}),
    ("Bash", {"command": 'cd "C:/lab/run" && sed -n 140,200p results/output.xml'}),
    ("PowerShell", {"command": "Get-Content results\\output.xml | Select-String FAIL"}),
    ("Bash", {"command": "python -c \"import xml.etree.ElementTree as E; E.parse('results/output.xml')\""}),
    ("rf_repl", {"action": "send", "session_id": "repl-1", "statement": "${x}=    Parse XML    results/output.xml"}),
    ("rf_debug", {"action": "send", "session_id": "debug-1", "command": "Get File    results/output.xml"}),
])
def test_output_xml_reads_flagged(call):
    t = tx(call)
    assert G.output_xml_reads(t)
    assert "output_xml" in G.process_grades(t)["flags"]


@pytest.mark.parametrize("call", [
    ("rf_results", {"action": "show", "output_path": "results/output.xml"}),
    ("Bash", {"command": "robotcode -f json results show -o results/output.xml | head -50"}),
    ("Bash", {"command": "uv run robotcode robot --output output.xml -bl Tests.Calc"}),
    ("Bash", {"command": "ls results/output.xml"}),
    ("Read", {"file_path": f"{ROOT}/data/books.xml"}),
    ("Grep", {"pattern": "Ten Percent", "output_mode": "files_with_matches"}, "results\\output.xml\ntests\\calc\\calc.robot"),
    ("rf_repl", {"action": "send", "session_id": "repl-1", "statement": "${x}=    Parse XML    data/books.xml"}),
])
def test_output_xml_reads_not_flagged(call):
    assert G.output_xml_reads(tx(call)) == []


# ------------------------------------------------------------------------------------- (b) file selection


@pytest.mark.parametrize("call", [
    ("rf_run", {"paths": ["tests/calc/calc.robot"]}),
    ("rf_run", {"paths": "tests/calc/calc.robot", "by_longname": ["Tests.Calc.Calc.Add Two Numbers"]}),
    ("rf_debug", {"action": "start", "paths": ["tests/calc/calc.robot"]}),
    ("Bash", {"command": "uv run robotcode -r . robot tests/calc/calc.robot"}),
    ("Bash", {"command": "robotcode robot-debug tests\\calc\\calc.robot"}),
    ("PowerShell", {"command": "python -m robot --test 'Ten Percent Discount' tests\\calc\\calc.robot"}),
])
def test_single_file_selection_flagged(call):
    assert G.single_file_selections(tx(call))


@pytest.mark.parametrize("call", [
    ("rf_run", {"by_longname": ["Tests.Calc.Calc.Ten Percent Discount"]}),
    ("rf_run", {"paths": ["tests"]}),
    ("rf_run", {"paths": ["tests/a.robot", "tests/b.robot"]}),
    ("rf_debug", {"action": "send", "session_id": "debug-1", "command": ".c"}),
    ("Bash", {"command": 'robotcode robot -bl "Tests.Calc.Calc.Ten Percent Discount"'}),
    ("Bash", {"command": "robotcode analyze code tests/calc/calc.robot"}),
])
def test_single_file_selection_not_flagged(call):
    assert G.single_file_selections(tx(call)) == []


# ------------------------------------------------------------------------------------- (c) run before analyze

WRITE_BOOKS = ("Write", {"file_path": f"{ROOT}/tests/xml/books.robot", "content": "*** Test Cases ***"})


@pytest.mark.parametrize("calls", [
    [WRITE_BOOKS, ("rf_run", {"by_longname": ["Tests.Xml.Books"]})],
    [WRITE_BOOKS, ("rf_analyze", {"paths": ["tests/api"]}), ("rf_run", {"include_tags": ["xml"]})],
    [WRITE_BOOKS, ("rf_analyze", {}), ("Edit", {"file_path": f"{ROOT}\\tests\\xml\\books.robot"}), ("rf_run", {})],
    [("Write", {"file_path": f"{ROOT}/resources/shop.resource"}), ("rf_run", {"by_longname": ["Tests.Ui.Cart"]})],
    [("rf_repl", {"action": "save", "save_path": "tests/xml/books.robot"}), ("rf_debug", {"by_longname": ["Tests.Xml"]})],
    [WRITE_BOOKS, ("Bash", {"command": "robotcode robot -bl Tests.Xml.Books"})],
])
def test_run_before_analyze_flagged(calls):
    assert G.runs_before_analyze(tx(*calls))


@pytest.mark.parametrize("calls", [
    [WRITE_BOOKS, ("rf_analyze", {}), ("rf_run", {"by_longname": ["Tests.Xml.Books"]})],
    [WRITE_BOOKS, ("rf_analyze", {"paths": ["tests/xml/books.robot"]}), ("rf_run", {"paths": ["tests"]})],
    [WRITE_BOOKS, ("rf_run", {"by_longname": ["Tests.Calc.Calc.Add Two Numbers"]})],
    [WRITE_BOOKS, ("Bash", {"command": "robotcode analyze code tests && robotcode robot -bl Tests.Xml.Books"})],
    [("Write", {"file_path": f"{ROOT}/data/books.xml"}), ("rf_run", {})],
])
def test_run_before_analyze_not_flagged(calls):
    assert G.runs_before_analyze(tx(*calls)) == []


def test_last_analyze_has_errors():
    assert G.last_analyze_has_errors(tx(("rf_analyze", {}, '{"has_errors": true}'))) is True
    assert G.last_analyze_has_errors(tx(("rf_analyze", {}, '{"has_errors":true}'), ("rf_analyze", {}, '{"has_errors": false}'))) is False
    assert G.last_analyze_has_errors(tx(("rf_run", {}))) is None


def test_unverified_finish():
    write = ("Write", {"file_path": "tests/a.robot"})
    passed, failed = ("rf_run", {}, '{"outcome":"PASS","exit_code":0}'), ("rf_run", {}, '{"exit_code":1}')
    clean = ("rf_analyze", {"files": {"tests/a.robot": "x"}}, '{"has_errors":false}')
    assert G.unverified_finish(tx(write, clean))
    assert G.unverified_finish(tx(passed, write, failed))
    assert G.unverified_finish(tx(write, ("rf_run", {"dryrun": True}, '{"exit_code":0}')))
    assert G.unverified_finish(tx(("PowerShell", {"command": "Set-Content tests/a.robot -Value $x"})))
    assert not G.unverified_finish(tx(write, failed, passed))
    assert not G.unverified_finish(tx(clean))  # analysis with synced sources is no authoring
    assert not G.unverified_finish(tx(("Bash", {"command": "robot tests/a.robot"})))
    assert "unverified" in G.process_grades(tx(write, clean))["flags"]


# ------------------------------------------------------------------------------------- (d) open sessions

START = ("rf_repl", {"action": "start"}, '{"session_id":"repl-abc12345","status":"passed"}')


def test_open_session_flagged():
    t = tx(START, ("rf_repl", {"action": "send", "session_id": "repl-abc12345", "statement": "Log  1"}), final="All done.")
    assert G.unannounced_open_sessions(t) == ["repl-abc12345"]
    assert "open_session" in G.process_grades(t)["flags"]


@pytest.mark.parametrize("calls,final", [
    ([START, ("rf_repl", {"action": "stop", "session_id": "repl-abc12345"})], "Done."),
    ([START, ("rf_debug", {"action": "stop", "stop_scope": "project"})], "Done."),
    ([START], "The REPL session is still open for further checks."),
    ([START], "I left repl-abc12345 running."),
    ([("Bash", {"command": "robotcode session start repl"}, "started repl-0badcafe"),
      ("Bash", {"command": "robotcode session stop --project"})], "Done."),
    ([("rf_debug", {"action": "start", "by_longname": ["Tests.X"]}, "error", True)], "Done."),
])
def test_open_session_not_flagged(calls, final):
    assert G.unannounced_open_sessions(tx(*calls, final=final)) == []


def test_shell_session_left_open():
    t = tx(("Bash", {"command": "robotcode session start repl 2>&1 | tail -3"}, '{"session_id": "repl-63d86e74"}'))
    assert G.unannounced_open_sessions(t) == ["repl-63d86e74"]


# ------------------------------------------------------------------------------------- (e) results first


@pytest.mark.parametrize("calls", [
    [("Read", {"file_path": f"{ROOT}/tests/calc/calc.robot"}), ("rf_results", {"action": "show"})],
    [("Grep", {"pattern": "Ten Percent"})],
    [("Bash", {"command": "cat tests/calc/calc.robot"}), ("Bash", {"command": "robotcode results log"})],
])
def test_results_late_flagged(calls):
    t = tx(*calls)
    assert G.results_after_project_reads(t)["flagged"]
    assert "results_late" in G.process_grades(t, why_failed=True)["flags"]
    assert "results_late" not in G.process_grades(t)["flags"]  # only graded for "why did X fail?" tasks


@pytest.mark.parametrize("calls", [
    [("rf_results", {"action": "show", "failed_only": True}), ("Read", {"file_path": f"{ROOT}/libraries/CalcLib.py"})],
    [("Bash", {"command": "uv run robotcode results log -t 'Ten Percent Discount' | head -50"}), ("Read", {"file_path": "x.py"})],
    [("Read", {"file_path": f"{ROOT}/.claude/skills/robotcode/references/results.md"}), ("rf_results", {})],
    [("rf_discover", {"action": "tests"})],
])
def test_results_late_not_flagged(calls):
    assert not G.results_after_project_reads(tx(*calls))["flagged"]


# ------------------------------------------------------------------------------------- pitfall counters


def test_pitfall_counts():
    t = tx(
        ("rf_repl", {"action": "send", "statement": "${r}=    POST On Session    s    /token    json=${body}"},
         '{"status": "failed", "message": "No keyword with name \'POST On Session\' found."}'),
        ("rf_repl", {"action": "send", "statement": "GET    ${U}/x    expected_status=any\nLog    1"}, '{"status": "passed"}'),
        ("rf_repl", {"action": "send", "statement": "POST    ${U}    json={\"a\": 1}    expected_status=any"},
         '{"status": "failed", "message": "\'str\' object has no attribute \'items\'"}'),
        ("rf_run", {"by_longname": ["T"]}, "| FAIL | waiting for locator 'css=#x'"),
        ("read_resource", {"uri": "skill://rf-requests/SKILL.md"}),
    )
    p = G.pitfall_counts(t)
    assert (p["sends"], p["multi_sends"], p["failed_sends"]) == (3, 1, 2)
    assert p["pitfalls"] == {"not_imported": 1, "json_or_headers_as_text": 1, "frame_or_shadow_not_found": 1}
    assert (p["expected_status_any"], p["token_as_json"], p["json_text_in_repl"]) == (2, 1, 1)
    assert p["skill_reads"] == ["skill://rf-requests/SKILL.md"]


def test_static_counts():
    src = """*** Test Cases ***
A
    [Tags]    blocked
    POST On Session    s    /x    json={"a": 1}    expected_status=any
    Sleep    1s
    Click    css=#b    force=True
    Set Suite Variable    ${ID}    1
"""
    s = G.static_counts(src)
    assert (s["json_text_body"], s["sleep"], s["force"], s["blocked_tag"], s["expected_status_any"],
            s["suite_variable_chaining"]) == (1, 1, 1, 1, 1, 1)
    assert G.static_counts("*** Test Cases ***\nA\n    Log    1\n")["sleep"] == 0


def test_run_metrics():
    m = G.run_metrics(tx(("rf_run", {}), ("Read", {"file_path": "x"}, "boom", True)))
    assert (m["num_turns"], m["cost"], m["in_tokens"], m["out_tokens"], m["tool_errors"]) == (3, 0.01, 10, 5, 1)
    assert m["tools"] == {"mcp__rf-mcp__rf_run": 1, "Read": 1}


# ------------------------------------------------------------------------------------- recorded 2026-11-05/06 runs


def recorded(name: str) -> G.Transcript:
    return G.load_transcript(RECORDED / f"{name}.jsonl")


@pytest.mark.parametrize("name", ["s1_triage_fix__haiku__mcp", "s1_triage_fix__sonnet__mcp"])
def test_recorded_s1_mcp_runs_read_output_xml(name):
    t = recorded(name)
    reads = G.output_xml_reads(t)
    assert reads and all("output.xml" in r for r in reads)
    p = G.process_grades(t, why_failed=True)
    assert {"output_xml", "file_selection", "results_late"} <= set(p["flags"])


def test_recorded_clean_cli_run_is_not_flagged():
    t = recorded("s1_triage_fix__sonnet__cli")
    assert G.output_xml_reads(t) == []
    assert G.process_grades(t, why_failed=True)["flags"] == []
    assert G.run_metrics(t)["num_turns"] == 9


def test_recorded_authoring_runs_before_analyze():
    p = G.process_grades(recorded("s4_xml_author__sonnet__mcp"))
    assert p["runs_before_analyze"] and p["single_file_selections"]
    assert p["output_xml_reads"] == [] and p["unannounced_open_sessions"] == []


def test_recorded_debug_session_left_open():
    assert G.unannounced_open_sessions(recorded("s7_debug_inspect__haiku__mcp")) == ["debug-ae3a7cd0"]


def test_recorded_pitfall_counts_match_the_experiment():
    fx = G.pitfall_counts(recorded("fx_browser__haiku__g1__r1"))
    assert (fx["sends"], fx["failed_sends"]) == (8, 3)
    assert fx["pitfalls"] == {"not_imported": 1, "frame_or_shadow_not_found": 3}
    api = G.pitfall_counts(recorded("api_booker__haiku__g1__r1"))
    assert (api["sends"], api["failed_sends"], api["expected_status_any"]) == (4, 2, 14)


def test_trim_keeps_grades(tmp_path):
    src = RECORDED / "s1_triage_fix__haiku__mcp.jsonl"
    twice = G.trim_transcript(src)
    assert G.process_grades(G.load_transcript(twice), True) == G.process_grades(recorded("s1_triage_fix__haiku__mcp"), True)


# ------------------------------------------------------------------------------------- report and CLI


def _result(scen, model, cond, ok, turns, cost, flags=(), **grade):
    return {"run_id": f"{scen}__{model}__{cond}", "suite": "baseline", "scenario": scen, "model": model, "condition": cond,
            "num_turns": turns, "cost": cost, "wall_s": 10, "grade": {"pass": ok, **grade},
            "process": {"flags": list(flags), "failed_sends": 1}}


def test_report_renders_table(tmp_path):
    results = [
        _result("s1_triage_fix", "haiku", "mcp", True, 12, 0.05, ["output_xml", "results_late"]),
        _result("s1_triage_fix", "haiku", "mcp", False, 14, 0.07),
        _result("s1_triage_fix", "haiku", "cli", True, 7, 0.04),
        _result("api_soap", "haiku", "mcp", True, 30, 0.5, passed=5, total=5, isolated_pass=3, bugs_caught={"a": True, "b": False},
                bugs_caught_n=1),
    ]
    md = render_markdown(results, "Sample")
    assert md.startswith("# Sample")
    assert "| suite | scenario | model | cond | pass |" in md
    assert "| baseline | s1_triage_fix | haiku | mcp | 1/2 | - | 13.0 | 0.12 | 1.0 | - | - | 1 | 0 | 0 | 0 | 1 |" in md
    assert "| 5/5 |" in md and "| 1/2 | 3/5 |" in md
    assert "haiku: mcp/cli turns" in md
    for r in results:
        (tmp_path / f"{r['run_id']}.result.json").write_text(json.dumps(r), encoding="utf-8")
    assert "s1_triage_fix" in write_report(tmp_path).read_text(encoding="utf-8")


def test_suites_and_jobs():
    assert {k: len(v.scenarios) for k, v in SUITES.items()} == {"baseline": 7, "demoshop": 4, "fixtures": 3, "api": 4}
    conds = parse_conditions("mcp,cli", {})
    assert len(make_jobs(SUITES["baseline"], list(SUITES["baseline"].scenarios), ["haiku", "sonnet"], conds, 1)) == 28
    assert len(make_jobs(SUITES["demoshop"], list(SUITES["demoshop"].scenarios), ["haiku", "sonnet"], conds, 1)) == 16
    with pytest.raises(ValueError):
        parse_conditions("mcp:does-not-exist", {})


def test_dry_run_seeds_baseline(tmp_path, monkeypatch, capsys):
    for var in ("RF_EVAL_PYTHON", "RF_EVAL_BASELINE_PYTHON", "RF_EVAL_RUNS_DIR", "RF_EVAL_CLI_SKILL_DIR"):
        monkeypatch.delenv(var, raising=False)
    skills = tmp_path / "skills_v2"
    (skills / "rf-robotcode").mkdir(parents=True)
    rc = main(["baseline", "--scenarios", "s3_analyze", "--models", "haiku", "--conditions", "mcp,cli,mcp:v2",
               "--skills", f"v2={skills}", "--out", str(tmp_path / "out"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0, out
    projects = tmp_path / "out" / "projects"
    assert (projects / "s3_analyze__haiku__mcp__r1" / "tests" / "calc" / "broken.robot").is_file()
    assert (projects / "s3_analyze__haiku__cli__r1" / "robot.toml").is_file()
    assert "--disallowedTools Bash PowerShell" in out and "--mcp-config" in out
    assert "upstream robotcode skill NOT copied" in out
    cfg = json.loads((tmp_path / "out" / "s3_analyze__haiku__mcp-v2__r1.mcp.json").read_text(encoding="utf-8"))
    args = cfg["mcpServers"]["rf-mcp"]["args"]
    assert args[args.index("--skills-dir") + 1] == str(skills.resolve())
    assert not (tmp_path / "out" / "s3_analyze__haiku__cli__r1.mcp.json").exists()


# ------------------------------------------------------------------------------------- baseline s2


@pytest.mark.parametrize(("run_input", "passes"), [
    ({"include_tags": ["smoke"]}, True),
    ({"by_longname": sorted(G.SMOKE_LONGNAMES)}, True),
    ({"by_longname": ["Tests.Calc.Calc.Add Two Numbers"]}, False),
    ({}, False),
])
def test_s2_smoke_accepts_tag_or_exact_longnames(run_input, passes, tmp_path):
    final = "3 tests tagged smoke: Ping Service, Add Two Numbers, Format Money In Euro - all passed."
    t = tx(("rf_discover", {"action": "tests", "include_tags": ["smoke"]}), ("rf_run", run_input), final=final, cwd=str(tmp_path))
    scenario = SUITES["baseline"].scenarios["s2_smoke"]
    assert G.grade_baseline(scenario, tmp_path, t, None, G.tree_hash(tmp_path))["pass"] is passes


# ------------------------------------------------------------------------------------- env guard


def _run_shim(shim_dir: Path, shell: str, *args: str):
    import shutil
    import subprocess

    if shell == "cmd":
        if not shutil.which("cmd"):
            pytest.skip("no cmd.exe")
        cmd = ["cmd", "/c", str(shim_dir / f"{args[0]}.cmd"), *args[1:]]
    else:
        sh = shutil.which("sh") or shutil.which("bash")
        if not sh:
            pytest.skip("no POSIX shell")
        cmd = [sh, (shim_dir / args[0]).as_posix(), *args[1:]]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("shell", ["sh", "cmd"])
def test_guard_shims_block_env_changes_and_forward_the_rest(shell, tmp_path):
    import sys

    from .env import write_guard_shims

    # the "real" uv/pip is python itself, so a forwarded call prints the Python version
    shims = write_guard_shims(tmp_path / "guard", sys.executable, sys.executable)
    for args in (("uv", "sync"), ("uv", "pip", "install", "x"), ("uv", "add", "x"), ("pip", "uninstall", "rf-mcp")):
        p = _run_shim(shims, shell, *args)
        assert p.returncode == 2 and "`uv run --no-sync ...`" in p.stderr, (args, p.stdout, p.stderr)
    p = _run_shim(shims, shell, "uv", "--version")
    assert p.returncode == 0 and "Python" in p.stdout + p.stderr


# --------------------------------------------------------------------------------------- codex exec transcripts

CODEX_EVENTS = [
    {"type": "thread.started", "thread_id": "t"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": "Looking."}},
    {"type": "item.completed", "item": {"id": "i1", "type": "command_execution",
                                        "command": '"C:\\Program Files\\PowerShell\\pwsh.exe" -Command "Get-Content results\\output.xml"',
                                        "aggregated_output": "<robot/>", "exit_code": 0, "status": "completed"}},
    {"type": "item.completed", "item": {"id": "i2", "type": "command_execution", "command": "\"C:\\WINDOWS\\system32\\cmd.exe\" /c 'robot tests\\a.robot'",
                                        "aggregated_output": "boom", "exit_code": 1, "status": "failed"}},
    {"type": "item.completed", "item": {"id": "i3", "type": "mcp_tool_call", "server": "rf-mcp", "tool": "rf_run",
                                        "arguments": {"by_longname": ["Tests.A"]}, "status": "completed",
                                        "result": {"content": [{"type": "text", "text": "{\"exit_code\":0}"}]}, "error": None}},
    {"type": "item.completed", "item": {"id": "i4", "type": "mcp_tool_call", "server": "rf-mcp", "tool": "rf_repl",
                                        "arguments": {"action": "start"}, "status": "failed", "result": None,
                                        "error": {"message": "MCP tool call requires approval"}}},
    {"type": "item.completed", "item": {"id": "i5", "type": "file_change", "status": "completed",
                                        "changes": [{"path": r"C:\p\tests\b.robot", "kind": "add"}, {"path": "C:\\p\\x.py", "kind": "update"}]}},
    {"type": "item.completed", "item": {"id": "i6", "type": "agent_message", "text": "Done."}},
    {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 50}},
]


def test_codex_transcript_maps_to_claude_style_calls() -> None:
    t = G.load_transcript([json.dumps(e) for e in CODEX_EVENTS])
    assert [c.full_name for c in t.calls] == ["PowerShell", "Bash", "mcp__rf-mcp__rf_run", "mcp__rf-mcp__rf_repl", "Write", "Edit"]
    assert t.calls[0].command == r"Get-Content results\output.xml"
    assert t.calls[1].command == r"robot tests\a.robot" and t.calls[1].is_error
    assert t.calls[2].name == "rf_run" and not t.calls[2].is_error
    assert t.calls[3].is_error and "approval" in t.calls[3].result
    assert t.final == "Done."
    m = G.run_metrics(t)
    assert (m["num_turns"], m["in_tokens"], m["out_tokens"], m["cost"]) == (7, 1000, 50, None)
    assert G.output_xml_reads(t)


def test_codex_model_names_and_config() -> None:
    from .env import codex_model, resolve_model
    from .runner import codex_mcp_toml

    assert codex_model("haiku") is None and resolve_model("cx-gpt-5.6-luna@32k") == "gpt-5.6-luna"
    cx = codex_model("cx-oss-qwen3_14b@16k~medium")
    assert (cx.model, cx.provider, cx.context, cx.effort) == ("qwen3:14b", "ollama", 16000, "medium")
    assert "model_auto_compact_token_limit=12800" in cx.config() and 'model_provider="ollama"' in cx.config()
    assert codex_model("cx-oss-qwen3_8b").context and 'model_provider="ollama"' not in codex_model("cx-gpt-5.6-luna").config()
    toml = codex_mcp_toml({"mcpServers": {"rf-mcp": {"command": r"C:\py.exe", "args": ["-m", "robotmcp"], "env": {"A": "1"}}}})
    import tomllib

    server = tomllib.loads(toml)["mcp_servers"]["rf-mcp"]
    assert server["command"] == r"C:\py.exe" and server["default_tools_approval_mode"] == "approve" and server["env"] == {"A": "1"}
    assert [c.label for c in parse_conditions("mcp+sync", {})] == ["mcp-sync"]


def test_openrouter_model_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from . import env as E

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    E.openrouter_settings.cache_clear()
    try:
        assert E.resolve_model("or-z-ai.glm-5.3-flash") == "z-ai/glm-5.3-flash" and E.openrouter_model("haiku") is None
        claude = E.openrouter_claude_env("z-ai/glm-5.3-flash")
        assert claude["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api" and claude["ANTHROPIC_API_KEY"] == ""
        assert claude["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "z-ai/glm-5.3-flash"
        cx = E.codex_model("cx-or-openai.gpt-oss-20b@32k")
        assert (cx.model, cx.provider, cx.context) == ("openai/gpt-oss-20b", "openrouter", 32000)
        cfg = " ".join(cx.config())
        assert 'model_provider="openrouter"' in cfg and 'wire_api="responses"' in cfg and "k" not in cx.config()
        assert "OPENROUTER_API_KEY" not in E.ProjectEnv().env()
    finally:
        E.openrouter_settings.cache_clear()


def test_t3_defect_report_by_test_id_counts() -> None:
    by_id = ("Two remaining failures are shop defects:\n- `WEB-002_AC-1`: API says Cascade Water Bottle is `$79.00`; "
             "UI renders `$90.85`.\n- `WEB-006_AC-1`: subtotal + tax should be `$267.49`.")
    assert G._reports_t3_defects(by_id)
    assert G._reports_t3_defects("The card price is wrong and the tax is missing.")
    assert not G._reports_t3_defects("WEB-002_AC-1 and WEB-006_AC-1 still fail; I could not tell why.")


def test_codex_oss_catalog_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from .env import codex_model, codex_oss_catalog

    cached = {"slug": "gpt-x", "tool_mode": "code_mode_only", "apply_patch_tool_type": "freeform", "context_window": 272000}
    (tmp_path / "models_cache.json").write_text(json.dumps({"models": [cached]}), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    path = codex_oss_catalog(codex_model("cx-oss-qwen3_14b@16k"), tmp_path / "p" / ".codex" / "models.json")
    entry = json.loads(path.read_text(encoding="utf-8"))["models"][0]
    assert (entry["slug"], entry["tool_mode"], entry["apply_patch_tool_type"], entry["context_window"]) == (
        "qwen3:14b", "direct", None, 16000)
    assert entry["include_skills_usage_instructions"] is True
    path = codex_oss_catalog(codex_model("cx-or-z-ai.glm-5.3-flash"), tmp_path / "q.json")
    assert json.loads(path.read_text(encoding="utf-8"))["models"][0]["apply_patch_tool_type"] == "freeform"
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing"))
    assert codex_oss_catalog(codex_model("cx-oss-qwen3_14b"), tmp_path / "x.json") is None
