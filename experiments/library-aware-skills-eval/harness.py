"""Evaluate library-aware skill retrieval (design D8 of openspec/changes/library-aware-skill-retrieval).

An OpenRouter chat model with tool calling works against an in-memory rf-mcp server until it stops calling tools.
The arms:

- A: rf-mcp's built-in skills only (pointers on, from the built-in catalogue);
- B: the official skills from a robotframework-plugins checkout through --skills-dir, with pointers;
- C: as B, with RF_MCP_SKILL_POINTERS=0 (no project_skills, no skill links on hints).

Grading, through the same server's tools: rf_analyze clean, rf_run passing with the required test names, and the
suite failing against a broken copy of the app (mutation).

Secrets: OPENROUTER_API_KEY and OPENROUTER_BASE_URL come from the environment or the repository's .env; they are
never printed or written to the results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from fastmcp.client import Client

from robotmcp.config import ServerConfig
from robotmcp.server import create_server

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
MAX_TOOL_CHARS = 16000


# --- secrets ----------------------------------------------------------------------------------------------------


def load_env(path: Path = REPO / ".env") -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip("'\"")
    return {**values, **{k: v for k, v in os.environ.items() if k.startswith("OPENROUTER_")}}


# --- fixtures ---------------------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except urllib.error.HTTPError:
            return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"{url} did not come up")


@contextmanager
def serve(cmd: list[str], url: str, cwd: Path) -> Iterator[str]:
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_http(url)
        yield url
    finally:
        proc.kill()
        proc.wait(timeout=10)


@dataclass
class Fixtures:
    web: str
    web_broken: str
    ledger: str
    ledger_broken: str


@contextmanager
def fixtures(python: str, workdir: Path) -> Iterator[Fixtures]:
    """The widget lab (and a copy whose payment page reports the wrong card), and the ledger API (and its vat bug)."""
    broken = workdir / "site-broken"
    if broken.exists():
        shutil.rmtree(broken)
    shutil.copytree(FIXTURES / "web" / "site", broken)
    # The card is 4242424242424242, so its first four digits equal its last four: report three instead.
    for name, old, new in (("pay.html", ".slice(-4)", ".slice(-3)"), ("coupon.html", "' applied'", "' rejected'")):
        page = broken / name
        text = page.read_text(encoding="utf-8")
        assert old in text, f"{name} changed; update the mutation"
        page.write_text(text.replace(old, new), encoding="utf-8")
    with ExitStack() as stack:
        def http(directory: Path, port: int) -> str:
            cmd = [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(directory)]
            return stack.enter_context(serve(cmd, f"http://127.0.0.1:{port}/", directory))

        try:
            http(FIXTURES / "web" / "xsite", 8702)  # index.html embeds the cross-origin frame at this port
        except RuntimeError:
            pass
        web = http(FIXTURES / "web" / "site", free_port())
        web_broken = http(broken, free_port())

        def ledger(bug: str) -> str:
            port = free_port()
            cmd = [python, str(FIXTURES / "ledger" / "ledger_api.py"), str(port), *([bug] if bug else [])]
            return stack.enter_context(serve(cmd, f"http://127.0.0.1:{port}/", FIXTURES / "ledger")).rstrip("/")

        yield Fixtures(web + "index.html", web_broken + "index.html", ledger(""), ledger("vat"))


# --- tasks ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    name: str
    suite: str
    variable: str
    tests: tuple[str, ...]
    prompt: str

    def url(self, fx: Fixtures, broken: bool = False) -> str:
        if self.name == "browser":
            return fx.web_broken if broken else fx.web
        return fx.ledger_broken if broken else fx.ledger


TASKS = {
    "browser": Task(
        "browser", "tests/widgets.robot", "URL", ("Pay With Card", "Apply Coupon"),
        "The web app under test is the widget lab page at {url}. Write the Robot Framework suite "
        "tests/widgets.robot using the Browser library (headless Chromium) with these tests:\n"
        "1. `Pay With Card`: in the Payment section, enter the card number 4242424242424242, press Pay, and verify "
        "the page says `Paid with card ending 4242`.\n"
        "2. `Apply Coupon`: in the Coupon section, enter the code SAVE10, press Apply, and verify the page says "
        "`Coupon SAVE10 applied`.\n"
        "Define `${{URL}}` in the suite's Variables section with the URL above, so it can be overridden with -v. "
        "Each test opens its own page. You are done when rf_analyze is clean and rf_run passes.",
    ),
    "requests": Task(
        "requests", "tests/invoices.robot", "BASE_URL",
        ("Create Invoice Computes Totals", "Reject Invoice Without Lines", "Mark Invoice Paid"),
        "The Ledger invoicing API runs at {url}. It has no OpenAPI; GET / lists the endpoints, and 422 responses "
        "describe invalid payloads. Get a bearer token with POST /token (form fields username=tester, "
        "password=Ledger-2026!). Write the Robot Framework suite tests/invoices.robot using RequestsLibrary with "
        "these tests:\n"
        "1. `Create Invoice Computes Totals`: create an invoice for the customer named `ACME GmbH` with the lines "
        "sku A (qty 3, unit_price 19.99) and sku B (qty 1, unit_price 5.00); verify status 201 and net 64.97, "
        "vat 12.34 and gross 77.31.\n"
        "2. `Reject Invoice Without Lines`: creating an invoice with an empty lines list returns 422.\n"
        "3. `Mark Invoice Paid`: create an invoice, PATCH it to paid, and verify GET /invoices/{{id}} shows paid "
        "true.\n"
        "Define `${{BASE_URL}}` in the suite's Variables section with the URL above, so it can be overridden with -v. "
        "Each test creates its own data. You are done when rf_analyze is clean and rf_run passes.",
    ),
}


def make_project(path: Path, python: str) -> Path:
    if path.exists():
        shutil.rmtree(path)
    (path / "tests").mkdir(parents=True)
    (path / ".robotframework").mkdir()
    # Without the output folder, Browser's New Browser in rf_repl fails on results/playwright-log.txt and the next
    # New Context hangs (rf-mcp bug found in the dry runs, outside this change); every arm would pay for it.
    (path / "results").mkdir()
    (path / "robot.toml").write_text('paths = ["tests"]\noutput-dir = "results"\n', encoding="utf-8")
    (path / ".robotframework" / "rf-mcp.toml").write_text(f"python = {json.dumps(python)}\n", encoding="utf-8")
    return path


# --- the agent loop ---------------------------------------------------------------------------------------------


@dataclass
class Run:
    model: str
    task: str
    arm: str
    rep: int
    passed: bool = False
    analyze_clean: bool = False
    run_passed: bool = False
    names_ok: bool = False
    bug_caught: bool = False
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    pointed: list[str] = field(default_factory=list)
    read: list[str] = field(default_factory=list)
    followed: int = 0
    tool_calls: int = 0
    seconds: float = 0.0
    stop: str = ""
    error: str = ""

    @property
    def strict_pass(self) -> bool:
        return self.passed and self.bug_caught


def chat(env: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    base = env.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    headers = {
        "Authorization": f"Bearer {env['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "X-Title": "rf-mcp library-aware skill eval",
    }
    ranking = env.get("OPENROUTER_RANKINGS", "")
    if ranking.startswith("http"):
        headers["HTTP-Referer"] = ranking
    data = json.dumps(body).encode("utf-8")
    last: Exception | None = None
    for attempt in range(5):
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            if "choices" in out:
                return out
            last = RuntimeError(str(out.get("error", out))[:300])
        except urllib.error.HTTPError as exc:
            last = RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}")
            if exc.code not in (408, 429, 500, 502, 503, 504):
                break
        except (OSError, ValueError) as exc:
            last = exc
        time.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"chat failed: {last}")


def tool_schemas(tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {
            "name": t.name,
            "description": (t.description or "")[:1000],
            "parameters": getattr(t, "input_schema", None) or {"type": "object"},
        }}
        for t in tools
    ]


# The agent's own file tools, as a coding agent has them; rf-mcp itself does not write project files.
LOCAL_TOOLS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file in the project, e.g. tests/login.robot.",
        "parameters": {"type": "object", "required": ["path", "content"], "properties": {
            "path": {"type": "string", "description": "Path relative to the project root."},
            "content": {"type": "string", "description": "The whole file content."}}},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file of the project.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "Path relative to the project root."}}},
    }},
]


def local_tool(project: Path, name: str, args: dict[str, Any]) -> str:
    target = (project / str(args.get("path", ""))).resolve()
    if project.resolve() not in target.parents:
        return f"error: {args.get('path')!r} is outside the project"
    if name == "write_file":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(args.get("content", "")), encoding="utf-8")
        return f"wrote {target.relative_to(project.resolve()).as_posix()} ({len(str(args.get('content', '')))} chars)"
    if not target.is_file():
        return f"error: {args.get('path')!r} does not exist"
    return target.read_text(encoding="utf-8", errors="replace")


def _pointers(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except ValueError:
        return []
    if not isinstance(data, dict):
        return []
    out = [e["skill"] for e in data.get("project_skills", []) if isinstance(e, dict) and "skill" in e]
    out += [h["skill"] for h in data.get("hints", []) if isinstance(h, dict) and "skill" in h]
    return out


async def call(client: Client, name: str, args: dict[str, Any]) -> tuple[str, Any]:
    res = await client.call_tool(name, args, raise_on_error=False)
    text = "\n".join(getattr(c, "text", "") for c in res.content)
    return text, res


async def agent_loop(env: dict[str, str], client: Client, instructions: str, prompt: str, run: Run,
                     max_turns: int, transcript: list[dict[str, Any]], project: Path) -> None:
    tools = tool_schemas(await client.list_tools()) + LOCAL_TOOLS
    messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}, {"role": "user", "content": prompt}]
    pointed: set[str] = set()
    for _ in range(max_turns):
        run.turns += 1
        out = await asyncio.to_thread(chat, env, {
            "model": run.model, "messages": messages, "tools": tools, "tool_choice": "auto",
            "usage": {"include": True}, "max_tokens": 8000,
        })
        usage = out.get("usage") or {}
        run.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        run.completion_tokens += int(usage.get("completion_tokens") or 0)
        run.cost += float(usage.get("cost") or 0.0)
        msg = out["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")} | {"role": "assistant"})
        transcript.append({"assistant": msg.get("content"), "tool_calls": [c["function"] for c in calls]})
        if not calls:
            run.stop = "done"
            return
        for c in calls:
            run.tool_calls += 1
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            if name == "read_resource" and str(args.get("uri", "")).startswith("skill://"):
                uri = str(args["uri"])
                run.read.append(uri)
                if uri in pointed or uri.replace("/SKILL.md", "") in {p.replace("/SKILL.md", "") for p in pointed}:
                    run.followed += 1
            if not isinstance(args, dict):
                args = {}
            try:
                if name in ("write_file", "read_file"):
                    text = local_tool(project, name, args)
                else:
                    text, _res = await asyncio.wait_for(call(client, name, args), 900)
            except Exception as exc:  # a tool that hangs or the client failing
                text = f"error: {exc}"
            for uri in _pointers(text):
                pointed.add(uri)
                run.pointed.append(uri)
            transcript.append({"tool": name, "result": text[:2000]})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": text[:MAX_TOOL_CHARS]})
    run.stop = "max_turns"


# --- grading ----------------------------------------------------------------------------------------------------


async def grade(client: Client, task: Task, fx: Fixtures, project: Path, run: Run) -> None:
    suite = project / task.suite
    if not suite.is_file():
        run.error = run.error or f"{task.suite} was not written"
        return
    text, res = await call(client, "rf_analyze", {"paths": [task.suite]})
    data = res.structured_content or {}
    errors = ((data.get("data") or {}).get("summary") or {}).get("errors")
    run.analyze_clean = not res.is_error and not errors
    good = f"{task.variable}:{task.url(fx)}"
    text, res = await call(client, "rf_run", {"paths": [task.suite], "variables": [good]})
    data = res.structured_content or {}
    run.run_passed = not res.is_error and data.get("exit_code") == 0
    suite_text = suite.read_text(encoding="utf-8", errors="replace")
    run.names_ok = all(name.lower() in suite_text.lower() for name in task.tests)
    run.passed = run.analyze_clean and run.run_passed and run.names_ok
    bad = f"{task.variable}:{task.url(fx, broken=True)}"
    text, res = await call(client, "rf_run", {"paths": [task.suite], "variables": [bad]})
    data = res.structured_content or {}
    run.bug_caught = run.passed and data.get("exit_code") not in (0, None)


# --- runs -------------------------------------------------------------------------------------------------------


async def one_run(env: dict[str, str], args: argparse.Namespace, fx: Fixtures, model: str, task: Task, arm: str,
                  rep: int) -> Run:
    run = Run(model, task.name, arm, rep)
    slug = f"{model.replace('/', '_')}-{task.name}-{arm}-{rep}"
    project = make_project(Path(args.workdir) / "runs" / slug, args.python)
    official = [Path(args.official)] if arm in ("B", "C") else []
    cfg = ServerConfig(execution_mode="server", workspace_root=project, skills_dirs=official,
                       skill_pointers=arm != "C")
    server = create_server(cfg)
    transcript: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        async with Client(server, mode="legacy") as client:
            try:
                await asyncio.wait_for(
                    agent_loop(env, client, server.instructions or "", task.prompt.format(url=task.url(fx)), run,
                               args.max_turns, transcript, project),
                    args.run_timeout,
                )
            except asyncio.TimeoutError:
                run.stop = "timeout"
            except Exception as exc:
                run.stop, run.error = "error", str(exc)[:500]
            await grade(client, task, fx, project, run)
            await call(client, "rf_repl", {"action": "stop", "stop_scope": "all"})
    except Exception as exc:
        run.error = run.error or str(exc)[:500]
    run.seconds = round(time.monotonic() - started, 1)
    (project / "transcript.json").write_text(json.dumps(transcript, indent=1)[:2_000_000], encoding="utf-8")
    return run


async def main_async(args: argparse.Namespace) -> None:
    env = load_env()
    if not env.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set (environment or .env)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["model"], r["task"], r["arm"], r["rep"]))
    jobs = [(m, TASKS[t], a, r) for r in range(args.reps) for m in args.models for t in args.tasks for a in args.arms
            if (m, t, a, r) not in done]
    random.Random(7).shuffle(jobs)
    print(f"{len(jobs)} runs to do ({len(done)} already in {out})", flush=True)
    sem = asyncio.Semaphore(args.jobs)
    with fixtures(args.python, Path(args.workdir)) as fx:
        async def guarded(job):
            async with sem:
                run = await one_run(env, args, fx, *job)
                row = asdict(run) | {"strict_pass": run.strict_pass}
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{run.model} {run.task} {run.arm} r{run.rep}: pass={run.passed} bug={run.bug_caught} "
                      f"turns={run.turns} pointed={len(run.pointed)} read={len(run.read)} {run.stop} {run.error[:80]}",
                      flush=True)

        await asyncio.gather(*(guarded(j) for j in jobs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--python", required=True, help="interpreter with robotframework-browser, -requests and the fixtures' deps")
    p.add_argument("--official", required=True, help="checkout of MarketSquare/robotframework-plugins")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--arms", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--run-timeout", type=float, default=1500)
    p.add_argument("--workdir", required=True, help="scratch folder for projects and fixtures (outside the repo)")
    p.add_argument("--out", required=True, help="results JSONL; existing rows are kept and skipped")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
