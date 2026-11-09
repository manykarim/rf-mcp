"""Run scenarios with `claude -p` in isolated project copies and grade them.

Conditions:
- `mcp`: rf-mcp tools (+ Read/Write/Edit/Glob/Grep), Bash/PowerShell disallowed, built-in skills over MCP.
- `mcp:<label>`: like `mcp`, with the server started with `--skills-dir <dir>` (`--skills label=dir`, or a path),
  so only the skill text differs between conditions (skill A/B).
- `cli`: Bash/PowerShell + the upstream robotcode agent skill copied to `.claude/skills/robotcode`
  (`--cli-skill` or `RF_EVAL_CLI_SKILL_DIR`), no MCP servers.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import graders as G
from .env import (ProjectEnv, claude_executable, codex_executable, codex_model, codex_oss_catalog, openrouter_claude_env,
                  openrouter_model, openrouter_settings, project_python, resolve_model, uv_executable, without_store_apps,
                  write_guard_shims)
from .fixtures import Demoshop, FixtureError, LedgerFixture, WebFixture
from .scenarios import Scenario, SeedError, SuiteSpec, ledger_bugs

MCP_TOOLS = ["mcp__rf-mcp", "Read", "Write", "Edit", "Glob", "Grep"]
CLI_TOOLS = ["Bash", "PowerShell", "Read", "Write", "Edit", "Glob", "Grep", "Skill"]


@dataclass(frozen=True)
class Condition:
    label: str
    kind: str  # "mcp" | "cli"
    skills_dir: Path | None = None
    setup: tuple[str, ...] = ()  # rf-mcp onboarding steps run in the seeded project: "sync" (sync-skills)


SETUPS = {"mcp+sync": ("sync",)}
CODEX_DISABLED_FEATURES = ("apps", "plugins", "browser_use", "computer_use", "image_generation", "multi_agent", "goals")


def parse_conditions(spec: str, skills: dict[str, str]) -> list[Condition]:
    out = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        if item in ("mcp", "cli"):
            out.append(Condition(item, item))
            continue
        if item in SETUPS:
            out.append(Condition(item.replace("+", "-"), "mcp", setup=SETUPS[item]))
            continue
        kind, _, variant = item.partition(":")
        if kind != "mcp" or not variant:
            raise ValueError(f"unknown condition {item!r} (use mcp, cli or mcp:<skills label|dir>)")
        path = Path(skills.get(variant, variant))
        if not path.is_dir():
            raise ValueError(f"condition {item!r}: skills dir {path} does not exist (pass --skills {variant}=<dir>)")
        label = variant if variant in skills else path.name
        out.append(Condition(f"mcp-{label}", "mcp", path.resolve()))
    return out


@dataclass
class Job:
    scenario: Scenario
    model: str
    condition: Condition
    rep: int

    @property
    def run_id(self) -> str:
        return f"{self.scenario.name}__{self.model}__{self.condition.label}__r{self.rep}"


@dataclass
class RunContext:
    space: str = field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:10]}")
    user: str = field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:8]}")
    web_url: str = ""
    ledger_url: str = ""
    shop: Demoshop | None = None
    ledger: LedgerFixture | None = None


class Harness:
    def __init__(self, suite: SuiteSpec, out: Path, *, dry_run: bool = False, cli_skill: Path | None = None,
                 max_turns: int | None = None, timeout: int | None = None, runs_dir: Path | None = None):
        self.suite = suite
        self.out = out
        self.dry_run = dry_run
        self.cli_skill = cli_skill
        self.max_turns = max_turns or suite.max_turns
        self.timeout = timeout or suite.timeout
        self.runs_dir = runs_dir or Path(os.environ.get("RF_EVAL_RUNS_DIR") or out / "projects")
        self.env = ProjectEnv(project_python(suite.name))
        self.shop = Demoshop()
        self.web: WebFixture | None = None
        self.ledger: LedgerFixture | None = None
        self.shim_dir: Path | None = None
        self.env_fingerprint: str | None = None

    # ------------------------------------------------------------------ fixtures

    def start_fixtures(self, scenarios: list[Scenario]) -> None:
        if self.dry_run:
            return
        if self.suite.name == "fixtures" and any(s.name in ("fx_browser", "fx_selenium") for s in scenarios):
            self.web = WebFixture(self.out).start()
        if self.suite.name == "api" and any(s.name != "api_booker" for s in scenarios):
            self.ledger = LedgerFixture(ledger_bugs(scenarios)).start()

    def stop_fixtures(self) -> None:
        for fx in (self.web, self.ledger):
            if fx:
                fx.stop()

    def context(self) -> RunContext:
        ctx = RunContext(shop=self.shop, ledger=self.ledger)
        ctx.web_url = self.web.url if self.web else "http://127.0.0.1:<web-port>/index.html"
        ctx.ledger_url = self.ledger.url if self.ledger else "http://127.0.0.1:<ledger-port>"
        return ctx

    # ------------------------------------------------------------------ seeding and commands

    def prepare(self, job: Job, ctx: RunContext) -> tuple[Path, list[str]]:
        """Seed the isolated project; return (dir, steps skipped because of --dry-run)."""
        sc = job.scenario
        dest = self.runs_dir / job.run_id
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.suite.seed(dest, sc, ctx)
        skipped = []
        cx = codex_model(job.model)
        if cx and job.condition.kind == "mcp":
            # Codex reads MCP servers from the trusted project's .codex/config.toml (its -c overrides were not
            # picked up for MCP servers in codex-cli 0.166)
            (dest / ".codex").mkdir(exist_ok=True)
            (dest / ".codex" / "config.toml").write_text(codex_mcp_toml(self.mcp_config(job, dest, ctx)), encoding="utf-8")
        if "sync" in job.condition.setup:
            from robotmcp.onboarding.cli import sync_skills_to_directory

            skills_dirs = [job.condition.skills_dir] if job.condition.skills_dir else []
            sync_skills_to_directory(root=dest, agent="codex" if cx else "claude", skills_dirs=skills_dirs)
        if job.condition.kind == "cli":
            if self.cli_skill and self.cli_skill.is_dir():
                skills = dest / (".agents" if cx else ".claude") / "skills" / "robotcode"
                shutil.copytree(self.cli_skill, skills)
            elif self.dry_run:
                skipped.append("upstream robotcode skill NOT copied: set RF_EVAL_CLI_SKILL_DIR or --cli-skill")
            else:
                raise SeedError("the cli condition needs the upstream robotcode skill (RF_EVAL_CLI_SKILL_DIR / --cli-skill)")
        if sc.preset:
            if self.dry_run:
                skipped.append(f"demoshop preset {sc.preset!r} in space {ctx.space}")
            else:
                self.shop.set_preset(ctx.space, sc.preset)
        if sc.prerun:
            if self.dry_run:
                skipped.append("prerun `robotcode robot` (fills results/)")
            else:
                self.env.rc(dest, "robot", **({"SHOP_SPACE": ctx.space} if self.suite.name == "demoshop" else {}))
        return dest, skipped

    def mcp_config(self, job: Job, dest: Path, ctx: RunContext) -> dict | None:
        if job.condition.kind != "mcp":
            return None
        extra = {"SHOP_SPACE": ctx.space} if self.suite.name == "demoshop" else {}
        return {"mcpServers": {"rf-mcp": self.env.mcp_server(dest, job.condition.skills_dir, extra)}}

    def command(self, job: Job, mcp_path: Path | None, dest: Path | None = None) -> list[str]:
        web = job.scenario.web
        cx = codex_model(job.model)
        if cx:
            # Codex's own sandbox: writes only in the project (+ temp), network on for the API and web suites. On
            # Windows it refuses the Store pwsh.exe, so run() drops WindowsApps from PATH.
            root = str(dest or "")
            cmd = [codex_executable(), "exec", "--json", "--ephemeral", "--skip-git-repo-check", "-s", "workspace-write",
                   "-c", "sandbox_workspace_write.network_access=true", "-c", 'approval_policy="never"',
                   "-c", f"web_search=\"{'live' if web else 'disabled'}\"", *cx.config()]
            catalog = cx.provider and dest and codex_oss_catalog(cx, dest / ".codex" / "models.json")
            if catalog:
                cmd += ["-c", f"model_catalog_json={json.dumps(str(catalog))}"]
            for feature in CODEX_DISABLED_FEATURES:  # the user's apps/plugins add ~30k chars of MCP resources
                cmd += ["--disable", feature]
            # one inline table: Codex splits a dotted `projects.'<path>'.trust_level` key at dots inside the quoted
            # path too (run ids of `cx-or-<vendor>.<model>` contain them), and ignores the setting
            trusted = ", ".join(f"'{key}' = {{ trust_level = \"trusted\" }}" for key in sorted({root, root.lower()}))
            cmd += ["-c", f"projects={{ {trusted} }}"]
            return cmd + ["-m", cx.model, "-C", root, "-"]
        cmd = [claude_executable(), "-p", "--model", resolve_model(job.model), "--output-format", "stream-json",
               "--verbose", "--no-session-persistence", "--setting-sources", "project", "--strict-mcp-config",
               "--max-turns", str(self.max_turns)]
        denied_web = ["WebSearch"] + ([] if web else ["WebFetch"])
        if job.condition.kind == "mcp":
            cmd += ["--mcp-config", str(mcp_path), "--allowedTools", *MCP_TOOLS, *(["WebFetch"] if web else []),
                    "--disallowedTools", "Bash", "PowerShell", *denied_web]
        else:
            cmd += ["--allowedTools", *CLI_TOOLS, *(["WebFetch"] if web else []), "--disallowedTools", *denied_web]
        return cmd

    # ------------------------------------------------------------------ one run

    def dry(self, job: Job) -> bool:
        ctx = self.context()
        print(f"\n== {job.run_id}")
        try:
            dest, skipped = self.prepare(job, ctx)
        except (SeedError, FixtureError, OSError) as e:
            print(f"   SEED FAILED: {e}")
            return False
        n_files = sum(1 for p in dest.rglob("*") if p.is_file())
        print(f"   project: {dest} ({n_files} files)")
        for s in skipped:
            print(f"   skipped (dry run): {s}")
        cfg = self.mcp_config(job, dest, ctx)
        mcp_path = self._write_mcp(job, cfg)
        prompt_path = self.out / f"{job.run_id}.prompt.txt"
        prompt_path.write_text(job.scenario.prompt, encoding="utf-8")
        if cfg:
            print(f"   mcp config ({mcp_path}): {json.dumps(cfg)}")
        print(f"   cwd: {dest}")
        print(f"   command: {_cmdline(self.command(job, mcp_path, dest))} < {prompt_path}")
        return True

    def _write_mcp(self, job: Job, cfg: dict | None) -> Path | None:
        if not cfg:
            return None
        path = self.out / f"{job.run_id}.mcp.json"
        path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        return path

    def run(self, job: Job) -> dict:
        ctx = self.context()
        sc = job.scenario
        res: dict = dict(run_id=job.run_id, suite=self.suite.name, scenario=sc.name, model=job.model,
                         model_id=resolve_model(job.model), condition=job.condition.label, rep=job.rep)
        log = self.out / f"{job.run_id}.jsonl"
        try:
            dest, _ = self.prepare(job, ctx)
        except Exception as e:  # noqa: BLE001 - recorded as a failed run
            res.update(exit="seed_error", grade={"pass": False, "seed_error": repr(e)})
            self._save(res)
            return res
        if self.env_fingerprint is not None and self.env.fingerprint() != self.env_fingerprint:
            # an earlier run changed the shared project env: do not spend a paid run in a broken environment
            res.update(exit="env_changed", grade={"pass": False, "skipped": "project env changed by an earlier run"})
            self._save(res)
            print(f"SKIPPED {job.run_id}: the project env changed since the sweep started", flush=True)
            return res
        before = G.tree_hash(dest)
        cmd = self.command(job, self._write_mcp(job, self.mcp_config(job, dest, ctx)), dest)
        t0 = time.time()
        try:
            with open(log, "w", encoding="utf-8") as fh:
                env = self.env.agent_env(self.shim_dir)
                cx = codex_model(job.model)
                if cx:
                    env["PATH"] = without_store_apps(env["PATH"])
                    if cx.provider == "openrouter":  # Codex's default shell env policy keeps *KEY* vars from commands
                        env["OPENROUTER_API_KEY"] = openrouter_settings()[1]
                elif openrouter_model(job.model):
                    env.update(openrouter_claude_env(openrouter_model(job.model)))
                p = subprocess.run(cmd, cwd=dest, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=self.timeout, input=sc.prompt, text=True, encoding="utf-8", check=False)
            res["exit"] = p.returncode
        except subprocess.TimeoutExpired:
            res["exit"] = "timeout"
        res["wall_s"] = round(time.time() - t0, 1)
        self.env.rc(dest, "session", "stop", "--project", timeout=120)
        if self.env_fingerprint is not None and self.env.fingerprint() != self.env_fingerprint:
            res["env_changed"] = True  # this run (or one running in parallel) installed or removed packages
            print(f"WARNING {job.run_id}: the project env changed during this run", flush=True)
        t = G.load_transcript(log)
        t.cwd = t.cwd or str(dest)
        res.update(G.run_metrics(t))
        res["process"] = G.process_grades(t, why_failed=sc.why_failed)
        try:
            res["grade"] = self.grade(sc, dest, t, before, ctx)
            if G.written_suite_files(t, dest):
                res["grade"]["written_analyze_errors"] = G.analyze_errors(self.env, dest, *G.written_suite_files(t, dest))
        except Exception as e:  # noqa: BLE001
            res["grade"] = {"pass": False, "grader_error": repr(e), "trace": traceback.format_exc()[-800:]}
        if self.suite.name in ("demoshop", "fixtures"):
            self.shop.reset(ctx.space)
        self._save(res)
        g = res["grade"]
        print(f"done {job.run_id}: pass={g.get('pass')} tests={g.get('passed')}/{g.get('total')} turns={res.get('num_turns')} "
              f"cost={res.get('cost')} flags={res['process']['flags']} {res['wall_s']:.0f}s", flush=True)
        return res

    def grade(self, sc: Scenario, dest: Path, t: G.Transcript, before: dict, ctx: RunContext) -> dict:
        if self.suite.name == "baseline":
            return G.grade_baseline(sc, dest, t, self.env, before)
        if self.suite.name == "demoshop":
            return G.grade_demoshop(sc, dest, t, self.env, self.shop)
        if self.suite.name == "fixtures":
            return G.grade_fixtures(sc, dest, t, self.env, ctx)
        return G.grade_api(sc, dest, t, self.env, ctx)

    def _save(self, res: dict) -> None:
        (self.out / f"{res['run_id']}.result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ all runs

    def execute(self, jobs: list[Job], parallel: int) -> list[dict] | bool:
        self.out.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            print(f"dry run: {len(jobs)} runs of suite {self.suite.name!r}; project env python: "
                  f"{self.env.python or '(harness interpreter)'}; rf-mcp served from "
                  f"{'the project env' if self.env.serves_rf_mcp() else 'uv run --directory <repo>'}; "
                  "fixtures are not started (URLs below are placeholders)")
            return all([self.dry(j) for j in jobs])
        if any(j.condition.kind == "cli" for j in jobs) and not (self.cli_skill and self.cli_skill.is_dir()):
            raise SystemExit("the cli condition needs the upstream robotcode skill: set RF_EVAL_CLI_SKILL_DIR or --cli-skill")
        if any(j.condition.kind == "cli" for j in jobs):
            self.shim_dir = write_guard_shims(self.out / "_guard", uv_executable(), self.env.pip_executable())
        self.env_fingerprint = self.env.fingerprint()
        self.start_fixtures(list({j.scenario.name: j.scenario for j in jobs}.values()))
        try:
            with cf.ThreadPoolExecutor(max(1, parallel)) as ex:
                return list(ex.map(self.run, jobs))
        finally:
            self.stop_fixtures()


def make_jobs(suite: SuiteSpec, scenarios: list[str], models: list[str], conditions: list[Condition], reps: int) -> list[Job]:
    return [Job(suite.scenarios[s], m, c, r) for r in range(1, reps + 1) for s in scenarios for m in models for c in conditions]


def codex_mcp_toml(cfg: dict | None) -> str:
    """`[mcp_servers.<name>]` tables for a Codex config.toml from an `mcpServers` dict."""
    lines = []
    for name, server in ((cfg or {}).get("mcpServers") or {}).items():
        lines += [f"[mcp_servers.{name}]", f"command = {json.dumps(server['command'])}",
                  f"args = {json.dumps(server.get('args') or [])}", "startup_timeout_sec = 120", "tool_timeout_sec = 900",
                  # like --allowedTools mcp__rf-mcp for claude: under approval_policy="never" Codex otherwise rejects
                  # every tool not annotated read-only (rf_run, rf_repl, rf_debug)
                  'default_tools_approval_mode = "approve"']
        if server.get("env"):
            lines.append("env = { " + ", ".join(f"{json.dumps(k)} = {json.dumps(v)}" for k, v in server["env"].items()) + " }")
        lines.append("")
    return "\n".join(lines)


def _cmdline(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    import shlex

    return shlex.join(cmd)
