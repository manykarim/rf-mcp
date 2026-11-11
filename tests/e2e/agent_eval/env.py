"""Interpreters, robotcode invocation and process environments for the agent evaluation.

Every interpreter is configurable, so the harness runs on any machine:

- `RF_EVAL_<SUITE>_PYTHON` (`RF_EVAL_BASELINE_PYTHON`, `RF_EVAL_DEMOSHOP_PYTHON`, `RF_EVAL_FIXTURES_PYTHON`,
  `RF_EVAL_API_PYTHON`): Python of the environment the agent's project runs in (robotcode and the libraries
  the suite needs; rf-mcp installed there too makes the MCP server see the same libraries).
- `RF_MCP_COOKBOOK_PYTHON`: fallback for the `fixtures` and `api` suites (same meaning as in tests/test_cookbook.py).
- `RF_EVAL_PYTHON`: fallback for every suite. Without any of them the interpreter running the harness is used.
- `RF_MCP_FIXTURE_PYTHON`: Python with fastapi, uvicorn, python-multipart, strawberry-graphql for the ledger fixture.
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`: OpenRouter access for `or-` and `cx-or-` models; read from the
  environment or the repo's `.env` (git-ignored). Only the agent process gets the key.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIXTURES_DIR = REPO / "tests" / "fixtures"
PROJECTS_DIR = HERE / "projects"

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}

FIXTURE_PY = os.environ.get("RF_MCP_FIXTURE_PYTHON", sys.executable)


def project_python(suite: str) -> str | None:
    """Return the configured interpreter for `suite`, or None for the harness interpreter."""
    names = [f"RF_EVAL_{suite.upper()}_PYTHON"]
    if suite in ("fixtures", "api"):
        names.append("RF_MCP_COOKBOOK_PYTHON")
    names.append("RF_EVAL_PYTHON")
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    return None


def resolve_model(name: str) -> str:
    """Map a short model name (haiku, sonnet, cx-..., or-...) to its id; pass other ids through."""
    cx = codex_model(name)
    if cx:
        return cx.model
    return openrouter_model(name) or MODELS.get(name, name)


def openrouter_slug(name: str) -> str:
    """`z-ai.glm-5.3-flash` -> `z-ai/glm-5.3-flash`: model names in run ids (file names) cannot contain `/`."""
    vendor, _, model = name.partition(".")
    return f"{vendor}/{model}"


def openrouter_model(name: str) -> str | None:
    """OpenRouter slug of a Claude Code model named `or-<vendor>.<model>`; None for other names."""
    return openrouter_slug(name[3:]) if name.startswith("or-") else None


@functools.cache
def openrouter_settings() -> tuple[str, str]:
    """(base URL, API key) from the environment, else from the repo's `.env`."""
    values = {k: os.environ.get(k, "") for k in ("OPENROUTER_BASE_URL", "OPENROUTER_API_KEY")}
    dotenv = REPO / ".env"
    if not all(values.values()) and dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() in values and not values[key.strip()]:
                values[key.strip()] = value.strip().strip("'\"")
    if not values["OPENROUTER_API_KEY"]:
        raise RuntimeError("OpenRouter models need OPENROUTER_API_KEY (environment or .env)")
    return (values["OPENROUTER_BASE_URL"] or "https://openrouter.ai/api/v1").rstrip("/"), values["OPENROUTER_API_KEY"]


def openrouter_claude_env(slug: str) -> dict[str, str]:
    """Claude Code on OpenRouter's Anthropic-compatible API, with every model role (background calls, subagents)
    on `slug` so no request goes to an Anthropic model."""
    base, key = openrouter_settings()
    env = dict(ANTHROPIC_BASE_URL=base.removesuffix("/v1"), ANTHROPIC_AUTH_TOKEN=key, ANTHROPIC_API_KEY="",
               CLAUDE_CODE_SUBAGENT_MODEL=slug, ANTHROPIC_SMALL_FAST_MODEL=slug,
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    for role in ("HAIKU", "SONNET", "OPUS"):
        env[f"ANTHROPIC_DEFAULT_{role}_MODEL"] = slug
    return env


def claude_executable() -> str:
    return os.environ.get("RF_EVAL_CLAUDE") or shutil.which("claude.cmd") or shutil.which("claude") or "claude"


def codex_executable() -> str:
    return os.environ.get("RF_EVAL_CODEX") or shutil.which("codex") or "codex"


@dataclass(frozen=True)
class CodexModel:
    """A model run through `codex exec`, named `cx-<model>[@<N>k][~<effort>]`, `cx-oss-<model>[@<N>k]` or
    `cx-or-<vendor>.<model>[@<N>k]`.

    `@32k` caps the context window Codex assumes (`model_context_window`, auto-compaction at 80%), so a large
    model can stand in for a small-context one. `cx-oss-qwen3_14b` runs `qwen3:14b` on the local Ollama provider
    (`model_provider="ollama"`; `--oss` fails with a provider mismatch in Codex 0.166). Codex has no metadata for
    Ollama models, so their window defaults to Ollama's `OLLAMA_CONTEXT_LENGTH` (32768 if unset), which the
    Ollama server must be started with. `cx-or-z-ai.glm-5.3-flash` runs `z-ai/glm-5.3-flash` on OpenRouter's
    Responses API with a 128k window unless `@<N>k` says otherwise.
    """

    model: str
    provider: str = ""  # "" (OpenAI), "ollama" or "openrouter"
    context: int | None = None
    effort: str = "low"

    def config(self) -> list[str]:
        args = ["-c", f'model_reasoning_effort="{self.effort}"']
        if self.provider == "ollama":
            args += ["-c", 'model_provider="ollama"']
        elif self.provider == "openrouter":
            base, _ = openrouter_settings()
            p = "model_providers.openrouter"
            args += ["-c", 'model_provider="openrouter"', "-c", f'{p}.name="OpenRouter"', "-c", f"{p}.base_url={json.dumps(base)}",
                     "-c", f'{p}.env_key="OPENROUTER_API_KEY"', "-c", f'{p}.wire_api="responses"']
        if self.context:
            args += ["-c", f"model_context_window={self.context}",
                     "-c", f"model_auto_compact_token_limit={int(self.context * 0.8)}"]
        return args


def codex_oss_catalog(cx: CodexModel, path: Path) -> Path | None:
    """Write a Codex model catalog with an entry for a non-OpenAI model; None if Codex has no cached catalog to copy.

    Without one Codex runs Ollama and OpenRouter models on fallback metadata. The entry copies a cached OpenAI model and sets:
    direct tools (the cached models are `code_mode_only`, which hides MCP tools behind `exec`), the skills usage
    instructions, the real window, and no `apply_patch` for Ollama: Codex only offers it as a freeform tool, and
    through Ollama every call failed with "tool apply_patch invoked with incompatible payload" (2026-11-07,
    qwen3:8b). OpenRouter passes it through (2026-11-09, glm-5.3-flash and gpt-oss-20b created files with it).
    """
    cache = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "models_cache.json"
    try:
        models = json.loads(cache.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError):
        return None
    if not models:
        return None
    entry = dict(models[-1], slug=cx.model, display_name=cx.model, description=f"{cx.provider} model",
                 context_window=cx.context, max_context_window=cx.context, tool_mode="direct",
                 include_skills_usage_instructions=True, input_modalities=["text"],
                 supports_search_tool=False, use_responses_lite=False, additional_speed_tiers=[], service_tiers=[],
                 experimental_supported_tools=[], multi_agent_version=None)
    if cx.provider == "ollama":
        entry["apply_patch_tool_type"] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": [entry]}), encoding="utf-8")
    return path


def codex_model(name: str) -> CodexModel | None:
    """Parse a `cx-...` model name; None for Claude models."""
    if not name.startswith("cx-"):
        return None
    rest, effort = name[3:], "low"
    if "~" in rest:
        rest, effort = rest.split("~", 1)
    context = None
    if "@" in rest:
        rest, size = rest.split("@", 1)
        context = int(size.lower().rstrip("k")) * 1000
    provider = ""
    if rest.startswith("oss-"):
        provider, rest = "ollama", rest[4:].replace("_", ":")
        context = context or int(os.environ.get("OLLAMA_CONTEXT_LENGTH") or 32768)
    elif rest.startswith("or-"):
        provider, rest = "openrouter", openrouter_slug(rest[3:])
        context = context or 128_000
    return CodexModel(rest, provider, context, effort)


def without_store_apps(path: str) -> str:
    """PATH without `WindowsApps`: Codex's Windows sandbox cannot start the Store `pwsh.exe` there
    (CreateProcessAsUserW access denied), so Codex falls back to System32 `powershell.exe`, which it can start.
    Weak models gave up after two refused shell calls (2026-11-07, qwen3:8b)."""
    return os.pathsep.join(p for p in path.split(os.pathsep) if "windowsapps" not in p.lower())


def uv_executable() -> str:
    return shutil.which("uv") or "uv"


# Commands that change a Python environment. Shell agents share the project env with every parallel run (and
# UV_PROJECT_ENVIRONMENT points `uv` at it), so one `uv sync` against the project's lockfile uninstalls rf-mcp
# for every run after it (2026-11-06: a Haiku CLI run did exactly that mid-sweep).
GUARD_MESSAGE = ("blocked by the rf-mcp evaluation harness: the project environment is pre-installed and shared; "
                 "do not install, sync or remove packages. Use `uv run --no-sync ...` or the tools on PATH.")
_UV_BLOCKED = ("sync", "add", "remove", "lock", "venv")
_PIP_BLOCKED = ("install", "uninstall", "sync")


def write_guard_shims(directory: Path, real_uv: str, real_pip: str) -> Path:
    """Write `uv` / `pip` shims (POSIX sh and .cmd) that refuse environment changes and forward everything else."""
    directory.mkdir(parents=True, exist_ok=True)
    uv_cases = "|".join(_UV_BLOCKED)
    pip_cases = "|".join(_PIP_BLOCKED)
    msg = f"'{GUARD_MESSAGE}'"  # single-quoted: the backticks in the message must not run in sh
    sh_uv = (f'#!/bin/sh\ncase "$1" in {uv_cases}) echo "uv $1:" {msg} >&2; exit 2;; esac\n'
             f'if [ "$1" = pip ]; then case "$2" in {pip_cases}) echo "uv pip $2:" {msg} >&2; exit 2;; esac; fi\n'
             f'exec "{Path(real_uv).as_posix()}" "$@"\n')
    sh_pip = (f'#!/bin/sh\ncase "$1" in {pip_cases}) echo "pip $1:" {msg} >&2; exit 2;; esac\n'
              f'exec "{Path(real_pip).as_posix()}" "$@"\n')
    cmd_checks_uv = "".join(f'if /I "%~1"=="{c}" goto blocked\r\n' for c in _UV_BLOCKED)
    cmd_checks_uv += "".join(f'if /I "%~1"=="pip" if /I "%~2"=="{c}" goto blocked\r\n' for c in _PIP_BLOCKED)
    cmd_checks_pip = "".join(f'if /I "%~1"=="{c}" goto blocked\r\n' for c in _PIP_BLOCKED)
    for name, checks, real in (("uv", cmd_checks_uv, real_uv), ("pip", cmd_checks_pip, real_pip)):
        (directory / f"{name}.cmd").write_text(
            f'@echo off\r\n{checks}"{real}" %*\r\nexit /b %ERRORLEVEL%\r\n:blocked\r\necho {name} %*: {GUARD_MESSAGE} 1>&2\r\nexit /b 2\r\n',
            encoding="utf-8")
    for name, body in (("uv", sh_uv), ("pip", sh_pip), ("pip3", sh_pip)):
        path = directory / name
        path.write_text(body, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    return directory


@functools.lru_cache(maxsize=None)
def has_modules(python: str, *modules: str) -> bool:
    code = "import importlib, sys; [importlib.import_module(m) for m in sys.argv[1:]]"
    try:
        return subprocess.run([python, "-c", code, *modules], capture_output=True, check=False, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@dataclass
class ProjectEnv:
    """The Python environment an evaluated project runs in."""

    python: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def interpreter(self) -> str:
        return self.python or sys.executable

    @property
    def scripts_dir(self) -> Path:
        return Path(self.interpreter).parent

    def robotcode(self) -> list[str]:
        for name in ("robotcode.exe", "robotcode"):
            candidate = self.scripts_dir / name
            if candidate.is_file():
                return [str(candidate)]
        return [shutil.which("robotcode") or "robotcode"]

    def env(self, **extra: str) -> dict[str, str]:
        """Environment for claude, the MCP server and grader runs: the project env's scripts first on PATH."""
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)
        env.pop("OPENROUTER_API_KEY", None)  # the agent process gets it back for OpenRouter models only
        env["PATH"] = str(self.scripts_dir) + os.pathsep + env.get("PATH", "")
        if self.python:
            venv = str(self.scripts_dir.parent)
            env.update(VIRTUAL_ENV=venv, UV_PROJECT_ENVIRONMENT=venv, UV_NO_SYNC="1")
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", ROBOTCODE_FORCE_AI_AGENT="1", PYTHONWARNINGS="ignore")
        env.update(self.extra_env)
        env.update({k: v for k, v in extra.items() if v is not None})
        return env

    def agent_env(self, shim_dir: Path | None) -> dict[str, str]:
        """Environment for the agent process: env() with the `uv` / `pip` guard shims first on PATH."""
        env = self.env()
        if shim_dir:
            env["PATH"] = str(shim_dir) + os.pathsep + env["PATH"]
        return env

    def pip_executable(self) -> str:
        for name in ("pip.exe", "pip"):
            if (self.scripts_dir / name).is_file():
                return str(self.scripts_dir / name)
        return shutil.which("pip") or "pip"

    def fingerprint(self) -> str:
        """`uv pip freeze` of the project env: changes when anything installs or removes a package."""
        try:
            p = subprocess.run([uv_executable(), "pip", "freeze", "--python", self.interpreter], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"error: {exc}"
        return p.stdout if p.returncode == 0 else f"error: {p.stderr.strip()}"

    def rc(self, root: Path, *args: str, timeout: int = 900, **extra_env: str) -> tuple[int, str]:
        """Run `robotcode -r <root> <args>` in the project env; return (exit code, stdout+stderr)."""
        try:
            p = subprocess.run(
                [*self.robotcode(), "-r", str(root), *args],
                cwd=root,
                env=self.env(**extra_env),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return 124, f"timeout after {timeout}s: {exc}"
        except OSError as exc:
            return 127, f"cannot run robotcode: {exc}"
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    def run_python(self, root: Path, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.interpreter, *args], cwd=root, env=self.env(), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )

    def serves_rf_mcp(self) -> bool:
        """True when rf-mcp is importable in the project env (then the server runs there, next to its libraries)."""
        return bool(self.python) and has_modules(self.interpreter, "robotmcp")

    def mcp_server(self, root: Path, skills_dir: Path | None = None, env: dict[str, str] | None = None) -> dict:
        """stdio MCP server entry: rf-mcp from the project env, else `uv run --directory <repo> rf-mcp`."""
        flags = ["--mode", "server", "--root", str(root)]
        if skills_dir:
            flags += ["--skills-dir", str(skills_dir)]
        if self.serves_rf_mcp():
            server = {"command": self.interpreter, "args": ["-m", "robotmcp", *flags]}
        else:
            server = {"command": uv_executable(), "args": ["run", "--directory", str(REPO), "rf-mcp", *flags]}
        server_env = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "ROBOTCODE_FORCE_AI_AGENT": "1",
                      "PYTHONWARNINGS": "ignore"}
        if os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
            server_env["PLAYWRIGHT_BROWSERS_PATH"] = os.environ["PLAYWRIGHT_BROWSERS_PATH"]
        server_env.update(self.extra_env)
        server_env.update(env or {})
        return {"type": "stdio", **server, "env": server_env}
