"""Register / unregister the rf-mcp MCP server into coding-agent config files.

Merge-in-place and never overwrite: an agent's other MCP servers are preserved.
Every write is recorded in the hash-tracked manifest so uninstall is reversible
and safe. ``--what`` other than ``mcp`` is accepted but currently finds no bundled
assets (extension seam for future skills/subagents/hooks).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from robotmcp.onboarding import adapters as A
from robotmcp.onboarding import codecs
from robotmcp.onboarding.manifest import Manifest, value_hash

WHAT_ALL = ["mcp", "skills", "agents", "hooks"]
WHAT_IMPLEMENTED = {"mcp"}


def resolved_command() -> str:
    """Absolute path to the installed ``robotmcp`` executable if resolvable, so the
    agent launches the tool-installed binary regardless of PATH context."""
    exe = shutil.which("robotmcp")
    if exe:
        return exe
    # running as `python -m robotmcp.server` / dev: fall back to a plain name
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0.endswith("robotmcp") and Path(argv0).exists():
        return str(Path(argv0).resolve())
    return "robotmcp"


# --- project-aware launch resolution (change: installer-project-aware-launch) ----

def _own_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("rf-mcp")
    except Exception:
        return None


def _rfmcp_with_args() -> List[str]:
    """uv ``--with`` args that add the SAME rf-mcp the user has installed:
    ``--with-editable <src>`` for a local/editable/dev install (its version may not
    be on PyPI), otherwise a version-pinned ``--with rf-mcp==<ver>`` for a published
    install, falling back to an unpinned ``--with rf-mcp``."""
    try:
        from importlib.metadata import distribution
        raw = distribution("rf-mcp").read_text("direct_url.json")
        if raw:
            info = json.loads(raw)
            url = info.get("url", "")
            editable = bool(info.get("dir_info", {}).get("editable"))
            if (editable or url.startswith("file:")) and url:
                path = url[7:] if url.startswith("file://") else url[5:] if url.startswith("file:") else url
                if path and Path(path).exists():
                    return ["--with-editable", path]
    except Exception:
        pass
    ver = _own_version()
    return ["--with", f"rf-mcp=={ver}"] if ver else ["--with", "rf-mcp"]


def _venv_shim(python: Path) -> Optional[Path]:
    """The ``robotmcp`` console script next to a venv's interpreter, if present."""
    for name in ("robotmcp", "robotmcp.exe"):
        p = python.parent / name
        if p.exists():
            return p
    return None


def _attach_env(attach: Optional[str]) -> Dict[str, str]:
    """Env for an attach-bridge entry. ``attach`` may be ``host``/``host:port``;
    defaults to 127.0.0.1:7317. Token uses the bridge default unless overridden."""
    host, port = "127.0.0.1", "7317"
    if attach and attach not in ("1", "true", "yes", "auto"):
        host = attach.split(":", 1)[0] or host
        if ":" in attach:
            port = attach.split(":", 1)[1] or port
    return {"ROBOTMCP_ATTACH_HOST": host, "ROBOTMCP_ATTACH_PORT": port}


@dataclass
class LaunchPlan:
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    strategy: str = "own-shim"   # own-shim|in-project|uv-overlay|attach|into-project|fallback
    note: str = ""               # human note surfaced by the CLI
    verify_lib: Optional[str] = None   # a project library to confirm reachable, if any


def _in_project_plan(env, verify_lib=None, *, strategy: str = "in-project") -> LaunchPlan:
    """Launch rf-mcp using the project's interpreter (it is importable there)."""
    shim = _venv_shim(env.python) if env.python else None
    if shim is not None:
        return LaunchPlan(str(shim), [], {}, strategy,
                          f"runs rf-mcp from the project's {env.type} env ({shim})",
                          verify_lib)
    return LaunchPlan(str(env.python), ["-m", "robotmcp.server"], {}, strategy,
                      f"runs rf-mcp via the project's {env.type} interpreter", verify_lib)


def resolve_launch(*, scope: str, project_dir: Optional[Path] = None,
                   into_project: bool = False, attach: Optional[str] = None,
                   command_override: Optional[str] = None,
                   env_override: Optional[Dict[str, str]] = None) -> LaunchPlan:
    """Resolve command+args+env so the running rf-mcp can see the target project's
    Robot Framework libraries (change: installer-project-aware-launch). Preferring
    uv; the global ``uvx``/``uv tool`` case (no project env) keeps the plain own-shim.
    """
    from robotmcp.onboarding import project_env as pe

    extra_env = dict(env_override or {})

    # explicit overrides win, unconditionally
    if command_override:
        return LaunchPlan(command_override, [], extra_env, "override", "explicit --command")
    if attach:
        e = _attach_env(attach); e.update(extra_env)
        return LaunchPlan(resolved_command(), [], e, "attach",
                          "attach bridge (explicit --attach) - start the project's RF "
                          "process with the McpAttach library")

    # user scope / non-project -> rf-mcp's own command (the easy global default)
    if scope != "project":
        return LaunchPlan(resolved_command(), [], extra_env, "own-shim", "user-scope own command")

    env = pe.detect(project_dir)
    if not env.has_env:
        return LaunchPlan(resolved_command(), [], extra_env, "own-shim",
                          "no project environment detected - global rf-mcp")

    extras = pe.project_extra_libraries(env)
    vlib = extras[0] if extras else None

    # (1) rf-mcp already importable in the project env -> run it there
    if pe.rfmcp_in_project(env.python):
        p = _in_project_plan(env, vlib); p.env.update(extra_env); return p

    # (6) irreconcilable version conflict -> attach bridge (never a silent overlay)
    conflict = pe.rf_conflict(env)
    if conflict:
        e = _attach_env(None); e.update(extra_env)
        return LaunchPlan(resolved_command(), [], e, "attach",
                          f"{conflict}. Routed to the attach bridge; run the project's "
                          f"own RF process with McpAttach.")

    # (3) project needs only libraries rf-mcp already bundles -> own command
    if not extras:
        return LaunchPlan(resolved_command(), [], extra_env, "own-shim",
                          "project uses only bundled libraries - global rf-mcp")

    # project has extra libraries and rf-mcp is not in its env
    if into_project:
        ok, _detail = _install_into_project(env)
        if ok:
            p = _in_project_plan(env, vlib, strategy="into-project")
            p.env.update(extra_env); p.note = "installed rf-mcp into the project env; " + p.note
            return p
        # fall through to overlay/fallback if the into-project install failed
    uv_available = shutil.which("uv") is not None
    if uv_available and env.is_venv:
        # --no-project + --python <venv-py> layers the venv's ACTUAL site-packages
        # (including ad-hoc installs) and is cwd-independent; --project only syncs
        # declared deps and would miss undeclared libraries. rf-mcp is added by an
        # editable-aware spec so the overlay matches the installed rf-mcp even when
        # its version is not on PyPI.
        args = (["run", "--no-project", "--python", str(env.python)]
                + _rfmcp_with_args() + ["robotmcp"])
        return LaunchPlan("uv", args, extra_env, "uv-overlay",
                          f"uv overlay: rf-mcp layered onto the project's {env.type} env "
                          f"so it sees {', '.join(extras[:4])}"
                          + ("..." if len(extras) > 4 else ""), vlib)

    # (4)/(5) non-venv (conda/global) or no uv -> cannot overlay; guide to co-install.
    # verify_lib is set so verification (below) refuses this blind config unless the
    # user overrides - writing an own-shim that can't see the project libs is exactly
    # the "wrong environment" outcome we must not persist silently.
    return LaunchPlan(resolved_command(), [], extra_env, "fallback",
                      f"project needs {', '.join(extras[:4])} but its {env.type} env "
                      f"cannot be overlaid by uv; install rf-mcp INTO that env "
                      f"(re-run with --into-project) or use --attach", vlib)


def _install_into_project(env) -> Tuple[bool, str]:
    """Opt-in: install rf-mcp into the detected project env (mutating). Best-effort."""
    ver = _own_version()
    spec = f"rf-mcp=={ver}" if ver else "rf-mcp"
    if shutil.which("uv") and env.python:
        try:
            r = subprocess.run(["uv", "pip", "install", "--python", str(env.python), spec],
                               capture_output=True, text=True, timeout=300,
                               stdin=subprocess.DEVNULL)
            return r.returncode == 0, (r.stdout + r.stderr)[-500:]
        except Exception as exc:  # pragma: no cover - env dependent
            return False, str(exc)
    return False, "uv not available to install rf-mcp into the project env"


# --- launch verification (never write a config that can't do the job) -----------

def _plan_interpreter(plan: LaunchPlan) -> Optional[List[str]]:
    """An argv prefix that runs a Python IN the plan's target environment, so an
    import probe confirms library reachability. None when it can't be derived."""
    if plan.strategy == "uv-overlay" and plan.args and plan.args[-1] == "robotmcp":
        return ["uv", *plan.args[:-1], "python"]   # swap the server for a python
    cmd = Path(plan.command)
    if cmd.name in ("robotmcp", "robotmcp.exe"):     # abs console shim -> sibling python
        for n in ("python", "python3", "python.exe"):
            p = cmd.parent / n
            if p.exists():
                return [str(p)]
        return None
    if cmd.name.startswith("python"):                # in-project `<python> -m robotmcp.server`
        return [str(cmd)]
    return None                                      # bare name (no path) -> can't probe


def _mcp_initialize_probe(argv: List[str], extra_env: Dict[str, str],
                          timeout: float = 40.0) -> Tuple[bool, str]:
    """Launch the command as an agent would and complete the MCP ``initialize``
    handshake with a non-inherited env. Returns (started_ok, detail)."""
    env = {"HOME": os.path.expanduser("~"), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    env.update(extra_env)
    try:
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=env, text=True, bufsize=1)
    except Exception as exc:
        return False, f"spawn failed: {exc}"
    resp: Dict[str, object] = {}

    def _read():
        for line in p.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if line.startswith("{"):
                try:
                    m = json.loads(line)
                    if m.get("id") == 1:
                        resp["m"] = m
                        break
                except Exception:
                    pass
    t = threading.Thread(target=_read, daemon=True); t.start()
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
           "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                      "clientInfo": {"name": "install-verify", "version": "0"}}}
    try:
        p.stdin.write(json.dumps(req) + "\n"); p.stdin.flush()  # type: ignore[union-attr]
    except Exception as exc:
        err = (p.stderr.read() or "")[-300:] if p.stderr else ""
        p.terminate()
        return False, f"process died before handshake: {exc} {err}"
    t.join(timeout=timeout)
    ok = "m" in resp
    detail = "" if ok else "no MCP initialize response " + ((p.stderr.read() or "")[-300:] if p.stderr else "")
    try:
        p.terminate()
    except Exception:
        pass
    return ok, detail


def verify_launch(plan: LaunchPlan, timeout: float = 60.0) -> Tuple[bool, str]:
    """Confirm the resolved command starts the server and - when the plan targets a
    project environment with a specific library - that the library is reachable there.
    Fast library-import probe where possible; MCP handshake otherwise."""
    if plan.verify_lib:
        interp = _plan_interpreter(plan)
        if interp is not None:
            probe = [*interp, "-c", f"import {plan.verify_lib}; import robotmcp; print('ok')"]
            try:
                r = subprocess.run(probe, capture_output=True, text=True, timeout=timeout,
                                   stdin=subprocess.DEVNULL)
                if r.returncode == 0 and "ok" in r.stdout:
                    return True, f"rf-mcp + '{plan.verify_lib}' importable in the project env"
                return False, (f"'{plan.verify_lib}' not reachable via the resolved command: "
                               + (r.stderr or r.stdout)[-300:])
            except Exception as exc:
                return False, f"library probe failed: {exc}"
    # own-shim / attach / no specific lib -> confirm the server actually starts
    ok, detail = _mcp_initialize_probe([plan.command, *plan.args], plan.env, timeout=timeout)
    if plan.strategy == "attach":
        return ok, ("server starts; attach-host reachability is verified when the "
                    "project's RF process is running" if ok else detail)
    return ok, ("server starts" if ok else detail)


@dataclass
class Result:
    agent: str
    scope: str
    what: str
    status: str          # installed | updated | already-present | removed | kept-user-modified | skipped | planned | no-assets | absent
    path: Optional[str] = None
    detail: str = ""


def _targets(agents_spec: str, home: Optional[Path]) -> List[A.AgentAdapter]:
    return A.resolve_selection(agents_spec, home=home)


def install(*, agents: str = "detected", scope: str = "project",
            whats: Optional[List[str]] = None, dry_run: bool = False,
            force: bool = False, command: Optional[str] = None,
            env: Optional[Dict[str, str]] = None,
            manifest: Optional[Manifest] = None,
            home: Optional[Path] = None, cwd: Optional[Path] = None,
            project_dir: Optional[Path] = None, into_project: bool = False,
            attach: Optional[str] = None, no_verify: bool = False) -> List[Result]:
    whats = whats or ["mcp"]
    manifest = manifest or Manifest()
    results: List[Result] = []

    # Resolve the launch command ONCE (project-aware), then verify it ONCE before
    # writing it to every targeted agent (change: installer-project-aware-launch).
    plan = resolve_launch(scope=scope, project_dir=project_dir or cwd,
                          into_project=into_project, attach=attach,
                          command_override=command, env_override=env)
    verified, verify_detail = True, "skipped"
    if not (no_verify or dry_run):
        verified, verify_detail = verify_launch(plan)
    plan_note = plan.note + (f" | verify: {verify_detail}" if verify_detail else "")

    for what in whats:
        if what not in WHAT_IMPLEMENTED:
            results.append(Result("*", scope, what, "no-assets",
                                  detail="no bundled assets of this kind yet"))
            continue
        for ad in _targets(agents, home):
            if ad.status != "supported":
                results.append(Result(ad.id, scope, what, "planned",
                                      detail="adapter convention unconfirmed"))
                continue
            if not ad.supports_scope(scope):
                other = "user" if scope == "project" else "project"
                results.append(Result(ad.id, scope, what, "skipped",
                                      detail=f"{scope} scope unsupported; try --scope {other}"))
                continue
            # Refuse to persist a command that failed verification unless forced.
            if not verified and not force:
                results.append(Result(ad.id, scope, what, "unverified", path=None,
                                      detail=f"{plan.strategy}: {verify_detail} "
                                             f"(use --into-project/--attach, or --no-verify/--force)"))
                continue
            path = ad.resolve_path(scope, cwd=cwd, home=home)
            data, existed = codecs.load(path, ad.fmt)
            container = codecs.ensure_container(data, ad.container)
            entry = ad.build_entry(plan.command, plan.args, plan.env)
            present = A.SERVER_NAME in container
            if present and not force:
                results.append(Result(ad.id, scope, what, "already-present",
                                      path=str(path), detail="use --force to overwrite"))
                continue
            if dry_run:
                results.append(Result(ad.id, scope, what,
                                      "updated" if present else "installed",
                                      path=str(path),
                                      detail=f"dry-run [{plan.strategy}] {plan_note}"))
                continue
            container[A.SERVER_NAME] = entry
            codecs.dump(path, ad.fmt, data)
            manifest.record(agent=ad.id, scope=scope, what=what, path=str(path),
                            value=entry, created_file=not existed)
            results.append(Result(ad.id, scope, what,
                                  "updated" if present else "installed", path=str(path),
                                  detail=f"[{plan.strategy}] {plan_note}"))
    if not dry_run:
        manifest.save()
    return results


def uninstall(*, agents: str = "detected", scope: Optional[str] = None,
              whats: Optional[List[str]] = None, dry_run: bool = False,
              manifest: Optional[Manifest] = None,
              home: Optional[Path] = None, cwd: Optional[Path] = None) -> List[Result]:
    whats = whats or WHAT_ALL
    manifest = manifest or Manifest()
    results: List[Result] = []
    # Selection by id; 'all'/'detected' still resolve to concrete adapters.
    wanted_ids = [a.id for a in _targets(agents, home)] if agents not in ("", None) else A.SUPPORTED_IDS
    if agents == "all":
        wanted_ids = A.SUPPORTED_IDS

    for e in list(manifest.entries_for(wanted_ids, scope, whats)):
        ad = A.get(e["agent"])
        path = Path(e["path"])
        data, existed = codecs.load(path, ad.fmt) if ad else ({}, False)
        container = codecs.get_container(data, ad.container) if ad else None
        if not container or A.SERVER_NAME not in container:
            if not dry_run:
                manifest.drop(e)
            results.append(Result(e["agent"], e["scope"], e["what"], "absent", path=str(path)))
            continue
        current_hash = value_hash(container[A.SERVER_NAME])
        if current_hash != e["value_hash"]:
            results.append(Result(e["agent"], e["scope"], e["what"], "kept-user-modified",
                                  path=str(path), detail="entry changed since install; left intact"))
            continue
        if dry_run:
            results.append(Result(e["agent"], e["scope"], e["what"], "removed",
                                  path=str(path), detail="dry-run"))
            continue
        del container[A.SERVER_NAME]
        codecs.prune_empty(data, ad.container)
        if e.get("created_file") and not data:
            path.unlink(missing_ok=True)
        else:
            codecs.dump(path, ad.fmt, data)
        manifest.drop(e)
        results.append(Result(e["agent"], e["scope"], e["what"], "removed", path=str(path)))
    if not dry_run:
        manifest.save()
    return results


def list_agents(*, home: Optional[Path] = None, cwd: Optional[Path] = None,
                manifest: Optional[Manifest] = None) -> List[Dict[str, str]]:
    manifest = manifest or Manifest()
    installed = {(e["agent"], e["scope"]) for e in manifest.entries}
    rows: List[Dict[str, str]] = []
    for ad in A.REGISTRY:
        rows.append({
            "id": ad.id, "name": ad.name, "status": ad.status,
            "detected": "yes" if ad.detect(home=home) else "no",
            "registered": "yes" if any(a == ad.id for a, _ in installed) else "no",
            "format": ad.fmt,
        })
    return rows
