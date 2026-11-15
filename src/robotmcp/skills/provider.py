"""Skills over MCP provider (`SkillsDirectoryProvider` + SEP-2640 + `ResourcesAsTools`)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from fastmcp import FastMCP
from fastmcp.resources.base import Resource
from fastmcp.resources.template import ResourceTemplate
from fastmcp.server.dependencies import get_context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.providers import Provider
from fastmcp.server.transforms import ResourcesAsTools
from fastmcp.utilities.versions import VersionSpec

try:  # fastmcp moved the provider; the old path warns with FastMCPDeprecationWarning
    from fastmcp.skills.provider import SkillsDirectoryProvider
except ImportError:  # pragma: no cover - depends on the installed fastmcp
    from fastmcp.server.providers.skills import SkillsDirectoryProvider

from robotmcp.client_roots import ClientRoots
from robotmcp.config import ServerConfig
from robotmcp.telemetry import record_skill_read

BUILTIN_SKILLS_ROOT = Path(__file__).resolve().parent / "builtin"

INSTRUCTIONS_MAX_CHARS = 2500

_SKILL_URI_RE = re.compile(r"^skill://([^/]+)/")


def _expand_skill_root_candidates(path: Path) -> list[Path]:
    """Expand a user-provided skill path into concrete skill root directories.

    Supports:
    - A directory directly containing `<skill-name>/SKILL.md`
    - A single skill directory containing `SKILL.md` directly (uses its parent)
    - A plugin repository root such as `robotframework-agent-plugins` containing `plugins/*/skills`
    - A repository containing `skills/` or `.agents/skills/`
    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return []
    if resolved.is_file() and resolved.name == "SKILL.md":
        return [resolved.parent.parent]

    if (resolved / "SKILL.md").is_file():
        return [resolved.parent]

    found: list[Path] = []
    # Check if immediate children have SKILL.md
    if any(
        child.is_dir() and (child / "SKILL.md").is_file()
        for child in resolved.iterdir()
    ):
        found.append(resolved)

    # Check common nested layouts (e.g. robotframework-agent-plugins `plugins/*/skills` or `skills/`)
    plugins_dir = resolved / "plugins"
    if plugins_dir.is_dir():
        for plugin_skills in sorted(plugins_dir.glob("*/skills")):
            if plugin_skills.is_dir():
                found.append(plugin_skills.resolve())

    for sub in ("skills", ".agents/skills"):
        sub_dir = resolved / sub
        if sub_dir.is_dir() and sub_dir.resolve() not in found:
            found.append(sub_dir.resolve())

    return found or [resolved]


def resolve_skill_roots(
    config: ServerConfig,
    root_override: str | Path | None = None,
    *,
    project_roots: Sequence[Path] | None = None,
) -> list[Path]:
    """Resolve layered skill root directories in precedence order:
    1. Project-local `<root>/.robotframework/skills/`, for each of `project_roots` in order
       (default: the one root `config.resolve_root(root_override)`)
    2. External `--skills-dir` / `RF_MCP_SKILLS_DIR` directories
    3. Built-in `src/robotmcp/skills/builtin/`
    """
    roots: list[Path] = []
    if project_roots is None:
        project_roots = [config.resolve_root(root_override)]
    for ws_root in project_roots:
        project_skills = Path(ws_root) / ".robotframework" / "skills"
        if project_skills.is_dir():
            for expanded in _expand_skill_root_candidates(project_skills):
                if expanded not in roots:
                    roots.append(expanded)

    for ext in config.skills_dirs:
        for expanded in _expand_skill_root_candidates(ext):
            if expanded not in roots:
                roots.append(expanded)

    if BUILTIN_SKILLS_ROOT.is_dir() and BUILTIN_SKILLS_ROOT not in roots:
        roots.append(BUILTIN_SKILLS_ROOT)

    return roots


def _parse_frontmatter(md_text: str, default_name: str) -> tuple[str, str]:
    """Extract (name, description) from YAML frontmatter of a SKILL.md file."""
    name = default_name
    description = "Robot Framework skill"
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", md_text, re.DOTALL)
    if not m:
        return name, description
    frontmatter = m.group(1)
    name_match = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
    if name_match:
        name = name_match.group(1).strip().strip("'\"")
    desc_match = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)
    if desc_match:
        description = desc_match.group(1).strip().strip("'\"")
    return name, description


def discover_skill_summaries(roots: Sequence[Path]) -> list[dict[str, str]]:
    """Return deduplicated metadata summaries for all `*/SKILL.md` across `roots`."""
    summaries: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.is_file():
                continue
            folder_name = skill_dir.name
            if folder_name in seen_names:
                continue
            seen_names.add(folder_name)
            try:
                content = skill_md.read_text(encoding="utf-8")
            except OSError:
                continue
            _, desc = _parse_frontmatter(content, folder_name)
            summaries.append(
                {
                    "name": folder_name,
                    "uri": f"skill://{folder_name}/SKILL.md",
                    "description": desc,
                    "path": str(skill_dir.resolve()),
                }
            )
    return summaries


_ROUTING = """\
# rf-mcp: Robot Framework through robotcode (mode: {mode})

| Intent | Tool |
|---|---|
| list tests, tags, suites, files, profiles | rf_discover - never grep .robot files |
{read_row}| why did X fail? / last run's results | rf_results first - never read or grep output.xml |
| check files without running | rf_analyze |
| keyword names, arguments, docs | rf_doc, or .kw / .doc in rf_repl |
| run tests | rf_run with by_longname or tags |
| an existing test fails | rf_debug on its longname, not the REPL; several probes per send |
| explore a page or API, write a new test | rf_repl |
{ui_row}| an error the attached hints do not explain | rf_diagnose |

## Authoring loop
{steps}

Done only when rf_analyze is clean and rf_run passes. Stop the sessions you started, or say which are still open.

Rules: select tests by the longname from rf_discover, never by a single file path; read rf_analyze diagnostics before every run.
"""

_READ_ROW = (
    "| read specs, docs, robot.toml, resources, tests | rf_discover action=read, all paths/globs "
    "in one call - not one Read per file |\n"
)
_PROJECT_SKILLS_NOTE = "Project skills: list_resources shows them once your workspace is known."
_UI_ROW ="| write or fix UI steps (Browser, Selenium, Appium) | read skill://rf-locators/SKILL.md first |\n"
_READ_STEP = "Read every file you need in one rf_discover action=read call (globs work), not file by file."
_STEPS = (
    "Start one rf_repl session and reuse it.",
    "Look keywords up (rf_doc, .kw) instead of guessing.",
    "Send several known steps per send. After a failure fix and retry once; never repeat the same failing call.",
    "Write the files, rf_analyze until clean, rf_run by longname until passing.",
    "Every test creates its own data and passes alone; never pass ids between tests with Set Suite Variable.",
)


def build_server_instructions(
    config: ServerConfig,
    skills: Sequence[dict[str, str]],
    *,
    hint_load_errors: Sequence[str] = (),
) -> str:
    """Build the MCP server instructions: an intent routing table, the authoring loop, and the skill catalog."""
    mode = config.resolved_mode()
    steps = ((_READ_STEP,) if mode == "server" else ()) + _STEPS
    text = _ROUTING.format(
        mode=mode,
        read_row=_READ_ROW if mode == "server" else "",
        ui_row=_UI_ROW if any(s["name"] == "rf-locators" for s in skills) else "",
        steps="\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)),
    )
    if hint_load_errors:
        text += "\nCustom hints not loaded: " + "; ".join(hint_load_errors)[:300] + "\n"
    catalog = ["", "## Skills (read with read_resource)"]
    if mode == "server" and config.workspace_root is None and config.startup_root()[1] == "cwd":
        catalog.append(_PROJECT_SKILLS_NOTE)  # started outside any project, e.g. VS Code user entries in the home folder
    catalog += ["| Skill | URI | About |", "|---|---|---|"]
    for s in skills:
        desc = s["description"]
        if len(desc) > 110:
            desc = desc[:107].rstrip() + "..."
        catalog.append(f"| {s['name']} | {s['uri']} | {desc} |")
    lines = [text.rstrip("\n")]
    for line in catalog:
        if len("\n".join(lines + [line])) > INSTRUCTIONS_MAX_CHARS - 40:
            lines.append("| ... | list_resources | more skills |")
            break
        lines.append(line)
    return "\n".join(lines)


class SkillReadMetrics(Middleware):
    """Count every `skill://` read, whether by resource read or the `read_resource` tool."""

    async def on_read_resource(self, context: MiddlewareContext, call_next) -> Any:
        uri = str(getattr(context.message, "uri", ""))
        match = _SKILL_URI_RE.match(uri)
        if match:
            record_skill_read(match.group(1))
        return await call_next(context)


class ProjectSkillsProvider(Provider):
    """Serve skills for the project roots of each request, so project skills follow the client's workspace.

    The roots follow the tool-call rules: the pinned root, else the client's MCP roots in server mode (all of
    them, in the client's order), else the startup root. Each distinct folder list gets its own
    `SkillsDirectoryProvider`, whose first-name-wins rule keeps project > `--skills-dir` > built-in.
    """

    def __init__(self, config: ServerConfig, client_roots: ClientRoots | None = None) -> None:
        super().__init__()
        self._config = config
        self._client_roots = client_roots
        self._providers: dict[tuple[Path, ...], SkillsDirectoryProvider] = {}

    async def _project_roots(self) -> list[Path]:
        cfg = self._config
        if cfg.workspace_root is not None:
            return [cfg.workspace_root]
        if self._client_roots is not None and cfg.resolved_mode() == "server":
            try:
                ctx = get_context()
            except RuntimeError:  # no request, e.g. a listing at startup
                ctx = None
            if ctx is not None:
                roots = await self._client_roots.roots_for(ctx)
                if roots:
                    return roots
        return [cfg.startup_root()[0]]

    async def _current(self) -> SkillsDirectoryProvider | None:
        roots = tuple(resolve_skill_roots(self._config, project_roots=await self._project_roots()))
        if not roots:
            return None
        provider = self._providers.get(roots)
        if provider is None:
            provider = self._providers[roots] = SkillsDirectoryProvider(roots=list(roots), reload=True)
        return provider

    async def _list_resources(self) -> Sequence[Resource]:
        provider = await self._current()
        return await provider.list_resources() if provider else []

    async def _get_resource(self, uri: str, version: VersionSpec | None = None) -> Resource | None:
        provider = await self._current()
        return await provider.get_resource(uri, version) if provider else None

    async def _list_resource_templates(self) -> Sequence[ResourceTemplate]:
        provider = await self._current()
        return await provider.list_resource_templates() if provider else []

    async def _get_resource_template(self, uri: str, version: VersionSpec | None = None) -> ResourceTemplate | None:
        provider = await self._current()
        return await provider.get_resource_template(uri, version) if provider else None


def mount_skills_provider(
    mcp: FastMCP,
    config: ServerConfig,
    *,
    client_roots: ClientRoots | None = None,
    enable_resources_as_tools: bool = True,
) -> list[dict[str, str]]:
    """Mount the per-request skills provider and `ResourcesAsTools` onto `mcp`; return the startup catalog."""
    roots = resolve_skill_roots(config)
    mcp.add_provider(ProjectSkillsProvider(config, client_roots))
    mcp.add_middleware(SkillReadMetrics())
    if enable_resources_as_tools:
        mcp.add_transform(ResourcesAsTools(mcp))
    return discover_skill_summaries(roots)
