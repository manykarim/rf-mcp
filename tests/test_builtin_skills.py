"""Built-in skills: reference files are served over MCP and carry the key rules."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastmcp.client import Client

from robotmcp.config import ServerConfig
from robotmcp.server import create_server
from robotmcp.skills.provider import BUILTIN_SKILLS_ROOT

RF_ROBOTCODE = BUILTIN_SKILLS_ROOT / "rf-robotcode"
REFERENCES = sorted(p.name for p in (RF_ROBOTCODE / "references").glob("*.md"))

# Key rules per reference file (P-numbers of the upstream practice checklist).
KEY_RULES = {
    "SKILL.md": [
        "Longnames, not files",  # P9
        "Recorded run first",  # P7
        "Debugger for real tests",  # P6
        "Analyze before running",  # P38
        "Every test stands alone",
        "Done means verified",
    ],
    "triage.md": ["Workflow B", "Workflow A", "Workflow E", "rerunfailed", "merge_inputs", "252"],
    "debugging.md": ["by_longname", "--output NONE --log NONE --report NONE", ".vars --user", "start with a dot", "Write it into", "fallback_command"],
    "repl-authoring.md": ["several lines", "stop at the first failure", "Run Keywords", ".kw", ".save", "Set Suite Variable", "must pass when run alone"],
    "results.md": ["Never read, grep or parse", "output_path", "search=", "extract_dir", "Counts first"],
    "analyze.md": ["bitmask", "collect_unused", "robotcode: ignore", "KeywordNotFound"],
    "config-profiles.md": ["rf-mcp doctor", "profile_show", "relative to the output directory", "RPA", "llms-full.txt"],
    "large-projects.md": ["never grep", "action=\"tags\"", "contains no tests after model modifiers", "Filter at the source"],
}


def test_reference_files_exist() -> None:
    for name in (
        "triage.md",
        "debugging.md",
        "repl-authoring.md",
        "results.md",
        "analyze.md",
        "config-profiles.md",
        "large-projects.md",
        "practices.md",
    ):
        assert name in REFERENCES, name


@pytest.mark.parametrize("name", sorted(KEY_RULES))
def test_key_rules_are_present(name: str) -> None:
    path = RF_ROBOTCODE / name if name == "SKILL.md" else RF_ROBOTCODE / "references" / name
    text = path.read_text(encoding="utf-8")
    for rule in KEY_RULES[name]:
        assert rule in text, f"{name} lacks {rule!r}"


def test_every_upstream_practice_is_mapped() -> None:
    text = (RF_ROBOTCODE / "references" / "practices.md").read_text(encoding="utf-8")
    rows = dict(re.findall(r"^\| (P\d+) \| [^|]+ \| (.+) \|$", text, re.MULTILINE))
    assert sorted(rows, key=lambda p: int(p[1:])) == [f"P{i}" for i in range(1, 47)]
    for practice, where in rows.items():
        files = re.findall(r"[\w-]+\.md", where)
        assert files or "not applicable" in where or "superseded" in where, practice
        for f in files:
            target = RF_ROBOTCODE / f if f == "SKILL.md" else RF_ROBOTCODE / "references" / f
            assert target.is_file(), f"{practice} points to missing {f}"


def test_skill_links_resolve() -> None:
    text = (RF_ROBOTCODE / "SKILL.md").read_text(encoding="utf-8")
    for link in re.findall(r"\]\((references/[^)]+)\)", text):
        assert (RF_ROBOTCODE / link).is_file(), link


async def test_references_are_readable_over_mcp(tmp_path: Path) -> None:
    mcp = create_server(ServerConfig(execution_mode="client", workspace_root=tmp_path))
    async with Client(mcp) as client:
        for name in REFERENCES:
            res = await client.call_tool(
                "read_resource", {"uri": f"skill://rf-robotcode/references/{name}"}
            )
            assert "#" in str(res.data), name
        direct = await client.read_resource("skill://rf-robotcode/references/triage.md")
        assert "Workflow E" in direct[0].text


RF_REQUESTS_RULES = [
    "Import Library    RequestsLibrary",
    "expected_status=any",
    "Every test creates its own data",
    "Never hand ids from one test to the next with `Set Suite Variable`",
    "Bodies: pass a dictionary, never JSON text",
    "`&{{\"k\": \"v\"}}`",
    "data=${{",
    "files=${{",
    "params=${{",
    "Catenate    SEPARATOR=",
    "STRIPS namespaces",
    "Errors come with status 200",
    "`Run Keywords` cannot assign",
]


def test_rf_requests_skill_rules_and_cookbook_suites() -> None:
    text = (BUILTIN_SKILLS_ROOT / "rf-requests" / "SKILL.md").read_text(encoding="utf-8")
    for rule in RF_REQUESTS_RULES:
        assert rule in text, rule
    cookbook = Path(__file__).parent / "cookbook"
    for suite in ("requests_import.robot", "requests_cookbook.robot", "ws_cookbook.robot"):
        assert (cookbook / suite).is_file(), suite


RF_LOCATORS = BUILTIN_SKILLS_ROOT / "rf-locators"
LOCATOR_RULES = {
    "SKILL.md": ["Inspect first, never guess", "Stability order", "data-testid", "Test data by identity", "- iframe",
                 "Closed shadow root", "attachShadow", "Page.addScriptToEvaluateOnNewDocument", ">>>", "shadow_root"],
    "browser.md": ["Strict mode", "force=True", "Press Keys", "role=button[name=", "${None}"],
    "selenium.md": ["Unselect Frame", "Handle Alert", "Switch Window", "Never import Browser and SeleniumLibrary"],
    "shadow-dom-iframes.md": [
        "XPath",
        ">>>",
        "shadow_root",
        "DEEP_QUERY",
        "attachShadow",
        "Page.addScriptToEvaluateOnNewDocument",
        "DOM.getDocument",
        "DOM.performSearch",
        "missing from `Get Aria Snapshot`",
    ],
    "appium.md": ["accessibility_id", "NATIVE_APP", "Get Source"],
    "platynui.md": ["/app:", "control:Frame", "AutomationId"],
    "keyword-intents.md": ["Fill Text", "Input Text", "Select Options By", "Wait Until Element Is Visible", "Pointer Click"],
    "library-chooser.md": ["Browser vs SeleniumLibrary", "AppiumLibrary", "RequestsLibrary", "PlatynUI"],
    "visual-checks.md": ["Take Screenshot", "Capture Page Screenshot", "structured"],
}


@pytest.mark.parametrize("name", sorted(LOCATOR_RULES))
def test_rf_locators_key_rules(name: str) -> None:
    path = RF_LOCATORS / name if name == "SKILL.md" else RF_LOCATORS / "references" / name
    text = path.read_text(encoding="utf-8")
    for rule in LOCATOR_RULES[name]:
        assert rule in text, f"{name} lacks {rule!r}"


@pytest.mark.parametrize("doc", ["references/shadow-dom-iframes.md", "SKILL.md"])
def test_shadow_dom_snippets_are_the_executed_ones(doc: str) -> None:
    """Each code line of the shadow DOM recipes appears verbatim in a cookbook suite."""
    ref = (RF_LOCATORS / doc).read_text(encoding="utf-8")
    suites = "\n".join(p.read_text(encoding="utf-8") for p in (Path(__file__).parent / "cookbook").glob("*.robot"))
    blocks = re.findall(r"```\n(.*?)```", ref, re.DOTALL)
    assert len(blocks) >= (5 if doc != "SKILL.md" else 2)
    for block in blocks:
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("***") or stripped in ("Deep Button", "Read Closed Shadow Via CDP Keeps State", "Closed Shadow Outer HTML"):
                continue
            normalized = re.sub(r"\s{2,}", "  ", stripped)
            assert normalized in re.sub(r"[ \t]{2,}", "  ", suites), f"not executed: {stripped}"


async def test_all_skill_references_are_readable_over_mcp(tmp_path: Path) -> None:
    mcp = create_server(ServerConfig(execution_mode="client", workspace_root=tmp_path))
    async with Client(mcp) as client:
        for skill in ("rf-locators", "rf-requests"):
            res = await client.call_tool("read_resource", {"uri": f"skill://{skill}/SKILL.md"})
            assert skill in str(res.data)
        for ref in sorted((RF_LOCATORS / "references").glob("*.md")):
            res = await client.call_tool("read_resource", {"uri": f"skill://rf-locators/references/{ref.name}"})
            assert "#" in str(res.data), ref.name
