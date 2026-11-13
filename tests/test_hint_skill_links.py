"""library-aware-skill-retrieval: hints link to skill sections; the server attaches the first served link."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
from fastmcp.client import Client

from robotmcp.config import ServerConfig
from robotmcp.hints.builtin_rules import BUILTIN_HINT_RULES
from robotmcp.hints.loader import load_custom_rules_with_errors
from robotmcp.server import create_server
from robotmcp.skills import BUILTIN_SKILLS_ROOT
from robotmcp.skills.pointers import resolve_skill_uri
from robotmcp.skills.provider import served_skills
from tests.conftest import make_skill

_LEGACY = {"mode": "legacy"} if "mode" in inspect.signature(Client.__init__).parameters else {}

STRICT = "strict mode violation: getByText('Add') resolved to 2 elements"

# A checkout of MarketSquare/robotframework-plugins at the commit rf-mcp's links were written against.
OFFICIAL_PLUGINS_DIR = os.environ.get("RF_MCP_OFFICIAL_PLUGINS_DIR")
OFFICIAL_PLUGINS_COMMIT = "37562367fc8c330db778ecadec287cef24054fc5"


async def _diagnose(server, error_text: str, **args) -> dict:
    async with Client(server, **_LEGACY) as client:
        res = await client.call_tool("rf_diagnose", {"error_text": error_text, **args})
    return res.structured_content


def _hint(data: dict, hint_id: str) -> dict:
    hints = {h["id"]: h for h in data["hints"]}
    assert all("skills" not in h for h in data["hints"])  # the candidate list never reaches the agent
    return hints[hint_id]


async def test_official_skill_served(tmp_path: Path, fake_bundle) -> None:
    data = await _diagnose(create_server(ServerConfig(execution_mode="server", workspace_root=tmp_path)), STRICT)
    assert _hint(data, "browser-strict-mode-violation")["skill"] == "skill://rf-browser/references/locators.md"


async def test_official_skills_absent(tmp_path: Path) -> None:
    data = await _diagnose(create_server(ServerConfig(execution_mode="server", workspace_root=tmp_path)), STRICT)
    assert _hint(data, "browser-strict-mode-violation")["skill"] == "skill://rf-locators/references/browser.md"


async def test_client_mode_resolves_links_too(tmp_path: Path) -> None:
    data = await _diagnose(create_server(ServerConfig(execution_mode="client", workspace_root=tmp_path)), STRICT)
    assert _hint(data, "browser-strict-mode-violation")["skill"] == "skill://rf-locators/references/browser.md"


def _company(tmp_path: Path, links: str) -> tuple[Path, Path]:
    project = tmp_path / "shop"
    (project / ".robotframework").mkdir(parents=True)
    (project / "robot.toml").write_text("", encoding="utf-8")
    (project / ".robotframework" / "hints.toml").write_text(
        "[[hints]]\n"
        'id = "corp-auth-expired"\n'
        'error_pattern = "CorpAuthExpiredError"\n'
        'message = "Refresh the token."\n'
        f"skills = {links}\n"
        "relevance = 99\n",
        encoding="utf-8",
    )
    company = tmp_path / "acme-skills"
    make_skill(company, "acme-auth", "tokens.md")
    return project, company


async def test_project_rule_links_a_company_skill(tmp_path: Path) -> None:
    project, company = _company(tmp_path, '["skill://acme-auth/references/tokens.md"]')
    server = create_server(ServerConfig(execution_mode="server", skills_dirs=[company]))
    data = await _diagnose(server, "CorpAuthExpiredError: token expired", root=str(project))
    assert _hint(data, "corp-auth-expired")["skill"] == "skill://acme-auth/references/tokens.md"


async def test_linked_skill_not_served(tmp_path: Path) -> None:
    project, _company_dir = _company(tmp_path, '["skill://acme-auth/SKILL.md", "skill://acme-auth/references/gone.md"]')
    server = create_server(ServerConfig(execution_mode="server"))  # the company folder is not configured
    data = await _diagnose(server, "CorpAuthExpiredError: token expired", root=str(project))
    hint = _hint(data, "corp-auth-expired")
    assert "skill" not in hint
    assert hint["title"] == "corp-auth-expired" and hint["message"] == "Refresh the token." and hint["relevance"] == 99


async def test_switched_off_links_nothing(tmp_path: Path, fake_bundle) -> None:
    cfg = ServerConfig(execution_mode="server", workspace_root=tmp_path, skill_pointers=False)
    data = await _diagnose(create_server(cfg), STRICT)
    assert "skill" not in _hint(data, "browser-strict-mode-violation")


def test_resolve_skill_uri_checks_the_reference_file(tmp_path: Path) -> None:
    make_skill(tmp_path / "skills", "s", "a.md")
    served = served_skills([tmp_path / "skills"])
    assert resolve_skill_uri("skill://s/SKILL.md", served)
    assert resolve_skill_uri("skill://s/references/a.md", served)
    assert not resolve_skill_uri("skill://s/references/b.md", served)
    assert not resolve_skill_uri("skill://s/../../etc/passwd", served)
    assert not resolve_skill_uri("skill://other/SKILL.md", served)
    assert not resolve_skill_uri("https://example.com", served)


def test_hints_toml_rejects_a_link_that_is_not_a_skill_uri(tmp_path: Path) -> None:
    hints = tmp_path / "hints.toml"
    hints.write_text(
        '[[hints]]\nid = "bad-link"\nerror_pattern = "X"\nmessage = "m"\nskills = ["docs/tokens.md"]\n\n'
        '[[hints]]\nid = "good"\nerror_pattern = "Y"\nmessage = "m"\nskills = "skill://a/SKILL.md"\n',
        encoding="utf-8",
    )
    rules, errors = load_custom_rules_with_errors(hints)
    assert [r.id for r in rules] == ["good"] and rules[0].skills == ["skill://a/SKILL.md"]
    assert len(errors) == 1 and "bad-link" in errors[0] and "skill://" in errors[0]


# --- 6.2 the built-in links stay valid ---------------------------------------------------------------------------

_LINKED = [(rule.id, uri) for rule in BUILTIN_HINT_RULES for uri in rule.skills]
_BUILTIN_SERVED = served_skills([BUILTIN_SKILLS_ROOT])


def _official_served() -> dict[str, Path]:
    base = Path(OFFICIAL_PLUGINS_DIR or "")
    return served_skills(sorted((base / "plugins").glob("*/skills")))


def test_every_linked_rule_has_a_builtin_fallback() -> None:
    linked = {rule.id for rule in BUILTIN_HINT_RULES if rule.skills}
    assert len(linked) >= 10
    for rule in BUILTIN_HINT_RULES:
        if rule.skills:
            assert any(resolve_skill_uri(u, _BUILTIN_SERVED) for u in rule.skills), rule.id


@pytest.mark.skipif(not OFFICIAL_PLUGINS_DIR, reason=f"set RF_MCP_OFFICIAL_PLUGINS_DIR to a checkout of "
                    f"MarketSquare/robotframework-plugins at {OFFICIAL_PLUGINS_COMMIT}")
@pytest.mark.parametrize(("rule_id", "uri"), _LINKED, ids=[f"{r}:{u}" for r, u in _LINKED])
def test_builtin_links_resolve(rule_id: str, uri: str) -> None:
    official = _official_served()
    assert official, f"no plugins/*/skills in {OFFICIAL_PLUGINS_DIR}"
    assert resolve_skill_uri(uri, official) or resolve_skill_uri(uri, _BUILTIN_SERVED), f"{rule_id}: {uri}"


def test_builtin_links_resolve_in_the_builtin_skills_when_they_name_them() -> None:
    builtin_named = [(r, u) for r, u in _LINKED if u.split("/")[2] in ("rf-locators", "rf-robotcode")]
    assert builtin_named
    for rule_id, uri in builtin_named:
        assert resolve_skill_uri(uri, _BUILTIN_SERVED), f"{rule_id}: {uri}"


# --- 6.3 the documented examples load ----------------------------------------------------------------------------

_CONFIGURATION = Path(__file__).resolve().parent.parent / "docs" / "CONFIGURATION.md"


def _block(lang: str, contains: str) -> str:
    import re

    blocks = re.findall(rf"```{lang}\n(.*?)```", _CONFIGURATION.read_text(encoding="utf-8"), re.DOTALL)
    return next(b for b in blocks if contains in b)


def test_the_documented_hints_toml_example_loads(tmp_path: Path) -> None:
    hints = tmp_path / "hints.toml"
    hints.write_text(_block("toml", "skills = "), encoding="utf-8")
    rules, errors = load_custom_rules_with_errors(hints)
    assert errors == [] and rules[0].skills == ["skill://corp-auth/references/sso.md"]


def test_the_documented_company_catalogue_loads(tmp_path: Path) -> None:
    from robotmcp.skills.catalog import match, skills_dir_entries

    plugin = tmp_path / "plugins" / "acme"
    make_skill(plugin / "skills", "acme-robot")
    (plugin / "catalog.yaml").write_text(_block("yaml", "acme-robot"), encoding="utf-8")
    [m] = match({"acme-robot": "3.2"}, skills_dir_entries([tmp_path]), ["acme-robot"])
    assert (m.library, m.uri, m.supported) == ("AcmeLibrary", "skill://acme-robot/SKILL.md", True)
