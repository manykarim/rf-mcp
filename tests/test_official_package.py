"""library-aware-skill-retrieval 9.1: rf-mcp against the real robotframework-plugins package, when installed."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
from fastmcp.client import Client

from robotmcp.config import ServerConfig
from robotmcp.server import create_server
from robotmcp.skills.catalog import catalog_entries, match
from robotmcp.skills.official import load_official_skills
from robotmcp.skills.provider import resolve_skill_roots, served_skills

_LEGACY = {"mode": "legacy"} if "mode" in inspect.signature(Client.__init__).parameters else {}


def _installed() -> bool:
    try:
        return importlib.util.find_spec("robotframework_plugins") is not None
    except ValueError:  # hidden by conftest (sys.modules entry None)
        return False


pytestmark = pytest.mark.skipif(not _installed(), reason="robotframework-plugins is not installed")


@pytest.fixture
def real_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo conftest's hiding of the installed package."""
    monkeypatch.delitem(sys.modules, "robotframework_plugins", raising=False)
    load_official_skills.cache_clear()


async def test_the_installed_package_is_served_and_matched(real_bundle, tmp_path: Path) -> None:
    official = load_official_skills()
    assert official.catalog_format == 1 and official.warning is None
    assert {"rf-core", "rf-web", "rf-api"} <= set(official.skills_dirs)

    (tmp_path / "robot.toml").write_text("", encoding="utf-8")
    cfg = ServerConfig(execution_mode="server", workspace_root=tmp_path)
    served = served_skills(resolve_skill_roots(cfg))
    inventory = {"robotframework-browser": "20.6.0", "robotframework-requests": "0.9.7"}
    assert [(m.library, m.skill, m.supported) for m in match(inventory, catalog_entries(cfg), served)] == [
        ("RequestsLibrary", "rf-requests", True), ("Browser", "rf-browser", True),
    ]

    server = create_server(cfg)
    assert "robotcode repl = rf_repl" in (server.instructions or "")
    assert len(server.instructions or "") <= 2500
    async with Client(server, **_LEGACY) as client:
        uris = {str(r.uri) for r in await client.list_resources()}
        assert {"skill://rf-browser/SKILL.md", "skill://rf-robotcode/SKILL.md"} <= uris
        res = await client.call_tool("rf_diagnose", {"error_text": "strict mode violation: x resolved to 2 elements"})
        assert res.structured_content["hints"][0]["skill"] == "skill://rf-browser/references/locators.md"


def test_every_builtin_hint_link_resolves(real_bundle) -> None:
    from robotmcp.hints.builtin_rules import BUILTIN_HINT_RULES
    from robotmcp.skills.pointers import resolve_skill_uri

    served = served_skills(resolve_skill_roots(ServerConfig()))
    for rule in BUILTIN_HINT_RULES:
        for uri in rule.skills:
            assert resolve_skill_uri(uri, served), f"{rule.id}: {uri}"
