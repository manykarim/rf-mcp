"""library-aware-skill-retrieval: skill catalogues and matching them against a project's installed packages."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from robotmcp.config import ServerConfig
from robotmcp.skills import BUILTIN_SKILLS_ROOT
from robotmcp.skills.catalog import (
    BUILTIN_CATALOG,
    builtin_entries,
    catalog_entries,
    match,
    official_entries,
    skills_dir_entries,
)
from robotmcp.skills.official import load_official_skills

ALL_SERVED = ("rf-browser", "rf-selenium", "rf-requests", "rf-language", "rf-locators", "rf-robotcode", "acme-robot")


def test_fixtures_fake_the_official_package_and_a_plugin_repo(fake_bundle, plugin_repo) -> None:
    from robotframework_plugins import bundle  # the fake from conftest

    assert bundle.CATALOG_FORMAT == 1
    assert set(bundle.skills_dirs()) == {"rf-core", "rf-web", "rf-api"}
    assert (fake_bundle["rf-web"] / "rf-browser" / "references" / "locators.md").is_file()
    repo = plugin_repo({"acme": {"skills": {"acme-robot": ()}, "libraries": [{"name": "Acme", "package": "acme-robot"}]}})
    assert (repo / "plugins" / "acme" / "catalog.yaml").is_file()
    assert load_official_skills().skills_dirs == {k: v.resolve() for k, v in fake_bundle.items()}


def test_without_the_package_there_are_no_official_skills() -> None:
    assert sys.modules.get("robotframework_plugins") is None  # conftest hides an installed copy
    assert load_official_skills().skills_dirs == {}
    assert official_entries() == []


def test_supported_version(fake_bundle) -> None:
    [m] = match({"robotframework-browser": "20.1.0"}, official_entries(), ALL_SERVED)
    assert m.to_dict() == {"library": "Browser", "installed": "20.1.0", "skill": "skill://rf-browser/SKILL.md", "supported": True}


def test_version_outside_the_range(fake_bundle) -> None:
    [m] = match({"robotframework-browser": "19.4.0"}, official_entries(), ALL_SERVED)
    assert m.to_dict() == {
        "library": "Browser", "installed": "19.4.0", "skill": "skill://rf-browser/SKILL.md",
        "supported": False, "versions": ">=20,<21",
    }


def test_package_name_normalisation(plugin_repo) -> None:
    repo = plugin_repo({"rf-desktop": {
        "skills": {"rf-platynui": ()},
        "libraries": [{"name": "PlatynUI", "package": "robotframework-PlatynUI", "versions": ">=0.12.0.dev0"}],
    }})
    [m] = match({"robotframework_platynui": "0.12.0"}, skills_dir_entries([repo]), ["rf-platynui"])
    assert (m.library, m.skill, m.supported) == ("PlatynUI", "rf-platynui", True)


def test_catalogue_names_no_skill_for_the_library(plugin_repo) -> None:
    """robotframework-plugins' catalog.yaml lists libraries per plugin, not per skill (as of 37562367)."""
    repo = plugin_repo({
        "rf-web": {
            "skills": {"rf-browser": (), "rf-selenium": ()},
            "libraries": [{"name": "Browser", "package": "robotframework-browser"},
                          {"name": "SeleniumLibrary", "package": "robotframework-seleniumlibrary"}],
        },
        "rf-misc": {"skills": {"widgets": (), "gadgets": ()}, "libraries": [{"name": "Gizmo", "package": "gizmo"}]},
    })
    entries = skills_dir_entries([repo])
    served = ["rf-browser", "rf-selenium", "widgets", "gadgets"]
    # the plugin's skill named after the library
    assert [m.skill for m in match({"robotframework-browser": "20.1.0"}, entries, served)] == ["rf-browser"]
    assert [m.skill for m in match({"robotframework-seleniumlibrary": "6.9"}, entries, served)] == ["rf-selenium"]
    # no skill named after it: every skill of the plugin
    assert [m.skill for m in match({"gizmo": "1"}, entries, served)] == ["gadgets", "widgets"]
    # only served skills count
    assert match({"robotframework-browser": "20.1.0"}, entries, ["rf-selenium"]) == []


def test_each_library_gets_its_first_skill_before_any_second_one(plugin_repo) -> None:
    repo = plugin_repo({
        "rf-api": {"skills": {"rf-requests": (), "rf-requests-extra": ()},
                   "libraries": [{"name": "RequestsLibrary", "package": "robotframework-requests"}]},
        "rf-web": {"skills": {"rf-browser": ()}, "libraries": [{"name": "Browser", "package": "robotframework-browser"}]},
    })
    inventory = {"robotframework-requests": "0.9.7", "robotframework-browser": "20.6.0"}
    matches = match(inventory, skills_dir_entries([repo]), ["rf-requests", "rf-requests-extra", "rf-browser"])
    assert [m.skill for m in matches] == ["rf-requests", "rf-browser", "rf-requests-extra"]


def test_the_pinned_official_catalogue_names_one_skill_per_library() -> None:
    """Against a checkout of MarketSquare/robotframework-plugins (RF_MCP_OFFICIAL_PLUGINS_DIR), as in the evaluation."""
    import os

    import pytest

    base = os.environ.get("RF_MCP_OFFICIAL_PLUGINS_DIR")
    if not base:
        pytest.skip("set RF_MCP_OFFICIAL_PLUGINS_DIR to a checkout of MarketSquare/robotframework-plugins")
    from robotmcp.skills.provider import served_skills

    entries = skills_dir_entries([Path(base)])
    served = served_skills(sorted((Path(base) / "plugins").glob("*/skills")))
    inventory = {"robotframework-browser": "20.6.0", "robotframework-requests": "0.9.7",
                 "robotframework-seleniumlibrary": "6.9.0", "robotframework-appiumlibrary": "3.1"}
    assert [(m.library, m.skill) for m in match(inventory, entries, served)] == [
        ("RequestsLibrary", "rf-requests"), ("AppiumLibrary", "rf-appium"),
        ("Browser", "rf-browser"), ("SeleniumLibrary", "rf-selenium"),
    ]


def test_company_catalogue_in_a_skills_folder(plugin_repo) -> None:
    repo = plugin_repo({"acme": {
        "skills": {"acme-robot": ()},
        "libraries": [{"name": "AcmeLibrary", "package": "acme-robot", "skills": ["acme-robot"]}],
    }})
    cfg = ServerConfig(skills_dirs=[repo])
    [m] = match({"acme-robot": "3.1"}, catalog_entries(cfg), ALL_SERVED)
    assert m.uri == "skill://acme-robot/SKILL.md"


def test_no_official_package_falls_back_to_the_builtin_catalogue() -> None:
    [m] = match({"robotframework-seleniumlibrary": "6.9.0"}, catalog_entries(ServerConfig()), ALL_SERVED)
    assert (m.library, m.uri, m.supported) == ("SeleniumLibrary", "skill://rf-locators/SKILL.md", True)


def test_an_earlier_layer_covers_a_library_for_the_later_ones(fake_bundle) -> None:
    inventory = {"robotframework-browser": "20.1.0", "robotframework-requests": "0.9.7", "robotframework-appiumlibrary": "3.0"}
    matches = match(inventory, catalog_entries(ServerConfig()), ALL_SERVED)
    assert [(m.library, m.skill) for m in matches] == [
        ("Browser", "rf-browser"),
        ("RequestsLibrary", "rf-requests"),
        ("AppiumLibrary", "rf-locators"),  # not in the official catalogue: the built-in one fills in
    ]


def test_builtin_catalogue_ships_with_the_skills() -> None:
    assert BUILTIN_CATALOG.parent == BUILTIN_SKILLS_ROOT and BUILTIN_CATALOG.is_file()
    libs = {lib.package: lib.skills for e in builtin_entries() for lib in e.libraries}
    assert libs["robotframework-browser"] == ("rf-locators",)
    assert libs["robotframework-requests"] == ("rf-requests",)
    for skills in libs.values():
        assert all((BUILTIN_SKILLS_ROOT / s / "SKILL.md").is_file() for s in skills)


def test_an_invalid_catalogue_is_skipped_with_a_warning(plugin_repo, caplog) -> None:
    repo = plugin_repo({"good": {"skills": {"good-skill": ()}, "libraries": [{"name": "G", "package": "g"}]}})
    bad = repo / "plugins" / "bad"
    (bad / "skills").mkdir(parents=True)
    (bad / "catalog.yaml").write_text("libraries: [unclosed\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        entries = skills_dir_entries([repo])
    assert [e.plugin for e in entries] == ["good"]
    assert "bad" in caplog.text and "catalog.yaml" in caplog.text


def test_a_library_without_a_package_is_skipped(plugin_repo, caplog) -> None:
    repo = plugin_repo({"p": {"skills": {"s": ()}, "libraries": [{"name": "NoPackage"}, {"name": "L", "package": "l"}]}})
    with caplog.at_level(logging.WARNING):
        [entry] = skills_dir_entries([repo])
    assert [lib.name for lib in entry.libraries] == ["L"]
    assert "NoPackage" in caplog.text


def test_unparsable_versions_count_as_supported(plugin_repo) -> None:
    repo = plugin_repo({"p": {"skills": {"s": ()}, "libraries": [{"name": "L", "package": "l", "versions": "not a range"}]}})
    [m] = match({"l": "weird"}, skills_dir_entries([repo]), ["s"])
    assert m.supported


def test_skills_dir_without_plugins_has_no_catalogue(tmp_path: Path) -> None:
    assert skills_dir_entries([tmp_path]) == []
