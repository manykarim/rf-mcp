"""Unit tests ensuring Windows desktop automation support is wired up."""

import sys

from robotmcp.config.library_registry import get_library_config
from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
from robotmcp.components.library_recommender import LibraryRecommender
from robotmcp.components.execution.session_manager import SessionManager
from robotmcp.models.session_models import ExecutionSession, SessionType, PlatformType


def test_library_registry_includes_rpa_windows() -> None:
    config = get_library_config("RPA.Windows")
    assert config is not None
    assert config.package_name == "rpaframework-windows"
    assert "desktop" in {cat.value for cat in config.categories}
    assert config.load_priority == 18


def test_session_detects_desktop_profile() -> None:
    session = ExecutionSession(session_id="desktop-test")
    scenario = "Automate a Windows desktop app: open Notepad, type text, and validate the window title."
    session.configure_from_scenario(scenario)
    SessionManager().initialize_desktop_session(session, scenario)

    assert session.session_type == SessionType.DESKTOP_AUTOMATION
    assert session.platform_type == PlatformType.DESKTOP

    if sys.platform.startswith("win"):
        assert session.desktop_supported is True
        libraries_to_load = session.get_libraries_to_load()
        assert "RPA.Windows" in libraries_to_load
        assert session.search_order[0] == "RPA.Windows"
    else:
        assert session.desktop_supported is False
        assert "RPA.Windows" not in session.get_search_order()


def test_session_manager_detects_desktop_platform() -> None:
    manager = SessionManager()
    scenario = "Launch calculator desktop app and capture window tree"
    platform = manager.detect_platform_from_scenario(scenario, context="desktop")
    assert platform == PlatformType.DESKTOP


def test_recommender_prioritizes_rpa_windows() -> None:
    recommender = LibraryRecommender()
    result = recommender.recommend_libraries(
        scenario="Validate data entry in a Windows desktop application",
        context="desktop",
        max_recommendations=3,
    )
    assert result["success"] is True
    recommended_names = [rec["library_name"] for rec in result["recommendations"]]
    assert "RPA.Windows" in recommended_names
    assert recommended_names[0] == "RPA.Windows"


def test_keyword_discovery_exposes_rpa_windows_keywords() -> None:
    orchestrator = get_keyword_discovery()
    # Ensure library is loaded into discovery
    orchestrator.library_manager.load_library_on_demand(
        "RPA.Windows", orchestrator.keyword_discovery
    )
    keyword_info = orchestrator.find_keyword(
        "List Windows", session_libraries=["RPA.Windows"]
    )
    assert keyword_info is not None
    assert keyword_info.library == "RPA.Windows"
