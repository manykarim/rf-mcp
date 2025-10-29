import json
from datetime import datetime

import pytest

from robotmcp.core.event_bus import FrontendEvent, event_bus
from robotmcp.frontend.bridge import bridge
from robotmcp.frontend.config import FrontendConfig
from robotmcp.frontend.django_app import get_django_application
from robotmcp.server import execution_engine
from robotmcp.models.execution_models import ExecutionStep


@pytest.fixture(scope="module", autouse=True)
def configure_django():
    """Configure Django settings once for the test module."""

    get_django_application(FrontendConfig(enabled=True, debug=False))


@pytest.fixture
def client():
    from django.test import Client

    return Client()


@pytest.fixture(autouse=True)
def reset_frontend_history():
    # Clear replay buffer and cached history to avoid cross-test bleed
    event_bus._replay.clear()  # type: ignore[attr-defined]
    bridge._history_sessions.clear()  # type: ignore[attr-defined]
    bridge._history_steps.clear()  # type: ignore[attr-defined]
    bridge._history_order.clear()  # type: ignore[attr-defined]
    bridge._history_snapshot_size = 0  # type: ignore[attr-defined]
    bridge._history_snapshot_marker = None  # type: ignore[attr-defined]
    yield
    event_bus._replay.clear()  # type: ignore[attr-defined]
    bridge._history_sessions.clear()  # type: ignore[attr-defined]
    bridge._history_steps.clear()  # type: ignore[attr-defined]
    bridge._history_order.clear()  # type: ignore[attr-defined]
    bridge._history_snapshot_size = 0  # type: ignore[attr-defined]
    bridge._history_snapshot_marker = None  # type: ignore[attr-defined]


@pytest.fixture
def sample_session():
    session = execution_engine.session_manager.create_session("test-session")
    session.imported_libraries.extend(["BuiltIn", "Browser"])
    session.browser_state.browser_type = "chromium"
    session.browser_state.active_library = "Browser"
    session.browser_state.current_url = "https://example.com"  # type: ignore[attr-defined]
    session.variables["CITY"] = "Helsinki"

    step = ExecutionStep(
        step_id="step-1",
        keyword="Log",
        arguments=["Hello from RobotMCP"],
    )
    step.mark_running()
    step.mark_success(result="OK")
    session.steps.append(step)
    session.last_activity = datetime.now()

    assign_step = ExecutionStep(
        step_id="step-2",
        keyword="Get Value",
        arguments=["source"],
    )
    assign_step.mark_running()
    assign_step.mark_success(result="42")
    assign_step.assigned_variables = ["${TOTAL}"]
    assign_step.assignment_type = "single"
    assign_step.variables = {"${TOTAL}": 42}
    session.steps.append(assign_step)
    session.variables["${TOTAL}"] = 42

    # Emulate frontend events so the historical cache can be built without
    # running full keyword execution.
    event_bus.publish_sync(
        FrontendEvent(
            event_type="step_started",
            session_id=session.session_id,
            step_id=step.step_id,
            payload={"keyword": step.keyword, "arguments": step.arguments},
        )
    )
    event_bus.publish_sync(
        FrontendEvent(
            event_type="step_completed",
            session_id=session.session_id,
            step_id=step.step_id,
            payload={"keyword": step.keyword, "arguments": step.arguments},
        )
    )
    event_bus.publish_sync(
        FrontendEvent(
            event_type="step_started",
            session_id=session.session_id,
            step_id=assign_step.step_id,
            payload={"keyword": assign_step.keyword, "arguments": assign_step.arguments},
        )
    )
    event_bus.publish_sync(
        FrontendEvent(
            event_type="step_completed",
            session_id=session.session_id,
            step_id=assign_step.step_id,
            payload={
                "keyword": assign_step.keyword,
                "arguments": assign_step.arguments,
                "assigned_variables": ["${TOTAL}"],
                "assignment_type": "single",
                "assigned_values": {"${TOTAL}": 42},
                "result": "42",
            },
        )
    )

    yield session

    execution_engine.session_manager.remove_session(session.session_id)


def test_sessions_api_returns_active_sessions(client, sample_session):
    response = client.get("/api/sessions/")
    assert response.status_code == 200
    payload = response.json()
    assert "sessions" in payload
    assert any(
        session["session_id"] == sample_session.session_id
        for session in payload["sessions"]
    )


def test_suite_preview_supports_overrides(client, sample_session):
    url = f"/api/sessions/{sample_session.session_id}/suite/"
    response = client.get(url)
    assert response.status_code == 200
    original = response.json()
    assert original["success"] is True
    assert "${TOTAL} =    Get Value" in original["rf_text"]

    payload = {"excluded": ["step-1"]}
    override_response = client.post(
        url,
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert override_response.status_code == 200
    overridden = override_response.json()
    assert overridden["success"] is False or overridden["metadata"]["applied_overrides"][
        "excluded"
    ] == ["step-1"]


def test_suite_preview_reconstructs_after_removal(client, sample_session):
    session_id = sample_session.session_id
    execution_engine.session_manager.remove_session(session_id)

    response = client.get(f"/api/sessions/{session_id}/suite/")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["suite"]
    assert "${TOTAL} =    Get Value" in data["rf_text"]


def test_session_detail_after_removal_uses_history(client, sample_session):
    session_id = sample_session.session_id
    execution_engine.session_manager.remove_session(session_id)

    response = client.get(f"/api/sessions/{session_id}/")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["status"] == "archived"


def test_session_steps_after_removal_from_history(client, sample_session):
    session_id = sample_session.session_id
    execution_engine.session_manager.remove_session(session_id)

    response = client.get(f"/api/sessions/{session_id}/steps/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["steps"]
    assert payload["steps"][0]["keyword"] == "Log"


def test_index_page_renders_dashboard(client, sample_session):
    response = client.get("/")
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "RobotMCP Command Center" in content
    assert "static/frontend/app.js" in content


def test_recent_events_endpoint(client, sample_session):
    response = client.get("/api/events/recent/")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert isinstance(data["events"], list)


def test_static_assets_packaged():
    import importlib.resources as resources

    css_path = resources.files("robotmcp.frontend") / "static" / "frontend" / "base.css"
    assert css_path.is_file(), f"Missing packaged static asset: {css_path}"


def test_session_detail_includes_browser_state(client, sample_session):
    response = client.get(f"/api/sessions/{sample_session.session_id}/")
    assert response.status_code == 200
    data = response.json()
    browser_state = data.get("browser_state") or {}
    assert browser_state.get("browser_type") == "chromium"
    assert browser_state.get("active_library") == "Browser"
    assert browser_state.get("current_url") == "https://example.com"


def test_session_detail_populated_from_executed_steps(client):
    session = execution_engine.session_manager.create_session("detail-from-steps")
    session.imported_libraries.append("Browser")
    session.browser_state.active_library = "browser"

    execution_engine.keyword_executor._apply_state_updates(
        session,
        {"current_browser": {"type": "chromium"}},
    )
    execution_engine.keyword_executor._apply_state_updates(
        session,
        {"current_page": {"url": "https://robotframework.org"}},
    )

    step = ExecutionStep(
        step_id="step-url",
        keyword="Get Text",
        arguments=["css=.title"],
    )
    step.mark_running()
    step.mark_success(result="Products")
    step.assigned_variables = ["${PAGE_TITLE}"]
    step.assignment_type = "single"
    step.variables = {"${PAGE_TITLE}": "Products"}
    session.steps.append(step)
    session.variables["${PAGE_TITLE}"] = "Products"

    response = client.get(f"/api/sessions/{session.session_id}/")
    assert response.status_code == 200
    data = response.json()
    assert data["active_library"] == "browser"
    assert data["browser_state"]["browser_type"] == "chromium"
    assert data["browser_state"]["current_url"] == "https://robotframework.org"
    assert "Browser" in data["imported_libraries"]

    steps = client.get(f"/api/sessions/{session.session_id}/steps/").json()["steps"]
    assigned = next(step for step in steps if step["step_id"] == "step-url")
    assert assigned["assigned_variables"] == ["${PAGE_TITLE}"]
    assert assigned["variables"]["${PAGE_TITLE}"] == "Products"

    execution_engine.session_manager.remove_session(session.session_id)


def test_steps_include_assigned_variables(client, sample_session):
    response = client.get(f"/api/sessions/{sample_session.session_id}/steps/")
    assert response.status_code == 200
    steps = response.json()["steps"]
    assignment = next(step for step in steps if step["step_id"] == "step-2")
    assert assignment["assigned_variables"] == ["${TOTAL}"]
    assert assignment["variables"]["${TOTAL}"] == 42


def test_session_variables_endpoint_exposes_values(client, sample_session):
    response = client.get(f"/api/sessions/{sample_session.session_id}/variables/")
    assert response.status_code == 200
    variables = response.json()["variables"]
    assert variables["${TOTAL}"] == 42
