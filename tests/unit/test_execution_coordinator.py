from unittest.mock import AsyncMock

import pytest

from robotmcp.components.execution.execution_coordinator import ExecutionCoordinator
from robotmcp.models.session_models import ExecutionSession


class DummySessionManager:
    def __init__(self, session: ExecutionSession):
        self.session = session

    def get_or_create_session(self, session_id: str) -> ExecutionSession:
        assert session_id == self.session.session_id
        return self.session

    def get_session(self, session_id: str) -> ExecutionSession:
        assert session_id == self.session.session_id
        return self.session


@pytest.mark.asyncio
async def test_execute_step_happy_path(monkeypatch):
    session = ExecutionSession(session_id="coordinator")
    session.libraries_loaded = False
    coordinator = ExecutionCoordinator()
    coordinator.session_manager = DummySessionManager(session)
    coordinator._load_session_libraries = lambda s: setattr(s, "libraries_loaded", True)

    captured_arguments = {}

    async def fake_execute_keyword(**kwargs):
        captured_arguments.update(kwargs)
        return {"success": True, "keyword": kwargs["keyword"], "arguments": kwargs["arguments"]}

    coordinator.keyword_executor.execute_keyword = fake_execute_keyword
    coordinator._convert_locators_in_arguments = lambda args, sess: args + ["converted"]

    result = await coordinator.execute_step(
        keyword="Log",
        arguments=["Hello"],
        session_id="coordinator",
        use_context=True,
    )

    assert result["success"] is True
    assert captured_arguments["arguments"] == ["Hello", "converted"]
    assert session.libraries_loaded is True


@pytest.mark.asyncio
async def test_execute_step_handles_errors(monkeypatch):
    session = ExecutionSession(session_id="error")
    session.libraries_loaded = True
    coordinator = ExecutionCoordinator()
    coordinator.session_manager = DummySessionManager(session)
    coordinator.keyword_executor.execute_keyword = AsyncMock(side_effect=RuntimeError("boom"))

    result = await coordinator.execute_step(
        keyword="Log",
        arguments=[],
        session_id="error",
    )

    assert result["success"] is False
    assert "Execution coordinator error" in result["error"]


@pytest.mark.asyncio
async def test_get_page_source_delegates(monkeypatch):
    session = ExecutionSession(session_id="state")
    session.libraries_loaded = True
    coordinator = ExecutionCoordinator()
    coordinator.session_manager = DummySessionManager(session)

    async def fake_get_page_source(**kwargs):
        return {"success": True, "session_id": kwargs["session"].session_id}

    coordinator.page_source_service.get_page_source = fake_get_page_source  # type: ignore[assignment]

    result = await coordinator.get_page_source(session_id="state")

    assert result["success"] is True
    assert result["session_id"] == "state"


@pytest.mark.asyncio
async def test_get_page_source_missing_session():
    coordinator = ExecutionCoordinator()
    coordinator.session_manager = DummySessionManager(ExecutionSession("dummy"))
    coordinator.session_manager.get_session = lambda _sid: None  # type: ignore[assignment]

    result = await coordinator.get_page_source(session_id="missing")

    assert result["success"] is False
    assert "not found" in result["error"]
