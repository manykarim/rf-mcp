import types
import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.session_models import ExecutionSession


class DummyBrowserManager:
    def set_active_library(self, *args, **kwargs):
        return None

    def update_browser_state(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_execute_keyword_uses_plugin_hints():
    executor = KeywordExecutor()
    session = ExecutionSession(session_id="sess")

    async def fake_internal(self, session, step, browser_manager, library_prefix=None, resolved_arguments=None):
        return {
            "success": False,
            "error": "HTTPError: 500 Server Error",
            "library_name": "RequestsLibrary",
            "hints": [],
        }

    executor._execute_keyword_internal = types.MethodType(fake_internal, executor)

    def fake_generate(self, library_name, session, keyword, arguments, error_text):
        return [
            {"title": "Plugin Hint", "message": "Use RequestsLibrary session", "examples": []}
        ]

    executor.plugin_manager.generate_failure_hints = types.MethodType(
        fake_generate, executor.plugin_manager
    )

    result = await executor.execute_keyword(
        session,
        "POST",
        ["https://restful-booker.herokuapp.com/booking"],
        DummyBrowserManager(),
    )

    assert result["hints"][0]["title"] == "Plugin Hint"
