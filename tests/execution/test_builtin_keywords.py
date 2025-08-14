import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# Ensure src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.session_models import ExecutionSession


@pytest.mark.asyncio
async def test_builtin_keywords_delegated():
    session = ExecutionSession(session_id="s2")
    executor = KeywordExecutor()

    browser_manager = MagicMock()
    browser_manager.get_active_browser_library.return_value = (None, None)

    exec_mock = AsyncMock(return_value={"success": True})
    with patch("robotmcp.core.dynamic_keyword_orchestrator.DynamicKeywordDiscovery.execute_keyword", exec_mock):
        await executor.execute_keyword(session, "Log", ["Hello"], browser_manager)

    # Built-in keywords should go through dynamic discovery
    exec_mock.assert_called_once()
