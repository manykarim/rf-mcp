import queue

import pytest

from robotmcp.attach import mcp_attach


class StubBuiltIn:
    def __init__(self, storage):
        self.storage = storage

    def set_test_variable(self, name, value):
        self.storage[name] = value

    def run_keyword(self, name, *args):
        return {"keyword": name, "args": list(args)}

    def log_to_console(self, _message):
        return None


def test_assign_variables_handles_scalars(monkeypatch):
    storage = {}
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: StubBuiltIn(storage))
    attach = mcp_attach.McpAttach()

    assigned = attach._assign_variables("${RESULT}", 42)
    assert assigned == {"${RESULT}": 42}
    assert storage["${RESULT}"] == 42


def test_assign_variables_multiple_targets_with_missing_values(monkeypatch):
    storage = {}
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: StubBuiltIn(storage))
    attach = mcp_attach.McpAttach()

    assigned = attach._assign_variables(["RESULT", "SECOND"], ["one"])
    assert assigned == {"${RESULT}": "one", "${SECOND}": None}
    assert storage["${RESULT}"] == "one"
    assert storage["${SECOND}"] is None


def test_parse_run_keyword_payload_validation():
    attach = mcp_attach.McpAttach()
    with pytest.raises(ValueError):
        attach._parse_run_keyword_payload({"name": "Log", "args": "not-a-list"})


def test_execute_command_unknown(monkeypatch):
    storage = {}
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: StubBuiltIn(storage))
    attach = mcp_attach.McpAttach()
    response = attach._execute_command("does-not-exist", {})
    assert response["success"] is False
    assert "unknown verb" in response["error"]


def test_execute_command_stop_sets_flag(monkeypatch):
    storage = {}
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: StubBuiltIn(storage))
    attach = mcp_attach.McpAttach()
    attach._stop_flag = False
    response = attach._execute_command("stop", {})
    assert response["success"] is True
    assert attach._stop_flag is True


def test_mcp_process_once_handles_queue(monkeypatch):
    attach = mcp_attach.McpAttach()

    def fake_execute(self, verb, payload):
        return {"success": True, "verb": verb, "payload": payload}

    monkeypatch.setattr(mcp_attach.McpAttach, "_execute_command", fake_execute)

    replyq: "queue.Queue" = queue.Queue()
    attach._cmdq.put(mcp_attach._Command("diagnostics", {"a": 1}, replyq))

    attach.MCP_Process_Once()

    response = replyq.get_nowait()
    assert response["success"] is True
    assert response["verb"] == "diagnostics"


# --- Tests for new Phase 2 verbs ---


class StubBuiltInWithKeywords:
    """Extended stub that simulates keyword execution for page source tests."""

    def __init__(self, keyword_responses=None, storage=None):
        self.keyword_responses = keyword_responses or {}
        self.storage = storage if storage is not None else {}
        self.called_keywords = []

    def set_test_variable(self, name, value):
        self.storage[name] = value

    def run_keyword(self, name, *args):
        self.called_keywords.append((name, args))
        if name in self.keyword_responses:
            response = self.keyword_responses[name]
            if isinstance(response, Exception):
                raise response
            return response
        raise RuntimeError(f"No keyword with name '{name}' found")

    def log_to_console(self, _message):
        return None

    def get_variables(self):
        return self.storage


class StubExecutionContext:
    """Mock RF execution context for session info tests."""

    def __init__(self, libraries=None, suite_name=None, test_name=None):
        self.namespace = StubNamespace(libraries or {})
        self.suite = StubSuite(suite_name)
        self.test = StubTest(test_name)


class StubNamespace:
    def __init__(self, libraries):
        self.libraries = libraries


class StubSuite:
    def __init__(self, name):
        self.name = name


class StubTest:
    def __init__(self, name):
        self.name = name


def test_execute_command_get_page_source_dispatches(monkeypatch):
    """Test that get_page_source verb is dispatched correctly."""
    attach = mcp_attach.McpAttach()

    def fake_get_page_source(self, payload):
        return {"success": True, "result": "<html></html>", "library": "Browser"}

    monkeypatch.setattr(mcp_attach.McpAttach, "_get_page_source", fake_get_page_source)

    response = attach._execute_command("get_page_source", {})
    assert response["success"] is True
    assert response["library"] == "Browser"


def test_execute_command_get_aria_snapshot_dispatches(monkeypatch):
    """Test that get_aria_snapshot verb is dispatched correctly."""
    attach = mcp_attach.McpAttach()

    def fake_get_aria_snapshot(self, payload):
        return {
            "success": True,
            "result": "- button: Click me",
            "format": "yaml",
            "selector": "css=html",
            "library": "Browser",
        }

    monkeypatch.setattr(mcp_attach.McpAttach, "_get_aria_snapshot", fake_get_aria_snapshot)

    response = attach._execute_command("get_aria_snapshot", {"selector": "css=body"})
    assert response["success"] is True
    assert response["format"] == "yaml"


def test_execute_command_get_session_info_dispatches(monkeypatch):
    """Test that get_session_info verb is dispatched correctly."""
    attach = mcp_attach.McpAttach()

    def fake_get_session_info(self, payload):
        return {
            "success": True,
            "result": {
                "context_active": True,
                "variable_count": 5,
                "suite_name": "Test Suite",
                "test_name": "Test Case",
                "libraries": ["Browser", "BuiltIn"],
            },
        }

    monkeypatch.setattr(mcp_attach.McpAttach, "_get_session_info", fake_get_session_info)

    response = attach._execute_command("get_session_info", {})
    assert response["success"] is True
    assert response["result"]["context_active"] is True


def test_get_page_source_tries_browser_first(monkeypatch):
    """Test that _get_page_source tries Browser Library keyword first."""
    stub = StubBuiltInWithKeywords(
        keyword_responses={"Get Page Source": "<html>Browser HTML</html>"}
    )
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

    attach = mcp_attach.McpAttach()
    response = attach._get_page_source({})

    assert response["success"] is True
    assert response["result"] == "<html>Browser HTML</html>"
    assert response["library"] == "Browser"
    assert ("Get Page Source", ()) in stub.called_keywords


def test_get_page_source_falls_back_to_selenium(monkeypatch):
    """Test that _get_page_source falls back to SeleniumLibrary when Browser fails."""
    stub = StubBuiltInWithKeywords(
        keyword_responses={
            "Get Page Source": RuntimeError("No keyword with name"),
            "Get Source": "<html>Selenium HTML</html>",
        }
    )
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

    attach = mcp_attach.McpAttach()
    response = attach._get_page_source({})

    assert response["success"] is True
    assert response["result"] == "<html>Selenium HTML</html>"
    assert response["library"] == "SeleniumLibrary"


def test_get_page_source_fails_when_no_library(monkeypatch):
    """Test that _get_page_source returns error when no browser library available."""
    stub = StubBuiltInWithKeywords(keyword_responses={})
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

    attach = mcp_attach.McpAttach()
    response = attach._get_page_source({})

    assert response["success"] is False
    assert "No browser library loaded" in response["error"]


def test_get_aria_snapshot_success(monkeypatch):
    """Test _get_aria_snapshot with successful Browser Library response."""
    stub = StubBuiltInWithKeywords(
        keyword_responses={"Get Aria Snapshot": "- button: Submit"}
    )
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

    attach = mcp_attach.McpAttach()
    response = attach._get_aria_snapshot({"selector": "css=body", "format": "yaml"})

    assert response["success"] is True
    assert response["result"] == "- button: Submit"
    assert response["format"] == "yaml"
    assert response["selector"] == "css=body"
    assert response["library"] == "Browser"


def test_get_aria_snapshot_uses_defaults(monkeypatch):
    """Test _get_aria_snapshot uses default selector and format."""
    stub = StubBuiltInWithKeywords(
        keyword_responses={"Get Aria Snapshot": "- document"}
    )
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

    attach = mcp_attach.McpAttach()
    response = attach._get_aria_snapshot({})

    assert response["success"] is True
    assert response["selector"] == "css=html"
    assert response["format"] == "yaml"


def test_get_aria_snapshot_not_available(monkeypatch):
    """Test _get_aria_snapshot when Browser Library not loaded."""
    stub = StubBuiltInWithKeywords(keyword_responses={})
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

    attach = mcp_attach.McpAttach()
    response = attach._get_aria_snapshot({})

    assert response["success"] is False
    assert "Aria snapshot not available" in response["error"]
    assert "Browser Library (Playwright)" in response["note"]


def test_get_session_info_success(monkeypatch):
    """Test _get_session_info with active RF context."""
    stub = StubBuiltInWithKeywords(
        storage={"${VAR1}": "value1", "${VAR2}": "value2"}
    )
    ctx = StubExecutionContext(
        libraries={"Browser": None, "BuiltIn": None},
        suite_name="My Test Suite",
        test_name="My Test Case",
    )
    monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

    # Mock the EXECUTION_CONTEXTS module-level object with one that has our ctx
    class MockContexts:
        current = ctx

    monkeypatch.setattr(mcp_attach, "EXECUTION_CONTEXTS", MockContexts())

    attach = mcp_attach.McpAttach()
    response = attach._get_session_info({})

    assert response["success"] is True
    result = response["result"]
    assert result["context_active"] is True
    assert result["variable_count"] == 2
    assert result["suite_name"] == "My Test Suite"
    assert result["test_name"] == "My Test Case"
    assert "Browser" in result["libraries"]
    assert "BuiltIn" in result["libraries"]


def test_get_session_info_no_context(monkeypatch):
    """Test _get_session_info when no active RF context."""

    # Mock the EXECUTION_CONTEXTS module-level object with None current
    class MockContexts:
        current = None

    monkeypatch.setattr(mcp_attach, "EXECUTION_CONTEXTS", MockContexts())

    attach = mcp_attach.McpAttach()
    response = attach._get_session_info({})

    assert response["success"] is False
    assert "No active execution context" in response["error"]
