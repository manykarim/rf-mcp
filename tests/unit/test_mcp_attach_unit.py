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
