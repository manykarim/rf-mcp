from types import SimpleNamespace

import pytest

from robotmcp.components.execution.external_rf_client import ExternalRFClient


class DummyHTTPConnection:
    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.requests = []
        self.closed = False

    def request(self, method, path, body, headers):
        self.requests.append((method, path, body, headers))

    def getresponse(self):
        return SimpleNamespace(read=lambda: b'{"success": true, "result": 1}')

    def close(self):
        self.closed = True


def test_external_rf_client_success(monkeypatch):
    dummy = DummyHTTPConnection("127.0.0.1", 7317, 10)

    def factory(host, port, timeout):
        return dummy

    monkeypatch.setattr(
        "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
        factory,
    )

    client = ExternalRFClient(token="secret")
    response = client.run_keyword("Log", ["hello"])

    assert response["success"] is True
    assert dummy.requests
    assert dummy.closed is True


def test_external_rf_client_connection_error(monkeypatch):
    def factory(host, port, timeout):
        raise RuntimeError("cannot connect")

    monkeypatch.setattr(
        "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
        factory,
    )

    client = ExternalRFClient()
    response = client.diagnostics()
    assert response["success"] is False
    assert "connection error" in response["error"]


# --- Tests for Phase 2 new client methods ---


def test_get_page_source_calls_correct_endpoint(monkeypatch):
    """Test that get_page_source posts to the correct endpoint."""
    dummy = DummyHTTPConnection("127.0.0.1", 7317, 10)

    def factory(host, port, timeout):
        return dummy

    monkeypatch.setattr(
        "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
        factory,
    )

    client = ExternalRFClient(token="secret")
    response = client.get_page_source()

    assert response["success"] is True
    assert dummy.requests
    method, path, body, headers = dummy.requests[0]
    assert method == "POST"
    assert path == "/get_page_source"
    assert headers["X-MCP-Token"] == "secret"


def test_get_aria_snapshot_calls_correct_endpoint(monkeypatch):
    """Test that get_aria_snapshot posts to the correct endpoint with params."""
    dummy = DummyHTTPConnection("127.0.0.1", 7317, 10)

    def factory(host, port, timeout):
        return dummy

    monkeypatch.setattr(
        "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
        factory,
    )

    client = ExternalRFClient(token="secret")
    response = client.get_aria_snapshot(selector="css=body", format_type="json")

    assert response["success"] is True
    assert dummy.requests
    method, path, body, headers = dummy.requests[0]
    assert method == "POST"
    assert path == "/get_aria_snapshot"
    import json
    payload = json.loads(body)
    assert payload["selector"] == "css=body"
    assert payload["format"] == "json"


def test_get_aria_snapshot_uses_defaults(monkeypatch):
    """Test that get_aria_snapshot uses default selector and format."""
    dummy = DummyHTTPConnection("127.0.0.1", 7317, 10)

    def factory(host, port, timeout):
        return dummy

    monkeypatch.setattr(
        "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
        factory,
    )

    client = ExternalRFClient(token="secret")
    response = client.get_aria_snapshot()

    assert response["success"] is True
    import json
    payload = json.loads(dummy.requests[0][2])
    assert payload["selector"] == "css=html"
    assert payload["format"] == "yaml"


def test_get_session_info_calls_correct_endpoint(monkeypatch):
    """Test that get_session_info posts to the correct endpoint."""
    dummy = DummyHTTPConnection("127.0.0.1", 7317, 10)

    def factory(host, port, timeout):
        return dummy

    monkeypatch.setattr(
        "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
        factory,
    )

    client = ExternalRFClient(token="secret")
    response = client.get_session_info()

    assert response["success"] is True
    assert dummy.requests
    method, path, body, headers = dummy.requests[0]
    assert method == "POST"
    assert path == "/get_session_info"
