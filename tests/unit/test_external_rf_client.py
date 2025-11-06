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
