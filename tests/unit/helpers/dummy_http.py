"""Shared DummyHTTPConnection test double for ExternalRFClient tests.

Consolidates 3+ duplicate implementations used across test files.
"""

import json
from io import BytesIO


class DummyResponse:
    """Minimal HTTP response stand-in."""

    def __init__(self, status=200, body=None):
        self.status = status
        self._body = body or {}

    def read(self):
        return json.dumps(self._body).encode("utf-8")


class DummyHTTPConnection:
    """Stand-in for http.client.HTTPConnection.

    Records requests and returns configured responses for
    ExternalRFClient integration testing without real HTTP.
    """

    def __init__(self, host="localhost", port=8270, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.requests = []
        self._response = DummyResponse()
        self._connected = False

    def connect(self):
        self._connected = True

    def close(self):
        self._connected = False

    def request(self, method, path, body=None, headers=None):
        parsed_body = json.loads(body) if isinstance(body, str) else body
        self.requests.append({
            "method": method,
            "path": path,
            "body": parsed_body,
            "headers": headers or {},
        })

    def getresponse(self):
        return self._response

    def set_response(self, status=200, body=None):
        """Configure the response to return."""
        self._response = DummyResponse(status=status, body=body)
        return self

    def get_last_request(self):
        """Get the most recent request, or None."""
        return self.requests[-1] if self.requests else None
