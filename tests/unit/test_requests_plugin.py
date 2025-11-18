from robotmcp.plugins.builtin.requests_plugin import RequestsLibraryPlugin
from robotmcp.models.session_models import ExecutionSession


def test_requests_plugin_generates_http_error_hint():
    plugin = RequestsLibraryPlugin()
    session = ExecutionSession(session_id="sess")
    hints = plugin.generate_failure_hints(
        session,
        "POST",
        ["https://restful-booker.herokuapp.com/booking"],
        "HTTPError: 500 Server Error",
    )
    assert hints, "Expected plugin to return hints for HTTPError"
    assert any("http" in h["title"].lower() for h in hints)
    assert hints[0]["examples"], "Hints should include examples"


def test_requests_plugin_missing_schema_hint():
    plugin = RequestsLibraryPlugin()
    session = ExecutionSession(session_id="sess")
    hints = plugin.generate_failure_hints(
        session,
        "RequestsLibrary.GET",
        ["restful-booker", "/booking/1"],
        "MissingSchema: Invalid URL 'restful-booker'",
    )
    assert hints, "Expected plugin to hint when URL is missing scheme"
    assert any("url" in h["message"].lower() for h in hints)
