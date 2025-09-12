"""Unit tests for hint generation rules that don't require network/e2e.

Validates that certain error patterns produce actionable hints.
"""

from robotmcp.utils.hints import HintContext, generate_hints


def test_hints_for_requests_post_400_json_string():
    ctx = HintContext(
        session_id="s",
        keyword="POST",
        arguments=[
            "https://restful-booker.herokuapp.com/booking",
            "json={'a':1}",
        ],
        error_text="HTTPError: 400 Client Error: Bad Request for url: https://restful-booker.herokuapp.com/booking",
    )
    hints = generate_hints(ctx)
    joined = "\n".join(h.get("title", "") + h.get("message", "") for h in hints)
    assert "payload guidance" in joined or "payload" in joined.lower()
    # Expect examples to include either json=${var} or headers guidance
    example_blob = str(hints)
    assert ("json=${" in example_blob or "headers=${" in example_blob)
    # Should include sessionful flow suggestion as well
    assert "Create Session" in example_blob and ("Post On Session" in example_blob)


def test_hints_for_requests_post_on_session_415():
    ctx = HintContext(
        session_id="s",
        keyword="Post On Session",
        arguments=["rb", "/booking", "json={'a':1}"],
        error_text="415 Unsupported Media Type",
    )
    hints = generate_hints(ctx)
    assert any("payload guidance" in (h.get("title", "") or "") for h in hints)
