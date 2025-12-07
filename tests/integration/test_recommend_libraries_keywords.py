import pytest

from tests.fastmcp.test_toolkit import mcp_client  # reuse fixture


@pytest.mark.asyncio
async def test_recommend_libraries_with_keywords(mcp_client):
    res = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": "Open a web page and click a button",
            "context": "web",
            "include_keywords": True,
        },
    )

    assert res.data["success"] is True
    recs = res.data.get("recommendations", [])
    assert recs
    top = recs[0]
    assert "keywords" in top
    assert isinstance(top["keywords"], list)
    assert len(top["keywords"]) > 0
    assert "keyword_hint" in top
    assert "keyword_source" in top
