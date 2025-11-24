from robotmcp.prompt.tool_schema import get_openai_tools


def test_filters_non_allowed_tools():
    class Schema:
        def model_dump(self, mode="json"):
            return {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            }

    tool = type(
        "Tool",
        (),
        {"name": "execute_step", "description": "Executes a keyword", "inputSchema": Schema()},
    )()

    tools = get_openai_tools([tool])
    assert len(tools) == 1
    fn = tools[0]["function"]
    assert fn["name"] == "execute_step"
    assert fn["parameters"]["properties"]["message"]["type"] == "string"

    extra_tool = type(
        "Tool",
        (),
        {"name": "custom_tool", "description": "ignored", "inputSchema": Schema()},
    )()
    tools = get_openai_tools([tool, extra_tool])
    assert len(tools) == 2
