from types import SimpleNamespace

from mcp.types import TextContent, Tool

from robotmcp.prompt.config import PromptRuntimeConfig
from robotmcp.prompt.llm_client import LlmResponse, LlmToolCall
from robotmcp.prompt.runner import PromptRunner


class FakeLlmClient:
    def __init__(self):
        self.calls = 0

    def complete_chat(self, **kwargs):  # noqa: ANN003 - signature mirrors protocol
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                content="",
                tool_calls=[
                    LlmToolCall(
                        id="call-1",
                        name="execute_step",
                        arguments={"keyword": "Log", "arguments": ["hello"]},
                    )
                ],
                raw_response=None,
            )
        return LlmResponse(content="All done", tool_calls=[], raw_response=None)


def fake_client_builder():
    schema = {"type": "object"}
    tool = Tool(name="execute_step", description="run step", inputSchema=schema)

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def list_tools(self):
            return [tool]

        async def get_prompt(self, prompt_key, arguments):
            _ = (prompt_key, arguments)
            message = SimpleNamespace(role="system", content="Use execute_step")
            return SimpleNamespace(messages=[message])

        async def call_tool_mcp(self, name, arguments):
            assert name == "execute_step"
            assert arguments["session_id"] == "suite"
            return SimpleNamespace(
                isError=False,
                content=[TextContent(type="text", text="Step executed")],
                structuredContent=None,
            )

    return _Client()


def test_prompt_runner_happy_path(monkeypatch):
    config = PromptRuntimeConfig(api_key="key", model="model")
    runner = PromptRunner(
        llm_client_factory=lambda _: FakeLlmClient(),
        client_builder=fake_client_builder,
    )

    result = runner.run(
        scenario="Log hello",
        prompt_key="automate",
        session_id="suite",
        config=config,
    )

    assert result.success is True
    assert result.final_response == "All done"
    assert result.executed_calls
    assert result.executed_calls[0].name == "execute_step"


def test_chat_runner(monkeypatch):
    config = PromptRuntimeConfig(api_key="key", model="model")
    runner = PromptRunner(
        llm_client_factory=lambda _: FakeLlmClient(),
        client_builder=fake_client_builder,
    )

    result = runner.run_chat(
        message="Open website xyz.com and add two items",
        session_id="suite",
        config=config,
    )

    assert result.success
    assert any(call.name == "execute_step" for call in result.executed_calls)
