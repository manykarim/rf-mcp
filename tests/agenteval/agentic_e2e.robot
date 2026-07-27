*** Settings ***
Documentation    Agentic (Tier-3) e2e: a real in-process LLM agent (pydantic-ai,
...              via agenteval's `in-process` adapter) drives the rf-mcp MCP server,
...              and deterministic readers project the tool-call trace + token metrics
...              off the recorded run. Requires a model credential and the `[agent]`
...              extra; SKIPS cleanly without one, so PR CI stays green keyless.
...
...              Config (read from the environment, never a RF variable):
...                AGENTEVAL_API_KEY   e.g. the MiniMax key
...                AGENTEVAL_BASE_URL  e.g. https://api.minimax.io/v1
...                AGENTEVAL_MODEL     e.g. MiniMax-M3
...              See ./README.md.
Library          MCPLibrary
Library          MetricsLibrary
Library          Collections

*** Variables ***
${RFMCP}         ${CURDIR}${/}..${/}..${/}.venv${/}bin${/}robotmcp

*** Test Cases ***
An LLM Agent Discovers And Calls Rf-mcp Tools Over MCP
    Skip If    '%{AGENTEVAL_API_KEY=}' == ''
    ...    No model credential set (AGENTEVAL_API_KEY/BASE_URL/MODEL) - agentic tier skipped
    ${handle}=    MCP.Start Server    robotmcp    stdio    command=${RFMCP}    args=${{ [] }}
    ${session}=    MCP.Connect To Server    ${handle}
    ${guide}=    MCP.Get Server Instructions    ${session}
    ${toolset}=    MCP.As Agent Toolset    ${handle}
    # v0.4.0: steer the agent with rf-mcp's own MCP `instructions` (the WORKFLOW GUIDE).
    ${adapter}=    Evaluate
    ...    AgentEval._core.adapter.get_adapter('in-process', toolsets=[$toolset], instructions=$guide)
    ${result}=    Evaluate
    ...    $adapter.run("You are testing a web app with Robot Framework MCP tools. Begin by calling analyze_scenario with scenario 'Open https://www.saucedemo.com and verify the page title' to set up the session.")
    ${count}=    MCP.Get Tool Call Count    ${result}
    ${names}=    MCP.Get Tool Call Names    ${result}
    Log    AGENTIC RUN -> tool_calls=${count} names=${names}    console=True
    Should Be True    ${count} >= 1    msg=The agent should have executed at least one rf-mcp tool
    MCP.Was Tool Called    ${result}    analyze_scenario
    ${usage}=    Metric.Get Token Usage    ${result}
    Log    tokens input=${usage}[input] output=${usage}[output]    console=True
    [Teardown]    Run Keyword And Ignore Error    MCP.Stop Server    ${handle}
