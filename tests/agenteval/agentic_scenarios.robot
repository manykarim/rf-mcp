*** Settings ***
Documentation    Phase-1 ported agentic e2e scenarios (change: adopt-agenteval-harness).
...
...              The scenario DATA (scenarios/*.yaml) is unchanged from the bespoke
...              tests/e2e harness; only the RUNNER changed. A real in-process agent
...              drives the spawned rf-mcp server, and agenteval's deterministic readers
...              assert the tool-call trace reached the scenario's expected tools at or
...              above its `min_tool_hit_rate` gate — the same robust signal the bespoke
...              quality gate used (never model self-report).
...
...              Gated: the API scenario needs AGENTEVAL_API_KEY (skips otherwise); the
...              web scenario additionally needs a browser (set AGENTEVAL_WEB=1 after
...              `rfbrowser init`). See ./README.md.
Library          MCPLibrary
Library          MetricsLibrary
Library          Collections
Library          scenario_lib.py

*** Variables ***
${RFMCP}         ${CURDIR}${/}..${/}..${/}.venv${/}bin${/}robotmcp

*** Test Cases ***
Restful Booker API Scenario Reaches Its Tool Surface
    [Tags]    agentic    api
    # Skipped by DEFAULT: this scenario exceeds agenteval 0.3.0's in-process
    # request_limit=50, which the adapter does not let us raise. Running it burns
    # ~50 live model requests only to fail on the cap, so it is off unless opted in.
    # Set AGENTEVAL_ALLOW_LONG=1 to run it (e.g. once an upstream usage-limit knob or
    # a CLI adapter is wired). See README - Findings.
    Skip If    '%{AGENTEVAL_ALLOW_LONG=}' != '1'
    ...    restful-booker exceeds agenteval 0.3.0 in-process request_limit=50 - opt in with AGENTEVAL_ALLOW_LONG=1
    ${scn}=    Load Agentic Scenario    ${CURDIR}${/}scenarios${/}restful_booker_api.yaml
    Drive Scenario And Assert Tool Parity    ${scn}

DemoShop Data-Driven Cart Scenario Reaches Its Tool Surface
    [Tags]    agentic    web
    Skip If    '%{AGENTEVAL_WEB=}' != '1'
    ...    Web scenario needs a browser - set AGENTEVAL_WEB=1 after 'uv run rfbrowser init'
    ${scn}=    Load Agentic Scenario    ${CURDIR}${/}scenarios${/}demoshop_dd_cart.yaml
    Drive Scenario And Assert Tool Parity    ${scn}

*** Keywords ***
Drive Scenario And Assert Tool Parity
    [Arguments]    ${scn}
    Skip If    '%{AGENTEVAL_API_KEY=}' == ''    No model credential set - agentic tier skipped
    ${handle}=    MCP.Start Server    robotmcp    stdio    command=${RFMCP}    args=${{ [] }}
    MCP.Connect To Server    ${handle}
    ${toolset}=    MCP.As Agent Toolset    ${handle}
    ${adapter}=    Evaluate    AgentEval._core.adapter.get_adapter('in-process', toolsets=[$toolset])
    # agenteval 0.3.0's in-process adapter runs agent.run(prompt) with pydantic-ai's
    # DEFAULT usage limit (request_limit=50) and exposes NO override. Long scenarios
    # (many tool calls + reasoning) hit it - a documented upstream gap, not a port
    # defect, so surface it as a skip rather than a hard fail. See README.
    ${status}    ${result}=    Run Keyword And Ignore Error    Evaluate    $adapter.run($scn['prompt'])
    IF    '${status}' == 'FAIL'
        Skip If    'UsageLimitExceeded' in '''${result}'''
        ...    ${scn}[id]: exceeds agenteval 0.3.0 in-process request_limit=50 (no override in this release) - needs an upstream usage-limit knob, or drive it through a coding-agent CLI adapter
        Fail    ${scn}[id] agent run failed: ${result}
    END
    ${count}=    MCP.Get Tool Call Count    ${result}
    ${names}=    MCP.Get Tool Call Names    ${result}
    ${hit}=    MCP.Get Tool Hit Rate    ${result}    ${scn}[expected_tool_names]
    ${success}=    MCP.Get Tool Success Rate    ${result}
    ${noise}=    MCP.Get Unnecessary Call Rate    ${result}    ${scn}[expected_tool_names]
    ${usage}=    Metric.Get Token Usage    ${result}
    Log    ${scn}[id]: calls=${count} hit_rate=${hit} success_rate=${success} noise=${noise} tokens=${usage}[input]/${usage}[output]    console=True
    Log    ${scn}[id] tool sequence: ${names}    console=True
    Should Be True    ${count} >= 1    msg=Agent made no rf-mcp tool calls for ${scn}[id]
    Should Be True    ${hit} >= ${scn}[min_tool_hit_rate]
    ...    msg=hit_rate ${hit} below the scenario gate ${scn}[min_tool_hit_rate] for ${scn}[id]
    [Teardown]    Run Keyword And Ignore Error    MCP.Stop Server    ${handle}
