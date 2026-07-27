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
    # A long scenario (~50-100 live model requests): read + create + authenticate +
    # delete + build. With agenteval >=0.4.0 it COMPLETES (instructions injected +
    # request_limit raised); it is off by default only to keep per-push CI cheap.
    # Opt in with AGENTEVAL_ALLOW_LONG=1.
    Skip If    '%{AGENTEVAL_ALLOW_LONG=}' != '1'
    ...    restful-booker is a long/costly scenario - opt in with AGENTEVAL_ALLOW_LONG=1
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
    ${session}=    MCP.Connect To Server    ${handle}
    ${guide}=    MCP.Get Server Instructions    ${session}
    ${toolset}=    MCP.As Agent Toolset    ${handle}
    # agenteval >=0.4.0: inject rf-mcp's own MCP `instructions` (the WORKFLOW GUIDE) so
    # the agent is steered like a compliant MCP client, and raise the request limit so
    # long-but-legitimate scenarios complete (pydantic-ai defaults to 50; rf-mcp's own
    # bespoke harness uses 100). Both were unavailable in 0.3.0 - see README - Findings.
    ${adapter}=    Evaluate
    ...    AgentEval._core.adapter.get_adapter('in-process', toolsets=[$toolset], instructions=$guide, request_limit=120)
    ${status}    ${result}=    Run Keyword And Ignore Error    Evaluate    $adapter.run($scn['prompt'])
    IF    '${status}' == 'FAIL'
        Skip If    'UsageLimitExceeded' in '''${result}'''
        ...    ${scn}[id]: exceeded request_limit=120 - raise it further if the scenario is legitimately longer
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
