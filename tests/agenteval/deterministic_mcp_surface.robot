*** Settings ***
Documentation    Deterministic (Tier-1) MCP-surface checks for rf-mcp, driven by
...              robotframework-agenteval. Spawns rf-mcp as a subprocess over the real
...              MCP protocol and asserts on its advertised tool surface and on tool
...              results — no model, no API key. This is the always-on CI gate.
...              See ./README.md for the isolation model and how to run.
Library          MCPLibrary

*** Variables ***
# rf-mcp launches from its OWN venv (repo-relative), spawned as a subprocess.
${RFMCP}         ${CURDIR}${/}..${/}..${/}.venv${/}bin${/}robotmcp

*** Test Cases ***
Mcp Launch Descriptor Is Well-Formed
    ${servers}=    MCP.Get Server Config    ${CURDIR}${/}.mcp.json
    Should Be Equal    ${servers}[robotmcp][transport]    stdio

Rf-mcp Advertises Its Tool Surface Over MCP
    ${handle}=    MCP.Start Server    robotmcp    stdio    command=${RFMCP}    args=${{ [] }}
    ${session}=    MCP.Connect To Server    ${handle}
    Should Not Be Empty    ${session.protocol_version}
    @{tools}=    MCP.List Tools    ${handle}
    ${names}=    Evaluate    sorted(t.name for t in $tools)
    Should Contain    ${names}    analyze_scenario
    Should Contain    ${names}    execute_step
    Should Contain    ${names}    build_test_suite
    [Teardown]    MCP.Stop Server    ${handle}

Rf-mcp analyze_scenario Answers Deterministically
    ${handle}=    MCP.Start Server    robotmcp    stdio    command=${RFMCP}    args=${{ [] }}
    MCP.Connect To Server    ${handle}
    ${result}=    MCP.Call Tool    ${handle}    analyze_scenario
    ...    scenario=Open the SauceDemo login page and verify the page title
    Should Be Equal    ${result.is_error}    ${False}
    [Teardown]    MCP.Stop Server    ${handle}
