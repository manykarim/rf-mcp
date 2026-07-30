*** Settings ***
Documentation    Shared setup for the ported black-box integration suites
...              (change: agenteval-port-blackbox-integration). One rf-mcp server per
...              suite; tests isolate via distinct session_id values (rf-mcp keys state
...              by session_id), mirroring the pytest originals' shared-server model.
Library          MCPLibrary
Library          mcp_result.py

*** Variables ***
${RFMCP}         ${CURDIR}${/}..${/}..${/}..${/}.venv${/}bin${/}robotmcp

*** Keywords ***
Start Rf-mcp Server
    ${handle}=    MCP.Start Server    robotmcp    stdio    command=${RFMCP}    args=${{ [] }}
    MCP.Connect To Server    ${handle}
    Set Suite Variable    ${HANDLE}    ${handle}

Stop Rf-mcp Server
    Run Keyword And Ignore Error    MCP.Stop Server    ${HANDLE}
