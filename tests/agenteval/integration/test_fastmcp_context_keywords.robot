*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_fastmcp_context_keywords.py: context-only BuiltIn
...              keywords driven through rf-mcp over the REAL MCP protocol (the original
...              used an in-memory fastmcp Client). Deterministic - no model key.
Resource         rfmcp.resource
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Test Cases ***
Set Test Variable And Get Value
    [Documentation]    Set a test variable in RF native context and retrieve it.
    ${set}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Set Test Variable', 'arguments': ['\${X}', '123'], 'session_id': 'ctx_vars_session', 'raise_on_failure': True} }}
    Result Field Should Be    ${set}    success    ${True}
    ${get}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Get Variable Value', 'arguments': ['\${X}'], 'session_id': 'ctx_vars_session', 'assign_to': 'val', 'raise_on_failure': True} }}
    Result Field Should Be    ${get}    success    ${True}
    ${assigned}=    Set Variable    ${get}[assigned_variables]
    Should Be True    $assigned.get('\${val}') in ('123', 123)
