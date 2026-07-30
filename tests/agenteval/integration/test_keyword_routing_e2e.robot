*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_keyword_routing_e2e.py: E2E keyword routing exercising
...              library preference enforcement, keyword validation, and cross-library routing.
...              Real MCP protocol; deterministic.
Resource         rfmcp.resource
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Test Cases ***
Shared Keyword In BuiltIn Session
    [Documentation]    Shared BuiltIn keywords work in any session.
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': 'kwroute-shared', 'action': 'init', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': ['Shared keyword test'], 'session_id': 'kwroute-shared'} }}
    Result Field Should Be    ${r}    success    ${True}

No Preference Session Allows BuiltIn Keywords
    [Documentation]    No-preference session allows all BuiltIn keywords.
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': 'kwroute-nopref', 'action': 'init', 'libraries': ['BuiltIn']} }}
    ${s1}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': ['test'], 'session_id': 'kwroute-nopref'} }}
    Result Field Should Be    ${s1}    success    ${True}
    ${s2}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Set Variable', 'arguments': ['value'], 'session_id': 'kwroute-nopref'} }}
    Result Field Should Be    ${s2}    success    ${True}
    ${s3}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Should Be True', 'arguments': ['True'], 'session_id': 'kwroute-nopref'} }}
    Result Field Should Be    ${s3}    success    ${True}

Full 5 Tool Sequence
    [Documentation]    Complete tool call sequence exercises keyword routing at each stage.
    ${analyze}=    Rf Tool    ${HANDLE}    analyze_scenario
    ...    ${{ {'scenario': 'Test data processing with BuiltIn and Collections keywords', 'context': 'api'} }}
    Result Field Should Be    ${analyze}    success    ${True}
    ${sid}=    Set Variable    ${analyze}[session_id]
    ${recommend}=    Rf Tool    ${HANDLE}    recommend_libraries
    ...    ${{ {'scenario': 'Data processing with lists and dictionaries'} }}
    Result Field Should Be    ${recommend}    success    ${True}
    ${manage}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': $sid, 'action': 'import_library', 'library_name': 'Collections'} }}
    Result Field Should Be    ${manage}    success    ${True}
    ${step1}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': ['Starting data processing'], 'session_id': $sid} }}
    Result Field Should Be    ${step1}    success    ${True}
    ${step2}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Create Dictionary', 'arguments': ['name=Alice', 'age=30'], 'session_id': $sid, 'assign_to': 'USER'} }}
    Result Field Should Be    ${step2}    success    ${True}
    ${state}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': $sid, 'sections': ['summary', 'variables', 'libraries']} }}
    Result Field Should Be    ${state}    success    ${True}

Find Keywords With Session Id
    [Documentation]    find_keywords with session_id should use session's library context.
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': 'kwroute-fksess', 'action': 'init', 'libraries': ['BuiltIn', 'String']} }}
    ${r}=    Rf Tool    ${HANDLE}    find_keywords
    ...    ${{ {'query': 'Convert To Upper Case', 'session_id': 'kwroute-fksess'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${keywords}=    Evaluate    $r.get('result', [])
    Should Be True    len($keywords) > 0

Find Keywords Without Session Searches Globally
    [Documentation]    find_keywords without session_id searches the global cache.
    ${r}=    Rf Tool    ${HANDLE}    find_keywords    ${{ {'query': 'Log'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${keywords}=    Evaluate    $r.get('result', [])
    Should Be True    len($keywords) > 0
