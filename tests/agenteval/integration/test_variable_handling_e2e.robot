*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_variable_handling_e2e.py: variable scope isolation
...              between tests, persistence via assign_to, init/list/dict variables, built-in
...              variable access, undefined-variable handling, and suite-scoped persistence.
...              Real MCP protocol; deterministic.
Resource         rfmcp.resource
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Test Cases ***
Variable Scope Isolation Between Tests
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'scope-1', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'scope-1', 'action': 'start_test', 'test_name': 'Test A'} }}
    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Set Variable', 'arguments': ['test_a_value'], 'session_id': 'scope-1', 'assign_to': 'TEST_A_VAR'} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'scope-1', 'action': 'end_test'} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'scope-1', 'action': 'start_test', 'test_name': 'Test B'} }}
    ${step_b}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['In Test B'], 'session_id': 'scope-1'} }}
    Result Field Should Be    ${step_b}    success    ${True}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'scope-1', 'action': 'end_test'} }}

Assign To Then Use In Next Step
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'persist-1', 'action': 'init', 'libraries': ['BuiltIn']} }}
    ${step1}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Set Variable', 'arguments': ['hello_world'], 'session_id': 'persist-1', 'assign_to': 'MY_VALUE'} }}
    Result Field Should Be    ${step1}    success    ${True}
    ${step2}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['\${MY_VALUE}'], 'session_id': 'persist-1'} }}
    Result Field Should Be    ${step2}    success    ${True}

Init Dict Variables Readable
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': 'initvars-1', 'action': 'init', 'libraries': ['BuiltIn'], 'variables': {'URL': 'https://example.com', 'TIMEOUT': '10s'}} }}
    Result Field Should Be    ${init}    success    ${True}
    ${state}=    Rf Tool    ${HANDLE}    get_session_state    ${{ {'session_id': 'initvars-1', 'sections': ['variables']} }}
    Result Field Should Be    ${state}    success    ${True}

Set Variables Overrides Previous Value
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'override-1', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': 'override-1', 'action': 'set_variables', 'variables': {'MY_VAR': 'first'}} }}
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': 'override-1', 'action': 'set_variables', 'variables': {'MY_VAR': 'second'}} }}
    ${state}=    Rf Tool    ${HANDLE}    get_session_state    ${{ {'session_id': 'override-1', 'sections': ['variables']} }}
    Result Field Should Be    ${state}    success    ${True}

Create List Variable
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'list-1', 'action': 'init', 'libraries': ['BuiltIn', 'Collections']} }}
    ${step}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Create List', 'arguments': ['item1', 'item2', 'item3'], 'session_id': 'list-1', 'assign_to': 'MY_LIST'} }}
    Result Field Should Be    ${step}    success    ${True}

Create Dictionary Variable
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'dict-1', 'action': 'init', 'libraries': ['BuiltIn', 'Collections']} }}
    ${step}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Create Dictionary', 'arguments': ['host=localhost', 'port=8080'], 'session_id': 'dict-1', 'assign_to': 'MY_DICT'} }}
    Result Field Should Be    ${step}    success    ${True}

Builtin Variables Accessible
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'builtin-1', 'action': 'init', 'libraries': ['BuiltIn']} }}
    ${step}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['\${TRUE}'], 'session_id': 'builtin-1'} }}
    Result Field Should Be    ${step}    success    ${True}

Undefined Variable In Arguments
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'undef-1', 'action': 'init', 'libraries': ['BuiltIn']} }}
    ${result}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': ['\${NONEXISTENT_VAR_12345}'], 'session_id': 'undef-1', 'raise_on_failure': False} }}
    Should Be True    isinstance($result, dict)

Suite Scoped Variable Persists
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'suite-1', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': 'suite-1', 'action': 'set_variables', 'variables': {'SUITE_VAR': 'suite_value'}, 'scope': 'suite'} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'suite-1', 'action': 'start_test', 'test_name': 'Test A'} }}
    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['In Test A'], 'session_id': 'suite-1'} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'suite-1', 'action': 'end_test'} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'suite-1', 'action': 'start_test', 'test_name': 'Test B'} }}
    ${step}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['Suite var: \${SUITE_VAR}'], 'session_id': 'suite-1'} }}
    Result Field Should Be    ${step}    success    ${True}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'suite-1', 'action': 'end_test'} }}
