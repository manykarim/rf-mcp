*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_mcp_e2e_builtin_only.py: BuiltIn-only end-to-end
...              flows driven through rf-mcp over the REAL MCP protocol (the original used
...              an in-memory fastmcp Client). Deterministic - no browser, no model key.
...              One server per suite; each test isolates via a distinct session_id.
Resource         rfmcp.resource
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Test Cases ***
Full Session Lifecycle Analyze To Build
    ${analyze}=    Rf Tool    ${HANDLE}    analyze_scenario
    ...    ${{ {'scenario': 'Test string operations with Robot Framework BuiltIn keywords', 'context': 'api'} }}
    Result Field Should Be    ${analyze}    success    ${True}
    ${sid}=    Set Variable    ${analyze}[session_id]
    ${set_vars}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': $sid, 'action': 'set_variables', 'variables': {'TEXT': 'Hello World', 'EXPECTED': 'HELLO WORLD'}} }}
    Result Field Should Be    ${set_vars}    success    ${True}
    ${step1}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['Starting test'], 'session_id': $sid} }}
    Result Field Should Be    ${step1}    success    ${True}
    ${step2}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Convert To Upper Case', 'arguments': ['\${TEXT}'], 'session_id': $sid, 'assign_to': 'RESULT'} }}
    Result Field Should Be    ${step2}    success    ${True}
    ${step3}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Should Be Equal', 'arguments': ['\${RESULT}', '\${EXPECTED}'], 'session_id': $sid} }}
    Result Field Should Be    ${step3}    success    ${True}
    ${step4}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['Test completed'], 'session_id': $sid} }}
    Result Field Should Be    ${step4}    success    ${True}
    ${state}=    Rf Tool    ${HANDLE}    get_session_state    ${{ {'session_id': $sid, 'sections': ['summary', 'variables']} }}
    Result Field Should Be    ${state}    success    ${True}
    ${build}=    Rf Tool    ${HANDLE}    build_test_suite
    ...    ${{ {'session_id': $sid, 'test_name': 'String Operations Test', 'documentation': 'Validates string conversion'} }}
    Result Field Should Be    ${build}    success    ${True}
    Should Contain    ${build}[rf_text]    *** Test Cases ***
    Should Contain    ${build}[rf_text]    Convert To Upper Case

Multi Test Session Via MCP
    ${init}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-multi', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Result Field Should Be    ${init}    success    ${True}
    ${setup}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': 'e2e-multi', 'action': 'set_suite_setup', 'keyword': 'Log', 'args': ['Suite starting']} }}
    Result Field Should Be    ${setup}    success    ${True}
    ${start_a}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-multi', 'action': 'start_test', 'test_name': 'Test A - Log'} }}
    Result Field Should Be    ${start_a}    success    ${True}
    ${step_a}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['In Test A'], 'session_id': 'e2e-multi'} }}
    Result Field Should Be    ${step_a}    success    ${True}
    ${end_a}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-multi', 'action': 'end_test'} }}
    Result Field Should Be    ${end_a}    success    ${True}
    ${start_b}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-multi', 'action': 'start_test', 'test_name': 'Test B - Convert'} }}
    Result Field Should Be    ${start_b}    success    ${True}
    ${step_b}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Convert To Upper Case', 'arguments': ['hello'], 'session_id': 'e2e-multi', 'assign_to': 'UPPER'} }}
    Result Field Should Be    ${step_b}    success    ${True}
    ${end_b}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-multi', 'action': 'end_test'} }}
    Result Field Should Be    ${end_b}    success    ${True}
    ${lt}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-multi', 'action': 'list_tests'} }}
    Result Field Should Be    ${lt}    success    ${True}
    ${names}=    Evaluate    [t.get('name', t.get('test_name', '')) for t in $lt.get('tests', [])]
    Should Contain    ${names}    Test A - Log
    Should Contain    ${names}    Test B - Convert
    ${build}=    Rf Tool    ${HANDLE}    build_test_suite    ${{ {'session_id': 'e2e-multi', 'test_name': 'Multi Test Suite'} }}
    Result Field Should Be    ${build}    success    ${True}
    Should Contain    ${build}[rf_text]    Test A
    Should Contain    ${build}[rf_text]    Test B

Auto Created Session Lifecycle
    ${step1}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Set Variable', 'arguments': ['hello_world'], 'session_id': 'default', 'assign_to': 'MY_VAR'} }}
    Result Field Should Be    ${step1}    success    ${True}
    ${step2}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['\${MY_VAR}'], 'session_id': 'default'} }}
    Result Field Should Be    ${step2}    success    ${True}
    ${state}=    Rf Tool    ${HANDLE}    get_session_state    ${{ {'session_id': 'default', 'sections': ['summary', 'variables']} }}
    Result Field Should Be    ${state}    success    ${True}
    ${build}=    Rf Tool    ${HANDLE}    build_test_suite    ${{ {'session_id': 'default', 'test_name': 'Auto Created Test'} }}
    Result Field Should Be    ${build}    success    ${True}
    Should Contain    ${build}[rf_text]    *** Test Cases ***

Error Recovery After Failed Step
    ${init}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-err', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Result Field Should Be    ${init}    success    ${True}
    ${failed}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Should Be Equal', 'arguments': ['abc', 'xyz'], 'session_id': 'e2e-err', 'raise_on_failure': False} }}
    Should Be True    isinstance($failed, dict)
    ${recovery}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['Recovered!'], 'session_id': 'e2e-err'} }}
    Result Field Should Be    ${recovery}    success    ${True}

Dynamic Library Import Enables Keywords
    ${init}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-dynlib', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Result Field Should Be    ${init}    success    ${True}
    ${imp}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-dynlib', 'action': 'import_library', 'library_name': 'Collections'} }}
    Result Field Should Be    ${imp}    success    ${True}
    ${step}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Create Dictionary', 'arguments': ['key1=val1', 'key2=val2'], 'session_id': 'e2e-dynlib', 'assign_to': 'MY_DICT'} }}
    Result Field Should Be    ${step}    success    ${True}

Get Session State All Sections
    ${init}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-state', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Result Field Should Be    ${init}    success    ${True}
    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['test'], 'session_id': 'e2e-state'} }}
    ${state}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': 'e2e-state', 'sections': ['summary', 'variables', 'page_source', 'validation', 'libraries', 'rf_context', 'application_state']} }}
    Result Field Should Be    ${state}    success    ${True}
    Should Contain    ${state}[sections]    summary

Concurrent Sessions State Isolation
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-iso1', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-iso2', 'action': 'init', 'libraries': ['BuiltIn', 'Collections']} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-iso3', 'action': 'init', 'libraries': ['BuiltIn', 'String']} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-iso1', 'action': 'set_variables', 'variables': {'VAR': 'session1'}} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-iso2', 'action': 'set_variables', 'variables': {'VAR': 'session2'}} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-iso3', 'action': 'set_variables', 'variables': {'VAR': 'session3'}} }}
    ${s1}=    Rf Tool    ${HANDLE}    get_session_state    ${{ {'session_id': 'e2e-iso1', 'sections': ['variables']} }}
    ${s2}=    Rf Tool    ${HANDLE}    get_session_state    ${{ {'session_id': 'e2e-iso2', 'sections': ['variables']} }}
    ${s3}=    Rf Tool    ${HANDLE}    get_session_state    ${{ {'session_id': 'e2e-iso3', 'sections': ['variables']} }}
    Result Field Should Be    ${s1}    success    ${True}
    Result Field Should Be    ${s2}    success    ${True}
    Result Field Should Be    ${s3}    success    ${True}

Build Test Suite With Variables Docs Tags
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-build', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': 'e2e-build', 'action': 'set_variables', 'variables': {'URL': 'https://example.com'}, 'scope': 'suite'} }}
    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['hello'], 'session_id': 'e2e-build'} }}
    ${build}=    Rf Tool    ${HANDLE}    build_test_suite
    ...    ${{ {'session_id': 'e2e-build', 'test_name': 'Tagged Test', 'documentation': 'Test docs', 'tags': ['smoke', 'regression']} }}
    Result Field Should Be    ${build}    success    ${True}
    Should Contain    ${build}[rf_text]    *** Test Cases ***

Find Keywords After Dynamic Import
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-fk', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Rf Tool    ${HANDLE}    find_keywords    ${{ {'query': 'Create Dictionary', 'session_id': 'e2e-fk'} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-fk', 'action': 'import_library', 'library_name': 'Collections'} }}
    ${after}=    Rf Tool    ${HANDLE}    find_keywords    ${{ {'query': 'Create Dictionary', 'session_id': 'e2e-fk'} }}
    Result Field Should Be    ${after}    success    ${True}

Check Library Availability
    ${r}=    Rf Tool    ${HANDLE}    check_library_availability    ${{ {'libraries': ['BuiltIn', 'NonExistentLibrary123']} }}
    Result Field Should Be    ${r}    success    ${True}
    ${avail}=    Evaluate    $r.get('availability', $r.get('libraries', {}))
    Should Be True    isinstance($avail, (dict, list))

Set Library Search Order Via MCP
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-so', 'action': 'init', 'libraries': ['BuiltIn', 'String']} }}
    ${r}=    Rf Tool    ${HANDLE}    set_library_search_order    ${{ {'libraries': ['String', 'BuiltIn'], 'session_id': 'e2e-so'} }}
    Result Field Should Be    ${r}    success    ${True}

Analyze Recommend Init Chain
    ${analysis}=    Rf Tool    ${HANDLE}    analyze_scenario    ${{ {'scenario': 'Test REST API endpoints with HTTP requests', 'context': 'api'} }}
    Result Field Should Be    ${analysis}    success    ${True}
    ${sid}=    Set Variable    ${analysis}[session_id]
    ${recommend}=    Rf Tool    ${HANDLE}    recommend_libraries    ${{ {'scenario': 'API testing with HTTP methods'} }}
    Result Field Should Be    ${recommend}    success    ${True}
    ${step}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['API test step'], 'session_id': $sid} }}
    Result Field Should Be    ${step}    success    ${True}

Suite Setup Teardown In Generated Robot
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-stsd', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-stsd', 'action': 'set_suite_setup', 'keyword': 'Log', 'args': ['Suite setup']} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-stsd', 'action': 'set_suite_teardown', 'keyword': 'Log', 'args': ['Suite teardown']} }}
    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['test step'], 'session_id': 'e2e-stsd'} }}
    ${build}=    Rf Tool    ${HANDLE}    build_test_suite    ${{ {'session_id': 'e2e-stsd', 'test_name': 'With Setup'} }}
    Result Field Should Be    ${build}    success    ${True}
    Should Be True    '*** Settings ***' in $build['rf_text'] or 'Suite Setup' in $build['rf_text']

Execute Step Invalid Keyword Graceful Error
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-grace', 'action': 'init', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Nonexistent Keyword That Doesnt Exist', 'arguments': [], 'session_id': 'e2e-grace', 'raise_on_failure': False} }}
    Should Be True    isinstance($r, dict)

Get Session State Nonexistent Session
    ${r}=    Rf Tool    ${HANDLE}    get_session_state    ${{ {'session_id': 'nonexistent-session-id-12345', 'sections': ['summary']} }}
    Should Be True    isinstance($r, dict)

Recommend Libraries Returns Suggestions
    ${r}=    Rf Tool    ${HANDLE}    recommend_libraries    ${{ {'scenario': 'Validate JSON responses from a REST API'} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Be True    isinstance($r, dict)

Variable Roundtrip Via Execute Step
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-varrt', 'action': 'init', 'libraries': ['BuiltIn']} }}
    ${set_result}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Set Variable', 'arguments': ['test_value_123'], 'session_id': 'e2e-varrt', 'assign_to': 'MY_VAR'} }}
    Result Field Should Be    ${set_result}    success    ${True}
    ${log_result}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': ['Variable value is: \${MY_VAR}'], 'session_id': 'e2e-varrt'} }}
    Result Field Should Be    ${log_result}    success    ${True}

Set Variables Via Manage Session Then Use
    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-setvar', 'action': 'init', 'libraries': ['BuiltIn']} }}
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': 'e2e-setvar', 'action': 'set_variables', 'variables': {'GREETING': 'Hello', 'TARGET': 'World'}} }}
    ${step}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': ['\${GREETING} \${TARGET}'], 'session_id': 'e2e-setvar'} }}
    Result Field Should Be    ${step}    success    ${True}

BuiltIn After Analyze Recommend Import
    ${analyze}=    Rf Tool    ${HANDLE}    analyze_scenario    ${{ {'scenario': 'Web UI test: verify text on page', 'context': 'web'} }}
    Result Field Should Be    ${analyze}    success    ${True}
    ${sid}=    Set Variable    ${analyze}[session_id]
    ${rec}=    Rf Tool    ${HANDLE}    recommend_libraries
    ...    ${{ {'scenario': 'Web UI test: verify text on page', 'session_id': $sid, 'check_availability': True, 'apply_search_order': True} }}
    Result Field Should Be    ${rec}    success    ${True}
    ${imp}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': $sid, 'action': 'import_library', 'library_name': 'OperatingSystem'} }}
    Result Field Should Be    ${imp}    success    ${True}
    ${step}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Should Be Equal', 'arguments': ['hello', 'hello'], 'session_id': $sid} }}
    Result Field Should Be    ${step}    success    ${True}

BuiltIn After Import Library Without Analyze
    ${init}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-bi-imp', 'action': 'init'} }}
    Result Field Should Be    ${init}    success    ${True}
    ${imp}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-bi-imp', 'action': 'import_library', 'library_name': 'Collections'} }}
    Result Field Should Be    ${imp}    success    ${True}
    ${step}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Should Be Equal', 'arguments': ['42', '42'], 'session_id': 'e2e-bi-imp'} }}
    Result Field Should Be    ${step}    success    ${True}

BuiltIn Log After Multiple Import Library
    ${init}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-bi-multi', 'action': 'init'} }}
    Result Field Should Be    ${init}    success    ${True}
    FOR    ${lib}    IN    Collections    String    OperatingSystem
        ${imp}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'session_id': 'e2e-bi-multi', 'action': 'import_library', 'library_name': $lib} }}
        Result Field Should Be    ${imp}    success    ${True}
    END
    ${s1}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['Still working'], 'session_id': 'e2e-bi-multi'} }}
    Result Field Should Be    ${s1}    success    ${True}
    ${s2}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Should Be Equal', 'arguments': ['a', 'a'], 'session_id': 'e2e-bi-multi'} }}
    Result Field Should Be    ${s2}    success    ${True}
    ${s3}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Set Variable', 'arguments': ['test_value'], 'session_id': 'e2e-bi-multi'} }}
    Result Field Should Be    ${s3}    success    ${True}
