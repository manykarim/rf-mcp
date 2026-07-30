*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_adr010_e2e.py: ADR-010 small-LLM resilience -
...              array coercion (JSON string / CSV / single value), session-id
...              auto-generation, next-step guidance, and catalog empty-hint - driven
...              through rf-mcp over the REAL MCP protocol; deterministic.
Resource         rfmcp.resource
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Test Cases ***
Libraries As JSON String
    [Documentation]    I1: manage_session accepts libraries as a JSON array string.
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-lib-1', 'libraries': '["BuiltIn"]'} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Contain    ${r}[libraries_loaded]    BuiltIn

Libraries As JSON String Multiple
    [Documentation]    I1: manage_session accepts multi-element JSON string.
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-lib-2', 'libraries': '["BuiltIn", "Collections"]'} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Contain    ${r}[libraries_loaded]    BuiltIn
    Should Contain    ${r}[libraries_loaded]    Collections

Libraries As Comma Separated
    [Documentation]    I2: manage_session accepts comma-separated string.
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-lib-3', 'libraries': 'BuiltIn, Collections'} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Contain    ${r}[libraries_loaded]    BuiltIn
    Should Contain    ${r}[libraries_loaded]    Collections

Libraries As Single String
    [Documentation]    I2: manage_session accepts bare single-library string.
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-lib-4', 'libraries': 'BuiltIn'} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Contain    ${r}[libraries_loaded]    BuiltIn

Libraries As Normal List
    [Documentation]    Baseline: standard list still works after coercion changes.
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-lib-5', 'libraries': ['BuiltIn']} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Contain    ${r}[libraries_loaded]    BuiltIn

Sections As JSON String
    [Documentation]    I1: get_session_state accepts sections as JSON string.
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-sec-1', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': 'adr010-sec-1', 'sections': '["summary"]'} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Should Contain Field    ${r}    sections
    Should Contain    ${r}[sections]    summary

Sections As Comma Separated
    [Documentation]    I2: get_session_state accepts sections as CSV string.
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-sec-2', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': 'adr010-sec-2', 'sections': 'summary, variables'} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Should Contain Field    ${r}    sections

Sections As Normal List
    [Documentation]    Baseline: standard list for sections still works.
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-sec-3', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': 'adr010-sec-3', 'sections': ['summary']} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Should Contain Field    ${r}    sections
    Should Contain    ${r}[sections]    summary

Arguments As JSON String
    [Documentation]    I1: execute_step accepts arguments as JSON array string.
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-arg-1', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': '["Hello from ADR-010"]', 'session_id': 'adr010-arg-1'} }}
    Result Field Should Be    ${r}    success    ${True}

Arguments As Single String
    [Documentation]    I2: execute_step accepts a bare single argument string.
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-arg-2', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': 'Hello single string', 'session_id': 'adr010-arg-2'} }}
    Result Field Should Be    ${r}    success    ${True}

Arguments As Normal List
    [Documentation]    Baseline: standard list arguments still work.
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-arg-3', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': ['Hello normal list'], 'session_id': 'adr010-arg-3'} }}
    Result Field Should Be    ${r}    success    ${True}

Arguments JSON String With Should Be Equal
    [Documentation]    I1: JSON string arguments with a keyword that needs two args.
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-arg-4', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Should Be Equal', 'arguments': '["hello", "hello"]', 'session_id': 'adr010-arg-4'} }}
    Result Field Should Be    ${r}    success    ${True}

Test Tags As JSON String
    [Documentation]    I1: start_test accepts test_tags as JSON string.
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-tags-1', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'start_test', 'session_id': 'adr010-tags-1', 'test_name': 'ADR-010 Tag Test', 'test_tags': '["smoke", "adr010"]'} }}
    Result Field Should Be    ${r}    success    ${True}

Test Tags As Comma Separated
    [Documentation]    I2: start_test accepts test_tags as CSV string.
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-tags-2', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'start_test', 'session_id': 'adr010-tags-2', 'test_name': 'ADR-010 CSV Tag Test', 'test_tags': 'smoke, adr010'} }}
    Result Field Should Be    ${r}    success    ${True}

Empty String Session Id
    [Documentation]    I5: manage_session auto-generates session_id when empty string.
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': '', 'libraries': ['BuiltIn']} }}
    Result Field Should Be    ${r}    success    ${True}
    ${returned_sid}=    Evaluate    $r.get('session_id', '')
    Should Be True    $returned_sid is not None
    Should Be True    isinstance($returned_sid, str)

Omitted Session Id Defaults
    [Documentation]    I5: manage_session works when session_id not provided (default).
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'libraries': ['BuiltIn']} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Should Contain Field    ${r}    session_id

Auto Session Id Is Usable
    [Documentation]    I5: auto-generated session_id can be used for execute_step.
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': '', 'libraries': ['BuiltIn']} }}
    Result Field Should Be    ${init}    success    ${True}
    ${sid}=    Set Variable    ${init}[session_id]
    ${step}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': ['Using auto-generated session'], 'session_id': $sid} }}
    Result Field Should Be    ${step}    success    ${True}

Init Response Has Next Step
    [Documentation]    I6: init response includes next_step guidance field.
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-next-1', 'libraries': ['BuiltIn']} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Should Contain Field    ${r}    next_step

Next Step Contains Session Id
    [Documentation]    I6: next_step guidance mentions the session_id for reuse.
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-next-2', 'libraries': ['BuiltIn']} }}
    Result Should Contain Field    ${r}    next_step
    Should Contain    ${r}[next_step]    adr010-next-2

Next Step Is String
    [Documentation]    I6: next_step is a human-readable string, not a dict.
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-next-3', 'libraries': ['BuiltIn']} }}
    Should Be True    isinstance($r['next_step'], str)

Catalog Empty Without Session Has Hint
    [Documentation]    I3: catalog returns hint when called without active session.
    ${r}=    Rf Tool    ${HANDLE}    find_keywords
    ...    ${{ {'query': 'nonexistent_xyz', 'strategy': 'catalog'} }}
    ${results}=    Evaluate    $r.get('results')
    IF    not $results
        Result Should Contain Field    ${r}    hint
        ${hint_lower}=    Evaluate    $r['hint'].lower()
        Should Be True    'semantic' in $hint_lower or 'session' in $hint_lower
    END

Catalog With Session No Hint
    [Documentation]    I3: catalog with active session does NOT include hint.
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'adr010-cat-1', 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    find_keywords
    ...    ${{ {'query': 'Log', 'strategy': 'catalog', 'session_id': 'adr010-cat-1'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${results}=    Evaluate    $r.get('results')
    Should Be True    $results
    Should Be True    'hint' not in $r or not $r['hint']

Full Workflow String Inputs Only
    [Documentation]    Full lifecycle using string inputs a small LLM would produce.
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'small-llm-1', 'libraries': '["BuiltIn", "String"]'} }}
    Result Field Should Be    ${init}    success    ${True}
    Result Should Contain Field    ${init}    next_step
    ${step1}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Convert To Upper Case', 'arguments': '["hello world"]', 'session_id': 'small-llm-1', 'assign_to': 'RESULT'} }}
    Result Field Should Be    ${step1}    success    ${True}
    ${step2}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Should Be Equal', 'arguments': '["\${RESULT}", "HELLO WORLD"]', 'session_id': 'small-llm-1'} }}
    Result Field Should Be    ${step2}    success    ${True}
    ${state}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': 'small-llm-1', 'sections': '["summary", "variables"]'} }}
    Result Field Should Be    ${state}    success    ${True}
    Result Should Contain Field    ${state}    sections
    ${build}=    Rf Tool    ${HANDLE}    build_test_suite
    ...    ${{ {'session_id': 'small-llm-1', 'test_name': 'Small LLM Resilience Test'} }}
    Result Field Should Be    ${build}    success    ${True}
    Should Contain    ${build}[rf_text]    *** Test Cases ***
    Should Contain    ${build}[rf_text]    Convert To Upper Case

Multi Test Workflow With Coerced Tags
    [Documentation]    Multi-test workflow using string tags (small LLM pattern).
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'multi-coerce-1', 'libraries': 'BuiltIn, Collections'} }}
    Result Field Should Be    ${init}    success    ${True}
    ${start}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'start_test', 'session_id': 'multi-coerce-1', 'test_name': 'Coerced Tag Test', 'test_tags': '["smoke", "resilience"]'} }}
    Result Field Should Be    ${start}    success    ${True}
    ${step}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Create Dictionary', 'arguments': '["key=value"]', 'session_id': 'multi-coerce-1', 'assign_to': 'MY_DICT'} }}
    Result Field Should Be    ${step}    success    ${True}
    ${end}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'end_test', 'session_id': 'multi-coerce-1'} }}
    Result Field Should Be    ${end}    success    ${True}
