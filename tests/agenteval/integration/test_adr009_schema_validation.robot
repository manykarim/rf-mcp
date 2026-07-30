*** Settings ***
Documentation     ADR-009 type-constrained tool schemas — ported from
...               tests/integration/test_adr009_schema_validation.py (change:
...               agenteval-port-partial-integration). Asserts the LIVE FastMCP-generated
...               inputSchema the spawned server advertises (via ``MCP.List Tools``) carries
...               the enum/required constraints, and that runtime calls honor them — the same
...               observable facts the pytest read from ``client.list_tools()[i].inputSchema``.
Resource          rfmcp.resource
Suite Setup       Capture Live Tool Listing
Suite Teardown    Stop Rf-mcp Server

*** Keywords ***
Capture Live Tool Listing
    Start Rf-mcp Server
    ${tools}=    MCP.List Tools    ${HANDLE}
    Set Suite Variable    ${TOOLS}    ${tools}

*** Test Cases ***
# --- 1. manage_session schema ---
manage_session action Has Enum
    Param Enum Should Be    ${TOOLS}    manage_session    action
    ...    ${{ ['init','initialize','bootstrap','import_library','library','import_resource','resource','set_variables','variables','import_variables','load_variables','start_test','end_test','start_task','end_task','add_data_row','data_row','list_tests','set_suite_setup','set_suite_teardown','set_tool_profile','tool_profile'] }}

manage_session test_status Has Enum
    Param Enum Should Be    ${TOOLS}    manage_session    test_status    ${{ ['pass','fail'] }}

manage_session tool_profile Has Enum
    Param Enum Should Be    ${TOOLS}    manage_session    tool_profile
    ...    ${{ ['browser_exec','api_exec','discovery','minimal_exec','desktop_exec','slim_exec','full'] }}

manage_session model_tier Has Enum
    Param Enum Should Be    ${TOOLS}    manage_session    model_tier
    ...    ${{ ['small_7b','small_context','medium_13b','standard','large_context','hosted'] }}

manage_session profile Has Enum
    Param Enum Should Be    ${TOOLS}    manage_session    profile
    ...    ${{ ['browser_exec','api_exec','discovery','minimal_exec','desktop_exec','slim_exec','full'] }}

manage_session action Is Required
    Param Should Be Required    ${TOOLS}    manage_session    action

# --- 2. intent_action schema ---
intent_action intent Has Enum
    Param Enum Should Be    ${TOOLS}    intent_action    intent
    ...    ${{ ['navigate','click','fill','hover','select','assert_visible','extract_text','wait_for','extract'] }}

intent_action intent Is Required
    Param Should Be Required    ${TOOLS}    intent_action    intent

# --- 3. find_keywords schema ---
find_keywords strategy Has Enum
    Param Enum Should Be    ${TOOLS}    find_keywords    strategy    ${{ ['semantic','pattern','catalog','session'] }}

find_keywords context Has Enum
    Param Enum Should Be    ${TOOLS}    find_keywords    context    ${{ ['web','mobile','api','desktop','generic','database'] }}

# --- 4. execute_step schema ---
execute_step mode Has Enum
    Param Enum Should Be    ${TOOLS}    execute_step    mode    ${{ ['keyword','evaluate'] }}

execute_step detail_level Has Enum
    Param Enum Should Be    ${TOOLS}    execute_step    detail_level    ${{ ['minimal','standard','full'] }}

# --- 5. execute_flow schema ---
execute_flow structure Has Enum
    Param Enum Should Be    ${TOOLS}    execute_flow    structure    ${{ ['if','for','try'] }}

# --- 6. run_test_suite schema ---
run_test_suite mode Has Enum
    Param Enum Should Be    ${TOOLS}    run_test_suite    mode    ${{ ['dry','validate','full'] }}

run_test_suite validation_level Has Enum
    Param Enum Should Be    ${TOOLS}    run_test_suite    validation_level    ${{ ['minimal','standard','strict'] }}

run_test_suite output_level Enum Is Informational
    [Documentation]    Ported from an xfail(strict=False) test: output_level uses
    ...    'detailed' not 'full' so it may not match the DetailLevel alias. Non-enforcing —
    ...    logs the outcome without failing the suite, preserving the original's intent.
    ${status}    ${err}=    Run Keyword And Ignore Error
    ...    Param Should Have Enum    ${TOOLS}    run_test_suite    output_level
    Log    run_test_suite.output_level enum present: ${status} (${err})

# --- 7. additional tools with type aliases ---
manage_library_plugins action Has Enum
    Param Enum Should Be    ${TOOLS}    manage_library_plugins    action    ${{ ['list','reload','diagnose'] }}

manage_attach action Has Enum
    Param Enum Should Be    ${TOOLS}    manage_attach    action
    ...    ${{ ['status','info','stop','shutdown','cleanup','clean','reset','reconnect','disconnect_all','terminate','force_stop'] }}

recommend_libraries context Has Enum
    Param Enum Should Be    ${TOOLS}    recommend_libraries    context    ${{ ['web','mobile','api','desktop','generic','database'] }}

recommend_libraries mode Has Enum
    Param Enum Should Be    ${TOOLS}    recommend_libraries    mode    ${{ ['direct','sampling_prompt','sampling','merge_samples','merge'] }}

# --- 8. enum coverage metrics ---
At Least 18 Enum Constrained Params
    ${count}=    Count Enum Constrained Params    ${TOOLS}
    Should Be True    ${count} >= 18    Expected >= 18 enum params, got ${count}

Enum Param Inventory Has At Least 10
    ${names}=    Enum Constrained Param Names    ${TOOLS}
    ${n}=    Get Length    ${names}
    Should Be True    ${n} >= 10    Expected >= 10 enum params, got ${n}: ${names}

No Free Text Action Params
    ${offenders}=    Tools With Unconstrained Action Param    ${TOOLS}
    Should Be Empty    ${offenders}    Tools with unconstrained 'action': ${offenders}

# --- 9. runtime validation (call the live tool) ---
manage_session Rejects Invalid Action
    [Documentation]    The invalid enum is rejected — the server raises ``Invalid arguments``
    ...    at the schema boundary (or, if it returned, success is falsey). Mirrors the pytest
    ...    original's try/except leniency; the observable fact is the rejection.
    ${status}    ${r}=    Run Keyword And Ignore Error
    ...    Rf Tool    ${HANDLE}    manage_session    ${{ {'action':'bogus_action','session_id':'adr009-bogus'} }}
    IF    '${status}' == 'PASS'
        Should Not Be True    ${r.get('success', False)}    manage_session should reject 'bogus_action'
    END

manage_session Accepts Valid Action Case Insensitive
    ${r}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'action':'INIT','session_id':'adr009-init'} }}
    Result Field Should Be    ${r}    success    ${True}

intent_action Rejects Invalid Intent
    ${status}    ${r}=    Run Keyword And Ignore Error
    ...    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent':'destroy','target':'#button','session_id':'adr009-intent'} }}
    IF    '${status}' == 'PASS'
        Should Not Be True    ${r.get('success', False)}    intent_action should reject 'destroy'
    END