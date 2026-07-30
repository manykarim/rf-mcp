*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_library_preferences.py: library-preference detection
...              and per-session isolation via analyze_scenario / execute_step /
...              get_session_state. Real MCP protocol; deterministic.
Resource         rfmcp.resource
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Test Cases ***
Analyze Scenario Selenium Preference
    ${r}=    Rf Tool    ${HANDLE}    analyze_scenario
    ...    ${{ {'scenario': 'Use SeleniumLibrary for classic web automation', 'context': 'web'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${analysis}=    Set Variable    ${r}[analysis]
    Result Field Should Be    ${analysis}    explicit_library_preference    SeleniumLibrary
    Result Field Should Be    ${analysis}    detected_session_type    web_automation
    # analyze_scenario returns session_id directly, not under session_info
    Result Should Contain Field    ${r}    session_id
    # Library preference is in the analysis sub-dict (duplicate assertion preserved)
    Result Field Should Be    ${analysis}    explicit_library_preference    SeleniumLibrary

Analyze Scenario Browser Preference
    ${r}=    Rf Tool    ${HANDLE}    analyze_scenario
    ...    ${{ {'scenario': 'Use Browser Library for modern cross-browser automation', 'context': 'web'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${analysis}=    Set Variable    ${r}[analysis]
    Result Field Should Be    ${analysis}    explicit_library_preference    Browser
    Result Should Contain Field    ${r}    session_id

Analyze Scenario No Preference
    ${r}=    Rf Tool    ${HANDLE}    analyze_scenario
    ...    ${{ {'scenario': 'Create an automation suite without naming any library', 'context': 'web'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${analysis}=    Set Variable    ${r}[analysis]
    Result Field Should Be    ${analysis}    explicit_library_preference    ${None}

Execute Step Respects Preference
    ${analyze}=    Rf Tool    ${HANDLE}    analyze_scenario
    ...    ${{ {'scenario': 'Use SeleniumLibrary for web automation', 'context': 'web'} }}
    ${sid}=    Set Variable    ${analyze}[session_id]
    ${result}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': ['Preference check'], 'session_id': $sid} }}
    Result Field Should Be    ${result}    success    ${True}
    ${state}=    Rf Tool    ${HANDLE}    get_session_state    ${{ {'session_id': $sid, 'sections': ['summary']} }}
    ${summary}=    Extract Summary    ${state}
    Result Field Should Be    ${summary}    explicit_library_preference    SeleniumLibrary
    Should Be Equal    ${summary}[search_order][0]    SeleniumLibrary

Validation Section Available
    ${analyze}=    Rf Tool    ${HANDLE}    analyze_scenario
    ...    ${{ {'scenario': 'Use SeleniumLibrary to validate login', 'context': 'web'} }}
    ${sid}=    Set Variable    ${analyze}[session_id]
    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Log', 'arguments': ['validation'], 'session_id': $sid} }}
    ${state}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': $sid, 'sections': ['summary', 'validation']} }}
    ${summary}=    Extract Summary    ${state}
    Result Field Should Be    ${summary}    session_id    ${sid}
    ${validation}=    Evaluate    $state['sections'].get('validation')
    Should Be True    $validation
    Result Field Should Be    ${validation}    success    ${True}

Multiple Sessions Isolated
    ${selenium}=    Rf Tool    ${HANDLE}    analyze_scenario
    ...    ${{ {'scenario': 'Use SeleniumLibrary', 'context': 'web'} }}
    ${browser}=    Rf Tool    ${HANDLE}    analyze_scenario
    ...    ${{ {'scenario': 'Use Browser Library', 'context': 'web'} }}
    ${selenium_id}=    Set Variable    ${selenium}[session_id]
    ${browser_id}=    Set Variable    ${browser}[session_id]
    Should Be True    $selenium_id != $browser_id
    ${selenium_state}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': $selenium_id, 'sections': ['summary']} }}
    ${browser_state}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': $browser_id, 'sections': ['summary']} }}
    ${selenium_summary}=    Extract Summary    ${selenium_state}
    ${browser_summary}=    Extract Summary    ${browser_state}
    Result Field Should Be    ${selenium_summary}    explicit_library_preference    SeleniumLibrary
    Result Field Should Be    ${browser_summary}    explicit_library_preference    Browser
    ${sel_search_order}=    Evaluate    $selenium_summary.get('search_order', [])
    Should Not Contain    ${sel_search_order}    Browser
    ${br_search_order}=    Evaluate    $browser_summary.get('search_order', [])
    Should Not Contain    ${br_search_order}    SeleniumLibrary

*** Keywords ***
Extract Summary
    [Documentation]    Mirror of the pytest _extract_summary helper: assert the summary
    ...                section is present and successful, then return its session_info dict.
    [Arguments]    ${state}
    ${summary}=    Evaluate    $state['sections'].get('summary')
    Should Be True    $summary
    Result Field Should Be    ${summary}    success    ${True}
    ${info}=    Evaluate    $summary.get('session_info', {})
    Should Be True    $info
    RETURN    ${info}
