*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_adr_integration.py: ADR-006/007/008 server wiring —
...              the intent_action tool, tool-profile management via manage_session, and the
...              combined init->intent->execute workflow, over the REAL MCP protocol (the
...              original used an in-memory fastmcp Client). Real MCP protocol; deterministic.
...              One server per suite; tests isolate via distinct session_id values. Profile
...              activation is process-global, so profile-switching tests restore the full
...              tool surface via manage_session (present in every profile), mirroring the
...              pytest _restore_full_profile helper.
Resource         rfmcp.resource
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Test Cases ***
# ============================================================
# ADR-007: intent_action tool
# ============================================================

Intent Action Tool Listed
    [Documentation]    intent_action should appear in the tool list.
    @{tools}=    MCP.List Tools    ${HANDLE}
    ${names}=    Evaluate    [t.name for t in $tools]
    Should Contain    ${names}    intent_action

Navigate Intent Resolves For Browser
    [Documentation]    navigate intent with Browser library should resolve to Go To.
    ${sid}=    Set Variable    adr-nav-browser-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['Browser', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent': 'navigate', 'target': 'https://example.com', 'session_id': $sid} }}
    Should Be True    'Go To' in str($r) or 'success' in str($r).lower()

Navigate Intent Resolves For Selenium
    [Documentation]    navigate intent with SeleniumLibrary should resolve to Go To.
    ${sid}=    Set Variable    adr-nav-selenium-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['SeleniumLibrary', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent': 'navigate', 'target': 'https://example.com', 'session_id': $sid} }}
    Should Be True    'Go To' in str($r) or 'success' in str($r).lower()

Click Intent Browser Resolves To Click
    ${sid}=    Set Variable    adr-click-browser-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['Browser', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent': 'click', 'target': 'text=Login', 'session_id': $sid} }}
    Should Be True    'Click' in str($r) or 'success' in str($r).lower()

Click Intent Selenium Resolves To Click Element
    ${sid}=    Set Variable    adr-click-selenium-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['SeleniumLibrary', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent': 'click', 'target': 'id=submit', 'session_id': $sid} }}
    Should Be True    'Click Element' in str($r) or 'success' in str($r).lower()

Fill Intent Requires Value
    ${sid}=    Set Variable    adr-fill-novalue-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['Browser', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent': 'fill', 'target': 'id=username', 'session_id': $sid} }}
    Should Be True    'error' in str($r).lower() or 'value' in str($r).lower() or 'requires' in str($r).lower()

Fill Intent With Value Resolves
    ${sid}=    Set Variable    adr-fill-value-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['Browser', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent': 'fill', 'target': 'id=username', 'value': 'testuser', 'session_id': $sid} }}
    Should Be True    'Fill Text' in str($r) or 'Input Text' in str($r) or 'success' in str($r).lower()

Invalid Intent Returns Error
    [Documentation]    ADR-009 type constraints reject unknown intents at validation.
    ${sid}=    Set Variable    adr-err-intent-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['BuiltIn']} }}
    ${status}    ${value}=    Run Keyword And Ignore Error
    ...    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent': 'invalid_intent_xyz', 'session_id': $sid} }}
    ${errored}=    Evaluate    $status == 'FAIL' or (isinstance($value, dict) and bool($value.get('is_error')))
    Should Be True    $errored    invalid intent should surface an error (raise or is_error result)
    ${text}=    Evaluate    str($value).lower()
    Should Be True    'error' in $text or 'validation' in $text or 'literal' in $text

Intent Without Session Returns Error
    ${r}=    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent': 'click', 'target': 'text=Login', 'session_id': 'nonexistent-xyz'} }}
    Should Be True    'error' in str($r).lower() or 'session' in str($r).lower()

Intent Without Web Library Returns Error
    ${sid}=    Set Variable    adr-nolib-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent': 'click', 'target': 'text=Login', 'session_id': $sid} }}
    Should Be True    'error' in str($r).lower() or 'library' in str($r).lower()

# ============================================================
# ADR-006: Tool Profile via manage_session
# ============================================================

Init With Model Tier Small Context
    [Documentation]    Init with model_tier=small_context should activate a profile.
    ${sid}=    Set Variable    adr-tier-small-1
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': $sid, 'scenario': 'browser test', 'libraries': ['Browser', 'BuiltIn'], 'model_tier': 'small_context'} }}
    Should Be True    'success' in str($r).lower() or $sid in str($r)
    Restore Full Profile

Init With Model Tier Large Context
    [Documentation]    Init with model_tier=large_context should keep full tool set.
    ${sid}=    Set Variable    adr-tier-large-1
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['BuiltIn'], 'model_tier': 'large_context'} }}
    Should Be True    'success' in str($r).lower() or $sid in str($r)
    Restore Full Profile

Set Tool Profile Browser Exec
    ${sid}=    Set Variable    adr-prof-browser-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'set_tool_profile', 'session_id': $sid, 'profile': 'browser_exec'} }}
    Should Be True    'browser_exec' in str($r) or 'success' in str($r).lower()
    Restore Full Profile

Set Tool Profile Discovery
    ${sid}=    Set Variable    adr-prof-discovery-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'set_tool_profile', 'session_id': $sid, 'profile': 'discovery'} }}
    Should Be True    'discovery' in str($r) or 'success' in str($r).lower()
    Restore Full Profile

Set Tool Profile Minimal Exec
    ${sid}=    Set Variable    adr-prof-minimal-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'set_tool_profile', 'session_id': $sid, 'profile': 'minimal_exec'} }}
    Should Be True    'minimal_exec' in str($r) or 'success' in str($r).lower()
    Restore Full Profile

Set Invalid Profile Returns Error
    [Documentation]    ADR-009 type constraints reject unknown profile names at validation.
    ${sid}=    Set Variable    adr-err-profile-1
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['BuiltIn']} }}
    ${status}    ${value}=    Run Keyword And Ignore Error
    ...    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'set_tool_profile', 'session_id': $sid, 'profile': 'nonexistent_xyz'} }}
    ${errored}=    Evaluate    $status == 'FAIL' or (isinstance($value, dict) and bool($value.get('is_error')))
    Should Be True    $errored    invalid profile should surface an error (raise or is_error result)
    ${text}=    Evaluate    str($value).lower()
    Should Be True    'error' in $text or 'validation' in $text or 'literal' in $text

Tools Reduced After Small Profile
    [Documentation]    After switching to a small profile, tool count should decrease.
    @{initial}=    MCP.List Tools    ${HANDLE}
    ${initial_count}=    Get Length    ${initial}
    ${sid}=    Set Variable    adr-reduce-1
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['Browser', 'BuiltIn'], 'model_tier': 'small_context'} }}
    @{reduced}=    MCP.List Tools    ${HANDLE}
    ${reduced_count}=    Get Length    ${reduced}
    # Restore before assertion (in case it fails), mirroring the pytest ordering.
    Restore Full Profile
    Should Be True    ${reduced_count} < ${initial_count}
    ...    Expected tools to be reduced: ${reduced_count} >= ${initial_count}

# ============================================================
# Combined: Intent + BuiltIn execution workflow
# ============================================================

Full Workflow Init With Model Tier Then Builtin
    [Documentation]    Full workflow: init with model_tier, execute BuiltIn, verify.
    ${sid}=    Set Variable    adr-combo-1
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['BuiltIn'], 'model_tier': 'small_context'} }}
    # Restore profile so execute_step is available.
    Restore Full Profile
    ${r}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['ADR integration test'], 'session_id': $sid} }}
    Should Be True    'pass' in str($r).lower() or 'success' in str($r).lower()

Intent Then Execute Step Same Session
    [Documentation]    Intent resolution followed by execute_step in same session.
    ${sid}=    Set Variable    adr-combo-2
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['Browser', 'BuiltIn']} }}
    ${intent}=    Rf Tool    ${HANDLE}    intent_action    ${{ {'intent': 'navigate', 'target': 'https://example.com', 'session_id': $sid} }}
    Should Not Be Equal    ${intent}    ${None}
    ${r}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['Post-intent test'], 'session_id': $sid} }}
    Should Be True    'pass' in str($r).lower() or 'success' in str($r).lower()

Profile Switch Then Execute
    [Documentation]    Switch profile, then verify execute_step still works.
    ${sid}=    Set Variable    adr-combo-3
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'init', 'session_id': $sid, 'libraries': ['BuiltIn']} }}
    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'set_tool_profile', 'session_id': $sid, 'profile': 'minimal_exec'} }}
    ${r}=    Rf Tool    ${HANDLE}    execute_step    ${{ {'keyword': 'Log', 'arguments': ['After profile switch'], 'session_id': $sid} }}
    Should Be True    'pass' in str($r).lower() or 'success' in str($r).lower()
    Restore Full Profile

*** Keywords ***
Restore Full Profile
    [Documentation]    Mirror the pytest _restore_full_profile helper. Tool-profile
    ...    activation is process-global (FastMCP tool visibility), so tests that switch
    ...    to a small profile restore the full surface afterward. manage_session is a
    ...    member of every profile, so the MCP-level restore always reaches the server.
    ...    Best-effort, like the original's try/except.
    Run Keyword And Ignore Error
    ...    Rf Tool    ${HANDLE}    manage_session    ${{ {'action': 'set_tool_profile', 'session_id': 'restore', 'profile': 'full'} }}
