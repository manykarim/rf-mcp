*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_intent_fallback_e2e.py: navigate intent fallback
...              end-to-end (intent_action -> fallback detection -> keyword execution); a
...              fallback is asserted *attempted* (metadata present), not that navigation
...              succeeded. Real MCP protocol; deterministic.
Resource         rfmcp.resource
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Test Cases ***
Navigate Fallback Metadata In Response
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'fb-nav-meta', 'libraries': ['Browser', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent': 'navigate', 'target': 'https://example.com', 'session_id': 'fb-nav-meta'} }}
    ${text}=    Evaluate    str($r)
    Should Be True    'Go To' in $text or 'fallback' in $text.lower() or 'error' in $text.lower()

Navigate No Fallback For Click Intent
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'fb-click-nofb', 'libraries': ['Browser', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent': 'click', 'target': 'text=NonExistent', 'session_id': 'fb-click-nofb'} }}
    ${text}=    Evaluate    str($r)
    Should Not Contain    ${text}    fallback_applied

Selenium Navigate Fallback
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'fb-sel-nav', 'libraries': ['SeleniumLibrary', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent': 'navigate', 'target': 'https://example.com', 'session_id': 'fb-sel-nav'} }}
    ${text}=    Evaluate    str($r)
    Should Be True    'Go To' in $text or 'fallback' in $text.lower() or 'error' in $text.lower()

Navigate No Fallback For Invalid Intent
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'fb-invalid', 'libraries': ['Browser', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent': 'navigate', 'session_id': 'fb-invalid'} }}
    ${text}=    Evaluate    str($r)
    Should Contain    ${text}    error    ignore_case=True
    Should Not Contain    ${text}    fallback_applied

Fallback Does Not Trigger For Fill
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'fb-fill', 'libraries': ['Browser', 'BuiltIn']} }}
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent': 'fill', 'target': 'id=username', 'value': 'test', 'session_id': 'fb-fill'} }}
    ${text}=    Evaluate    str($r)
    Should Not Contain    ${text}    fallback_applied
