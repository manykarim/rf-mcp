*** Settings ***
Documentation     Real SeleniumLibrary (headless Chrome) keyword-execution and page-source
...               tests — the MCP-observable subset ported from
...               tests/integration/test_real_selenium_prevalidation.py (change:
...               agenteval-port-partial-integration). The internal ``_pre_validate_element``
...               tests stay in the trimmed pytest file. Skips cleanly when headless Chrome is
...               not provisioned (mirrors the pytest ``skipif``). This suite spawns its OWN
...               rf-mcp subprocess, so SeleniumLibrary and Browser Library never share a process
...               (the web_automation exclusion group that forced separate pytest invocations).
Resource          rfmcp.resource
Suite Setup       Establish Selenium Session Or Skip
Suite Teardown    Close Browsers And Stop

*** Variables ***
${SSID}           real_selenium_preval

*** Keywords ***
Establish Selenium Session Or Skip
    Start Rf-mcp Server
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action':'init','session_id':'${SSID}','libraries':['SeleniumLibrary','BuiltIn']} }}
    IF    not ${init.get('success', False)}
        Skip    Selenium session init failed: ${init}
    END
    ${ob}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Open Browser','arguments':['https://example.com','headlesschrome'],'session_id':'${SSID}'} }}
    IF    not ${ob.get('success', False)}
        Skip    Headless Chrome unavailable (chrome/chromedriver not provisioned?): ${ob.get('error', '')}
    END

Close Browsers And Stop
    Run Keyword And Ignore Error    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Close All Browsers','arguments':[],'session_id':'${SSID}'} }}
    Stop Rf-mcp Server

*** Test Cases ***
# --- keyword execution (on example.com) ---
Get Title Returns Text
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Get Title','arguments':[],'session_id':'${SSID}','assign_to':'title'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${val}=    Assigned Variable    ${r}    title
    Should Contain    ${val.lower()}    example    Expected 'example' in title, got: ${val}

Get Location Returns Url
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Get Location','arguments':[],'session_id':'${SSID}','assign_to':'url'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${val}=    Assigned Variable    ${r}    url
    Should Contain    ${val}    example.com    Expected 'example.com' in URL, got: ${val}

Get Source Returns Html
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Get Source','arguments':[],'session_id':'${SSID}','assign_to':'source'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${val}=    Assigned Variable    ${r}    source
    Should Be True    '<html' in $val.lower() or '<body' in $val.lower()    Expected HTML, got: ${val}[:200]

Timeout Not Injected For Action Keywords
    [Documentation]    P0 fix: Click Element must NOT receive an injected ``timeout=`` argument.
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Click Element','arguments':['css=h1'],'session_id':'${SSID}'} }}
    Result Field Should Be    ${r}    success    ${True}

# --- page source via get_session_state ---
Page Source Retrieval Returns Html
    ${r}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id':'${SSID}','sections':['page_source']} }}
    Result Field Should Be    ${r}    success    ${True}
    ${ps}=    Set Variable    ${r}[sections][page_source]
    Should Be True    ${ps.get('success', False)}    Page source failed: ${ps}
    Should Be True    ${ps.get('page_source_length', 0)} > 0

Page Source Has Context
    ${r}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id':'${SSID}','sections':['page_source']} }}
    ${ctx}=    Set Variable    ${r}[sections][page_source][context]
    ${title}=    Set Variable    ${ctx.get('page_title', '')}
    Should Not Be Empty    ${title}    Expected non-empty page_title in context, got: ${ctx}