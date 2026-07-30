*** Settings ***
Documentation     Real Browser-Library (Playwright, headless Chromium) keyword-execution,
...               page-source, and intent_action(extract) tests — the MCP-observable subset
...               ported from tests/integration/test_real_browser_prevalidation.py (change:
...               agenteval-port-partial-integration). The internal ``_pre_validate_element``
...               tests and the OBS-01 verdict-equivalence tests (which reach into the executor)
...               stay in the trimmed pytest file. Skips cleanly when headless Chromium is not
...               provisioned (mirrors the pytest ``skipif``), so the always-on CI tier stays green.
Resource          rfmcp.resource
Suite Setup       Establish Browser Session Or Skip
Suite Teardown    Close Browser And Stop

*** Variables ***
${BSID}           real_browser_preval

*** Keywords ***
Establish Browser Session Or Skip
    Start Rf-mcp Server
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action':'init','session_id':'${BSID}','libraries':['Browser','BuiltIn']} }}
    IF    not ${init.get('success', False)}
        Skip    Browser session init failed: ${init}
    END
    ${nb}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'New Browser','arguments':['chromium','headless=True'],'session_id':'${BSID}'} }}
    IF    not ${nb.get('success', False)}
        Skip    Headless Chromium unavailable (rfbrowser init not run?): ${nb.get('error', '')}
    END
    ${np}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'New Page','arguments':['https://example.com'],'session_id':'${BSID}'} }}
    IF    not ${np.get('success', False)}
        Skip    New Page failed (network to example.com?): ${np.get('error', '')}
    END

Close Browser And Stop
    Run Keyword And Ignore Error    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Close Browser','arguments':['ALL'],'session_id':'${BSID}'} }}
    Stop Rf-mcp Server

Navigate To OBS06 Fixture
    ${url}=    Set Variable    data:text/html,<html><head><title>OBS-06 Fixture</title></head><body><div id='order-display' data-order-id='ORD-1007696'>Order ORD-1007696</div><div class='item'>A</div><div class='item'>B</div><div class='item'>C</div><input id='user-name' value='alice' /></body></html>
    ${nav}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Go To','arguments':[$url],'session_id':'${BSID}'} }}
    Result Field Should Be    ${nav}    success    ${True}

*** Test Cases ***
# --- keyword execution (on example.com) ---
Get Title Returns Text
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Get Title','arguments':[],'session_id':'${BSID}','assign_to':'title'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${val}=    Assigned Variable    ${r}    title
    Should Contain    ${val.lower()}    example    Expected 'example' in title, got: ${val}

Get Url Returns Url
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Get Url','arguments':[],'session_id':'${BSID}','assign_to':'url'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${val}=    Assigned Variable    ${r}    url
    Should Contain    ${val}    example.com    Expected 'example.com' in URL, got: ${val}

Get Page Source Returns Html
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Get Page Source','arguments':[],'session_id':'${BSID}','assign_to':'source'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${val}=    Assigned Variable    ${r}    source
    Should Be True    '<html' in $val.lower() or '<body' in $val.lower()    Expected HTML, got: ${val}[:200]

# --- page source via get_session_state ---
Page Source Retrieval Returns Html
    ${r}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id':'${BSID}','sections':['page_source']} }}
    Result Field Should Be    ${r}    success    ${True}
    ${ps}=    Set Variable    ${r}[sections][page_source]
    Should Be True    ${ps.get('success', False)}    Page source failed: ${ps}
    Should Be True    ${ps.get('page_source_length', 0)} > 0

Page Source Has Context
    ${r}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id':'${BSID}','sections':['page_source']} }}
    ${ctx}=    Set Variable    ${r}[sections][page_source][context]
    ${title}=    Set Variable    ${ctx.get('page_title', '')}
    Should Not Be Empty    ${title}    Expected non-empty page_title in context, got: ${ctx}

# --- intent_action(extract) end-to-end (OBS-06, self-navigates to data: fixtures) ---
Extract Text Returns Element Text
    Navigate To OBS06 Fixture
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent':'extract','target':'id=order-display','mode':'text','session_id':'${BSID}'} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Field Should Be    ${r}    extracted_value    Order ORD-1007696

Extract Attribute Returns Attribute Value
    Navigate To OBS06 Fixture
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent':'extract','target':'id=order-display','mode':'attribute','attribute_name':'data-order-id','session_id':'${BSID}'} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Field Should Be    ${r}    extracted_value    ORD-1007696

Extract Count Returns Match Count
    Navigate To OBS06 Fixture
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent':'extract','target':'css=.item','mode':'count','session_id':'${BSID}'} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Be True    int(${r}[extracted_value]) == 3

Extract Value Returns Input Value
    Navigate To OBS06 Fixture
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent':'extract','target':'id=user-name','mode':'value','session_id':'${BSID}'} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Field Should Be    ${r}    extracted_value    alice

Extract Title Returns Page Title
    Navigate To OBS06 Fixture
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent':'extract','mode':'title','session_id':'${BSID}'} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Field Should Be    ${r}    extracted_value    OBS-06 Fixture

Extract Url Returns Current Url
    Navigate To OBS06 Fixture
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent':'extract','mode':'url','session_id':'${BSID}'} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Contain    ${r}[extracted_value]    data:text/html    expected data: URL, got: ${r}[extracted_value]

Extract With Assign To Captures Variable
    Navigate To OBS06 Fixture
    ${r}=    Rf Tool    ${HANDLE}    intent_action
    ...    ${{ {'intent':'extract','target':'id=order-display','mode':'attribute','attribute_name':'data-order-id','assign_to':'order_id','session_id':'${BSID}'} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Field Should Be    ${r}    extracted_value    ORD-1007696

# --- OBS-10: Drag And Drop pre-scrolls an off-screen source into view ---
Drag And Drop Pre Scrolls Off Screen Source
    [Documentation]    OBS-10 contract: Drag And Drop must pre-scroll an off-screen source
    ...    into the viewport. Asserts window scroll-top before (~0) and after (>1000) the drag,
    ...    reading only the tool-result ``output`` (MCP-observable). The unit-wiring counterpart
    ...    stays in tests/unit/test_browser_drag_and_drop_prescroll.py.
    ${url}=    Set Variable    data:text/html,<html><head><title>OBS-10 Fixture</title></head><body><div id='target' style='width:200px;height:80px;border:2px dashed blue;background:lightyellow'>drop here</div><div style='height:2200px'></div><div id='source' draggable='true' style='width:120px;height:40px;background:lightblue;cursor:grab'>drag me</div></body></html>
    ${nav}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Go To','arguments':[$url],'session_id':'${BSID}'} }}
    Result Field Should Be    ${nav}    success    ${True}
    ${pre}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Get Scroll Position','arguments':[],'session_id':'${BSID}'} }}
    Result Field Should Be    ${pre}    success    ${True}
    ${pre_top}=    Scroll Top    ${pre.get('output')}
    Should Be True    ${pre_top} is not None and ${pre_top} < 100    Precondition: page should start at scroll-top ~0; got ${pre_top}
    ${drag}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Drag And Drop','arguments':['id=source','id=target'],'session_id':'${BSID}'} }}
    Result Field Should Be    ${drag}    success    ${True}
    ${post}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword':'Get Scroll Position','arguments':[],'session_id':'${BSID}'} }}
    Result Field Should Be    ${post}    success    ${True}
    ${post_top}=    Scroll Top    ${post.get('output')}
    Should Be True    ${post_top} is not None and ${post_top} > 1000    OBS-10: page should scroll to bring off-screen source into view; pre=${pre_top} post=${post_top}