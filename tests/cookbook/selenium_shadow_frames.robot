*** Settings ***
Library           SeleniumLibrary    timeout=4s
Suite Setup       Open Browser    ${URL}    headlesschrome
Suite Teardown    Close All Browsers
Test Setup        Go To    ${URL}

*** Variables ***
${DEEP_QUERY}     SEPARATOR=\n
...    const find = (root, sel) => {
...      const hit = root.querySelector(sel); if (hit) return hit;
...      for (const el of root.querySelectorAll('*')) { if (el.shadowRoot) { const f = find(el.shadowRoot, sel); if (f) return f; } }
...      return null; };
...    return find(document, arguments[0]);

*** Test Cases ***
S01 Css Locator Does Not Pierce Shadow
    Page Should Not Contain Element    css=#user

S02 Js Returns Shadow Element As WebElement
    ${user}=    Execute Javascript    return document.querySelector('login-card').shadowRoot.querySelector('#user')
    Input Text    ${user}    alice
    ${btn}=    Execute Javascript    return document.querySelector('login-card').shadowRoot.querySelector('button')
    Click Element    ${btn}
    ${res}=    Execute Javascript    return document.querySelector('login-card').shadowRoot.querySelector('#login-result').textContent
    Should Be Equal    ${res}    Welcome, alice

S03 Selenium4 Shadow Root Property
    ${host}=    Get WebElement    css=login-card
    ${user}=    Evaluate    $host.shadow_root.find_element("css selector", "#user")
    Input Text    ${user}    carol
    ${btn}=    Evaluate    $host.shadow_root.find_element("css selector", "button")
    Click Element    ${btn}
    ${res}=    Evaluate    $host.shadow_root.find_element("css selector", "#login-result").text
    Should Be Equal    ${res}    Welcome, carol

S04 Deep Query Snippet Through Nested Shadow
    ${btn}=    Execute Javascript    ${DEEP_QUERY}    ARGUMENTS    button.deep
    Click Element    ${btn}
    ${st}=    Execute Javascript    ${DEEP_QUERY}    ARGUMENTS    .deep-status
    Element Text Should Be    ${st}    deep ok

S05 Same Origin Frame
    Select Frame    id=payframe
    Input Text    id=card    4111111111111111
    Click Button    Pay
    Element Text Should Be    id=pay-result    Paid with card ending 1111
    [Teardown]    Unselect Frame

S06 Nested Frames
    Select Frame    id=outer
    Select Frame    id=inner
    Click Button    Confirm
    Element Text Should Be    id=c    Confirmed
    [Teardown]    Unselect Frame

S07 Cross Origin Frame
    Select Frame    id=xframe
    Click Button    Accept cookies
    Element Text Should Be    id=consent    Accepted
    [Teardown]    Unselect Frame

S08 Iframe Inside Shadow Root
    ${frame}=    Execute Javascript    ${DEEP_QUERY}    ARGUMENTS    iframe#couponframe
    Select Frame    ${frame}
    Input Text    id=coupon    SAVE10
    Click Button    Apply
    Element Text Should Be    id=cr    Coupon SAVE10 applied
    [Teardown]    Unselect Frame

S09 Shadow Root Inside Iframe
    Select Frame    id=cardframe
    ${host}=    Get WebElement    css=card-input
    ${inp}=    Evaluate    $host.shadow_root.find_element("css selector", "#cardno")
    Input Text    ${inp}    4000123412349999
    Click Element    ${{ $host.shadow_root.find_element("css selector", "button") }}
    ${st}=    Evaluate    $host.shadow_root.find_element("css selector", "#card-status").text
    Should Be Equal    ${st}    card 9999 verified
    [Teardown]    Unselect Frame

S10 Forgotten Unselect Frame Hides Main Page
    Select Frame    id=payframe
    Run Keyword And Expect Error    *    Page Should Contain Element    css=h1
    Unselect Frame
    Page Should Contain Element    css=h1

S11 Closed Shadow Root Is Null
    ${root}=    Execute Javascript    return document.querySelector('closed-counter').shadowRoot
    Should Be Equal    ${root}    ${None}

S12 Closed Shadow Forced Open Via CDP Init Script
    ${lib}=    Get Library Instance    SeleniumLibrary
    Evaluate    $lib.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": "const o = Element.prototype.attachShadow; Element.prototype.attachShadow = function (i) { return o.call(this, {...i, mode: 'open'}); };"})
    Reload Page
    ${host}=    Get WebElement    css=closed-counter
    Click Element    ${{ $host.shadow_root.find_element("css selector", "button") }}
    ${v}=    Evaluate    $host.shadow_root.find_element("css selector", "output").text
    Should Be Equal    ${v}    1
