*** Settings ***
Library           Browser    timeout=4s
Suite Setup       New Browser    chromium    headless=True
Test Setup        New Page    ${URL}
Test Teardown     Close Page

*** Test Cases ***
B01 Open Shadow CSS Pierces Automatically
    Fill Text    css=#user    alice
    Click    text=Sign in
    Get Text    css=#login-result    ==    Welcome, alice

B02 Host Scoped Chain Into Open Shadow
    Fill Text    login-card >> input[name="user"]    bob
    Click    login-card >> role=button[name="Sign in"]
    Get Text    login-card >> role=status    ==    Welcome, bob

B03 XPath Does Not Pierce Shadow
    Get Element Count    xpath=//input[@id="user"]    ==    0

B04 Nested Open Shadow Pierces Too
    Click    text=Deep action
    Get Text    css=.deep-status    ==    deep ok

B05 Same Origin Iframe With Frame Piercing
    Fill Text    id=payframe >>> id=card    4111111111111111
    Click    id=payframe >>> text=Pay
    Get Text    id=payframe >>> id=pay-result    ==    Paid with card ending 1111

B06 Nested Iframes Chain
    Click    id=outer >>> id=inner >>> text=Confirm
    Get Text    id=outer >>> id=inner >>> id=c    ==    Confirmed

B07 Cross Origin Iframe
    Click    id=xframe >>> text=Accept cookies
    Get Text    id=xframe >>> id=consent    ==    Accepted

B08 Iframe Inside Open Shadow Root
    Fill Text    css=shadow-frame >> iframe >>> id=coupon    SAVE10
    Click    css=shadow-frame >> iframe >>> text=Apply
    Get Text    css=shadow-frame >> iframe >>> id=cr    ==    Coupon SAVE10 applied

B09 Shadow Root Inside Iframe
    Fill Text    id=cardframe >>> id=cardno    4000123412349999
    Click    id=cardframe >>> text=Verify card
    Get Text    id=cardframe >>> id=card-status    ==    card 9999 verified

B10 Frame Without Piercing Fails
    Run Keyword And Expect Error    *    Click    text=Accept cookies

B11 Closed Shadow Is Not Reachable By Selector
    Run Keyword And Expect Error    *    Click    closed-counter >> button

B16 Aria Snapshot Of Page Shows Shadow And Frames
    ${snap}=    Get Aria Snapshot    css=body
    Log    ${snap}
    Should Contain    ${snap}    Sign in
    Should Contain    ${snap}    Deep action

B17 Page Map Snippet Lists Shadow Hosts And Frames
    ${map}=    Evaluate JavaScript    ${None}
    ...    () => { const out = []; const walk = (root, path) => { for (const el of root.querySelectorAll('*')) { if (el.shadowRoot) { out.push(path + el.localName + ' [open shadow]'); walk(el.shadowRoot, path + el.localName + ' >> '); } else if (el.localName.includes('-') && !el.shadowRoot) { out.push(path + el.localName + ' [custom element: closed shadow or none]'); } if (el.localName === 'iframe') { out.push(path + 'iframe#' + el.id + ' src=' + el.getAttribute('src') + ' -> use >>>'); } } }; walk(document, ''); return out; }
    Log    ${map}
    Should Contain    ${map}    login-card [open shadow]
    Should Contain    ${map}    closed-counter [custom element: closed shadow or none]
    Should Contain    ${map}    shadow-frame >> iframe#couponframe src=coupon.html -> use >>>
