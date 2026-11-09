*** Settings ***
Library           Browser    timeout=4s
Suite Setup       New Browser    chromium    headless=True
Test Setup        New Page    ${URL}
Test Teardown     Close Page

*** Test Cases ***
D01 Aria Snapshot Inside Frame
    ${snap}=    Get Aria Snapshot    id=xframe >>> body
    Should Contain    ${snap}    Accept cookies
    ${snap}=    Get Aria Snapshot    id=outer >>> id=inner >>> body
    Should Contain    ${snap}    Confirm

D02 Evaluate JavaScript Inside Frame
    ${v}=    Evaluate JavaScript    id=payframe >>> id=card    (el) => { el.value = '5555444433332222'; return el.ownerDocument.title || el.ownerDocument.URL; }
    Should Contain    ${v}    pay.html
    Get Property    id=payframe >>> id=card    value    ==    5555444433332222

D03 Force Closed Shadow Open By Recreating Element
    Evaluate JavaScript    ${None}
    ...    () => { const orig = Element.prototype.attachShadow; Element.prototype.attachShadow = function (o) { return orig.call(this, {...o, mode: 'open'}); }; for (const el of document.querySelectorAll('closed-counter')) { el.replaceWith(document.createElement(el.localName)); } }
    Click    closed-counter >> role=button[name="Increment"]
    Get Text    closed-counter >> output    ==    1

D04 Shadow Host Text Via JS For Open Roots
    ${t}=    Evaluate JavaScript    login-card    (host) => host.shadowRoot.querySelector('h2').textContent
    Should Be Equal    ${t}    Account

D05 Strict Mode Over Shadow Duplicates
    ${n}=    Get Element Count    css=button
    Log    ${n}
    Run Keyword And Expect Error    *strict mode violation*    Click    css=button
