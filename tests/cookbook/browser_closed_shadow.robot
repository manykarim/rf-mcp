*** Settings ***
Documentation     Browser library and closed shadow roots / frames in page snapshots: the negatives the
...               rf-locators cookbook states (closed roots are invisible to selectors, snapshots and host
...               text; a page snapshot shows each iframe only as `- iframe`).
Library           Browser    timeout=4s
Suite Setup       New Browser    chromium    headless=True
Test Setup        New Page    ${URL}
Test Teardown     Close Page

*** Test Cases ***
C01 Host Get Text Does Not Include Closed Shadow Text
    ${text}=    Get Text    css=closed-counter
    Should Not Contain    ${text}    Count

C02 Closed Shadow Is Missing From Aria Snapshot
    ${snap}=    Get Aria Snapshot    css=closed-counter
    Should Not Contain    ${snap}    Increment

C03 Role Selector Does Not Reach Closed Shadow
    Run Keyword And Expect Error    *Timeout*    Click    role=button[name="Increment"]

C04 Text Selector Does Not Reach Closed Shadow
    Run Keyword And Expect Error    *    Click    text=+1

C05 Page Snapshot Shows Each Iframe Only As A Placeholder
    ${snap}=    Get Aria Snapshot    css=body
    Should Contain    ${snap}    - iframe
    Should Not Contain    ${snap}    Accept cookies
    ${frame}=    Get Aria Snapshot    id=xframe >>> body
    Should Contain    ${frame}    Accept cookies
