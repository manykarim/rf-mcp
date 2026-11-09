*** Settings ***
Library           SeleniumLibrary    timeout=4s
Suite Setup       Open Browser    ${URL}    headlesschrome
Suite Teardown    Close All Browsers

*** Test Cases ***
S13 Read Closed Shadow Via CDP Keeps State
    ${w}    ${h}=    Get Element Size    css=closed-counter
    Click Element At Coordinates    css=closed-counter    ${{ int(10 - $w / 2) }}    0
    Click Element At Coordinates    css=closed-counter    ${{ int(10 - $w / 2) }}    0
    ${html}=    Closed Shadow Outer HTML    output[aria-label="Count value"]
    Should Contain    ${html}    >2</output>

*** Keywords ***
Closed Shadow Outer HTML
    [Documentation]    Chrome only: DevTools search pierces closed shadow roots.
    [Arguments]    ${css}
    ${lib}=    Get Library Instance    SeleniumLibrary
    ${drv}=    Set Variable    ${lib.driver}
    Evaluate    $drv.execute_cdp_cmd("DOM.getDocument", {"depth": -1, "pierce": True})
    ${search}=    Evaluate    $drv.execute_cdp_cmd("DOM.performSearch", {"query": $css})
    ${hits}=    Evaluate    $drv.execute_cdp_cmd("DOM.getSearchResults", {"searchId": $search["searchId"], "fromIndex": 0, "toIndex": 1})
    ${out}=    Evaluate    $drv.execute_cdp_cmd("DOM.getOuterHTML", {"nodeId": $hits["nodeIds"][0]})
    RETURN    ${out}[outerHTML]
