*** Settings ***
Library    CalcLib
Library    Collections

*** Keywords ***
Price After Discount Should Be
    [Arguments]    ${price}    ${percent}    ${expected}
    ${result}=    Apply Discount    ${price}    ${percent}
    Should Be Equal As Numbers    ${result}    ${expected}

Create Cart
    [Arguments]    @{items}
    ${cart}=    Create List    @{items}
    RETURN    ${cart}
