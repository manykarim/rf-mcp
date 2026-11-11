*** Settings ***
Resource    common.resource
Metadata    Component    pricing

*** Test Cases ***
Add Two Numbers
    [Tags]    smoke    calc
    ${sum}=    Add Numbers    2    3
    Should Be Equal As Numbers    ${sum}    5

Ten Percent Discount
    [Tags]    calc    regression
    [Documentation]    Fails because Apply Discount has a bug
    Price After Discount Should Be    200    10    180

Format Money In Euro
    [Tags]    smoke
    ${s}=    Format Money    3.5
    Should Be Equal    ${s}    3.50 EUR

Cart Contains Items
    [Tags]    regression
    ${cart}=    Create Cart    apple    pear
    Length Should Be    ${cart}    3
