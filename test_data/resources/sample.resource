*** Settings ***

*** Variables ***
${HELLO}        world
${SUM}          0

*** Keywords ***
Hello Message
    [Arguments]    ${name}=Robot
    ${msg}=    Set Variable    Hello, ${name}!
    RETURN    ${msg}

Add Numbers
    [Arguments]    ${a}    ${b}
    ${res}=    Evaluate    int(${a}) + int(${b})
    Set Test Variable    ${SUM}    ${res}
    RETURN    ${res}

No Operation
    Log    Doing nothing

Greet And Log
    [Arguments]    ${who}=World
    ${text}=    Hello Message    ${who}
    Log    ${text}

