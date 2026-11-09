*** Settings ***
Library     RequestsLibrary
Library     Collections
Suite Setup    Create Session    booker    ${BASE_URL}    verify=True

*** Test Cases ***
Ping Service
    [Tags]    smoke    api
    ${resp}=    GET On Session    booker    /ping    expected_status=201

Get Booking Ids
    [Tags]    api
    ${resp}=    GET On Session    booker    /booking
    Should Not Be Empty    ${resp.json()}

Create Booking
    [Tags]    api    regression
    ${dates}=    Create Dictionary    checkin=2026-01-01    checkout=2026-01-05
    ${body}=    Create Dictionary    firstname=Robot    lastname=Lab    totalprice=111
    ...    depositpaid=${True}    bookingdates=${dates}    additionalneeds=Breakfast
    ${resp}=    POST On Session    booker    /booking    json=${body}
    Should Be Equal As Strings    ${resp.json()}[booking][firstname]    Robot
