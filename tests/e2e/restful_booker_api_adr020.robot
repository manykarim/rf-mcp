*** Settings ***
Documentation    End-to-end API test suite for the Restful Booker API
...              covering CRUD operations and authentication.
...              Generated via rf-mcp MCP tools (ADR-020 workflow).
Library         RequestsLibrary
Library         Collections
Suite Setup     Create Session    booker    https://restful-booker.herokuapp.com
Test Tags       api    restful-booker    e2e

*** Test Cases ***
Read Booking
    [Documentation]    Verify that we can read booking data from the Restful Booker API
    [Tags]    api    get    smoke
    ${response} =    GET On Session    booker    /booking
    Should Be Equal As Strings    ${response.status_code}    200
    ${all_bookings} =    GET On Session    booker    /booking
    ${first_id} =    Evaluate    $all_bookings.json()[0]['bookingid']
    ${booking} =    GET On Session    booker    /booking/${first_id}
    Should Not Be Empty    ${booking.json()}[firstname]
    Should Not Be Empty    ${booking.json()}[lastname]

Create New Booking
    [Documentation]    Verify that we can create a new booking via POST request
    [Tags]    api    post    crud
    ${booking_body} =    Evaluate    {'firstname': 'Jim', 'lastname': 'Brown', 'totalprice': 111, 'depositpaid': True, 'bookingdates': {'checkin': '2024-01-01', 'checkout': '2024-01-05'}, 'additionalneeds': 'Breakfast'}
    ${create_response} =    POST On Session    booker    /booking    json=${booking_body}
    Should Be Equal As Strings    ${create_response.status_code}    200
    Dictionary Should Contain Key    ${create_response.json()}    bookingid
    Should Be Equal As Strings    ${create_response.json()}[booking][firstname]    Jim

Authenticate Admin
    [Documentation]    Verify that we can authenticate as admin and receive a token
    [Tags]    api    auth
    ${auth_data} =    Create Dictionary    username=admin    password=password123
    ${auth_response} =    POST On Session    booker    /auth    json=${auth_data}
    Should Be Equal As Strings    ${auth_response.status_code}    200
    Dictionary Should Contain Key    ${auth_response.json()}    token
    Should Not Be Empty    ${auth_response.json()}[token]

Delete Booking While Authenticated
    [Documentation]    Create a booking, authenticate, then delete it and verify deletion
    [Tags]    api    delete    auth    crud
    ${delete_body} =    Evaluate    {'firstname': 'ToDelete', 'lastname': 'User', 'totalprice': 50, 'depositpaid': True, 'bookingdates': {'checkin': '2024-06-01', 'checkout': '2024-06-05'}, 'additionalneeds': 'None'}
    ${new_booking} =    POST On Session    booker    /booking    json=${delete_body}
    ${booking_id} =    Evaluate    str($new_booking.json()['bookingid'])
    ${auth_creds} =    Create Dictionary    username=admin    password=password123
    ${token_response} =    POST On Session    booker    /auth    json=${auth_creds}
    ${token} =    Evaluate    $token_response.json()['token']
    ${delete_headers} =    Create Dictionary    Cookie=token=${token}
    ${delete_response} =    DELETE On Session    booker    /booking/${booking_id}    expected_status=201    headers=${delete_headers}
    Should Be Equal As Strings    ${delete_response.status_code}    201
    ${verify_deleted} =    GET On Session    booker    /booking/${booking_id}    expected_status=404
    Should Be Equal As Strings    ${verify_deleted.status_code}    404
