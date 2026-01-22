*** Settings ***
Documentation    Form validation and input testing with AILibrary
...
...    Examples of testing various form scenarios using
...    natural language descriptions.

Library    Browser
Library    robotmcp.lib.AILibrary
...    provider=anthropic
...    api_key=%{ANTHROPIC_API_KEY}

Suite Teardown    Close Browser

*** Variables ***
${FORM_URL}    https://www.saucedemo.com


*** Test Cases ***
Test Login Form Validation
    [Documentation]    Test form validation using AI keywords

    New Browser    chromium    headless=true
    New Page    ${FORM_URL}

    # Test empty submission
    Do    Click the login button without entering any credentials
    Check    Error message is displayed for missing username

    # Test missing password
    Do    Enter standard_user as username
    Do    Click login
    Check    Error message shows password is required

    # Test invalid credentials
    Do    Clear the form and enter invalid_user with wrong_password
    Do    Submit the login form
    Check    Error indicates invalid credentials


Test Form Input Types
    [Documentation]    Test various input field types

    New Browser    chromium    headless=true
    New Page    ${FORM_URL}

    # Text input
    Do    Type standard_user into the username field
    ${username}=    Ask    What text is in the username field?
    Should Contain    ${username}    standard_user

    # Password input (masked)
    Do    Enter secret_sauce in the password field
    Check    Password field shows masked characters


Test Form Clearing
    [Documentation]    Test clearing form fields

    New Browser    chromium    headless=true
    New Page    ${FORM_URL}

    Do    Fill username with test_value
    Do    Fill password with test_pass

    Do    Clear all form fields
    Check    Username field is empty
    Check    Password field is empty


Test Checkout Form
    [Documentation]    Complete checkout form testing

    New Browser    chromium    headless=true
    New Page    ${FORM_URL}

    # Setup - login and add item
    Do    Login as standard_user with password secret_sauce
    Do    Add any product to cart
    Do    Go to cart and proceed to checkout

    # Test form fields
    Do    Enter John in the first name field
    Do    Enter Doe in the last name field
    Do    Enter 12345 in the postal code field

    Check    All checkout fields are filled

    # Submit and verify
    Do    Continue to the next step
    Check    Order overview page is displayed


Test Form Field Limits
    [Documentation]    Test input field character limits and validation

    New Browser    chromium    headless=true
    New Page    ${FORM_URL}

    # Test long input
    ${long_text}=    Set Variable    ${{ 'a' * 100 }}
    Do    Enter ${long_text} in the username field

    ${entered}=    Ask    How many characters are in the username field?
    Log    Characters entered: ${entered}


Test Form Tab Navigation
    [Documentation]    Test keyboard navigation in forms

    New Browser    chromium    headless=true
    New Page    ${FORM_URL}

    Do    Click on the username field
    Do    Type standard_user
    Do    Press Tab to move to password field
    Do    Type secret_sauce
    Do    Press Enter to submit

    Check    Login was successful or error is shown


Test Form Error Messages
    [Documentation]    Verify specific error message content

    New Browser    chromium    headless=true
    New Page    ${FORM_URL}

    Do    Try to login with locked_out_user and secret_sauce
    Check    Error message mentions locked out

    ${error}=    Ask    What is the exact error message shown?
    Should Contain    ${error}    locked out    ignore_case=True


Test Form Autocomplete
    [Documentation]    Test form field autocomplete behavior

    New Browser    chromium    headless=true
    New Page    ${FORM_URL}

    # First login
    Do    Login as standard_user with password secret_sauce
    Check    Login successful

    # Logout and check autocomplete
    Do    Open the menu and logout
    Check    Login page is displayed again

    # Check if username might be remembered
    ${username_value}=    Ask    Is there any pre-filled text in the username field?
    Log    Pre-filled username: ${username_value}
