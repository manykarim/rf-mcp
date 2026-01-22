*** Settings ***
Documentation    Basic usage examples for AILibrary
...
...    This example demonstrates the three core keywords:
...    - Do: Execute actions
...    - Check: Verify conditions
...    - Ask: Extract information

Library    Browser
Library    robotmcp.lib.AILibrary
...    provider=openai
...    api_key=%{OPENAI_API_KEY}
...    model=gpt-5-mini

Test Setup    Open Demo Store
Test Teardown    Close And Stop Recording


*** Variables ***
${STORE_URL}    https://www.saucedemo.com


*** Test Cases ***
Login With Valid Credentials
    [Documentation]    Use Do keyword to perform login action
    Do    Login as standard_user with password secret_sauce
    Check    Products page is displayed with inventory items
    Export Test Suite    path=${OUTPUT_DIR}/login_test_suite.robot
    


Extract Product Information
    [Documentation]    Use Ask keyword to get information from the page

    Do    Login as standard_user with password secret_sauce
    ${product_name}=    Ask    What is the name of the first product?
    ${product_price}=    Ask    What is the price of the first product?

    Log    Product: ${product_name}
    Log    Price: ${product_price}

    Should Not Be Empty    ${product_name}
    Should Contain    ${product_price}    $



Add Product To Cart
    [Documentation]    Use Do and Check keywords for shopping cart
    Do    Login as standard_user with password secret_sauce
    Do    Add the Sauce Labs Backpack to the shopping cart
    ${cart_count}=    Ask    How many items are in the cart badge?
    Log    Cart shows: ${cart_count}


Complete Purchase Flow
    [Documentation]    Full end-to-end purchase using AI keywords
    Do    Login as standard_user with password secret_sauce
    # Add items
    Do    Add the Sauce Labs Backpack to cart
    Do    Click on shopping cart
    Do    Click the Checkout button

    # Fill form
    Do    Fill the first name field with John
    Do    Fill the last name field with Doe
    Do    Fill the zip code field with 12345
    Do    Click the Continue button

    # Complete
    Do    Click the Finish button
    Check    Order confirmation message is displayed

    ${confirmation}=    Ask    What does the confirmation message say?
    Log    Order confirmed: ${confirmation}


*** Keywords ***
Open Demo Store
    Start Recording
    New Browser    chromium    headless=false
    New Page    ${STORE_URL}

Close And Stop Recording
    Close Browser
    Stop Recording
    Export Test Suite    path=${OUTPUT_DIR}/${{ $TEST_NAME.replace(" ", "_") }}.robot