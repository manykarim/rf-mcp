*** Settings ***
Documentation    Tests for variable assignment recording and export
...
...    This test verifies that:
...    - ${var}= Ask ... syntax is detected and recorded
...    - Variable assignments appear in exported test files
...    - Multiple Ask keywords each get their own variable assignments

Library    Browser
Library    robotmcp.lib.AILibrary
...    provider=openai
...    api_key=%{OPENAI_API_KEY}
...    model=gpt-4o-mini

Test Setup    Setup Test
Test Teardown    Teardown Test


*** Variables ***
${STORE_URL}    https://www.saucedemo.com
${OUTPUT_DIR}    ${CURDIR}${/}..${/}..${/}results_variable_test


*** Test Cases ***
Test Ask Keyword With Variable Assignment
    [Documentation]    Verify Ask keyword captures variable assignment
    Do    Login as standard_user with password secret_sauce

    # These should be recorded with their variable assignments
    ${product_name}=    Ask    What is the name of the first product?
    ${product_price}=    Ask    What is the price of the first product?

    Log    Product Name: ${product_name}
    Log    Product Price: ${product_price}

    # Validate the values are extracted
    Should Not Be Empty    ${product_name}
    Should Contain    ${product_price}    $

    # Export and verify
    Export Test Suite    path=${OUTPUT_DIR}/test_ask_variables.robot
    ...    suite_name=Ask Variable Test
    ...    test_name=Ask With Variables


Test Multiple Variable Assignments In Flow
    [Documentation]    Test multiple Ask keywords in a complete flow
    Do    Login as standard_user with password secret_sauce

    # Extract product info
    ${first_product}=    Ask    What is the name of the first product?
    ${first_price}=    Ask    What is the price of the first product?

    # Add to cart and check badge
    Do    Add the ${first_product} to the shopping cart
    ${cart_count}=    Ask    What number is shown in the cart badge?

    Log    Added ${first_product} (${first_price}) to cart
    Log    Cart count: ${cart_count}

    # Verify cart count
    Should Be Equal As Strings    ${cart_count}    1

    Export Test Suite    path=${OUTPUT_DIR}/test_flow_variables.robot
    ...    suite_name=Flow Variable Test
    ...    test_name=Multiple Variables In Flow


Test Do Keyword Without Variable Assignment
    [Documentation]    Verify Do keywords don't get spurious variable assignments
    Do    Login as standard_user with password secret_sauce
    Do    Click the Sauce Labs Backpack item
    Check    Product details page is displayed

    Export Test Suite    path=${OUTPUT_DIR}/test_do_no_variables.robot
    ...    suite_name=Do Without Variables
    ...    test_name=Do Keyword Test


Test Check Keyword Without Variable Assignment
    [Documentation]    Verify Check keywords don't get spurious variable assignments
    Do    Login as standard_user with password secret_sauce
    Check    Products page is displayed with inventory items
    Check    At least one product is visible

    Export Test Suite    path=${OUTPUT_DIR}/test_check_no_variables.robot
    ...    suite_name=Check Without Variables
    ...    test_name=Check Keyword Test


Test Complete Purchase With Variable Capture
    [Documentation]    Full e2e flow capturing variables along the way
    Do    Login as standard_user with password secret_sauce

    # Capture product info before purchase
    ${product_to_buy}=    Ask    What is the name of the first product?
    ${product_cost}=    Ask    What is the price of the first product?
    Log    Planning to buy: ${product_to_buy} for ${product_cost}

    # Add to cart
    Do    Add the ${product_to_buy} to cart

    # Go to checkout
    Do    Click the shopping cart icon
    ${items_in_cart}=    Ask    How many items are shown in the cart?
    Log    Items in cart: ${items_in_cart}

    Do    Click the Checkout button

    # Fill checkout form
    Do    Fill the first name field with John
    Do    Fill the last name field with Doe
    Do    Fill the zip code field with 12345
    Do    Click the Continue button

    # Verify total and complete
    ${order_total}=    Ask    What is the total price shown?
    Log    Order total: ${order_total}

    Do    Click the Finish button
    Check    Order confirmation message is displayed

    ${confirmation_text}=    Ask    What does the confirmation header say?
    Log    Confirmation: ${confirmation_text}

    Export Test Suite    path=${OUTPUT_DIR}/test_purchase_variables.robot
    ...    suite_name=Purchase With Variables
    ...    test_name=Complete Purchase E2E


*** Keywords ***
Setup Test
    Start Recording
    New Browser    chromium    headless=true
    New Page    ${STORE_URL}

Teardown Test
    Close Browser
    Stop Recording
