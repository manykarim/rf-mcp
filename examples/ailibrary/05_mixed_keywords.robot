*** Settings ***
Documentation    Combining AI keywords with traditional RF keywords
...
...    Best practice: Use AI keywords for complex interactions,
...    traditional keywords for precise control and assertions.

Library    Browser
Library    robotmcp.lib.AILibrary
...    provider=anthropic
...    api_key=%{ANTHROPIC_API_KEY}
Library    String
Library    Collections

Suite Setup    Initialize Test Environment
Suite Teardown    Close Browser

*** Variables ***
${BASE_URL}         https://www.saucedemo.com
${VALID_USER}       standard_user
${VALID_PASS}       secret_sauce
@{PRODUCTS}         Sauce Labs Backpack    Sauce Labs Bike Light


*** Test Cases ***
Login With Mixed Approach
    [Documentation]    Traditional setup, AI login, traditional verification

    # Traditional: Navigate
    New Page    ${BASE_URL}

    # AI: Complex interaction
    Do    Login as ${VALID_USER} with password ${VALID_PASS}

    # Traditional: Precise verification
    ${url}=    Get Url
    Should Contain    ${url}    inventory.html

    ${title}=    Get Title
    Should Be Equal    ${title}    Swag Labs


Data-Driven With AI
    [Documentation]    Use AI with data from variables

    New Page    ${BASE_URL}
    Do    Login with standard credentials

    FOR    ${product}    IN    @{PRODUCTS}
        # AI: Add specific product
        Do    Add ${product} to the shopping cart

        # Traditional: Verify cart count incremented
        ${badge}=    Get Text    .shopping_cart_badge
        Should Not Be Empty    ${badge}
    END

    # AI: Final verification
    Check    Cart shows ${PRODUCTS.__len__()} items


Extract And Process Data
    [Documentation]    Use AI to extract, RF to process

    New Page    ${BASE_URL}
    Do    Login as standard_user

    # AI: Extract data
    ${price_text}=    Ask    What is the price of the Sauce Labs Backpack?

    # Traditional: Process the data
    ${price}=    Remove String    ${price_text}    $
    ${price_float}=    Convert To Number    ${price}

    Should Be True    ${price_float} > 0
    Log    Backpack costs: $${price_float}


Conditional Flow With AI
    [Documentation]    Use AI for detection, RF for conditional logic

    New Page    ${BASE_URL}
    Do    Login with valid credentials

    # AI: Check state
    ${has_items}=    Run Keyword And Return Status
    ...    Check    Cart is not empty

    IF    ${has_items}
        Log    Cart has items - clearing first
        Do    Remove all items from cart
    END

    # Continue with test
    Do    Add the backpack to cart
    Check    Cart has exactly 1 item


Complex Form Handling
    [Documentation]    AI fills form, RF validates specific fields

    New Page    ${BASE_URL}
    Do    Login and add a product to cart
    Do    Go to checkout

    # AI: Fill the form naturally
    Do    Fill the form with first name Test, last name User, zip 90210

    # Traditional: Verify specific field values
    ${first_name}=    Get Text    id=first-name    attribute=value
    ${last_name}=    Get Text    id=last-name    attribute=value
    ${zip}=    Get Text    id=postal-code    attribute=value

    Should Be Equal    ${first_name}    Test
    Should Be Equal    ${last_name}    User
    Should Be Equal    ${zip}    90210


API Response With AI Analysis
    [Documentation]    Traditional API call, AI analyzes response

    # This example shows concept - actual implementation depends on your API
    ${response}=    Set Variable    {"user": "john", "role": "admin", "active": true}

    # Store for AI analysis
    Set Test Variable    ${API_RESPONSE}    ${response}

    # AI can analyze complex responses
    # Note: In real scenario, you'd set page context or use custom prompts
    Log    Response to analyze: ${response}


Screenshot And AI Analysis
    [Documentation]    Take screenshot, describe with AI

    New Page    ${BASE_URL}
    Do    Login as standard_user

    # Traditional: Take screenshot
    ${screenshot}=    Take Screenshot    ${OUTPUT_DIR}/page_state.png

    # AI: Describe current state
    ${description}=    Ask    Describe what is visible on the products page
    Log    Page description: ${description}

    # Traditional: Log for reporting
    Log    Screenshot: ${screenshot}
    Log    AI Description: ${description}


*** Keywords ***
Initialize Test Environment
    New Browser    chromium    headless=true
    Set Browser Timeout    30s
