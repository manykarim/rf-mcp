*** Settings ***
Documentation    BDD-style end-to-end test: open demoshop, add two items to cart,
...              verify cart counts, checkout, and assert order confirmation.
...              Generated via RobotMCP MCP tools with BDD grouping.
Library          Browser
Test Tags        bdd    purchase    demoshop    e2e

*** Variables ***
${DEMOSHOP_URL}           https://demoshop.makrocode.de/
${FIRST_PRODUCT_BUTTON}   text="Add to cart" >> nth=0
${SECOND_PRODUCT_BUTTON}  button >> text="Add to cart" >> nth=1
${CART_LINK}              css=nav a[href='/cart']
${EMAIL_FIELD}            css=input[placeholder='you@productstudio.com']
${NAME_FIELD}             css=input[placeholder='Jamie Product']
${ADDRESS_FIELD}          css=textarea[placeholder*='123 Flow Street']
${ALERT_VISIBLE}          css=[role='alert'] >> visible=true

*** Test Cases ***
Demoshop BDD Purchase Workflow
    [Documentation]    BDD-style end-to-end test: open demoshop, add two items
    ...    to cart, verify cart counts, checkout, and assert order confirmation.
    Given the demoshop is open
    When the user adds the first product to cart
    Then the cart should contain 1 item
    When the user adds the second product to cart
    Then the cart should contain 2 items
    When the user proceeds to checkout
    And the user fills in the checkout form
    And the user places the order
    Then the order confirmation should be displayed
    [Teardown]    Close Browser

*** Keywords ***
the demoshop is open
    New Browser    chromium
    New Context
    New Page    ${DEMOSHOP_URL}

the user adds the first product to cart
    Click    ${FIRST_PRODUCT_BUTTON}

the cart should contain 1 item
    ${cart_text}=    Get Text    ${CART_LINK}
    Should Contain    ${cart_text}    1

the user adds the second product to cart
    Click    ${SECOND_PRODUCT_BUTTON}

the cart should contain 2 items
    ${cart_text}=    Get Text    ${CART_LINK}
    Should Contain    ${cart_text}    2

the user proceeds to checkout
    Click    ${CART_LINK}
    Click    text=Proceed to checkout

the user fills in the checkout form
    Fill Text    ${EMAIL_FIELD}    test@example.com
    Fill Text    ${NAME_FIELD}    Test User
    Fill Text    ${ADDRESS_FIELD}    123 Test Street\nTest City, TC 12345

the user places the order
    Click    text=Place order

the order confirmation should be displayed
    ${confirmation}=    Get Text    ${ALERT_VISIBLE}
    Should Contain    ${confirmation}    confirmed
