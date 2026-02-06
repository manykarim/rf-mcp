*** Settings ***
Documentation    Recording and Export functionality examples
...
...    This example shows how to:
...    - Record keyword executions
...    - Export recordings to different formats
...    - Generate traditional RF test files from AI interactions

Library    Browser
Library    robotmcp.lib.AILibrary
...    provider=anthropic
...    api_key=%{ANTHROPIC_API_KEY}
Library    OperatingSystem

Suite Teardown    Close Browser

*** Variables ***
${STORE_URL}    https://www.saucedemo.com
${OUTPUT}       ${CURDIR}/generated


*** Test Cases ***
Record Login Flow And Export
    [Documentation]    Record a login flow and export as .robot file

    # Start recording
    Start Recording

    # Perform the login flow
    New Browser    chromium    headless=true
    New Page    ${STORE_URL}

    Do    Enter standard_user in the username field
    Do    Enter secret_sauce in the password field
    Do    Click the login button

    Check    Products page is displayed

    # Stop recording
    Stop Recording

    # Export to Robot format
    ${robot_path}=    Export Test Suite
    ...    ${OUTPUT}/login_flow.robot
    ...    suite_name=Login Tests
    ...    test_name=Standard User Login
    ...    include_comments=True

    Log    Exported Robot file: ${robot_path}
    File Should Exist    ${robot_path}


Record Shopping Flow And Export As JSON
    [Documentation]    Export recording in JSON format for processing

    Start Recording

    New Browser    chromium    headless=true
    New Page    ${STORE_URL}

    Do    Login as standard_user with password secret_sauce
    Do    Add the first product to cart
    Do    Go to cart
    Check    Cart contains one item

    Stop Recording

    # Export as JSON
    ${json_path}=    Export Test Suite
    ...    ${OUTPUT}/shopping_flow.json
    ...    format=json
    ...    suite_name=Shopping Tests
    ...    test_name=Add Product To Cart

    Log    Exported JSON file: ${json_path}
    File Should Exist    ${json_path}

    # Read and log JSON content
    ${content}=    Get File    ${json_path}
    Log    JSON Content: ${content}


Export Without AI Comments
    [Documentation]    Export clean RF file without AI prompt comments

    Start Recording

    New Browser    chromium    headless=true
    New Page    ${STORE_URL}

    Do    Login and add backpack to cart

    Stop Recording

    ${path}=    Export Test Suite
    ...    ${OUTPUT}/clean_test.robot
    ...    include_comments=False
    ...    suite_name=Clean Tests
    ...    test_name=Quick Purchase

    ${content}=    Get File    ${path}
    Should Not Contain    ${content}    # AI:


Inspect Recorded Steps
    [Documentation]    Get recorded steps for programmatic inspection

    Start Recording

    New Browser    chromium    headless=true
    New Page    ${STORE_URL}

    Do    Login as standard_user with password secret_sauce
    Do    Add product to cart

    # Get steps before stopping (can also do after)
    ${steps}=    Get Recorded Steps

    Log    Total steps recorded: ${steps.__len__()}

    FOR    ${step}    IN    @{steps}
        Log    Keyword: ${step}[keyword]
        Log    Library: ${step}[library]
        Log    Type: ${step}[type]
        Log    ---
    END

    Stop Recording


Export To YAML Format
    [Documentation]    Export in YAML format for CI/CD integration

    Start Recording

    New Browser    chromium    headless=true
    New Page    ${STORE_URL}

    Do    Complete login with standard_user credentials

    Stop Recording

    ${yaml_path}=    Export Test Suite
    ...    ${OUTPUT}/test_flow.yaml
    ...    format=yaml

    File Should Exist    ${yaml_path}
    ${content}=    Get File    ${yaml_path}
    Should Contain    ${content}    metadata:
    Should Contain    ${content}    steps:
