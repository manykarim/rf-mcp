*** Settings ***
Documentation    Retry configuration examples for handling dynamic content
...
...    Shows how to configure retries for:
...    - Slow loading elements
...    - Dynamic content
...    - Network delays

Library    Browser
Library    robotmcp.lib.AILibrary
...    provider=anthropic
...    api_key=%{ANTHROPIC_API_KEY}
...    retries=3
...    retry_delay=1s

Suite Teardown    Close Browser

*** Variables ***
${SLOW_APP_URL}    https://the-internet.herokuapp.com


*** Test Cases ***
Handle Slow Loading Elements
    [Documentation]    Use increased retries for slow elements

    New Browser    chromium    headless=true
    New Page    ${SLOW_APP_URL}/dynamic_loading/1

    # Start the loading
    Do    Click the Start button

    # Wait for dynamic content with increased retries
    Check    Hello World text is displayed    retries=10    retry_delay=500ms


Handle Dynamic Content
    [Documentation]    Retry when content changes dynamically

    New Browser    chromium    headless=true
    New Page    ${SLOW_APP_URL}/dynamic_content

    # Content changes on each load - retry to handle variations
    ${content}=    Ask    What text is displayed in the first paragraph?    retries=3
    Should Not Be Empty    ${content}


Handle AJAX Loading
    [Documentation]    Wait for AJAX content to load

    New Browser    chromium    headless=true
    New Page    ${SLOW_APP_URL}/dynamic_loading/2

    Do    Click Start to begin loading
    Check    The loading indicator disappears    retries=15    retry_delay=1s
    Check    Hello World message is visible


Quick Retry For Fast Elements
    [Documentation]    Use short retry delay for responsive elements

    New Browser    chromium    headless=true
    New Page    ${SLOW_APP_URL}/add_remove_elements/

    Do    Click Add Element button
    Check    Delete button appears    retries=3    retry_delay=100ms

    Do    Click the Delete button
    Check    Delete button is removed    retries=3    retry_delay=100ms


Progressive Retry Strategy
    [Documentation]    Start with fast retries, increase on failure

    New Browser    chromium    headless=true
    New Page    ${SLOW_APP_URL}/dynamic_loading/1

    Do    Start the loading process

    # Try quick check first
    ${status}=    Run Keyword And Return Status
    ...    Check    Content is loaded    retries=2    retry_delay=200ms

    # If quick check fails, try with longer waits
    IF    not ${status}
        Check    Content is loaded    retries=10    retry_delay=1s
    END


Handle Disappearing Elements
    [Documentation]    Retry for elements that may temporarily disappear

    New Browser    chromium    headless=true
    New Page    ${SLOW_APP_URL}/disappearing_elements/

    # Elements may not always appear - retry to catch them
    ${gallery_exists}=    Run Keyword And Return Status
    ...    Check    Gallery link is present    retries=5    retry_delay=500ms

    Log    Gallery link found: ${gallery_exists}
