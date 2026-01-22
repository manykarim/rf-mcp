*** Settings ***
Documentation    Examples using different AI providers
...
...    AILibrary supports multiple providers:
...    - Anthropic (Claude)
...    - OpenAI (GPT)
...    - Ollama (local models)
...    - Azure OpenAI
...
...    Run with the appropriate environment variables set.

Library    Browser


*** Test Cases ***
# ============================================================================
# ANTHROPIC EXAMPLES
# ============================================================================

Test With Anthropic Claude
    [Documentation]    Using Anthropic's Claude model
    [Tags]    anthropic

    # Import with Anthropic configuration
    Import Library    robotmcp.lib.AILibrary
    ...    provider=anthropic
    ...    api_key=%{ANTHROPIC_API_KEY}
    ...    model=claude-sonnet-4-20250514
    ...    max_tokens=4096

    New Browser    chromium    headless=true
    New Page    https://example.com

    Do    Find the main heading on the page
    ${heading}=    Ask    What is the main heading text?

    Log    Using Claude: ${heading}
    Close Browser


# ============================================================================
# OPENAI EXAMPLES
# ============================================================================

Test With OpenAI GPT-4
    [Documentation]    Using OpenAI's GPT-4 model
    [Tags]    openai

    Import Library    robotmcp.lib.AILibrary
    ...    provider=openai
    ...    api_key=%{OPENAI_API_KEY}
    ...    model=gpt-4-turbo

    New Browser    chromium    headless=true
    New Page    https://example.com

    Do    Locate the main content area
    Check    Page has loaded successfully

    Close Browser


Test With OpenAI GPT-3.5
    [Documentation]    Using GPT-3.5 for faster, cheaper operations
    [Tags]    openai

    Import Library    robotmcp.lib.AILibrary
    ...    provider=openai
    ...    api_key=%{OPENAI_API_KEY}
    ...    model=gpt-3.5-turbo

    New Browser    chromium    headless=true
    New Page    https://example.com

    ${title}=    Ask    What is the page title?
    Log    Page title: ${title}

    Close Browser


# ============================================================================
# OLLAMA (LOCAL) EXAMPLES
# ============================================================================

Test With Ollama Local Model
    [Documentation]    Using Ollama for local/offline testing
    [Tags]    ollama    local

    # No API key needed for local Ollama
    Import Library    robotmcp.lib.AILibrary
    ...    provider=ollama
    ...    model=llama2
    ...    base_url=http://localhost:11434

    New Browser    chromium    headless=true
    New Page    https://example.com

    Do    Click on any visible link
    ${current_url}=    Ask    What is the current page URL?

    Log    Navigated to: ${current_url}
    Close Browser


Test With Ollama Mistral
    [Documentation]    Using Mistral model via Ollama
    [Tags]    ollama    local

    Import Library    robotmcp.lib.AILibrary
    ...    provider=ollama
    ...    model=mistral
    ...    base_url=http://localhost:11434

    New Browser    chromium    headless=true
    New Page    https://example.com

    Check    Page contains some text content

    Close Browser


# ============================================================================
# AZURE OPENAI EXAMPLES
# ============================================================================

Test With Azure OpenAI
    [Documentation]    Using Azure-hosted OpenAI
    [Tags]    azure

    Import Library    robotmcp.lib.AILibrary
    ...    provider=azure
    ...    api_key=%{AZURE_OPENAI_KEY}
    ...    model=gpt-4-turbo
    ...    base_url=%{AZURE_OPENAI_ENDPOINT}

    New Browser    chromium    headless=true
    New Page    https://example.com

    Do    Navigate to the about section if available
    ${content}=    Ask    What information is available on this page?

    Log    Page content: ${content}
    Close Browser


# ============================================================================
# CONFIGURATION FILE EXAMPLE
# ============================================================================

Test With YAML Configuration
    [Documentation]    Load configuration from YAML file
    [Tags]    config

    # Configuration loaded from file
    Import Library    robotmcp.lib.AILibrary
    ...    config=${CURDIR}/ai_config.yaml

    New Browser    chromium    headless=true
    New Page    https://example.com

    Do    Explore the page structure
    Check    Page is accessible

    Close Browser
