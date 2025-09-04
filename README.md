# 🤖 RobotMCP - AI-Powered Test Automation Bridge

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Robot Framework](https://img.shields.io/badge/robot%20framework-6.0+-green.svg)](https://robotframework.org)
[![FastMCP](https://img.shields.io/badge/fastmcp-2.0+-orange.svg)](https://github.com/jlowin/fastmcp)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Transform natural language into production-ready Robot Framework tests using AI agents and MCP protocol.**

RobotMCP is a comprehensive Model Context Protocol (MCP) server that bridges the gap between human language and Robot Framework automation. It enables AI agents to understand test intentions, execute steps interactively, and generate complete test suites from successful executions.

https://github.com/user-attachments/assets/ad89064f-cab3-4ae6-a4c4-5e8c241301a1

---

## ✨ Quick Start

### 1️⃣ Install
```bash
pip install rf-mcp
```

### 2️⃣ Add to VS Code (Cline/Claude Desktop)
```json
{
  "servers": {
    "robotmcp": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "robotmcp.server"]
    }
  }
}
```

### 3️⃣ Start Testing with AI
```
Use RobotMCP to create a TestSuite and execute it step wise.
Create a test for https://www.saucedemo.com/ that:
- Logs in to https://www.saucedemo.com/ with valid credentials
- Adds two items to cart
- Completes checkout process
- Verifies success message

Use Selenium Library.
Execute the test suite stepwise and build the final version afterwards.
```

**That's it!** RobotMCP will guide the AI through the entire testing workflow.

---

## 🎯 Core Concept

**Traditional Way:**
1. Write Robot Framework code manually
2. Debug syntax and locator issues
3. Maintain test suites over time

**RobotMCP Way:**
1. **Describe** what you want to test in natural language
2. **Execute** steps interactively with AI guidance
3. **Generate** production-ready Robot Framework suites automatically

---

## 🚀 Key Features

### 🧠 **Natural Language Processing**
- Convert human test descriptions into structured actions
- Intelligent scenario analysis and library recommendations
- Context-aware test planning (web, mobile, API, database)

### ⚡ **Interactive Step Execution**
- Execute Robot Framework keywords step-by-step
- Real-time state tracking and session management
- Smart error handling with actionable suggestions

### 🔍 **Intelligent Element Location**
- Advanced locator guidance for Browser Library & SeleniumLibrary
- Cross-library locator conversion (Browser ↔ Selenium)
- DOM filtering and element discovery

### 📋 **Production-Ready Suite Generation**
- Generate optimized Robot Framework test suites
- Maintain proper imports, setup/teardown, and documentation
- Support for tags, variables, and test organization

### 🌐 **Multi-Platform Support**
- **Web**: Browser Library (Playwright) & SeleniumLibrary
- **Mobile**: AppiumLibrary for iOS/Android testing
- **API**: RequestsLibrary for HTTP/REST testing
- **Database**: DatabaseLibrary for SQL operations

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Robot Framework 6.0+

### Method 1: PyPI Installation (Recommended)
```bash
# Install RobotMCP
pip install rf-mcp

# Install browser automation libraries
pip install robotframework-browser
playwright install  # Install browser binaries

# Or install Selenium support
pip install robotframework-seleniumlibrary

# For API testing
pip install robotframework-requests

# For mobile testing
pip install robotframework-appiumlibrary
```

### Method 2: Development Installation
```bash
# Clone repository
git clone https://github.com/manykarim/rf-mcp.git
cd rf-mcp

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Hint: When using a venv 

If you are using a virtual environment (venv) for your project, I recommend to install the `rf-mcp` package within the same venv.
When starting the MCP server, make sure to use the Python interpreter from that venv.

---

## 🔧 MCP Integration

### VS Code (GitHub Code)

```json
{
  "servers": {
    "robotmcp": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "-m", "robotmcp.server"]
    }
  }
}
```

```json
{
  "servers": {
    "robotmcp": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "robotmcp.server"]
    }
  }
}
```

**Hint:** 
If you set up a virtual environment, make sure to also use the python executable from that venv to start the server.

### Claude Desktop

**Location:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "robotmcp": {
      "command": "python",
      "args": ["-m", "robotmcp.server"]
    }
  }
}
```

### Other AI Agents
RobotMCP works with any MCP-compatible AI agent. Use the stdio configuration above.

---

## 🎪 Example Workflows

### 🌐 Web Application Testing

**Prompt:**
```
Use RobotMCP to create a TestSuite and execute it step wise.
Create a test for https://www.saucedemo.com/ that:
- Logs in to https://www.saucedemo.com/ with valid credentials
- Adds two items to cart
- Completes checkout process
- Verifies success message

Use Selenium Library.
Execute the test suite stepwise and build the final version afterwards.

```

**Result:** Complete Robot Framework test suite with proper locators, assertions, and structure.

### 📱 Mobile App Testing

**Prompt:**
```
Use RobotMCP to create a TestSuite and execute it step wise.
It shall:
- Launch app from tests/appium/SauceLabs.apk
- Perform login flow
- Add products to cart
- Complete purchase

Appium server is running at http://localhost:4723
Execute the test suite stepwise and build the final version afterwards.
```

**Result:** Mobile test suite with AppiumLibrary keywords and device capabilities.

### 🔌 API Testing

**Prompt:**
```
Read the Restful Booker API documentation at https://restful-booker.herokuapp.com.
Use RobotMCP to create a TestSuite and execute it step wise.
It shall:

- Create a new booking
- Authenticate as admin
- Update the booking
- Delete the booking
- Verify each response

Execute the test suite stepwise and build the final version afterwards.
```

**Result:** API test suite using RequestsLibrary with proper error handling.

### 🧪 XML/Database Testing

**Prompt:**
```
Create a xml file with books and authors.
Use RobotMCP to create a TestSuite and execute it step wise.
It shall:
- Parse XML structure
- Validate specific nodes and attributes
- Assert content values
- Check XML schema compliance

Execute the test suite stepwise and build the final version afterwards.
```

**Result:** XML processing test using Robot Framework's XML library.

---

## 🔍 MCP Tools Overview

RobotMCP provides **20 comprehensive MCP tools** organized into functional categories:

### Core Execution
- `analyze_scenario` - Convert natural language to structured test intent
- `execute_step` - Execute individual Robot Framework keywords
- `discover_keywords` - Find appropriate keywords for actions

### State & Context Management
- `get_application_state` - Capture current application state
- `get_page_source` - Extract DOM with intelligent filtering
- `get_session_info` - Session configuration and status

### Test Suite Generation
- `build_test_suite` - Generate Robot Framework test files
- `run_test_suite_dry` - Validate test syntax before execution
- `run_test_suite` - Execute complete test suites

### Library Discovery
- `recommend_libraries` - Suggest appropriate RF libraries
- `check_library_availability` - Verify library installation
- `get_available_keywords` - List all available keywords
- `search_keywords` - Find keywords by pattern

### Locator Guidance
- `get_selenium_locator_guidance` - SeleniumLibrary selector help
- `get_browser_locator_guidance` - Browser Library (Playwright) guidance
- `get_appium_locator_guidance` - Mobile locator strategies

### Advanced Features
- `set_library_search_order` - Control keyword resolution precedence
- `initialize_context` - Set up test sessions with variables
- `get_session_validation_status` - Check test readiness

*For detailed tool documentation, see the [Tools Reference](#-tools-reference) section.*

---

## 🏗️ Architecture

### Service-Oriented Design
```
📦 ExecutionCoordinator (Main Orchestrator)
├── 🔤 SessionManager - Session lifecycle & library management
├── ⚙️ KeywordExecutor - RF keyword execution engine
├── 🌐 BrowserLibraryManager - Browser/Selenium library switching
├── 📊 PageSourceService - DOM extraction & filtering
├── 🔄 LocatorConverter - Cross-library locator translation
└── 📋 SuiteExecutionService - Test suite generation & execution
```

### Native Robot Framework Integration
- **ArgumentResolver** - Native RF argument parsing
- **TypeConverter** - RF type conversion (string → int/bool/etc.)
- **LibDoc API** - Direct RF documentation access
- **Keyword Discovery** - Runtime detection using RF internals

### Session Management
- Auto-configuration based on scenario analysis
- Browser library conflict resolution (Browser vs Selenium)
- Cross-session state persistence
- Mobile capability detection and setup

---

## 📚 Tools Reference

### `analyze_scenario`
Convert natural language test descriptions into structured test intents with automatic session creation.

```python
{
  "scenario": "Test user login with valid credentials",
  "context": "web",
  "session_id": "optional-session-id"
}
```

### `execute_step`
Execute individual Robot Framework keywords with advanced session management.

```python
{
  "keyword": "Fill Text",
  "arguments": ["css=input[name='username']", "testuser"],
  "session_id": "default",
  "detail_level": "minimal"
}
```

### `build_test_suite`
Generate production-ready Robot Framework test suites from executed steps.

```python
{
  "test_name": "User Login Test",
  "session_id": "default",
  "tags": ["smoke", "login"],
  "documentation": "Test successful user login flow"
}
```

### `get_browser_locator_guidance`
Get comprehensive Browser Library locator strategies and error guidance.

```python
{
  "error_message": "Strict mode violation: multiple elements found",
  "keyword_name": "Click"
}
```

**Returns:**
- 10 Playwright locator strategies (css=, xpath=, text=, id=, etc.)
- Advanced features (cascaded selectors, iframe piercing, shadow DOM)
- Error-specific guidance and suggestions
- Best practices for element location

### `get_selenium_locator_guidance`
Get comprehensive SeleniumLibrary locator strategies and troubleshooting.

```python
{
  "error_message": "Element not found: name=firstname",
  "keyword_name": "Input Text"
}
```

**Returns:**
- 14 SeleniumLibrary locator strategies (id:, name:, css:, xpath:, etc.)
- Locator format analysis and recommendations
- Timeout and waiting strategy guidance
- Element location best practices

*For complete tool documentation, see the source code docstrings.*

---

## 🧪 Example Generated Test Suite

```robot
*** Settings ***
Documentation     Test suite for validating the complete checkout process on Sauce Demo website
Library           Browser
Library           Collections
Force Tags        e2e  checkout  smoke

*** Variables ***
${URL}                      https://www.saucedemo.com/
${USERNAME}                 standard_user
${PASSWORD}                 secret_sauce
${FIRST_NAME}               John
${LAST_NAME}                Doe
${POSTAL_CODE}              12345
${EXPECTED_SUCCESS_MSG}     Thank you for your order!

*** Test Cases ***
Complete Checkout Process Test
    [Documentation]    Validates the complete checkout process on Sauce Demo:
    ...    1. Opens the website
    ...    2. Logs in with valid credentials
    ...    3. Adds items to cart
    ...    4. Completes checkout process
    
    # Setup and login
    Open Browser And Navigate To Login Page
    Login With Valid Credentials
    Verify Successful Login
    
    # Add items to cart
    Add Item To Cart    id=add-to-cart-sauce-labs-backpack
    Verify Item Count In Cart    1
    Add Item To Cart    id=add-to-cart-sauce-labs-bike-light
    Verify Item Count In Cart    2
    
    # Checkout process
    Go To Cart
    Start Checkout
    Fill Checkout Information
    Complete Checkout
    
    # Verify successful checkout
    Verify Checkout Success
    
    # Cleanup
    Close Browser

*** Keywords ***
Open Browser And Navigate To Login Page
    New Browser    chromium    headless=False
    New Context    viewport={'width': 1280, 'height': 720}
    New Page    ${URL}
    
Login With Valid Credentials
    Fill Text    id=user-name    ${USERNAME}
    Fill Text    id=password    ${PASSWORD}
    Click    id=login-button

Verify Successful Login
    Wait For Elements State    .inventory_list    visible
    ${current_url}=    Get Url
    Should Contain    ${current_url}    inventory.html

Add Item To Cart
    [Arguments]    ${item_id}
    Click    ${item_id}

Verify Item Count In Cart
    [Arguments]    ${expected_count}
    ${cart_count}=    Get Text    .shopping_cart_badge
    Should Be Equal As Strings    ${cart_count}    ${expected_count}

Go To Cart
    Click    .shopping_cart_link

Start Checkout
    Click    id=checkout

Fill Checkout Information
    Fill Text    id=first-name    ${FIRST_NAME}
    Fill Text    id=last-name    ${LAST_NAME}
    Fill Text    id=postal-code    ${POSTAL_CODE}
    Click    id=continue

Complete Checkout
    Click    id=finish

Verify Checkout Success
    ${success_message}=    Get Text    h2
    Should Be Equal As Strings    ${success_message}    ${EXPECTED_SUCCESS_MSG}
```

Original prompt:
```bash
Use RobotMCP to create a TestSuite and execute it step wise.

- Open https://www.saucedemo.com/
- Login with valid user
- Assert login was successful
- Add item to cart
- Assert item was added to cart
- Add another item to cart
- Assert another item was added to cart
- Checkout
- Assert checkout was successful

Execute step by step and build final test suite afterwards.
Make a clean and maintainable test suite
```

---

## 🔄 Recommended Workflow

### 1. **Analysis Phase**
```
Use analyze_scenario to understand test requirements and create session
```

### 2. **Library Setup**
```
Get recommendations with recommend_libraries
Check availability with check_library_availability
```

### 3. **Interactive Development**
```
Execute steps one by one with execute_step
Get page state with get_page_source
Use locator guidance tools for element issues
```

### 4. **Suite Generation**
```
Validate session with get_session_validation_status
Generate suite with build_test_suite
Validate syntax with run_test_suite_dry
Execute with run_test_suite
```

---

## 🎯 Pro Tips

### 🔍 **Element Location**
- Use `get_page_source` with `filtered=true` to see automation-relevant elements
- Leverage locator guidance tools when elements aren't found
- Browser Library supports modern selectors (text=, data-testid=, etc.)

### ⚡ **Performance**
- Use `detail_level="minimal"` to reduce response size by 80-90%
- Enable DOM filtering to focus on interactive elements
- Session management maintains state across interactions

### 🛡️ **Reliability**
- Execute steps individually before building suites
- Use `run_test_suite_dry` to catch issues early
- Leverage native RF integration for maximum compatibility

### 🌐 **Cross-Platform**
- Sessions auto-detect context (web/mobile/api) from scenarios
- Library conflicts are automatically resolved
- Mobile sessions configure Appium capabilities automatically

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Clone** your fork locally
3. **Install** development dependencies: `uv sync`
4. **Create** a feature branch
5. **Add** comprehensive tests for new functionality
6. **Run** tests: `uv run pytest tests/`
7. **Submit** a pull request

### Development Commands
```bash
# Run tests
uv run pytest tests/

# Format code
uv run black src/

# Type checking
uv run mypy src/

# Start development server
uv run python -m robotmcp.server
```

---

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

## 🌟 Why RobotMCP?

### For AI Agents
- **🤖 Agent-Optimized**: Structured responses designed for AI processing
- **🧠 Context-Aware**: Rich error messages with actionable guidance
- **⚡ Token-Efficient**: Minimal response mode reduces costs significantly

### For Test Engineers
- **🛡️ Production-Ready**: Native Robot Framework integration
- **🔧 Flexible**: Multi-library support (Browser, Selenium, Appium, etc.)
- **📊 Comprehensive**: 20 tools covering complete automation workflow

### For Teams
- **📝 Maintainable**: Generates clean, documented Robot Framework code
- **🔄 Iterative**: Step-by-step development and validation
- **🌐 Scalable**: Session-based architecture supports complex scenarios

---

## 💬 Support & Community

- 🐛 **Issues**: [GitHub Issues](https://github.com/manykarim/rf-mcp/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/manykarim/rf-mcp/discussions)
- 📖 **Documentation**: Tool docstrings and examples
- 🚀 **Latest Updates**: Check releases for new features

---

**⭐ Star us on GitHub if RobotMCP helps your test automation journey!**

Made with ❤️ for the Robot Framework and AI automation community.
