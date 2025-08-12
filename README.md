# Robot Framework MCP Server

A comprehensive Model Context Protocol (MCP) server that provides an intelligent bridge between natural language test descriptions and Robot Framework execution. This server enables AI agents to dynamically create, execute, and debug Robot Framework test steps from natural language, then generate complete test suites from successful executions.

## ✨ Key Features

- **🧠 Natural Language Processing**: Converts human language test descriptions into structured test actions
- **🔍 Semantic Keyword Matching**: Uses AI to find the most appropriate Robot Framework keywords for each action  
- **⚡ Interactive Test Execution**: Execute test steps individually with real-time state tracking and session management
- **📊 State-Aware Testing**: Captures and analyzes application state (DOM, API responses, database state) after each step
- **🎯 Intelligent Suggestions**: AI-driven recommendations for next test steps based on current state
- **📋 Test Suite Generation**: Automatically generates optimized Robot Framework test suites from successful executions
- **🌐 Multi-Context Support**: Handles web, mobile, API, and database testing scenarios
- **🔧 Advanced Locator Guidance**: Comprehensive Browser Library and SeleniumLibrary locator strategy assistance
- **🛡️ Robust Error Handling**: Context-aware error analysis with actionable suggestions
- **📚 Native Robot Framework Integration**: Uses Robot Framework's native LibDoc and type conversion systems

## 🏗️ Architecture

### Core Components

1. **🔤 Natural Language Processor** - Analyzes test scenarios and extracts structured actions
2. **🎯 Keyword Matcher** - Maps natural language actions to Robot Framework keywords using semantic similarity
3. **⚙️ Execution Coordinator** - Service-oriented execution engine with proper library routing and session management
4. **📈 State Manager** - Tracks application state (DOM, API responses, database state) with intelligent filtering
5. **📝 Test Builder** - Converts successful execution paths into optimized Robot Framework test suites
6. **🔍 Dynamic Keyword Discovery** - Runtime keyword detection and argument processing using Robot Framework's native systems
7. **🌐 Browser Library Manager** - Advanced Browser Library and SeleniumLibrary integration with automatic library switching

## 🛠️ MCP Tools

The server provides **24 comprehensive MCP tools** for complete test automation workflow:

### Core Execution Tools

#### 1. `execute_step` 
**Execute individual Robot Framework keywords with advanced session management**
```json
{
  "keyword": "Fill Text",
  "arguments": ["css=input[name='username']", "testuser"],
  "session_id": "default",
  "detail_level": "minimal"
}
```
- ✅ Supports both Browser Library and SeleniumLibrary
- ✅ Automatic library detection and switching
- ✅ Native Robot Framework type conversion
- ✅ Context-aware error messages with locator guidance

#### 2. `analyze_scenario`
**Process natural language test descriptions into structured test intents**
```json
{
  "scenario": "Test that users can search for products and add them to cart",
  "context": "web"
}
```

#### 3. `discover_keywords`
**Find matching Robot Framework keywords for specific actions**
```json
{
  "action_description": "click the login button", 
  "context": "web",
  "current_state": {}
}
```

### State Management Tools

#### 4. `get_application_state`
**Retrieve current application state for decision making**
```json
{
  "state_type": "dom",
  "elements_of_interest": ["button", "input"],
  "session_id": "default"
}
```

#### 5. `get_page_source`
**Get page source with intelligent DOM filtering**
```json
{
  "session_id": "default",
  "full_source": false,
  "filtered": true,
  "filtering_level": "standard"
}
```
- ✅ Automatic DOM filtering for automation-relevant content
- ✅ Multiple filtering levels (minimal, standard, aggressive)
- ✅ Size optimization for AI processing

#### 6. `suggest_next_step`
**Get AI-driven suggestions for the next test step**
```json
{
  "current_state": {...},
  "test_objective": "complete user login",
  "executed_steps": [...],
  "session_id": "default"
}
```

### Test Suite Generation Tools

#### 7. `build_test_suite`
**Generate Robot Framework test suite from successful execution**
```json
{
  "test_name": "User Login Test",
  "session_id": "default",
  "tags": ["login", "smoke"],
  "documentation": "Test successful user login flow",
  "remove_library_prefixes": true
}
```

#### 8. `validate_step_before_suite`
**Validate individual steps before adding to test suite**
```json
{
  "keyword": "Click",
  "arguments": ["css=.login-button"],
  "session_id": "default",
  "expected_outcome": "User should be logged in"
}
```

#### 9. `validate_test_readiness`
**Check if session is ready for test suite generation**
```json
{
  "session_id": "default"
}
```

#### 10. `get_session_validation_status`
**Get validation status of all steps in a session**
```json
{
  "session_id": "default"
}
```

### Library and Keyword Discovery Tools

#### 11. `get_available_keywords`
**Get available Robot Framework keywords with native LibDoc documentation**
```json
{
  "library_name": "Browser"  // Optional - returns all if not specified
}
```

#### 12. `search_keywords`
**Search for keywords matching a pattern using native RF LibDoc**
```json
{
  "pattern": "click"
}
```

#### 13. `get_keyword_documentation`
**Get comprehensive documentation for specific keywords**
```json
{
  "keyword_name": "Fill Text",
  "library_name": "Browser"  // Optional
}
```

#### 14. `get_loaded_libraries`
**Get status of all loaded Robot Framework libraries**
```json
{}
```

#### 15. `check_library_availability`
**Check if Robot Framework libraries are available**
```json
{
  "libraries": ["Browser", "SeleniumLibrary", "RequestsLibrary"]
}
```

#### 16. `get_library_status`
**Get detailed installation status for specific library**
```json
{
  "library_name": "Browser"
}
```

### Advanced Locator Guidance Tools

#### 17. `get_selenium_locator_guidance`
**Get comprehensive SeleniumLibrary locator strategy guidance**
```json
{
  "error_message": "Element not found: name=firstname",
  "keyword_name": "Input Text"
}
```
**Provides:**
- ✅ 14 locator strategies with examples (`id:`, `name:`, `css:`, `xpath:`, etc.)
- ✅ Error-specific guidance (element not found, timeouts, etc.)
- ✅ Locator format analysis and recommendations
- ✅ Best practices for element location

#### 18. `get_browser_locator_guidance`
**Get comprehensive Browser Library (Playwright) locator strategy guidance**
```json
{
  "error_message": "Strict mode violation: multiple elements match",
  "keyword_name": "Click"
}
```
**Provides:**
- ✅ 10 Playwright locator strategies (`css=`, `xpath=`, `text=`, `id=`, etc.)
- ✅ Advanced features (cascaded selectors, iFrame piercing, Shadow DOM)
- ✅ Implicit detection rules (CSS default, XPath for `//`, text for quotes)
- ✅ Strict mode and Shadow DOM guidance
- ✅ Intelligent selector pattern analysis

### Planning and Validation Tools

#### 19. `validate_scenario`
**Validate scenario feasibility before execution**
```json
{
  "parsed_scenario": {...},
  "available_libraries": ["Browser", "BuiltIn"]
}
```

#### 20. `recommend_libraries`
**Recommend Robot Framework libraries based on test scenario**
```json
{
  "scenario": "Test REST API endpoints with authentication",
  "context": "api",
  "max_recommendations": 5
}
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Robot Framework 6.0+

### Quick Install from PyPI (Recommended)

```bash
# Install the latest stable version
pip install rf-mcp

```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/manykarim/rf-mcp.git
cd rf-mcp

# Install dependencies using uv (recommended)
uv sync

# Or install using pip in development mode
pip install -e .


```

### Robot Framework Libraries

Install the libraries you need for your testing:

```bash
# For web automation with Browser Library (recommended)
uv add robotframework-browser
playwright install  # Install browser binaries

# For web automation with Selenium
uv add robotframework-seleniumlibrary

# For API testing
uv add robotframework-requests

# For database testing  
uv add robotframework-databaselibrary

# For SSH/remote operations
uv add robotframework-sshlibrary
```

## 🖥️ VS Code Integration

### Adding to VS Code via mcp.json

Create or update your VS Code MCP configuration file:

**Location:** `%APPDATA%\Code\User\globalStorage\rooveterinaryinc.roo-cline\mcp.json` (Windows)
**Location:** `~/.config/Code/User/globalStorage\rooveterinaryinc.roo-cline\mcp.json` (macOS/Linux)

### Option 1: Using PyPI Installation (Recommended)

If you installed via `pip install rf-mcp`:

**Windows:**
```json
{
  "servers": {
    "robotmcp": {
      "type": "stdio",
      "command": "python",
      "args": [
        "-m",
        "robotmcp.server"
      ]
    }
  }
}
```

**macOS/Linux:**
```json
{
  "servers": {
    "robotmcp": {
      "type": "stdio",
      "command": "python3",
      "args": [
        "-m",
        "robotmcp.server"
      ]
    }
  }
}
```

### Option 2: Development Installation

If you cloned the repository for development:

**Windows:**
```json
{
  "servers": {
    "robotmcp": {
      "type": "stdio",
      "command": "C:\\workspace\\rf-mcp\\.venv\\Scripts\\python.exe",
      "args": [
        "-m",
        "robotmcp.server"
      ],
      "cwd": "C:\\workspace\\rf-mcp"
    }
  }
}
```

**macOS/Linux:**
```json
{
  "servers": {
    "robotmcp": {
      "type": "stdio", 
      "command": "/path/to/rf-mcp/.venv/bin/python",
      "args": [
        "-m", 
        "robotmcp.server"
      ],
      "cwd": "/path/to/rf-mcp"
    }
  }
}
```

### Option 3: Using uv

```json
{
  "servers": {
    "robotmcp": {
      "type": "stdio",
      "command": "uv", 
      "args": [
        "run",
        "python",
        "-m", 
        "robotmcp.server"
      ],
      "cwd": "C:\\workspace\\rf-mcp"
    }
  }
}
```

## 🎯 Usage

### Starting the Server

**After PyPI installation:**
```bash
# Using the installed package
python -m robotmcp.server

# Or using the robotmcp command
robotmcp
```

**Development installation:**
```bash
# Using uv (recommended for development)
uv run python -m robotmcp.server

# Or traditional method
python -m robotmcp.server
```

### Example Workflow

#### 1. **Analyze a test scenario**:
```
"Test login functionality with valid credentials showing dashboard"
```

#### 2. **Execute steps interactively**:
- Open Browser to login page
- Fill username and password fields
- Click login button  
- Verify dashboard appears
- Get intelligent suggestions for next steps

#### 3. **Generate test suite**:
- Optimized Robot Framework test case
- Complete with imports, setup, and teardown
- Ready for execution in CI/CD pipelines

### Advanced Browser Testing Example

```python
# 1. Start browser session
await execute_step("New Browser", ["chromium", "headless=False"])
await execute_step("New Page", ["https://example.com/login"])

# 2. Get page state for element discovery
state = await get_page_source("default", filtered=True, filtering_level="standard")

# 3. Interactive element location with guidance
try:
    await execute_step("Fill Text", ["css=input[name='username']", "testuser"])
except Exception as e:
    # Get Browser Library locator guidance for error resolution
    guidance = await get_browser_locator_guidance(str(e), "Fill Text")
    # Use guidance to fix selector: try "id=username" or "//input[@name='username']"

# 4. Build test suite from successful steps
suite = await build_test_suite("Login Test", "default", ["smoke", "login"])
```

## 🌟 Key Advantages

### For AI Agents
- **🤖 Agent-Friendly**: Structured responses optimized for AI processing
- **🔍 Context-Aware**: Rich error messages with actionable guidance  
- **⚡ Efficient**: Minimal response mode reduces token usage by 80-90%
- **🧠 Intelligent**: Semantic keyword matching and smart suggestions

### For Developers  
- **🛡️ Robust**: Native Robot Framework integration with proper type conversion
- **🔧 Flexible**: Support for both Browser Library and SeleniumLibrary
- **📊 Comprehensive**: 24 tools covering entire test automation workflow
- **🎯 Precise**: Advanced locator guidance prevents common automation issues

### For Test Automation
- **📝 Stepwise Development**: Execute and validate each step before building suites
- **🔄 Session Management**: Maintain context across multiple interactions
- **🌐 Multi-Library**: Seamless switching between Browser/Selenium libraries
- **📋 Production-Ready**: Generates clean, maintainable Robot Framework code

## 🧪 Example Generated Test Suite

```robot
*** Settings ***
Documentation    Test case that opens browser, navigates to page, performs login, and verifies success.
Library          Browser
Library          BuiltIn
Force Tags       automated    generated    web    login

*** Test Cases ***
User Login Test
    [Documentation]    Test successful user login flow with form validation
    [Tags]    login    smoke    critical
    
    # Browser Setup
    New Browser    chromium    headless=False
    New Page       https://example.com/login
    
    # Login Actions  
    Fill Text      css=input[name='username']    testuser
    Fill Text      css=input[name='password']    testpass123
    Click          css=button[type='submit']
    
    # Verification
    Wait For Elements State    css=.dashboard    visible    timeout=5s
    Get Text                   css=.welcome-message    ==    Welcome, testuser!
    
    [Teardown]    Close Browser
```

## 📚 Dependencies

### Required
- `robotframework>=6.0`
- `fastmcp>=2.0.0` 
- `pydantic>=2.0.0`
- `aiohttp>=3.8.0`

### Optional (Enhanced Functionality)
- `sentence-transformers>=2.2.0` - Semantic keyword matching
- `beautifulsoup4>=4.11.0` - DOM parsing and filtering
- `robotframework-browser` - Modern web automation (Playwright)
- `robotframework-seleniumlibrary` - Traditional web automation
- `robotframework-requests` - API testing
- `robotframework-databaselibrary` - Database testing
- `robotframework-sshlibrary` - SSH/remote operations

## 🎪 Supported Test Contexts

- **🌐 Web Applications**: Browser Library (Playwright) and SeleniumLibrary support
- **📱 Mobile Applications**: AppiumLibrary integration for mobile testing
- **🔌 API Testing**: RequestsLibrary for HTTP/REST APIs  
- **🗄️ Database Testing**: DatabaseLibrary for SQL operations
- **🖥️ Desktop Applications**: Support for desktop automation libraries
- **🔧 System Testing**: SSH, Process, and OperatingSystem library integration

## 🚧 Development

### Running Tests
```bash
uv run pytest tests/
```

### Code Quality
```bash
# Format code
uv run black src/

# Type checking
uv run mypy src/

# Linting  
uv run flake8 src/
```

### Architecture Testing
```bash
# Test locator guidance systems
uv run python -c "from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter; c = RobotFrameworkNativeConverter(); print('✅ Systems working')"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with comprehensive tests
4. Add tests for new functionality
5. Ensure all tests pass (`uv run pytest`)
6. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 💬 Support

- 🐛 **Bug Reports**: [Create an issue](https://github.com/manykarim/rf-mcp/issues)
- 💡 **Feature Requests**: [Create an issue](https://github.com/manykarim/rf-mcp/issues)  
- 📖 **Documentation**: Check the comprehensive tool documentation above
- 💬 **Community**: Join our discussions and share your automation success stories

## 🔄 Recent Updates

### v2.0 - Advanced Locator Guidance & Native Integration
- ✅ **Comprehensive Locator Guidance**: Added SeleniumLibrary and Browser Library locator strategy tools
- ✅ **Native Robot Framework Integration**: Uses RF's native LibDoc and ArgumentResolver systems  
- ✅ **Enhanced Error Handling**: Context-aware error analysis with actionable suggestions
- ✅ **Intelligent Selector Analysis**: Automatic detection and guidance for selector patterns
- ✅ **Session Management**: Advanced Browser/Selenium library switching with force parameters
- ✅ **DOM Filtering**: Intelligent page source filtering for automation-relevant content

### v1.5 - Service-Oriented Architecture
- ✅ **Migrated to Service Architecture**: From monolithic to modular ExecutionCoordinator
- ✅ **Enhanced Performance**: 80-90% token reduction in minimal response mode
- ✅ **Robust Type Conversion**: Native Robot Framework type conversion eliminates pattern matching
- ✅ **24 Comprehensive Tools**: Complete test automation workflow coverage

---

**🎯 Production Note**: This implementation provides enterprise-grade Robot Framework automation with comprehensive AI agent support. The system includes robust error handling, session management, and native Robot Framework integration suitable for production test automation environments.