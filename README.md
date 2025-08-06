# Robot Framework MCP Server

A Model Context Protocol (MCP) server that provides an intelligent bridge between natural language test descriptions and Robot Framework execution. This server enables AI agents to dynamically create and execute Robot Framework test steps from natural language, then generate complete test suites from successful executions.

## Features

- **Natural Language Processing**: Converts human language test descriptions into structured test actions
- **Semantic Keyword Matching**: Uses AI to find the most appropriate Robot Framework keywords for each action
- **Interactive Test Execution**: Execute test steps individually with real-time state tracking
- **State-Aware Testing**: Captures and analyzes application state after each step
- **Intelligent Suggestions**: AI-driven recommendations for next test steps
- **Test Suite Generation**: Automatically generates optimized Robot Framework test suites
- **Multi-Context Support**: Handles web, mobile, API, and database testing scenarios

## Architecture

### Core Components

1. **Natural Language Processor** - Analyzes test scenarios and extracts structured actions
2. **Keyword Matcher** - Maps natural language actions to Robot Framework keywords using semantic similarity
3. **Execution Engine** - Executes Robot Framework keywords and manages test sessions
4. **State Manager** - Tracks application state (DOM, API responses, database state)
5. **Test Builder** - Converts successful execution paths into optimized Robot Framework test suites

## MCP Tools

The server provides 7 MCP tools for comprehensive test automation:

### 1. `analyze_scenario`
Process natural language test descriptions into structured test intents.

```json
{
  "scenario": "Test that users can search for products and add them to cart",
  "context": "web"
}
```

### 2. `discover_keywords` 
Find matching Robot Framework keywords for specific actions.

```json
{
  "action_description": "click the login button",
  "context": "web",
  "current_state": {}
}
```

### 3. `execute_step`
Execute individual Robot Framework keywords with session management.

```json
{
  "keyword": "Open Browser",
  "arguments": ["https://example.com", "chrome"],
  "session_id": "default"
}
```

### 4. `get_application_state`
Retrieve current application state for decision making.

```json
{
  "state_type": "dom",
  "elements_of_interest": ["button", "input"],
  "session_id": "default"
}
```

### 5. `suggest_next_step`
Get AI-driven suggestions for the next test step.

```json
{
  "current_state": {...},
  "test_objective": "complete user login",
  "executed_steps": [...],
  "session_id": "default"
}
```

### 6. `build_test_suite`
Generate Robot Framework test suite from successful execution.

```json
{
  "session_id": "default",
  "test_name": "User Login Test",
  "tags": ["login", "smoke"],
  "documentation": "Test successful user login flow"
}
```

### 7. `validate_scenario`
Validate scenario feasibility before execution.

```json
{
  "parsed_scenario": {...},
  "available_libraries": ["SeleniumLibrary", "BuiltIn"]
}
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd robotmcp

# Install dependencies using uv (recommended)
uv sync

# Or install using pip
pip install -e .

# Install optional dependencies for full functionality
pip install sentence-transformers beautifulsoup4
```

## Usage

### Starting the Server

Using uv (recommended):
```bash
uv run python src/robotmcp/server.py
```

Or using the FastMCP CLI:
```bash
uv run fastmcp run src/robotmcp/server.py
```

Traditional method:
```bash
robotmcp
```

### Example Workflow

1. **Analyze a test scenario**:
   ```
   "Test login functionality with valid credentials showing dashboard"
   ```

2. **Execute steps interactively**:
   - Import SeleniumLibrary
   - Open Browser to login page
   - Enter username and password
   - Click login button
   - Verify dashboard appears

3. **Generate test suite**:
   - Optimized Robot Framework test case
   - Complete with imports, setup, and teardown
   - Ready for execution in CI/CD pipelines

### Integration with Claude Desktop

Add to your Claude Desktop MCP configuration using FastMCP 2.0:

```json
{
  "mcpServers": {
    "robotmcp": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "/path/to/robotmcp/src/robotmcp/server.py"
      ],
      "cwd": "/path/to/robotmcp"
    }
  }
}
```

Or generate the configuration automatically using FastMCP CLI:
```bash
uv run fastmcp install mcp-json src/robotmcp/server.py --name "Robot Framework MCP"
```

## Dependencies

### Required
- `robotframework>=6.0`
- `fastmcp>=2.0.0`
- `pydantic>=2.0.0`
- `aiohttp>=3.8.0`

### Optional (for enhanced functionality)
- `sentence-transformers>=2.2.0` - For semantic keyword matching
- `beautifulsoup4>=4.11.0` - For DOM parsing
- `robotframework-seleniumlibrary` - For web automation
- `robotframework-requests` - For API testing
- `robotframework-databaselibrary` - For database testing

## Supported Test Contexts

- **Web Applications**: Using SeleniumLibrary for browser automation
- **Mobile Applications**: Using AppiumLibrary for mobile testing  
- **API Testing**: Using RequestsLibrary for HTTP/REST APIs
- **Database Testing**: Using DatabaseLibrary for SQL operations

## Example Generated Test Suite

```robot
*** Settings ***
Documentation    Test case that opens browser, navigates to page, enters data, performs click action, verifies result.
Library          SeleniumLibrary
Library          BuiltIn
Force Tags       automated    generated    web

*** Test Cases ***
User Login Test
    [Documentation]    Test successful user login flow
    [Tags]    login    smoke
    Open Browser    https://example.com    chrome    # Open chrome and navigate to https://example.com
    Go To    https://example.com/login
    Input Text    id=username    testuser    # Enter 'testuser' into id=username
    Input Text    id=password    testpass    # Enter 'testpass' into id=password
    Click Button    id=login-btn    # Click on id=login-btn
    Page Should Contain    Welcome    # Verify page contains 'Welcome'
    [Teardown]    Close Browser    # Cleanup: Close browser
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/

# Type checking  
mypy src/

# Linting
flake8 src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Create issues for bugs or feature requests
- Check the documentation for detailed usage examples
- Join our community discussions

## FastMCP 2.0 Migration

This project has been successfully migrated from MCP 1.0 to FastMCP 2.0, bringing the following improvements:

### Key Changes
- **Simplified Server Definition**: Uses FastMCP's declarative approach with decorators
- **Improved Performance**: Better transport handling and connection management
- **Enhanced Developer Experience**: More intuitive API and better error messages
- **Better Integration**: Seamless integration with modern MCP clients

### Migration Notes
- Dependencies updated from `mcp>=1.0.0` to `fastmcp>=2.0.0`
- Server implementation converted to use FastMCP's decorator-based tools
- Maintained full backward compatibility with existing MCP protocol features
- All 7 MCP tools (analyze_scenario, discover_keywords, execute_step, etc.) preserved

### Running with FastMCP 2.0
The server can be run in multiple ways:
1. **Direct Python execution**: `uv run python src/robotmcp/server.py`
2. **FastMCP CLI**: `uv run fastmcp run src/robotmcp/server.py`
3. **MCP Client Integration**: Use generated JSON configuration

---

**Note**: This is an MVP implementation focused on demonstrating the core concept. Production use would require additional error handling, security considerations, and integration with actual Robot Framework execution environments.