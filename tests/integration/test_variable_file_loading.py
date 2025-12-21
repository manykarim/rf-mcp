"""Integration tests for variable file loading functionality."""

import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

from fastmcp import FastMCP
from fastmcp.client import Client
import robotmcp.server

# Get the MCP server instance
mcp = robotmcp.server.mcp


class TestVariableFileLoading:
    """Test variable file loading with different file formats and scenarios."""
    
    @pytest_asyncio.fixture
    async def mcp_client(self):
        """Create an MCP client for testing."""
        async with Client(mcp) as client:
            yield client
    
    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory path."""
        return Path(__file__).parent.parent.parent / "test_data"
    
    @pytest.mark.asyncio
    async def test_static_python_variable_file(self, mcp_client, test_data_dir):
        """Test loading static Python variable file."""
        session_id = "test_static_py_vars"
        
        # Import static Python variable file
        result = await mcp_client.call_tool("manage_session", {
            "action": "import_variables",
            "session_id": session_id,
            "variable_file_path": str(test_data_dir / "static_variables.py")
        })
        
        assert result.data["success"] is True
        assert result.data["action"] == "import_variables"
        assert result.data["session_id"] == session_id
        assert result.data["variable_file"] == str(test_data_dir / "static_variables.py")
        
        # Check that variables were loaded
        variables_loaded = result.data["variables_loaded"]
        assert "SCALAR_VAR" in variables_loaded
        assert "NUMBER_VAR" in variables_loaded 
        assert "BOOLEAN_VAR" in variables_loaded
        assert "@{MY_LIST}" in variables_loaded  # Robot Framework converts LIST__ prefix to @{}
        assert "&{MY_DICT}" in variables_loaded  # Robot Framework converts DICT__ prefix to &{}
        
        # Check variable values in variables_map
        variables_map = result.data["variables_map"]
        assert variables_map["${SCALAR_VAR}"] == "test_value"
        assert variables_map["${NUMBER_VAR}"] == 42
        assert variables_map["${BOOLEAN_VAR}"] is True
        assert variables_map["@{MY_LIST}"] == ["item1", "item2", "item3"]
        assert variables_map["&{MY_DICT}"]["key1"] == "value1"
        
    @pytest.mark.asyncio
    async def test_dynamic_python_variable_file_with_args(self, mcp_client, test_data_dir):
        """Test loading dynamic Python variable file with arguments."""
        session_id = "test_dynamic_py_vars"
        
        # Import dynamic Python variable file with arguments
        result = await mcp_client.call_tool("manage_session", {
            "action": "import_variables", 
            "session_id": session_id,
            "variable_file_path": str(test_data_dir / "dynamic_variables.py"),
            "args": ["prod", "secret123"]
        })
        
        assert result.data["success"] is True
        assert result.data["args"] == ["prod", "secret123"]
        
        # Check that environment-specific variables were loaded
        variables_map = result.data["variables_map"]
        assert variables_map["${ENVIRONMENT}"] == "prod"
        assert variables_map["${API_KEY}"] == "secret123"
        assert variables_map["${BASE_URL}"] == "https://prod.example.com"
        assert "PROD_ONLY_VAR" in result.data["variables_loaded"]
        assert variables_map["${PROD_ONLY_VAR}"] == "production setting"
        
        # Check list and dict variables
        assert variables_map["@{ENDPOINTS}"] == ["/prod/api", "/prod/health"]
        config = variables_map["&{CONFIG}"]
        assert config["env"] == "prod"
        assert config["debug"] is False
        assert config["timeout"] == 30
        
    @pytest.mark.asyncio
    async def test_dynamic_python_variable_file_default_args(self, mcp_client, test_data_dir):
        """Test loading dynamic Python variable file with default arguments."""
        session_id = "test_dynamic_py_default"
        
        # Import without arguments (should use defaults)
        result = await mcp_client.call_tool("manage_session", {
            "action": "import_variables",
            "session_id": session_id, 
            "variable_file_path": str(test_data_dir / "dynamic_variables.py")
        })
        
        assert result.data["success"] is True
        assert result.data["args"] == []
        
        variables_map = result.data["variables_map"]
        assert variables_map["${ENVIRONMENT}"] == "test"  # default
        assert variables_map["${API_KEY}"] == "default"   # default
        assert "PROD_ONLY_VAR" not in result.data["variables_loaded"]  # should not exist for test env
        
    @pytest.mark.asyncio
    async def test_yaml_variable_file(self, mcp_client, test_data_dir):
        """Test loading YAML variable file."""
        session_id = "test_yaml_vars"
        
        result = await mcp_client.call_tool("manage_session", {
            "action": "import_variables",
            "session_id": session_id,
            "variable_file_path": str(test_data_dir / "test_variables.yaml")
        })
        
        assert result.data["success"] is True
        
        variables_map = result.data["variables_map"]
        assert variables_map["${YAML_SCALAR}"] == "yaml_value"
        assert variables_map["${YAML_NUMBER}"] == 123
        assert variables_map["${YAML_BOOLEAN}"] is True
        assert variables_map["@{YAML_LIST}"] == ["first_item", "second_item", 42, False]
        
        # Check nested dictionary
        yaml_dict = variables_map["&{YAML_DICT}"]
        assert yaml_dict["database"]["host"] == "localhost"
        assert yaml_dict["api"]["version"] == "v1"
        
    @pytest.mark.asyncio 
    async def test_json_variable_file(self, mcp_client, test_data_dir):
        """Test loading JSON variable file."""
        session_id = "test_json_vars"
        
        result = await mcp_client.call_tool("manage_session", {
            "action": "import_variables",
            "session_id": session_id,
            "variable_file_path": str(test_data_dir / "test_variables.json")
        })
        
        assert result.data["success"] is True
        
        variables_map = result.data["variables_map"]
        assert variables_map["${JSON_SCALAR}"] == "json_value"
        assert variables_map["${JSON_NUMBER}"] == 456
        assert variables_map["${JSON_BOOLEAN}"] is False
        assert variables_map["@{JSON_LIST}"] == ["json_item1", "json_item2", 789]

        # Check nested structure
        json_dict = variables_map["&{JSON_DICT}"]
        assert json_dict["settings"]["theme"] == "dark"
        
    @pytest.mark.asyncio
    async def test_nonexistent_file_error(self, mcp_client):
        """Test error handling for nonexistent file."""
        session_id = "test_error_nonexistent"
        
        result = await mcp_client.call_tool("manage_session", {
            "action": "import_variables",
            "session_id": session_id,
            "variable_file_path": "/path/to/nonexistent/file.py"
        })
        
        assert result.data["success"] is False
        assert "error" in result.data
        assert "nonexistent" in result.data["error"] or "not found" in result.data["error"].lower()
        
    @pytest.mark.asyncio
    async def test_invalid_syntax_error(self, mcp_client, test_data_dir):
        """Test error handling for Python file with syntax error."""
        session_id = "test_error_syntax"
        
        result = await mcp_client.call_tool("manage_session", {
            "action": "import_variables",
            "session_id": session_id,
            "variable_file_path": str(test_data_dir / "invalid_syntax.py")
        })
        
        assert result.data["success"] is False
        assert "error" in result.data
        # Should contain descriptive error message
        assert result.data["variable_file"] == str(test_data_dir / "invalid_syntax.py")
        
    @pytest.mark.asyncio
    async def test_missing_variable_file_path(self, mcp_client):
        """Test error when variable_file_path is not provided."""
        session_id = "test_missing_path"
        
        result = await mcp_client.call_tool("manage_session", {
            "action": "import_variables",
            "session_id": session_id
            # missing variable_file_path
        })
        
        assert result.data["success"] is False
        assert result.data["error"] == "variable_file_path is required"
        
    @pytest.mark.asyncio
    async def test_variable_overwrite_behavior(self, mcp_client, test_data_dir):
        """Test that re-importing variables overwrites existing ones."""
        session_id = "test_overwrite"
        
        # Set some initial variables
        await mcp_client.call_tool("manage_session", {
            "action": "set_variables", 
            "session_id": session_id,
            "variables": {"SCALAR_VAR": "initial_value"}
        })
        
        # Import variable file that contains SCALAR_VAR
        result = await mcp_client.call_tool("manage_session", {
            "action": "import_variables",
            "session_id": session_id,
            "variable_file_path": str(test_data_dir / "static_variables.py")
        })
        
        assert result.data["success"] is True
        # Should overwrite the initial value
        assert result.data["variables_map"]["${SCALAR_VAR}"] == "test_value"
        
    @pytest.mark.asyncio
    async def test_session_variable_tracking(self, mcp_client, test_data_dir):
        """Test that variable file imports are tracked in session metadata.""" 
        session_id = "test_tracking"
        
        # Import first variable file
        result1 = await mcp_client.call_tool("manage_session", {
            "action": "import_variables",
            "session_id": session_id,
            "variable_file_path": str(test_data_dir / "static_variables.py")
        })
        assert result1.data["success"] is True
        
        # Import second variable file
        result2 = await mcp_client.call_tool("manage_session", {
            "action": "import_variables", 
            "session_id": session_id,
            "variable_file_path": str(test_data_dir / "test_variables.yaml")
        })
        assert result2.data["success"] is True
        
        # Check session state contains both imports
        session_state = await mcp_client.call_tool("get_session_state", {
            "session_id": session_id,
            "sections": ["variables"]
        })
        
        assert session_state.data["success"] is True
        # Should contain variables from both files
        session_data = session_state.data["sections"]["variables"]
        actual_vars = session_data.get("variables", {})
        assert "SCALAR_VAR" in actual_vars
        assert "YAML_SCALAR" in actual_vars
        
    @pytest.mark.asyncio
    async def test_yaml_requires_pyyaml(self, mcp_client):
        """Test that YAML import provides helpful error if PyYAML not available."""
        # Note: This test assumes PyYAML is available - in a real test environment
        # you might want to mock the import to simulate missing PyYAML
        session_id = "test_yaml_dependency"
        
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("TEST_VAR: yaml_test_value\n")
            yaml_path = f.name
        
        try:
            result = await mcp_client.call_tool("manage_session", {
                "action": "import_variables",
                "session_id": session_id,
                "variable_file_path": yaml_path
            })
            
            # If PyYAML is available, this should succeed
            # If not available, should get descriptive error
            if result.data["success"]:
                assert "TEST_VAR" in result.data["variables_loaded"]
            else:
                assert "yaml" in result.data["error"].lower() or "pyyaml" in result.data["error"].lower()
                
        finally:
            os.unlink(yaml_path)
        
    @pytest.mark.asyncio
    async def test_relative_path_resolution(self, mcp_client):
        """Test that relative paths work correctly."""
        session_id = "test_relative_path"
        
        # Use relative path from current working directory
        result = await mcp_client.call_tool("manage_session", {
            "action": "import_variables",
            "session_id": session_id,
            "variable_file_path": "test_data/static_variables.py"
        })
        
        # Should work with relative path
        assert result.data["success"] is True
        assert "SCALAR_VAR" in result.data["variables_loaded"]
        
    @pytest.mark.asyncio
    async def test_alternative_action_names(self, mcp_client, test_data_dir):
        """Test that 'load_variables' action alias works."""
        session_id = "test_alias"
        
        result = await mcp_client.call_tool("manage_session", {
            "action": "load_variables",  # Alternative name 
            "session_id": session_id,
            "variable_file_path": str(test_data_dir / "static_variables.py")
        })
        
        assert result.data["success"] is True
        assert result.data["action"] == "import_variables"  # Should normalize
        assert "SCALAR_VAR" in result.data["variables_loaded"]