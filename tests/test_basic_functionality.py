"""Basic functionality tests for Robot Framework MCP Server."""

import asyncio
import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from robotmcp.components.nlp_processor import NaturalLanguageProcessor
from robotmcp.components.keyword_matcher import KeywordMatcher
from robotmcp.components.execution_engine import ExecutionEngine
from robotmcp.components.state_manager import StateManager
from robotmcp.components.test_builder import TestBuilder
from robotmcp.utils.validation import validate_mcp_input, InputValidator

class TestNaturalLanguageProcessor:
    """Test the Natural Language Processor component."""
    
    @pytest.fixture
    def nlp_processor(self):
        return NaturalLanguageProcessor()
    
    @pytest.mark.asyncio
    async def test_analyze_scenario_basic(self, nlp_processor):
        scenario = "Test user login with valid credentials"
        result = await nlp_processor.analyze_scenario(scenario, "web")
        
        assert result["success"] is True
        assert "scenario" in result
        assert result["scenario"]["title"]
        assert result["scenario"]["context"] == "web"
        assert len(result["scenario"]["actions"]) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_scenario_complex(self, nlp_processor):
        scenario = """
        Test e-commerce checkout flow:
        1. Navigate to product page
        2. Add item to cart
        3. Go to checkout
        4. Fill in shipping information
        5. Complete payment
        6. Verify order confirmation
        """
        result = await nlp_processor.analyze_scenario(scenario, "web")
        
        assert result["success"] is True
        assert len(result["scenario"]["actions"]) >= 4
        assert result["analysis"]["complexity"] in ["medium", "complex"]
    
    @pytest.mark.asyncio
    async def test_validate_scenario(self, nlp_processor):
        parsed_scenario = {
            "scenario": {
                "actions": [
                    {"action_type": "navigate", "description": "go to login page"},
                    {"action_type": "input", "description": "enter username"}
                ],
                "required_capabilities": ["SeleniumLibrary"]
            }
        }
        
        result = await nlp_processor.validate_scenario(
            parsed_scenario, 
            ["SeleniumLibrary", "BuiltIn"]
        )
        
        assert result["success"] is True
        assert result["feasible"] is True
        assert len(result["missing_capabilities"]) == 0

class TestKeywordMatcher:
    """Test the Keyword Matcher component."""
    
    @pytest.fixture
    def keyword_matcher(self):
        return KeywordMatcher()
    
    @pytest.mark.asyncio
    async def test_discover_keywords_click(self, keyword_matcher):
        result = await keyword_matcher.discover_keywords("click the login button", "web")
        
        assert result["success"] is True
        assert result["action_type"] == "click"
        assert len(result["matches"]) > 0
        
        # Should find click-related keywords
        click_keywords = [m for m in result["matches"] if "click" in m["keyword_name"].lower()]
        assert len(click_keywords) > 0
    
    @pytest.mark.asyncio
    async def test_discover_keywords_input(self, keyword_matcher):
        result = await keyword_matcher.discover_keywords("enter text in username field", "web")
        
        assert result["success"] is True
        assert result["action_type"] == "input"
        assert len(result["matches"]) > 0
    
    def test_normalize_action(self, keyword_matcher):
        action = "Click on the 'Login' button"
        normalized = keyword_matcher._normalize_action(action)
        
        assert "login" in normalized
        assert "button" in normalized
        assert "'" not in normalized  # Quotes should be removed
    
    def test_classify_action(self, keyword_matcher):
        assert keyword_matcher._classify_action("click the button") == "click"
        assert keyword_matcher._classify_action("type username") == "input"
        assert keyword_matcher._classify_action("navigate to page") == "navigate"
        assert keyword_matcher._classify_action("verify text appears") == "verify"

class TestExecutionEngine:
    """Test the Execution Engine component."""
    
    @pytest.fixture
    def execution_engine(self):
        return ExecutionEngine()
    
    @pytest.mark.asyncio
    async def test_execute_step_basic(self, execution_engine):
        result = await execution_engine.execute_step("Log", ["Hello World"], "test_session")
        
        assert result["success"] is True
        assert result["status"] == "pass"
        assert result["keyword"] == "Log"
        assert result["arguments"] == ["Hello World"]
    
    @pytest.mark.asyncio
    async def test_execute_step_set_variable(self, execution_engine):
        result = await execution_engine.execute_step(
            "Set Variable", 
            ["test_var", "test_value"], 
            "test_session"
        )
        
        assert result["success"] is True
        assert result["status"] == "pass"
        assert "test_var" in result["session_variables"]
        assert result["session_variables"]["test_var"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_session_management(self, execution_engine):
        session_id = "test_session_mgmt"
        
        # Execute a step to create session
        await execution_engine.execute_step("Log", ["Test"], session_id)
        
        # Get session info
        info = await execution_engine.get_session_info(session_id)
        assert info["success"] is True
        assert info["session_id"] == session_id
        assert info["total_steps"] == 1
        
        # Clear session
        result = await execution_engine.clear_session(session_id)
        assert result["success"] is True
        
        # Verify session is cleared
        info = await execution_engine.get_session_info(session_id)
        assert info["success"] is False

class TestStateManager:
    """Test the State Manager component."""
    
    @pytest.fixture
    def state_manager(self):
        return StateManager()
    
    @pytest.mark.asyncio
    async def test_get_state_basic(self, state_manager):
        result = await state_manager.get_state("all", [], "test_session")
        
        assert result["success"] is True
        assert result["session_id"] == "test_session"
        assert "dom" in result
        assert "api" in result
        assert "database" in result
        assert "variables" in result
    
    @pytest.mark.asyncio
    async def test_get_dom_state(self, state_manager):
        result = await state_manager.get_state("dom", ["button", "input"], "test_session")
        
        assert result["success"] is True
        assert "dom" in result
        assert "elements" in result["dom"]
        assert "element_count" in result["dom"]
    
    @pytest.mark.asyncio
    async def test_update_variables(self, state_manager):
        session_id = "test_vars"
        variables = {"test_key": "test_value", "another_key": 123}
        
        await state_manager.update_variables(session_id, variables)
        
        state = await state_manager.get_state("all", [], session_id)
        assert state["variables"]["test_key"] == "test_value"
        assert state["variables"]["another_key"] == 123
    
    @pytest.mark.asyncio
    async def test_state_history(self, state_manager):
        session_id = "test_history"
        
        # Create some state changes
        await state_manager.update_variables(session_id, {"step": 1})
        await state_manager.get_state("all", [], session_id)
        
        await state_manager.update_variables(session_id, {"step": 2})
        await state_manager.get_state("all", [], session_id)
        
        # Get history
        history = await state_manager.get_state_history(session_id, 10)
        assert history["success"] is True
        assert history["total_states"] >= 2

class TestTestBuilder:
    """Test the Test Builder component."""
    
    @pytest.fixture
    def test_builder(self):
        return TestBuilder()
    
    @pytest.mark.asyncio
    async def test_build_suite_basic(self, test_builder):
        result = await test_builder.build_suite(
            session_id="test_session",
            test_name="Basic Test",
            tags=["smoke", "basic"],
            documentation="A basic test case"
        )
        
        assert result["success"] is True
        assert result["suite"]["name"] == "Generated_Suite_test_session"
        assert len(result["suite"]["test_cases"]) == 1
        assert result["suite"]["test_cases"][0]["name"] == "Basic Test"
        assert "smoke" in result["suite"]["test_cases"][0]["tags"]
    
    @pytest.mark.asyncio
    async def test_rf_text_generation(self, test_builder):
        result = await test_builder.build_suite(
            session_id="test_rf_text",
            test_name="RF Text Test",
            tags=["rf"],
            documentation="Test RF text generation"
        )
        
        assert result["success"] is True
        assert result["rf_text"]
        
        # Check for Robot Framework syntax
        rf_text = result["rf_text"]
        assert "*** Settings ***" in rf_text
        assert "*** Test Cases ***" in rf_text
        assert "RF Text Test" in rf_text
        assert "Library" in rf_text

class TestValidation:
    """Test the validation utilities."""
    
    def test_validate_scenario(self):
        # Valid scenario
        valid, error = InputValidator.validate_scenario("Test user login functionality")
        assert valid is True
        assert error is None
        
        # Invalid scenarios
        valid, error = InputValidator.validate_scenario("")
        assert valid is False
        assert "empty" in error.lower()
        
        valid, error = InputValidator.validate_scenario("123")
        assert valid is False
        assert "short" in error.lower()
    
    def test_validate_keyword(self):
        # Valid keyword
        valid, error = InputValidator.validate_keyword("Click Button")
        assert valid is True
        assert error is None
        
        # Invalid keywords
        valid, error = InputValidator.validate_keyword("")
        assert valid is False
        
        valid, error = InputValidator.validate_keyword("Invalid<>Keyword")
        assert valid is False
        assert "invalid characters" in error.lower()
    
    def test_validate_arguments(self):
        # Valid arguments
        valid, error = InputValidator.validate_arguments(["arg1", "arg2"])
        assert valid is True
        assert error is None
        
        # Invalid arguments
        valid, error = InputValidator.validate_arguments("not a list")
        assert valid is False
        
        valid, error = InputValidator.validate_arguments([123, "string"])
        assert valid is False
        assert "must be a string" in error.lower()
    
    def test_validate_mcp_input(self):
        # Test analyze_scenario validation
        result = validate_mcp_input(
            "analyze_scenario",
            scenario="Test login functionality",
            context="web"
        )
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Test with invalid input
        result = validate_mcp_input(
            "analyze_scenario",
            scenario="",
            context="invalid_context"
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0

# Test runner for manual execution
if __name__ == "__main__":
    # Run a simple test to verify everything works
    async def run_basic_test():
        print("Running basic functionality test...")
        
        # Test NLP processor
        nlp = NaturalLanguageProcessor()
        result = await nlp.analyze_scenario("Test login with username and password", "web")
        print(f"NLP Test: {'PASS' if result['success'] else 'FAIL'}")
        
        # Test keyword matcher
        matcher = KeywordMatcher()
        result = await matcher.discover_keywords("click button", "web")
        print(f"Keyword Matcher Test: {'PASS' if result['success'] else 'FAIL'}")
        
        # Test execution engine
        engine = ExecutionEngine()
        result = await engine.execute_step("Log", ["Test message"], "basic_test")
        print(f"Execution Engine Test: {'PASS' if result['success'] else 'FAIL'}")
        
        # Test state manager
        state_mgr = StateManager()
        result = await state_mgr.get_state("all", [], "basic_test")
        print(f"State Manager Test: {'PASS' if result['success'] else 'FAIL'}")
        
        # Test test builder
        builder = TestBuilder()
        result = await builder.build_suite("basic_test", "Basic Test", ["test"], "Basic test")
        print(f"Test Builder Test: {'PASS' if result['success'] else 'FAIL'}")
        
        print("Basic functionality test completed!")
    
    asyncio.run(run_basic_test())