"""Integration tests for analyze_scenario library preference functionality."""

import pytest

from robotmcp.models.session_models import ExecutionSession, SessionType


class TestRealScenarioConfiguration:
    """Real integration tests without mocking for scenario configuration."""

    def test_selenium_preference_end_to_end_configuration(self):
        """Test complete scenario configuration for Selenium Library preference."""
        scenario = """
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
        
        Execute the test suite stepwise and build the final version afterwards.
        Use Selenium Library
        """

        session = ExecutionSession(session_id="integration_selenium")
        session.configure_from_scenario(scenario)

        # Verify explicit preference detection
        assert session.explicit_library_preference == "SeleniumLibrary"
        assert session.session_type == SessionType.WEB_AUTOMATION

        # Verify exclusive library loading
        assert "SeleniumLibrary" in session.imported_libraries
        assert "Browser" not in session.imported_libraries

        # Verify search order prioritizes SeleniumLibrary
        assert session.search_order[0] == "SeleniumLibrary"
        assert "Browser" not in session.search_order

        # Verify core libraries are present
        assert "BuiltIn" in session.search_order
        assert "Collections" in session.search_order
        assert "String" in session.search_order

    def test_browser_preference_end_to_end_configuration(self):
        """Test complete scenario configuration for Browser Library preference."""
        scenario = """
        Use RobotMCP to create a TestSuite and execute it step wise.
        
        - Open https://www.saucedemo.com/
        - Login with valid user
        - Assert login was successful
        - Add item to cart
        - Checkout
        
        Use Browser Library for modern web automation.
        """

        session = ExecutionSession(session_id="integration_browser")
        session.configure_from_scenario(scenario)

        # Verify explicit preference detection
        assert session.explicit_library_preference == "Browser"
        assert session.session_type == SessionType.WEB_AUTOMATION

        # Verify exclusive library loading
        assert "Browser" in session.imported_libraries
        assert "SeleniumLibrary" not in session.imported_libraries

        # Verify search order prioritizes Browser Library
        assert session.search_order[0] == "Browser"
        assert "SeleniumLibrary" not in session.search_order

        # Verify core libraries are present
        assert "BuiltIn" in session.search_order
        assert "Collections" in session.search_order
        assert "String" in session.search_order

    def test_no_preference_neutral_configuration(self):
        """Test scenario configuration without explicit library preference."""
        scenario = """
        Create a comprehensive web testing suite for an e-commerce application.
        
        - Navigate to the application
        - Test user authentication
        - Test product browsing
        - Test shopping cart functionality
        - Test checkout process
        - Validate order confirmation
        """

        session = ExecutionSession(session_id="integration_neutral")
        session.configure_from_scenario(scenario)

        # Verify no explicit preference
        assert session.explicit_library_preference is None
        # Current implementation may classify broader scenarios as WEB_AUTOMATION or MIXED
        assert session.session_type in [SessionType.WEB_AUTOMATION, SessionType.MIXED]

        # No explicit web library import until used
        assert "Browser" not in session.imported_libraries
        assert "SeleniumLibrary" not in session.imported_libraries

        # Verify core libraries present in search order (Browser may be included by profile)
        assert "BuiltIn" in session.search_order
        assert "Collections" in session.search_order
        assert "String" in session.search_order


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_selenium_preference_workflow(self):
        """Test complete workflow from scenario to session configuration for Selenium."""
        scenario = """
        Use RobotMCP with SeleniumLibrary to create a TestSuite and execute it step wise.
        - Open https://www.saucedemo.com/
        - Login with valid user
        - Assert login was successful
        """

        session = ExecutionSession(session_id="e2e_selenium_test")
        session.configure_from_scenario(scenario)

        # Verify end-to-end configuration
        assert session.explicit_library_preference == "SeleniumLibrary"
        assert session.session_type == SessionType.WEB_AUTOMATION
        assert "SeleniumLibrary" in session.imported_libraries
        assert "Browser" not in session.imported_libraries
        assert session.search_order[0] == "SeleniumLibrary"

        # Verify library conflict resolution: do not include the other web lib in search order
        assert "Browser" not in session.search_order

    def test_browser_preference_workflow(self):
        """Test complete workflow from scenario to session configuration for Browser Library."""
        scenario = """
        Use RobotMCP with Browser Library to create a TestSuite and execute it step wise.
        - Open https://www.saucedemo.com/ 
        - Login with valid user
        - Assert login was successful
        """

        session = ExecutionSession(session_id="e2e_browser_test")
        session.configure_from_scenario(scenario)

        # Verify end-to-end configuration
        assert session.explicit_library_preference == "Browser"
        assert session.session_type == SessionType.WEB_AUTOMATION
        assert "Browser" in session.imported_libraries
        assert "SeleniumLibrary" not in session.imported_libraries
        assert session.search_order[0] == "Browser"

        # Verify library conflict resolution: do not include the other web lib in search order
        assert "SeleniumLibrary" not in session.search_order

    def test_scenario_library_switching(self):
        """Test that session can properly switch libraries when explicitly requested."""
        session = ExecutionSession(session_id="e2e_switching_test")

        # Start with Browser Library
        browser_scenario = "Use Browser Library for web automation testing"
        session.configure_from_scenario(browser_scenario)

        assert session.explicit_library_preference == "Browser"
        assert "Browser" in session.imported_libraries
        assert "SeleniumLibrary" not in session.imported_libraries

        # Reset session for new configuration
        session.imported_libraries.clear()
        session.search_order.clear()
        session.auto_configured = False

        # Switch to SeleniumLibrary
        selenium_scenario = "Use SeleniumLibrary for comprehensive web testing"
        session.configure_from_scenario(selenium_scenario)

        assert session.explicit_library_preference == "SeleniumLibrary"
        assert "SeleniumLibrary" in session.imported_libraries
        assert "Browser" not in session.imported_libraries
        assert session.search_order[0] == "SeleniumLibrary"

    def test_concurrent_session_isolation(self):
        """Test that multiple sessions with different preferences work independently."""
        # Create sessions with different preferences
        session1 = ExecutionSession(session_id="concurrent_selenium")
        session2 = ExecutionSession(session_id="concurrent_browser")

        scenario1 = "Use SeleniumLibrary for web testing"
        scenario2 = "Use Browser Library for modern automation"

        # Configure both sessions
        session1.configure_from_scenario(scenario1)
        session2.configure_from_scenario(scenario2)

        # Verify complete isolation
        assert session1.explicit_library_preference == "SeleniumLibrary"
        assert session2.explicit_library_preference == "Browser"

        assert "SeleniumLibrary" in session1.imported_libraries
        assert "Browser" not in session1.imported_libraries

        assert "Browser" in session2.imported_libraries
        assert "SeleniumLibrary" not in session2.imported_libraries

        assert session1.search_order[0] == "SeleniumLibrary"
        assert session2.search_order[0] == "Browser"

        # Verify no cross-contamination
        assert "Browser" not in session1.search_order
        assert "SeleniumLibrary" not in session2.search_order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
