"""Regression tests for session configuration bugs."""

import pytest

from robotmcp.models.session_models import ExecutionSession, SessionType


class TestSessionConfigurationRegression:
    """Regression tests to prevent future session configuration bugs."""

    def test_explicit_selenium_preference_regression(self):
        """
        REGRESSION TEST: Ensure Selenium Library preference doesn't load Browser Library.

        This test specifically addresses the bug where analyze_scenario created sessions
        with Browser Library even when Selenium Library was explicitly requested.

        Bug Report: Session created with "Use Selenium Library" contained both
        Browser and SeleniumLibrary, violating exclusive library principle.
        """
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

        session = ExecutionSession(session_id="regression_selenium")
        session.configure_from_scenario(scenario)

        # CRITICAL: Verify Browser Library is NOT present
        assert "Browser" not in session.imported_libraries, (
            "REGRESSION: Browser Library should not be imported when SeleniumLibrary is explicitly preferred"
        )

        assert "Browser" not in session.search_order, (
            "REGRESSION: Browser Library should not be in search order when SeleniumLibrary is explicitly preferred"
        )

        # Verify SeleniumLibrary IS present and prioritized
        assert session.explicit_library_preference == "SeleniumLibrary"
        assert "SeleniumLibrary" in session.imported_libraries
        assert session.search_order[0] == "SeleniumLibrary"

    def test_explicit_browser_preference_regression(self):
        """
        REGRESSION TEST: Ensure Browser Library preference doesn't load SeleniumLibrary.

        Mirror test for Browser Library to ensure the fix works both ways.
        """
        scenario = """
        Use RobotMCP to create a TestSuite and execute it step wise.
        
        - Open https://www.saucedemo.com/
        - Login with valid user
        - Checkout process
        
        Use Browser Library for modern web automation.
        """

        session = ExecutionSession(session_id="regression_browser")
        session.configure_from_scenario(scenario)

        # CRITICAL: Verify SeleniumLibrary is NOT present
        assert "SeleniumLibrary" not in session.imported_libraries, (
            "REGRESSION: SeleniumLibrary should not be imported when Browser Library is explicitly preferred"
        )

        assert "SeleniumLibrary" not in session.search_order, (
            "REGRESSION: SeleniumLibrary should not be in search order when Browser Library is explicitly preferred"
        )

        # Verify Browser Library IS present and prioritized
        assert session.explicit_library_preference == "Browser"
        assert "Browser" in session.imported_libraries
        assert session.search_order[0] == "Browser"

    def test_no_double_library_loading_regression(self):
        """
        REGRESSION TEST: Ensure session configuration doesn't load both web libraries.

        This addresses the core issue where both Browser and SeleniumLibrary
        would be loaded simultaneously, causing keyword conflicts.
        """
        # Test with various explicit preference scenarios
        test_cases = [
            ("Use SeleniumLibrary for testing", "SeleniumLibrary", "Browser"),
            ("Use Browser Library for testing", "Browser", "SeleniumLibrary"),
            ("Use selenium for web automation", "SeleniumLibrary", "Browser"),
            ("Use browser library with playwright", "Browser", "SeleniumLibrary"),
        ]

        for scenario, expected_lib, forbidden_lib in test_cases:
            session = ExecutionSession(session_id=f"regression_{expected_lib.lower()}")
            session.configure_from_scenario(scenario)

            # Verify only the expected library is loaded
            assert expected_lib in session.imported_libraries, (
                f"Expected library {expected_lib} should be imported for scenario: {scenario}"
            )

            assert forbidden_lib not in session.imported_libraries, (
                f"Forbidden library {forbidden_lib} should NOT be imported for scenario: {scenario}"
            )

            assert expected_lib in session.search_order, (
                f"Expected library {expected_lib} should be in search order for scenario: {scenario}"
            )

            assert forbidden_lib not in session.search_order, (
                f"Forbidden library {forbidden_lib} should NOT be in search order for scenario: {scenario}"
            )

    def test_search_order_consistency_regression(self):
        """
        REGRESSION TEST: Ensure search order is consistent and logical.

        Previously, search order could contain conflicting libraries or
        have inconsistent ordering based on configuration timing.
        """
        session = ExecutionSession(session_id="regression_search_order")
        scenario = "Use SeleniumLibrary for comprehensive web testing"

        session.configure_from_scenario(scenario)

        # Verify search order properties
        search_order = session.search_order

        # No duplicates
        assert len(search_order) == len(set(search_order)), (
            "REGRESSION: Search order should not contain duplicates"
        )

        # Explicit preference comes first
        assert search_order[0] == session.explicit_library_preference, (
            "REGRESSION: Explicit library preference should be first in search order"
        )

        # No conflicting libraries present (other web lib not in search order)
        other = "Browser" if session.explicit_library_preference == "SeleniumLibrary" else "SeleniumLibrary"
        assert other not in search_order

        # BuiltIn should always be present (core Robot Framework library)
        assert "BuiltIn" in search_order, (
            "REGRESSION: BuiltIn library should always be in search order"
        )

    def test_profile_override_regression(self):
        """
        REGRESSION TEST: Ensure custom profiles properly override default profiles.

        Previously, default WEB_AUTOMATION profile would leak Browser Library
        even when custom SeleniumLibrary profile was created.
        """
        session = ExecutionSession(session_id="regression_profile_override")
        scenario = "Use SeleniumLibrary for web automation testing"

        # Configure session
        session.configure_from_scenario(scenario)

        # Get the profile that should be applied
        profiles = session._get_session_profiles()
        applied_profile = session._get_profile_for_preferences(profiles)

        # Verify custom profile was created for SeleniumLibrary
        assert applied_profile is not None
        assert applied_profile.session_type == SessionType.WEB_AUTOMATION

        # Verify custom profile has correct libraries
        assert "SeleniumLibrary" in applied_profile.core_libraries
        assert "Browser" not in applied_profile.core_libraries, (
            "REGRESSION: Custom SeleniumLibrary profile should not contain Browser Library"
        )

        # Verify search order reflects custom profile
        assert "SeleniumLibrary" in applied_profile.search_order
        assert "Browser" not in applied_profile.search_order, (
            "REGRESSION: Custom profile search order should not contain conflicting libraries"
        )

    def test_concurrent_session_isolation_regression(self):
        """
        REGRESSION TEST: Ensure multiple sessions don't interfere with each other.

        Verify that library preferences in one session don't affect another session.
        """
        # Create two sessions with different preferences
        session1 = ExecutionSession(session_id="regression_session1")
        session2 = ExecutionSession(session_id="regression_session2")

        scenario1 = "Use SeleniumLibrary for web testing"
        scenario2 = "Use Browser Library for modern automation"

        # Configure both sessions
        session1.configure_from_scenario(scenario1)
        session2.configure_from_scenario(scenario2)

        # Verify session isolation
        assert session1.explicit_library_preference == "SeleniumLibrary"
        assert session2.explicit_library_preference == "Browser"

        # Verify libraries don't cross-contaminate
        assert "SeleniumLibrary" in session1.imported_libraries
        assert "Browser" not in session1.imported_libraries

        assert "Browser" in session2.imported_libraries
        assert "SeleniumLibrary" not in session2.imported_libraries

        # Verify search orders are independent
        assert session1.search_order[0] == "SeleniumLibrary"
        assert session2.search_order[0] == "Browser"

        assert "Browser" not in session1.search_order
        assert "SeleniumLibrary" not in session2.search_order

    def test_library_removal_idempotent_regression(self):
        """
        REGRESSION TEST: Ensure library removal is idempotent and safe.

        Removing a library that's not present should not cause errors.
        """
        session = ExecutionSession(session_id="regression_removal")

        # Library removal API not exposed; ensure re-import is idempotent
        session.import_library("Browser")
        session.import_library("Browser")
        assert session.imported_libraries.count("Browser") == 1

        # Should still be able to configure normally
        scenario = "Use SeleniumLibrary for testing"
        session.configure_from_scenario(scenario)

        assert session.explicit_library_preference == "SeleniumLibrary"
        assert "SeleniumLibrary" in session.imported_libraries

    def test_performance_regression(self):
        """
        REGRESSION TEST: Ensure session configuration performance is acceptable.

        Session configuration should complete quickly even with complex scenarios.
        """
        import time

        complex_scenario = """
        Use SeleniumLibrary to create a comprehensive TestSuite and execute it step wise.
        
        This is a complex scenario with many requirements:
        - Open https://www.saucedemo.com/
        - Login with multiple user types (standard, problem, performance, locked)
        - Navigate through product catalog with filtering and sorting
        - Add multiple items to cart with quantity variations
        - Modify cart contents (add, remove, update quantities)
        - Proceed through multi-step checkout process
        - Validate order confirmation and details
        - Test logout functionality
        - Verify session persistence and security
        - Test error handling for invalid inputs
        - Validate responsive design elements
        - Test accessibility features
        - Perform cross-browser compatibility checks
        
        Execute the test suite stepwise and build the final version afterwards.
        Use Selenium Library for all web automation tasks.
        """

        start_time = time.time()

        session = ExecutionSession(session_id="regression_performance")
        session.configure_from_scenario(complex_scenario)

        end_time = time.time()
        configuration_time = end_time - start_time

        # Configuration should complete within reasonable time (< 1 second)
        assert configuration_time < 1.0, (
            f"REGRESSION: Session configuration took {configuration_time:.3f}s, should be < 1.0s"
        )

        # Verify configuration still worked correctly despite complexity
        assert session.explicit_library_preference == "SeleniumLibrary"
        assert "SeleniumLibrary" in session.imported_libraries
        assert "Browser" not in session.imported_libraries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
