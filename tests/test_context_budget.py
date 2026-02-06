"""Unit tests for context.py budget-based element formatting.

Tests that _format_context_for_ai properly handles:
1. Category-based budgets (links, inputs, buttons, text_elements)
2. Deduplication of identical selectors
3. Complex pages with many elements
"""

import pytest
from robotmcp.lib.context import RFContextBridge


class TestFormatContextBudgets:
    """Test budget system for _format_context_for_ai."""

    @pytest.fixture
    def bridge(self):
        """Create a bridge instance for testing (without RF)."""
        return RFContextBridge()

    def test_links_budget_limit(self, bridge):
        """Links should be limited to budget (12)."""
        # Create 20 links - more than budget
        links = [
            {"text": f"Link {i}", "data_test": f"link-{i}"}
            for i in range(20)
        ]
        context = {"links": links}

        result = bridge._format_context_for_ai(context)

        # Should have exactly 12 links (budget limit)
        link_count = result.count("link '")
        assert link_count == 12, f"Expected 12 links, got {link_count}"

    def test_inputs_budget_limit(self, bridge):
        """Inputs should be limited to budget (12)."""
        inputs = [
            {"type": "text", "data_test": f"input-{i}"}
            for i in range(20)
        ]
        context = {"inputs": inputs}

        result = bridge._format_context_for_ai(context)

        input_count = result.count("input[type=")
        assert input_count == 12, f"Expected 12 inputs, got {input_count}"

    def test_buttons_budget_limit(self, bridge):
        """Buttons should be limited to budget (12)."""
        buttons = [
            {"text": f"Button {i}", "data_test": f"button-{i}"}
            for i in range(20)
        ]
        context = {"buttons": buttons}

        result = bridge._format_context_for_ai(context)

        button_count = result.count("button '")
        assert button_count == 12, f"Expected 12 buttons, got {button_count}"

    def test_text_elements_budget_limit(self, bridge):
        """Text elements should be limited to budget (14)."""
        text_elements = [
            {"text": f"Text {i}", "data_test": f"text-{i}", "type": "span"}
            for i in range(30)
        ]
        context = {"text_elements": text_elements}

        result = bridge._format_context_for_ai(context)

        text_count = result.count("span '")
        assert text_count == 14, f"Expected 14 text elements, got {text_count}"

    def test_complex_page_all_categories_represented(self, bridge):
        """Complex page with many elements should have all categories."""
        context = {
            "links": [
                {"text": f"Link {i}", "data_test": f"link-{i}"}
                for i in range(50)
            ],
            "inputs": [
                {"type": "text", "data_test": f"input-{i}"}
                for i in range(50)
            ],
            "buttons": [
                {"text": f"Button {i}", "data_test": f"btn-{i}"}
                for i in range(50)
            ],
            "text_elements": [
                {"text": f"Product {i}", "data_test": f"prod-{i}", "type": "div"}
                for i in range(50)
            ],
        }

        result = bridge._format_context_for_ai(context)

        # All categories should be represented
        assert "link '" in result, "Links should be included"
        assert "input[type=" in result, "Inputs should be included"
        assert "button '" in result, "Buttons should be included"
        assert "div '" in result, "Text elements should be included"

        # Total elements should be around 50 (12+12+12+14)
        lines = [line for line in result.split("\n") if line.strip().startswith("- ")]
        assert len(lines) == 50, f"Expected 50 total elements, got {len(lines)}"

    def test_deduplication(self, bridge):
        """Duplicate selectors should be removed."""
        # Same selector appears in links and buttons
        links = [
            {"text": "Cart", "data_test": "shopping-cart"},
            {"text": "Cart Icon", "data_test": "shopping-cart"},  # Duplicate!
        ]
        context = {"links": links}

        result = bridge._format_context_for_ai(context)

        # Should only have one shopping-cart selector
        assert result.count("shopping-cart") == 1

    def test_shopping_cart_included(self, bridge):
        """Shopping cart link should be included (original bug test)."""
        context = {
            "links": [
                {"text": "", "class": "shopping_cart_link", "data_test": "shopping-cart-link"},
                {"text": "Products", "data_test": "products"},
            ],
            "buttons": [
                {"text": f"Add to cart {i}", "data_test": f"add-to-cart-{i}"}
                for i in range(15)  # Many buttons
            ],
            "text_elements": [
                {"text": f"Product {i}", "data_test": f"item-name-{i}", "type": "div"}
                for i in range(20)
            ],
        }

        result = bridge._format_context_for_ai(context)

        # Shopping cart link MUST be included (this was the original bug)
        assert "shopping-cart-link" in result, "Shopping cart link must be included!"

    def test_priority_data_test_over_id(self, bridge):
        """data-test attribute should be preferred over id."""
        links = [
            {"text": "Link", "id": "my-link", "data_test": "preferred-selector"},
        ]
        context = {"links": links}

        result = bridge._format_context_for_ai(context)

        assert "preferred-selector" in result
        assert "#my-link" not in result

    def test_empty_context(self, bridge):
        """Empty context should return empty string."""
        result = bridge._format_context_for_ai({})
        assert result == ""

    def test_links_without_text_use_data_test_description(self, bridge):
        """Links without text should use data-test as description."""
        links = [
            {"text": "", "data_test": "shopping-cart-link"},
        ]
        context = {"links": links}

        result = bridge._format_context_for_ai(context)

        # Should generate readable description from data-test
        assert "shopping cart link" in result.lower()
