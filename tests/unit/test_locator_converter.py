from robotmcp.components.execution.locator_converter import LocatorConverter
from robotmcp.models.config_models import ExecutionConfig


def _converter_with_conversion_enabled() -> LocatorConverter:
    config = ExecutionConfig()
    config.ENABLE_LOCATOR_CONVERSION = True
    return LocatorConverter(config=config)


def test_add_explicit_strategy_prefix_for_suite_text():
    converter = LocatorConverter()
    locator = "Log In Now"
    result = converter.add_explicit_strategy_prefix(
        locator, for_test_suite=True, target_library="Browser"
    )
    assert result == "text=Log In Now"


def test_add_explicit_strategy_prefix_preserves_existing():
    converter = LocatorConverter()
    locator = "css=.login-button"
    assert converter.add_explicit_strategy_prefix(locator, for_test_suite=True) == locator


def test_convert_locator_for_selenium_css():
    converter = _converter_with_conversion_enabled()
    locator = ".login-button"
    assert converter.convert_locator_for_library(locator, "SeleniumLibrary") == "css=.login-button"


def test_convert_locator_for_selenium_text_selector():
    converter = _converter_with_conversion_enabled()
    locator = "text=Submit the Form"
    converted = converter.convert_locator_for_library(locator, "SeleniumLibrary")
    assert converted.startswith("xpath=")
    assert 'Submit the Form' in converted


def test_convert_locator_for_browser_preserves_prefix():
    converter = _converter_with_conversion_enabled()
    locator = "css=.nav >> text=Docs"
    assert converter.convert_locator_for_library(locator, "Browser") == locator


def test_convert_jquery_selector_to_css(monkeypatch):
    converter = _converter_with_conversion_enabled()
    jquery = "#menu li:first a:contains('Docs')"
    converted = converter._convert_jquery_to_css(jquery)
    assert ":first-child" in converted
    assert "[title*=\"Docs\"]" in converted or "[alt*=\"Docs\"]" in converted


# Security: XPath Injection Prevention Tests
class TestXPathEscaping:
    """Test suite for XPath injection prevention."""

    def test_escape_xpath_string_simple_text(self):
        """Simple text without quotes should be wrapped in double quotes."""
        converter = LocatorConverter()
        result = converter._escape_xpath_string("Hello World")
        assert result == '"Hello World"'

    def test_escape_xpath_string_with_single_quote(self):
        """Text with single quote should be wrapped in double quotes."""
        converter = LocatorConverter()
        result = converter._escape_xpath_string("It's a test")
        assert result == '"It\'s a test"'

    def test_escape_xpath_string_with_double_quote(self):
        """Text with double quote should be wrapped in single quotes."""
        converter = LocatorConverter()
        result = converter._escape_xpath_string('Say "Hello"')
        assert result == '\'Say "Hello"\''

    def test_escape_xpath_string_with_both_quotes(self):
        """Text with both quote types should use concat()."""
        converter = LocatorConverter()
        result = converter._escape_xpath_string("It's \"complex\"")
        assert result.startswith("concat(")
        assert "It" in result
        assert "complex" in result

    def test_convert_text_locator_prevents_injection(self):
        """Ensure malicious input cannot break out of XPath expression."""
        converter = _converter_with_conversion_enabled()
        # Attempt XPath injection with closing quote and comment
        malicious = 'text=Hello")]|//*[contains(@id,"'
        result = converter.convert_locator_for_library(malicious, "SeleniumLibrary")
        # The injection attempt should be safely escaped
        assert "xpath=" in result
        # The malicious content should be inside single quotes (because it contains double quotes)
        # This prevents the injection from breaking out of the XPath expression
        # The result should be: xpath=//*[contains(text(),'Hello")]|//*[contains(@id,"')]
        # where the entire malicious string is safely contained within single quotes
        assert result == 'xpath=//*[contains(text(),\'Hello")]|//*[contains(@id,"\')]'

    def test_convert_text_locator_prevents_complex_injection(self):
        """Ensure complex injection with both quote types is safely handled."""
        converter = _converter_with_conversion_enabled()
        # Injection attempt using both quote types to try to escape
        malicious = "text=Test')] | //*[@id=\"admin"
        result = converter.convert_locator_for_library(malicious, "SeleniumLibrary")
        # Should use concat() because input contains both quote types
        assert "concat(" in result
        # Verify the XPath structure is maintained
        assert result.startswith("xpath=//*[contains(text(),")

    def test_convert_text_locator_simple(self):
        """Simple text locator conversion should work correctly."""
        converter = _converter_with_conversion_enabled()
        locator = "text=Login"
        result = converter.convert_locator_for_library(locator, "SeleniumLibrary")
        assert result == 'xpath=//*[contains(text(),"Login")]'

    def test_convert_text_locator_with_single_quote(self):
        """Text locator with single quote should be safely escaped."""
        converter = _converter_with_conversion_enabled()
        locator = "text=It's working"
        result = converter.convert_locator_for_library(locator, "SeleniumLibrary")
        assert 'xpath=//*[contains(text(),' in result
        assert "It's working" in result

    def test_convert_text_locator_with_double_quote(self):
        """Text locator with double quote should use single quote wrapping."""
        converter = _converter_with_conversion_enabled()
        locator = 'text=Say "Hello"'
        result = converter.convert_locator_for_library(locator, "SeleniumLibrary")
        assert 'xpath=//*[contains(text(),' in result
        # Should use single quotes for the value
        assert "'Say \"Hello\"'" in result


class TestInputValidation:
    """Test suite for input validation."""

    def test_validate_locator_rejects_null_bytes(self):
        """Locators with null bytes should be rejected."""
        converter = LocatorConverter()
        import pytest
        with pytest.raises(ValueError, match="null bytes"):
            converter._validate_locator_input("test\x00injection")

    def test_validate_locator_rejects_too_long(self):
        """Extremely long locators should be rejected."""
        converter = LocatorConverter()
        import pytest
        with pytest.raises(ValueError, match="maximum length"):
            converter._validate_locator_input("a" * 20000)

    def test_validate_locator_allows_normal_input(self):
        """Normal locators should pass validation."""
        converter = LocatorConverter()
        result = converter._validate_locator_input("text=Hello World")
        assert result == "text=Hello World"

    def test_validate_locator_allows_empty(self):
        """Empty locators should pass through."""
        converter = LocatorConverter()
        result = converter._validate_locator_input("")
        assert result == ""
