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
