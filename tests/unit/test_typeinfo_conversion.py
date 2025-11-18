from enum import Enum
import inspect

from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter


class DemoBrowsers(Enum):
    chromium = "chromium"
    firefox = "firefox"


def demo_keyword(browser: DemoBrowsers, retries: int, headless: bool = False):
    return browser, retries, headless


def test_typeinfo_conversion_success():
    converter = RobotFrameworkNativeConverter()
    positional = ["firefox", "3"]
    named = {"headless": "False"}
    new_pos, new_named = converter._apply_typeinfo_conversions(  # type: ignore[attr-defined]
        "Demo Keyword",
        "DemoLib",
        positional,
        named,
        demo_keyword,
        inspect.signature(demo_keyword),
    )
    assert new_pos[0] == DemoBrowsers.firefox
    assert isinstance(new_pos[0], DemoBrowsers)
    assert new_pos[1] == 3
    assert new_named["headless"] is False


def test_typeinfo_conversion_failure():
    converter = RobotFrameworkNativeConverter()
    positional = ["safari", "1"]
    named = {}
    try:
        converter._apply_typeinfo_conversions(  # type: ignore[attr-defined]
            "Demo Keyword",
            "DemoLib",
            positional,
            named,
            demo_keyword,
            inspect.signature(demo_keyword),
        )
    except ValueError as exc:
        assert "safari" in str(exc)
    else:
        raise AssertionError("conversion did not raise for invalid enum")
