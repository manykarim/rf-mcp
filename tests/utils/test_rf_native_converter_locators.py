import os
import sys

# Ensure src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter
from robotmcp.models.library_models import KeywordInfo


def test_locator_vs_named_argument_detection():
    converter = RobotFrameworkNativeConverter()
    keyword = KeywordInfo(name="Click", library="Browser", method_name="click", args=["locator"])

    positional, named = converter._split_args_into_positional_and_named(["id=foo"], keyword.args)

    # Locator strings like "id=foo" should remain positional
    assert positional == ["id=foo"]
    assert named == {}
