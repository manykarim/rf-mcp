import os
import sys

# Ensure src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter


def test_named_arg_with_equals_in_value():
    conv = RobotFrameworkNativeConverter()
    pos, named = conv._split_args_into_positional_and_named(["key=a=b"], ["key: str"])
    assert pos == []
    assert named == {"key": "a=b"}


def test_positional_arg_with_equals_in_value():
    conv = RobotFrameworkNativeConverter()
    pos, named = conv._split_args_into_positional_and_named(["a=b", "c"], ["first: str", "second: str"])
    assert pos == ["a=b", "c"]
    assert named == {}


def test_combined_positional_and_named():
    conv = RobotFrameworkNativeConverter()
    pos, named = conv._split_args_into_positional_and_named(["val1", "key=val"], ["arg1: str", "key: str = None"])
    assert pos == ["val1"]
    assert named == {"key": "val"}


def test_named_before_required_becomes_positional():
    conv = RobotFrameworkNativeConverter()
    pos, named = conv._split_args_into_positional_and_named(["key=value"], ["first: str", "key: str"])
    assert pos == ["key=value"]
    assert named == {}


def test_varargs_and_kwargs():
    conv = RobotFrameworkNativeConverter()
    pos, named = conv._split_args_into_positional_and_named(
        ["1", "2", "a=3", "b=4"], ["first: str", "*rest", "**kw"]
    )
    assert pos == ["1", "2"]
    assert named == {"a": "3", "b": "4"}
