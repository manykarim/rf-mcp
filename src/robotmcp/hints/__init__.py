"""Public exports for the RobotCode-native error hinting package."""

from robotmcp.hints.builtin_rules import BUILTIN_HINT_RULES
from robotmcp.hints.engine import Hint, HintContext, HintEngine, HintRule
from robotmcp.hints.loader import load_custom_rules, load_custom_rules_with_errors

__all__ = [
    "BUILTIN_HINT_RULES",
    "Hint",
    "HintContext",
    "HintEngine",
    "HintRule",
    "load_custom_rules",
    "load_custom_rules_with_errors",
]
