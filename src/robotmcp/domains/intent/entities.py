"""Intent Domain Entities.

Entities have identity and mutable state. Each IntentMapping is
identified by its (intent_verb, library) composite key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .value_objects import IntentTarget, IntentVerb, NormalizedLocator


class ArgumentTransformer(Protocol):
    """Protocol for intent-to-keyword argument transformation.

    Each IntentMapping can have a custom transformer that converts
    the intent's (target, value, options) into the keyword's expected
    argument list.
    """
    def __call__(
        self,
        target: Optional[IntentTarget],
        value: Optional[str],
        normalized_locator: Optional[NormalizedLocator],
        options: Optional[Dict[str, str]],
    ) -> List[str]:
        """Transform intent arguments to keyword arguments.

        Args:
            target: The resolved target (locator), if applicable
            value: The value argument (e.g., text to type), if applicable
            normalized_locator: Locator after normalization
            options: Additional named options

        Returns:
            List of positional string arguments for the RF keyword
        """
        ...


@dataclass
class IntentMapping:
    """Maps an intent verb to a concrete RF keyword for a specific library.

    Attributes:
        intent_verb: The abstract intent (e.g., CLICK)
        library: The RF library name (e.g., "Browser", "SeleniumLibrary")
        keyword: The concrete RF keyword name
        requires_target: Whether this intent requires a locator target
        requires_value: Whether this intent requires a value argument
        argument_transformer: Optional custom argument builder
        notes: Human-readable notes about edge cases
        timeout_category: Timeout category for the TimeoutPolicy domain
            ("action", "navigation", "assertion", "read")
    """
    intent_verb: IntentVerb
    library: str
    keyword: str
    requires_target: bool = True
    requires_value: bool = False
    argument_transformer: Optional[ArgumentTransformer] = None
    notes: str = ""
    timeout_category: str = "action"

    @property
    def mapping_key(self) -> Tuple[IntentVerb, str]:
        """Composite key for registry lookup."""
        return (self.intent_verb, self.library)

    def build_arguments(
        self,
        target: Optional[IntentTarget],
        value: Optional[str],
        normalized_locator: Optional[NormalizedLocator],
        options: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Build the keyword argument list from intent parameters.

        If a custom argument_transformer is provided, delegates to it.
        Otherwise, applies the default positional argument convention:
            [locator] [value] [named_options...]

        Args:
            target: Resolved target
            value: Value argument
            normalized_locator: Normalized locator for target library
            options: Named keyword options

        Returns:
            List of string arguments for execute_step
        """
        if self.argument_transformer:
            return self.argument_transformer(
                target, value, normalized_locator, options
            )

        args: List[str] = []

        # Position 0: locator (if target required)
        if self.requires_target and normalized_locator:
            args.append(normalized_locator.value)
        elif self.requires_target and target:
            args.append(target.locator)

        # Position 1: value (if value required)
        if self.requires_value and value is not None:
            args.append(value)

        # Named options as "key=value" strings
        if options:
            for k, v in options.items():
                args.append(f"{k}={v}")

        return args
