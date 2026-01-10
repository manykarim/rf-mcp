"""Integration script for the enhanced serialization system and token-efficient output.

This module initializes the enhanced serialization system and integrates
it with the execution coordinator in the server.py module. It also provides
token-efficient output utilities for MCP tool responses.
"""

import logging
import os
from typing import Any, Dict

from robotmcp.utils.enhanced_serialization_integration import (
    apply_enhanced_serialization,
)
from robotmcp.utils.token_efficient_output import (
    TokenEfficientOutput,
    compact_response,
    error_response,
    estimate_tokens,
    optimize_execution_result,
    optimize_keyword_list,
    optimize_library_list,
    optimize_output,
    optimize_session_info,
    success_response,
    token_budget_check,
)

logger = logging.getLogger(__name__)

# Default verbosity level - can be configured via environment variable
DEFAULT_VERBOSITY = os.getenv("ROBOTMCP_OUTPUT_VERBOSITY", "standard")

# Global token-efficient output handler
_output_handler: TokenEfficientOutput = None


def get_output_handler() -> TokenEfficientOutput:
    """Get the global token-efficient output handler."""
    global _output_handler
    if _output_handler is None:
        _output_handler = TokenEfficientOutput(
            verbosity=DEFAULT_VERBOSITY,
            abbreviate_fields=(DEFAULT_VERBOSITY == "compact"),
            omit_empty=True,
        )
    return _output_handler


def set_output_verbosity(verbosity: str) -> None:
    """
    Set the global output verbosity level.

    Args:
        verbosity: Verbosity level ('compact', 'standard', 'verbose')
    """
    global _output_handler
    _output_handler = TokenEfficientOutput(
        verbosity=verbosity,
        abbreviate_fields=(verbosity == "compact"),
        omit_empty=True,
    )
    logger.info(f"Output verbosity set to: {verbosity}")


def initialize_enhanced_serialization(execution_engine):
    """
    Initialize enhanced serialization for the MCP server.

    This function should be called during server initialization to set up
    the enhanced serialization system.

    Args:
        execution_engine: The execution coordinator instance from the server
    """
    logger.info("Initializing enhanced serialization system...")
    apply_enhanced_serialization(execution_engine)
    logger.info("Enhanced serialization system initialized")

    # Initialize token-efficient output handler
    global _output_handler
    _output_handler = get_output_handler()
    logger.info(f"Token-efficient output initialized (verbosity: {DEFAULT_VERBOSITY})")


def optimize_tool_response(
    response: Dict[str, Any], verbosity: str = None
) -> Dict[str, Any]:
    """
    Optimize an MCP tool response for token efficiency.

    This is a convenience function for optimizing tool responses. It uses
    the global verbosity setting unless overridden.

    Args:
        response: The response dictionary to optimize
        verbosity: Optional verbosity override

    Returns:
        Optimized response dictionary
    """
    v = verbosity or DEFAULT_VERBOSITY
    return optimize_output(response, verbosity=v)


# Re-export commonly used functions for convenience
__all__ = [
    "initialize_enhanced_serialization",
    "get_output_handler",
    "set_output_verbosity",
    "optimize_tool_response",
    "compact_response",
    "optimize_output",
    "optimize_execution_result",
    "optimize_keyword_list",
    "optimize_library_list",
    "optimize_session_info",
    "success_response",
    "error_response",
    "estimate_tokens",
    "token_budget_check",
    "DEFAULT_VERBOSITY",
]
