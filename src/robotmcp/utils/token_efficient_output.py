"""Token-efficient output utilities for MCP tool responses.

This module provides utilities to reduce token consumption in MCP tool outputs
while preserving all information necessary for agents to effectively use the tools.

Key Optimization Strategies:
1. Omit null/empty fields by default
2. Use abbreviations for common field names
3. Provide configurable verbosity levels (compact, standard, verbose)
4. Truncate large content with intelligent summarization
5. Flatten unnecessary nesting
6. Remove redundant metadata

Usage:
    from robotmcp.utils.token_efficient_output import compact_response, optimize_output

    # Compact a response dict
    result = compact_response({"success": True, "result": data, "metadata": None})

    # Apply optimization with verbosity level
    result = optimize_output(response, verbosity="compact")
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


# Field name abbreviations for compact mode
# These save tokens in high-frequency responses
FIELD_ABBREVIATIONS = {
    "success": "ok",
    "error": "err",
    "message": "msg",
    "result": "res",
    "status": "st",
    "keyword": "kw",
    "arguments": "args",
    "session_id": "sid",
    "execution_time": "time",
    "step_id": "id",
    "output": "out",
    "description": "desc",
    "library": "lib",
    "libraries": "libs",
    "keywords": "kws",
    "count": "n",
    "total": "tot",
    "items": "items",  # Keep as-is (common)
    "metadata": "meta",
    "original_type": "type",
    "attributes": "attrs",
    "documentation": "doc",
    "short_doc": "sdoc",
    "assigned_variables": "vars",
    "session_variables": "svars",
    "resolved_arguments": "rargs",
    "active_library": "alib",
    "browser_state": "bstate",
}

# Fields that should be omitted when their value is empty/null/default
OMIT_WHEN_EMPTY = {
    "error",
    "message",
    "metadata",
    "description",
    "hint",
    "guidance",
    "warnings",
    "notes",
    "details",
    "extra",
    "context",
    "trace",
    "traceback",
    "stack_trace",
    "debug_info",
    "assigned_variables",
    "resolved_arguments",
    "state_updates",
}

# Fields that are redundant when success is determinable from other fields
REDUNDANT_FIELDS = {
    "status",  # Often redundant with "success"
}

# Maximum lengths for various content types
MAX_LENGTHS = {
    "string": 500,  # Default max string length
    "list": 20,  # Default max list items
    "dict": 20,  # Default max dict items
    "error": 300,  # Max error message length
    "output": 1000,  # Max output field length
    "doc": 200,  # Max documentation length
    "traceback": 500,  # Max traceback length
}


class TokenEfficientOutput:
    """Class for creating token-efficient MCP outputs."""

    def __init__(
        self,
        verbosity: str = "standard",
        abbreviate_fields: bool = False,
        omit_empty: bool = True,
        max_string_length: int = 500,
        max_list_items: int = 20,
        max_dict_items: int = 20,
    ):
        """
        Initialize the token-efficient output handler.

        Args:
            verbosity: Output verbosity level ('compact', 'standard', 'verbose')
            abbreviate_fields: Whether to use abbreviated field names
            omit_empty: Whether to omit null/empty fields
            max_string_length: Maximum length for string values
            max_list_items: Maximum number of list items to include
            max_dict_items: Maximum number of dict entries to include
        """
        self.verbosity = verbosity
        self.abbreviate_fields = abbreviate_fields or (verbosity == "compact")
        self.omit_empty = omit_empty
        self.max_string_length = max_string_length
        self.max_list_items = max_list_items
        self.max_dict_items = max_dict_items

    def optimize(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a response dictionary for token efficiency.

        Args:
            response: The response dictionary to optimize

        Returns:
            Optimized response dictionary
        """
        if not isinstance(response, dict):
            return response

        optimized = {}

        for key, value in response.items():
            # Skip empty values if configured
            if self.omit_empty and self._is_empty(value) and key in OMIT_WHEN_EMPTY:
                continue

            # Skip redundant fields
            if key in REDUNDANT_FIELDS and self._is_redundant(key, response):
                continue

            # Abbreviate field name if configured
            output_key = (
                FIELD_ABBREVIATIONS.get(key, key) if self.abbreviate_fields else key
            )

            # Optimize the value based on its type
            optimized[output_key] = self._optimize_value(key, value)

        return optimized

    def _is_empty(self, value: Any) -> bool:
        """Check if a value is considered empty."""
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
        return False

    def _is_redundant(self, key: str, response: Dict[str, Any]) -> bool:
        """Check if a field is redundant given other fields."""
        if key == "status" and "success" in response:
            # Status is redundant if success is present and they match
            success = response.get("success")
            status = response.get("status")
            if isinstance(success, bool):
                return (success and status in ("success", "completed", "passed")) or (
                    not success and status in ("failed", "error")
                )
        return False

    def _optimize_value(self, key: str, value: Any) -> Any:
        """Optimize a value based on its type and the field key."""
        if value is None:
            return None

        # Handle strings
        if isinstance(value, str):
            # Use configured max_string_length if no specific max for this key
            max_len = min(
                MAX_LENGTHS.get(key, self.max_string_length), self.max_string_length
            )
            if len(value) > max_len:
                return value[:max_len] + "..." + f"[+{len(value) - max_len}]"
            return value

        # Handle lists
        if isinstance(value, list):
            if len(value) > self.max_list_items:
                truncated = [
                    self._optimize_value(key, item)
                    for item in value[: self.max_list_items]
                ]
                return {"items": truncated, "n": len(value), "truncated": True}
            return [self._optimize_value(key, item) for item in value]

        # Handle dicts recursively
        if isinstance(value, dict):
            if len(value) > self.max_dict_items:
                items = list(value.items())[: self.max_dict_items]
                truncated = {k: self._optimize_value(k, v) for k, v in items}
                truncated["_n"] = len(value)
                truncated["_truncated"] = True
                return truncated
            return {k: self._optimize_value(k, v) for k, v in value.items()}

        # Return other types as-is
        return value


def compact_response(
    response: Dict[str, Any],
    omit_empty: bool = True,
    abbreviate: bool = False,
) -> Dict[str, Any]:
    """
    Create a compact version of a response dictionary.

    This is a convenience function that applies common optimizations:
    - Removes null/empty optional fields
    - Optionally abbreviates field names
    - Truncates long strings

    Args:
        response: Response dictionary to compact
        omit_empty: Whether to omit empty fields
        abbreviate: Whether to use abbreviated field names

    Returns:
        Compacted response dictionary
    """
    handler = TokenEfficientOutput(
        verbosity="compact" if abbreviate else "standard",
        abbreviate_fields=abbreviate,
        omit_empty=omit_empty,
    )
    return handler.optimize(response)


def optimize_output(
    response: Dict[str, Any], verbosity: str = "standard"
) -> Dict[str, Any]:
    """
    Optimize output based on verbosity level.

    Args:
        response: Response dictionary to optimize
        verbosity: Verbosity level ('compact', 'standard', 'verbose')

    Returns:
        Optimized response dictionary
    """
    handler = TokenEfficientOutput(verbosity=verbosity)
    return handler.optimize(response)


# ============================================================================
# Tool-specific optimizers
# ============================================================================


def optimize_keyword_list(
    keywords: List[Dict[str, Any]], verbosity: str = "standard"
) -> List[Dict[str, Any]]:
    """
    Optimize a list of keywords for minimal token usage.

    Args:
        keywords: List of keyword dictionaries
        verbosity: Verbosity level

    Returns:
        Optimized keyword list
    """
    if verbosity == "compact":
        # Return just essential fields: name, library, arg count
        return [
            {
                "name": kw.get("name"),
                "lib": kw.get("library"),
                "args": len(kw.get("args", kw.get("arguments", []))),
            }
            for kw in keywords
        ]
    elif verbosity == "standard":
        # Return name, library, args (names only), short_doc (truncated)
        return [
            {
                "name": kw.get("name"),
                "library": kw.get("library"),
                "args": kw.get("args", kw.get("arguments", [])),
                "doc": (kw.get("short_doc", "") or "")[:100],
            }
            for kw in keywords
        ]
    else:  # verbose
        return keywords


def optimize_library_list(
    libraries: List[Dict[str, Any]], verbosity: str = "standard"
) -> List[Dict[str, Any]]:
    """
    Optimize a list of libraries for minimal token usage.

    Args:
        libraries: List of library dictionaries
        verbosity: Verbosity level

    Returns:
        Optimized library list
    """
    if verbosity == "compact":
        return [
            {"name": lib.get("name"), "kw_count": lib.get("keyword_count", 0)}
            for lib in libraries
        ]
    elif verbosity == "standard":
        return [
            {
                "name": lib.get("name"),
                "keywords": lib.get("keyword_count", 0),
                "status": lib.get("status", "unknown"),
            }
            for lib in libraries
        ]
    else:  # verbose
        return libraries


def optimize_execution_result(
    result: Dict[str, Any], verbosity: str = "standard"
) -> Dict[str, Any]:
    """
    Optimize an execution result for minimal token usage.

    Args:
        result: Execution result dictionary
        verbosity: Verbosity level

    Returns:
        Optimized execution result
    """
    # Always include success/ok status
    optimized = {"ok": result.get("success", False)}

    # Include error if present
    if not result.get("success") and result.get("error"):
        err = result.get("error", "")
        optimized["err"] = err[:300] if len(err) > 300 else err

    if verbosity == "compact":
        # Minimal: just success/error and essential output
        if result.get("output"):
            out = str(result["output"])
            optimized["out"] = out[:200] if len(out) > 200 else out
    elif verbosity == "standard":
        # Standard: include keyword, output, execution time
        if result.get("keyword"):
            optimized["kw"] = result["keyword"]
        if result.get("output"):
            out = str(result["output"])
            optimized["out"] = out[:500] if len(out) > 500 else out
        if result.get("execution_time"):
            optimized["time"] = round(result["execution_time"], 3)
        if result.get("assigned_variables"):
            optimized["vars"] = result["assigned_variables"]
    else:  # verbose
        # Return full result with basic optimization
        handler = TokenEfficientOutput(verbosity="verbose", omit_empty=True)
        return handler.optimize(result)

    return optimized


def optimize_session_info(
    session_info: Dict[str, Any], verbosity: str = "standard"
) -> Dict[str, Any]:
    """
    Optimize session info for minimal token usage.

    Args:
        session_info: Session information dictionary
        verbosity: Verbosity level

    Returns:
        Optimized session info
    """
    if verbosity == "compact":
        return {
            "id": session_info.get("session_id"),
            "libs": len(session_info.get("libraries", [])),
            "steps": session_info.get("step_count", 0),
        }
    elif verbosity == "standard":
        return {
            "session_id": session_info.get("session_id"),
            "libraries": [
                lib if isinstance(lib, str) else lib.get("name")
                for lib in session_info.get("libraries", [])
            ],
            "step_count": session_info.get("step_count", 0),
            "active": session_info.get("active", True),
        }
    else:  # verbose
        return session_info


# ============================================================================
# Response builder helpers
# ============================================================================


def success_response(
    result: Any = None,
    message: str = None,
    verbosity: str = "standard",
    **extras,
) -> Dict[str, Any]:
    """
    Build a standardized success response.

    Args:
        result: Optional result data
        message: Optional success message
        verbosity: Verbosity level
        **extras: Additional fields to include

    Returns:
        Success response dictionary
    """
    response = {"success": True}

    if result is not None:
        response["result"] = result

    if message and verbosity != "compact":
        response["message"] = message

    # Add extras (filtered for verbosity)
    handler = TokenEfficientOutput(verbosity=verbosity, omit_empty=True)
    for key, value in extras.items():
        if not handler._is_empty(value):
            response[key] = handler._optimize_value(key, value)

    return response


def error_response(
    error: str,
    code: str = None,
    details: Any = None,
    verbosity: str = "standard",
    **extras,
) -> Dict[str, Any]:
    """
    Build a standardized error response.

    Args:
        error: Error message
        code: Optional error code
        details: Optional error details
        verbosity: Verbosity level
        **extras: Additional fields to include

    Returns:
        Error response dictionary
    """
    # Truncate error message if needed
    max_err_len = 300 if verbosity == "compact" else 500
    if len(error) > max_err_len:
        error = error[:max_err_len] + "..."

    response = {"success": False, "error": error}

    if code and verbosity != "compact":
        response["code"] = code

    if details and verbosity == "verbose":
        response["details"] = details

    # Add extras (filtered for verbosity)
    handler = TokenEfficientOutput(verbosity=verbosity, omit_empty=True)
    for key, value in extras.items():
        if not handler._is_empty(value):
            response[key] = handler._optimize_value(key, value)

    return response


# ============================================================================
# Decorators for automatic output optimization
# ============================================================================


def with_optimized_output(verbosity: str = "standard"):
    """
    Decorator to automatically optimize function output.

    Args:
        verbosity: Verbosity level for optimization

    Usage:
        @with_optimized_output("compact")
        async def my_tool():
            return {"success": True, "data": large_data}
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            if isinstance(result, dict):
                return optimize_output(result, verbosity=verbosity)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                return optimize_output(result, verbosity=verbosity)
            return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================================================
# Token estimation utilities
# ============================================================================


def estimate_tokens(obj: Any) -> int:
    """
    Estimate the number of tokens in a JSON-serializable object.

    This provides a rough estimate based on string lengths and structure.
    Actual token counts may vary by model.

    Args:
        obj: Object to estimate tokens for

    Returns:
        Estimated token count
    """
    import json

    try:
        json_str = json.dumps(obj, default=str)
        # Rough estimate: ~4 characters per token for JSON
        return len(json_str) // 4
    except Exception:
        return len(str(obj)) // 4


def token_budget_check(obj: Any, max_tokens: int = 4000) -> Dict[str, Any]:
    """
    Check if an object fits within a token budget.

    Args:
        obj: Object to check
        max_tokens: Maximum allowed tokens

    Returns:
        Dictionary with budget info and optimization suggestions
    """
    estimated = estimate_tokens(obj)
    within_budget = estimated <= max_tokens

    result = {
        "estimated_tokens": estimated,
        "max_tokens": max_tokens,
        "within_budget": within_budget,
    }

    if not within_budget:
        result["suggestion"] = (
            f"Response exceeds budget by ~{estimated - max_tokens} tokens. "
            "Consider using 'compact' verbosity or truncating large fields."
        )

    return result
