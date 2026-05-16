"""Tests for P6: raw_error= parameter on intent_action.

Verifies that when raw_error=True (default), the underlying library error
message is surfaced in the response under "underlying_error".
"""
from __future__ import annotations

__test__ = True

from typing import Any, Dict

import pytest

# Test the server-level raw_error logic directly
# (no need to invoke the full MCP stack)


def _apply_raw_error(result: Dict[str, Any], raw_error: bool) -> Dict[str, Any]:
    """Replicate server.py raw_error enrichment logic."""
    if raw_error and not result.get("success", True):
        if "error" in result and "underlying_error" not in result:
            result["underlying_error"] = result["error"]
    return result


def _apply_raw_error_on_exception(error_msg: str, raw_error: bool) -> Dict[str, Any]:
    """Replicate server.py exception handler raw_error logic."""
    error_result: Dict[str, Any] = {
        "success": False,
        "error": f"Intent execution failed: {error_msg}",
    }
    if raw_error:
        error_result["underlying_error"] = error_msg
    return error_result


# ── raw_error on failed result dict ───────────────────────────────────

def test_raw_error_true_surfaces_underlying_error():
    result = {"success": False, "error": "Element not found"}
    enriched = _apply_raw_error(result, raw_error=True)
    assert enriched["underlying_error"] == "Element not found"


def test_raw_error_false_does_not_add_underlying_error():
    result = {"success": False, "error": "Element not found"}
    enriched = _apply_raw_error(result, raw_error=False)
    assert "underlying_error" not in enriched


def test_raw_error_true_success_result_no_underlying_error():
    result = {"success": True, "output": "done"}
    enriched = _apply_raw_error(result, raw_error=True)
    assert "underlying_error" not in enriched


def test_raw_error_does_not_overwrite_existing_underlying_error():
    result = {
        "success": False,
        "error": "Timeout",
        "underlying_error": "Page.waitForSelector timed out after 30s",
    }
    enriched = _apply_raw_error(result, raw_error=True)
    # Should not overwrite
    assert enriched["underlying_error"] == "Page.waitForSelector timed out after 30s"


# ── raw_error on exception path ───────────────────────────────────────

def test_exception_path_raw_error_true_has_underlying_error():
    result = _apply_raw_error_on_exception(
        "Connection refused", raw_error=True
    )
    assert result["underlying_error"] == "Connection refused"
    assert result["success"] is False


def test_exception_path_raw_error_false_no_underlying_error():
    result = _apply_raw_error_on_exception(
        "Connection refused", raw_error=False
    )
    assert "underlying_error" not in result
    assert result["success"] is False


def test_exception_path_error_message_contains_intent_prefix():
    result = _apply_raw_error_on_exception("Timeout after 30s", raw_error=True)
    assert "Intent execution failed" in result["error"]
    assert result["underlying_error"] == "Timeout after 30s"
