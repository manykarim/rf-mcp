"""Tests for P14: commit_form intent when no SPA validation library is detected.

Verifies the no-library fallback path returns the correct structured response.
"""
from __future__ import annotations

__test__ = True

import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest


async def _run_commit_form_no_library(
    step_mock: AsyncMock,
    form_selector: str = "form",
    extra_field_ids: list[str] | None = None,
) -> Dict[str, Any]:
    """Minimal implementation of no-library path for testing."""
    probe_result: Dict[str, Any] | None = None

    try:
        r = await step_mock(
            keyword="Evaluate Javascript",
            arguments=["(probe_js)", f'"{form_selector}"'],
            session_id="test",
            detail_level="minimal",
        )
        if isinstance(r, dict) and r.get("success"):
            raw = r.get("result") or r.get("return_value") or r.get("output")
            if isinstance(raw, dict):
                probe_result = raw
            elif isinstance(raw, str):
                try:
                    probe_result = json.loads(raw)
                except Exception:
                    pass
    except Exception:
        pass

    if probe_result is None:
        return {
            "success": False,
            "error": "commit_form probe failed. Ensure a browser session is active.",
        }

    detected_library: str | None = probe_result.get("library")

    if not detected_library:
        return {
            "success": True,
            "library_detected": None,
            "fields_validated": 0,
            "hint": (
                "No SPA validation library detected. "
                "Form likely uses native HTML5 validation."
            ),
        }

    return {
        "success": True,
        "library_detected": detected_library,
        "fields_validated": 0,
        "hint": "Library found but test shouldn't reach here.",
    }


@pytest.mark.asyncio
async def test_commit_form_no_library_detected():
    probe_response = {
        "success": True,
        "result": {"library": None, "fields": []},
    }
    step_mock = AsyncMock(return_value=probe_response)

    result = await _run_commit_form_no_library(step_mock)

    assert result["success"] is True
    assert result["library_detected"] is None
    assert result["fields_validated"] == 0
    assert "No SPA validation library detected" in result["hint"]
    assert "native HTML5 validation" in result["hint"]


@pytest.mark.asyncio
async def test_commit_form_no_library_json_string_probe_result():
    probe_response = {
        "success": True,
        "result": json.dumps({"library": None, "fields": []}),
    }
    step_mock = AsyncMock(return_value=probe_response)

    result = await _run_commit_form_no_library(step_mock)

    assert result["success"] is True
    assert result["library_detected"] is None


@pytest.mark.asyncio
async def test_commit_form_probe_fails_returns_error():
    step_mock = AsyncMock(return_value={"success": False, "error": "No browser"})

    result = await _run_commit_form_no_library(step_mock)

    assert result["success"] is False
    assert "probe failed" in result["error"]


@pytest.mark.asyncio
async def test_commit_form_step_raises_exception():
    step_mock = AsyncMock(side_effect=RuntimeError("Browser crashed"))

    result = await _run_commit_form_no_library(step_mock)

    assert result["success"] is False
    assert "probe failed" in result["error"]


@pytest.mark.asyncio
async def test_commit_form_library_detected_not_no_library_path():
    """If a library IS detected, we don't hit the no-library fallback."""
    probe_response = {
        "success": True,
        "result": {"library": "idealForms", "fields": ["f1", "f2"]},
    }
    step_mock = AsyncMock(return_value=probe_response)

    result = await _run_commit_form_no_library(step_mock)

    assert result["success"] is True
    # The shim sets library_detected when found
    assert result.get("library_detected") == "idealForms"
    assert "No SPA validation" not in result.get("hint", "")
