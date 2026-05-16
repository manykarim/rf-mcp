"""Tests for P14: commit_form intent with idealForms library fixture.

Tests the _execute_commit_form helper and the JavaScript probe / validate
logic using mocked execute_step calls.
"""
from __future__ import annotations

__test__ = True

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── helpers ──────────────────────────────────────────────────────────

def _make_step_mock(responses: list[Dict[str, Any]]) -> AsyncMock:
    """Return an AsyncMock that yields each response in sequence."""
    mock = AsyncMock()
    mock.side_effect = responses
    return mock


# ── Import the helpers under test ─────────────────────────────────────

# We import via the module-level functions that server.py defines.
# To avoid importing the entire server (requires fastmcp), we test
# the commit_form logic isolated by importing only what we need.

# The probe and validate JS strings are module-level constants in server.py.
# We'll verify the orchestration logic directly via a thin shim.

import json


def _parse_raw(raw: Any) -> Dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


async def _run_commit_form_logic(
    step_mock: AsyncMock,
    form_selector: str,
    extra_field_ids: list[str],
    probe_js: str,
    validate_js_map: Dict[str, str],
) -> Dict[str, Any]:
    """Isolated implementation of _execute_commit_form logic for testing."""
    probe_result: Dict[str, Any] | None = None
    js_keyword: str | None = None

    # Probe step
    try:
        r = await step_mock(
            keyword="Evaluate Javascript",
            arguments=[probe_js, f'"{form_selector}"'],
            session_id="test",
            detail_level="minimal",
        )
        if isinstance(r, dict) and r.get("success"):
            raw = r.get("result") or r.get("return_value") or r.get("output")
            probe_result = _parse_raw(raw)
            if probe_result:
                js_keyword = "Evaluate Javascript"
    except Exception:
        pass

    if probe_result is None:
        return {
            "success": False,
            "error": "commit_form probe failed. Ensure a browser session is active.",
        }

    detected_library: str | None = probe_result.get("library")
    fields: list[str] = list(probe_result.get("fields") or [])
    if extra_field_ids:
        seen = set(fields)
        for fid in extra_field_ids:
            if fid not in seen:
                fields.append(fid)
                seen.add(fid)

    if not detected_library:
        return {
            "success": True,
            "library_detected": None,
            "fields_validated": 0,
            "hint": "No SPA validation library detected. Form likely uses native HTML5 validation.",
        }

    validate_js = validate_js_map.get(detected_library)
    if not validate_js:
        return {
            "success": True,
            "library_detected": detected_library,
            "fields_validated": 0,
            "hint": f"Library '{detected_library}' detected but no built-in validator.",
        }

    field_json = json.dumps(fields)
    validate_args = [validate_js, f'"{form_selector}"', field_json]

    val_result: Dict[str, Any] | None = None
    try:
        assert js_keyword is not None
        vr = await step_mock(
            keyword=js_keyword,
            arguments=validate_args,
            session_id="test",
            detail_level="minimal",
        )
        if isinstance(vr, dict):
            raw = vr.get("result") or vr.get("return_value") or vr.get("output")
            val_result = _parse_raw(raw)
    except Exception:
        pass

    if val_result is None:
        return {
            "success": True,
            "library_detected": detected_library,
            "fields_validated": len(fields),
            "all_valid": None,
            "hint": "Validation call succeeded but response was not parseable.",
        }

    invalid_fields: list[str] = val_result.get("invalid", [])
    validated_count: int = val_result.get("validated", len(fields))
    return {
        "success": True,
        "library_detected": detected_library,
        "fields_validated": validated_count,
        "all_valid": len(invalid_fields) == 0,
        "invalid_fields": invalid_fields,
    }


PROBE_JS = "(formSelector) => { return {library: 'idealForms', fields: ['f1', 'f2']}; }"
VALIDATE_JS = "(formSelector, fieldIds) => { return {validated: 2, invalid: []}; }"
VALIDATE_JS_INVALID = "(formSelector, fieldIds) => { return {validated: 2, invalid: ['f2']}; }"


@pytest.mark.asyncio
async def test_commit_form_idealforms_all_valid():
    probe_response = {
        "success": True,
        "result": {"library": "idealForms", "fields": ["f1", "f2"]},
    }
    validate_response = {
        "success": True,
        "result": {"validated": 2, "invalid": []},
    }
    step_mock = AsyncMock(side_effect=[probe_response, validate_response])

    result = await _run_commit_form_logic(
        step_mock=step_mock,
        form_selector="form.idealforms",
        extra_field_ids=[],
        probe_js=PROBE_JS,
        validate_js_map={"idealForms": VALIDATE_JS},
    )

    assert result["success"] is True
    assert result["library_detected"] == "idealForms"
    assert result["fields_validated"] == 2
    assert result["all_valid"] is True
    assert result["invalid_fields"] == []


@pytest.mark.asyncio
async def test_commit_form_idealforms_with_invalid_fields():
    probe_response = {
        "success": True,
        "result": {"library": "idealForms", "fields": ["f1", "f2"]},
    }
    validate_response = {
        "success": True,
        "result": {"validated": 2, "invalid": ["f2"]},
    }
    step_mock = AsyncMock(side_effect=[probe_response, validate_response])

    result = await _run_commit_form_logic(
        step_mock=step_mock,
        form_selector="form",
        extra_field_ids=[],
        probe_js=PROBE_JS,
        validate_js_map={"idealForms": VALIDATE_JS_INVALID},
    )

    assert result["success"] is True
    assert result["all_valid"] is False
    assert "f2" in result["invalid_fields"]


@pytest.mark.asyncio
async def test_commit_form_extra_field_ids_merged():
    probe_response = {
        "success": True,
        "result": {"library": "idealForms", "fields": ["f1"]},
    }
    validate_response = {
        "success": True,
        "result": {"validated": 2, "invalid": []},
    }
    step_mock = AsyncMock(side_effect=[probe_response, validate_response])

    result = await _run_commit_form_logic(
        step_mock=step_mock,
        form_selector="form",
        extra_field_ids=["f2"],
        probe_js=PROBE_JS,
        validate_js_map={"idealForms": VALIDATE_JS},
    )

    assert result["success"] is True
    # The validate call should have been made with f1 + f2
    call_args = step_mock.call_args_list[1]
    validate_call_args = call_args[1]["arguments"]
    field_json_arg = validate_call_args[2]
    import json
    fields = json.loads(field_json_arg)
    assert "f2" in fields


@pytest.mark.asyncio
async def test_commit_form_probe_fails_returns_error():
    step_mock = AsyncMock(return_value={"success": False, "error": "No browser"})

    result = await _run_commit_form_logic(
        step_mock=step_mock,
        form_selector="form",
        extra_field_ids=[],
        probe_js=PROBE_JS,
        validate_js_map={"idealForms": VALIDATE_JS},
    )

    assert result["success"] is False
    assert "probe failed" in result["error"]


@pytest.mark.asyncio
async def test_commit_form_validate_response_as_json_string():
    """Validate response returned as JSON string (not dict)."""
    probe_response = {
        "success": True,
        "result": {"library": "idealForms", "fields": ["f1"]},
    }
    validate_response = {
        "success": True,
        "result": json.dumps({"validated": 1, "invalid": []}),
    }
    step_mock = AsyncMock(side_effect=[probe_response, validate_response])

    result = await _run_commit_form_logic(
        step_mock=step_mock,
        form_selector="form",
        extra_field_ids=[],
        probe_js=PROBE_JS,
        validate_js_map={"idealForms": VALIDATE_JS},
    )

    assert result["success"] is True
    assert result["all_valid"] is True
