"""Tests for high-level flow/control tools: evaluate_expression, set_variables,
execute_if, execute_for_each, execute_try_except.

Covers multiple scenarios and error conditions.
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


# -------------------------
# evaluate_expression tests
# -------------------------

@pytest.mark.asyncio
async def test_evaluate_expression_success_and_assign(mcp_client):
    session_id = "flow_eval_assign"
    res = await mcp_client.call_tool(
        "evaluate_expression",
        {"session_id": session_id, "expression": "1+2", "assign_to": "SUM"},
    )
    assert res.data.get("success") is True
    # Output is string '3'; result may not be included in minimal step response
    assert str(res.data.get("output")) == "3"

    vars_res = await mcp_client.call_tool(
        "get_context_variables", {"session_id": session_id}
    )
    assert vars_res.data.get("success") is True
    # SUM should be set in Variables
    assert vars_res.data.get("variables", {}).get("SUM") == 3


@pytest.mark.asyncio
async def test_evaluate_expression_error(mcp_client):
    session_id = "flow_eval_error"
    res = await mcp_client.call_tool(
        "evaluate_expression",
        {"session_id": session_id, "expression": "1/0"},
    )
    assert res.data.get("success") is False
    assert "division" in (res.data.get("error", "") or "").lower()


# ----------------
# set_variables
# ----------------

@pytest.mark.asyncio
async def test_set_variables_dict_and_list(mcp_client):
    session_id = "flow_set_vars"
    # Dict
    r1 = await mcp_client.call_tool(
        "set_variables",
        {"session_id": session_id, "variables": {"A": 5, "B": "x"}},
    )
    assert r1.data.get("success") is True
    # List
    r2 = await mcp_client.call_tool(
        "set_variables",
        {"session_id": session_id, "variables": ["C=7", "D=text"]},
    )
    assert r2.data.get("success") is True

    vars_res = await mcp_client.call_tool(
        "get_context_variables", {"session_id": session_id}
    )
    v = vars_res.data.get("variables", {})
    assert v.get("A") == 5
    assert v.get("B") == "x"
    assert v.get("C") == "7" or v.get("C") == 7  # string or parsed
    assert v.get("D") == "text"


# ---------
# execute_if
# ---------

@pytest.mark.asyncio
async def test_execute_if_then_branch(mcp_client):
    session_id = "flow_if_then"
    # set X=1
    await mcp_client.call_tool(
        "set_variables",
        {"session_id": session_id, "variables": {"X": 1}},
    )

    res = await mcp_client.call_tool(
        "execute_if",
        {
            "session_id": session_id,
            "condition": "int(${X}) == 1",
            "then_steps": [{"keyword": "Log", "arguments": ["then"]}],
            "else_steps": [{"keyword": "Log", "arguments": ["else"]}],
        },
    )
    assert res.data.get("success") is True
    assert res.data.get("branch_taken") == "then"
    assert res.data.get("steps") and res.data["steps"][0]["success"] is True


@pytest.mark.asyncio
async def test_execute_if_else_branch_with_failure_stop(mcp_client):
    session_id = "flow_if_else_stop"
    await mcp_client.call_tool(
        "set_variables",
        {"session_id": session_id, "variables": {"X": 2}},
    )

    res = await mcp_client.call_tool(
        "execute_if",
        {
            "session_id": session_id,
            "condition": "int(${X}) == 1",
            "then_steps": [{"keyword": "No Such Keyword"}],
            "else_steps": [
                {"keyword": "No Such Keyword"},
                {"keyword": "Log", "arguments": ["should not run"]},
            ],
            "stop_on_failure": True,
        },
    )
    # else branch with first step failing, so success False and only first step attempted
    assert res.data.get("success") is False
    assert res.data.get("branch_taken") == "else"
    steps = res.data.get("steps", [])
    assert len(steps) == 1
    assert steps[0].get("success") is False


# ---------------
# execute_for_each
# ---------------

@pytest.mark.asyncio
async def test_execute_for_each_success(mcp_client):
    session_id = "flow_for_each_ok"
    res = await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session_id,
            "items": [1, 2, 3],
            "steps": [
                {"keyword": "Evaluate", "arguments": ["int(${item}) * 2"], "assign_to": "TWICE"}
            ],
        },
    )
    assert res.data.get("success") is True
    assert res.data.get("count") == 3
    iters = res.data.get("iterations", [])
    assert len(iters) == 3
    # First iteration step success
    assert iters[0]["steps"][0]["success"] is True


@pytest.mark.asyncio
async def test_execute_for_each_stop_on_failure_false(mcp_client):
    session_id = "flow_for_each_continue"
    res = await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session_id,
            "items": ["a", "b"],
            "steps": [
                {"keyword": "No Such Keyword"},
            ],
            "stop_on_failure": False,
        },
    )
    # Continues through all items despite failures, overall success False
    assert res.data.get("success") is False
    assert res.data.get("count") == 2
    iters = res.data.get("iterations", [])
    assert len(iters) == 2
    assert iters[0]["steps"][0]["success"] is False


@pytest.mark.asyncio
async def test_execute_for_each_max_iterations(mcp_client):
    session_id = "flow_for_each_max"
    res = await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session_id,
            "items": [1, 2, 3, 4, 5],
            "steps": [{"keyword": "Log", "arguments": ["ok"]}],
            "max_iterations": 2,
        },
    )
    assert res.data.get("success") is True
    assert res.data.get("count") == 2
    assert len(res.data.get("iterations", [])) == 2


# -----------------
# execute_try_except
# -----------------

@pytest.mark.asyncio
async def test_execute_try_except_success_path(mcp_client):
    session_id = "flow_try_ok"
    res = await mcp_client.call_tool(
        "execute_try_except",
        {
            "session_id": session_id,
            "try_steps": [{"keyword": "Log", "arguments": ["ok"]}],
        },
    )
    assert res.data.get("success") is True
    assert res.data.get("handled") is False


@pytest.mark.asyncio
async def test_execute_try_except_handled_failure(mcp_client):
    session_id = "flow_try_handled"
    res = await mcp_client.call_tool(
        "execute_try_except",
        {
            "session_id": session_id,
            "try_steps": [{"keyword": "No Such Keyword"}],
            "except_patterns": ["no such keyword"],
            "except_steps": [{"keyword": "Log", "arguments": ["handled"]}],
        },
    )
    assert res.data.get("handled") is True
    assert res.data.get("success") is True
    assert res.data.get("except_results")[0]["success"] is True


@pytest.mark.asyncio
async def test_execute_try_except_unhandled_failure(mcp_client):
    session_id = "flow_try_unhandled"
    res = await mcp_client.call_tool(
        "execute_try_except",
        {
            "session_id": session_id,
            "try_steps": [{"keyword": "No Such Keyword"}],
            "except_patterns": ["different pattern"],
            "rethrow": False,
        },
    )
    assert res.data.get("handled") is False
    assert res.data.get("success") is False
    assert isinstance(res.data.get("error", ""), str)
