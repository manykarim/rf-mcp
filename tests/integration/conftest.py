"""Shared fixtures for rf-mcp integration tests."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


def _sid(prefix: str = "test") -> str:
    """Generate a unique session ID with optional prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    """Provide an MCP client connected to the rf-mcp server.

    Yields a Client instance with active connection.
    """
    async with Client(mcp) as client:
        yield client


@pytest.fixture
def session_id():
    """Generate a unique session ID for test isolation."""
    return _sid("integration")


@pytest.fixture
def test_data_dir():
    """Path to the shared test data directory."""
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "test_data"
