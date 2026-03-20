"""Shared fixtures for E2E Copilot CLI tests."""

import time

import pytest

from tests.e2e.copilot_cli_runner import INTER_TEST_DELAY_SECONDS


@pytest.fixture(autouse=True)
def _copilot_inter_test_delay(request):
    """Add a delay after each copilot_cli test to avoid Copilot burst rate limits.

    Only applies to tests marked with ``copilot_cli``.  The delay runs
    *after* the test (in teardown) so it doesn't inflate the first test's
    timing.  Set ``COPILOT_INTER_TEST_DELAY=0`` to disable.
    """
    yield
    if request.node.get_closest_marker("copilot_cli") and INTER_TEST_DELAY_SECONDS > 0:
        time.sleep(INTER_TEST_DELAY_SECONDS)


def skip_if_rate_limited(result) -> None:
    """Call after ``run_copilot_cli`` to skip on rate-limit or auth errors.

    Usage::

        result = run_copilot_cli(...)
        skip_if_rate_limited(result)
    """
    if result.auth_error:
        pytest.skip(f"Copilot auth failed: {result.auth_error_message[:120]}")
    if result.rate_limited:
        pytest.skip(f"Copilot rate limited: {result.rate_limit_message[:120]}")
