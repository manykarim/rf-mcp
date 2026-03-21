import os
import uuid

import pytest

from robotmcp.components.execution.rf_native_context_manager import (
    get_rf_native_context_manager,
)


@pytest.fixture(autouse=True)
def _isolate_rf_context():
    """Ensure no stale RF context from prior tests affects these tests.

    The RobotFrameworkNativeContextManager is a process-global singleton.
    Its ``_session_contexts`` dict and RF's ``EXECUTION_CONTEXTS`` can
    carry state between tests when running the full suite.  We save the
    entire singleton state, reset it, and restore after each test.
    """
    from robot.running.context import EXECUTION_CONTEXTS

    mgr = get_rf_native_context_manager()

    # Save and clear singleton + RF globals
    saved_contexts = list(getattr(EXECUTION_CONTEXTS, "_contexts", []))
    saved_current = getattr(EXECUTION_CONTEXTS, "_context", None)
    saved_sessions = dict(mgr._session_contexts)
    saved_active = getattr(mgr, "_active_context", None)

    EXECUTION_CONTEXTS._contexts = []
    EXECUTION_CONTEXTS._context = None
    mgr._session_contexts.clear()
    mgr._active_context = None

    yield

    # Restore original state (best-effort)
    try:
        while EXECUTION_CONTEXTS.current is not None:
            try:
                EXECUTION_CONTEXTS.end_suite()
            except Exception:
                break
        EXECUTION_CONTEXTS._contexts = saved_contexts
        EXECUTION_CONTEXTS._context = saved_current
        mgr._session_contexts.update(saved_sessions)
        mgr._active_context = saved_active
    except Exception:
        pass


def _cleanup_context(session_id: str, context_info: dict | None) -> None:
    from robot.running.context import EXECUTION_CONTEXTS

    mgr = get_rf_native_context_manager()
    if context_info is None:
        context_info = mgr._session_contexts.get(session_id)
    mgr._session_contexts.pop(session_id, None)

    suite_name = None
    if context_info and context_info.get("suite") is not None:
        suite_name = getattr(context_info["suite"], "name", None)
    if suite_name is None:
        suite_name = f"MCP_Suite_{session_id}"

    # Prune execution contexts associated with the suite
    filtered = []
    for ctx in getattr(EXECUTION_CONTEXTS, "_contexts", []):
        if getattr(ctx, "_mcp_session", None) == session_id:
            continue
        ctx_suite = getattr(getattr(ctx, "suite", None), "name", None)
        if ctx_suite != suite_name:
            filtered.append(ctx)
    EXECUTION_CONTEXTS._contexts = filtered
    EXECUTION_CONTEXTS._context = filtered[-1] if filtered else None


def test_browser_library_import_registers_listener():
    pytest.importorskip("Browser")
    mgr = get_rf_native_context_manager()
    session_id = f"test-browser-{uuid.uuid4()}"
    context_info = None
    try:
        result = mgr.create_context_for_session(session_id, libraries=["Browser"])
        assert result["success"], result
        context_info = mgr._session_contexts[session_id]
        namespace = context_info["namespace"]
        libraries = namespace.libraries
        if hasattr(libraries, "keys"):
            names = list(libraries.keys())
        else:
            names = [
                getattr(lib, "name", getattr(getattr(lib, "__class__", None), "__name__", type(lib).__name__))
                for lib in libraries
            ]
        assert any("Browser" in name for name in names)
    finally:
        _cleanup_context(session_id, context_info)


def test_context_sets_standard_output_variables():
    mgr = get_rf_native_context_manager()
    session_id = f"test-output-{uuid.uuid4()}"
    context_info = None
    try:
        result = mgr.create_context_for_session(session_id)
        assert result["success"], result
        context_info = mgr._session_contexts[session_id]
        variables = context_info["variables"]
        output_dir = variables["${OUTPUTDIR}"]
        output_log = variables["${LOGFILE}"]
        output_file = variables["${OUTPUT}"]

        assert output_dir and os.path.isdir(output_dir)
        assert output_log and output_log.startswith(output_dir)
        assert output_file and output_file.startswith(output_dir)
    finally:
        _cleanup_context(session_id, context_info)
