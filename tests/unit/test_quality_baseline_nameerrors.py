"""Regression tests for the three NameError defects found by the Quality job
(change: quality-baseline-cleanup).

All three sit on ERROR or DEGRADED paths, which is why 7,300 passing tests never
reached them:

  suite_execution_service.py:28   runs only if Robot Framework is UNIMPORTABLE
  aggregates.py:150               runs only on the snapshot COMPRESSION path
  rf_native_context_manager:975   runs only on the BuiltIn FALLBACK

Each test below fails against the unfixed code. That is the point: a regression test
for a NameError that passes before the fix is testing nothing.

Run: uv run pytest tests/unit/test_quality_baseline_nameerrors.py -q
"""
from __future__ import annotations

import builtins
import importlib
import re
import sys

import pytest


# =============================================================================
# 1. suite_execution_service - the ImportError handler used `logger` before it existed
# =============================================================================


def test_import_failure_handler_warns_instead_of_raising(monkeypatch):
    """The handler that reports "Robot Framework not available" must be able to run.

    It called `logger.warning(...)` while `logger` was assigned four lines LATER, so a
    missing Robot Framework turned a degraded-mode warning into
    `NameError: name 'logger' is not defined` at import time - a worse failure than the
    one being reported.
    """
    real_import = builtins.__import__

    blocked = {"robot", "robot.api", "robot.running", "robot.parsing",
               "robot.libraries.BuiltIn"}

    def fake_import(name, *args, **kwargs):
        # Only the names the guarded try-block imports. Blocking every `robot.*`
        # would also break imports that are NOT inside the try, so the ImportError
        # would escape the module instead of driving its except branch - testing the
        # fake rather than the fix.
        if name in blocked:
            raise ImportError("simulated: Robot Framework unavailable")
        return real_import(name, *args, **kwargs)

    mod_name = "robotmcp.components.execution.suite_execution_service"

    # Warm the whole transitive dependency graph BEFORE patching, then evict only the
    # module under test. Sibling modules pulled in transitively (e.g.
    # robotmcp/attach/mcp_attach.py) import BuiltIn with no try/except; if their bodies
    # ran under the patch the simulated ImportError would surface from a frame that has
    # no handler and the test would be exercising the fake, not the fix. Already-imported
    # modules do not re-execute, so after this only suite_execution_service's own body
    # runs against fake_import.
    saved = importlib.import_module(mod_name)
    sys.modules.pop(mod_name, None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        # Against the unfixed code this raises NameError rather than importing.
        mod = importlib.import_module(mod_name)
        assert mod.ROBOT_AVAILABLE is False, (
            "the simulated ImportError should have driven the except branch"
        )
    finally:
        monkeypatch.undo()
        # Restore the real module so the degraded copy (ROBOT_AVAILABLE=False) cannot
        # leak into any later test in the session.
        #
        # Restoring sys.modules alone is NOT enough, and getting this wrong is silent:
        # `importlib.import_module` ALSO rebinds the attribute on the parent package, and
        # `from robotmcp.components.execution import suite_execution_service` resolves
        # through that attribute, while `from ...suite_execution_service import X`
        # resolves through sys.modules. Leave the two disagreeing and a later test
        # patches one module object while the code under test reads the other - which is
        # exactly what happened to test_subprocess_stdin_isolation.py's
        # test_dry_run_timeout_reaps_process_tree: it patched `_kill_process_tree` on the
        # degraded copy, the real one ran unpatched, and the reap assertion failed on
        # three CI matrix cells while every file passed in isolation.
        sys.modules[mod_name] = saved
        parent_name, _, child_name = mod_name.rpartition(".")
        setattr(sys.modules[parent_name], child_name, saved)

    # Guard the restore itself. Without this the leak is invisible here and only shows up
    # as an unrelated test failing somewhere later in the session.
    from robotmcp.components.execution import suite_execution_service as via_parent

    assert via_parent is sys.modules[mod_name], (
        "the parent-package attribute and sys.modules must point at the SAME module "
        "object, or a later test will patch one and exercise the other"
    )
    assert via_parent.ROBOT_AVAILABLE is True, (
        "the degraded ROBOT_AVAILABLE=False copy must not survive this test"
    )


# =============================================================================
# 2. aggregates - a name referenced inside the expression that binds it
# =============================================================================


def _snapshot_with_repetitive_list():
    """Build a tree via the real kernel API (AriaNode has no `.create`; that belongs
    to the OTHER AriaNode in domains/snapshot/models.py - aggregates.py uses the
    shared-kernel one re-exported through entities.py)."""
    from robotmcp.domains.shared.kernel import AriaNode, AriaRole, ElementRef
    from robotmcp.domains.snapshot.aggregates import PageSnapshot
    from robotmcp.domains.snapshot.entities import AriaTree
    from robotmcp.domains.snapshot.value_objects import SnapshotId

    def node(index, role, name=None, children=None):
        return AriaNode(
            ref=ElementRef.from_index(index),
            role=AriaRole(role),
            name=name,
            children=children or [],
        )

    # Near-identical siblings so the SimHash similarity check actually folds.
    items = [node(i, "listitem", f"Item {i}") for i in range(3, 23)]
    root = node(1, "document", "page", [node(2, "list", "results", items)])
    return PageSnapshot(
        snapshot_id=SnapshotId.generate(),
        session_id="quality-baseline",
        aria_tree=AriaTree(root=root),
    )


def test_fold_lists_does_not_reference_its_own_result():
    """`fold_lists` built its result while referencing the not-yet-bound name.

        new_snapshot = PageSnapshot(
            ...
            token_estimate_after=new_snapshot._estimate_tokens_for_tree(new_tree),
        )

    so the compression path raised NameError. The fix must PRESERVE the estimate, not
    drop the field to silence the linter - hence the assertion on the value below.
    """
    snapshot = _snapshot_with_repetitive_list()

    folded = snapshot.fold_lists()

    assert folded is not None
    stats = folded.compression_stats
    assert stats.token_estimate_before > 0, "before-estimate should be populated"
    assert stats.token_estimate_after > 0, (
        "after-estimate must still be computed - dropping the field would also make "
        "the NameError go away, and would be wrong"
    )
    assert stats.folded_lists >= 1, (
        "the list must actually fold - otherwise this test never reaches the "
        "compression code it is supposed to be guarding"
    )


def test_folded_summary_reports_a_usable_ref_range():
    """Second defect on the same path, found by the test above once the NameError no
    longer masked it: `_create_folded_list` read `ElementRef.index`, which does not
    exist (the type has `.value` and `to_index()`), so folding raised AttributeError.

    The old f-string also prefixed a literal "e" onto a value that already starts with
    one, so a naive `.to_index` swap would have produced "ee3". Assert the real shape.
    """
    folded = _snapshot_with_repetitive_list().fold_lists()

    summaries = [
        n for n in folded.aria_tree.root.traverse()
        if n.properties.get("folded")
    ]
    assert summaries, "expected a folded summary node"
    ref_range = summaries[0].properties["folded_refs"]

    assert re.fullmatch(r"e\d+-e\d+", ref_range), (
        f"ref range should read like 'e4-e22', got {ref_range!r}"
    )
    first, last = (int(part[1:]) for part in ref_range.split("-"))
    assert first < last, "range should run low -> high across the folded items"


# =============================================================================
# 3. rf_native_context_manager - a nested helper called from another scope
# =============================================================================


def test_normalize_arg_is_reachable_from_the_builtin_fallback():
    """`_normalize_arg` is used at the BuiltIn fallback but was defined as a nested
    function inside a DIFFERENT function, so the fallback raised NameError - on the
    path taken when execution has already gone wrong.

    Asserting on scope rather than driving a full fallback execution: reaching that
    branch needs a failing primary path inside a live RF context, which would make this
    test an integration test of everything except the defect.
    """
    import robotmcp.components.execution.rf_native_context_manager as rf

    fn = getattr(rf, "_normalize_arg", None)
    assert callable(fn), (
        "_normalize_arg must be resolvable at module scope so the BuiltIn fallback "
        "at rf_native_context_manager.py can call it"
    )

    # Behaviour must match the nested original: Windows drive paths get RF's ${/}
    # separator so RF does not de-escape \t, \r, \b inside the path.
    assert fn(r"C:\temp\report.html") == "C:${/}temp${/}report.html"
    assert fn("/posix/path") == "/posix/path"
    assert fn(123) == 123


def test_no_undefined_names_remain_in_src():
    """The whole point of F821 is that it means something. While annotation-only
    findings sit in the report, real crashes are camouflaged among them."""
    import subprocess
    import sys as _sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [_sys.executable, "-m", "ruff", "check", "src/", "--select", "F821",
         "--output-format", "concise"],
        cwd=repo, capture_output=True, text=True,
    )
    if proc.returncode not in (0, 1):  # ruff missing or misconfigured
        pytest.skip(f"ruff unavailable: {proc.stderr[:120]}")
    findings = [ln for ln in proc.stdout.splitlines() if "F821" in ln]
    assert not findings, "undefined names remain:\n" + "\n".join(findings)
