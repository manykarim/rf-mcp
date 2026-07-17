"""AUT process lineage (change: desktop-aut-process-lineage).

Run 5 (2026-06-12): bare pid equality flagged every legitimate click —
the agent launched LibreOffice via a bash wrapper (captured pid = bash,
target = soffice re-parented to init after oosplash forked), and the
reopen was served by the original soffice (single-instance handoff).
Lineage = pid identity OR ancestor chain OR session-id match; warn only
on a confirmed foreign process.
"""

from __future__ import annotations

import os

import pytest

from robotmcp.components.execution.platynui_focus import pid_in_aut_lineage


def _tree(parents: dict, sids: dict):
    """Fake /proc readers: parents maps pid->ppid, sids maps pid->sid."""
    return {
        "_ppid": lambda pid: parents.get(pid),
        "_sid": lambda pid: sids.get(pid),
    }


class TestLineageTiers:
    def test_identical_pid(self):
        assert pid_in_aut_lineage(100, 100, None, **_tree({}, {})) is True

    def test_wrapper_child_in_scope(self):
        # bash(1000) -> oosplash(1001) -> soffice.bin(1002); launcher alive.
        readers = _tree(
            {1002: 1001, 1001: 1000, 1000: 500}, {1002: 50, 1000: 50}
        )
        assert pid_in_aut_lineage(1002, 1000, 50, **readers) is True

    def test_daemonized_reparented_same_sid_in_scope(self):
        # Run-5 shape: soffice(992769) re-parented to init; bash(992737)
        # exited. Ancestor walk dead-ends at 1; sid matches.
        readers = _tree({992769: 1}, {992769: 4242})
        assert pid_in_aut_lineage(992769, 992737, 4242, **readers) is True

    def test_single_instance_handoff_in_scope(self):
        # New launcher pid (996321) recorded, but the target is the ORIGINAL
        # soffice (992769) — same server session id.
        readers = _tree({992769: 1}, {992769: 4242})
        assert pid_in_aut_lineage(992769, 996321, 4242, **readers) is True

    def test_confirmed_foreign_process(self):
        # Both signals resolve: different sid, no ancestor relation.
        readers = _tree({7777: 1}, {7777: 9000})
        assert pid_in_aut_lineage(7777, 1000, 4242, **readers) is False

    def test_dead_launcher_unreadable_target_indeterminate(self):
        readers = _tree({}, {})  # every read fails
        assert pid_in_aut_lineage(7777, 1000, 4242, **readers) is None

    def test_no_recorded_sid_indeterminate(self):
        # Target readable but no aut_sid was captured: tier 2 alone can
        # never confirm foreignness (the launcher may simply be dead).
        readers = _tree({7777: 1}, {7777: 9000})
        assert pid_in_aut_lineage(7777, 1000, None, **readers) is None

    def test_ancestor_hop_limit(self):
        # 20-deep chain that WOULD reach the aut beyond the 15-hop bound:
        # must not loop forever; sid mismatch then confirms foreign.
        parents = {i: i - 1 for i in range(2000, 2020)}
        parents[2000] = 1
        readers = _tree(parents, {2019: 9000})
        assert pid_in_aut_lineage(2019, 1999, 4242, **readers) is False

    def test_ancestor_cycle_terminates(self):
        # Defensive: corrupt ppid data forming a cycle must terminate via
        # the hop bound, then fall through to the sid tier.
        readers = _tree({10: 11, 11: 10}, {10: 4242})
        assert pid_in_aut_lineage(10, 99, 4242, **readers) is True


class TestLiveProc:
    def test_own_process_lineage_via_real_proc(self):
        # Live smoke: this test process is in our own session's lineage.
        me = os.getpid()
        sid = os.getsid(me)
        parent = os.getppid()
        assert pid_in_aut_lineage(me, parent, None) is True  # ancestor tier
        assert pid_in_aut_lineage(me, 999999, sid) is True  # sid tier

    def test_live_wrapper_child(self):
        # Spawn a wrapper that spawns a sleeper; the sleeper must be in the
        # wrapper's lineage via real /proc readers.
        import subprocess
        import time

        wrapper = subprocess.Popen(
            ["/bin/bash", "-c", "sleep 3 & echo $!; wait"],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            child_pid = int(wrapper.stdout.readline().strip())
            time.sleep(0.1)
            assert pid_in_aut_lineage(child_pid, wrapper.pid, None) is True
        finally:
            wrapper.kill()
            wrapper.wait()
