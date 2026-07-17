#!/usr/bin/env python3
"""Deterministic no-LLM acceptance gates for the desktop-steering fixes,
run against REAL AT-SPI inside the docker desktop harness.

  G2  desktop-launch-env-generalization (R5/R6): the launch-env overlay makes a
      NON-allowlisted GTK app (mousepad) and LibreOffice come up with a NON-EMPTY
      AT-SPI object tree, with GTK_A11Y unset in the parent env — proving the
      overlay (not a global image env var) is what enables accessibility.
  G3  desktop-steering-confidence-gate (R3): a keystroke that LANDS yields
      `confirmed`; a keystroke sent while a DIFFERENT window is focused leaves the
      target's CharacterCount unchanged and yields `contradicted`.
  G6  desktop-aware-batch-execution §4 (discovery): measure the native
      bad-descriptor resolution time and locate the real PlatynUI timeout knob so
      the retry-timeout cap targets the correct attribute.

Usage: gate_drivers.py <g2|g3|g6|all>   (PYTHONPATH must include /app/src)
Exit 0 = PASS.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ART = os.environ.get("ROBOTMCP_SCREENSHOT_DIR", "/artifacts")

import platynui_native as pn  # noqa: E402

rt = pn.Runtime()


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _esc(name: str) -> str:
    return (name or "").replace("'", "&apos;")


def resolve_by_pid(pid: int, timeout_s: float = 40.0, name_substrings=()):
    """Return the /app:* node whose ProcessId == pid, or (fallback) whose name
    contains one of ``name_substrings`` — LibreOffice forks soffice.bin so its
    AT-SPI ProcessId differs from the launcher pid. Else None."""
    subs = [s.lower() for s in name_substrings]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        _try(rt.clear_cache)
        for a in _try(lambda: rt.evaluate("/app:*"), []) or []:
            apid = _try(lambda: a.attribute("ProcessId"))
            try:
                if apid is not None and int(apid) == pid:
                    return a
            except Exception:
                pass
            nm = (getattr(a, "name", "") or "").lower()
            if subs and any(s in nm for s in subs):
                return a
        time.sleep(0.5)
    return None


def subtree_control_count(app) -> int:
    name = _esc(app.name)
    nodes = _try(lambda: list(rt.evaluate(f"/app:*[@Name='{name}']//control:*")), None)
    return len(nodes) if nodes is not None else -1


def _read_cc(node):
    # The working read (docker probe 2026-07-17): the bare string accessor
    # returns None; enumerate attributes() and call .value() on the match.
    for a in _try(lambda: list(node.attributes()), []) or []:
        if getattr(a, "name", None) == "Text.CharacterCount":
            return _try(lambda: int(a.value()))
    v = _try(lambda: node.attribute("native:Text.CharacterCount"))
    if v is not None:
        return _try(lambda: int(v.value() if hasattr(v, "value") else v))
    return None


def char_count(app, debug=None):
    """native:Text.CharacterCount for the app's text node, or None. First tries
    a DIRECT descriptor resolution of the Text node (the RF Get Attribute path
    the product uses), then falls back to a descendant scan."""
    name = _esc(app.name)
    # Direct resolution mirrors the product's evaluate_single(descriptor) path.
    for xp in (
        f"/app:*[@Name='{name}']//control:Text",
        f"/app:*[@Name='{name}']//Text",
        f"/app:*[@Name='{name}']//control:Document",
    ):
        node = _try(lambda x=xp: rt.evaluate_single(x))
        if node is not None:
            iv = _read_cc(node)
            if iv is not None:
                if debug is not None:
                    debug["hit_xpath"] = xp
                return iv
            if debug is not None and "text_node_attrs" not in debug:
                debug["text_node_xpath"] = xp
                debug["text_node_dir"] = [
                    a for a in (_try(lambda: dir(node), []) or [])
                    if not a.startswith("__")
                ]
                debug["text_node_attributes"] = _try(
                    lambda: [str(a) for a in node.attributes()][:40]
                ) or _try(lambda: [str(a) for a in node.attribute_names()][:40])
    roles = []
    for xp in (f"/app:*[@Name='{name}']//*", f"/app:*[@Name='{name}']//control:*"):
        for n in _try(lambda x=xp: list(rt.evaluate(x)), []) or []:
            roles.append(getattr(n, "role", "?"))
            for attr in ("native:Text.CharacterCount", "Text.CharacterCount", "CharacterCount"):
                v = _try(lambda a=attr: n.attribute(a))
                if v is not None:
                    iv = _try(lambda: int(v.value() if hasattr(v, "value") else v))
                    if iv is not None:
                        if debug is not None:
                            debug["hit_role"] = getattr(n, "role", "?")
                            debug["hit_attr"] = attr
                        return iv
    if debug is not None:
        debug["roles_seen"] = sorted(set(roles))[:20]
    return None


def launch(binary, args, *, overlay: bool):
    """Popen `binary args` with (overlay=True) the launch-env accessibility
    overlay applied, GTK_A11Y removed from the parent so the overlay is the ONLY
    source. overlay=False is the negative control."""
    from robotmcp.components.execution.desktop_launch_env import (
        build_desktop_launch_env,
        gui_launch_overrides,
    )

    parent = dict(os.environ)
    parent.pop("GTK_A11Y", None)  # prove the overlay adds it, not the image env
    if overlay:
        ov = gui_launch_overrides(binary, parent_env=parent)
        env = build_desktop_launch_env(binary, parent_env=parent, display_env=ov)
    else:
        env = dict(parent)
        env["DISPLAY"] = os.environ.get("DISPLAY", ":99")
    proc = subprocess.Popen([binary] + list(args), env=env)
    return proc, env


# ── G2 ──────────────────────────────────────────────────────────────────────
def g2():
    r = {"gate": "G2", "apps": {}, "passed": False}
    targets = [
        ("mousepad", [], 40.0, ("mousepad",)),
        ("soffice", ["--writer", "--norestore"], 90.0, ("libre", "writer", "soffice")),
    ]
    ok = True
    procs = []
    for binary, args, tmo, subs in targets:
        entry = {}
        proc, env = launch(binary, args, overlay=True)
        procs.append(proc)
        entry["overlay_gtk_a11y"] = env.get("GTK_A11Y")
        app = resolve_by_pid(proc.pid, tmo, name_substrings=subs)
        entry["resolved"] = app is not None
        entry["subtree_controls"] = subtree_control_count(app) if app else 0
        entry["app_name"] = getattr(app, "name", None)
        app_ok = (
            entry["overlay_gtk_a11y"] == "atspi"
            and entry["resolved"]
            and entry["subtree_controls"] > 0
        )
        entry["pass"] = app_ok
        ok = ok and app_ok
        r["apps"][binary] = entry
    # informational negative control (mousepad, no overlay)
    ncp, _ = launch("mousepad", [], overlay=False)
    procs.append(ncp)
    napp = resolve_by_pid(ncp.pid, 30.0)
    r["negative_control_mousepad_no_overlay"] = {
        "resolved": napp is not None,
        "subtree_controls": subtree_control_count(napp) if napp else 0,
        "note": "informational: GTK may still auto-register a11y when the bus is up",
    }
    for p in procs:
        _try(p.terminate)
    r["passed"] = ok
    return r


# ── G3 ──────────────────────────────────────────────────────────────────────
def g3():
    from robotmcp.components.execution.desktop_execution_signals import (
        steering_confidence,
        SC_CONFIRMED,
        SC_CONTRADICTED,
    )

    r = {"gate": "G3", "passed": False}
    a_proc, _ = launch("mousepad", [], overlay=True)
    b_proc = None
    try:
        app_a = resolve_by_pid(a_proc.pid, 40.0)
        if app_a is None:
            r["error"] = "mousepad A did not resolve"
            return r
        frame_a = _try(lambda: rt.evaluate_single(f"/app:*[@Name='{_esc(app_a.name)}']/*[1]"))
        _try(lambda: rt.bring_to_front(frame_a))
        time.sleep(0.8)

        # (1) landed keystroke -> effect observed -> confirmed
        dbg = {}
        before1 = char_count(app_a, debug=dbg)
        _try(lambda: rt.keyboard_type("hello"))
        time.sleep(0.8)
        after1 = char_count(app_a, debug=dbg)
        r["char_read_debug"] = dbg
        v1 = steering_confidence(
            keyword="Keyboard Type", success=True, verified_focus=True,
            state_before=before1, state_after=after1, wayland_risk=False,
        )
        r["landed"] = {"before": before1, "after": after1, "verdict": v1 and v1["verdict"]}

        # (2) keystroke while a DIFFERENT window is focused -> A unchanged -> contradicted
        b_proc, _ = launch("mousepad", [], overlay=True)
        app_b = resolve_by_pid(b_proc.pid, 40.0)
        frame_b = _try(lambda: rt.evaluate_single(f"/app:*[@Name='{_esc(app_b.name)}']/*[1]")) if app_b else None
        _try(lambda: rt.bring_to_front(frame_b))
        time.sleep(0.8)
        before2 = char_count(app_a)
        _try(lambda: rt.keyboard_type("world"))   # lands in B, not A
        time.sleep(0.8)
        after2 = char_count(app_a)
        v2 = steering_confidence(
            keyword="Keyboard Type", success=True, verified_focus=False,
            state_before=before2, state_after=after2, wayland_risk=False,
        )
        r["misdirected"] = {"a_before": before2, "a_after": after2, "verdict": v2 and v2["verdict"]}

        r["passed"] = (
            v1 is not None and v1["verdict"] == SC_CONFIRMED
            and v2 is not None and v2["verdict"] == SC_CONTRADICTED
            and before2 == after2
        )
    finally:
        _try(a_proc.terminate)
        if b_proc:
            _try(b_proc.terminate)
    return r


# ── G6 ──────────────────────────────────────────────────────────────────────
def g6():
    r = {"gate": "G6", "passed": False, "knob": {}}
    # 1) native bad-descriptor resolution time (the hang the cap must bound)
    t0 = time.time()
    _try(lambda: rt.evaluate_single(
        "/app:*[@Name='zzz-nonexistent-app']//control:Button[@Name='nope']"
    ))
    r["bad_descriptor_resolve_s"] = round(time.time() - t0, 2)

    # 2) locate the real timeout knob. The runtime has none (confirmed); the
    #    descriptor-resolution timeout lives on the RF BareMetal LIBRARY, so
    #    load it via RF's TestLibrary importer and inspect an instance.
    def _scan(obj, label):
        names = _try(lambda: dir(obj), []) or []
        hits = [n for n in names if any(k in n.lower() for k in ("timeout", "setting", "query"))]
        r["knob"][label] = hits

    _scan(rt, "runtime")
    # Instantiate the RF library the way rf-mcp/RF does.
    try:
        from robot.running.testlibraries import TestLibrary

        tl = TestLibrary.from_name("PlatynUI.BareMetal")
        inst = getattr(tl, "instance", None) or getattr(tl, "_libinst", None)
        r["knob"]["baremetal_library_class"] = type(inst).__name__ if inst else None
        _scan(inst, "baremetal_instance")
        qs = getattr(inst, "query_settings", None)
        if qs is not None:
            r["knob"]["query_settings_type"] = type(qs).__name__
            _scan(qs, "query_settings")
            r["knob"]["query_settings_timeout"] = _try(lambda: getattr(qs, "timeout"))
    except Exception as e:
        r["knob"]["baremetal_import_error"] = repr(e)

    # 3) confirm the cap helper + no-op-without-RF-context contract
    from robotmcp.adapters.recovery_adapter import (
        RecoveryServiceAdapter,
        _batch_retry_timeout_cap_seconds,
    )
    r["cap_seconds"] = _batch_retry_timeout_cap_seconds()

    class _KR:
        async def run_keyword(self, *a, **k):
            return None

    class _SM:
        def get_session(self, sid):
            class _S:
                def is_desktop_session(self_inner):
                    return True
            return _S()

    from robotmcp.domains.recovery import RecoveryEngine
    adapter = RecoveryServiceAdapter(
        engine=RecoveryEngine.with_defaults(), keyword_runner=_KR(), session_manager=_SM(),
    )
    r["baremetal_resolved_outside_rf"] = adapter._resolve_baremetal("s") is not None
    with adapter.retry_timeout_cap("s"):
        pass  # must not raise
    r["cap_noop_ok"] = True

    # PASS = we measured the native timeout and the cap machinery is sound.
    r["passed"] = r["bad_descriptor_resolve_s"] >= 0 and r["cap_noop_ok"]
    return r


# ── GUARD: strict isolation guard + XPID corroboration vs real Xvfb/proc ─────
def guard():
    from robotmcp.components.execution import desktop_display_safety as dds

    r = {"gate": "GUARD", "passed": False}
    env = dict(os.environ)
    r["display"] = env.get("DISPLAY")
    r["marker"] = env.get(dds.ISOLATION_MARKER_ENV)
    r["xpid"] = env.get(dds.ISOLATION_XPID_ENV)
    # (1) entrypoint set a real Xvfb XPID -> marker is corroborated -> isolated
    verified = dds.classify_bound_display_detailed(env)
    r["with_xpid"] = verified
    # (2) strip the XPID -> strict fail-closed -> unknown (refused)
    env_no = dict(env)
    env_no.pop(dds.ISOLATION_XPID_ENV, None)
    stripped = dds.classify_bound_display_detailed(env_no)
    r["without_xpid"] = stripped
    r["passed"] = (
        verified.get("isolation") == dds.ISOLATED
        and verified.get("isolation_source") == "marker"
        and stripped.get("isolation") == dds.UNKNOWN
    )
    return r


# ── PROBE: exactly how does node.attribute('native:Text.CharacterCount') read? ─
def probe():
    r = {"gate": "PROBE", "passed": False, "reads": []}
    proc, _ = launch("mousepad", [], overlay=True)
    try:
        app = resolve_by_pid(proc.pid, 40.0, name_substrings=("mousepad",))
        if app is None:
            r["error"] = "no mousepad"
            return r
        frame = _try(lambda: rt.evaluate_single(f"/app:*[@Name='{_esc(app.name)}']/*[1]"))
        _try(lambda: rt.bring_to_front(frame))
        time.sleep(0.6)
        _try(lambda: rt.keyboard_type("hello"))  # -> CharacterCount should be 5
        time.sleep(0.6)
        node = _try(lambda: rt.evaluate_single(f"/app:*[@Name='{_esc(app.name)}']//Text"))
        rec = {"node_role": getattr(node, "role", None)}
        # (a) the exact product call
        att = _try(lambda: node.attribute("native:Text.CharacterCount"))
        rec["attribute_obj_repr"] = repr(att)[:120]
        rec["value_via_method"] = _try(lambda: att.value())
        rec["value_attr"] = _try(lambda: getattr(att, "value"))
        # (b) match from attributes() list and read that
        for a in _try(lambda: list(node.attributes()), []) or []:
            if getattr(a, "name", "") == "Text.CharacterCount":
                rec["from_list_repr"] = repr(a)[:120]
                rec["from_list_value_method"] = _try(lambda: a.value())
                rec["from_list_value_attr"] = _try(lambda: getattr(a, "value"))
                # maybe reading requires the node, not the descriptor
                rec["node_read_attr_names"] = [x for x in dir(a) if not x.startswith("__")][:20]
                break
        # (c) pattern route
        rec["has_text_pattern"] = _try(lambda: node.has_pattern("Text"))
        tp = _try(lambda: node.get_pattern("Text"))
        rec["text_pattern_dir"] = [x for x in (_try(lambda: dir(tp), []) or []) if not x.startswith("__")][:25]
        rec["pattern_char_count"] = _try(lambda: tp.character_count) or _try(lambda: tp.CharacterCount)
        r["reads"].append(rec)
        r["passed"] = True
    finally:
        _try(proc.terminate)
    return r


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    gates = {"guard": guard, "g2": g2, "g3": g3, "g6": g6, "probe": probe}
    todo = list(gates) if which == "all" else [which]
    out = {}
    rc = 0
    for g in todo:
        try:
            res = gates[g]()
        except Exception as e:
            res = {"gate": g.upper(), "passed": False, "error": repr(e)}
        out[g] = res
        if not res.get("passed"):
            rc = 1
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "gate_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    sys.exit(rc)


if __name__ == "__main__":
    main()
