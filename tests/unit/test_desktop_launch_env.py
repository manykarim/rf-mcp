"""Unit tests: desktop launch-env sanitization
(change: platynui-desktop-safety-isolation, tasks 1.2/3.4)."""

import os
import sys

import pytest

from robotmcp.components.execution.desktop_launch_env import (
    build_desktop_launch_env,
    is_desktop_gui_launch,
)

SNAP_PARENT = {
    "LD_LIBRARY_PATH": "/snap/core20/current/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu",
    "GTK_PATH": "/snap/code/current/gtk:/usr/lib/gtk",
    "GIO_MODULE_DIR": "/snap/code/current/gio",
    "XDG_DATA_DIRS": "/var/lib/snapd/desktop:/usr/share:/snap/foo/share",
    "LD_PRELOAD": "/snap/core20/current/lib/libfoo.so",
    "FONTCONFIG_FILE": "/etc/fonts/fonts.conf",
    "PATH": "/usr/bin:/snap/bin",
}


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="snap/LD_LIBRARY_PATH stripping uses ':' pathsep — POSIX only",
)
def test_strips_snap_segments_from_path_lists():
    env = build_desktop_launch_env(
        "gnome-calculator", parent_env=SNAP_PARENT,
        display_env={"DISPLAY": ":99"},
    )
    assert env["LD_LIBRARY_PATH"] == "/usr/lib/x86_64-linux-gnu"
    assert "/snap/" not in env.get("GTK_PATH", "")
    assert "/snap/" not in env["XDG_DATA_DIRS"]
    assert "/var/lib/snapd/" not in env["XDG_DATA_DIRS"]
    assert env["XDG_DATA_DIRS"] == "/usr/share"


def test_drops_single_path_var_under_snap():
    env = build_desktop_launch_env(
        "gnome-calculator", parent_env=SNAP_PARENT, display_env={},
    )
    assert "LD_PRELOAD" not in env


def test_preserves_non_snap_vars():
    env = build_desktop_launch_env(
        "gnome-calculator", parent_env=SNAP_PARENT, display_env={},
    )
    assert env["FONTCONFIG_FILE"] == "/etc/fonts/fonts.conf"


def test_does_not_mutate_parent():
    parent = dict(SNAP_PARENT)
    build_desktop_launch_env("gnome-calculator", parent_env=parent, display_env={})
    assert parent["LD_LIBRARY_PATH"].startswith("/snap/")


def test_clean_env_passthrough():
    clean = {"LD_LIBRARY_PATH": "/usr/lib", "PATH": "/usr/bin"}
    env = build_desktop_launch_env(
        "gnome-calculator", parent_env=clean, display_env={"DISPLAY": ":99"},
    )
    assert env["LD_LIBRARY_PATH"] == "/usr/lib"
    assert env["DISPLAY"] == ":99"


def test_display_overlay_unsets_empty():
    env = build_desktop_launch_env(
        "gnome-calculator",
        parent_env={"WAYLAND_DISPLAY": "wayland-0", "PATH": "/usr/bin"},
        display_env={"DISPLAY": ":99", "WAYLAND_DISPLAY": ""},
    )
    assert env["DISPLAY"] == ":99"
    assert "WAYLAND_DISPLAY" not in env


def test_no_sanitize_hatch_keeps_snap_but_overlays_display():
    env = build_desktop_launch_env(
        "gnome-calculator", parent_env=SNAP_PARENT,
        display_env={"DISPLAY": ":99"}, sanitize=False,
    )
    assert env["LD_LIBRARY_PATH"].startswith("/snap/")
    assert env["DISPLAY"] == ":99"


def test_is_desktop_gui_launch():
    assert is_desktop_gui_launch(["gnome-calculator", "--new-window"]) == "gnome-calculator"
    assert is_desktop_gui_launch(["/usr/bin/gnome-text-editor"]) == "gnome-text-editor"
    assert is_desktop_gui_launch(["ls", "-la"]) is None
    assert is_desktop_gui_launch([]) is None


# ── desktop-launch-env-generalization ─────────────────────────────────────
from robotmcp.components.execution.desktop_launch_env import gui_launch_overrides


def _make_exe(tmp_path, name):
    p = tmp_path / name
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return p


def test_generalized_recognizes_nonallowlisted_gui(tmp_path):
    exe = _make_exe(tmp_path, "soffice")
    # Not in the gnome allow-set, but a resolvable binary in a desktop session.
    assert is_desktop_gui_launch([str(exe), "--writer"], is_desktop_session=True) == "soffice"


def test_generalized_denylist_excluded():
    assert is_desktop_gui_launch(["bash", "-c", "echo hi"], is_desktop_session=True) is None
    assert is_desktop_gui_launch(["python3", "s.py"], is_desktop_session=True) is None
    assert is_desktop_gui_launch(["xdotool", "key", "a"], is_desktop_session=True) is None


def test_generalized_requires_desktop_session(tmp_path):
    exe = _make_exe(tmp_path, "someapp")
    # Back-compat: without desktop context, only the allow-set is recognized.
    assert is_desktop_gui_launch([str(exe)]) is None
    assert is_desktop_gui_launch([str(exe)], is_desktop_session=False) is None
    assert is_desktop_gui_launch([str(exe)], is_desktop_session=True) == "someapp"


def test_allowlist_still_fastpath_without_session():
    assert is_desktop_gui_launch(["gnome-calculator"]) == "gnome-calculator"
    assert is_desktop_gui_launch(["/usr/bin/gedit"]) == "gedit"


def test_env_token_first_positional_ignored():
    assert is_desktop_gui_launch(["env:FOO=bar"], is_desktop_session=True) is None
    assert is_desktop_gui_launch(["KEY=val"], is_desktop_session=True) is None


def test_nonexistent_binary_not_recognized():
    assert is_desktop_gui_launch(
        ["/nonexistent/path/to/app-xyz"], is_desktop_session=True
    ) is None
    assert is_desktop_gui_launch(
        ["definitely-not-a-real-binary-xyz"], is_desktop_session=True
    ) is None


def test_gui_launch_overrides_defaults_atspi():
    ov = gui_launch_overrides("soffice", parent_env={})
    assert ov["GTK_A11Y"] == "atspi"
    assert ov["GDK_BACKEND"] == "x11"
    assert ov["QT_QPA_PLATFORM"] == "xcb"
    assert ov["XDG_SESSION_TYPE"] == "x11"


def test_gui_launch_overrides_preserves_explicit_gtk_a11y():
    ov = gui_launch_overrides("soffice", parent_env={"GTK_A11Y": "1"})
    assert ov["GTK_A11Y"] == "1"
    # empty value is treated as unset -> default applies
    ov2 = gui_launch_overrides("soffice", parent_env={"GTK_A11Y": ""})
    assert ov2["GTK_A11Y"] == "atspi"
