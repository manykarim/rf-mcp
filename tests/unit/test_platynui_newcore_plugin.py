"""Unit tests for the PlatynUI new-core plugin (ADR-025).

Covers:
- PLATYNUI_KEYWORDS surface (24 keywords) and get_keyword_library_map
  including _SHARED_WITH_BROWSER exclusions.
- ensure_x11_session_env() env-shim logic on a plain dict (no global mutation).
- generate_failure_hints() classification.
- on_session_start() firing only when libraries contain platynui.

Run with: uv run pytest tests/unit/test_platynui_newcore_plugin.py -q
"""

__test__ = True

import pytest

import robotmcp.plugins.builtin.platynui_plugin as plugin_mod
from robotmcp.plugins.builtin.platynui_plugin import (
    KEEP_WAYLAND_ENV,
    PLATYNUI_KEYWORDS,
    PlatynUILibraryPlugin,
    _SHARED_WITH_BROWSER,
    ensure_x11_session_env,
)


@pytest.fixture
def plugin():
    return PlatynUILibraryPlugin()


# =============================================================================
# Keyword surface
# =============================================================================


class TestKeywordSurface:
    def test_keyword_count_is_24(self):
        assert len(PLATYNUI_KEYWORDS) == 24

    def test_keywords_are_unique(self):
        assert len(set(PLATYNUI_KEYWORDS)) == len(PLATYNUI_KEYWORDS)

    def test_keywords_are_lowercase(self):
        for kw in PLATYNUI_KEYWORDS:
            assert kw == kw.lower()

    def test_expected_keywords_present(self):
        for kw in (
            "query",
            "pointer click",
            "keyboard type",
            "activate window",
            "bring to front",
            "take screenshot",
            "set root",
        ):
            assert kw in PLATYNUI_KEYWORDS

    def test_shared_with_browser_set(self):
        assert _SHARED_WITH_BROWSER == frozenset(
            {"focus", "get attribute", "take screenshot"}
        )


# =============================================================================
# get_keyword_library_map
# =============================================================================


class TestKeywordLibraryMap:
    def test_prefixed_entries_for_all_keywords(self, plugin):
        mapping = plugin.get_keyword_library_map()
        for kw in PLATYNUI_KEYWORDS:
            assert mapping[f"platynui.baremetal.{kw}"] == "PlatynUI.BareMetal"

    def test_unprefixed_entries_exclude_shared(self, plugin):
        mapping = plugin.get_keyword_library_map()
        for kw in PLATYNUI_KEYWORDS:
            if kw in _SHARED_WITH_BROWSER:
                assert kw not in mapping
            else:
                assert mapping[kw] == "PlatynUI.BareMetal"

    def test_shared_keywords_only_via_prefix(self, plugin):
        mapping = plugin.get_keyword_library_map()
        for kw in _SHARED_WITH_BROWSER:
            assert f"platynui.baremetal.{kw}" in mapping
            assert kw not in mapping

    def test_query_unprefixed_present(self, plugin):
        mapping = plugin.get_keyword_library_map()
        assert mapping["query"] == "PlatynUI.BareMetal"
        assert mapping["pointer click"] == "PlatynUI.BareMetal"


# =============================================================================
# ensure_x11_session_env — env-shim logic on a plain dict
# =============================================================================


class TestEnsureX11SessionEnv:
    def _linux(self, monkeypatch):
        monkeypatch.setattr(plugin_mod.sys, "platform", "linux")

    def test_wayland_with_display_forces_x11_and_returns_note(self, monkeypatch):
        self._linux(monkeypatch)
        env = {"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"}
        note = ensure_x11_session_env(env)
        assert note is not None
        assert "x11" in note.lower()
        assert env["XDG_SESSION_TYPE"] == "x11"

    def test_unset_session_type_with_wayland_display_forces_x11(self, monkeypatch):
        self._linux(monkeypatch)
        env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
        note = ensure_x11_session_env(env)
        assert note is not None
        assert env["XDG_SESSION_TYPE"] == "x11"

    def test_x11_session_returns_none(self, monkeypatch):
        self._linux(monkeypatch)
        env = {"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"}
        note = ensure_x11_session_env(env)
        assert note is None
        assert env["XDG_SESSION_TYPE"] == "x11"

    def test_no_display_returns_none_and_does_not_mutate(self, monkeypatch):
        self._linux(monkeypatch)
        env = {"XDG_SESSION_TYPE": "wayland"}
        note = ensure_x11_session_env(env)
        assert note is None
        # No DISPLAY -> cannot force X11, must leave session type unchanged
        assert env["XDG_SESSION_TYPE"] == "wayland"

    def test_opt_out_keep_wayland_returns_none(self, monkeypatch):
        self._linux(monkeypatch)
        env = {
            "XDG_SESSION_TYPE": "wayland",
            "DISPLAY": ":0",
            KEEP_WAYLAND_ENV: "1",
        }
        note = ensure_x11_session_env(env)
        assert note is None
        assert env["XDG_SESSION_TYPE"] == "wayland"

    @pytest.mark.parametrize("optval", ["true", "yes"])
    def test_opt_out_accepts_true_yes(self, monkeypatch, optval):
        self._linux(monkeypatch)
        env = {"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0", KEEP_WAYLAND_ENV: optval}
        assert ensure_x11_session_env(env) is None
        assert env["XDG_SESSION_TYPE"] == "wayland"

    def test_non_linux_returns_none(self, monkeypatch):
        monkeypatch.setattr(plugin_mod.sys, "platform", "win32")
        env = {"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"}
        note = ensure_x11_session_env(env)
        assert note is None
        assert env["XDG_SESSION_TYPE"] == "wayland"

    def test_no_display_no_wayland_returns_none_unchanged(self, monkeypatch):
        self._linux(monkeypatch)
        env = {}
        note = ensure_x11_session_env(env)
        assert note is None
        assert env == {}


# =============================================================================
# generate_failure_hints
# =============================================================================


class _StubSession:
    session_id = "s1"


class TestGenerateFailureHints:
    def _hints(self, plugin, keyword, args, error):
        return plugin.generate_failure_hints(_StubSession(), keyword, args, error)

    def test_import_error_yields_matched_set_hint(self, plugin):
        hints = self._hints(
            plugin, "Query", [], "ImportError: cannot import name 'WindowSurface'"
        )
        types = {h["type"] for h in hints}
        assert "platynui_matched_set" in types

    def test_element_not_found_yields_locator_hint(self, plugin):
        hints = self._hints(
            plugin, "Pointer Click", ["/app:*//control:Button"],
            "ElementNotFoundError: no nodes matched",
        )
        locator = [h for h in hints if h["type"] == "platynui_locator"]
        assert locator
        assert "//" in locator[0]["message"]

    def test_element_not_found_with_control_window_prepends_frame_hint(self, plugin):
        hints = self._hints(
            plugin, "Query", ["/app:*//control:Window[@Name='x']"],
            "element not found",
        )
        locator = [h for h in hints if h["type"] == "platynui_locator"]
        assert locator
        # Frame hint must be prepended before the actionability hint
        msg = locator[0]["message"]
        assert "Frame" in msg
        assert "Scope queries" in msg or "Scope queries to the target" in msg
        assert msg.index("Frame") < msg.lower().index("whole desktop")

    def test_element_not_found_without_control_window_omits_frame_hint(self, plugin):
        hints = self._hints(
            plugin, "Query", ["/app:*//control:Button"], "element not found"
        )
        locator = [h for h in hints if h["type"] == "platynui_locator"]
        assert locator
        assert "Frame" not in locator[0]["message"]

    def test_timeout_yields_scope_hint(self, plugin):
        hints = self._hints(plugin, "Query", [], "Operation timed out after 30s")
        types = {h["type"] for h in hints}
        assert "platynui_query_scope" in types

    def test_provider_error_with_mock_yields_mock_hint(self, plugin):
        hints = self._hints(
            plugin, "Query", [],
            "ProviderError: mock provider not available",
        )
        types = {h["type"] for h in hints}
        assert "platynui_mock_provider" in types

    def test_provider_error_without_mock_no_mock_hint(self, plugin):
        hints = self._hints(plugin, "Query", [], "ProviderError: provider failed")
        types = {h["type"] for h in hints}
        assert "platynui_mock_provider" not in types

    def test_clean_error_yields_no_hints(self, plugin):
        hints = self._hints(plugin, "Query", [], "some unrelated message")
        assert hints == []


# =============================================================================
# on_session_start
# =============================================================================


class TestOnSessionStart:
    """The hook checks imported_libraries + search_order +
    explicit_library_preference (NOT a ``libraries`` attribute) because it
    fires at session creation — see ADR-025 implementation notes. The
    deterministic shim trigger lives in the keyword executor and the
    library manager import chokepoint.
    """

    def test_fires_when_platynui_in_imported_libraries(self, plugin, monkeypatch):
        calls = []
        monkeypatch.setattr(
            plugin_mod, "ensure_x11_session_env", lambda *a, **k: calls.append(1)
        )

        class S:
            imported_libraries = ["PlatynUI.BareMetal", "BuiltIn"]
            search_order = []
            explicit_library_preference = ""

        plugin.on_session_start(S())
        assert calls == [1]

    def test_fires_when_platynui_is_explicit_preference(self, plugin, monkeypatch):
        calls = []
        monkeypatch.setattr(
            plugin_mod, "ensure_x11_session_env", lambda *a, **k: calls.append(1)
        )

        class S:
            imported_libraries = ["BuiltIn"]
            search_order = []
            explicit_library_preference = "PlatynUI.BareMetal"

        plugin.on_session_start(S())
        assert calls == [1]

    def test_does_not_fire_without_platynui(self, plugin, monkeypatch):
        calls = []
        monkeypatch.setattr(
            plugin_mod, "ensure_x11_session_env", lambda *a, **k: calls.append(1)
        )

        class S:
            imported_libraries = ["Browser", "BuiltIn"]
            search_order = ["Browser"]
            explicit_library_preference = ""

        plugin.on_session_start(S())
        assert calls == []

    def test_no_libraries_attribute_is_safe(self, plugin, monkeypatch):
        calls = []
        monkeypatch.setattr(
            plugin_mod, "ensure_x11_session_env", lambda *a, **k: calls.append(1)
        )

        class S:
            imported_libraries = None
            search_order = None
            explicit_library_preference = None

        plugin.on_session_start(S())
        assert calls == []
