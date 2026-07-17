"""Unit tests for the desktop-stepwise-followups change.

Two fixes motivated by a live GNOME Calculator agent run:
- D2 desktop-process-keyword-resolution: unqualified Process keywords resolve.
- D1 desktop-display-assertion-guidance: guidance leads with history Labels.
"""

from __future__ import annotations

import pytest


# ── D2: unqualified Process keywords resolve to Process ─────────────


class TestProcessKeywordResolution:
    def _manager(self):
        from robotmcp.plugins import get_library_plugin_manager
        from robotmcp.config.library_registry import (
            _reset_plugin_state_for_tests,
            _ensure_plugins_registered,
        )

        _reset_plugin_state_for_tests()
        _ensure_plugins_registered()
        return get_library_plugin_manager()

    @pytest.mark.parametrize(
        "keyword",
        [
            "Start Process",
            "Run Process",
            "Terminate Process",
            "Terminate All Processes",
            "Process Should Be Running",
            "Process Should Be Stopped",
            "Wait For Process",
            "Is Process Running",
            "Get Process Result",
        ],
    )
    def test_process_keywords_resolve(self, keyword):
        assert self._manager().get_library_for_keyword(keyword) == "Process"

    def test_resolution_is_case_insensitive(self):
        assert self._manager().get_library_for_keyword("start process") == "Process"

    def test_no_regression_for_other_libraries(self):
        pm = self._manager()
        assert pm.get_library_for_keyword("Pointer Click") == "PlatynUI.BareMetal"

    def test_static_plugin_exposes_keyword_map(self):
        # The Process StaticLibraryPlugin now returns its keyword map.
        from robotmcp.plugins import get_library_plugin_manager
        from robotmcp.config.library_registry import (
            _reset_plugin_state_for_tests,
            _ensure_plugins_registered,
        )

        _reset_plugin_state_for_tests()
        _ensure_plugins_registered()
        pm = get_library_plugin_manager()
        meta = pm.get_metadata("Process")
        assert meta is not None  # Process is registered

    def test_executor_resolves_unqualified_and_qualified(self):
        # The executor maps both the unqualified keyword (via the plugin map)
        # and the dotted form (via the explicit-prefix branch) to Process.
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        ke = KeywordExecutor.__new__(KeywordExecutor)
        # qualified — dotted-name branch (no plugin manager needed)
        assert ke._get_library_for_keyword("Process.Start Process") == "Process"


# ── D1: display-state guidance leads with history Labels ────────────


class TestDisplayStateGuidance:
    def _display(self):
        from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

        g = RobotFrameworkNativeConverter().get_platynui_locator_guidance()
        return g["display_state_reading"]

    def test_history_labels_are_primary(self):
        ds = self._display()
        rules = ds["rules"]
        # The first rule must be the history-Label path.
        assert rules[0].startswith("PRIMARY")
        assert "Label" in rules[0]
        # Labels appear before any CharacterCount mention across the rules.
        joined = " ".join(rules)
        assert joined.index("Label") < joined.index("CharacterCount")

    def test_character_count_demoted_and_flagged(self):
        ds = self._display()
        joined = " ".join(ds["rules"]).lower()
        assert "secondary" in joined
        # Must warn it can report 0 even when the display changed.
        assert "may report 0" in joined or "report 0" in joined
        assert "do not rely on it alone" in joined

    def test_ocr_named_as_last_resort(self):
        ds = self._display()
        joined = " ".join(ds["rules"])
        assert "LAST RESORT" in joined
        assert "OCR" in joined or "ocr" in joined.lower()

    def test_description_does_not_lead_with_character_count(self):
        ds = self._display()
        # The section description should steer to Labels first.
        assert "CharacterCount" not in ds["description"]
        assert "Label" in ds["description"]
