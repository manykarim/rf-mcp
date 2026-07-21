"""Desktop turn-economy guidance (change: desktop-turn-economy-guidance).

Spike levers #1 (deliver the PlatynUI keyword surface + locator crib at init, and
a desktop-focused instruction template) and #5 (Process in the DESKTOP core libs).
"""

from __future__ import annotations

import json

import pytest

import importlib.util as _ilu

from robotmcp.components.execution import desktop_guidance as dg
from robotmcp.domains.instruction.value_objects import InstructionTemplate
from robotmcp.domains.instruction.adapters.fastmcp_adapter import InstructionTemplateType
from robotmcp.models.session_models import ExecutionSession, SessionType

# The guidance bundle is derived live from LibraryDocumentation("PlatynUI.BareMetal");
# without the native library installed get_desktop_guidance() returns None by design.
requires_platynui = pytest.mark.skipif(
    _ilu.find_spec("platynui_native") is None or _ilu.find_spec("PlatynUI") is None,
    reason="PlatynUI (platynui_native) not installed — desktop keyword catalog unavailable",
)


@requires_platynui
class TestDesktopGuidanceBundle:
    def test_bundle_has_full_keyword_surface_and_crib(self):
        b = dg.get_desktop_guidance()
        assert b is not None
        assert b["keyword_count"] == len(b["keywords"]) >= 20
        # Take Screenshot signature shows the descriptor-first trap explicitly
        ts = [s for s in b["keywords"] if s.startswith("Take Screenshot")]
        assert ts and ts[0].startswith("Take Screenshot(descriptor")
        crib = " ".join(b["locator_crib"])
        assert "/app:*[@Name=" in crib
        assert "control:Frame" in crib
        assert "get_locator_guidance" in crib

    def test_bundle_is_bounded(self):
        b = dg.get_desktop_guidance()
        # ~4.5 KB bound. The cheat-sheet is derived from PlatynUI.BareMetal's keyword
        # surface, which grew with the new Rust core (0.13.x: 30 keywords, ~3.8 KB).
        # The bound stays a guard against unbounded growth, with headroom for the core.
        assert len(json.dumps(b)) <= 4608

    def test_bundle_is_process_cached(self):
        # second call returns the identical cached object (no libdoc re-parse)
        assert dg.get_desktop_guidance() is dg.get_desktop_guidance()


class TestDesktopTemplate:
    def test_get_by_name_desktop_focused(self):
        t = InstructionTemplate.get_by_name("desktop-focused")
        assert t.template_id == "desktop-focused"
        assert "PlatynUI" in t.content and "control:Frame" in t.content

    def test_enum_resolves_desktop_focused(self):
        assert InstructionTemplateType.from_string("desktop-focused") is InstructionTemplateType.DESKTOP_FOCUSED

    def test_existing_templates_unaffected(self):
        for name in ("minimal", "standard", "browser-focused", "api-focused"):
            assert InstructionTemplate.get_by_name(name) is not None


class TestDesktopProfileProcessCore:
    def _desktop_session(self):
        s = ExecutionSession(session_id="d")
        s.session_type = SessionType.DESKTOP_TESTING
        return s

    def test_process_is_loaded_for_desktop(self):
        libs = self._desktop_session().get_libraries_to_load()
        assert "Process" in libs
        assert "PlatynUI.BareMetal" in libs

    def test_platynui_leads_search_order(self):
        libs = self._desktop_session().get_libraries_to_load()
        # PlatynUI.BareMetal must remain first (search order derives from core order)
        assert libs[0] == "PlatynUI.BareMetal"
