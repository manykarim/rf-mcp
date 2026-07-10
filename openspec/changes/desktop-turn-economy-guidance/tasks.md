# Tasks: desktop-turn-economy-guidance

## 1. Desktop init guidance bundle (spike lever #1)
- [ ] 1.1 New builder module (e.g. `components/execution/desktop_guidance.py`): derive the 24-keyword cheat-sheet from `LibraryDocumentation("PlatynUI.BareMetal")` (name + one-line signature preserving arg order, compressed defaults), cache process-wide; define the locator crib text (app-scoping, no unscoped `//`, `Set Root`, Frame-not-Window on Linux, launch-before-query, `Get Attribute` read-back recipe, `Take Screenshot` arg order, pointer to `get_locator_guidance`). Keep the crib aligned with `get_platynui_locator_guidance` (`rf_native_type_converter.py:1682` ff.) — one authoritative rule set.
- [ ] 1.2 `server.py` init handler (result dict at `server.py:3178-3188`): attach `desktop_guidance` when the session resolves to `DESKTOP_TESTING` or `PlatynUI.BareMetal`/`PlatynUI` is among requested/loaded libraries. Soft-fail: a libdoc error must not fail init (omit the bundle, log debug).
- [ ] 1.3 Bound check: assert bundle ≤ ~3 KB in tests; no bundle on web/API/unknown sessions.

## 2. Desktop-focused instruction template (spike lever #1)
- [ ] 2.1 `instruction/value_objects.py`: add `desktop_focused()` factory (workflow init → `Start Process` → `Query` `control:Frame` → pointer/keyboard act → `Get Attribute` read-back; locator rules as in the crib; ~1000 chars like `browser_focused`); register `"desktop-focused"` in `get_by_name`.
- [ ] 2.2 `instruction/adapters/fastmcp_adapter.py:26-36`: add `DESKTOP_FOCUSED = "desktop-focused"` to `InstructionTemplateType` so `ROBOTMCP_INSTRUCTIONS_TEMPLATE=desktop-focused` resolves.
- [ ] 2.3 Add mirror `domains/instruction/templates/desktop_focused.txt` (reference copy, consistent with the other `.txt` mirrors).

## 3. Process → core for DESKTOP_TESTING (spike lever #5)
- [ ] 3.1 `session_models.py:585-586`: move `Process` from `optional_libraries` to `core_libraries` for `DESKTOP_TESTING`, keeping `PlatynUI.BareMetal` first (search-order builder derives from core order — comment at `session_models.py:581-584`).
- [ ] 3.2 Verify `get_libraries_to_load()` (`session_models.py:1280-1307`) now includes Process for desktop and the derived search order still leads with `PlatynUI.BareMetal`.

## 4. Tests + validation
- [ ] 4.1 `tests/unit/test_desktop_turn_economy_guidance.py`: (a) desktop init returns `desktop_guidance` with all libdoc keywords, arg order (incl. `Take Screenshot` descriptor-first) and crib markers (`/app:*[@Name=`, `control:Frame`, `get_locator_guidance`); size bound; process-wide cache (second init does not re-parse libdoc); no bundle for Browser/RequestsLibrary init; (b) `get_by_name("desktop-focused")` + `InstructionTemplateType.from_string("desktop-focused")` + env-var resolution; existing template names unaffected; (c) desktop profile: Process in libraries-to-load, PlatynUI.BareMetal first in search order.
- [ ] 4.2 Full unit suite green (no regressions in `test_platynui_newcore_plugin.py`, instruction-domain tests, session-model tests).
- [ ] 4.3 `openspec validate desktop-turn-economy-guidance --strict` passes.
