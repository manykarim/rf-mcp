## 1. Reproduction

- [x] 1.1 Test (red): `plugin_manager.get_library_for_keyword("Start Process")`
  returns `None` today (core static Process lib has no keyword map)
- [x] 1.2 Confirm: `get_platynui_locator_guidance()["display_state_reading"]`
  currently leads with `native:Text.CharacterCount` (finding 1)

## 2. Process keyword resolution (D2, D3)

- [x] 2.1 Extend the static-definition mechanism so a `definitions.py` entry can
  declare a keyword set, and `StaticLibraryPlugin.get_keyword_library_map()`
  returns `{kw: name}` when present
- [x] 2.2 Declare the `Process` keyword set on its definition (Start/Run/
  Terminate Process, Process Should Be Running/Stopped, Wait For Process, Get
  Process Id/Object/Result, Is Process Running, Send Signal To Process, Switch
  Process, Terminate All Processes, Split/Join Command Line)
- [x] 2.3 Verify `_ensure_library_registration` loads + registers Process for an
  unqualified `Start Process` in a desktop session (resolution → load → register)
- [x] 2.4 Unit tests: `get_library_for_keyword` resolves the Process keywords to
  `Process`; qualified `Process.Start Process` unchanged; non-desktop sessions
  unaffected

## 3. Display-state guidance (D1)

- [x] 3.1 Rewrite `display_state_reading` so history/result `Label` nodes are the
  PRIMARY assertion path
- [x] 3.2 Demote `native:Text.CharacterCount` to a SECONDARY proxy flagged "may
  report 0 on some GTK builds even when the display changed — do not rely on it
  alone"
- [x] 3.3 Name screenshot + OCR as the explicit LAST-RESORT fallback
- [x] 3.4 Soften the `node_attribute_api`/`tips` cross-references that steer to
  CharacterCount as the display hook
- [x] 3.5 Unit tests: guidance leads with Labels; CharacterCount carries the
  unreliability warning; OCR named as last resort

## 4. Validation + docs

- [x] 4.1 Full unit suite green; confirm web/api/mobile + existing desktop flows
  unaffected
- [x] 4.2 ADR noting both fixes + the live GNOME Calculator run evidence
- [x] 4.3 Release note for the two follow-up fixes
