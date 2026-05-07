# Appium Pre-Validation `tag_name` Bug — RESOLVED

## Issue Reference
`docs/issues/robotmcp-prevalidation-appium-analysis.md`

## Status
**RESOLVED** on branch `fix/appium_prevalidation`. Solution C from the
original analysis (substring matching) was implemented, plus a `className`
attribute fallback for React Native / Compose / Flutter wrappers.

## Problem Recap
`KeywordExecutor._run_appium_state_check` decides whether an Appium element
is "editable" by inspecting `element.tag_name`. The previous comparison
used **exact tuple membership**:

```python
if tag_name in ("edittext", "textfield", "input", "textarea"):
    current_states.add("editable")
```

On Android, `tag_name` returns the full Java class
(`android.widget.EditText`); on iOS it returns the XCUI element type
(`XCUIElementTypeTextField`). Neither equals one of the four tokens, so
`Input Text` / `Input Value` / `Clear Text` always failed pre-validation
with `Element missing required states: editable`, even on perfectly valid
edit fields. The only workaround was `Run Keyword AppiumLibrary.Input Text`,
which bypasses pre-validation entirely.

## Root Cause Confirmation
Reproduced before any fix with a parametrized unit test using realistic
mobile `tag_name` values:

```text
FAILED tests/unit/test_appium_prevalidation_tagname_fix.py::
  TestAppiumEditableDetectionRealistic::
  test_editable_detected_for_mobile_tag_names[android.widget.EditText]
AssertionError: tag_name='android.widget.EditText' should be classified
as editable; got missing=['editable']
```

The failure isolates the bug to a single line of the comparison; nothing
else along the validation pipeline misbehaves.

## Fix
File: `src/robotmcp/components/execution/keyword_executor.py`

1. **Class-level substring set** that covers HTML tags, Android Java class
   names, and iOS XCUI element types:

   ```python
   _EDITABLE_TAG_SUBSTRINGS = (
       "edittext",     # Android: *.EditText, AutoCompleteTextView, AppCompatEditText
       "textfield",    # iOS: XCUIElementTypeTextField, XCUIElementTypeSecureTextField
       "searchfield",  # iOS: XCUIElementTypeSearchField
       "input",        # Web fallback: <input>
       "textarea",     # Web fallback: <textarea>
       "textview",     # Some Android variants expose editable TextView
   )
   ```

2. **Substring containment** in `_run_appium_state_check`, replacing the
   exact-membership check:

   ```python
   tag_name = element.tag_name.lower() if hasattr(element, 'tag_name') else ""
   is_editable_tag = any(
       token in tag_name for token in self._EDITABLE_TAG_SUBSTRINGS
   )
   if not is_editable_tag and hasattr(element, 'get_attribute'):
       try:
           class_attr = (element.get_attribute("className") or "").lower()
       except Exception:
           class_attr = ""
       if class_attr and any(
           token in class_attr for token in self._EDITABLE_TAG_SUBSTRINGS
       ):
           is_editable_tag = True
   if is_editable_tag and "enabled" in current_states:
       current_states.add("editable")
   ```

3. **`className` attribute fallback** — Compose/React Native/Flutter often
   surface a generic wrapper as the leaf element while still exposing the
   real widget class via the `className` attribute. The fallback path
   inspects that attribute when `tag_name` alone does not match. The
   fallback is wrapped in a try/except so a `Stale element` or driver
   exception cannot crash pre-validation.

## Verification
**New regression suite:** `tests/unit/test_appium_prevalidation_tagname_fix.py`
(53 tests, all green) — three classes:

`TestAppiumEditableDetectionRealistic` (14 tests):

* `test_editable_detected_for_mobile_tag_names` — parametrized over
  Android (`android.widget.EditText`, `android.widget.AutoCompleteTextView`,
  `androidx.appcompat.widget.AppCompatEditText`), iOS
  (`XCUIElementTypeTextField`, `XCUIElementTypeSecureTextField`,
  `XCUIElementTypeSearchField`), web (`input`, `textarea`), and the prior
  lower-cased forms (`edittext`, `textfield`).
* `test_non_editable_tag_does_not_get_editable_state` — `android.widget.Button`
  must still report `editable` as missing.
* `test_disabled_edittext_does_not_get_editable_state` — `is_enabled()=False`
  blocks the editable promotion.
* `test_classname_attribute_fallback_for_react_native_wrapper` — wrapper tag
  + `className=android.widget.EditText` is detected via fallback.
* `test_classname_attribute_unavailable_does_not_crash` — `get_attribute`
  raising must NOT bubble out as a `Pre-validation error`.

`TestAppiumNonTextControlTypesAudit` (36 tests):

* `test_non_text_controls_pass_click_prevalidation` — parametrized over
  21 control types (Android `Button`/`ImageButton`/`ImageView`/`CheckBox`/
  `RadioButton`/`Switch`/`ToggleButton`/`Spinner`/`TextView`/
  `ViewGroup`/`View`/`RecyclerView`; iOS
  `Button`/`StaticText`/`Image`/`Switch`/`Cell`/`NavigationBar`/`TabBar`/
  `Picker`/`PickerWheel`) — click pre-validation must always succeed when
  the element is displayed and enabled, regardless of platform tag string.
* `test_non_text_controls_not_promoted_to_editable` — defensive check that
  buttons, switches, picker wheels etc. are never silently promoted to
  editable (would otherwise mask `Input Text` mistakes).

`TestAppiumAdditionalEditableTypes` (3 tests):

* `MultiAutoCompleteTextView` (matched via `textview` substring),
  `EditTextPreference` (matched via `edittext`),
  `XCUIElementTypeTextView` (multi-line iOS, matched via `textview`).

**Existing test suite:** unchanged after the fix.

```text
$ uv run pytest tests/unit/test_keyword_executor_pre_validation.py
============================== 59 passed in 1.52s ==============================

$ uv run pytest tests/unit/
============================ 5267 passed, 1 skipped ============================
```

The `test_editable_state_for_edittext_elements` test that ships with the
project still passes because lower-cased `"edittext"` is also a substring
match. No existing assertion needed updating.

## Behaviour Change Summary
| Tag returned by `element.tag_name`                      | Before  | After |
|---------------------------------------------------------|---------|-------|
| `android.widget.EditText`                               | not editable (BUG) | editable |
| `android.widget.AutoCompleteTextView`                   | not editable (BUG) | editable |
| `androidx.appcompat.widget.AppCompatEditText`           | not editable (BUG) | editable |
| `XCUIElementTypeTextField`                              | not editable (BUG) | editable |
| `XCUIElementTypeSecureTextField`                        | not editable (BUG) | editable |
| `XCUIElementTypeSearchField`                            | not editable (BUG) | editable |
| `input`, `textarea`, `edittext`, `textfield`            | editable           | editable (unchanged) |
| `android.widget.Button`, generic wrappers w/o className | not editable       | not editable (unchanged) |
| Disabled EditText                                       | not editable       | not editable (unchanged) |

## Workaround Status
The `Run Keyword AppiumLibrary.Input Text` workaround in the affected
`docs/issues/saucelabs.robot` test is no longer required after this fix —
plain `Input Text accessibility_id=test-Username standard_user` will pass
pre-validation on Android (Samsung Galaxy / SauceLabsDemo.apk) and iOS.
The `.robot` file in `docs/issues/` is left as-is for historical context;
new tests authored via `build_test_suite` no longer need the wrapper.

## Audit Conclusion — Other Appium Control Types
The audit confirmed the bug was confined to the editable-state heuristic.
Pre-validation only ever injects `attached`/`visible`/`enabled`/`editable`
into the state set. `visible` and `enabled` come from `is_displayed()`
and `is_enabled()` — driver-level evaluations that depend on the live
UI tree, not on the tag string — so click/tap/check/select on Android
`Button`/`ImageButton`/`ImageView`/`CheckBox`/`RadioButton`/`Switch`/
`ToggleButton`/`Spinner`/`ViewGroup`/`RecyclerView` and iOS
`Button`/`StaticText`/`Image`/`Switch`/`Cell`/`NavigationBar`/`TabBar`/
`Picker`/`PickerWheel` all pass pre-validation regardless of how the
driver formats `tag_name`. The new
`TestAppiumNonTextControlTypesAudit` class pins this contract for 21
control types.

The Selenium and Browser-library state checks (`_run_selenium_state_check`,
`_run_browser_get_states`) do not have an analogous bug: Selenium uses a
JavaScript snippet that compares `el.tagName === 'INPUT' || 'TEXTAREA'`
which is always the literal HTML tag, and Browser Library asks Playwright
itself for `Get Element States` (no tag comparison performed in
robotmcp).

## Local Validation Against SauceLabs
The original analysis was reproduced against
`https://ondemand.eu-central-1.saucelabs.com:443/wd/hub` using the
`SauceLabsDemo.apk` recipe in `docs/issues/saucelabs.robot`. Those files
contain a real SauceLabs username and access key and **must never be
committed**. This branch:

* Adds explicit `.gitignore` entries for
  `docs/issues/saucelabs.robot`, `docs/issues/saucelabs.txt`, and any
  `docs/issues/*credentials*` file.
* Confirms via `git ls-files` that no SauceLabs-related file is tracked
  in the repository.
* Confirms via grep that no CI workflow under `.github/workflows/*.yml`,
  no `tests/e2e/*.robot`, and no source file references
  `SAUCE_USERNAME` / `SAUCE_ACCESS_KEY`. SauceLabs is therefore **only**
  a local-machine validation aid for this fix; CI continues to rely on
  the existing unit-test contract.

## Files Changed
* `src/robotmcp/components/execution/keyword_executor.py` — fix and
  shared `_EDITABLE_TAG_SUBSTRINGS` constant.
* `tests/unit/test_appium_prevalidation_tagname_fix.py` — 53 regression
  tests covering Android, iOS, web, wrapper fallback, exception
  safety, and the non-text control-type audit.
* `.gitignore` — explicit blocks for the local SauceLabs reproduction
  artefacts.
* `docs/issues/robotmcp-prevalidation-appium-analysis_RESOLVED.md` — this
  resolution document.

## Notes / Future Work
* The `ROBOTMCP_PRE_VALIDATION=0` global escape hatch is still available
  but is no longer the recommended remedy for Appium text-input failures.
* Phase 4 of the original proposal (a dedicated `mobile_exec` tool profile)
  is **not** required for this fix; it remains a possible future
  refinement if mobile-only optimisations diverge further from the desktop
  defaults.
