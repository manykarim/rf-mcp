# Tasks: desktop-a11y-atspi-backend

## 1. Correct the remediation/guidance advice

- [x] 1.1 `ui_tree_service._build_exposure_diagnostic`: remediation recommends `GTK_A11Y=atspi` (backend name) and flags `GTK_A11Y=1` as rejected by modern GTK → empty tree; keep the AT-SPI-bus + relaunch steps
- [x] 1.2 `rf_native_type_converter.get_platynui_locator_guidance`: accessibility-exposure rule names `GTK_A11Y=atspi` (not `1`) as the bridge fix

## 2. Pin the bridge in replayed desktop suites

- [x] 2.1 `test_builder._inject_desktop_replay_environment`: add `Set Environment Variable    GTK_A11Y    atspi` to the `Prepare Desktop Display Environment` preamble (before `Remove Environment Variable WAYLAND_DISPLAY`); non-desktop suites untouched

## 3. Tests

- [x] 3.1 `test_suite_replay_environment.py`: self-sufficient test asserts the `GTK_A11Y    atspi` line; new `test_gtk_a11y_pins_atspi_backend_not_1` forbids the `GTK_A11Y    1` form
- [x] 3.2 `test_desktop_native_platynui_alignment.py`: `test_window_present_not_exposed` asserts `GTK_A11Y=atspi` in remediation + `1` only as a rejected anti-pattern; `test_accessibility_exposure_section` asserts guidance names `GTK_A11Y=atspi`, not `1`

## 4. Validation

- [x] 4.1 Affected unit files green (`test_suite_replay_environment.py`, `test_desktop_native_platynui_alignment.py` — 29 passed)
- [x] 4.2 Full unit suite green (no regressions) — 6850 passed + 1 skipped
