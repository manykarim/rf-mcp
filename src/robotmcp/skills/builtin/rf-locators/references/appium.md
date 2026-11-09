# AppiumLibrary locators

Ported from rf-mcp 0.36.0. Legacy source: main:src/robotmcp/utils/rf_native_type_converter.py (`get_appium_locator_guidance`, `_analyze_appium_error`); server.py (`get_appium_locator_guidance` docstring); utils/hints.py (`_check_mobile_context_mismatch`, `_check_mobile_scroll_needed`, `_check_click_intercepted`, `_check_element_not_interactable`, `_check_stale_element`, `_check_unexpected_alert`, `_check_timeout`); components/execution/page_source_service.py (`_parse_android_source`, `_parse_ios_source`); domains/intent/adapters/locator_normalizer_adapter.py.

## Strategies
| Strategy | Example | Notes |
|---|---|---|
| default (id) | `login_btn` | A plain string is looked up as an id |
| id | `id=com.app:id/submit` | |
| accessibility_id | `accessibility_id=submit-button` | Recommended for cross-platform tests |
| xpath | `//*[@text='Login']` or `xpath=//*[@type='android.widget.EditText']` | A leading `//` is detected automatically |
| class | `class=android.widget.Button` | |
| identifier | `identifier=my_element` | Matches the @id attribute |
| name | `name=my_element` | Selendroid only |
| android | `android=UiSelector().resourceId("com.app:id/submit").enabled(true)` | Android UiAutomator |
| predicate | `predicate=type == 'XCUIElementTypeButton' AND name == 'Submit'` | iOS NSPredicate |
| chain | `chain=XCUIElementTypeWindow[1]/XCUIElementTypeButton[@name='Login']` | iOS class chain |
| css | `css=.login-form .submit-btn` | WEBVIEW context only |

## Stability order
Legacy had no explicit ranking. It follows from the tips:
1. `accessibility_id` (content-desc on Android, name on iOS)
2. `id` (Android resource-id)
3. platform queries (`android=UiSelector()…`, `predicate=`, `chain=`)
4. `xpath`, best anchored on text, resource-id or content-desc
5. coordinates (`Swipe`) only for scrolling

## Android vs iOS
| | Android | iOS |
|---|---|---|
| Attributes in Get Source | `resource-id`, `text`, `content-desc`, `class`, `clickable`, `enabled`, `bounds` | `type`, `name`, `label`, `value`, `enabled`, `visible`, x, y, width, height |
| Rich query | `android=UiSelector().className('android.widget.Button').text('Login')`, `android=UiSelector().description('Search').clickable(true)` | `predicate=name BEGINSWITH 'login' AND visible == 1` |
| Hierarchy | XPath on class names, e.g. `//android.widget.EditText` | `chain=XCUIElementTypeWindow[1]/*` |
| Default automation | UiAutomator2 | XCUITest |

## Contexts (hybrid apps)
Native locators (id, accessibility_id, xpath) work only in `NATIVE_APP`. CSS works only in a `WEBVIEW_*` context. An "Unknown command" error usually means the session is in the wrong context.

```robotframework
@{contexts}=    Get Contexts
${ctx}=    Get Current Context
Switch To Context    WEBVIEW_1
Click Element    css=.login-button
Switch To Context    NATIVE_APP
```

## Inspect before guessing
`${xml}=    Get Source` returns the current screen hierarchy (XML). Read the real `resource-id`, `content-desc`, `text`, `name` and `label` values from it. Mobile screens render only what is visible, so an element that exists off-screen will not be in the source until you scroll. Legacy also pointed to Appium Inspector.

## Element references
AppiumLibrary 1.4 and later accepts WebElements:

```robotframework
@{els}=    Get Webelements    class=android.widget.Button
Click Element    ${els}[0]
```

Prefer locator strings over stored elements. References go stale after a screen change, so find the element again instead of reusing the old reference.

## Failure recipes
| Symptom | Fix |
|---|---|
| Element not found | Run `Get Source`. Then `Scroll Down    <loc>`, or `Swipe    start_x=500    start_y=1500    end_x=500    end_y=500`. Check `Get Current Context` |
| Not interactable, or tap intercepted | `Hide Keyboard` (the keyboard, toasts and permission dialogs often cover elements), then scroll and retry |
| Slow or async screen | `Wait Until Element Is Visible    <loc>    10s`, `Wait Until Page Contains Element    <loc>`, or `Set Appium Timeout    30s` |
| Stale element | Wait for the element and find it again (legacy suggested `Expect Element    <loc>    visible    timeout=10`; check with rf_doc) |
| System alert or permission | iOS: `Click Alert Button    Allow`. Android: `Click Text    Allow` (check `Click Alert Button` with rf_doc). Grant permissions up front when possible |
| Session or driver error | Check that `Open Application` received correct capabilities (platformName, deviceName/udid, app or appPackage+appActivity on Android, app or bundleId on iOS), that the Appium server is reachable, and that the device is connected |

Legacy normalization turned bare text and `text=X` into `accessibility_id=X` for Appium. Do the same when a Browser-style text locator is carried over.
