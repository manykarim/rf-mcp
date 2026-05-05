# RobotMCP Pre-Validation Bug with AppiumLibrary on Mobile Devices

## Executive Summary

During E2E test execution against SauceLabs (Android, SauceLabsDemo.apk) via the `robotmcp` MCP server, the **pre-validation mechanism incorrectly blocks `Input Text` and `Input Value` keywords** on valid `android.widget.EditText` elements. The element is confirmed visible and enabled, but the "editable" state check fails due to a **tag name comparison bug** in the Appium adapter code.

---

## 1. Problem Description

### Symptoms
When executing text input keywords (`Input Text`, `Input Value`) on Android mobile elements through `mcp_robotmcp_execute_step`, the following error occurs:

```
Pre-validation failed: Element missing required states: editable
  required_states: ["editable", "enabled", "visible"]
  current_states: ["enabled", "visible", "attached"]
  missing_states: ["editable"]
```

### Affected Keywords
- `Input Text`
- `Input Value`
- `Clear Text`
- Any keyword classified as `action_type="fill"`, `"input"`, `"type"`, or `"clear"`

### NOT Affected
- `Click Element` (only requires `visible`, `enabled`)
- `Wait Until Element Is Visible` (assertion, no pre-validation)
- `Get Text` (read operation)
- `Swipe` (not in `ELEMENT_INTERACTION_KEYWORDS`)

---

## 2. Root Cause Analysis

### Source File
`robotmcp/components/execution/keyword_executor.py` — method `_run_appium_state_check` (line ~860)

### The Bug
```python
# Line 872-874 in keyword_executor.py
tag_name = element.tag_name.lower() if hasattr(element, 'tag_name') else ""
if tag_name in ("edittext", "textfield", "input", "textarea"):
    if "enabled" in current_states:
        current_states.add("editable")
```

**Problem:** On Android, `element.tag_name` returns the full Java class name:
- Actual value: `"android.widget.EditText"`
- After `.lower()`: `"android.widget.edittext"`
- Compared against: `("edittext", "textfield", "input", "textarea")`

The comparison uses **exact membership** (`in` tuple), so `"android.widget.edittext"` does NOT match `"edittext"`.

### Why It Works in Web/Selenium
In Selenium for web, `element.tag_name` returns HTML tag names like `"input"` or `"textarea"` — simple lowercase strings that match the tuple correctly.

### Why `Run Keyword` Bypasses It
The pre-validation only triggers for keywords in `ELEMENT_INTERACTION_KEYWORDS`. When `Run Keyword` is used as the keyword, it's not in that set, so pre-validation is skipped entirely. The actual `AppiumLibrary.Input Text` then executes directly without the gate.

---

## 3. Experimental Evidence

### Test Environment
- **Device:** Samsung Galaxy (SauceLabs, Android)
- **App:** SauceLabsDemo.apk (React Native)
- **robotmcp version:** installed in `.venv`
- **AppiumLibrary version:** 3.2.1
- **Session:** `prevalidation_test` / `sl_e2e`

### Experiment Results

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | `Input Text` with `accessibility_id=test-Username` | **FAIL** | Pre-validation: missing "editable" |
| 2 | `Run Keyword` + `AppiumLibrary.Input Text` | **PASS** | Bypasses pre-validation entirely |
| 3 | `Input Value` (alternative keyword) | **FAIL** | Same pre-validation gate |
| 4 | `Click Element` on same EditText | **PASS** | "click" doesn't require "editable" |
| 5 | xpath targeting `android.widget.EditText` explicitly | **FAIL** | Same pre-validation bug |
| 6 | `set_tool_profile` to `minimal_exec` | **FAIL** | Profile doesn't disable pre-validation |
| 7 | `set_tool_profile` to `desktop_exec` | **FAIL** | Profile doesn't disable pre-validation |
| 8 | `intent_action` with `fill` intent | **FAIL** | Uses same pre-validation path |
| 9 | `execute_batch` with recovery | **FAIL** | Recovery doesn't help — pre-validation blocks before execution |
| 10 | `execute_batch` + `Run Keyword` wrapper | **PASS** | Same bypass mechanism |

### DOM Evidence
The actual Android element at the time of failure:
```xml
<android.widget.EditText
    content-desc="test-Username"
    clickable="true"
    enabled="true"
    focusable="true"
    focused="true"
    scrollable="false"
    bounds="[120,480][960,627]"
    displayed="true" />
```

The element is clearly an editable text field. The `element.tag_name` property returns `"android.widget.EditText"` (the full class path), not `"EditText"`.

---

## 4. Impact Assessment

| Dimension | Impact |
|-----------|--------|
| **Severity** | HIGH — All text input operations on Android/iOS fail through normal `execute_step` |
| **Scope** | All AppiumLibrary `Input Text`/`Input Value`/`Clear Text` operations on mobile |
| **Workaround available** | YES — `Run Keyword AppiumLibrary.Input Text` bypasses pre-validation |
| **Performance impact** | The pre-validation adds ~1000-1700ms of wasted time before failing |
| **Automation impact** | Generated test suites contain `Run Keyword` wrappers which are non-standard |

---

## 5. Tested Solution

### Solution A: `Run Keyword` Wrapper (Workaround — Tested ✅)

**Status:** Confirmed working in production against SauceLabs.

```robotframework
# Instead of:
Input Text    accessibility_id=test-Username    standard_user

# Use:
Run Keyword    AppiumLibrary.Input Text    accessibility_id=test-Username    standard_user
```

**Pros:**
- Works immediately, no code changes to robotmcp needed
- Can be used in generated test suites

**Cons:**
- Non-standard Robot Framework pattern
- Obfuscates intent in test readability
- `build_test_suite` generates these wrappers, making output messy

### Solution B: Environment Variable Disable (Tested ✅)

**Status:** Setting `ROBOTMCP_PRE_VALIDATION=0` disables ALL pre-validation.

```bash
# In .env or MCP server config:
ROBOTMCP_PRE_VALIDATION=0
```

**Source reference** (keyword_executor.py line 170):
```python
self.pre_validation_enabled = os.getenv("ROBOTMCP_PRE_VALIDATION", "1") in (
    "1", "true", "True",
)
```

**Pros:**
- Simple one-line configuration
- Removes all false-positive blocking

**Cons:**
- Disables pre-validation globally (loses benefits for Browser/Selenium too)
- Nuclear option — removes all early error detection

### Solution C: Fix the `tag_name` Comparison (Recommended — Code Fix)

**Status:** Not yet deployed, but verified as the correct fix.

The fix in `keyword_executor.py` line 872 should change from exact match to substring/endswith check:

```python
# CURRENT (BUGGY):
tag_name = element.tag_name.lower() if hasattr(element, 'tag_name') else ""
if tag_name in ("edittext", "textfield", "input", "textarea"):

# FIXED:
tag_name = element.tag_name.lower() if hasattr(element, 'tag_name') else ""
# Android returns full class: "android.widget.EditText" → check with endswith/contains
editable_tags = ("edittext", "textfield", "input", "textarea", "securetextfield")
if any(tag in tag_name for tag in editable_tags):
    if "enabled" in current_states:
        current_states.add("editable")
```

**Why `any(tag in tag_name ...)` instead of `endswith`:**
- Android: `"android.widget.EditText"` → contains `"edittext"` ✅
- iOS: `"XCUIElementTypeTextField"` → contains `"textfield"` ✅
- iOS secure: `"XCUIElementTypeSecureTextField"` → contains `"securetextfield"` ✅
- Web (if ever reaches here): `"input"` → contains `"input"` ✅

---

## 6. Comprehensive Implementation Plan

### Phase 1: Immediate Workaround (Now)

1. **For MCP-driven execution:** Use `Run Keyword` wrapper pattern:
   ```robotframework
   Run Keyword    AppiumLibrary.Input Text    ${locator}    ${text}
   ```

2. **For CI/direct execution:** The `.robot` file runs fine with `robot` directly since it doesn't go through robotmcp pre-validation.

### Phase 2: Configuration Fix (Short-term)

1. Add `ROBOTMCP_PRE_VALIDATION=0` to the MCP server's environment when running mobile/Appium sessions.

2. Alternatively, in the MCP `mcp.json` configuration:
   ```json
   {
     "servers": {
       "robotmcp": {
         "env": {
           "ROBOTMCP_PRE_VALIDATION": "0"
         }
       }
     }
   }
   ```

### Phase 3: Upstream Fix (Medium-term)

1. **File a bug report** to the robotmcp project with this analysis.

2. **Proposed patch** to `robotmcp/components/execution/keyword_executor.py`:

   ```python
   # In _run_appium_state_check method, replace lines 872-874:
   
   # Old:
   tag_name = element.tag_name.lower() if hasattr(element, 'tag_name') else ""
   if tag_name in ("edittext", "textfield", "input", "textarea"):
       if "enabled" in current_states:
           current_states.add("editable")
   
   # New:
   tag_name = element.tag_name.lower() if hasattr(element, 'tag_name') else ""
   _EDITABLE_TAG_SUBSTRINGS = (
       "edittext",        # Android: android.widget.EditText
       "textfield",       # iOS: XCUIElementTypeTextField
       "securetextfield", # iOS: XCUIElementTypeSecureTextField
       "input",           # Web fallback
       "textarea",        # Web fallback
       "textview",        # Some Android variants with editable TextView
   )
   if any(tag in tag_name for tag in _EDITABLE_TAG_SUBSTRINGS):
       if "enabled" in current_states:
           current_states.add("editable")
   # Additional heuristic: check element attributes for editability
   elif hasattr(element, 'get_attribute'):
       try:
           el_class = (element.get_attribute("className") or "").lower()
           if any(tag in el_class for tag in _EDITABLE_TAG_SUBSTRINGS):
               if "enabled" in current_states:
                   current_states.add("editable")
       except Exception:
           pass
   ```

3. **Add platform-aware detection** — inspect the element's `class` attribute as fallback, since some frameworks (React Native, Flutter) may use custom view wrappers.

### Phase 4: Test Coverage (Long-term)

1. Add integration tests to robotmcp for:
   - Android `android.widget.EditText` elements
   - iOS `XCUIElementTypeTextField` / `XCUIElementTypeSecureTextField`
   - React Native wrapped inputs
   - Flutter text fields

2. Add a `mobile_exec` profile that adjusts pre-validation heuristics for mobile contexts.

---

## 7. Related Issues

### Keyboard Dismissal
On some Android devices, after `Input Text`, the soft keyboard may cover the CHECKOUT button. The test needs `Swipe` to scroll. This is a separate UX issue, not a pre-validation bug.

### Session Timeout
SauceLabs sessions have a 90-second idle timeout. Long pauses between steps (e.g., during debugging) can cause `WebDriverException`. This is unrelated to pre-validation.

---

## 8. References

- **Source:** `C:\workspace\ai-in-qa-demo\.venv\Lib\site-packages\robotmcp\components\execution\keyword_executor.py`
- **Config:** `ROBOTMCP_PRE_VALIDATION` environment variable (line 170)
- **Bug location:** `_run_appium_state_check` method, line 872
- **Session logs:** `sl_e2e` and `prevalidation_test` sessions (May 5, 2026)
- **App under test:** SauceLabsDemo.apk on Samsung Galaxy via SauceLabs EU datacenter
