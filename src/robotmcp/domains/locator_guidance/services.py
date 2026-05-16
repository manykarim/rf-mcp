"""Service returning topic-based locator guidance payloads."""
from __future__ import annotations

from typing import Any

from robotmcp.domains.locator_guidance.value_objects import (
    BrowserLocatorCookbook,
    CookbookEntry,
    KnownValidationLibraries,
    SpaWizardGuidance,
)

_SPA_WIZARD_GUIDANCE = SpaWizardGuidance(
    jquery_validation={
        "description": "jQuery validation plugins intercept form submission and field blur events.",
        "detection": [
            "window.$.validator !== undefined",
            "window.$.fn.idealforms !== undefined",
            "$('#formId').data('idealforms') !== undefined",
        ],
        "commit_without_keystrokes": [
            "Call form.idealforms('validate', '#fieldId') per field after setting its value via JS.",
            "Call $('#formId').valid() to trigger jQuery Validate on the whole form.",
            "Use intent_action(commit_form=True) if Agent C's commit_form parameter is available.",
        ],
        "robot_snippet": (
            "# Commit idealForms state after JS value injection\n"
            "Execute JavaScript    document.querySelector('#fieldId').value = 'value';\n"
            "Execute JavaScript    $('#formId').idealforms('validate', '#fieldId');"
        ),
    },
    change_event_swallowing={
        "description": (
            "SPAs (idealForms, formvalidation.io) intercept native change/input events. "
            "Setting value via JS without dispatching a jQuery change event does NOT update "
            "validation state."
        ),
        "recommendation": [
            "Use Type Text (real keystrokes) so the browser fires native events.",
            "Or dispatch jQuery change: Execute JavaScript    $('#fieldId').trigger('change');",
            "Or call the library API directly (see jquery_validation section).",
        ],
        "anti_pattern": "document.querySelector('#id').value = 'x'  # silently skips validation",
    },
    auto_advancing_wizards={
        "description": (
            "Dispatching events in a wizard may trigger automatic tab/screen advancement "
            "before all fields are filled."
        ),
        "recommendation": [
            "Bulk-set all field values via JS first (no events yet).",
            "Call the validation library API to commit the state.",
            "Then click Next via JS: document.querySelector('.next-btn').click()",
            "Or use Execute JavaScript    document.querySelector('#nextBtn').dispatchEvent(new MouseEvent('click'));",
        ],
    },
    checkbox_minoption={
        "description": (
            "Checkbox groups with minoption rules (e.g., Hobbies — select at least 1) "
            "prevent downstream content from rendering until the group validation passes."
        ),
        "recommendation": [
            "Tick the first available checkbox in the group before clicking Next.",
            "Verify downstream content appears before proceeding.",
        ],
        "robot_snippet": (
            "# Tick first checkbox in a group with name 'Hobbies'\n"
            "Click    css=input[name='Hobbies']:not(:checked):first-of-type\n"
            "# Or using Browser Library nth:\n"
            "Click    css=input[name='Hobbies'] >> nth=0"
        ),
    },
    hidden_zero_size_elements={
        "description": (
            "idealSteps containers and <tfoot> rows may be zero-size or hidden but still "
            "present in the DOM, causing 'element not interactable' or pre-validation failures."
        ),
        "recommendation": [
            "Use pre_validate=False (Agent B parameter on execute_step) to bypass visibility checks.",
            "Use force=True (Agent C parameter on intent_action) to force interaction.",
            "Or scroll element into view: Execute JavaScript    el.scrollIntoView();",
        ],
        "robot_snippet": (
            "# Force-click a hidden element\n"
            "Execute JavaScript    document.querySelector('#hiddenBtn').click();"
        ),
    },
    datepicker_overlay={
        "description": (
            ".ui-datepicker overlay remains visible after date selection and intercepts "
            "subsequent clicks (e.g., on the Next button)."
        ),
        "recommendation": [
            "After selecting a date, hide the datepicker before clicking Next:",
            "Execute JavaScript    var dp = document.querySelector('.ui-datepicker'); if(dp) dp.style.display='none';",
            "Or wait for it to close: Wait For Elements State    .ui-datepicker    hidden",
        ],
        "robot_snippet": (
            "Select Date    css=#dateField    2024-06-15\n"
            "Execute JavaScript    var dp=document.querySelector('.ui-datepicker'); if(dp) dp.style.display='none';\n"
            "Click    css=#nextButton"
        ),
    },
    strict_mode_duplicates={
        "description": (
            "Some pages reuse the same id for desktop and mobile navigation elements. "
            "Browser Library strict mode raises an error when a selector matches multiple elements."
        ),
        "recommendation": [
            "Use >> nth=0 to target the first match: Click    css=#navItem >> nth=0",
            "Or use intent_action(..., nth=0) when Agent C's nth parameter is available.",
            "Or make the selector more specific: css=.desktop-nav #navItem",
            "Or disable strict mode: Set Strict Mode    False  (then re-enable after the step)",
        ],
        "robot_snippet": (
            "# Target first of duplicate elements\n"
            "Click    css=#duplicateId >> nth=0\n"
            "# Or with text disambiguation\n"
            "Click    xpath=(//button[@id='btn'])[1]"
        ),
    },
)

_BROWSER_LOCATOR_COOKBOOK = BrowserLocatorCookbook(
    entries=(
        CookbookEntry(
            title="Wrapper-label click (input INSIDE label)",
            locator_template="*css=label >> id=<input-id>",
            example="Click   *css=label >> id=newsletter",
            use_when=(
                "the <input> is an ancestor-descendant of a <label> and the input "
                "is visually hidden (styled radio/checkbox; common with custom "
                "form themes and SPA validation libraries that hide native inputs)"
            ),
        ),
        CookbookEntry(
            title="Wrapper-label check (input INSIDE label)",
            locator_template="*css=label >> id=<input-id>",
            example="Check Checkbox   *css=label >> id=accept_terms",
            use_when=(
                "checkbox is inside a <label> wrapper and Check Checkbox reports "
                "element not visible"
            ),
        ),
        CookbookEntry(
            title="Sibling label by 'for' attribute (Bootstrap form-check)",
            locator_template="css=label[for='<input-id>']",
            example="Click   css=label[for='checkbox1']",
            use_when=(
                "the <input> and <label> are siblings (label has for=<input-id> "
                "instead of wrapping the input). Common with Bootstrap form-check, "
                "Tailwind UI, plain HTML. The wrapping selector "
                "(*css=label >> id=...) will NOT match this case."
            ),
        ),
        CookbookEntry(
            title="Click visible label by text",
            locator_template="text=<label-text>",
            example="Click   text=I accept the terms",
            use_when=(
                "the wrapper label has clear visible text and no useful id is available"
            ),
        ),
        CookbookEntry(
            title="Visibility-scoped CSS chain",
            locator_template='section[style="display: block;"] >> text=<button-text>',
            example='Click   section[style="display: block;"] >> text=Next',
            use_when=(
                "multiple wizard steps exist in the DOM but only one is visible; "
                "prevents ambiguous-match errors in jQuery accordions, idealSteps, "
                "multi-page wizards"
            ),
        ),
        CookbookEntry(
            title="Sibling input by label text",
            locator_template='"<label-text>" >> .. >> input',
            example='Fill Text   "Email" >> .. >> input    user@example.com',
            use_when=(
                "the input has no useful id but a visible label sits as a sibling or ancestor"
            ),
        ),
        CookbookEntry(
            title="Value-attribute on grouped radio/checkbox",
            locator_template="*css=label >> css=[value=<value>]",
            example="Click   *css=label >> css=[value=premium]",
            use_when=(
                "radio button or checkbox groups where only the value attribute "
                "distinguishes options"
            ),
        ),
        CookbookEntry(
            title="Strict-mode disambiguation via nth",
            locator_template="<locator> >> nth=<n>",
            example="Click   id=nav-main >> nth=0",
            use_when=(
                "an id or selector appears multiple times (e.g. mobile vs desktop nav) "
                "and Browser Library strict mode raises an error"
            ),
        ),
        CookbookEntry(
            title="Network-await pattern",
            locator_template="Promise To Wait For Response  <url>  timeout=<s>",
            example=(
                "${promise}=    Promise To Wait For Response    **/api/submit    timeout=10\n"
                "Click    id=submit\n"
                "Wait For    ${promise}"
            ),
            use_when=(
                "verifying application state after a button click that triggers "
                "an HTTP request; avoids timing fragility of fixed sleeps"
            ),
        ),
        CookbookEntry(
            title="Force-click hidden element",
            locator_template="Click With Options    <selector>    force=True",
            example="Click With Options    id=hidden_submit    force=True",
            use_when=(
                "Pre-validation rejects an element as 'not visible' AND no "
                "visible wrapper or scoped section can be discovered. This is "
                "the ACCEPTABLE FALLBACK when natural locators (entries a-d) "
                "are not obvious. Browser library's Click With Options skips "
                "Playwright's actionability check when force=True. Prefer "
                "wrapper-locator patterns (entries a-b) first; this is the "
                "documented escape hatch. Also reachable via "
                "intent_action(intent='click', force=True)."
            ),
        ),
    ),
    see_also=("spa_wizards",),
)

_KNOWN_VALIDATION_LIBRARIES = KnownValidationLibraries(
    libraries=[
        {
            "name": "idealForms",
            "detection": [
                "window.$.fn.idealforms !== undefined",
                "$('#form').data('idealforms') !== undefined",
            ],
            "commit_api": "form.idealforms('validate', '#fieldId')",
            "notes": "Field-level validation; minoption rule for checkbox groups.",
        },
        {
            "name": "jQuery Validate",
            "detection": [
                "window.$.validator !== undefined",
                "$('#form').data('validator') !== undefined",
            ],
            "commit_api": "$('#form').valid()",
            "notes": "Validates entire form on submit; individual field via .element(field).",
        },
        {
            "name": "formvalidation.io",
            "detection": [
                "window.FormValidation !== undefined",
                "window.FormValidation.formValidation !== undefined",
            ],
            "commit_api": "fv.revalidateField('fieldName')",
            "notes": "Intercepts native input/change events; requires revalidateField per field.",
        },
        {
            "name": "vee-validate (Vue)",
            "detection": [
                "window.__VUE__ !== undefined",
                "document.querySelector('[data-vv-name]') !== null",
            ],
            "commit_api": "veeValidateInstance.validate()",
            "notes": "Vue-based; validation triggers on input events. Use real keystrokes.",
        },
    ]
)


class LocatorTopicService:
    """Returns pre-built guidance payloads for known locator topics."""

    SUPPORTED_TOPICS: frozenset[str] = frozenset(
        {"spa_wizards", "known_validation_libraries", "browser_locators"}
    )

    def get_spa_wizards(self) -> dict[str, Any]:
        result = _SPA_WIZARD_GUIDANCE.to_dict()
        result["success"] = True
        result.setdefault("see_also", ["browser_locators"])
        return result

    def get_known_validation_libraries(self) -> dict[str, Any]:
        result = _KNOWN_VALIDATION_LIBRARIES.to_dict()
        result["success"] = True
        return result

    def get_browser_locators(self) -> dict[str, Any]:
        result = _BROWSER_LOCATOR_COOKBOOK.to_dict()
        result["success"] = True
        return result

    def get_topic(self, topic: str) -> dict[str, Any] | None:
        """Dispatch to the appropriate handler for the given topic name.

        Args:
            topic: One of the supported topic names (case-sensitive).

        Returns:
            Guidance dict with ``success: True`` on match, or ``None`` if the
            topic is unknown (caller is responsible for building the error response).
        """
        if topic == "spa_wizards":
            return self.get_spa_wizards()
        if topic == "known_validation_libraries":
            return self.get_known_validation_libraries()
        if topic == "browser_locators":
            return self.get_browser_locators()
        return None
