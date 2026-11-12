"""Built-in recovery hint rules for Robot Framework libraries and RobotCode sessions."""

from __future__ import annotations

from robotmcp.hints.engine import HintRule

BUILTIN_HINT_RULES: list[HintRule] = [
    # 1. Browser: Evaluate JavaScript single-argument misuse
    HintRule(
        id="browser-evaluate-js-single-arg",
        title="Browser.Evaluate JavaScript requires (elements_selector, function) arguments",
        error_pattern=r"(Unexpected token|not a valid selector|SyntaxError|waiting for locator| Evaluating JavaScript)",
        keyword_pattern=r"Evaluate\s+JavaScript",
        message=(
            "In Browser library, the 1st argument of `Evaluate JavaScript` is the element selector "
            "(or `${None}` for page-level evaluation) and the 2nd argument is the JavaScript expression."
        ),
        fix_examples=[
            "Evaluate JavaScript    ${None}    () => document.title",
            "${url}=    Evaluate JavaScript    ${None}    () => window.location.href",
            "${text}=    Evaluate JavaScript    css=h1    (el) => el.textContent",
        ],
        relevance=96,
    ),
    # Also catch Evaluate JavaScript when called with 1 JS-looking argument regardless of exact error string
    HintRule(
        id="browser-evaluate-js-arg-shape",
        title="Pass ${None} as the first argument to Browser.Evaluate JavaScript for page-level scripts",
        error_pattern=r".+",
        keyword_pattern=r"Evaluate\s+JavaScript",
        arg_pattern=r"Evaluate\s+JavaScript\s+(\(\)\s*=>|document\.|window\.|return\s+)",
        message=(
            "`Browser.Evaluate JavaScript` takes `elements` as its 1st argument. When evaluating page-level JS, "
            "pass `${None}` as the 1st argument and the JS expression as the 2nd argument."
        ),
        fix_examples=[
            "Evaluate JavaScript    ${None}    () => document.title",
        ],
        relevance=95,
    ),
    # 2. Browser: Playwright strict mode violation
    HintRule(
        id="browser-strict-mode-violation",
        title="Playwright strict mode violation: selector matched multiple elements",
        error_pattern=r"strict mode violation:.*resolved to \d+ elements",
        keyword_pattern=r".*",
        message=(
            "Playwright strict mode requires locators to resolve to a single element. "
            "Disambiguate the selector with `>> nth=0`, `>> visible=true`, a parent chain, "
            "or inspect the accessible page tree using `Get Aria Snapshot    css=body`."
        ),
        fix_examples=[
            "Get Aria Snapshot    css=body",
            "Click    css=button.submit >> visible=true",
            "Click    text=Add to cart >> nth=0",
        ],
        skills=[
            "skill://rf-browser/references/locators.md",
            "skill://rf-locators/references/browser.md",
        ],
        relevance=95,
    ),
    # 3. Browser: Click intercepted by overlay/modal
    HintRule(
        id="browser-click-intercepted",
        title="Element click intercepted by another element or overlay",
        error_pattern=r"intercepts pointer events",
        keyword_pattern=r".*",
        message=(
            "Another element (such as a modal, cookie banner, or animation overlay) is covering the target. "
            "Inspect the page with `Get Aria Snapshot    css=body` to dismiss the blocking element, "
            "wait for it to disappear, or use `Click With Options    <selector>    force=True`."
        ),
        fix_examples=[
            "Get Aria Snapshot    css=body",
            "Wait For Elements State    css=.modal-backdrop    hidden    timeout=5s",
            "Click With Options    <selector>    force=True",
        ],
        skills=[
            "skill://rf-browser/references/troubleshooting.md",
            "skill://rf-locators/references/browser.md",
        ],
        relevance=92,
    ),
    # 4. Browser: Locator timeout / element not found
    HintRule(
        id="browser-locator-timeout",
        title="Browser locator timed out waiting for element",
        error_pattern=r"(TimeoutError: locator\.|waiting for locator\()",
        keyword_pattern=r".*",
        message=(
            "The selector did not match a visible/enabled element within the timeout. "
            "Capture the current page structure with `Get Aria Snapshot    css=body` or `Get Url` "
            "to verify the element's role, label, or state before retrying."
        ),
        fix_examples=[
            "Get Aria Snapshot    css=body",
            "Get Url",
            "Wait For Elements State    <selector>    visible    timeout=5s",
        ],
        skills=[
            "skill://rf-browser/references/troubleshooting.md",
            "skill://rf-locators/references/browser.md",
        ],
        relevance=88,
    ),
    # 5. Browser: No page or browser open
    HintRule(
        id="browser-no-page-open",
        title="No active Browser page is open",
        error_pattern=r"(No page open|Browser has not been initialized|Tried to do playwright action, but no open page)",
        keyword_pattern=r".*",
        message="Open a browser and page first using `New Browser` and `New Page`.",
        fix_examples=[
            "New Browser    chromium    headless=True",
            "New Page    https://example.com",
        ],
        relevance=93,
    ),
    # 6. REPL syntax: Section headers sent to robotcode repl / session
    HintRule(
        id="repl-section-header",
        title="Do not send *** Settings *** or *** Test Cases *** headers into robotcode repl",
        error_pattern=r"No keyword with name '\*\*\*\s*(Settings|Variables|Test Cases|Keywords|Tasks)\s*\*\*\*'",
        keyword_pattern=r".*",
        message=(
            "`robotcode session` / `robotcode repl` executes test-body statements directly. "
            "Use `Import Library`, `Import Resource`, and `Import Variables` at the prompt, "
            "and use `.save -t \"Test Name\" path/to/suite.robot` to export a complete `.robot` file with `*** Settings ***`."
        ),
        fix_examples=[
            "Import Library    Browser",
            "Import Resource    resources/common.resource",
            ".save -t \"Smoke Test\" tests/smoke.robot",
        ],
        relevance=97,
    ),
    # 7. REPL syntax: PowerShell UTF-8 BOM on stdin
    HintRule(
        id="repl-utf8-bom",
        title="UTF-8 BOM (\\ufeff) detected at start of REPL input",
        error_pattern=r"No keyword with name '(\ufeff|\\ufeff)",
        keyword_pattern=r".*",
        message=(
            "PowerShell pipelines prepend a UTF-8 BOM (`\\ufeff`) to the first line of stdin. "
            "Send statements as a single-quoted argument (`robotcode session send <id> '<statement>'`) instead of "
            "piping them, and set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`."
        ),
        fix_examples=[
            "robotcode -f json session send <id> 'Import Library    Collections'",
        ],
        relevance=98,
    ),
    # 8. RequestsLibrary / Python dictionary bracket access
    HintRule(
        id="rf-dict-bracket-access",
        title="Use Robot Framework bracket syntax ${dict}[key] for dictionary keys",
        error_pattern=r"('dict' object has no attribute|Resolving variable '\$\{[^}]+\.[^}]+\}' failed: AttributeError)",
        keyword_pattern=r".*",
        message=(
            "Plain Python dictionaries do not support dot-attribute lookup for arbitrary keys in Robot Framework. "
            "Use bracket notation `${dict}[key]` (or `${response.json()}[key]`)."
        ),
        fix_examples=[
            "${value}=    Set Variable    ${data}[key]",
            "Should Be Equal    ${response.json()}[status]    ok",
        ],
        skills=[
            "skill://rf-requests/references/response-validation.md",
            "skill://rf-requests/SKILL.md",
        ],
        relevance=91,
    ),
    # 9. RequestsLibrary: Session & named arguments
    HintRule(
        id="requests-named-args",
        title="RequestsLibrary session keywords expect named arguments (json=, params=, headers=, expected_status=)",
        error_pattern=r"(GET|POST|PUT|DELETE|PATCH)\s+On\s+Session.*(unexpected keyword argument|takes from|SessionNotFound|No session)",
        keyword_pattern=r"(GET|POST|PUT|DELETE|PATCH)\s+On\s+Session",
        message=(
            "Create a session first with `Create Session    <alias>    <base_url>`, then call "
            "`GET On Session` / `POST On Session` with `<alias>`, `<relative_url>`, and named options "
            "`json=${body}`, `params=${params}`, `headers=${headers}`, `expected_status=200`."
        ),
        fix_examples=[
            "Create Session    api    https://httpbin.org",
            "${resp}=    GET On Session    api    /get    expected_status=200",
            "${resp}=    POST On Session    api    /post    json=${payload}    expected_status=201",
        ],
        skills=[
            "skill://rf-requests/references/request-options.md",
            "skill://rf-requests/SKILL.md",
        ],
        relevance=90,
    ),
    # 10. SeleniumLibrary: Element not interactable / not found
    HintRule(
        id="selenium-element-interactable",
        title="Selenium element not interactable or not visible yet",
        error_pattern=r"(ElementNotInteractableException|ElementClickInterceptedException|Element with locator .* not found)",
        keyword_pattern=r".*",
        message=(
            "Wait for the element to be present and visible before interacting, and verify the locator prefix (`id:`, `css:`, `xpath:`)."
        ),
        fix_examples=[
            "Wait Until Element Is Visible    css:button.submit    timeout=10s",
            "Scroll Element Into View    css:button.submit",
            "Click Element    css:button.submit",
        ],
        skills=[
            "skill://rf-selenium/references/waiting-strategies.md",
            "skill://rf-locators/references/selenium.md",
        ],
        relevance=86,
    ),
    # 11. AppiumLibrary: Mobile element / session failure
    HintRule(
        id="appium-locator-or-source",
        title="Appium mobile element lookup or session error",
        error_pattern=r"(An element could not be located on the page using the given search parameters|InvalidSelectorException)",
        keyword_pattern=r".*",
        message=(
            "Wait for the mobile view to settle using `Wait Until Page Contains Element`, inspect the hierarchy with `Get Source`, "
            "and prefer `accessibility_id=<id>` over brittle XPath locators."
        ),
        fix_examples=[
            "Wait Until Page Contains Element    accessibility_id=login_button    timeout=10s",
            "${xml}=    Get Source",
        ],
        skills=[
            "skill://rf-appium/references/locators-mobile.md",
            "skill://rf-locators/references/appium.md",
        ],
        relevance=85,
    ),
    # 12. PlatynUI: Desktop control lookup
    HintRule(
        id="platynui-control-lookup",
        title="PlatynUI desktop control not found",
        error_pattern=r"(PlatynUI|Control .* not found|Desktop element .* not found)",
        keyword_pattern=r".*",
        message=(
            "Verify that the target desktop application window is open and focused, and check the PlatynUI role/name locator."
        ),
        fix_examples=[
            "robotcode doc keywords -i PlatynUI",
        ],
        skills=[
            "skill://rf-platynui/references/locators-and-queries.md",
            "skill://rf-locators/references/platynui.md",
        ],
        relevance=84,
    ),
    # 13. Variable not found / Shell ${var} expansion
    HintRule(
        id="rf-variable-not-found",
        title="Robot Framework variable not found (or expanded by shell)",
        error_pattern=r"(Variable '\$\{[^}]+\}' not found|Keyword name cannot be empty)",
        keyword_pattern=r".*",
        message=(
            "Check defined variables in the current session with `.vars --user`. "
            "If running via a shell CLI, always wrap Robot Framework statements in single quotes `'...'` "
            "so PowerShell/Bash does not expand `${var}` to an empty string before `robotcode` sees it."
        ),
        fix_examples=[
            ".vars --user",
            "${my_var}=    Set Variable    expected_value",
            "robotcode -f json session send <id> 'Log    ${my_var}'",
        ],
        relevance=89,
    ),
    # 13b. Static analysis: variable not defined in the file's scope
    HintRule(
        id="rf-variable-not-found-static",
        title="Variable is not defined where it is used",
        error_pattern=r"Variable '[\$\@\&%]\{[^}]+\}' not found",
        keyword_pattern=r".*",
        message=(
            "Define the variable in `*** Variables ***`, assign it earlier in the test or keyword, pass it as an "
            "argument, or import the resource/variable file that defines it. Variables set at run time "
            "(`Set Suite Variable`, `-v`) are not visible to static analysis; check with `rf_run` before changing code."
        ),
        fix_examples=[
            "*** Variables ***\n${BASE_URL}    https://example.com",
            "Resource    resources/common.resource",
        ],
        relevance=89,
        applies_to=["static"],
    ),
    # 14. Missing keyword / KeywordNotFound
    HintRule(
        id="rf-keyword-not-found",
        title="Keyword not found — import the library/resource or search with rf_doc / .kw",
        error_pattern=r"No keyword with name '([^']+)' found",
        keyword_pattern=r".*",
        message=(
            "The keyword is either not imported in the current scope or misspelled. "
            "Search available keywords with `.kw <pattern>` inside the session or `rf_doc` (`robotcode doc keywords <pattern>`), "
            "then import the library or resource with `Import Library` / `Import Resource`."
        ),
        fix_examples=[
            ".kw <pattern>",
            "Import Library    Collections",
            "Import Resource    resources/common.resource",
        ],
        relevance=80,
        applies_to=["runtime", "static"],
    ),
    # 15. Suite contains no tests after model modifiers (-bl longname mismatch)
    HintRule(
        id="rf-no-tests-after-modifiers",
        title="No tests matched -bl (by_longname) or tag filters",
        error_pattern=r"contains no tests after model modifiers",
        keyword_pattern=r".*",
        message=(
            "`-bl` (`by_longname`) performs an exact match against the full suite/test `longname` "
            "(including the top-level directory suite name when running against `.` or `paths`). "
            "Run `rf_discover(action='tests')` (`robotcode discover tests`) first and pass the exact `longname` it reports. "
            "If you also passed a single file in `paths`, that file became the top suite and its longnames lost "
            "their parent prefix: drop `paths`."
        ),
        fix_examples=[
            "rf_discover(action='tests')",
            "rf_run(by_longname=['Tests.Orders.Order Total Fails'])",
        ],
        relevance=94,
    ),
    # 16. robotcode executable not on global PATH in client mode
    HintRule(
        id="rf-robotcode-not-on-path",
        title="`robotcode` CLI is in a local virtualenv — prefix command with `uv run`",
        error_pattern=r"(The term 'robotcode' is not recognized|'robotcode' is not recognized|command not found:\s*robotcode|No such file or directory.*robotcode)",
        keyword_pattern=r".*",
        message=(
            "`robotcode` is installed in the project's Python/uv virtual environment rather than on the global system PATH. "
            "Prefix the `robotcode` command with `uv run ` (or `.venv/Scripts/robotcode`) when executing locally in `client` mode."
        ),
        fix_examples=[
            "uv run robotcode -f json discover tests",
            "uv run robotcode robot -i smoke",
        ],
        relevance=97,
    ),
    # 17. Bare debugger command sent without leading dot in robotcode debug session
    HintRule(
        id="rf-debug-dot-command",
        title="RobotCode debugger commands start with a dot (`.c`, `.n`, `.s`, `.o`, `.where`, `.vars`)",
        error_pattern=r"No keyword with name '(c|continue|n|next|s|step|o|out|w|where|l|list|vars|abort|q|quit)' found",
        keyword_pattern=r".*",
        message=(
            "In `robotcode` debug sessions, bare input is evaluated as a Robot Framework keyword, "
            "while debugger control commands start with a dot (`.c`, `.n`, `.s`, `.o`, `.where`, `.list`, `.vars --user`, `.abort`)."
        ),
        fix_examples=[
            ".c",
            ".n",
            ".s",
            ".where",
            ".vars --user",
            ".abort",
        ],
        relevance=96,
    ),
    # --- Rules carried over from the legacy 0.36.0 hint engine (src/robotmcp/utils/hints.py) ---
    # 18. Invalid selector syntax (CSS / XPath parse error)
    HintRule(
        id="locator-invalid-selector",
        title="Invalid selector syntax",
        error_pattern=r"invalid selector|not a valid XPath expression|Unexpected token|Unsupported token|selector syntax error",
        message=(
            "The locator expression has a syntax error; waiting or retrying will not help. Fix the CSS or XPath, "
            "or switch strategy: Browser `css=`, `xpath=`, `text=`, `id=`; SeleniumLibrary `css:`, `xpath:`, `id:`."
        ),
        fix_examples=[
            "Click    css=button[type='submit']",
            "Click Element    xpath://button[normalize-space()='Save']",
        ],
        relevance=90,
    ),
    # 19. Browser: SeleniumLibrary-style role prefix (`button=Save`)
    HintRule(
        id="browser-role-prefix-misuse",
        title="Browser library does not support `button=` / `link=` style prefixes",
        arg_pattern=r"^\S.*?\s{2,}(button|link|input|select|textarea|checkbox|radio|heading)\s*=\s*\S",
        library_pattern=r"\bBrowser\b|waiting for locator|Unsupported token|Unknown engine|strict mode",
        message=(
            "`button=Label` is SeleniumLibrary-style syntax. Browser selector prefixes are `css=`, `xpath=`, `text=`, "
            "`id=` and Playwright engines; `button \"Label\"` in an ARIA snapshot is role + name, not a locator."
        ),
        fix_examples=[
            "Click    text=Calculate",
            "Click    css=button:text-is('Calculate')",
            "Click    role=button[name=\"Calculate\"]",
        ],
        skills=[
            "skill://rf-browser/references/locators.md",
            "skill://rf-locators/references/browser.md",
        ],
        relevance=92,
    ),
    # 20. Element outside the viewport
    HintRule(
        id="element-outside-viewport",
        title="Element is outside the visible viewport",
        error_pattern=r"element is outside of the viewport|outside.*viewport",
        message=(
            "The element exists but is off-screen (often a hidden input styled by CSS). Scroll it into view, click "
            "its visible label instead, or force the click. On mobile (AppiumLibrary), scroll or swipe to it."
        ),
        fix_examples=[
            "Scroll To Element    <selector>",
            "Click With Options    <selector>    force=True",
            "Scroll Down    <locator>",
        ],
        relevance=88,
    ),
    # 21. Stale element / detached from the DOM
    HintRule(
        id="element-stale",
        title="Stale element - the DOM was rebuilt",
        error_pattern=(
            r"stale element reference|element is not attached|not attached to the page document|"
            r"element reference.*is stale|expired from the internal cache"
        ),
        message=(
            "The element was removed and re-added (navigation, AJAX update, SPA route change). Wait for it again "
            "and locate it fresh instead of reusing an old reference."
        ),
        fix_examples=[
            "Wait For Elements State    <selector>    attached    timeout=10s",
            "Wait Until Element Is Visible    <locator>    timeout=10s",
        ],
        relevance=85,
    ),
    # 22. Unexpected alert / system dialog
    HintRule(
        id="unexpected-alert",
        title="An alert or dialog is blocking interaction",
        error_pattern=r"unexpected alert open|UnexpectedAlertPresentException|alert.*present",
        message=(
            "A JavaScript alert, confirm or prompt (or a mobile permission dialog) blocks every command. Accept or "
            "dismiss it first. Browser library: register `Handle Future Dialogs` before the action that opens it."
        ),
        fix_examples=[
            "Handle Alert    ACCEPT",
            "Handle Future Dialogs    action=accept",
            "Click Alert Button    Allow",
        ],
        relevance=85,
    ),
    # 23. Frame not found / detached
    HintRule(
        id="frame-not-found",
        title="Frame not found or detached",
        error_pattern=r"no such frame|NoSuchFrameException|frame was detached|ERR_ABORTED.*frame",
        message=(
            "The frame does not exist yet or was replaced. SeleniumLibrary: return to the main document with "
            "`Unselect Frame`, then `Select Frame` again. Browser: address the frame with `>>>` "
            "(`iframe#pay >>> css=button`) after the page settles."
        ),
        fix_examples=[
            "Unselect Frame",
            "Select Frame    css:iframe#payment",
            "Click    css=iframe#payment >>> css=button.pay",
        ],
        skills=[
            "skill://rf-locators/references/shadow-dom-iframes.md",
        ],
        relevance=80,
    ),
    # 24. Window or tab not found
    HintRule(
        id="window-not-found",
        title="Window or tab not found",
        error_pattern=r"no such window|NoSuchWindowException",
        message="The window or tab was closed or never existed. Switch to the main window or the new one.",
        fix_examples=[
            "Switch Window    MAIN",
            "Switch Window    NEW",
        ],
        relevance=80,
    ),
    # 25. Process: dash-prefixed argument with `=` is parsed as a named argument
    HintRule(
        id="process-named-arg-misparse",
        title="Process argument may be parsed as a named argument",
        keyword_pattern=r"^(Process\.)?(Start|Run) Process$",
        arg_pattern=r"(^|\s{2,})-[^\s=]*[^\s=\\]=",
        message=(
            "Robot Framework parses an argument like `-env:UserInstallation=file:///tmp` as the named argument "
            "`-env:UserInstallation` and drops it from the command line. Escape the first `=` as `\\=` so it stays "
            "a positional command-line argument."
        ),
        fix_examples=[
            "Start Process    soffice    -env:UserInstallation\\=file:///tmp/profile",
        ],
        relevance=85,
    ),
    # 26. AppiumLibrary: command not available in the current context
    HintRule(
        id="mobile-context-mismatch",
        title="Wrong context for the mobile command",
        error_pattern=r"UnknownCommandError|unknown command",
        library_pattern=r"Appium",
        message=(
            "Hybrid apps have NATIVE_APP and WEBVIEW contexts and each command works in only one of them. Check the "
            "current context and switch."
        ),
        fix_examples=[
            "Get Current Context",
            "Get Contexts",
            "Switch To Context    NATIVE_APP",
        ],
        skills=[
            "skill://rf-appium/references/troubleshooting.md",
            "skill://rf-locators/references/appium.md",
        ],
        relevance=85,
    ),
]

