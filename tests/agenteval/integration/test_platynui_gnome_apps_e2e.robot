*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_platynui_gnome_apps_e2e.py: live GNOME Calculator +
...              Text Editor desktop automation via PlatynUI (ADR-025). Real MCP protocol;
...              deterministic. DESKTOP-GATED: set AGENTEVAL_DESKTOP=1 to run.
Resource         rfmcp.resource
Library          Process
Library          OperatingSystem
Library          Collections
Suite Setup      Setup Desktop Suite
Suite Teardown   Teardown Desktop Suite

*** Variables ***
${E2E_DISPLAY}          :99
${CALC_APP}             /app:*[@Name='gnome-calculator']
${EDITOR_APP}           /app:*[@Name='gnome-text-editor']
${EDITOR_FILE_NAME}     platynui-e2e.txt
${EDITOR_FILE}          ${EMPTY}

*** Test Cases ***
# ===========================================================================
# Calculator - every test asserts a DISTINCT value (history-pollution-proof)
# ===========================================================================
Multiplication 7x8
    [Documentation]    Canonical scenario from the ADR-025 agent validation.
    Init Session    calc-mul
    Verified Calc Sequence    calc-mul    ${CALC_APP}    ${{ ['C', '7', '×', '8', '='] }}    56
    Assert History Result    calc-mul    ${CALC_APP}    56    7×8

Addition Multi Digit
    [Documentation]    Sequential multi-digit entry: 12 + 34 = 46.
    Init Session    calc-add
    Verified Calc Sequence    calc-add    ${CALC_APP}    ${{ ['C', '1', '2', '+', '3', '4', '='] }}    46
    Assert History Result    calc-add    ${CALC_APP}    46    12+34

Division Integer
    [Documentation]    54 / 6 = 9 (integer result avoids locale decimal separators).
    Init Session    calc-div
    Verified Calc Sequence    calc-div    ${CALC_APP}    ${{ ['C', '5', '4', '÷', '6', '='] }}    9
    Assert History Result    calc-div    ${CALC_APP}    9    54÷6

Subtraction
    [Documentation]    100 - 13 = 87 (uses the U+2212 minus glyph from the keypad).
    Init Session    calc-sub
    Verified Calc Sequence    calc-sub    ${CALC_APP}    ${{ ['C', '1', '0', '0', '−', '1', '3', '='] }}    87
    Assert History Result    calc-sub    ${CALC_APP}    87    100−13

Chained Operations
    [Documentation]    2 x 3 x 4 = 24 - intermediate '=' free chaining.
    Init Session    calc-chain
    Verified Calc Sequence    calc-chain    ${CALC_APP}    ${{ ['C', '2', '×', '3', '×', '4', '='] }}    24
    Assert History Result    calc-chain    ${CALC_APP}    24    2×3×4

Keyboard Entry
    [Documentation]    Keyboard path: click entry to focus, type '9*9<Return>' -> 81.
    Init Session    calc-kbd
    Click Element    calc-kbd    ${CALC_APP}//control:Button[@Name='C']
    ${len0}=    Calc Entry Length    calc-kbd    ${CALC_APP}
    Should Be Equal As Integers    ${len0}    0
    Click Element    calc-kbd    ${CALC_APP}//control:Text
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Keyboard Type', 'arguments': ['\${None}', '9*9<Return>'], 'session_id': 'calc-kbd'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${len1}=    Calc Entry Length    calc-kbd    ${CALC_APP}
    Should Be Equal As Integers    ${len1}    2
    Assert History Result    calc-kbd    ${CALC_APP}    81    9×9

Intent Action Click
    [Documentation]    intent_action(click) resolves to PlatynUI Pointer Click and really clicks: 5 x 5 = 25.
    Init Session    calc-intent
    ${expected_len}=    Set Variable    ${None}
    FOR    ${name}    IN    @{{ ['C', '5', '×', '5', '='] }}
        ${target}=    Set Variable    ${CALC_APP}//control:Button[@Name='${name}']
        ${r}=    Rf Tool    ${HANDLE}    intent_action
        ...    ${{ {'intent': 'click', 'target': $target, 'session_id': 'calc-intent'} }}
        Result Field Should Be    ${r}    success    ${True}
        Should Be True    $r.get('keyword') in ('Pointer Click', None)
        IF    $name == 'C'
            ${expected_len}=    Set Variable    ${0}
        ELSE IF    $name == '='
            ${expected_len}=    Evaluate    len('25')
        ELSE
            ${expected_len}=    Evaluate    $expected_len + len($name)
        END
        ${actual}=    Calc Entry Length    calc-intent    ${CALC_APP}
        Should Be Equal As Integers    ${actual}    ${expected_len}
    END
    Assert History Result    calc-intent    ${CALC_APP}    25    5×5

Ui Tree Exposes Buttons
    [Documentation]    get_session_state ui_tree expands the calculator subtree.
    Init Session    calc-tree
    ${state}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': 'calc-tree', 'sections': ['ui_tree'], 'elements_of_interest': ['gnome-calculator']} }}
    ${ui_tree}=    Evaluate    $state['sections']['ui_tree']
    Result Field Should Be    ${ui_tree}    success    ${True}
    ${exp_apps}=    Evaluate    $ui_tree.get('expanded_applications', 0)
    Should Be True    ${exp_apps} >= 1
    ${expanded}=    Evaluate    [a for a in $ui_tree['applications'] if a.get('expanded')]
    Should Be True    len($expanded) >= 1
    ${roles}=    Collect Roles    ${expanded}
    Should Contain    ${roles}    Frame
    Should Be True    any(a.get('children') for a in $expanded)

Suite Generation No Browser Teardown
    [Documentation]    Desktop suites must not get a 'Close Browser' teardown (ADR-025 test_builder fix).
    Init Session    calc-suite
    Verified Calc Sequence    calc-suite    ${CALC_APP}    ${{ ['C', '8', '×', '9', '='] }}    72
    Assert History Result    calc-suite    ${CALC_APP}    72    8×9
    ${suite}=    Rf Tool    ${HANDLE}    build_test_suite
    ...    ${{ {'session_id': 'calc-suite', 'test_name': 'Calculator 8x9'} }}
    Result Field Should Be    ${suite}    success    ${True}
    ${rf_text}=    Resolve Rf Text    ${suite}
    Should Not Be Empty    ${rf_text}
    Should Contain    ${rf_text}    PlatynUI.BareMetal
    Should Contain    ${rf_text}    Pointer Click
    Should Not Contain    ${rf_text}    Close Browser

# ===========================================================================
# Text editor - GTK4 text content is NOT AT-SPI readable; assert via
# CharacterCount and on-disk save roundtrips.
# ===========================================================================
Window Title Contains Filename
    Init Session    ed-title
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Get Attribute', 'arguments': [$EDITOR_APP + '//control:Frame', 'Name'], 'session_id': 'ed-title', 'assign_to': 'title'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${title}=    Evaluate    str($r.get('assigned_variables', {}).get('\${title}', ''))
    Should Be True    $EDITOR_FILE_NAME in $title or 'Text Editor' in $title

Type Text Character Count
    [Documentation]    Replace seed content; verify typed length via native Text.CharacterCount.
    Init Session    ed-type
    Editor Type    ed-type    <Ctrl+A><Delete>Hello PlatynUI
    ${count}=    Editor Char Count    ed-type    count
    Should Be Equal As Integers    ${count}    ${{ len('Hello PlatynUI') }}

Select All Delete Empties Buffer
    Init Session    ed-clear
    ${baseline}=    Editor Char Count    ed-clear    baseline
    Editor Type    ed-clear    scratch content
    ${typed}=    Editor Char Count    ed-clear    typed_count
    Should Be Equal As Integers    ${typed}    ${{ $baseline + len('scratch content') }}
    Editor Type    ed-clear    <Ctrl+A><Delete>
    ${count}=    Editor Char Count    ed-clear    empty_count
    Should Be Equal As Integers    ${count}    0

Undo Restores Text
    Init Session    ed-undo
    Editor Type    ed-undo    <Ctrl+A><Delete>undo target
    ${before}=    Editor Char Count    ed-undo    before
    Should Be Equal As Integers    ${before}    ${{ len('undo target') }}
    Editor Type    ed-undo    <Ctrl+A><Delete>
    ${deleted}=    Editor Char Count    ed-undo    deleted
    Should Be Equal As Integers    ${deleted}    0
    Editor Type    ed-undo    <Ctrl+Z>
    ${after}=    Editor Char Count    ed-undo    after
    Should Be Equal As Integers    ${after}    ${before}

Save Roundtrip To Disk
    [Documentation]    Type unique text, Ctrl+S (in-place save), assert the bytes reached disk.
    Init Session    ed-save
    ${marker}=    Set Variable    saved-by-platynui-${{ __import__('uuid').uuid4().hex[:8] }}
    Editor Type    ed-save    <Ctrl+A><Delete>${marker}
    ${typed}=    Editor Char Count    ed-save    marker_count
    Should Be Equal As Integers    ${typed}    ${{ len($marker) }}
    Editor Type    ed-save    <Ctrl+S>
    ${content}=    Set Variable    ${EMPTY}
    FOR    ${i}    IN RANGE    20
        ${content}=    Get File    ${EDITOR_FILE}
        ${found}=    Evaluate    $marker in $content
        IF    $found    BREAK
        Sleep    0.5s
    END
    Should Contain    ${content}    ${marker}

# ===========================================================================
# Cross-application scoping
# ===========================================================================
Scoped Queries Disambiguate Apps
    [Documentation]    With both apps alive, app-scoped Frame queries resolve per process.
    Init Session    xapp-scope
    FOR    ${app}    IN    ${CALC_APP}    ${EDITOR_APP}
        ${r}=    Rf Tool    ${HANDLE}    execute_step
        ...    ${{ {'keyword': 'Get Attribute', 'arguments': [$app + '//control:Frame', 'native:Accessible.Application'], 'session_id': 'xapp-scope', 'assign_to': 'appref'} }}
        ${ok}=    Evaluate    $r.get('success') is True
        IF    not $ok
            ${r}=    Rf Tool    ${HANDLE}    execute_step
            ...    ${{ {'keyword': 'Get Attribute', 'arguments': [$app + '//control:Frame', 'Name'], 'session_id': 'xapp-scope', 'assign_to': 'appref'} }}
        END
        Result Field Should Be    ${r}    success    ${True}
    END

Ui Tree Lists Both Apps
    Init Session    xapp-tree
    ${state}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': 'xapp-tree', 'sections': ['ui_tree']} }}
    ${ui_tree}=    Evaluate    $state['sections']['ui_tree']
    Result Field Should Be    ${ui_tree}    success    ${True}
    ${names}=    Evaluate    [a.get('name') for a in $ui_tree['applications']]
    Should Contain    ${names}    gnome-calculator
    Should Contain    ${names}    gnome-text-editor

Calculator Then Editor In One Session
    [Documentation]    One MCP session driving two desktop apps back-to-back: 6 x 7 = 42, then type in editor.
    Init Session    xapp-flow
    Raise X11 Window    Calculator
    Verified Calc Sequence    xapp-flow    ${CALC_APP}    ${{ ['C', '6', '×', '7', '='] }}    42
    Assert History Result    xapp-flow    ${CALC_APP}    42    6×7
    Raise X11 Window    platynui-e2e.txt
    Editor Type    xapp-flow    <Ctrl+A><Delete>answer is 42
    ${count}=    Editor Char Count    xapp-flow    flow_count
    Should Be Equal As Integers    ${count}    ${{ len('answer is 42') }}

# ===========================================================================
# Maintainer-report workflow end-to-end (change: desktop-mcp-workflow-correctness)
# ===========================================================================
Full Report Flow Reaches Real Interactions
    # 1. analyze_scenario(context=desktop) -> desktop session, PlatynUI-led, no Appium.
    ${analysis}=    Rf Tool    ${HANDLE}    analyze_scenario
    ...    ${{ {'scenario': 'Open GNOME Calculator desktop application, perform several calculations and assert each entered value and the result', 'context': 'desktop', 'session_id': 'report-flow'} }}
    Result Field Should Be    ${analysis}    success    ${True}
    ${caps}=    Evaluate    list($analysis['scenario']['required_capabilities'])
    Should Contain    ${caps}    PlatynUI.BareMetal
    Should Not Contain    ${caps}    AppiumLibrary
    ${dst}=    Evaluate    $analysis['analysis']['detected_session_type']
    Should Be Equal    ${dst}    desktop_testing
    # 2. find_keywords surfaces PlatynUI desktop keywords.
    ${kws}=    Rf Tool    ${HANDLE}    find_keywords
    ...    ${{ {'query': '', 'strategy': 'catalog', 'library_name': 'PlatynUI', 'session_id': 'report-flow'} }}
    Result Field Should Be    ${kws}    library    PlatynUI.BareMetal
    ${kw_names}=    Evaluate    [c['name'] for c in $kws['results']]
    Should Contain    ${kw_names}    Pointer Click
    # 3. get_session_state must inspect ui_tree, never the mobile source.
    ${state}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': 'report-flow', 'sections': ['page_source'], 'elements_of_interest': ['gnome-calculator']} }}
    ${ps}=    Evaluate    $state['sections']['page_source']
    Result Field Should Be    ${ps}    source    desktop
    ${msg}=    Evaluate    ($ps.get('message') or '').lower()
    Should Not Contain    ${msg}    mobile source
    ${has_ui}=    Evaluate    'ui_tree' in $state['sections']
    Should Be True    ${has_ui}
    # 4. stepwise interactions with per-entry + result assertions.
    Raise X11 Window    Calculator
    Verified Calc Sequence    report-flow    ${CALC_APP}    ${{ ['C', '3', '×', '9', '='] }}    27
    Assert History Result    report-flow    ${CALC_APP}    27    3×9
    # 5. build_test_suite reflects real interactions, reports pre-start accounting.
    ${suite}=    Rf Tool    ${HANDLE}    build_test_suite
    ...    ${{ {'session_id': 'report-flow', 'test_name': 'Calculator 3x9'} }}
    Result Field Should Be    ${suite}    success    ${True}
    ${has_excl}=    Evaluate    'excluded_pre_start_count' in $suite
    Should Be True    ${has_excl}
    ${rf_text}=    Resolve Rf Text    ${suite}
    Should Contain    ${rf_text}    Pointer Click
    Should Not Contain    ${rf_text}    Close Browser

Execute Batch Preserves Arguments
    [Documentation]    execute_batch must honor the canonical 'arguments' key (finding #8).
    Init Session    report-batch
    ${batch}=    Rf Tool    ${HANDLE}    execute_batch
    ...    ${{ {'session_id': 'report-batch', 'steps': [{'keyword': 'Pointer Click', 'arguments': [$CALC_APP + "//control:Button[@Name='C']"]}, {'keyword': 'Pointer Click', 'arguments': [$CALC_APP + "//control:Button[@Name='4']"]}]} }}
    Should Be True    $batch.get('status') in ('PASS', 'RECOVERED')

Generated Suite Is Clean Not A Debug Trace
    [Documentation]    build_test_suite serializes validated intent, not investigation history.
    Init Session    report-clean
    Raise X11 Window    Calculator
    # Mix exploratory introspection probes with real interactions.
    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Query', 'arguments': ["//app:*[@Name='gnome-calculator']"], 'session_id': 'report-clean', 'assign_to': 'probe_nodes'} }}
    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Evaluate', 'arguments': ['[1, 2, 3]'], 'session_id': 'report-clean', 'assign_to': 'probe_list'} }}
    Verified Calc Sequence    report-clean    ${CALC_APP}    ${{ ['C', '8', '×', '9', '='] }}    72
    Assert History Result    report-clean    ${CALC_APP}    72    8×9
    ${suite}=    Rf Tool    ${HANDLE}    build_test_suite
    ...    ${{ {'session_id': 'report-clean', 'test_name': 'Calculator Clean 8x9'} }}
    Result Field Should Be    ${suite}    success    ${True}
    ${filtered}=    Evaluate    $suite.get('introspection_filtered_count', 0)
    Should Be True    ${filtered} >= 1
    ${rf_text}=    Resolve Rf Text    ${suite}
    Should Contain    ${rf_text}    Pointer Click
    Should Not Contain    ${rf_text}    [1, 2, 3]

*** Keywords ***
Setup Desktop Suite
    Skip If    '%{AGENTEVAL_DESKTOP=}' != '1'    Needs a desktop + GNOME apps - set AGENTEVAL_DESKTOP=1
    # Environment must be fixed BEFORE the rf-mcp server subprocess starts so it
    # inherits it (the PlatynUI Rust core caches the session type once per process).
    Set Environment Variable    XDG_SESSION_TYPE    x11
    Set Environment Variable    DISPLAY    ${E2E_DISPLAY}
    Remove Environment Variable    WAYLAND_DISPLAY
    Set Environment Variable    ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY    ${E2E_DISPLAY}
    Ensure Xvfb
    # Pre-created file so Ctrl+S saves in place (no save dialog).
    ${dir}=    Join Path    ${TEMPDIR}    rfmcp-e2e-editor
    Create Directory    ${dir}
    ${file}=    Join Path    ${dir}    ${EDITOR_FILE_NAME}
    Create File    ${file}    seed\n
    Set Suite Variable    ${EDITOR_FILE}    ${file}
    Start Rf-mcp Server
    Systemd Run    rfmcp-e2e-calc    /usr/bin/gnome-calculator
    Systemd Run    rfmcp-e2e-editor    /usr/bin/gnome-text-editor    ${EDITOR_FILE}
    Wait For App Window    Calculator
    Wait For App Window    ${EDITOR_FILE_NAME}

Teardown Desktop Suite
    Run Keyword And Ignore Error    Systemd Stop    rfmcp-e2e-calc
    Run Keyword And Ignore Error    Systemd Stop    rfmcp-e2e-editor
    Run Keyword And Ignore Error    Stop Rf-mcp Server
    Run Keyword And Ignore Error    Terminate Process    xvfb

Ensure Xvfb
    ${status}=    Run Keyword And Return Status    Xdpyinfo Ready
    IF    not ${status}
        Start Process    Xvfb    ${E2E_DISPLAY}    -screen    0    1280x1024x24    alias=xvfb
        Wait Until Keyword Succeeds    20x    0.5s    Xdpyinfo Ready
    END

Xdpyinfo Ready
    ${r}=    Run Process    xdpyinfo
    Should Be Equal As Integers    ${r.rc}    0

Systemd Run
    [Arguments]    ${unit}    @{cmd}
    Systemd Stop    ${unit}
    Run Process    systemd-run    --user    --unit\=${unit}
    ...    --setenv\=DISPLAY\=${E2E_DISPLAY}    --setenv\=GDK_BACKEND\=x11    --setenv\=GSK_RENDERER\=cairo
    ...    @{cmd}

Systemd Stop
    [Arguments]    ${unit}
    Run Process    systemctl    --user    stop    ${unit}.service
    Run Process    systemctl    --user    reset-failed    ${unit}.service

Wait For App Window
    [Arguments]    ${title}
    ${status}=    Run Keyword And Return Status    Wait Until Keyword Succeeds    20x    1s    X11 Window Present    ${title}
    IF    not ${status}    Skip    App window '${title}' did not appear on AT-SPI/X11

X11 Window Present
    [Arguments]    ${title}
    ${r}=    Run Process    xwininfo    -root    -tree
    Should Contain    ${r.stdout}    ${title}

Raise X11 Window
    [Documentation]    Raise the X11 window whose xwininfo tree entry matches ${title}
    ...                (WM-less Xvfb: restack via XRaiseWindow, ignoring 1x1 utility windows).
    [Arguments]    ${title}
    ${code}=    Catenate    SEPARATOR=${\n}
    ...    import ctypes,os,subprocess,sys,time
    ...    t\=sys.argv[1]; d\=":99"
    ...    out\=subprocess.run(["xwininfo","-root","-tree"],capture_output\=True,text\=True,env\=dict(os.environ,DISPLAY\=d)).stdout
    ...    wid\=next((int(l.strip().split()[0],16) for l in out.splitlines() if t in l and '"' in l and " 1x1+" not in l), None)
    ...    assert wid is not None, "X11 window not found: "+t
    ...    x\=ctypes.CDLL("libX11.so.6"); dp\=x.XOpenDisplay(d.encode()); assert dp; x.XRaiseWindow(dp,wid); x.XFlush(dp); x.XCloseDisplay(dp); time.sleep(0.5)
    ${res}=    Run Process    python3    -c    ${code}    ${title}
    Should Be Equal As Integers    ${res.rc}    0    Raise window failed: ${res.stderr}

Init Session
    [Arguments]    ${sid}
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'session_id': $sid, 'action': 'init', 'scenario': 'Native desktop automation of GNOME apps with PlatynUI', 'libraries': ['PlatynUI.BareMetal', 'BuiltIn']} }}
    Result Field Should Be    ${init}    success    ${True}

Click Element
    [Arguments]    ${sid}    ${locator}
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Pointer Click', 'arguments': [$locator], 'session_id': $sid} }}
    Result Field Should Be    ${r}    success    ${True}

Calc Entry Length
    [Documentation]    Displayed-entry length proxy via native:Text.CharacterCount (GTK4 hides content).
    [Arguments]    ${sid}    ${app}
    ${loc}=    Set Variable    ${app}//control:Text
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Get Attribute', 'arguments': [$loc, 'native:Text.CharacterCount'], 'session_id': $sid, 'assign_to': 'calc_len'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${val}=    Evaluate    int(str($r.get('assigned_variables', {}).get('\${calc_len}')))
    RETURN    ${val}

Verified Calc Sequence
    [Documentation]    Click a calculator sequence with a per-action display assertion.
    [Arguments]    ${sid}    ${app}    ${buttons}    ${result}
    ${expected_len}=    Set Variable    ${None}
    FOR    ${name}    IN    @{buttons}
        ${btn}=    Set Variable    ${app}//control:Button[@Name='${name}']
        Click Element    ${sid}    ${btn}
        IF    $name == 'C'
            ${expected_len}=    Set Variable    ${0}
        ELSE IF    $name == '='
            ${expected_len}=    Evaluate    len($result)
        ELSE IF    $expected_len is not None
            ${expected_len}=    Evaluate    $expected_len + len($name)
        END
        IF    $expected_len is not None
            ${actual}=    Calc Entry Length    ${sid}    ${app}
            Should Be Equal As Integers    ${actual}    ${expected_len}
        END
    END

Assert History Result
    [Documentation]    Calculator results appear in history as Labels (equation + result value).
    [Arguments]    ${sid}    ${app}    ${expected}    ${equation}=${None}
    ${label}=    Set Variable    ${app}//control:Label[@Name='${expected}']
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Get Attribute', 'arguments': [$label, 'Name'], 'session_id': $sid, 'assign_to': 'result'} }}
    Result Field Should Be    ${r}    success    ${True}
    ${check}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Should Be Equal As Strings', 'arguments': ['\${result}', $expected], 'session_id': $sid} }}
    Result Field Should Be    ${check}    success    ${True}
    IF    $equation is not None
        ${eq_label}=    Set Variable    ${app}//control:Label[@Name='${equation}']
        ${eq}=    Rf Tool    ${HANDLE}    execute_step
        ...    ${{ {'keyword': 'Get Attribute', 'arguments': [$eq_label, 'Name'], 'session_id': $sid, 'assign_to': 'equation'} }}
        Result Field Should Be    ${eq}    success    ${True}
        ${eq_check}=    Rf Tool    ${HANDLE}    execute_step
        ...    ${{ {'keyword': 'Should Be Equal As Strings', 'arguments': ['\${equation}', $equation], 'session_id': $sid} }}
        Result Field Should Be    ${eq_check}    success    ${True}
    END

Editor Char Count
    [Arguments]    ${sid}    ${var}
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Get Attribute', 'arguments': [$EDITOR_APP + '//control:Text', 'native:Text.CharacterCount'], 'session_id': $sid, 'assign_to': $var} }}
    Result Field Should Be    ${r}    success    ${True}
    ${val}=    Evaluate    int(str($r.get('assigned_variables', {}).get('\${' + $var + '}')))
    RETURN    ${val}

Editor Type
    [Documentation]    Click into the text view (sets X input focus via PointerRoot), then type.
    [Arguments]    ${sid}    ${sequence}
    Click Element    ${sid}    ${EDITOR_APP}//control:Text
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Keyboard Type', 'arguments': ['\${None}', $sequence], 'session_id': $sid} }}
    Result Field Should Be    ${r}    success    ${True}

Collect Roles
    [Documentation]    Flatten every node role reachable from the given application nodes.
    [Arguments]    ${apps}
    ${roles}=    Create List
    ${stack}=    Evaluate    list($apps)
    WHILE    len($stack) > 0    limit=100000
        ${node}=    Evaluate    $stack.pop()
        ${role}=    Evaluate    $node.get('role')
        Append To List    ${roles}    ${role}
        ${stack}=    Evaluate    $stack + list($node.get('children') or [])
    END
    RETURN    ${roles}

Resolve Rf Text
    [Documentation]    Follow the ADR-015 artifact pointer if the suite response was externalized.
    [Arguments]    ${data}
    ${rf_text}=    Evaluate    $data.get('rf_text') or ''
    ${externalized}=    Evaluate    'Content saved to ' in $rf_text and '.robotmcp_artifacts' in $rf_text
    IF    $externalized
        ${artifact}=    Evaluate    $rf_text.split('Content saved to ', 1)[1].split(' (', 1)[0]
        ${rf_text}=    Get File    ${artifact}
    END
    RETURN    ${rf_text}
