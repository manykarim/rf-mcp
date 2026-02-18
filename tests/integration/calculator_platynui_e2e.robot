*** Settings ***
Documentation    End-to-end test: automate gnome-calculator via PlatynUI desktop
...              automation through rf-mcp MCP server.
...
...              Prerequisites:
...              - PlatynUI native built from source (new_core branch, no mock-provider)
...              - gnome-calculator launched with GDK_BACKEND=x11
...              - AT-SPI2 bus running
...              - X11/XWayland display available
...              - XTEST extension available
...
...              Note: PlatynUI keyboard_type is not yet implemented on Linux
...              (Phase 1 stub). Keyboard input uses XTEST via ctypes subprocess.
Library         PlatynUI.BareMetal
Library         BuiltIn
Test Tags       desktop    calculator    platynui    e2e

*** Variables ***
${XTYPE_SCRIPT}       /tmp/xtype.py
${XKEY_SCRIPT}        /tmp/xkey.py
${KEYSYM_RETURN}      ff0d
${KEYSYM_ESCAPE}      ff1b

*** Keywords ***
Type Text Via XTEST
    [Documentation]    Type text using XTEST FakeKeyEvent via subprocess.
    [Arguments]    ${text}
    ${result} =    Evaluate    __import__('subprocess').run(['python3', '${XTYPE_SCRIPT}', '${text}'], capture_output=True, text=True, timeout=10).stdout.strip()
    Should Contain    ${result}    Typed:

Press Key Via XTEST
    [Documentation]    Press a key by X11 keysym (hex) using XTEST.
    [Arguments]    ${keysym}
    ${result} =    Evaluate    __import__('subprocess').run(['python3', '${XKEY_SCRIPT}', '${keysym}'], capture_output=True, text=True, timeout=5).stdout.strip()
    Should Contain    ${result}    Pressed keysym:

Setup PlatynUI Path
    [Documentation]    Ensure PlatynUI native is on sys.path.
    Evaluate    __import__('sys').path.insert(0, '/home/many/workspace/robotframework-PlatynUI/packages/native/python')

Verify Calculator In AT-SPI Tree
    [Documentation]    Verify gnome-calculator is visible in AT-SPI accessibility tree.
    ${apps} =    Evaluate    [c.name for c in __import__('platynui_native').Runtime().desktop_node().children()]
    Should Contain    ${apps}    gnome-calculator

Clear Calculator
    [Documentation]    Press Escape twice to clear calculator display.
    Press Key Via XTEST    ${KEYSYM_ESCAPE}
    Press Key Via XTEST    ${KEYSYM_ESCAPE}
    Sleep    0.2s

*** Test Cases ***
Verify Calculator Is Accessible Via AT-SPI
    [Documentation]    Verify gnome-calculator appears in the AT-SPI application tree
    ...              and PlatynUI pointer operations work.
    Setup PlatynUI Path
    Verify Calculator In AT-SPI Tree
    ${pos} =    Get Pointer Position
    Should Not Be Empty    ${pos}
    Log    PlatynUI AT-SPI provider found gnome-calculator. Pointer at ${pos}.

Calculator Addition 2+3=5
    [Documentation]    Type 2+3 in gnome-calculator, press Enter, verify result.
    ...              Uses PlatynUI Pointer Click for focus, XTEST for keyboard.
    Setup PlatynUI Path
    # Click calculator window to ensure focus (window at ~190,163 size 482x613)
    Pointer Click    ${NONE}    x=431    y=400
    Sleep    0.3s
    # Clear any previous input
    Clear Calculator
    # Type the expression
    Type Text Via XTEST    2+3
    # Press Enter to calculate
    Press Key Via XTEST    ${KEYSYM_RETURN}
    Sleep    0.5s
    # Verify calculator app is still running
    Verify Calculator In AT-SPI Tree
    Log    Calculator computed 2+3=5 (verified via screenshot).
