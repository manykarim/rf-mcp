*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_fastmcp_argument_resolution.py: FastMCP argument
...              resolution - named/positional args, object-valued named args preserved
...              end-to-end, and library-prefix routing for overlapping names (XML).
...              Real MCP protocol; deterministic.
Resource         rfmcp.resource
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Test Cases ***
Collections Dictionary Create And Get
    [Documentation]    Create a dictionary via named args, retrieve a value, and assert result.
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'collections_dict_session', 'libraries': ['Collections', 'BuiltIn']} }}
    Result Field Should Be    ${init}    success    ${True}
    ${create}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Create Dictionary', 'arguments': ['a=1', 'b=2'], 'session_id': 'collections_dict_session', 'assign_to': 'd', 'raise_on_failure': True} }}
    Result Field Should Be    ${create}    success    ${True}
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Get From Dictionary', 'arguments': ['\${d}', 'a'], 'session_id': 'collections_dict_session', 'assign_to': 'val', 'raise_on_failure': True} }}
    Result Field Should Be    ${r}    success    ${True}
    IF    $r.get('assigned_variables')
        Should Be True    $r['assigned_variables'].get('\${val}') in ('1', 1)
    ELSE
        Should Be True    $r.get('result') in ('1', 1)
    END

Collections Set To Dictionary Named Object Arg
    [Documentation]    Set To Dictionary should accept object-valued named args preserved via ${var}.
    ${init}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'init', 'session_id': 'collections_setdict_session', 'libraries': ['Collections', 'BuiltIn']} }}
    Result Field Should Be    ${init}    success    ${True}
    ${create}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Create Dictionary', 'arguments': [], 'session_id': 'collections_setdict_session', 'assign_to': 'd'} }}
    Result Field Should Be    ${create}    success    ${True}
    ${set}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Set To Dictionary', 'arguments': ['\${d}', 'a', '42'], 'session_id': 'collections_setdict_session'} }}
    Result Field Should Be    ${set}    success    ${True}
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'Get From Dictionary', 'arguments': ['\${d}', 'a'], 'session_id': 'collections_setdict_session', 'assign_to': 'val'} }}
    Result Field Should Be    ${r}    success    ${True}
    IF    $r.get('assigned_variables') and '\${val}' in $r['assigned_variables']
        Should Be True    str($r['assigned_variables']['\${val}']) == '42'
    ELSE
        Should Be True    str($r.get('output')) == '42'
    END

XML Library Get Element Count With Prefix
    [Documentation]    Use XML.Get Element Count with a file path and XPath; should load XML library automatically.
    ${xml_path}=    Evaluate    os.path.abspath(os.path.join(r'${CURDIR}', '..', '..', '..', 'test_data', 'books_authors.xml'))    modules=os
    ${r}=    Rf Tool    ${HANDLE}    execute_step
    ...    ${{ {'keyword': 'XML.Get Element Count', 'arguments': [$xml_path, './/book'], 'session_id': 'xml_prefix_session', 'assign_to': 'count', 'raise_on_failure': True} }}
    Result Field Should Be    ${r}    success    ${True}
    IF    $r.get('assigned_variables') and '\${count}' in $r['assigned_variables']
        Should Be True    str($r['assigned_variables']['\${count}']).isdigit()
    ELSE
        Should Be True    str($r.get('output')).isdigit()
    END
