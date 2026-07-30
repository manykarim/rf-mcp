*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_variable_file_loading.py: variable-file loading
...              (static/dynamic Python, YAML, JSON, error handling) via manage_session
...              import_variables/load_variables. Real MCP protocol; deterministic.
Resource         rfmcp.resource
Library          OperatingSystem
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Variables ***
${TEST_DATA}     ${CURDIR}${/}..${/}..${/}..${/}test_data

*** Test Cases ***
Static Python Variable File
    ${vf}=    Set Variable    ${TEST_DATA}${/}static_variables.py
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_static_py_vars', 'variable_file_path': $vf} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Field Should Be    ${r}    action    import_variables
    Result Field Should Be    ${r}    session_id    test_static_py_vars
    Result Field Should Be    ${r}    variable_file    ${vf}
    Should Contain    ${r}[variables_loaded]    SCALAR_VAR
    Should Contain    ${r}[variables_loaded]    NUMBER_VAR
    Should Contain    ${r}[variables_loaded]    BOOLEAN_VAR
    Should Contain    ${r}[variables_loaded]    \@{MY_LIST}
    Should Contain    ${r}[variables_loaded]    \&{MY_DICT}
    Should Be True    $r['variables_map']['\${SCALAR_VAR}'] == 'test_value'
    Should Be True    $r['variables_map']['\${NUMBER_VAR}'] == 42
    Should Be True    $r['variables_map']['\${BOOLEAN_VAR}'] is True
    Should Be True    $r['variables_map']['\@{MY_LIST}'] == ['item1', 'item2', 'item3']
    Should Be True    $r['variables_map']['\&{MY_DICT}']['key1'] == 'value1'

Dynamic Python Variable File With Args
    ${vf}=    Set Variable    ${TEST_DATA}${/}dynamic_variables.py
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_dynamic_py_vars', 'variable_file_path': $vf, 'args': ['prod', 'secret123']} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Be True    $r['args'] == ['prod', 'secret123']
    Should Be True    $r['variables_map']['\${ENVIRONMENT}'] == 'prod'
    Should Be True    $r['variables_map']['\${API_KEY}'] == 'secret123'
    Should Be True    $r['variables_map']['\${BASE_URL}'] == 'https://prod.example.com'
    Should Contain    ${r}[variables_loaded]    PROD_ONLY_VAR
    Should Be True    $r['variables_map']['\${PROD_ONLY_VAR}'] == 'production setting'
    Should Be True    $r['variables_map']['\@{ENDPOINTS}'] == ['/prod/api', '/prod/health']
    Should Be True    $r['variables_map']['\&{CONFIG}']['env'] == 'prod'
    Should Be True    $r['variables_map']['\&{CONFIG}']['debug'] is False
    Should Be True    $r['variables_map']['\&{CONFIG}']['timeout'] == 30

Dynamic Python Variable File Default Args
    ${vf}=    Set Variable    ${TEST_DATA}${/}dynamic_variables.py
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_dynamic_py_default', 'variable_file_path': $vf} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Be True    $r['args'] == []
    Should Be True    $r['variables_map']['\${ENVIRONMENT}'] == 'test'
    Should Be True    $r['variables_map']['\${API_KEY}'] == 'default'
    Should Not Contain    ${r}[variables_loaded]    PROD_ONLY_VAR

Yaml Variable File
    ${vf}=    Set Variable    ${TEST_DATA}${/}test_variables.yaml
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_yaml_vars', 'variable_file_path': $vf} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Be True    $r['variables_map']['\${YAML_SCALAR}'] == 'yaml_value'
    Should Be True    $r['variables_map']['\${YAML_NUMBER}'] == 123
    Should Be True    $r['variables_map']['\${YAML_BOOLEAN}'] is True
    Should Be True    $r['variables_map']['\@{YAML_LIST}'] == ['first_item', 'second_item', 42, False]
    Should Be True    $r['variables_map']['\&{YAML_DICT}']['database']['host'] == 'localhost'
    Should Be True    $r['variables_map']['\&{YAML_DICT}']['api']['version'] == 'v1'

Json Variable File
    ${vf}=    Set Variable    ${TEST_DATA}${/}test_variables.json
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_json_vars', 'variable_file_path': $vf} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Be True    $r['variables_map']['\${JSON_SCALAR}'] == 'json_value'
    Should Be True    $r['variables_map']['\${JSON_NUMBER}'] == 456
    Should Be True    $r['variables_map']['\${JSON_BOOLEAN}'] is False
    Should Be True    $r['variables_map']['\@{JSON_LIST}'] == ['json_item1', 'json_item2', 789]
    Should Be True    $r['variables_map']['\&{JSON_DICT}']['settings']['theme'] == 'dark'

Nonexistent File Error
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_error_nonexistent', 'variable_file_path': '/path/to/nonexistent/file.py'} }}
    Result Field Should Be    ${r}    success    ${False}
    Result Should Contain Field    ${r}    error
    Should Be True    'nonexistent' in $r['error'] or 'not found' in $r['error'].lower()

Invalid Syntax Error
    ${vf}=    Set Variable    ${TEST_DATA}${/}invalid_syntax.py
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_error_syntax', 'variable_file_path': $vf} }}
    Result Field Should Be    ${r}    success    ${False}
    Result Should Contain Field    ${r}    error
    Result Field Should Be    ${r}    variable_file    ${vf}

Missing Variable File Path
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_missing_path'} }}
    Result Field Should Be    ${r}    success    ${False}
    Result Field Should Be    ${r}    error    variable_file_path is required

Variable Overwrite Behavior
    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'set_variables', 'session_id': 'test_overwrite', 'variables': {'SCALAR_VAR': 'initial_value'}} }}
    ${vf}=    Set Variable    ${TEST_DATA}${/}static_variables.py
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_overwrite', 'variable_file_path': $vf} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Be True    $r['variables_map']['\${SCALAR_VAR}'] == 'test_value'

Session Variable Tracking
    ${vf1}=    Set Variable    ${TEST_DATA}${/}static_variables.py
    ${r1}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_tracking', 'variable_file_path': $vf1} }}
    Result Field Should Be    ${r1}    success    ${True}
    ${vf2}=    Set Variable    ${TEST_DATA}${/}test_variables.yaml
    ${r2}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_tracking', 'variable_file_path': $vf2} }}
    Result Field Should Be    ${r2}    success    ${True}
    ${state}=    Rf Tool    ${HANDLE}    get_session_state
    ...    ${{ {'session_id': 'test_tracking', 'sections': ['variables']} }}
    Result Field Should Be    ${state}    success    ${True}
    Should Be True    'SCALAR_VAR' in $state['sections']['variables'].get('variables', {})
    Should Be True    'YAML_SCALAR' in $state['sections']['variables'].get('variables', {})

Yaml Import Provides Helpful Error Or Loads
    ${yaml_path}=    Evaluate    __import__('os').path.join(__import__('tempfile').gettempdir(), 'rfmcp_yaml_dep_test.yaml')
    Create File    ${yaml_path}    TEST_VAR: yaml_test_value\n
    TRY
        ${r}=    Rf Tool    ${HANDLE}    manage_session
        ...    ${{ {'action': 'import_variables', 'session_id': 'test_yaml_dependency', 'variable_file_path': $yaml_path} }}
        IF    $r['success']
            Should Contain    ${r}[variables_loaded]    TEST_VAR
        ELSE
            Should Be True    'yaml' in $r['error'].lower() or 'pyyaml' in $r['error'].lower()
        END
    FINALLY
        Run Keyword And Ignore Error    Remove File    ${yaml_path}
    END

Relative Path Resolution
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'import_variables', 'session_id': 'test_relative_path', 'variable_file_path': 'test_data/static_variables.py'} }}
    Result Field Should Be    ${r}    success    ${True}
    Should Contain    ${r}[variables_loaded]    SCALAR_VAR

Alternative Action Names
    ${vf}=    Set Variable    ${TEST_DATA}${/}static_variables.py
    ${r}=    Rf Tool    ${HANDLE}    manage_session
    ...    ${{ {'action': 'load_variables', 'session_id': 'test_alias', 'variable_file_path': $vf} }}
    Result Field Should Be    ${r}    success    ${True}
    Result Field Should Be    ${r}    action    import_variables
    Should Contain    ${r}[variables_loaded]    SCALAR_VAR
