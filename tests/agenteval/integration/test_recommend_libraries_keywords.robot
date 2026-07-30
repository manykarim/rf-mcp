*** Settings ***
Documentation    Ported (Phase 2a, change agenteval-port-blackbox-integration) from
...              tests/integration/test_recommend_libraries_keywords.py: recommend_libraries
...              with include_keywords returns per-recommendation keyword hints. Real MCP protocol; deterministic.
Resource         rfmcp.resource
Suite Setup      Start Rf-mcp Server
Suite Teardown   Stop Rf-mcp Server

*** Test Cases ***
Recommend Libraries With Keywords
    ${r}=    Rf Tool    ${HANDLE}    recommend_libraries
    ...    ${{ {'scenario': 'Open a web page and click a button', 'context': 'web', 'include_keywords': True} }}
    Result Field Should Be    ${r}    success    ${True}
    ${recs}=    Evaluate    $r.get('recommendations', [])
    Should Not Be Empty    ${recs}
    ${top}=    Set Variable    ${recs}[0]
    Should Be True    'keywords' in $top
    Should Be True    isinstance($top['keywords'], list)
    Should Be True    len($top['keywords']) > 0
    Should Be True    'keyword_hint' in $top
    Should Be True    'keyword_source' in $top
