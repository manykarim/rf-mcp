*** Settings ***
Documentation     Executes every SOAP/XML and GraphQL snippet of skill://rf-requests/SKILL.md against the
...               local ledger fixture (tests/fixtures/ledger/ledger_ws.py). Each W test records one fact.
Library           RequestsLibrary
Library           XML
Library           Collections
Library           OperatingSystem
Suite Setup       Login


*** Variables ***
${LEDGER_URL}     http://127.0.0.1:8711
${LEDGER_USER}    cookbook-ws
${ENV_OPEN}       <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" xmlns:l="urn:ledger:v1"><soap:Body>
${ENV_CLOSE}      </soap:Body></soap:Envelope>


*** Test Cases ***
W01 Double Space Inside XML Splits The Cell Into A List
    ${xml}=    Set Variable    <a  x="1"/>
    Should Be True    isinstance($xml, list)
    Should Be Equal    ${xml}    ${{ ["<a", 'x="1"/>'] }}

W02 Single Line Envelope With Catenate Works
    ${body}=    Catenate    SEPARATOR=
    ...    ${ENV_OPEN}<l:CreateInvoice><l:Customer>ACME GmbH</l:Customer>
    ...    <l:Line><l:Sku>A-1</l:Sku><l:Qty>2</l:Qty><l:UnitPrice>19.99</l:UnitPrice></l:Line>
    ...    </l:CreateInvoice>${ENV_CLOSE}
    ${r}=    Soap    CreateInvoice    ${body}
    Should Be Equal As Integers    ${r.status_code}    200

W03 Catenate Default Separator Inserts Spaces But XML Still Parses
    ${body}=    Catenate
    ...    ${ENV_OPEN}<l:GetInvoice><l:Id>1</l:Id>
    ...    </l:GetInvoice>${ENV_CLOSE}
    Should Contain    ${body}    <l:Id>1</l:Id> </l:GetInvoice>
    ${r}=    Soap    GetInvoice    ${body}    expected_status=any
    Should Not Contain    ${r.text}    Malformed

W04 Soap Fault Raises HTTPError Without expected_status
    ${body}=    Set Variable    ${ENV_OPEN}<l:GetInvoice><l:Id>999999</l:Id></l:GetInvoice>${ENV_CLOSE}
    &{h}=    Create Dictionary    Content-Type=text/xml; charset=utf-8    SOAPAction=GetInvoice
    ${msg}=    Run Keyword And Expect Error    *    POST On Session    ledger    /soap    data=${body}    headers=${h}
    Should Contain    ${msg}    HTTPError: 500
    Should Not Contain    ${msg}    not found

W05 Parse XML Strips Namespaces By Default
    ${r}=    Create Invoice Soap
    ${root}=    Parse XML    ${r.text}
    Should Be Equal    ${root.tag}    Envelope
    ${gross}=    Get Element Text    ${root}    .//Gross
    Should Be Equal    ${gross}    53.53

W06 Prefixed XPath Fails After Parse XML
    ${r}=    Create Invoice Soap
    ${root}=    Parse XML    ${r.text}
    Run Keyword And Expect Error    *    Get Element Text    ${root}    .//l:Gross

W07 Path From Root Omits Envelope
    ${r}=    Create Invoice Soap
    ${root}=    Parse XML    ${r.text}
    ${n}=    Get Element Text    ${root}    Body/CreateInvoiceResponse/Invoice/Net
    Should Be Equal    ${n}    44.98

W08 Clark Notation When Namespaces Kept
    ${r}=    Create Invoice Soap
    ${root}=    Parse XML    ${r.text}    keep_clark_notation=True
    ${n}=    Get Element Text    ${root}    .//{urn:ledger:v1}Net
    Should Be Equal    ${n}    44.98

W09 Fault Text And Detail Fields
    ${body}=    Set Variable    ${ENV_OPEN}<l:CreateInvoice><l:Customer>ACME</l:Customer><l:Line><l:Sku>A</l:Sku><l:Qty>0</l:Qty><l:UnitPrice>1</l:UnitPrice></l:Line></l:CreateInvoice>${ENV_CLOSE}
    ${r}=    Soap    CreateInvoice    ${body}    expected_status=500
    ${root}=    Parse XML    ${r.content}
    Element Text Should Be    ${root}    Validation failed    xpath=.//faultstring
    @{fields}=    Get Elements Texts    ${root}    .//ValidationError/Field
    Should Be Equal    ${fields}    ${{ ["Qty"] }}

W10 XML Element Texts Are Strings
    ${r}=    Create Invoice Soap
    ${root}=    Parse XML    ${r.text}
    ${q}=    Get Element Text    ${root}    .//Line[1]/Qty
    Run Keyword And Expect Error    *(string)*(integer)*    Should Be Equal    ${q}    ${2}
    Should Be Equal As Numbers    ${{ float($root.find(".//Net").text) + float($root.find(".//Vat").text) }}    53.53    precision=2

W12 GraphQL Query In A Variable With Spaced Braces Works
    ${query}=    Set Variable    query { invoices { id net } }
    ${r}=    POST On Session    ledger    /graphql    json=${{ {"query": $query} }}
    Dictionary Should Not Contain Key    ${r.json()}    errors

W13 GraphQL Variables Keep Types Via Dict
    ${q}=    Set Variable    mutation($i: InvoiceInput!) { createInvoice(input: $i) { id gross } }
    &{line}=    Create Dictionary    sku=A-1    qty=${2}    unitPrice=${19.99}
    &{input}=    Create Dictionary    customer=ACME GmbH    lines=${{ [$line] }}
    ${r}=    POST On Session    ledger    /graphql    json=${{ {"query": $q, "variables": {"i": $input}} }}
    Dictionary Should Not Contain Key    ${r.json()}    errors
    Should Be Equal    ${r.json()}[data][createInvoice][gross]    ${47.58}

W14 GraphQL Errors Come With Status 200
    ${q}=    Set Variable    mutation($i: InvoiceInput!) { createInvoice(input: $i) { id } }
    ${r}=    POST On Session    ledger    /graphql    json=${{ {"query": $q, "variables": {"i": {"customer": "ACME", "lines": [{"sku": "A", "qty": 0, "unitPrice": 1.0}]}}} }}
    Should Be Equal As Integers    ${r.status_code}    200
    Should Be Equal    ${r.json()}[data]    ${None}
    Should Contain    ${r.json()}[errors][0][message]    qty

W15 GraphQL Dollar Variable Is Literal In Robot Strings
    ${q}=    Set Variable    query($c: String) { invoices(customer: $c) { id } }
    Should Contain    ${q}    $c
    ${r}=    POST On Session    ledger    /graphql    json=${{ {"query": $q, "variables": {"c": "nobody"}} }}
    Should Be Equal    ${r.json()}[data][invoices]    ${{ [] }}

W16 Query From A File
    Create File    ${OUTPUT DIR}/create.graphql    mutation($i: InvoiceInput!) {\n createInvoice(input: $i) { id }\n}
    ${q}=    Get File    ${OUTPUT DIR}/create.graphql
    ${r}=    POST On Session    ledger    /graphql    json=${{ {"query": $q, "variables": {"i": {"customer": "ACME", "lines": [{"sku": "A", "qty": 1, "unitPrice": 1.0}]}}} }}
    Dictionary Should Not Contain Key    ${r.json()}    errors

W17 Introspection Query Lists Fields
    ${q}=    Set Variable    query { __schema { queryType { fields { name } } mutationType { fields { name args { name } } } } }
    ${r}=    POST On Session    ledger    /graphql    json=${{ {"query": $q} }}
    ${names}=    Evaluate    [f["name"] for f in $r.json()["data"]["__schema"]["mutationType"]["fields"]]
    Should Be Equal    ${names}    ${{ ["createInvoice", "markPaid"] }}


*** Keywords ***
Login
    Create Session    anon    ${LEDGER_URL}
    ${r}=    POST On Session    anon    /token    data=${{ {"username": $LEDGER_USER, "password": "Ledger-2026!"} }}
    Create Session    ledger    ${LEDGER_URL}    headers=${{ {"Authorization": "Bearer " + $r.json()["access_token"]} }}

Soap
    [Arguments]    ${action}    ${body}    ${expected_status}=200
    &{h}=    Create Dictionary    Content-Type=text/xml; charset=utf-8    SOAPAction=${action}
    ${r}=    POST On Session    ledger    /soap    data=${body}    headers=${h}    expected_status=${expected_status}
    RETURN    ${r}

Create Invoice Soap
    ${body}=    Catenate    SEPARATOR=
    ...    ${ENV_OPEN}<l:CreateInvoice><l:Customer>ACME GmbH</l:Customer>
    ...    <l:Line><l:Sku>A-1</l:Sku><l:Qty>2</l:Qty><l:UnitPrice>19.99</l:UnitPrice></l:Line>
    ...    <l:Line><l:Sku>B-7</l:Sku><l:Qty>1</l:Qty><l:UnitPrice>5.00</l:UnitPrice></l:Line>
    ...    </l:CreateInvoice>${ENV_CLOSE}
    ${r}=    Soap    CreateInvoice    ${body}
    RETURN    ${r}
