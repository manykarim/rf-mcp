*** Settings ***
Documentation     skill://rf-requests: a fresh REPL has BuiltIn only, so RequestsLibrary keywords are unknown
...               until `Import Library    RequestsLibrary`. This suite deliberately imports nothing.


*** Variables ***
${LEDGER_URL}     http://127.0.0.1:8711


*** Test Cases ***
Create Session Is Unknown Until RequestsLibrary Is Imported
    Run Keyword And Expect Error    No keyword with name 'Create Session' found.
    ...    Create Session    api    ${LEDGER_URL}
    Import Library    RequestsLibrary
    Import Library    Collections
    Create Session    api    ${LEDGER_URL}
    ${r}=    GET On Session    api    /    expected_status=200
    Dictionary Should Contain Key    ${r.json()}    endpoints
