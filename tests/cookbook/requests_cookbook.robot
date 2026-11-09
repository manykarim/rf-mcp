*** Settings ***
Documentation     Executes every RequestsLibrary snippet of skill://rf-requests/SKILL.md (REST part)
...               against the local ledger fixture (tests/fixtures/ledger). One test per skill fact.
Library           RequestsLibrary
Library           Collections
Library           OperatingSystem
Suite Setup       Open Sessions


*** Variables ***
${LEDGER_URL}         http://127.0.0.1:8711
${LEDGER_USER}        cookbook
${LEDGER_PASSWORD}    Ledger-2026!


*** Test Cases ***
Probe With expected_status=any Shows The Error Body
    ${r}=    POST On Session    api    /token    json=${{ {} }}    expected_status=any
    Log    ${r.status_code} ${r.text}
    Should Be Equal As Integers    ${r.status_code}    422
    Should Contain    ${r.text}    Field required

HTTPError Without expected_status Hides The Body
    ${msg}=    Run Keyword And Expect Error    *    POST On Session    api    /token    json=${{ {} }}
    Should Contain    ${msg}    HTTPError: 422 Client Error
    Should Not Contain    ${msg}    Field required

Each Test Creates Its Own Record
    ${id}=    Create Test Invoice    ACME GmbH
    ${r}=     GET On Session    ledger    /invoices/${id}
    Should Be Equal    ${r.json()}[customer][name]    ACME GmbH

Session Headers Go With Every Request And Per Request Headers Add
    &{headers}=    Create Dictionary    X-Workshop-Space=space-1    X-Session-ID=cart-1
    Create Session    shop    ${LEDGER_URL}    headers=${headers}
    &{h}=    Create Dictionary    X-Extra=yes
    ${r}=    GET On Session    shop    /echo/cart    headers=${h}
    Should Be Equal    ${r.json()}[headers][x-workshop-space]    space-1
    Should Be Equal    ${r.json()}[headers][x-session-id]    cart-1
    Should Be Equal    ${r.json()}[headers][x-extra]    yes

JSON Text Is Sent As A JSON String
    ${r}=    POST On Session    api    /echo/items    json={"product_id": 1}
    Should Be Equal    ${r.json()}[json]    {"product_id": 1}
    ${body}=    Catenate    {"customer": {"name": "Text Co"}, "lines": [{"sku": "A", "qty": 1, "unit_price": 1.0}]}
    ${s}=    POST On Session    ledger    /invoices    json=${body}    expected_status=422
    Should Contain    ${s.text}    valid dictionary

Headers As Text Fail With Items Error
    Run Keyword And Expect Error    *'str' object has no attribute 'items'*
    ...    POST On Session    api    /echo/items    json=${{ {"product_id": 1} }}    headers={"X-Session-ID": "x"}

Inline Python Dict Needs The Braces Inside
    ${r}=    POST On Session    api    /echo/items    json=${{ {"product_id": 1, "quantity": 2} }}
    Should Be Equal    ${r.json()}[json][quantity]    ${2}

Ampersand Inline Dict Is A Syntax Error
    Run Keyword And Expect Error    *SyntaxError: invalid syntax*
    ...    POST On Session    api    /echo/items    json=&{{"k": "v"}}

Create Dictionary With Typed Values
    &{body}=    Create Dictionary    product_id=${7}    quantity=${3}
    ${r}=    POST On Session    api    /echo/items    json=${body}
    Should Be Equal    ${r.json()}[json][quantity]    ${3}
    Should Be Equal    ${r.json()}[json][product_id]    ${7}

Create Dictionary Values Without ${} Are Strings
    &{body}=    Create Dictionary    quantity=3
    ${r}=    POST On Session    api    /echo/items    json=${body}
    Should Be Equal    ${r.json()}[json][quantity]    3
    Should Be Equal    ${{ type($r.json()["json"]["quantity"]).__name__ }}    str

Booleans And Floats With ${}
    &{parcel}=    Create Dictionary    weight_kg=${2.5}    fragile=${True}    express=${False}
    ${r}=    POST On Session    api    /echo/parcels    json=${parcel}
    Should Be Equal    ${r.json()}[json]    ${{ {"weight_kg": 2.5, "fragile": True, "express": False} }}

Nested Payload From Inner Dictionaries
    &{address}=    Create Dictionary    city=Berlin    zip=10115
    &{parcel}=     Create Dictionary    weight_kg=${2.5}    fragile=${True}
    @{parcels}=    Create List    ${parcel}
    &{shipment}=   Create Dictionary    address=${address}    parcels=${parcels}    express=${False}
    ${r}=    POST On Session    api    /echo/shipments    json=${shipment}
    Should Be Equal    ${r.json()}[json][parcels][0][weight_kg]    ${2.5}
    Should Be Equal    ${r.json()}[json][parcels][0][fragile]    ${True}
    Should Be Equal    ${r.json()}[json][address][zip]    10115
    Should Be Equal    ${r.json()}[json][express]    ${False}

Negative Test With expected_status Then Inspect The Body
    ${body}=    Evaluate    {"customer": {"name": "Neg Co"}, "lines": [{"sku": "A", "qty": 0, "unit_price": 1.0}]}
    ${r}=    POST On Session    ledger    /invoices    json=${body}    expected_status=422
    ${fields}=    Evaluate    [e["loc"][-1] for e in $r.json()["detail"]]
    Should Contain    ${fields}    qty
    ${any}=    POST On Session    ledger    /invoices    json=${body}    expected_status=any
    Should Be Equal As Integers    ${any.status_code}    422

Reading Nested Responses
    ${r}=    Post Cart
    ${cart}=    Set Variable    ${r.json()}[json]
    Should Be Equal    ${cart}[items][0][quantity]    ${2}
    FOR    ${item}    IN    @{cart}[items]
        Should Be Equal As Numbers    ${item}[total_price]    ${{ $item["unit_price"] * $item["quantity"] }}    precision=2
    END
    ${ids}=    Evaluate    [i["product_id"] for i in $cart["items"]]
    Should Be Equal    ${ids}    ${{ [1, 7] }}

Loop Directly Over A Response List
    ${r}=    POST On Session    api    /echo/cart    json=${{ {"items": [{"unit_price": 19.99, "quantity": 2, "total_price": 39.98}]} }}
    ${data}=    Set Variable    ${r.json()}[json]
    FOR    ${item}    IN    @{data}[items]
        Should Be Equal As Numbers    ${item}[total_price]    ${{ $item["unit_price"] * $item["quantity"] }}    precision=2
    END
    ${s}=    POST On Session    api    /echo/items-list    json=${{ [{"id": 1}, {"id": 2}] }}
    ${n}=    Set Variable    ${0}
    FOR    ${item}    IN    @{s.json()}[json]
        ${n}=    Evaluate    $n + $item["id"]
    END
    Should Be Equal    ${n}    ${3}

JSON Numbers Stay Numbers
    &{body}=    Create Dictionary    quantity=${3}
    ${r}=    POST On Session    api    /echo/items    json=${body}
    Run Keyword And Expect Error    3 (integer) != 3 (string)    Should Be Equal    ${r.json()}[json][quantity]    3
    Should Be Equal As Integers    ${r.json()}[json][quantity]    3
    Should Be Equal    ${r.json()}[json][quantity]    ${3}

Arithmetic Must Happen Inside Inline Python
    ${r}=    POST On Session    api    /echo/order    json=${{ {"subtotal": 39.98, "tax": 7.6, "total": 47.58} }}
    ${o}=    Set Variable    ${r.json()}[json]
    Run Keyword And Expect Error    *cannot be converted to a floating point number*
    ...    Should Be Equal As Numbers    ${o}[total]    ${o}[subtotal] + ${o}[tax]
    Should Be Equal As Numbers    ${o}[total]    ${{ $o["subtotal"] + $o["tax"] }}    precision=2
    Run Keyword And Expect Error    *    Should Be Equal    ${o}[total]    ${{ $o["subtotal"] + $o["tax"] + 0.001 }}

Missing Key Message And Optional Headers
    ${r}=    GET On Session    api    /echo/cart
    Run Keyword And Expect Error    *has no key 'name'*    Log    ${r.json()}[name]
    Should Contain    ${r.headers}[content-type]    json
    Should Contain    ${r.headers}[Content-Type]    json
    ${missing}=    Evaluate    $r.headers.get("X-Not-There")
    Should Be Equal    ${missing}    ${None}

Variables Inside Evaluate Use Dollar Name
    @{items}=    Create List    a    b    c
    ${n}=    Evaluate    len($items)
    Should Be Equal    ${n}    ${3}

Form Body With data=
    ${r}=    POST On Session    api    /echo/token    data=${{ {"username": "u", "password": "p"} }}
    Should Start With    ${r.json()}[content_type]    application/x-www-form-urlencoded
    Should Be Equal    ${r.json()}[form][username]    u

Form Sent As json= Is Not A Form
    ${r}=    POST On Session    api    /token    json=${{ {"username": "u", "password": "p"} }}    expected_status=422
    ${fields}=    Evaluate    [e["loc"][-1] for e in $r.json()["detail"]]
    Should Contain    ${fields}    username
    Should Contain    ${r.text}    Field required

Multipart File Upload From A File
    ${path}=    Set Variable    ${OUTPUT DIR}${/}receipt.txt
    Create File    ${path}    total 53.53 EUR
    ${size}=    Get File Size    ${path}
    ${f}=    Evaluate    open($path, "rb")
    ${r}=    POST On Session    api    /echo/upload    files=${{ {"file": ("receipt.txt", $f, "text/plain")} }}
    Evaluate    $f.close()
    Should Be Equal    ${r.json()}[files][file][filename]    receipt.txt
    Should Be Equal As Integers    ${r.json()}[files][file][size]    ${size}
    Should Be Equal    ${r.json()}[files][file][content_type]    text/plain

Multipart From Bytes
    ${r}=    POST On Session    api    /echo/upload    files=${{ {"file": ("a.txt", b"hello", "text/plain")} }}
    Should Be Equal As Integers    ${r.json()}[files][file][size]    5

Query Parameters As A Dictionary
    ${r}=    GET On Session    api    /echo/invoices    params=${{ {"customer": "ACME GmbH", "paid": "false"} }}
    Should Be Equal    ${r.json()}[query]    ${{ {"customer": "ACME GmbH", "paid": "false"} }}

Python False In params Is Sent As Text
    ${r}=    GET On Session    api    /echo/invoices    params=${{ {"paid": False} }}
    Should Be Equal    ${r.json()}[query][paid]    False

Bearer Token On Every Request
    ${r}=    POST On Session    api    /token    data=${{ {"username": $LEDGER_USER, "password": $LEDGER_PASSWORD} }}
    ${token}=    Set Variable    ${r.json()}[access_token]
    Create Session    authed    ${LEDGER_URL}    headers=${{ {"Authorization": "Bearer " + $token} }}
    GET On Session    authed    /invoices    expected_status=200
    GET On Session    api    /invoices    expected_status=401

No Content Has No JSON
    ${id}=    Create Test Invoice    Delete Co
    ${d}=    DELETE On Session    ledger    /invoices/${id}    expected_status=204
    Should Be Empty    ${d.content}
    Run Keyword And Expect Error    *JSONDecodeError*    Evaluate    $d.json()

Location Header Is A Path For The Session
    ${body}=    Evaluate    {"customer": {"name": "Loc Co"}, "lines": [{"sku": "X", "qty": 1, "unit_price": 1.0}]}
    ${c}=    POST On Session    ledger    /invoices    json=${body}    expected_status=201
    ${g}=    GET On Session    ledger    ${c.headers}[Location]
    Should Be Equal    ${g.json()}[id]    ${c.json()}[id]

Auth Token As Cookie
    ${login}=    POST On Session    api    /token    data=${{ {"username": $LEDGER_USER, "password": $LEDGER_PASSWORD} }}
    &{h}=    Create Dictionary    Cookie=token=${login.json()}[access_token]
    ${r}=    PATCH On Session    api    /echo/shipments/1    json=${{ {"express": True} }}    headers=${h}
    Should Be Equal    ${r.json()}[headers][cookie]    token=${login.json()}[access_token]
    Should Be Equal    ${r.json()}[json][express]    ${True}

Run Keywords Cannot Assign
    Run Keyword And Expect Error    *Variable '\${r}' not found*
    ...    Run Keywords    ${r}=    Set Variable    1    AND    Log    ${r}


*** Keywords ***
Open Sessions
    Create Session    api    ${LEDGER_URL}
    ${r}=    POST On Session    api    /token    data=${{ {"username": $LEDGER_USER, "password": $LEDGER_PASSWORD} }}
    Create Session    ledger    ${LEDGER_URL}    headers=${{ {"Authorization": "Bearer " + $r.json()["access_token"]} }}

Create Test Invoice
    [Arguments]    ${customer}
    &{line}=    Create Dictionary    sku=A-1    qty=${2}    unit_price=${19.99}
    ${body}=    Create Dictionary    customer=${{ {"name": $customer} }}    lines=${{ [$line] }}
    ${r}=    POST On Session    ledger    /invoices    json=${body}    expected_status=201
    RETURN    ${r.json()}[id]

Post Cart
    &{a}=    Create Dictionary    product_id=${1}    unit_price=${19.99}    quantity=${2}    total_price=${39.98}
    &{b}=    Create Dictionary    product_id=${7}    unit_price=${5.0}    quantity=${1}    total_price=${5.0}
    ${r}=    POST On Session    api    /echo/cart    json=${{ {"items": [$a, $b]} }}
    RETURN    ${r}
