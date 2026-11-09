---
name: rf-requests
description: RequestsLibrary for REST, SOAP/XML and GraphQL steps with rf_repl - payloads, auth, status, responses, XML namespaces. Read before writing API or web-service tests.
---

# RequestsLibrary (API steps)

## In rf_repl, import first

The REPL starts with BuiltIn only - `Create Session` is "not found" until you send `Import Library    RequestsLibrary` (and `Import Library    Collections`), or `Import Resource    resources/api.resource` for project keywords.

## Exploring an API without a schema

Without `expected_status`, a 4xx/5xx raises `HTTPError: 422 Client Error ...` and the response body (which names the bad field) is NOT in the message. Probe with `expected_status=any` and log the body:

```
${r}=    POST On Session    api    /orders    json=${body}    expected_status=any
Log    ${r.status_code} ${r.text}
```

## Every test creates its own data

A test must pass when run alone (`rf_run` by longname, rerun of failed tests). Never hand ids from one test to the next with `Set Suite Variable` - "Get Invoice" then fails without "Create Invoice". Put creation in a keyword and call it in each test that needs the record:

```
Get Invoice Returns Lines
    ${id}=    Create Test Invoice    ACME GmbH
    ${r}=     GET On Session    api    /invoices/${id}
```

Suite Setup is for what every test only reads (the authenticated session, base data).

## Sessions and headers

```
&{headers}=    Create Dictionary    X-Workshop-Space=${SHOP_SPACE}    X-Session-ID=cart-1
Create Session    shop    ${SHOP_URL}    headers=${headers}
```

Session headers go with every request; per-request `headers=${h}` adds to them.

## Bodies: pass a dictionary, never JSON text

| Write | Result |
|---|---|
| `json={"product_id": 1}` | sends the JSON *string* `"{\"product_id\": 1}"` - API answers 422 "Input should be a valid dictionary" |
| `headers={"X-Session-ID": "x"}` | fails: `'str' object has no attribute 'items'` |
| `json=${{ {"product_id": 1, "quantity": 2} }}` | dict - correct (inline Python needs the dict braces *inside* `${{ }}`) |
| `&{{"k": "v"}}` | fails: `SyntaxError: invalid syntax` (no dict braces inside) |
| `&{body}=    Create Dictionary    product_id=${7}    quantity=${3}` then `json=${body}` | dict with ints - correct |
| `Create Dictionary    quantity=3` | value is the STRING `"3"` - use `${3}` for numbers, `${True}` for booleans |

Nested payloads - build the inner dictionaries and lists first:

```
&{address}=    Create Dictionary    city=Berlin    zip=10115
&{parcel}=     Create Dictionary    weight_kg=${2.5}    fragile=${True}
@{parcels}=    Create List    ${parcel}
&{shipment}=   Create Dictionary    address=${address}    parcels=${parcels}    express=${False}
${r}=    POST On Session    api    /shipments    json=${shipment}
```

## Status codes

- Without `expected_status`, any 4xx/5xx fails the step with `HTTPError: 422 Client Error ...`.
- Negative tests: `${r}=    POST On Session    shop    /api/checkout/    json=${body}    expected_status=422`, then inspect `${r.json()}`. `expected_status=any` accepts everything.

## Reading responses

```
${cart}=    Set Variable    ${r.json()}
Should Be Equal    ${cart}[items][0][quantity]    ${2}
FOR    ${item}    IN    @{r.json()}[items]
    Should Be Equal As Numbers    ${item}[total_price]    ${{ $item["unit_price"] * $item["quantity"] }}    precision=2
END
${ids}=       Evaluate    [i["product_id"] for i in $cart["items"]]
${fields}=    Evaluate    [e["loc"][-1] for e in $r.json()["detail"]]
```

- JSON numbers stay numbers: `Should Be Equal    ${x}    3` fails with `3 (integer) != 3 (string)` - compare with `${3}` or `Should Be Equal As Integers` / `As Numbers`.
- Arithmetic must happen inside `${{ }}`: `Should Be Equal As Numbers    ${o}[total]    ${o}[subtotal] + ${o}[tax]` fails ("cannot be converted to a floating point number"); write `${{ $o["subtotal"] + $o["tax"] }}` and add `precision=2` for money.
- A missing key fails with `Dictionary ... has no key 'name'`. Headers are case-insensitive (`${r.headers}[content-type]`) but may be absent (`Content-Length` with gzip): `Evaluate    $r.headers.get("Content-Length")`.
- Inside `Evaluate` / `${{ }}` refer to variables as `$name`, not `${name}`.

## Forms, files, query parameters

| Need | Write |
|---|---|
| form body (OAuth2 password grant, HTML forms) | `data=${{ {"username": $USER, "password": $PASSWORD} }}` - `json=` is not a form (the API answers 422 "username Field required") |
| file upload (multipart) | `files=${{ {"file": ("receipt.txt", $f, "text/plain")} }}` with `${f}=  Evaluate  open($path, "rb")`, then `Evaluate  $f.close()`; bytes work too: `("a.txt", b"hello", "text/plain")` |
| query string | `params=${{ {"customer": "ACME GmbH", "paid": "false"} }}` - a Python `False` is sent as the text `False` |
| bearer token on every request | `Create Session    api    ${URL}    headers=${{ {"Authorization": "Bearer " + $token} }}` |

- 204 No Content has an empty body: `${r.json()}` raises `JSONDecodeError`; assert `Should Be Empty    ${r.content}`.
- A `Location` header is usually a path: `GET On Session    api    ${r.headers}[Location]` works with the session base URL.

## Auth token as cookie

```
${login}=    POST On Session    api    /login    json=${{ {"user": "demo", "password": "secret"} }}
&{h}=        Create Dictionary    Cookie=token=${login.json()}[token]
PATCH On Session    api    /shipments/${id}    json=${{ {"express": True} }}    headers=${h}
```

## SOAP / XML services

Build the envelope in ONE cell. Two spaces inside a cell split it (`Set Variable    <a  x="1"/>` returns a list), so use `Catenate    SEPARATOR=` with `...` lines and single spaces:

```
${ENV_OPEN}=    Set Variable    <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" xmlns:l="urn:example:v1"><soap:Body>
${body}=    Catenate    SEPARATOR=
...    ${ENV_OPEN}<l:GetOrder><l:Id>${id}</l:Id></l:GetOrder>
...    </soap:Body></soap:Envelope>
&{h}=    Create Dictionary    Content-Type=text/xml; charset=utf-8    SOAPAction=GetOrder
${r}=    POST On Session    api    /soap    data=${body}    headers=${h}
```

- A SOAP fault comes back as HTTP 500: use `expected_status=500` (or `any` while exploring). The fault text names the missing element or namespace.
- Parse with the XML library (`Import Library    XML`): `${root}=    Parse XML    ${r.text}`. It STRIPS namespaces by default, so XPaths use plain names and start below the root element `Envelope`:

```
${total}=     Get Element Text     ${root}    .//Total
${net}=       Get Element Text     ${root}    Body/GetOrderResponse/Order/Net
@{fields}=    Get Elements Texts   ${root}    .//ValidationError/Field
Element Text Should Be    ${root}    Validation failed    xpath=.//faultstring
```

  Prefixed XPaths (`.//l:Total`, `soap:Body/...`) fail. With `keep_clark_notation=True` write `.//{urn:example:v1}Total`. `.//Line/Qty` with two lines fails ("Multiple elements (2) matching") - use `.//Line[1]/Qty` or `Get Elements Texts`.
- XML texts are strings: compare numbers with `Should Be Equal As Numbers` / `As Integers`, or convert inside `${{ float($root.find(".//Net").text) }}`.

## GraphQL

One POST with `query` and `variables`; keep the query in a variable and pass variables as a dictionary so numbers stay numbers:

```
${q}=    Set Variable    mutation($i: OrderInput!) { createOrder(input: $i) { id total } }
&{line}=    Create Dictionary    sku=A-1    qty=${2}    unitPrice=${19.99}
${r}=    POST On Session    api    /graphql    json=${{ {"query": $q, "variables": {"i": {"customer": "ACME", "lines": [$line]}}} }}
Dictionary Should Not Contain Key    ${r.json()}    errors
```

- `$i` inside a Robot string is literal (only `${...}` is a Robot variable).
- Errors come with status 200: `{"data": null, "errors": [{"message": "..."}]}` - always assert that `errors` is absent, and for negative tests check `${r.json()}[errors][0][message]`.
- The schema is discoverable: `query { __schema { queryType { fields { name } } mutationType { fields { name args { name } } } } }`; a wrong field name answers "Cannot query field 'x' on type 'T'".

## In rf_repl

`Run Keywords` cannot assign (`${r}=` inside it fails with "Variable '${r}' not found"). Send the steps as several lines in one `rf_repl send` instead.
