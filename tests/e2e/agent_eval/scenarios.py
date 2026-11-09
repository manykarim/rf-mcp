"""Scenario definitions, prompts and project seeding for the four evaluation suites.

- baseline (7 scenarios, from the 2026-11-05 matrix): small calc/booker project in projects/baseline
- demoshop (T1, T2, T3, T6): the workshop repository against the shared demoshop, one X-Workshop-Space per run
- fixtures: Browser / SeleniumLibrary shadow DOM + iframe widgets on the local Widget Lab page, demoshop cart API
- api: schema-less ledger REST / SOAP / GraphQL (local fixture with bug servers) and restful-booker
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .env import PROJECTS_DIR
from .fixtures import SHOP_URL, FixtureError, demoshop_source


class SeedError(RuntimeError):
    """A project could not be seeded."""


@dataclass
class Scenario:
    name: str
    suite: str
    prompt: str
    file: str | None = None  # the suite the agent must write (authoring tasks)
    longname: str | None = None  # longname of `file` (baseline authoring grades)
    prerun: bool = False  # run the suite once before the agent starts (results/ for triage tasks)
    why_failed: bool = False  # "why did X fail?" task: rf_results should come before reading project files
    web: bool = False  # WebFetch allowed (public API docs)
    preset: str | None = None  # demoshop preset
    inject: list[tuple[str, str, str]] = field(default_factory=list)  # (file, old, new) test bugs
    remove: list[str] = field(default_factory=list)  # files deleted before the run
    extra: dict[str, str] = field(default_factory=dict)  # files added before the run
    bugs: list[str] = field(default_factory=list)  # ledger bug servers the suite must catch


@dataclass
class SuiteSpec:
    name: str
    scenarios: dict[str, Scenario]
    models: list[str]
    conditions: list[str]
    max_turns: int
    timeout: int
    seed: Callable  # seed(dest, scenario, ctx) -> None


# --------------------------------------------------------------------------------------- baseline

BROKEN = """*** Settings ***
Resource    common.resource

*** Test Cases ***
Broken Reference
    [Tags]    calc
    ${x}=    Add Numbrs    1    2
    Format Money    1    EUR    extra
"""

BASELINE = [
    Scenario("s1_triage_fix", "baseline", prerun=True, why_failed=True, prompt=(
        "The last test run had failures. Find out why the test 'Ten Percent Discount' failed, "
        "fix the root cause (do not change the test's expected values), and verify the fix.")),
    Scenario("s2_smoke", "baseline", prompt=(
        "Which tests in this project are tagged smoke? Run only those and report the outcome.")),
    Scenario("s3_analyze", "baseline", extra={"tests/calc/broken.robot": BROKEN}, prompt=(
        "Check this Robot Framework project for broken keyword references or wrong keyword arguments "
        "WITHOUT running any tests. List every problem with file and line.")),
    Scenario("s4_xml_author", "baseline", file="tests/xml/books.robot", longname="Tests.Xml.Books", prompt=(
        "Use Robot Framework to create a test suite and execute it step wise. It shall:\n"
        "- Create an example XML file data/books.xml for Books and Authors\n"
        "- Parse the XML file with the XML library, check several nodes and attributes and do assertions\n"
        "Execute it step by step and afterwards build the final test suite as tests/xml/books.robot.")),
    Scenario("s5_api_author", "baseline", file="tests/api/booking_crud.robot", longname="Tests.Api.Booking Crud", prompt=(
        "Use Robot Framework with RequestsLibrary to create a test suite tests/api/booking_crud.robot "
        "against https://restful-booker.herokuapp.com (API docs: https://restful-booker.herokuapp.com/apidoc/index.html). It shall:\n"
        "- Create a new booking and assert it\n"
        "- Read that booking back by its id and assert the response and its values "
        "(the server is public and other bookings change constantly, so assert only data your test created)\n"
        "- Authenticate as admin (username admin, password password123)\n"
        "- Delete a booking while authenticated\n"
        "Execute it step by step and build the final test suite afterwards.")),
    Scenario("s6_web_author", "baseline", file="tests/web/saucedemo.robot", longname="Tests.Web.Saucedemo", prompt=(
        "Use Robot Framework with the Browser library (headless) to create a test suite tests/web/saucedemo.robot "
        "for https://www.saucedemo.com/. It shall:\n- Login with valid user\n- Assert login was successful\n"
        "- Add item to cart\n- Assert item was added to cart\n- Checkout\n- Assert checkout was successful\n"
        "Execute it step by step and build the final test suite afterwards.")),
    Scenario("s7_debug_inspect", "baseline", prerun=True, prompt=(
        "The test 'Cart Contains Items' fails. Inspect the live variables at the point of failure inside the "
        "real test run and tell me the exact value of the cart variable. Do not modify any files.")),
]


def seed_baseline(dest: Path, sc: Scenario, ctx) -> None:
    shutil.copytree(PROJECTS_DIR / "baseline", dest, ignore=shutil.ignore_patterns("__pycache__", "results"))
    for rel, text in sc.extra.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------------------- demoshop

T6_INJECT = [
    ("tests/ui/checkout.robot", "    Cart Should Say It Is Empty\n", "    Cart Should Say Its Empty\n"),
    ("tests/ui/checkout.robot", "    Get Url    $=    /checkout\n", "    Get Url    $=    /checkout/confirm\n"),
    ("tests/ui/catalogue.robot", "    Check Category    Audio\n", "    Check Category    Audios\n"),
]

DEMOSHOP = [
    Scenario("t1_checkout", "demoshop", preset="clean", file="tests/ui/checkout.robot",
             remove=["tests/ui/checkout.robot", "resources/checkout.resource"], prompt=(
        "Implement the acceptance criteria WEB-006_AC-1, AC-7, AC-8, AC-10, AC-11 and AC-12 from "
        "openspec/specs/shop/checkout/spec.md as Robot Framework UI tests in tests/ui/checkout.robot, with all locators "
        "and steps as keywords in resources/checkout.resource (reuse resources/shop.resource and resources/api.resource). "
        "Follow docs/conventions.md. AC-8 must prove that both document links deliver real PDFs whose text contains the "
        "order number. Run the tests until they pass. The shop is already configured in .env.")),
    Scenario("t2_hybrid", "demoshop", preset="clean", file="tests/ui/cart_hybrid.robot", prompt=(
        "Write tests/ui/cart_hybrid.robot covering WEB-005_AC-3 and WEB-005_AC-4 (openspec/specs/shop/cart/spec.md): "
        "create the cart through the cart API (Aurora Neural Headphones x2 and product 7 x1) with a session id you "
        "choose, make the browser use that same cart, and verify the /cart lines and the summary. Then place the order "
        "through the UI and verify through GET /api/cart/ that the same session's cart is empty. Follow "
        "docs/conventions.md, keep locators in resources/, and run it until it passes. The shop is configured in .env.")),
    Scenario("t3_drift", "demoshop", preset="drift_and_bug", prerun=True, why_failed=True, prompt=(
        "The UI suite fails against the current shop (the last run's results are in results/). For every failing test "
        "find out why, using the recorded results and the debugger. Repair what is a test problem onto the stable "
        "contract of docs/conventions.md, and report what is a shop defect, with evidence. Tests tagged `broken` are "
        "known and out of scope. Never change expected values, never edit openspec/specs/shop, and never reset or change "
        "the shop's presets.")),
    Scenario("t6_triage", "demoshop", preset="buggy", prerun=True, why_failed=True, inject=T6_INJECT, prompt=(
        "Triage the last test run (results/) without re-running the whole suite and without opening results/output.xml "
        "directly. Cluster the failures by root cause, classify each cluster as test bug, shop defect, or known "
        "(tagged `broken`), cite evidence, and write results/triage.md. Then fix only the test bugs and rerun only the "
        "affected tests. Never change the shop's presets.")),
]

_DEMOSHOP_IGNORE = shutil.ignore_patterns(".git", ".claude", ".agents", ".github", "results", "uv.lock", ".venv",
                                          "website", "__pycache__", ".robotcode_cache")


def seed_demoshop(dest: Path, sc: Scenario, ctx) -> None:
    try:
        src = demoshop_source()
    except FixtureError as e:
        raise SeedError(str(e)) from e
    shutil.copytree(src, dest, ignore=_DEMOSHOP_IGNORE)
    (dest / ".env").write_text(f"SHOP_URL={SHOP_URL}\nSHOP_SPACE={ctx.space}\n", encoding="utf-8")
    rt = dest / "robot.toml"
    rt.write_text(rt.read_text(encoding="utf-8").replace('default-profiles = ["local"]', 'default-profiles = ["shared"]'),
                  encoding="utf-8")
    for rel in sc.remove:
        p = dest / rel
        if not p.is_file():
            raise SeedError(f"{sc.name}: {rel} is missing in the demoshop checkout")
        p.unlink()
    for rel, old, new in sc.inject:
        p = dest / rel
        text = p.read_text(encoding="utf-8") if p.is_file() else ""
        if old not in text:
            raise SeedError(f"{sc.name}: cannot inject a test bug, {rel} no longer contains {old.strip()!r}")
        p.write_text(text.replace(old, new, 1), encoding="utf-8")


# --------------------------------------------------------------------------------------- fixtures (skill A/B)

WIDGET_ITEMS = """1. Sign in: enter user `alice` (any password) and click `Sign in`; verify the message `Welcome, alice`.
2. Deep action: click `Deep action`; verify the status text `deep ok`.
3. Payment: pay with card `4111111111111111`; verify `Paid with card ending 1111`.
4. Confirmation: click `Confirm`; verify `Confirmed`.
5. Consent: click `Accept cookies`; verify `Accepted`.
6. Coupon: apply coupon `SAVE10`; verify `Coupon SAVE10 applied`.
7. Card: verify card `4000123412349999`; verify `card 9999 verified`.
8. Counter: click the counter's `+1` button twice; verify the count shows `2`."""
TAIL = ("Run the suite until all tests pass. If a test cannot be made to pass, keep it, tag it `blocked` and explain why "
        "in its [Documentation]. Do not modify the page under test.")
# The closed-root counter needs JavaScript run by the test (forcing attachShadow open); "do not modify the page" made
# agents tag it `blocked` instead (2026-11-06 sweeps), so the UI scenarios say what is allowed.
UI_TAIL = ("Run the suite until all tests pass. If a test cannot be made to pass, keep it, tag it `blocked` and explain "
           "why in its [Documentation]. Do not edit the Widget Lab's source files; JavaScript that the test itself runs in "
           "the browser is allowed.")

FIXTURES = [
    Scenario("fx_browser", "fixtures", file="tests/ui/widgets.robot", prompt=(
        "Write `tests/ui/widgets.robot` using the Browser library (headless chromium) for the Widget Lab page at `${URL}` "
        "(the variable is defined in robot.toml). One test per item:\n" + WIDGET_ITEMS + "\n" + UI_TAIL)),
    Scenario("fx_selenium", "fixtures", file="tests/ui/widgets_selenium.robot", prompt=(
        "Write `tests/ui/widgets_selenium.robot` using SeleniumLibrary (`Open Browser    ${URL}    headlesschrome`) for the "
        "Widget Lab page at `${URL}` (the variable is defined in robot.toml). One test per item:\n" + WIDGET_ITEMS + "\n" + UI_TAIL)),
    Scenario("fx_api", "fixtures", file="tests/api/cart_api.robot", preset="clean", prompt=(
        "Write `tests/api/cart_api.robot` using RequestsLibrary for the demoshop REST API at `${SHOP_URL}` (variables are in robot.toml). "
        "Every request must send the header `X-Workshop-Space: ${SHOP_SPACE}`; the cart is selected by the header `X-Session-ID` "
        "(use a unique value per test and clear the cart first with DELETE /api/cart/).\n"
        "Endpoints: POST /api/cart/items body {\"product_id\": int, \"quantity\": int >= 1}; GET /api/cart/; DELETE /api/cart/; "
        "POST /api/checkout/ body {\"name\": str (min 2), \"email\": str, \"address\": str (min 5)}. Validation errors return 422 "
        "with a FastAPI `detail` list whose entries have `loc` and `msg`.\nTests:\n"
        "1. Cart Totals: add product 1 with quantity 2 and product 7 with quantity 1; verify the cart has exactly 2 items, every item's "
        "total_price equals unit_price * quantity, and the cart total equals the sum of the item totals.\n"
        "2. Invalid Add Is Rejected: quantity 0 gives 422 with an error whose loc ends with `quantity`; a missing product_id gives 422 "
        "naming `product_id`; product 99999 gives 404.\n"
        "3. Checkout Validation: checkout with name `A`, no email and address `x` gives 422 and the error fields are exactly name, email "
        "and address.\n"
        "4. Checkout Order: with products 1 (x2) and 7 (x1) in the cart, checkout succeeds; the order subtotal equals the cart total, the "
        "order total equals subtotal + tax (to 2 decimals), the order items contain products 1 and 7 with their quantities, and the cart "
        "is empty afterwards.\n" + TAIL)),
]


def seed_fixtures(dest: Path, sc: Scenario, ctx) -> None:
    (dest / "tests" / "ui").mkdir(parents=True)
    (dest / "tests" / "api").mkdir(parents=True)
    (dest / "robot.toml").write_text(
        f'output-dir = "results"\n\n[variables]\nURL = "{ctx.web_url}"\n'
        f'SHOP_URL = "{SHOP_URL}"\nSHOP_SPACE = "{ctx.space}"\n', encoding="utf-8")


# --------------------------------------------------------------------------------------- api (schema-less)

LEDGER_PASSWORD = "Ledger-2026!"
REST_BUGS = ["vat", "qty0", "filter", "size", "delete", "auth"]
SOAP_BUGS = ["soapvat", "soaplines", "soapnotfound"]
GQL_BUGS = ["gqlfilter", "gqlpaid", "gqlqty0"]
_LEDGER_AUTH = ("Authenticate like the REST API does: `POST ${LEDGER_URL}/token` is a standard OAuth2 password-grant "
                "endpoint (credentials `LEDGER_USER` / `LEDGER_PASSWORD` from robot.toml); send the token as a bearer token.")

API = [
    Scenario("api_ledger", "api", file="tests/api/ledger_api.robot", bugs=REST_BUGS, prompt=(
        "Write `tests/api/ledger_api.robot` using RequestsLibrary for the Ledger invoicing API at `${LEDGER_URL}` "
        "(variables `LEDGER_URL`, `LEDGER_USER` and `LEDGER_PASSWORD` are in robot.toml). There is no published schema: "
        "`GET /` lists the endpoints; discover the request payloads from the API's error responses. The token endpoint is a "
        "standard OAuth2 password-grant endpoint; send the token as a bearer token.\nTests:\n"
        "1. Authentication: valid credentials give a bearer token; a wrong password gives 401; listing invoices without a token gives 401.\n"
        "2. Invoice Totals: create an unpaid EUR invoice for customer `ACME GmbH` with two lines: sku `A-1`, 2 units at 19.99; "
        "sku `B-7`, 1 unit at 5.00. Verify the status is 201, the Location header points to the new invoice (a GET on it returns "
        "the same id), net equals the sum of quantity x unit price, VAT is 19% of net rounded to 2 decimals, and gross = net + VAT.\n"
        "3. Invalid Invoices Are Rejected: a quantity of 0, a quantity sent as the text `two`, and a missing customer name are each "
        "rejected with 422, and the error names the offending field.\n"
        "4. Paid Filter: create two invoices for a customer name unique to the test and mark one as paid; listing the unpaid "
        "invoices of that customer returns exactly the other one.\n"
        "5. Attachment: attach a small text file `receipt.txt` (created by the test) to an invoice; the response reports the file "
        "name and its size in bytes.\n"
        "6. Delete: deleting an invoice returns 204, and a GET on it afterwards returns 404.\n"
        "The suite must also pass when run again with a different user.\n" + TAIL)),
    Scenario("api_booker", "api", file="tests/api/booker_api.robot", web=True, prompt=(
        "Write `tests/api/booker_api.robot` using RequestsLibrary for the public Restful Booker API at `${BOOKER_URL}` "
        "(variable in robot.toml; documentation: https://restful-booker.herokuapp.com/apidoc/index.html). Admin credentials: "
        "`admin` / `password123`. The API is shared with the public and reset from time to time: only rely on bookings your test "
        "creates.\nTests:\n"
        "1. Create Booking: create a booking for a first name unique to the test, last name `Tester`, total price 245, deposit "
        "paid, check-in 2026-12-01, check-out 2026-12-05, additional needs `Breakfast`; verify the response echoes every field "
        "with the right type (number, boolean, nested dates) and a GET by the returned id returns the same booking.\n"
        "2. Search: searching bookings by that first and last name returns the created id.\n"
        "3. Partial Update Needs Auth: a partial update of the first name without a token is rejected with 403; with a token it "
        "succeeds and only the first name changes.\n"
        "4. Full Update: replace the booking (total price 300, deposit not paid, check-in 2027-01-10, check-out 2027-01-12, "
        "additional needs `Late checkout`) and verify all fields.\n"
        "5. Delete: deleting without a token is rejected; with a token it succeeds and a GET afterwards returns 404.\n"
        "6. Wrong Password: requesting a token with a wrong password returns no token.\n" + TAIL)),
    Scenario("api_soap", "api", file="tests/api/ledger_soap.robot", bugs=SOAP_BUGS, prompt=(
        "Write `tests/api/ledger_soap.robot` for the Ledger SOAP 1.1 service at `${LEDGER_URL}/soap` using RequestsLibrary "
        "(variables are in robot.toml). There is no WSDL: `GET ${LEDGER_URL}/soap` lists the operations; discover the element "
        "names and the namespace from the SOAP faults. " + _LEDGER_AUTH + " \nTests:\n"
        "1. Create Invoice: CreateInvoice for customer `ACME GmbH` with two lines: sku `A-1`, 2 units at 19.99; sku `B-7`, 1 unit "
        "at 5.00. Verify the returned invoice has two lines, net equals the sum of quantity x unit price, VAT is 19% of net "
        "rounded to 2 decimals, and gross = net + VAT.\n"
        "2. Get Invoice: GetInvoice for the created id returns the same customer, both lines with their quantities, and unpaid.\n"
        "3. Mark Paid: after MarkInvoicePaid, GetInvoice shows the invoice as paid.\n"
        "4. Unknown Invoice: GetInvoice for id 999999 returns a SOAP fault whose faultstring says the invoice was not found.\n"
        "5. Validation Fault: a line with quantity 0 returns a SOAP fault whose detail names the quantity element.\n" + TAIL)),
    Scenario("api_graphql", "api", file="tests/api/ledger_graphql.robot", bugs=GQL_BUGS, prompt=(
        "Write `tests/api/ledger_graphql.robot` for the Ledger GraphQL API at `${LEDGER_URL}/graphql` using RequestsLibrary "
        "(variables are in robot.toml). No schema is given: discover it from the API. " + _LEDGER_AUTH + " \nTests:\n"
        "1. Create Invoice: create an invoice for customer `ACME GmbH` with two lines: sku `A-1`, 2 units at 19.99; sku `B-7`, "
        "1 unit at 5.00. Verify net equals the sum of quantity x unit price, VAT is 19% of net rounded to 2 decimals, and "
        "gross = net + VAT.\n"
        "2. Customer Filter: create invoices for two customer names unique to the test; querying the invoices of one customer "
        "returns only that customer's invoices.\n"
        "3. Mark Paid: mark an invoice as paid; querying it shows it paid, and the unpaid invoices of its customer no longer "
        "include it.\n"
        "4. Validation: an invoice line with quantity 0 is rejected: the response carries an error message that mentions the "
        "quantity field and no created invoice.\n"
        "5. Unauthenticated: a query without a token returns the error `Not authenticated`.\n" + TAIL)),
]


def seed_api(dest: Path, sc: Scenario, ctx) -> None:
    (dest / "tests" / "api").mkdir(parents=True)
    (dest / "robot.toml").write_text(
        f'output-dir = "results"\n\n[variables]\nLEDGER_URL = "{ctx.ledger_url}"\n'
        f'LEDGER_USER = "{ctx.user}"\nLEDGER_PASSWORD = "{LEDGER_PASSWORD}"\n'
        'BOOKER_URL = "https://restful-booker.herokuapp.com"\n', encoding="utf-8")


SUITES: dict[str, SuiteSpec] = {
    "baseline": SuiteSpec("baseline", {s.name: s for s in BASELINE}, ["haiku", "sonnet"], ["mcp", "cli"], 60, 1200,
                          seed_baseline),
    "demoshop": SuiteSpec("demoshop", {s.name: s for s in DEMOSHOP}, ["haiku", "sonnet"], ["mcp", "cli"], 150, 2700,
                          seed_demoshop),
    "fixtures": SuiteSpec("fixtures", {s.name: s for s in FIXTURES}, ["haiku"], ["mcp"], 120, 2400, seed_fixtures),
    "api": SuiteSpec("api", {s.name: s for s in API}, ["haiku"], ["mcp"], 120, 2400, seed_api),
}


def ledger_bugs(scenarios) -> list[str]:
    return sorted({b for s in scenarios for b in s.bugs})
