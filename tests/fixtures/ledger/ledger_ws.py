"""Ledger plus SOAP 1.1 (/soap) and GraphQL (/graphql) faces over the same data, for web-service skill experiments.

No WSDL. SOAP operations live in namespace urn:ledger:v1; faults (HTTP 500) name the missing or bad element.
Extra bugs: soapvat, soapnotfound, soaplines, gqlpaid, gqlqty0, gqlfilter.
Run: python ledger_ws.py <port> [bug]
"""
import sys
import xml.etree.ElementTree as ET
from decimal import Decimal
from typing import Optional
from xml.sax.saxutils import escape

import strawberry
import uvicorn
from fastapi import Request, Response
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

import ledger_api as L

BUG = L.BUG
NS = "urn:ledger:v1"
SOAP = "http://schemas.xmlsoap.org/soap/envelope/"
app = L.app


def new_invoice(u: str, customer: str, lines: list[dict], currency: str = "EUR", vat_rate: str = "0.19") -> dict:
    L.SEQ["n"] += 1
    i = L.SEQ["n"]
    ls = [dict(sku=l["sku"], qty=l["qty"], unit_price=l["unit_price"],
               line_total=L.money(Decimal(str(l["unit_price"])) * l["qty"])) for l in lines]
    net = L.money(sum(Decimal(str(l["line_total"])) for l in ls))
    vat = L.money(Decimal(str(net)) * Decimal(vat_rate))
    inv = dict(id=i, number=f"INV-{i:05d}", customer={"name": customer, "vat_id": None}, lines=ls, currency=currency,
               paid=False, net=net, vat=vat, gross=L.money(Decimal(str(net)) + Decimal(str(vat))), attachments=[])
    L.DATA.setdefault(u, {})[i] = inv
    return inv


def token_user(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.startswith("Bearer "):
        return L.TOKENS.get(authorization[7:])
    return None


# ---------------------------------------------------------------- SOAP
class Fault(Exception):
    def __init__(self, msg, code="soap:Client", errors=()):
        self.msg, self.code, self.errors = msg, code, list(errors)


def envelope(inner: str, status=200) -> Response:
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<soap:Envelope xmlns:soap="{SOAP}" xmlns:l="{NS}"><soap:Body>{inner}</soap:Body></soap:Envelope>')
    return Response(xml, status_code=status, media_type="text/xml; charset=utf-8")


def fault(f: Fault) -> Response:
    det = "".join(f"<l:ValidationError><l:Field>{escape(a)}</l:Field><l:Message>{escape(b)}</l:Message></l:ValidationError>"
                  for a, b in f.errors)
    return envelope(f"<soap:Fault><faultcode>{f.code}</faultcode><faultstring>{escape(f.msg)}</faultstring>"
                    + (f"<detail>{det}</detail>" if det else "") + "</soap:Fault>", 500)


def inv_xml(inv: dict, all_lines=True) -> str:
    lines = inv["lines"] if all_lines else inv["lines"][:1]
    ls = "".join(f"<l:Line><l:Sku>{escape(l['sku'])}</l:Sku><l:Qty>{l['qty']}</l:Qty><l:UnitPrice>{l['unit_price']:.2f}</l:UnitPrice>"
                 f"<l:LineTotal>{l['line_total']:.2f}</l:LineTotal></l:Line>" for l in lines)
    return (f"<l:Invoice><l:Id>{inv['id']}</l:Id><l:Number>{inv['number']}</l:Number>"
            f"<l:Customer>{escape(inv['customer']['name'])}</l:Customer><l:Currency>{inv['currency']}</l:Currency>{ls}"
            f"<l:Net>{inv['net']:.2f}</l:Net><l:Vat>{inv['vat']:.2f}</l:Vat><l:Gross>{inv['gross']:.2f}</l:Gross>"
            f"<l:Paid>{str(inv['paid']).lower()}</l:Paid></l:Invoice>")


def q(name):
    return f"{{{NS}}}{name}"


def text(el, name, errors, conv=str, check=None, msg="required"):
    c = el.find(q(name))
    if c is None or c.text is None or not c.text.strip():
        errors.append((name, "required"))
        return None
    try:
        v = conv(c.text.strip())
    except ValueError:
        errors.append((name, msg))
        return None
    if check and not check(v):
        errors.append((name, msg))
        return None
    return v


OPS = ("CreateInvoice", "GetInvoice", "MarkInvoicePaid")


def soap_op(u: str, op: ET.Element) -> str:
    tag = op.tag
    name = tag.split("}")[-1]
    if not tag.startswith("{" + NS + "}") or name not in OPS:
        raise Fault(f"Unknown operation '{tag}'. Operations are in namespace {NS}: {', '.join(OPS)}")
    errors = []
    if name == "CreateInvoice":
        cust = text(op, "Customer", errors, check=lambda s: len(s) >= 2, msg="at least 2 characters")
        cur = (op.findtext(q("Currency")) or "EUR").strip()
        lines = []
        les = op.findall(q("Line"))
        if not les:
            errors.append(("Line", "at least one Line required"))
        for le in les:
            lines.append(dict(sku=text(le, "Sku", errors),
                              qty=text(le, "Qty", errors, int, lambda v: v >= 1, "must be a positive integer"),
                              unit_price=text(le, "UnitPrice", errors, float, lambda v: v > 0, "must be a positive decimal")))
        if errors:
            raise Fault("Validation failed", errors=errors)
        inv = new_invoice(u, cust, lines, cur, "0.16" if BUG == "soapvat" else "0.19")
        return f"<l:CreateInvoiceResponse>{inv_xml(inv)}</l:CreateInvoiceResponse>"
    inv_id = text(op, "Id", errors, int, msg="must be an integer")
    if errors:
        raise Fault("Validation failed", errors=errors)
    inv = L.DATA.setdefault(u, {}).get(inv_id)
    if not inv:
        if BUG == "soapnotfound":
            return f"<l:{name}Response/>"
        raise Fault(f"Invoice {inv_id} not found")
    if name == "MarkInvoicePaid":
        inv["paid"] = True
    return f"<l:{name}Response>{inv_xml(inv, BUG != 'soaplines')}</l:{name}Response>"


@app.get("/soap")
def soap_info():
    return {"message": "SOAP 1.1 endpoint. POST an envelope with a SOAPAction header and a bearer token.",
            "operations": list(OPS)}


@app.post("/soap")
async def soap(request: Request):
    u = token_user(request.headers.get("authorization"))
    if not u:
        return Response("Not authenticated", status_code=401, headers={"WWW-Authenticate": "Bearer"})
    try:
        action = (request.headers.get("soapaction") or "").strip('"')
        if not action:
            raise Fault("SOAPAction header required")
        try:
            root = ET.fromstring(await request.body())
        except ET.ParseError as e:
            raise Fault(f"Malformed XML: {e}")
        body = root.find(f"{{{SOAP}}}Body")
        if root.tag != f"{{{SOAP}}}Envelope" or body is None or len(body) == 0:
            raise Fault(f"Expected a SOAP 1.1 Envelope ({SOAP}) with a Body")
        op = body[0]
        if action.split("/")[-1].split("#")[-1] != op.tag.split("}")[-1]:
            raise Fault(f"SOAPAction '{action}' does not match operation '{op.tag.split('}')[-1]}'")
        return envelope(soap_op(u, op))
    except Fault as f:
        return fault(f)


# ---------------------------------------------------------------- GraphQL
@strawberry.type
class LineT:
    sku: str
    qty: int
    unit_price: float
    line_total: float


@strawberry.type
class InvoiceT:
    id: int
    number: str
    customer: str
    currency: str
    lines: list[LineT]
    net: float
    vat: float
    gross: float
    paid: bool


@strawberry.input
class LineInput:
    sku: str
    qty: int
    unit_price: float


@strawberry.input
class InvoiceInput:
    customer: str
    lines: list[LineInput]
    currency: str = "EUR"


def to_t(inv: dict) -> InvoiceT:
    return InvoiceT(id=inv["id"], number=inv["number"], customer=inv["customer"]["name"], currency=inv["currency"],
                    lines=[LineT(**l) for l in inv["lines"]], net=inv["net"], vat=inv["vat"], gross=inv["gross"],
                    paid=inv["paid"])


def gql_user(info: Info) -> str:
    u = token_user(info.context["request"].headers.get("authorization"))
    if not u:
        raise PermissionError("Not authenticated")
    return u


@strawberry.type
class Query:
    @strawberry.field
    def invoice(self, info: Info, id: int) -> Optional[InvoiceT]:
        inv = L.DATA.setdefault(gql_user(info), {}).get(id)
        return to_t(inv) if inv else None

    @strawberry.field
    def invoices(self, info: Info, customer: Optional[str] = None, paid: Optional[bool] = None) -> list[InvoiceT]:
        items = list(L.DATA.setdefault(gql_user(info), {}).values())
        if customer is not None and BUG != "gqlfilter":
            items = [x for x in items if x["customer"]["name"] == customer]
        if paid is not None:
            items = [x for x in items if x["paid"] == paid]
        return [to_t(x) for x in items]


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_invoice(self, info: Info, input: InvoiceInput) -> InvoiceT:
        u = gql_user(info)
        if len(input.customer) < 2:
            raise ValueError("customer: at least 2 characters")
        if not input.lines:
            raise ValueError("lines: at least one line required")
        for n, l in enumerate(input.lines):
            if l.qty < (0 if BUG == "gqlqty0" else 1):
                raise ValueError(f"lines[{n}].qty: must be >= 1")
            if l.unit_price <= 0:
                raise ValueError(f"lines[{n}].unitPrice: must be > 0")
        return to_t(new_invoice(u, input.customer, [vars(l) for l in input.lines], input.currency))

    @strawberry.mutation
    def mark_paid(self, info: Info, id: int, paid: bool = True) -> InvoiceT:
        inv = L.DATA.setdefault(gql_user(info), {}).get(id)
        if not inv:
            raise ValueError(f"Invoice {id} not found")
        if BUG != "gqlpaid":
            inv["paid"] = paid
        return to_t(inv)


app.include_router(GraphQLRouter(strawberry.Schema(query=Query, mutation=Mutation), graphql_ide=None), prefix="/graphql")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
