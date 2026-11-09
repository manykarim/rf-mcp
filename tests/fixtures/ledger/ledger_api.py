"""Ledger: a strict invoicing API for RequestsLibrary skill experiments.

No OpenAPI/docs on purpose: agents discover payloads from 422 responses.
LEDGER_BUG=<name> injects one bug (vat, qty0, filter, size, delete, auth) for mutation grading.
Run: python ledger_api.py <port> [bug]
"""
import hashlib
import secrets
import sys
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field

BUG = sys.argv[2] if len(sys.argv) > 2 else ""
PASSWORD = "Ledger-2026!"
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
TOKENS: dict[str, str] = {}
DATA: dict[str, dict[int, dict]] = {}
SEQ = {"n": 0}


class Strict(BaseModel):
    model_config = ConfigDict(strict=True)


class Customer(Strict):
    name: Annotated[str, Field(min_length=2)]
    vat_id: Optional[str] = None


class Line(Strict):
    sku: Annotated[str, Field(min_length=1)]
    qty: Annotated[int, Field(ge=0 if BUG == "qty0" else 1)]
    unit_price: Annotated[float, Field(gt=0)]


class InvoiceIn(Strict):
    customer: Customer
    lines: Annotated[list[Line], Field(min_length=1)]
    currency: Literal["EUR", "USD"] = "EUR"
    paid: bool = False


class InvoicePatch(Strict):
    paid: Optional[bool] = None
    customer: Optional[Customer] = None


def money(x) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def user(authorization: Annotated[Optional[str], Header()] = None) -> str:
    if not authorization or not authorization.startswith("Bearer ") or authorization[7:] not in TOKENS:
        raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    return TOKENS[authorization[7:]]


def get_inv(u: str, inv_id: int) -> dict:
    inv = DATA.setdefault(u, {}).get(inv_id)
    if not inv:
        raise HTTPException(404, f"Invoice {inv_id} not found")
    return inv


@app.get("/")
def index():
    return {"name": "Ledger", "endpoints": ["POST /token", "GET /invoices", "POST /invoices", "GET /invoices/{id}",
                                            "PATCH /invoices/{id}", "DELETE /invoices/{id}",
                                            "POST /invoices/{id}/attachments"]}


@app.post("/token")
def token(username: Annotated[str, Form()], password: Annotated[str, Form()], grant_type: Annotated[Optional[str], Form()] = None):
    if password != PASSWORD and BUG != "auth":
        raise HTTPException(401, "Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    t = secrets.token_hex(16)
    TOKENS[t] = username
    return {"access_token": t, "token_type": "bearer", "expires_in": 3600}


@app.post("/invoices", status_code=201)
def create(body: InvoiceIn, response: Response, u: str = Depends(user)):
    SEQ["n"] += 1
    i = SEQ["n"]
    lines = [dict(l.model_dump(), line_total=money(Decimal(str(l.unit_price)) * l.qty)) for l in body.lines]
    net = money(sum(Decimal(str(l["line_total"])) for l in lines))
    vat = money(Decimal(str(net)) * Decimal("0.16" if BUG == "vat" else "0.19"))
    inv = dict(id=i, number=f"INV-{i:05d}", customer=body.customer.model_dump(), lines=lines, currency=body.currency,
               paid=body.paid, net=net, vat=vat, gross=money(Decimal(str(net)) + Decimal(str(vat))), attachments=[])
    DATA.setdefault(u, {})[i] = inv
    response.headers["Location"] = f"/invoices/{i}"
    return inv


@app.get("/invoices")
def list_invoices(u: str = Depends(user), paid: Optional[bool] = None, customer: Optional[str] = None,
                  limit: int = 20, offset: int = 0):
    items = list(DATA.setdefault(u, {}).values())
    if paid is not None and BUG != "filter":
        items = [x for x in items if x["paid"] == paid]
    if customer is not None:
        items = [x for x in items if x["customer"]["name"] == customer]
    return {"items": items[offset:offset + limit], "total": len(items), "limit": limit, "offset": offset}


@app.get("/invoices/{inv_id}")
def read(inv_id: int, u: str = Depends(user)):
    return get_inv(u, inv_id)


@app.patch("/invoices/{inv_id}")
def patch(inv_id: int, body: InvoicePatch, u: str = Depends(user)):
    inv = get_inv(u, inv_id)
    if body.paid is not None:
        inv["paid"] = body.paid
    if body.customer is not None:
        inv["customer"] = body.customer.model_dump()
    return inv


@app.delete("/invoices/{inv_id}", status_code=204)
def delete(inv_id: int, u: str = Depends(user)):
    get_inv(u, inv_id)
    if BUG != "delete":
        del DATA[u][inv_id]
    return Response(status_code=204)


@app.post("/invoices/{inv_id}/attachments", status_code=201)
async def attach(inv_id: int, file: Annotated[UploadFile, File()], u: str = Depends(user)):
    inv = get_inv(u, inv_id)
    data = await file.read()
    a = dict(filename=file.filename, size=len(data) + (1 if BUG == "size" else 0),
             content_type=file.content_type, sha256=hashlib.sha256(data).hexdigest())
    inv["attachments"].append(a)
    return a


@app.api_route("/echo/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
async def echo(path: str, request: Request):
    """Unlisted: echoes what RequestsLibrary sent, for verifying skill snippets."""
    ctype = request.headers.get("content-type", "")
    out = {"path": path, "method": request.method, "query": dict(request.query_params), "headers": dict(request.headers),
           "content_type": ctype}
    if ctype.startswith("application/json"):
        out["json"] = await request.json()
    elif ctype.startswith(("multipart/", "application/x-www-form-urlencoded")):
        form = await request.form()
        out["form"] = {k: v for k, v in form.items() if isinstance(v, str)}
        out["files"] = {k: {"filename": v.filename, "size": len(await v.read()), "content_type": v.content_type}
                        for k, v in form.items() if not isinstance(v, str)}
    else:
        out["raw"] = (await request.body()).decode("utf-8", "replace")
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
