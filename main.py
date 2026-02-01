import os
import math
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

APP_NAME = "TCD Quote API"
API_KEY = os.getenv("TCD_API_KEY", "").strip()

DATA_DIR = os.getenv("TCD_DATA_DIR", "../tcd_bot_inputs")
PRODUCTS_CSV = f"{DATA_DIR}/products_master.csv"

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description=(
        "Use this API to search products by SKU/name and generate quotes.\n\n"
        "Rules:\n"
        "- Prices are ex-GST in products_master.csv\n"
        "- If a SKU has no price (REVIEW_REQUIRED), do not quote it; ask staff.\n"
    )
)

def require_key(x_api_key: Optional[str]):
    if not API_KEY:
        return  # allow if not configured (dev only)
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def load_products():
    try:
        df = pd.read_csv(PRODUCTS_CSV)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Missing {PRODUCTS_CSV}")
    # normalize
    df["sku"] = df["sku"].astype(str).str.strip()
    return df

class QuoteItem(BaseModel):
    sku: str = Field(..., description="Product SKU")
    qty: int = Field(..., ge=1, description="Quantity")

class QuoteRequest(BaseModel):
    items: List[QuoteItem]
    include_gst: bool = Field(True, description="If true, includes GST totals (AU 10%)")
    customer_postcode: Optional[str] = Field(None, description="Optional (used later for delivery zones)")

@app.get("/products/search", summary="Search products by SKU or name")
def search_products(q: str, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    df = load_products()
    q2 = q.strip().lower()
    if not q2:
        return {"results": []}

    mask = df["sku"].str.lower().str.contains(q2, na=False) | df["name"].astype(str).str.lower().str.contains(q2, na=False)
    out = df[mask].head(20)[["sku","category","name","unit_price_ex_gst","currency","status","review_reason"]].fillna("").to_dict("records")
    return {"results": out}

@app.get("/products/{sku}", summary="Get a product by SKU")
def get_product(sku: str, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    df = load_products()
    row = df[df["sku"].str.lower() == sku.strip().lower()]
    if row.empty:
        raise HTTPException(status_code=404, detail="SKU not found")
    r = row.iloc[0].fillna("").to_dict()
    return r

@app.post("/quote", summary="Create a quote from SKU + quantity")
def create_quote(req: QuoteRequest, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    df = load_products()

    lines = []
    subtotal_ex = 0.0

    for it in req.items:
        sku = it.sku.strip()
        row = df[df["sku"].str.lower() == sku.lower()]
        if row.empty:
            raise HTTPException(status_code=400, detail=f"Unknown SKU: {sku}")

        r = row.iloc[0]
        status = str(r.get("status", "")).strip()
        price = r.get("unit_price_ex_gst", None)

        # pandas may give NaN for blanks
        if status != "OK" or price is None or (isinstance(price, float) and math.isnan(price)):
            reason = str(r.get("review_reason","")).strip()
            raise HTTPException(
                status_code=400,
                detail=f"SKU {sku} is not safe-to-quote (status={status}, reason={reason})."
            )

        unit_price = float(price)
        line_total = unit_price * int(it.qty)
        subtotal_ex += line_total

        lines.append({
            "sku": sku,
            "name": str(r.get("name","")),
            "category": str(r.get("category","")),
            "qty": int(it.qty),
            "unit_price_ex_gst": round(unit_price, 2),
            "line_total_ex_gst": round(line_total, 2),
            "currency": str(r.get("currency","AUD")) or "AUD"
        })

    gst = round(subtotal_ex * 0.10, 2) if req.include_gst else 0.0
    total = round(subtotal_ex + gst, 2)

    return {
        "currency": "AUD",
        "lines": lines,
        "subtotal_ex_gst": round(subtotal_ex, 2),
        "gst": gst,
        "total_inc_gst": total if req.include_gst else None,
        "notes": [
            "Delivery/assembly not included unless specified.",
            "If a SKU is REVIEW_REQUIRED, staff must confirm price before quoting."
        ]
    }
