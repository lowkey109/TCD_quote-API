import os
import math
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

# App metadata
APP_NAME = "TCD Quote API"
VERSION = "1.0.0"

# Configuration from environment
DATA_DIR = Path(os.getenv("TCD_DATA_DIR", "../tcd_bot_inputs"))
PRODUCTS_CSV = DATA_DIR / "products_master.csv"
API_KEY = os.getenv("TCD_API_KEY", "").strip()
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() in ("1", "true", "yes")

# FastAPI app (single instance)
app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description=(
        "Use this API to search products by SKU/name and generate quotes.\n\n"
        "Rules:\n"
        "- Prices are ex-GST in products_master.csv\n"
        "- If a SKU has no price (REVIEW_REQUIRED), do not quote it; ask staff.\n"
    ),
)

# Logger
logger = logging.getLogger("tcd_quote_api")
logging.basicConfig(level=logging.INFO)


@app.on_event("startup")
async def on_startup():
    if REQUIRE_API_KEY and not API_KEY:
        logger.error("REQUIRE_API_KEY is set but TCD_API_KEY is missing. Failing startup.")
        raise RuntimeError("Server misconfigured: TCD_API_KEY is required")
    if not API_KEY:
        logger.warning("TCD_API_KEY not set — API key checks are disabled (development mode)")
    else:
        logger.info("TCD_API_KEY loaded; API key protection enabled")


# Helpers

def require_key(x_api_key: Optional[str]):
    """Raise HTTPException if the provided x_api_key is invalid.

    If the server has no API_KEY configured, this function is a no-op (convenience for dev).
    """
    if not API_KEY:
        # Development convenience: allow requests if no key configured
        return
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@lru_cache(maxsize=1)
def load_products() -> pd.DataFrame:
    """Load and cache the products CSV. Cached until the process restarts.

    Raises HTTPException(500) if the CSV cannot be found/read.
    """
    if not PRODUCTS_CSV.exists():
        raise HTTPException(status_code=500, detail=f"Missing product CSV at {PRODUCTS_CSV}")
    try:
        df = pd.read_csv(PRODUCTS_CSV)
    except Exception as exc:
        logger.exception("Failed to read products CSV")
        raise HTTPException(status_code=500, detail=f"Failed to read products CSV: {exc}")

    # Normalise SKU column to string and strip whitespace
    if "sku" in df.columns:
        df["sku"] = df["sku"].astype(str).str.strip()
    return df


# Models
class QuoteItem(BaseModel):
    sku: str = Field(..., description="Product SKU")
    qty: int = Field(..., ge=1, description="Quantity")


class QuoteRequest(BaseModel):
    items: List[QuoteItem]
    include_gst: bool = Field(True, description="If true, includes GST totals (AU 10%)")
    customer_postcode: Optional[str] = Field(None, description="Optional (used later for delivery zones)")


# Middleware: require API key for non-public paths
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    # Allow docs, openapi, redoc, and related static assets without a key
    public_prefixes = ("/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect", "/static", "/favicon.ico")
    if request.url.path.startswith(public_prefixes):
        return await call_next(request)

    x_api_key = request.headers.get("x-api-key")
    require_key(x_api_key)
    return await call_next(request)


# Endpoints
@app.get("/products/search", summary="Search products by SKU or name")
def search_products(q: str, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    df = load_products()
    q2 = q.strip().lower()
    if not q2:
        return {"results": []}

    mask = (
        df.get("sku", "").astype(str).str.lower().str.contains(q2, na=False)
        | df.get("name", "").astype(str).str.lower().str.contains(q2, na=False)
    )
    out = (
        df[mask]
        .head(20)[["sku", "category", "name", "unit_price_ex_gst", "currency", "status", "review_reason"]]
        .fillna("")
        .to_dict("records")
    )
    return {"results": out}


@app.get("/products/{sku}", summary="Get a product by SKU")
def get_product(sku: str, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    df = load_products()
    row = df[df.get("sku", "").astype(str).str.lower() == sku.strip().lower()]
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
        row = df[df.get("sku", "").astype(str).str.lower() == sku.lower()]
        if row.empty:
            raise HTTPException(status_code=400, detail=f"Unknown SKU: {sku}")

        r = row.iloc[0]
        status = str(r.get("status", "")).strip()
        price = r.get("unit_price_ex_gst", None)

        # pandas may give NaN for blanks
        if status != "OK" or price is None or (isinstance(price, float) and math.isnan(price)):
            reason = str(r.get("review_reason", "")).strip()
            raise HTTPException(
                status_code=400,
                detail=f"SKU {sku} is not safe-to-quote (status={status}, reason={reason}).",
            )

        unit_price = float(price)
        line_total = unit_price * int(it.qty)
        subtotal_ex += line_total

        lines.append(
            {
                "sku": sku,
                "name": str(r.get("name", "")),
                "category": str(r.get("category", "")),
                "qty": int(it.qty),
                "unit_price_ex_gst": round(unit_price, 2),
                "line_total_ex_gst": round(line_total, 2),
                "currency": str(r.get("currency", "AUD")) or "AUD",
            }
        )

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
            "If a SKU is REVIEW_REQUIRED, staff must confirm price before quoting.",
        ],
    }