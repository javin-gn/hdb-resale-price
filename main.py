from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import httpx
import os
from dotenv import load_dotenv
from feature_generator import FeatureGenerator # Importing from the file in your root folder
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import r2_score
from typing import Optional
app = FastAPI()
templates = Jinja2Templates(directory="templates")
executor = ThreadPoolExecutor()

# --- 1. LOAD ML MODEL ---
# Updated to match the structure in pipeline.py: 
# results = {"LightGBM": {"pipeline": pipe, "mae": mae, ...}}
with open('data/hdb_model_pipeline.pkl', 'rb') as f:
    artifacts = pickle.load(f)

# Change these lines to access the dictionary keys directly
model = artifacts['pipeline']
model_r2 = artifacts['r2']
mae = artifacts['mae']
mape = artifacts['mape']
rmse = artifacts['rmse']
BASE_YEAR = artifacts.get('base_year', 2026)  # last year in training data

# Monthly growth: compound 3% annual growth into monthly steps (~0.247% per month).
# This means selecting any future month always shows a higher price than the previous
# month, which matches user expectations when browsing future valuations.
ANNUAL_GROWTH = 0.03  # 3% p.a. — adjust if needed
MONTHLY_GROWTH_RATE = (1 + ANNUAL_GROWTH) ** (1/12) - 1  # ~0.00247 per month

# --- 2. LOAD LOOKUP DATA EXPLICITLY ---
# This ignores the pickle file for lookups and goes straight to the JSONs
PSM_DATA = {}
GLOBAL_MEDIAN = 5500.0

if os.path.exists('data/psm_lookup.json'):
    with open('data/psm_lookup.json', 'r') as f:
        raw_list = json.load(f)
        # Convert list of dicts to a lookup dictionary: {(town, flat_type): psm}
        PSM_DATA = {(item['town'], item['flat_type']): item['hist_price_psm'] for item in raw_list}

if os.path.exists('data/psm_lookup_meta.json'):
    with open('data/psm_lookup_meta.json', 'r') as f:
        meta = json.load(f)
        GLOBAL_MEDIAN = meta.get("global_median_psm", 5500.0)

async def get_current_onemap_token():
    
    load_dotenv()
    """
    Automated credential authentication wrapper for the OneMap gateway engine using HTTPX (Async).
    """
    print("🔑 Authenticating developer credentials with OneMap Security Gateway...")
    login_url = "https://www.onemap.gov.sg/api/auth/post/getToken"
   
    # Securely retrieve constants from your root `.env` system environment file
    email = os.getenv("ONEMAP_EMAIL")
    password = os.getenv("ONEMAP_PASSWORD")

    if not email or not password:
        raise ValueError(
            "❌ Missing Credentials! Ensure ONEMAP_EMAIL and ONEMAP_PASSWORD "
            "are defined inside your workspace root `.env` configuration file."
        )
    
    payload = {
        "email": email,
        "password": password
    }

    try:
        # Use httpx.AsyncClient() for non-blocking asynchronous requests
        async with httpx.AsyncClient() as client:
            response_obj = await client.post(login_url, json=payload)
            
        if response_obj.status_code != 200:
            print(f"❌ Handshake Denied: Status {response_obj.status_code}")
            return None
        
        response = response_obj.json()
        token = response.get("access_token")
        print("✅ Handshake successful. Valid 72-hour API token generated.")
        return token
    
    except Exception as e:
        print(f"❌ Security gateway transmission failed: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the Bootstrap web page."""
    # New / Correct way
    return templates.TemplateResponse(request, "index.html")


@app.on_event("startup")
async def startup_event():
    """Runs automatically when the FastAPI server starts up."""
    global ONEMAP_TOKEN
    ONEMAP_TOKEN = await get_current_onemap_token()


@app.get("/api/search")
async def search_address(q: str):
    global ONEMAP_TOKEN
    
    if not q or len(q) < 3:
        return []

    # 1. Thread-safe Token Refresh Logic
    if not ONEMAP_TOKEN:
        async with token_lock:
            # Re-check inside the lock to avoid redundant calls
            if not ONEMAP_TOKEN:
                ONEMAP_TOKEN = await get_current_onemap_token()
                if not ONEMAP_TOKEN:
                    raise HTTPException(status_code=500, detail="Authentication failed")

    # 2. API Call with persistent error handling
    url = "https://www.onemap.gov.sg/api/common/elastic/search"
    headers = {"Authorization": ONEMAP_TOKEN, "User-Agent": "FastAPI-Bootstrap-Map-App/1.0"}
    params = {"searchVal": q, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": "1"}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers, timeout=5.0)
            
            # Handle token expiration specifically
            if response.status_code in [401, 403]:
                ONEMAP_TOKEN = None # Force re-auth on next request
                return {"error": "Token expired, please retry."}
                
            response.raise_for_status()
            return response.json().get("results", [])
            
        except httpx.HTTPError:
            return {"error": "OneMap API unreachable"}



@app.get("/api/calculate")
async def calculate_metrics(address: str):
    loop = asyncio.get_event_loop()
    # This runs the blocking genDS function in a separate thread
    metrics = await loop.run_in_executor(executor, FeatureGenerator, address)
    return metrics

@app.get("/api/config")
async def get_config():
    with open('data/hdb_constants.json', 'r') as f:
        data = json.load(f)
    return data

def _resolve_hist_price_psm(town: str, flat_type: str, provided: Optional[float]) -> float:
    if provided is not None:
        return provided
    
    # Use the loaded PSM_DATA dictionary, NOT the model object
    return PSM_DATA.get((town, flat_type), GLOBAL_MEDIAN)

@app.get("/market-price-psm")
def market_price_psm(town: str, flat_type: str):
    town = town.upper().strip()
    flat_type = flat_type.upper().strip()
    
    resolved = _resolve_hist_price_psm(town, flat_type, None)
    
    # Check if the specific key exists in our loaded data
    is_specific = (town, flat_type) in PSM_DATA
    
    return {
        "town": town,
        "flat_type": flat_type,
        "hist_price_psm": resolved,
        "source": "town_flat_type_median" if is_specific else "singapore_wide_fallback",
    }

# Define the schema for incoming data
class ValuationRequest(BaseModel):
    year: int
    month_num: int
    floor_area_sqm: float
    storey_midpoint: float  # Added this
    remaining_lease: int    # Matches index.html
    lease_commence_date: int
    flat_age: int
    is_mature: int = 0      # Ensure this is provided or set a default
    town: str
    flat_type: str
    flat_models: str        # Matches index.html
    dist_to_closest_mrt_km: float
    dist_to_closest_shopping_mall_km: float
    dist_to_closest_primary_school_km: float
    min_distance_to_regional_hub_km: float
    dist_cbd_km: float      # Ensure this is in your index.html payload
    sch_within_1km: int
    sch_within_2km: int
    mrt_within_500m: int
    mrt_within_1km: int
    malls_within_500m: int
    mall_count: int
    hist_price_psm: float 
    

# --- 2. PREDICT ENDPOINT ---
@app.post("/api/predict")
async def predict_valuation(data: ValuationRequest):
    try:
        data_dict = data.model_dump()

        # ── Year pinning ──────────────────────────────────────────────────────
        # LightGBM cannot extrapolate beyond its training range.
        # We pin year to BASE_YEAR for the raw prediction, then apply
        # a growth scalar + monthly seasonality as post-prediction adjustments.
        requested_year  = data_dict["year"]
        requested_month = data_dict["month_num"]
        data_dict["year"] = BASE_YEAR   # pin before sending to model

        df = pd.DataFrame([data_dict])
        df = df.rename(columns={
            "storey_midpoint": "storey_mid",
            "remaining_lease": "remaining_lease_yrs",
            "flat_models":     "flat_model",
            "sch_within_1km":  "primary_schools_within_1km",
            "sch_within_2km":  "primary_schools_within_2km",
            "mall_count":      "malls_within_1km",
        })

        # ── Raw prediction at base year ───────────────────────────────────────
        raw_price = float(model.predict(df)[0])

        # ── Apply time-based scaling ──────────────────────────────────────────
        # Compute total months elapsed from BASE_YEAR Jan as the reference point.
        # Every additional month adds ~0.247% so price always increases forward in time.
        BASE_MONTH    = 1   # January of BASE_YEAR is our reference (factor = 1.0)
        months_elapsed = (requested_year - BASE_YEAR) * 12 + (requested_month - BASE_MONTH)
        months_elapsed = max(0, months_elapsed)
        time_factor   = (1 + MONTHLY_GROWTH_RATE) ** months_elapsed
        final_price   = raw_price * time_factor

        print(f"  predict: year={requested_year} month={requested_month} "
              f"months_elapsed={months_elapsed} factor={time_factor:.4f} "
              f"raw=S${raw_price:,.0f} → S${final_price:,.0f}")

        return {
            "predicted_price": round(final_price, 0),
            "raw_base_price":  round(raw_price, 0),
            "time_factor":     round(time_factor, 4),
            "months_elapsed":  months_elapsed,
            "model_r2":        f"{model_r2 * 100:.2f}%",
            "mae":             f"S${mae:,.0f}",
            "mape":            f"{mape:.1%}",
            "rmse":            f"S${rmse:,.0f}",
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))