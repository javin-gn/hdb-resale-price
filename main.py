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

app = FastAPI()
templates = Jinja2Templates(directory="templates")
executor = ThreadPoolExecutor()

# --- 1. LOAD ML MODEL (Once at startup) ---
# Ensure your model is at data/hdb_model_pipeline.pkl
with open('data/hdb_model_pipeline.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
expected_features = artifacts['trained_features']
global_mean = artifacts['global_price_mean']
encoding_maps = artifacts['target_encoding_maps']


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

# Define the schema for incoming data
class ValuationRequest(BaseModel):
    floor_area_sqm: float
    flat_type: str
    flat_models: str  # Matches your payload key
    storey_range: str
    storey_midpoint: float
    remaining_lease: int
    lease_commence_date: int
    town: str
    month: str
    year: str
    month_num: int
    street_name: str
    dist_to_closest_primary_school_km: float
    min_distance_to_regional_hub_km: float
    dist_to_closest_shopping_mall_km: float
    dist_to_closest_mrt_km: float
    closest_primary_school_name: str
    closest_mrt_name: str
    mrt_within_1km: int
    mrt_within_500m: int
    sch_within_1km: int
    sch_within_2km: int
    mall_count: int
    malls_within_500m: int
    lrt_within_500m: int
    dist_to_closest_lrt_km: float

# --- 2. PREDICT ENDPOINT ---
@app.post("/api/predict")
async def predict_valuation(data: ValuationRequest):
    try:
        # 1. Convert Pydantic model to dict
        data_dict = data.dict()
        
        # 2. Create DataFrame
        df = pd.DataFrame([data_dict])
        
        # 3. Data Engineering (Hardcode/Fill context)
        df['year'] = 2026
        df['month_num'] = 6
        
        # 4. Encoding Logic (Safe Mapping)
        # We ensure that if a category is missing, it gets the global mean 
        # instead of causing a crash with NaN
        for col, mapping in encoding_maps.items():
            if col in df.columns:
                # Use .map() and fill missing with global_mean to prevent failures
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(global_mean)
        
        # Drop raw categorical columns after encoding
        cols_to_drop = [c for c in encoding_maps.keys() if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # 5. One-Hot Encoding for 'town' (or other dummies)
        # We ensure consistent columns matching the training set
        df_encoded = pd.get_dummies(df, columns=['town'], dtype=int)
        
        # 6. Reindex to match the exact features the model was trained on
        # This fills missing dummy columns with 0, preventing "feature mismatch" errors
        X_final = df_encoded.reindex(columns=expected_features, fill_value=0)
        
        # 7. Run Prediction
        prediction = float(model.predict(X_final)[0])
        
        return {"predicted_price": prediction}
    
    except Exception as e:
        # Print the error to your terminal so you can see why it failed
        print(f"--- PREDICTION ERROR ---")
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))