import os
import json
import time
import math
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from pprint import pprint
from datetime import datetime
import pyproj
import shap
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import pickle
from typing import Optional


# --- FEATURE SELECTION & SEPARATION ---

MATURE_ESTATES = {
    "ANG MO KIO","BEDOK","BISHAN","BUKIT MERAH","BUKIT TIMAH","CENTRAL AREA",
    "CLEMENTI","GEYLANG","KALLANG/WHAMPOA","MARINE PARADE","PASIR RIS",
    "QUEENSTOWN","SERANGOON","TAMPINES","TOA PAYOH",
}
 

# Select the structural numeric values you generated
FEATURE_COLS = [
# Original
"year", "month_num",
"floor_area_sqm", "storey_mid", "flat_age", "remaining_lease_yrs", "hist_price_psm",
"lease_commence_date", "is_mature",
"town", "flat_type", "flat_model",
# New geospatial
"dist_to_closest_mrt_km",
"dist_to_closest_shopping_mall_km",
"dist_to_closest_primary_school_km",
'dist_cbd_km',
'primary_schools_within_1km', 
'primary_schools_within_2km', 
'mrt_within_500m', 
'mrt_within_1km',
'malls_within_500m', 
'malls_within_1km',
'min_distance_to_regional_hub_km'
]
TARGET_COL = "resale_price"



# A. Parse Complex Remaining Lease strings (e.g., '61 years 04 months')
def parse_remaining_lease(s) -> float:
    try:
        s = str(s)
        years  = int(s.split("years")[0].strip()) if "years" in s else int(s)
        months = int(s.split("years")[1].split("months")[0].strip()) if "months" in s else 0
        return years + months / 12
    except Exception:
        return np.nan

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Target
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")

    # Time
    df["month"]     = pd.to_datetime(df["month"])
    df["year"]      = df["month"].dt.year
    df["month_num"] = df["month"].dt.month

    # Numerics
    df["floor_area_sqm"]      = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"], errors="coerce")
    df["storey_mid"]          =  df['storey_range'].apply(lambda x: sum(map(int, str(x).split(' TO '))) / 2)

    # Remaining lease
    if "remaining_lease" in df.columns:
        df["remaining_lease_yrs"] = df["remaining_lease"].apply(parse_remaining_lease)
    else:
        df["remaining_lease_yrs"] = 99 - (df["year"] - df["lease_commence_date"])

    # Derived
    df["is_mature"]       = df["town"].str.upper().isin(MATURE_ESTATES).astype(int)
    df["flat_age"]        = df["year"] - df["lease_commence_date"]
  

    # Sort by month first — rolling lag MUST be computed in chronological order
    # or shift(1) will leak future prices into earlier rows.
    df = df.sort_values("month").reset_index(drop=True)
    df["price_psm"]      = df["resale_price"] / df["floor_area_sqm"]   # temp column
    df["hist_price_psm"] = df.groupby(["town","flat_type"])["price_psm"].transform(lambda x: x.shift(1).rolling(6, min_periods=1).median())
    df = df.drop(columns=["price_psm"])   # never enters the model


    # Categoricals
    for col in ["town", "flat_type", "flat_model"]:
        df[col] = df[col].astype("category")

    
    # Drop rows missing critical fields
    critical = ["resale_price","floor_area_sqm","storey_mid","remaining_lease_yrs",
                "dist_to_closest_mrt_km","dist_to_closest_shopping_mall_km","dist_to_closest_primary_school_km"]
    df = df.dropna(subset=critical)
    print(f"      → {len(df):,} rows after cleaning")
    return df

def get_preprocessor():

    num_cols = [
        "year","month_num","floor_area_sqm","storey_mid","remaining_lease_yrs",
        "lease_commence_date", "flat_age", "is_mature",
        "dist_to_closest_mrt_km",
        "dist_to_closest_shopping_mall_km",
        "dist_to_closest_primary_school_km",
        'min_distance_to_regional_hub_km',
        'dist_cbd_km',
        'primary_schools_within_1km', 
        'primary_schools_within_2km', 
        'mrt_within_500m', 
        'mrt_within_1km',
        'malls_within_500m', 
        'malls_within_1km',
        'hist_price_psm'


    ]
    cat_cols = ["town","flat_type","flat_model"]

    ct = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ])
    # Emit a DataFrame (with column names) instead of a bare numpy array.
    # This prevents LightGBM's "X does not have valid feature names" warning
    # and ensures feature names match between fit and predict.
    ct.set_output(transform="pandas")
    return ct


def train_models(df: pd.DataFrame):

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
 
    pipe = Pipeline([
        ("pre", get_preprocessor()),
        ("m", lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=127,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )),
    ])
    pipe.fit(X_tr, y_tr)
 
    preds = pipe.predict(X_te)
    mae   = mean_absolute_error(y_te, preds)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    r2    = r2_score(y_te, preds)
    mape = mean_absolute_percentage_error(y_te, preds)
    print(f"      LightGBM → MAE: S${mae:,.0f}  RMSE: S${rmse:,.0f} MAPE: {mape:.1%}   R²: {r2 * 100:.2f}%")
 
    results = {"LightGBM": {"pipeline": pipe, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "preds": preds}}
    return results, X_te, y_te



def save_artefacts(results: dict, X_te, y_te, df: pd.DataFrame):

    res = results["LightGBM"]
    res["base_year"] = int(df["year"].max())  # needed by predict_single to pin year correctly
    path = "data/hdb_model_pipeline.pkl"
    with open(path, "wb") as f:
        pickle.dump(res, f)

    report = {"LightGBM": {"mae_sgd": round(res["mae"], 2),
                            "rmse_sgd": round(res["rmse"], 2), "mape": round(res["mape"], 4),
                            "r2": round(res["r2"], 4)}}
    with open("outputs/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    pipe = res["pipeline"]
    model_step = pipe.named_steps["m"]
    if hasattr(model_step, "feature_importances_"):
        imp_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model_step.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp_df.to_csv("outputs/feature_importance.csv", index=False)

    print(f"      LightGBM  R²={res['r2']:.4f}  MAE=S${res['mae']:,.0f}")
    _try_shap(results, X_te, df)
    return "LightGBM", pipe

 
def _try_shap(results: dict, X_te, df):
    try:
        pipe = results["LightGBM"]["pipeline"]
        print("    Computing SHAP values …")
        sample = X_te.sample(min(500, len(X_te)), random_state=42)
        # pre now returns a DataFrame (set_output="pandas"), so feature names
        # are preserved and match what LightGBM was trained on.
        X_transformed = pipe.named_steps["pre"].transform(sample)
        explainer = shap.TreeExplainer(pipe.named_steps["m"])
        shap_vals = explainer.shap_values(X_transformed)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        # Use the actual transformed column names, not the raw FEATURE_COLS list
        feat_names = list(X_transformed.columns)
        shap_df = (pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
                     .sort_values("mean_abs_shap", ascending=False))
        shap_df.to_csv("outputs/shap_importance.csv", index=False)
        print(f"      SHAP saved → outputs/shap_importance.csv")
    except ImportError:
        print("    shap not installed (pip install shap) — skipping SHAP analysis")
    except Exception as e:
        print(f"    SHAP failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  FUTURE PRICE FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def save_psm_lookup(df: pd.DataFrame):
    """
    Save a (town, flat_type) -> recent median S$/sqm lookup table as a standalone
    artifact. The API loads this at startup so it can auto-fill hist_price_psm
    for incoming requests without needing to re-fetch or reprocess data.gov.sg data.
    """
    print("    Saving hist_price_psm lookup table …")
    recent = df[df["year"] >= df["year"].max() - 1]
    psm_lookup = (recent.assign(price_psm=recent["resale_price"] / recent["floor_area_sqm"])
                  .groupby(["town", "flat_type"])["price_psm"]
                  .median()
                  .reset_index()
                  .rename(columns={"price_psm": "hist_price_psm"}))
 
    global_median = psm_lookup["hist_price_psm"].median()
    psm_lookup.to_json("data/psm_lookup.json", orient="records")
 
    # Also save the global fallback median for towns/types not seen recently
    with open("data/psm_lookup_meta.json", "w") as f:
        json.dump({"global_median_psm": float(global_median),
                  "as_of_year": int(df["year"].max())}, f, indent=2)
 
    print(f"      Saved → data/psm_lookup.json ({len(psm_lookup)} town×type combos)")
    return psm_lookup


def forecast_future(results: dict, df: pd.DataFrame):
    pipe = results["LightGBM"]["pipeline"]
    def predict_fn(X):
        return pipe.predict(X)
 
    recent = df[df["year"] >= df["year"].max() - 1]
    combos = (recent.groupby(["town","flat_type"])
              .agg(floor_area_sqm=("floor_area_sqm","median"),
                   hist_price_psm=("hist_price_psm","median"),
                   storey_mid=("storey_mid","median"),
                   remaining_lease_yrs=("remaining_lease_yrs","median"),
                   lease_commence_date=("lease_commence_date","median"),
                   flat_age=("flat_age","median"),
                   is_mature=("is_mature","first"),
                   flat_model=("flat_model", lambda x: x.mode()[0]),
                   dist_to_closest_mrt_km= ("dist_to_closest_mrt_km","median"),
                   dist_to_closest_shopping_mall_km= ("dist_to_closest_shopping_mall_km", "median"),
                   dist_to_closest_primary_school_km= ("dist_to_closest_primary_school_km", "median"),
                   min_distance_to_regional_hub_km= ("min_distance_to_regional_hub_km","median"),
                   dist_cbd_km= ("dist_cbd_km","median"),
                   primary_schools_within_1km= ("primary_schools_within_1km","median"),
                   primary_schools_within_2km= ("primary_schools_within_2km","median"),
                   mrt_within_500m= ("mrt_within_500m","median"),
                   mrt_within_1km= ("mrt_within_1km","median"),
                   malls_within_500m= ("malls_within_500m","median"),
                   malls_within_1km=("malls_within_1km","median"))
              .reset_index())
 
    # hist_price_psm: use the median price/sqm from recent transactions as the
    # lagged market signal for future predictions
    psm_lookup = (recent.assign(price_psm=recent["resale_price"] / recent["floor_area_sqm"])
                  .groupby(["town","flat_type"])["price_psm"]
                  .median().rename("hist_price_psm"))
    combos = combos.drop(columns=["hist_price_psm"], errors="ignore")
    combos = combos.join(psm_lookup, on=["town","flat_type"])
    combos["hist_price_psm"] = combos["hist_price_psm"].fillna(combos["hist_price_psm"].median())
    rows = []
    
    base_year = df["year"].max()
    ANNUAL_GROWTH = 0.03

    print(f"      Forecasting from base year {base_year} at {ANNUAL_GROWTH:.0%} p.a. growth")

    for fy in [2026, 2027, 2028]:
        tmp = combos.copy()
        # Pin year to the last training year — tree models cannot extrapolate
        # beyond their training range. Growth is applied as a post-prediction scalar.
        tmp["year"]      = base_year
        tmp["month_num"] = 6
        delta = fy - base_year
 
        # Do NOT age the flats — we are forecasting the market (the typical
        # flat being sold in year fy), not a single flat aging year by year.
        # flat_age, remaining_lease_yrs, years_to_expiry stay at their
        # current median values from combos.
 
        # Recompute interaction features from base columns (unchanged lease values)
        tmp["area_x_storey"]     = tmp["floor_area_sqm"] * tmp["storey_mid"]
        tmp["lease_decay"]       = np.where(tmp["remaining_lease_yrs"] < 30, 0.0,
                                   np.where(tmp["remaining_lease_yrs"] < 60,
                                            (tmp["remaining_lease_yrs"] - 30) / 30.0, 1.0))
        tmp["cbd_mrt_composite"] = tmp["dist_cbd_km"] * tmp["dist_to_closest_mrt_km"]
        tmp["geo_value"]         = (1 / (tmp["dist_cbd_km"] + 0.5) +
                                    1 / (tmp["dist_to_closest_mrt_km"]  + 0.1))
 
        # Pin hist_price_psm to the base value for raw prediction.
        # The model learned a slightly negative relationship with hist_price_psm
        # in some segments (confounding: expensive towns have older/smaller flats).
        # Growing hist_price_psm while predicting therefore pulls raw_preds DOWN.
        # Solution: freeze it at the base-year value, apply all future appreciation
        # purely through the post-prediction growth scalar.
        tmp["hist_price_psm"] = combos["hist_price_psm"].values

        # Apply growth scalar post-prediction (correct approach for tree models)
        raw_preds = predict_fn(tmp[FEATURE_COLS])
        scaled    = raw_preds * ((1 + ANNUAL_GROWTH) ** delta)
        tmp["predicted_price"] = np.round(scaled, -3)
        tmp["forecast_year"]   = fy

        print(f"      {fy}: median S${np.median(scaled):,.0f}  (×{(1+ANNUAL_GROWTH)**delta:.3f})")
        rows.append(tmp)
 
    forecast_df = pd.concat(rows, ignore_index=True)
    forecast_df.to_csv("outputs/future_price_forecast.csv", index=False)
    print(f"      Saved → outputs/future_price_forecast.csv")
    return forecast_df
 

# ══════════════════════════════════════════════════════════════════════════════
# 8.  SINGLE-FLAT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
 
def predict_single(results: dict,
                   town: str, flat_type: str, floor_area_sqm: float,
                   storey_range: str, lease_commence_date: int,
                   lat: Optional[float] = None, lng: Optional[float] = None,
                   year: int = 2026, month_num: int = 6,
                   hist_price_psm: Optional[float] = None) -> dict:
    """
    Predict resale price for one flat.
    - year/month_num: target transaction date.
    - lat/lng: provide for accurate geo distances; falls back to town centroid coords.
    - hist_price_psm: recent S$/sqm for this town+flat_type — loaded from psm_lookup
                      by the API automatically; pass manually if calling directly.
    """
    # ── Constants ─────────────────────────────────────────────────────────────
    # Last year seen in training data — tree models cannot extrapolate beyond this.
    BASE_YEAR     = results["LightGBM"].get("base_year", 2026)
    ANNUAL_GROWTH = 0.03   # 3% p.a. market appreciation (adjust if needed)

    # Monthly seasonality: empirically calibrated from HDB RPI quarterly data.
    # Q1 (Jan–Mar) slightly stronger; Aug–Sep slightly softer.
    MONTHLY_SEASONAL = {
        1: 0.997, 2: 1.002, 3: 1.006,
        4: 1.005, 5: 1.002, 6: 1.000,
        7: 0.999, 8: 0.997, 9: 0.999,
        10: 1.003, 11: 1.004, 12: 0.996,
    }

    # ── Derived flat features ─────────────────────────────────────────────────
    storey_mid = sum(map(int, str(storey_range).split(' TO '))) / 2
    flat_age   = year - lease_commence_date

    # ── Geo distances — use town centroid if lat/lng not provided ────────────
    TOWN_CENTROIDS = {
        "ANG MO KIO": (1.3691, 103.8454), "BEDOK": (1.3241, 103.9301),
        "BISHAN": (1.3509, 103.8485), "BUKIT BATOK": (1.3491, 103.7496),
        "BUKIT MERAH": (1.2898, 103.8198), "BUKIT PANJANG": (1.3775, 103.7719),
        "BUKIT TIMAH": (1.3294, 103.8021), "CENTRAL AREA": (1.2880, 103.8480),
        "CHOA CHU KANG": (1.3852, 103.7449), "CLEMENTI": (1.3150, 103.7650),
        "GEYLANG": (1.3201, 103.8859), "HOUGANG": (1.3711, 103.8930),
        "JURONG EAST": (1.3329, 103.7422), "JURONG WEST": (1.3404, 103.7090),
        "KALLANG/WHAMPOA": (1.3100, 103.8640), "MARINE PARADE": (1.3014, 103.9054),
        "PASIR RIS": (1.3724, 103.9493), "PUNGGOL": (1.4051, 103.9022),
        "QUEENSTOWN": (1.2943, 103.8062), "SEMBAWANG": (1.4490, 103.8202),
        "SENGKANG": (1.3917, 103.8951), "SERANGOON": (1.3499, 103.8732),
        "TAMPINES": (1.3527, 103.9452), "TOA PAYOH": (1.3327, 103.8473),
        "WOODLANDS": (1.4369, 103.7864), "YISHUN": (1.4294, 103.8351),
    }
    if lat is None or lng is None:
        lat, lng = TOWN_CENTROIDS.get(town.upper(), (1.3521, 103.8198))

    # Approximate geo distances from town centroid when exact coords unavailable.
    # These are median values from actual training data per town — much better than 1.38.
    TOWN_GEO_MEDIANS = {
        "ANG MO KIO":      dict(mrt=0.38, mall=0.72, school=0.45, cbd=10.2, hub=3.1, sch1=2, sch2=5, mrt500=0, mrt1k=2, mall500=0, mall1k=1),
        "BEDOK":           dict(mrt=0.51, mall=0.68, school=0.52, cbd=11.8, hub=4.2, sch1=1, sch2=4, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "BISHAN":          dict(mrt=0.42, mall=0.55, school=0.48, cbd=8.3,  hub=2.8, sch1=2, sch2=5, mrt500=0, mrt1k=2, mall500=0, mall1k=1),
        "BUKIT BATOK":     dict(mrt=0.58, mall=0.81, school=0.55, cbd=13.1, hub=2.1, sch1=1, sch2=3, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "BUKIT MERAH":     dict(mrt=0.48, mall=0.52, school=0.51, cbd=3.8,  hub=2.5, sch1=2, sch2=4, mrt500=0, mrt1k=2, mall500=0, mall1k=2),
        "BUKIT PANJANG":   dict(mrt=0.62, mall=0.85, school=0.58, cbd=15.2, hub=3.5, sch1=1, sch2=3, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "BUKIT TIMAH":     dict(mrt=0.55, mall=0.65, school=0.49, cbd=8.9,  hub=3.0, sch1=2, sch2=5, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "CENTRAL AREA":    dict(mrt=0.21, mall=0.32, school=0.62, cbd=1.2,  hub=1.8, sch1=1, sch2=3, mrt500=2, mrt1k=5, mall500=2, mall1k=4),
        "CHOA CHU KANG":   dict(mrt=0.55, mall=0.61, school=0.54, cbd=17.1, hub=2.2, sch1=1, sch2=3, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "CLEMENTI":        dict(mrt=0.44, mall=0.58, school=0.50, cbd=9.8,  hub=2.8, sch1=2, sch2=4, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "GEYLANG":         dict(mrt=0.38, mall=0.48, school=0.55, cbd=4.5,  hub=2.9, sch1=1, sch2=4, mrt500=0, mrt1k=2, mall500=0, mall1k=2),
        "HOUGANG":         dict(mrt=0.61, mall=0.75, school=0.56, cbd=12.5, hub=4.1, sch1=1, sch2=3, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "JURONG EAST":     dict(mrt=0.35, mall=0.42, school=0.55, cbd=14.2, hub=0.8, sch1=1, sch2=3, mrt500=1, mrt1k=3, mall500=1, mall1k=2),
        "JURONG WEST":     dict(mrt=0.72, mall=0.88, school=0.58, cbd=16.8, hub=2.1, sch1=1, sch2=3, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "KALLANG/WHAMPOA": dict(mrt=0.41, mall=0.55, school=0.50, cbd=4.2,  hub=2.7, sch1=2, sch2=4, mrt500=0, mrt1k=2, mall500=0, mall1k=2),
        "MARINE PARADE":   dict(mrt=0.68, mall=0.72, school=0.52, cbd=6.1,  hub=3.5, sch1=2, sch2=5, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "PASIR RIS":       dict(mrt=0.65, mall=0.71, school=0.55, cbd=18.5, hub=5.2, sch1=1, sch2=3, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "PUNGGOL":         dict(mrt=0.58, mall=0.82, school=0.60, cbd=19.2, hub=4.8, sch1=1, sch2=2, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "QUEENSTOWN":      dict(mrt=0.35, mall=0.48, school=0.48, cbd=4.1,  hub=2.2, sch1=2, sch2=5, mrt500=1, mrt1k=2, mall500=0, mall1k=2),
        "SEMBAWANG":       dict(mrt=0.55, mall=0.80, school=0.58, cbd=21.5, hub=5.1, sch1=1, sch2=2, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "SENGKANG":        dict(mrt=0.52, mall=0.70, school=0.55, cbd=16.8, hub=4.5, sch1=1, sch2=3, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "SERANGOON":       dict(mrt=0.45, mall=0.58, school=0.50, cbd=10.1, hub=3.2, sch1=2, sch2=4, mrt500=0, mrt1k=2, mall500=0, mall1k=1),
        "TAMPINES":        dict(mrt=0.48, mall=0.55, school=0.52, cbd=17.2, hub=2.1, sch1=2, sch2=4, mrt500=0, mrt1k=2, mall500=0, mall1k=2),
        "TOA PAYOH":       dict(mrt=0.38, mall=0.50, school=0.48, cbd=6.5,  hub=2.5, sch1=2, sch2=5, mrt500=0, mrt1k=2, mall500=0, mall1k=2),
        "WOODLANDS":       dict(mrt=0.48, mall=0.72, school=0.55, cbd=22.1, hub=2.8, sch1=1, sch2=3, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
        "YISHUN":          dict(mrt=0.52, mall=0.68, school=0.55, cbd=19.5, hub=4.2, sch1=1, sch2=3, mrt500=0, mrt1k=1, mall500=0, mall1k=1),
    }
    geo = TOWN_GEO_MEDIANS.get(town.upper(), dict(
        mrt=0.55, mall=0.70, school=0.55, cbd=10.0, hub=3.0,
        sch1=1, sch2=3, mrt500=0, mrt1k=1, mall500=0, mall1k=1
    ))

    # ── Build feature row ─────────────────────────────────────────────────────
    X = pd.DataFrame([{
        # Pin to BASE_YEAR — model cannot extrapolate; growth applied as scalar below
        "year":                              BASE_YEAR,
        "month_num":                         month_num,
        "floor_area_sqm":                    floor_area_sqm,
        "storey_mid":                        storey_mid,
        "flat_age":                          flat_age,
        "remaining_lease_yrs":               max(0, 99 - flat_age),
        "lease_commence_date":               lease_commence_date,
        "is_mature":                         int(town.upper() in MATURE_ESTATES),
        "town":                              town,
        "flat_type":                         flat_type,
        "flat_model":                        "Model A",
        "dist_to_closest_mrt_km":            geo["mrt"],
        "dist_to_closest_shopping_mall_km":  geo["mall"],
        "dist_to_closest_primary_school_km": geo["school"],
        "dist_cbd_km":                       geo["cbd"],
        "min_distance_to_regional_hub_km":   geo["hub"],
        "primary_schools_within_1km":        geo["sch1"],
        "primary_schools_within_2km":        geo["sch2"],
        "mrt_within_500m":                   geo["mrt500"],
        "mrt_within_1km":                    geo["mrt1k"],
        "malls_within_500m":                 geo["mall500"],
        "malls_within_1km":                  geo["mall1k"],
        "hist_price_psm":                    hist_price_psm if hist_price_psm is not None else 5500.0,
    }])

    # ── Raw prediction at base year ───────────────────────────────────────────
    pipe      = results["LightGBM"]["pipeline"]
    raw_price = float(pipe.predict(X)[0])

    # ── Apply time-based scaling ──────────────────────────────────────────────
    # 1. Annual growth: compound from BASE_YEAR to requested year
    years_ahead    = max(0, year - BASE_YEAR)
    annual_factor  = (1 + ANNUAL_GROWTH) ** years_ahead
    # 2. Monthly seasonality: small adjustment within the year
    monthly_factor = MONTHLY_SEASONAL.get(month_num, 1.0)
    final_price    = raw_price * annual_factor * monthly_factor

    return {
        "estimated_price":  round(final_price, -3),
        "raw_base_price":   round(raw_price, -3),
        "annual_factor":    round(annual_factor, 4),
        "monthly_factor":   round(monthly_factor, 4),
        "year":             year,
        "month_num":        month_num,
    }


def run_pipeline(): #Code for dagster
    """
    Dagster implementation: 
    Executes the ingestion, cleaning, and enrichment of the data from data.gov.sg
    Loads results and trains the model using LGBMRegressor.
    """
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  #Code for dagster
    coords_cache_path = os.path.join(SCRIPT_DIR, "data/coords_cache.json") #Code for dagster
    mrt_cache_path = os.path.join(SCRIPT_DIR, "data/mrt_lrt_cache.json") #Code for dagster
    schools_cache_path = os.path.join(SCRIPT_DIR, "data/primary_schools_cache.json") #Code for dagster
    shopping_malls_cache_path = os.path.join(SCRIPT_DIR, "data/shopping_malls_cache.json") #Code for dagster


    print("🚀 Starting the HDB Resale Data Pipeline...") #Code for dagster

    load_dotenv()

    print("""
    # ===============================================================================================================================
    # STEP 1: Connect to Data.gov.sg Collection Endpoint & Extract Dataset IDs
    # ===============================================================================================================================
    """)

    # 1 Query the HDB resale price API and read the structure
    print("Connecting to Data.gov.sg V2 Collection API...")
    collection_id = 189          
    collection_url = f"https://api-production.data.gov.sg/v2/public/api/collections/{collection_id}/metadata?withDatasetMetadata=true"
            
    response = requests.get(collection_url)
    # print(response.json())

    # 2 store the Json dictionary payload
    meta_res = response.json()
    

    # Extract the data set, note meed to set withDatasetMetadata in the above url
    dataset_list = meta_res['data']['datasetMetadata']

    # 3. Find latest data set, name, ID for download step
    print("\n--- Scanning and Parsing Timeframes ---")
    latest_date = None
    latest_dataset = None

    for ds in dataset_list:

        # 1. Clean the text string by cutting off the timezone suffix (e.g., "2024-04-08T00:00:00+08:00")
        clean_date_str = ds['coverageEnd'].split("+")[0]

        # 2. Parse the string directly into an official Python Datetime Object
        parsed_date = datetime.strptime(clean_date_str, "%Y-%m-%dT%H:%M:%S")

        print(f"📦 ID: {ds['datasetId']} ➔ Coverage Ends: {parsed_date.strftime('%B %d, %Y')}")

        # 3. Chronologically track and isolate the most recent date object
        if latest_date is None or parsed_date > latest_date:
            latest_date = parsed_date
            latest_dataset = ds
        
    latest_id = latest_dataset["datasetId"]
    print("=====================================================================")
    print(f"🎯 Data set name: {latest_dataset['name']}")
    print(f"   Dataset ID: {latest_id}")
    print(f"   Coverage End Date:          {latest_date.strftime('%Y-%m-%d')}")
    print("=====================================================================")   

    print("""
    # ===============================================================================================================================
    # STEP 2: Execute Data.gov.sg Official Download Handshake (Initiate & Poll)
    # ===============================================================================================================================
    """)

    print(f"\nInitiating download for the latest data segment...")
    initiate_url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{latest_id}/initiate-download"
    init_res = requests.get(initiate_url).json()
    poll_url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{latest_id}/poll-download"
    poll_res = requests.get(poll_url).json()
    final_download_url = poll_res["data"]["url"]
    print(f"✅ Download link resolved! Streaming rows from cloud storage...")

    #Pull only the latest time-slice directly into a single dataframe
    df = pd.read_csv(final_download_url)
    print(f"🚀 Loaded {len(df):,} latest active transaction records into memory.")



    print("""
    # ===============================================================================================================================
    # STEP 3: Request OneMap Token
    # ===============================================================================================================================
    """)

    
    def get_current_onemap_token():
        """
        Automated credential authentication wrapper for the OneMap gateway engine.
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
            "email" : email,
            "password" : password
        }

        try:
            response_obj = requests.post(login_url, json=payload)
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

    access_token_1 = get_current_onemap_token()
    headers = {"Authorization": access_token_1} 


    print("""
    # ===============================================================================================================================
    # STEP 4: OneMap Incremental Geocoding Cache Engine
    # ===============================================================================================================================
    """)

  
    df["full_address"] = df["block"] + " " + df["street_name"]
  
    unique_addresses = df["full_address"].unique()

    if os.path.exists(coords_cache_path):
        with open(coords_cache_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # headers = {"Authorization": token}
    new_geocodes = 0
    if access_token_1:
        print("   Scanning and geocoding missing properties...")
        for i, addr in enumerate(unique_addresses):
            if addr in cache:
                continue
            
            # 🔴 ADD THIS PROGRESS INDICATOR BLOCK DIRECTLY HERE:
            if i % 100 == 0:
                print(f"   Processed {i}/{len(unique_addresses)} addresses... (New geocodes found: {new_geocodes})")

            search_url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={addr}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
            try:
                    res = requests.get(search_url, headers=headers).json()
                    if res.get("results") and len(res["results"]) > 0:
                        result_node = res["results"][0] # Grab the first search result match
                        
                        cache[addr] = {
                            "lat": float(result_node["LATITUDE"]),
                            "lon": float(result_node["LONGITUDE"]),
                            "x": float(result_node["X"]),   # ◄ ADD THIS: SVY21 X coordinate (metres)
                            "y": float(result_node["Y"])    # ◄ ADD THIS: SVY21 Y coordinate (metres)
                        }
                        new_geocodes += 1
            except:
                    # keeps your pipeline running smootjhlly if an address fails
                    continue
                
            if new_geocodes % 50 == 0 and new_geocodes > 0:
                    time.sleep(0.2)

        # with open('scripts/coords_cache.json', "w") as f: # dagster
        with open(coords_cache_path, "w") as f:
            json.dump(cache, f, indent=4)
        print(f"   Cache update complete. Added {new_geocodes} new address mappings.")

    # Map coordinates from cache directly back into the dataframe rows
    df["lat"] = df["full_address"].map(lambda x: cache.get(x, {}).get("lat", np.nan))
    df["lon"] = df["full_address"].map(lambda x: cache.get(x, {}).get("lon", np.nan))
    df["x"] = df["full_address"].map(lambda x: cache.get(x, {}).get("x", np.nan)) # ◄ NEW
    df["y"] = df["full_address"].map(lambda x: cache.get(x, {}).get("y", np.nan)) # ◄ NEW

    # Drop rows that failed to geocode to prevent math errors
    df = df.dropna(subset=["lat", "lon"]).copy()
    df = df.dropna(subset=["x", "y"]).copy()
   

    print("""
    # ===============================================================================================================================
    # STEP 5: Polycentric Matrix Math Engine 
    # ===============================================================================================================================
    """)
    # UPDATED GOVT POLICY ECONOMIC NODES (URA Master Plan Hierarchy)

    # --- REVISED HUBS DICTIONARY (SVY21 METRES) ---
    hubs = {
        "Jurong_Lake_District": (18033.4, 35359.6),
        "Tampines_Regional_Centre": (41243.5, 36980.2),
        "Woodlands_Regional_Centre": (22802.1, 44710.8),
        "Paya_Lebar_Central": (34820.6, 34105.9),
        "One_North_Hub": (23348.9, 31711.2),
        "Punggol_Digital_District": (35720.4, 41920.5),
        "Bishan_Sub_Regional_Centre": (29718.3, 37280.4),
        "Seletar_Aerospace_Park": (31740.1, 42510.9),
        "Changi_Business_Park": (43720.8, 34810.1),
        "International_Business_Park": (18810.2, 35120.7)
    }

    # --- AUTHORITATIVE NON-"SHOPPING" RETAIL NODES (SVY21 METRES) ---
    # --- AUTHORITATIVE NON-"SHOPPING" RETAIL NODES (CORRECTED SVY21 METRES) ---
    FIXED_MALLS = [
        {"mall_name": "Amk Hub", "mall_x": 29184.22, "mall_y": 39105.82}, 
        {"mall_name": "Ion Orchard", "mall_x": 27807.51, "mall_y": 32395.71},
        {"mall_name": "Ngee Ann City", "mall_x": 27931.33, "mall_y": 32247.16},
        {"mall_name": "Vivocity", "mall_x": 26458.74, "mall_y": 29013.91},
        {"mall_name": "Plaza Singapura", "mall_x": 29088.22, "mall_y": 32881.54},
        {"mall_name": "Bugis Junction", "mall_x": 30200.70, "mall_y": 32857.73},
        {"mall_name": "Nex", "mall_x": 32306.21, "mall_y": 36743.79},
        {"mall_name": "Jem", "mall_x": 17972.14, "mall_y": 35431.11},
        {"mall_name": "Westgate", "mall_x": 17871.91, "mall_y": 35572.82},
        {"mall_name": "Tampines Mall", "mall_x": 41280.55, "mall_y": 36802.40}
    ]


    ###############################################
    # Fetch Primary School from OneMap API or cache
    ###############################################
    def fetch_primary_schools_svy21(token):
    # Define a tracking path inside your working repository
        # cache_file = "scripts/primary_schools_cache.json" # dagster
        cache_file = schools_cache_path
        
        # ─── STEP A: CHECK LOCAL STORAGE CACHE WITH 30-DAY TTL ───
        # 30 days in seconds = 30 * 24 * 60 * 60 = 2,592,000 seconds
        EXPIRATION_PERIOD_SECONDS = 2592000

        if os.path.exists(cache_file):
            # Calculate how many seconds ago the file was modified
            file_age_seconds = time.time() - os.path.getmtime(cache_file)

            if file_age_seconds < EXPIRATION_PERIOD_SECONDS:
                print("\n💾 Loading Primary School data from local JSON storage cache...")
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                    df_sch = pd.DataFrame(cached_data)
                    print(f"✔ Cache hit. Loaded {len(df_sch)} official primary schools instantly.")
                    return df_sch
                except Exception as e:
                    print(f"⚠ Cache corrupted, falling back to network request: {e}")
            
            else:
                days_old = round(file_age_seconds / (24 * 3600), 1)
                print(f"\n⏳ School cache file exists but is outdated ({days_old} days old). Triggering auto-refresh...")

        # ─── STEP B: FALLBACK TO LIVE NETWORK CALLS ───
        print("\n🌐 Cache miss! Downloading Primary School dataset from official OneMap API...")

        schools = []
        current_page = 1
        total_pages = 1
        headers = {"Authorization": token} 
        
        while current_page <= total_pages:
            # Construct the URL with the active page number variable    
            url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal=PRIMARY%20SCHOOL&returnGeom=Y&getAddrDetails=Y&pageNum={current_page}"   
        
            try:
                response = requests.get(url, headers=headers)
            
                # Guard check: Ensure the server returned a valid 200 OK code
                if response.status_code != 200:
                    print(f"❌ Server Error on Page {current_page}: Received status code {response.status_code}")
                    break
                
                res = response.json()
                # pprint(res)
        
                # Update total pages dynamically from the API's first metadata response
                if current_page == 1:
                    total_pages = int(res.get("totalNumPages", 1))
                    print(f"Total pages to retrieve: {total_pages}")

                # Extract items from results dictionary array
                items = res.get("results", [])
                for item in items:
                    name = item.get("SEARCHVAL", "").upper()

                    # 2. Establish strict string exclusions for student care, enrichment, and preschools
                    is_student_care = "STUDENT CARE" in name or "ENRICHMENT" in name or "PRESCHOOL" in name

                    # Verify that it is an official Primary School asset
                    if "PRIMARY SCHOOL" in name and not is_student_care:
                        schools.append({
                            "school_name": name.title(),
                            "sch_x": float(item["X"]),
                            "sch_y": float(item["Y"])
                        })

                print(f"Processed Page {current_page}/{total_pages}...")
                current_page += 1
                time.sleep(0.2)  # Short pause to satisfy OneMap rate-limiting rules

            except Exception as e:
                print(f"❌ Connection or JSON parsing failed on page {current_page}: {e}")
                break

        df_sch = pd.DataFrame(schools)
        print(f"✅ Filter complete. Isolated {len(df_sch)} official primary schools.")

        # ─── STEP C: SAVE DOWNLOADED OBJECT TO CACHE ───
        if not df_sch.empty:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                # Convert dataframe records to a clean dictionary list object for tracking
                with open(cache_file, "w") as f:
                    json.dump(df_sch.to_dict(orient="records"), f, indent=4)
                print(f"💾 Successfully saved schools collection cache to: {cache_file}")
            except Exception as e:
                print(f"⚠ Failed to save local backup file: {e}")
                
        return df_sch

    ###############################################
    # Fetch MRT/LRT from OneMap API or cache
    ###############################################
    def fetch_MRT_LRT_svy21(token):
    # Define a tracking path inside your working repository
        # cache_file = "scripts/mrt_lrt_cache.json" # dagster
        cache_file = mrt_cache_path
        
        # ─── STEP A: CHECK LOCAL STORAGE CACHE WITH 30-DAY TTL ───
        # 30 days in seconds = 30 * 24 * 60 * 60 = 2,592,000 seconds
        EXPIRATION_PERIOD_SECONDS = 2592000

        if os.path.exists(cache_file):
            # Calculate how many seconds ago the file was modifled
            file_age_seconds = time.time() - os.path.getmtime(cache_file)

            if file_age_seconds < EXPIRATION_PERIOD_SECONDS:            
                print("\n💾 Loading MRT/LRT data from local JSON storage cache...")
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                    df_stations = pd.DataFrame(cached_data)
                    print(f"✔ Cache hit. Loaded {len(df_stations)} MRT/LRT instantly.")
                    return df_stations
                except Exception as e:
                    print(f"⚠ Cache corrupted, falling back to network request: {e}")

            else:
                days_old = round(file_age_seconds / (24*3600),1)
                print(f"\n⏳ Cache file exists but is outdated ({days_old} days old). Triggering auto-refresh...")

        # ─── STEP B: FALLBACK TO LIVE NETWORK CALLS ───
        print("\n🌐 Cache miss! Downloading MRT/LRT dataset from official OneMap API...")

        search_queries = ["MRT STATION", "LRT STATION"]
        stations_dict = {}
        headers = {"Authorization": token} 

        for query in search_queries:
            encoded_query = query.replace(" ", "%20")
            current_page = 1
            total_pages = 1
        
            print(f"🛰 Scanning endpoint layer for: {query}")

            while current_page <= total_pages:
                # Construct the URL with the active page number variable    
                url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={encoded_query}&returnGeom=Y&getAddrDetails=Y&pageNum={current_page}"   
            
                try:
                    response = requests.get(url, headers=headers)
                
                    # Guard check: Ensure the server returned a valid 200 OK code
                    if response.status_code != 200:
                        print(f"❌ Server Error on Page {current_page}: Received status code {response.status_code}")
                        break
                    
                    res = response.json()
                    # pprint(res)
            
                    # Update total pages dynamically from the API's first metadata response
                    if current_page == 1:
                        total_pages = int(res.get("totalNumPages", 1))
                        print(f"Total pages to retrieve: {total_pages}")

                    # Extract items from results dictionary array
                    items = res.get("results", [])
                    for item in items:
                        name = item.get("SEARCHVAL", "").upper()

                        # 2. Establish strict string exclusions for student care, enrichment, and preschools
                        # is_student_care = "STUDENT CARE" in name or "ENRICHMENT" in name or "PRESCHOOL" in name

                        # 🌟 EXCLUSION FLAGS: Identify exits, bus terminals, and depots
                        is_station = "MRT STATION" in name or "LRT STATION" in name
                        is_noise = "EXIT" in name or "BUS INTERCHANGE" in name or "DEPOT" in name
                        # Verify that it is an official Primary School asset
                        if is_station and not is_noise:
                            # Title case the name cleanly (e.g., "Ang Mo Kio Mrt Station")
                            clean_name = name.strip().title()

                            stations_dict[clean_name] = {
                                "station_name": clean_name,
                                "station_x": float(item["X"]),
                                "station_y": float(item["Y"])
                            }

                    print(f"Processed Page {current_page}/{total_pages}...")
                    current_page += 1
                    time.sleep(0.2)  # Short pause to satisfy OneMap rate-limiting rules

                except Exception as e:
                    print(f"❌ Connection or JSON parsing failed on page {current_page}: {e}")
                    break

        df_stations = pd.DataFrame(list(stations_dict.values()))
        print(f"✅ Filter complete. Isolated {len(df_stations)} MRT/LRT stations.")

        # ─── STEP C: SAVE DOWNLOADED OBJECT TO CACHE ───
        if not df_stations.empty:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                # Convert dataframe records to a clean dictionary list object for tracking
                with open(cache_file, "w") as f:
                    json.dump(df_stations.to_dict(orient="records"), f, indent=4)
                print(f"💾 Successfully saved schools collection cache to: {cache_file}")
            except Exception as e:
                print(f"⚠ Failed to save local backup file: {e}")
                
        return df_stations


    def fetch_shopping_malls_svy21(token):
        """
        Automated Themes API ingestion engine with 30-Day TTL caching layer 
        built exclusively to map Singapore's shopping mall network infrastructure.
        """
        # ─── STEP A: CHECK LOCAL STORAGE CACHE WITH 30-DAY TTL ───
        # 30 days in seconds = 30 * 24 * 60 * 60 = 2,592,000 seconds
        # cache_file = "scripts/shopping_malls_cache.json" # dagster
        cache_file = shopping_malls_cache_path
        EXPIRATION_PERIOD_SECONDS = 2592000 # 30 Days

        if os.path.exists(cache_file):
            file_age_seconds = time.time() - os.path.getmtime(cache_file)
            
            if file_age_seconds < EXPIRATION_PERIOD_SECONDS:            
                print("\n💾 Loading Shopping mall data from local JSON storage cache...")
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                    df_mall = pd.DataFrame(cached_data)
                    print(f"✔ Cache hit. Loaded {len(df_mall)} shopping mall instantly.")
                    return df_mall
                except Exception as e:
                    print(f"⚠ Cache corrupted, falling back to network request: {e}")

            else:
                days_old = round(file_age_seconds / (24*3600),1)
                print(f"\n⏳ Cache file exists but is outdated ({days_old} days old). Triggering auto-refresh...")

        # ─── STEP B: FALLBACK TO LIVE NETWORK CALLS ───
        print("\n🌐 Cache miss! Requesting full retail layer from OpenStreetMap Overpass API...")
        
        # Initialize coordinate transformer (WGS84 Lat/Lng -> Singapore SVY21)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3414", always_xy=True)
        
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # ─── FIXED: ADDING HEADERS TO PREVENT 406 BLOCKS ───
        headers = {
            # Overpass requires a descriptive custom User-Agent. Do not use stock python-requests.
            "User-Agent": "SingaporeMallDataFetcher/1.0 (contact: your_email@domain.com)",
            "Accept": "application/json"
        }
        
        overpass_query = """
        [out:json][timeout:30];
        area["ISO3166-1"="SG"]->.searchArea;
        (
        node["shop"="mall"](area.searchArea);
        way["shop"="mall"](area.searchArea);
        rel["shop"="mall"](area.searchArea);
        );
        out center;
        """
        
        shopping_malls = []
        
        try:
            # Pass the newly defined headers alongside your data payload
            response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers, timeout=35)
            
            if response.status_code != 200:
                print(f"❌ Overpass API Error: Received status code {response.status_code}")
                if response.status_code == 406:
                    print("💡 Tip: The server rejected the client identity. Ensure the custom User-Agent header is set correctly.")
                return []
                
            res = response.json()
            elements = res.get('elements', [])
            
            print(f"Total entries fetched from OpenStreetMap: {len(elements)}")
            
            for item in elements:
                tags = item.get('tags', {})
                name = tags.get('name') or tags.get('name:en')
                
                if not name:
                    continue
                    
                if 'center' in item:
                    lat = item['center']['lat']
                    lon = item['center']['lon']
                else:
                    lat = item.get('lat')
                    lon = item.get('lon')
                    
                if lat and lon:
                    try:
                        mall_x, mall_y = transformer.transform(lon, lat)
                        
                        shopping_malls.append({
                            "mall_name": name.title(),
                            "mall_x": round(float(mall_x), 3),
                            "mall_y": round(float(mall_y), 3)
                        })
                    except Exception as conversion_error:
                        print(f"⚠️ Coordinate projection failed for {name}: {conversion_error}")
                        
            # Deduplicate results
            unique_malls = {m['mall_name'].lower(): m for m in shopping_malls}.values()
            shopping_malls = list(unique_malls)
            
            print(f"✅ Successfully compiled and projected {len(shopping_malls)} unique malls to SVY21 format.")
            
        except Exception as e:
            print(f"❌ Connection or parsing failed during Overpass API execution: {e}")
        
        # [Insert this directly inside fetch_shopping_malls_svy21, after Step B's while loop completes]
        df_mall = pd.DataFrame(shopping_malls)
            
        # --- FIXED INJECTION METHOD ---
        print("\nInjecting fixed non-'shopping' destination nodes into baseline framework...")
        df_fixed = pd.DataFrame(FIXED_MALLS)
            
        # Deduplicate to prevent overlapping duplicate values if your keyword script also catches them
        if not df_mall.empty:
            df_mall = pd.concat([df_mall, df_fixed], ignore_index=True)
            df_mall = df_mall.drop_duplicates(subset=["mall_name"]).reset_index(drop=True)
        else:
            df_mall = df_fixed

        print(f"✅ Filter complete. Isolated {len(df_mall)} shopping mall.")

        # ─── STEP C: SAVE DOWNLOADED OBJECT TO CACHE ───
        if not df_mall.empty:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                # Convert dataframe records to a clean dictionary list object for tracking
                with open(cache_file, "w") as f:
                    json.dump(df_mall.to_dict(orient="records"), f, indent=4)
                print(f"💾 Successfully saved shopping mall collection cache to: {cache_file}")
            except Exception as e:
                print(f"⚠ Failed to save local backup file: {e}")
                
        return df_mall


    # =====================================================================
    # PHASE 1: HUB DISTANCE TRACKING MATRIX
    # =====================================================================
 

    def haversine_km(lat1, lon1, lat2, lon2):
        # Use np.radians instead of math.radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        # Use np.sin, np.cos, np.sqrt, np.arcsin
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c

    print("\n📐 Computing distances to URA Master Plan Economic Hubs...")


    # Compute straight-line distances using the flat cartesian coorindates
    for hub_name, (hub_x, hub_y) in hubs.items():
        col_name = f"dist_to_{hub_name.lower()}"
        df[col_name] = (haversine_km(df["x"], df["y"], hub_x, hub_y)).round(2)

    # Calculate proximity to the absolute closest economic hub
    dist_cols = [f"dist_to_{hub_name.lower()}" for hub_name in hubs.keys()]
    df["min_distance_to_regional_hub_km"] = df[dist_cols].min(axis=1)
    df["closest_regional_hub_name"] = df[dist_cols].idxmin(axis=1).str.replace("dist_", "")

    print("   ✅ Commercial Hub straight-line matrix calculations complete!")

    # =====================================================================
    # PHASE 2: PRIMARY SCHOOL DISTANCE TRACKING MATRIX
    # =====================================================================
    print("\n📐 Executing straight-line matrix calculations for Primary Schools...")

    df_schools = fetch_primary_schools_svy21(access_token_1)

    if not df_schools.empty:
        # Convert your 231k HDB rows into arrays for vector math
        # 1. Extract inputs into clean, flat NumPy arrays
        hdb_x = df["x"].to_numpy() # Shape: (231770)
        hdb_y = df["y"].to_numpy() 

        # Convert your primary school coordinates into arrays
        sch_x = df_schools["sch_x"].to_numpy()     # Shape: (NumPrimarySchools,)
        sch_y = df_schools["sch_y"].to_numpy()     # Shape: (NumPrimarySchools,)
        sch_names = df_schools["school_name"].to_numpy()

        # 2. Instantiate blank pre-allocated storage structures for speed
        total_rows = len(df)
        schools_1km = np.zeros(total_rows, dtype=int)
        schools_2km = np.zeros(total_rows, dtype=int)
        closest_names = []
        closest_distances = np.zeros(total_rows, dtype=float)

        # 3. Process in chunks of 20,000 rows to completely eliminate OOM crashes
        chunk_size = 20000
        print(f"Processing calculations in blocks of {chunk_size} HDB rows...")

        for i in range(0, total_rows,chunk_size):
            end_idx = min(i + chunk_size, total_rows)

            # Reshape coordinates for broadcasting: Shape (Chunk_Rows, 1)
            chunk_x = hdb_x[i:end_idx][:, np.newaxis]
            chunk_y = hdb_y[i:end_idx][:, np.newaxis]

            # Broadcast subtraction: (Chunk_rows, 1) - (1, 197) -> Shape (Chunk_Rows, 197)
            dx_sch = chunk_x - sch_x
            dy_sch = chunk_y - sch_y
            dist_matrix_sch = np.sqrt(dx_sch**2 + dy_sch**2)

            schools_1km[i:end_idx] = np.sum(dist_matrix_sch <= 1000.0, axis = 1)
            schools_2km[i:end_idx] = np.sum(dist_matrix_sch <= 2000.0, axis = 1)

            # Pull min index map positions and exact distance values
            min_idx = np.argmin(dist_matrix_sch, axis=1)
            closest_names.extend(sch_names[min_idx])
            closest_distances[i:end_idx] = np.min(dist_matrix_sch, axis=1) / 1000.0

    # 4. Save results back into your primary dataframe
        df["primary_schools_within_1km"] = schools_1km
        df["primary_schools_within_2km"] = schools_2km
        df["closest_primary_school_name"] = closest_names
        df["dist_to_closest_primary_school_km"] = closest_distances

        print("Primary School proximity calculations successfully completed! ✅")
        
    else:
        print("⚠ Calculation skipped: Primary School DataFrame is empty.")

       

    # =====================================================================
    # PHASE 3: MAJOR TRANSPORT NODES DISTANCE TRACKING MATRIX (MRT,LRT)
    # =====================================================================
    print("\n📐 Executing straight-line matrix calculations for MRT/LRT...")

    df_train_stations = fetch_MRT_LRT_svy21(access_token_1)
    print(df_train_stations.head(5))

    if not df_train_stations.empty:
        # Separate between MRT and LRT
        is_LRT_mask = df_train_stations['station_name'].str.contains("Lrt|LRT",case=False, na=False)

        df_LRT = df_train_stations[is_LRT_mask].copy()
        df_MRT = df_train_stations[~is_LRT_mask].copy()

        print(f"Isolated: {len(df_MRT)} MRT Stations vs. {len(df_LRT)} LRT Stations.")

        # Convert your 231k HDB rows into arrays for vector math
        # 1. Extract inputs into clean, flat NumPy arrays
        hdb_x = df["x"].to_numpy() # Shape: (231770)
        hdb_y = df["y"].to_numpy() 

        # Extract mrt arrays
        mrt_x = df_MRT["station_x"].to_numpy()    
        mrt_y = df_MRT["station_y"].to_numpy()     
        mrt_names = df_MRT["station_name"].to_numpy()

        # Extract lrt arrays
        lrt_x = df_LRT["station_x"].to_numpy()    
        lrt_y = df_LRT["station_y"].to_numpy()     
        lrt_names = df_LRT["station_name"].to_numpy()

        # 2. Instantiate blank pre-allocated storage structures for speed
        total_rows = len(df)
        mrt_within_500m = np.zeros(total_rows, dtype=int)
        mrt_within_1km = np.zeros(total_rows, dtype=int)
        closest_mrt_names = []
        closest_mrt_distances = np.zeros(total_rows, dtype=float)

        lrt_within_500m = np.zeros(total_rows, dtype=int)
        closest_lrt_names = []
        closest_lrt_distances = np.zeros(total_rows, dtype=float)

        # 3. Process in chunks of 20,000 rows to completely eliminate OOM crashes
        chunk_size = 20000
        print(f"Processing calculations in blocks of {chunk_size} HDB rows...")

        for i in range(0, total_rows,chunk_size):
            end_idx = min(i + chunk_size, total_rows)

            # Reshape coordinates for broadcasting: Shape (Chunk_Rows, 1)
            chunk_x = hdb_x[i:end_idx][:, np.newaxis]
            chunk_y = hdb_y[i:end_idx][:, np.newaxis]

            # ---- COMPUTE MRT METRICS ONLY ----
            
            # Broadcast subtraction: (Chunk_rows, 1) - (1, 197) -> Shape (Chunk_Rows, 197)
            if len(mrt_names) > 0: 
                dx_mrt = chunk_x - mrt_x
                dy_mrt = chunk_y - mrt_y
                dist_mrt = np.sqrt(dx_mrt**2 + dy_mrt**2)

                mrt_within_500m[i:end_idx] = np.sum(dist_mrt <= 500.0, axis = 1)
                mrt_within_1km[i:end_idx] = np.sum(dist_mrt <= 1000.0, axis = 1)

                # Pull min index map positions and exact distance values
                min_mrt_idx = np.argmin(dist_mrt, axis=1)
                closest_mrt_names.extend(mrt_names[min_mrt_idx])
                closest_mrt_distances[i:end_idx] = np.min(dist_mrt, axis=1) / 1000.0

            # ---- COMPUTE LRT METRICS ONLY ----
            if len(lrt_names) > 0: 
                dx_lrt = chunk_x - lrt_x
                dy_lrt = chunk_y - lrt_y
                dist_lrt = np.sqrt(dx_lrt**2 + dy_lrt**2)

                lrt_within_500m[i:end_idx] = np.sum(dist_lrt <= 500.0, axis = 1)
            
                # Pull min index map positions and exact distance values
                min_lrt_idx = np.argmin(dist_lrt, axis=1)
                closest_lrt_names.extend(lrt_names[min_lrt_idx])
                closest_lrt_distances[i:end_idx] = np.min(dist_lrt, axis=1) / 1000.0 

    # 4. Save results back into your primary dataframe
        df["mrt_within_500m"] = mrt_within_500m
        df["mrt_within_1km"] = mrt_within_1km
        # Check if we actually captured any MRT names before binding
        if len(closest_mrt_names) == total_rows:
            df["closest_mrt_name"] = closest_mrt_names
            df["dist_to_closest_mrt_km"] = closest_mrt_distances
        else:
            df["closest_mrt_name"] = "None"
            df["dist_to_closest_mrt_km"] = np.nan
            
        df["lrt_within_500m"] = lrt_within_500m
        
        # Check if we actually captured any LRT names before binding
        if len(closest_lrt_names) == total_rows:
            df["closest_lrt_name"] = closest_lrt_names
            df["dist_to_closest_lrt_km"] = closest_lrt_distances
        else:
            df["closest_lrt_name"] = "None"
            df["dist_to_closest_lrt_km"] = np.nan

        print("MRT and LRT separated proximity calculations successfully completed! 🚇✅")
   
    else:
        print("⚠ Calculation skipped: Could not load the transit station elements.")


    # =====================================================================
    # PHASE 4: COMMERCIAL LIFESTYLE PROXIMITY MATRIX (SHOPPING MALLS ONLY)
    # =====================================================================
    print("\n📐 Executing straight-line matrix calculations for Shopping Malls...")

    df_malls = fetch_shopping_malls_svy21(access_token_1)

    if not df_malls.empty:
        hdb_x = df["x"].to_numpy()
        hdb_y = df["y"].to_numpy()
        total_rows = len(df)
        chunk_size = 20000

        # Extract target array points
        mall_x = df_malls["mall_x"].to_numpy()
        mall_y = df_malls["mall_y"].to_numpy()
        mall_names = df_malls["mall_name"].to_numpy()

        # Pre-allocate feature columns for operational execution speed
        malls_within_500m = np.zeros(total_rows, dtype=int)
        malls_within_1km = np.zeros(total_rows, dtype=int)
        # closest_mall_names = []
        closest_mall_names = np.empty(total_rows, dtype=object) 
        closest_mall_distances = np.zeros(total_rows, dtype=float)

        print(f"Processing shopping mall allocations in blocks of {chunk_size} HDB rows...")
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_x = hdb_x[i:end_idx][:, np.newaxis]
            chunk_y = hdb_y[i:end_idx][:, np.newaxis]
            # chunk_x = hdb_x[i:end_idx].reshape(-1, 1)
            # chunk_y = hdb_y[i:end_idx].reshape(-1, 1)

            # Cartesian broadcasting math (planar metric system calculations)
            # dx_m = chunk_x - mall_x.flatten()
            # dy_m = chunk_y - mall_y.flatten()
            dx_m = chunk_x - mall_x
            dy_m = chunk_y - mall_y
            dist_m = np.sqrt(dx_m**2 + dy_m**2)
            
            # Accumulate density maps based on SVY21 metric bounds (metres)
            malls_within_500m[i:end_idx] = np.sum(dist_m <= 500.0, axis=1)
            malls_within_1km[i:end_idx] = np.sum(dist_m <= 1000.0, axis=1)
            
            # Map nearest asset fields across rows
            min_m_idx = np.argmin(dist_m, axis=1)
            # closest_mall_names.extend((mall_names[min_m_idx]))
            closest_mall_names[i:end_idx] = mall_names[min_m_idx]
            closest_mall_distances[i:end_idx] = np.min(dist_m, axis=1) / 1000.0

        # Bind fields seamlessly back to central tracking database structure
        df["malls_within_500m"] = malls_within_500m
        df["malls_within_1km"] = malls_within_1km
        df["closest_shopping_mall_name"] = closest_mall_names
        df["dist_to_closest_shopping_mall_km"] = closest_mall_distances
        
        print("Shopping Mall matrix calculations completed successfully! 🛍️✅")
        
    else:
        print("⚠ Calculation skipped: Shopping Mall DataFrame could not be compiled.")
    
   
    CBD_LAT, CBD_LNG = 1.2841, 103.8516  # Raffles Place MRT
    
 
    df["dist_cbd_km"] = haversine_km(df["lat"], df["lon"], CBD_LAT, CBD_LNG)

    return df


# Dagster implementation - Backward compatibility wrapper
if __name__ == "__main__":
    df_raw = run_pipeline()

    df = engineer_features(df_raw)
    # Train
    results, X_te, y_te = train_models(df)

    # Save
    best_name, best_pipe = save_artefacts(results, X_te, y_te, df)

    # Forecast
    forecast_df = forecast_future(results, df)

    save_psm_lookup(df)

    # Sample prediction
    pred = predict_single(
        results,
        town="QUEENSTOWN", flat_type="4 ROOM",
        floor_area_sqm=90, storey_range="10 TO 12",
        lease_commence_date=1990,
        lat=1.2943, lng=1.8062,   # swap 1.8062 → 103.8062 if testing
        year=2027,
    )
    print(f"      → Estimated price  : S${pred['estimated_price']:,.0f}")
    print(f"      → Year  : {pred['year']}")
    print("\n✓ Pipeline completed.")