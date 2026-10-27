"""
Groundwater Quality ML API
--------------------------
FastAPI backend for the UDAAN UG Fellowship project
"Machine Learning Approaches for Water Quality Testing".

Endpoints
    GET  /                 API info
    GET  /health           lightweight status check (used by the frontend)
    GET  /summary          dataset headline numbers for the dashboard
    GET  /latest-wells     latest observation + risk for every monitoring well
    GET  /analytics        pre-aggregated data for the Analytics page
    GET  /model-info       model metrics, feature importance, input ranges, thresholds
    POST /predict          uranium prediction for one set of inputs
    GET  /monitoring-data  full dataset (kept for backward compatibility)
"""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =====================================================
# PATHS
# =====================================================

# Resolve files relative to this script, so the API works no matter
# which folder it is started from.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "groundwater_quality_model_final.pkl"
DATASET_PATH = BASE_DIR / "groundwater_monitoring_dataset_3000.csv"


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Groundwater Quality ML API",
    description="API for groundwater uranium prediction (research prototype)",
    version="1.1",
)


# =====================================================
# CORS
# =====================================================
# Only the deployed frontend and local development may call the API.
# Extra origins can be added on Render with the ALLOWED_ORIGINS
# environment variable (comma-separated).

DEFAULT_ORIGINS = [
    "https://groundwater-frontend-one.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

extra_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS + extra_origins,
    # Vercel preview deployments, e.g. groundwater-frontend-git-xyz.vercel.app
    allow_origin_regex=r"https://groundwater-frontend[a-z0-9-]*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =====================================================
# RISK THRESHOLDS (single source of truth)
# =====================================================
# Project-defined prototype thresholds, NOT regulatory limits.
# The frontend reads these from /model-info instead of hard-coding them.

MODERATE_THRESHOLD_PPM = 5.0
HIGH_THRESHOLD_PPM = 10.0


def classify_risk(uranium_value: float) -> str:
    if uranium_value < MODERATE_THRESHOLD_PPM:
        return "Low Risk"
    if uranium_value < HIGH_THRESHOLD_PPM:
        return "Moderate Risk"
    return "High Risk"


# =====================================================
# LOAD ML MODEL
# =====================================================

model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
features = model_package["features"]


# =====================================================
# LOAD DATASET
# =====================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"{DATASET_PATH.name} not found in the backend folder."
    )

dataset = pd.read_csv(DATASET_PATH)

# Dates in the CSV have nanosecond precision
# ("2023-01-01 08:45:46.515505168"), which browsers do not parse reliably.
# Convert them once to clean ISO strings ("2023-01-01T08:45:46").
dataset["Date"] = pd.to_datetime(dataset["Date"], format="mixed")
dataset = dataset.sort_values("Date").reset_index(drop=True)
dataset["Date"] = dataset["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
dataset["Risk"] = dataset["Uranium_ppm"].apply(classify_risk)


# =====================================================
# INPUT RANGES
# =====================================================
# The model was trained only on values inside these ranges, so
# predictions outside them are extrapolation and are rejected.

FEATURE_INFO = {
    "Distance_km": {"label": "Distance from ISR Site", "unit": "km"},
    "pH": {"label": "pH", "unit": ""},
    "TDS": {"label": "TDS", "unit": "mg/L"},
    "Conductivity": {"label": "Electrical Conductivity", "unit": "µS/cm"},
    "Hydraulic_K": {"label": "Hydraulic Conductivity", "unit": "m/day"},
    "Temperature_C": {"label": "Temperature", "unit": "°C"},
    "Dissolved_Oxygen": {"label": "Dissolved Oxygen", "unit": "mg/L"},
    "Nitrate": {"label": "Nitrate", "unit": "mg/L"},
    "Rainfall_mm": {"label": "Rainfall", "unit": "mm"},
    "Aquifer_Depth_m": {"label": "Aquifer Depth", "unit": "m"},
    "Soil_Permeability": {"label": "Soil Permeability", "unit": "m/s"},
    "Porosity": {"label": "Porosity", "unit": "fraction"},
}

def _round_outward(value, up, significant=3):
    """Round to 3 significant figures, away from the data (so the
    displayed range always contains every training value)."""
    if value == 0:
        return 0.0
    digits = significant - int(np.floor(np.log10(abs(value)))) - 1
    factor = 10.0 ** digits
    rounded = (np.ceil if up else np.floor)(value * factor) / factor
    return float(round(rounded, max(digits, 0)))


FEATURE_RANGES = {
    feature: {
        "min": _round_outward(dataset[feature].min(), up=False),
        "max": _round_outward(dataset[feature].max(), up=True),
        "default": float(dataset[feature].median()),
    }
    for feature in features
}


# =====================================================
# MODEL METRICS
# =====================================================
# Preferred: metrics saved inside the .pkl by train_model.py.
# Fallback (older .pkl files): compute them on the first request that
# needs them (not at startup, to keep Render cold starts short), then cache.

@lru_cache(maxsize=1)
def _compute_metrics():
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold, cross_validate

    scores = cross_validate(
        clone(model),
        dataset[features],
        dataset["Uranium_ppm"],
        groups=dataset["Well_ID"],
        cv=GroupKFold(n_splits=5),
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
    )
    return {
        "validation": "5-Fold Group Cross-Validation (grouped by Well_ID)",
        "r2_mean": float(np.mean(scores["test_r2"])),
        "r2_std": float(np.std(scores["test_r2"])),
        "mae_mean": float(-np.mean(scores["test_mae"])),
        "rmse_mean": float(-np.mean(scores["test_rmse"])),
    }


def get_metrics():
    return model_package.get("metrics") or _compute_metrics()

feature_importance = sorted(
    [
        {
            "feature": feature,
            "label": FEATURE_INFO.get(feature, {}).get("label", feature),
            "importance_pct": round(float(value) * 100, 2),
        }
        for feature, value in zip(features, model.feature_importances_)
    ],
    key=lambda item: item["importance_pct"],
    reverse=True,
)


# =====================================================
# INPUT MODEL
# =====================================================

class WaterQualityInput(BaseModel):
    Distance_km: float
    pH: float
    TDS: float
    Conductivity: float
    Hydraulic_K: float
    Temperature_C: float
    Dissolved_Oxygen: float
    Nitrate: float
    Rainfall_mm: float
    Aquifer_Depth_m: float
    Soil_Permeability: float
    Porosity: float


def latest_observation_per_well() -> pd.DataFrame:
    # dataset is already sorted by Date, so tail(1) is the latest row
    return dataset.groupby("Well_ID").tail(1).sort_values("Well_ID")


def r(value, digits=3):
    return round(float(value), digits)


# =====================================================
# ROUTES
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Groundwater Quality ML API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/summary")
def summary():
    latest = latest_observation_per_well()
    return {
        "total_samples": int(len(dataset)),
        "total_wells": int(dataset["Well_ID"].nunique()),
        "average_uranium_ppm": r(dataset["Uranium_ppm"].mean()),
        "maximum_uranium_ppm": r(dataset["Uranium_ppm"].max()),
        "date_start": dataset["Date"].iloc[0][:10],
        "date_end": dataset["Date"].iloc[-1][:10],
        "latest_risk_counts": {
            risk: int((latest["Risk"] == risk).sum())
            for risk in ["Low Risk", "Moderate Risk", "High Risk"]
        },
    }


@app.get("/latest-wells")
def latest_wells():
    latest = latest_observation_per_well()
    wells = [
        {
            "id": row.Well_ID,
            "date": row.Date,
            "lat": r(row.Latitude, 6),
            "lon": r(row.Longitude, 6),
            "distance_km": r(row.Distance_km),
            "uranium_ppm": r(row.Uranium_ppm),
            "pH": r(row.pH, 2),
            "TDS": r(row.TDS, 2),
            "conductivity": r(row.Conductivity, 2),
            "risk": row.Risk,
        }
        for row in latest.itertuples()
    ]
    wells.sort(key=lambda well: well["uranium_ppm"], reverse=True)
    return {"wells": wells}


@app.get("/analytics")
def analytics():
    risk_distribution = [
        {"name": risk, "count": int((dataset["Risk"] == risk).sum())}
        for risk in ["Low Risk", "Moderate Risk", "High Risk"]
    ]

    well_averages = (
        dataset.groupby("Well_ID")["Uranium_ppm"]
        .mean()
        .sort_values(ascending=False)
    )

    return {
        "risk_distribution": risk_distribution,
        "well_averages": [
            {"well": well, "uranium_ppm": r(value, 2)}
            for well, value in well_averages.items()
        ],
        "distance_scatter": [
            {"distance_km": r(d, 2), "uranium_ppm": r(u, 2)}
            for d, u in zip(dataset["Distance_km"], dataset["Uranium_ppm"])
        ],
    }


@app.get("/model-info")
def model_info():
    params = model.get_params()
    return {
        "algorithm": type(model).__name__,
        "config": {
            "n_estimators": params.get("n_estimators"),
            "learning_rate": params.get("learning_rate"),
            "max_depth": params.get("max_depth"),
            "random_state": params.get("random_state"),
        },
        "metrics": get_metrics(),
        "feature_importance": feature_importance,
        "features": [
            {"name": feature, **FEATURE_INFO.get(feature, {}), **FEATURE_RANGES[feature]}
            for feature in features
        ],
        "risk_thresholds": {
            "moderate_ppm": MODERATE_THRESHOLD_PPM,
            "high_ppm": HIGH_THRESHOLD_PPM,
            "note": "Project-defined prototype thresholds, not regulatory limits.",
        },
        "dataset": {
            "observations": int(len(dataset)),
            "wells": int(dataset["Well_ID"].nunique()),
            "features": len(features),
        },
    }


@app.post("/predict")
def predict(data: WaterQualityInput):
    values = data.model_dump()

    # Reject values outside the training range (model would extrapolate).
    out_of_range = []
    for feature in features:
        bounds = FEATURE_RANGES[feature]
        if not bounds["min"] <= values[feature] <= bounds["max"]:
            label = FEATURE_INFO.get(feature, {}).get("label", feature)
            out_of_range.append(
                f"{label} must be between {bounds['min']:g} and {bounds['max']:g}"
            )

    if out_of_range:
        raise HTTPException(status_code=422, detail=out_of_range)

    input_data = pd.DataFrame([[values[f] for f in features]], columns=features)
    prediction = float(model.predict(input_data)[0])

    return {
        "predicted_uranium_ppm": r(prediction),
        "risk": classify_risk(prediction),
    }


@app.get("/monitoring-data")
def monitoring_data():
    """Full dataset. The website no longer uses this (it is ~1.3 MB);
    it is kept so older clients and the API docs still work."""
    return {
        "total_samples": int(len(dataset)),
        "total_wells": int(dataset["Well_ID"].nunique()),
        "average_uranium_ppm": r(dataset["Uranium_ppm"].mean()),
        "maximum_uranium_ppm": r(dataset["Uranium_ppm"].max()),
        "data": dataset.to_dict(orient="records"),
    }
