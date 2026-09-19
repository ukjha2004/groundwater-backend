from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Groundwater Quality ML API",
    description="API for groundwater uranium prediction",
    version="1.0"
)


# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =====================================================
# LOAD ML MODEL
# =====================================================

model_package = joblib.load(
    "groundwater_quality_model_final.pkl"
)

model = model_package["model"]

features = model_package["features"]


# =====================================================
# LOAD DATASET
# =====================================================

DATASET_PATH = Path(
    "groundwater_monitoring_dataset_3000.csv"
)

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        "groundwater_monitoring_dataset_3000.csv "
        "not found in the backend folder."
    )


dataset = pd.read_csv(DATASET_PATH)


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


# =====================================================
# RISK CLASSIFICATION
# =====================================================

def classify_risk(uranium_value):

    if uranium_value < 5:
        return "Low Risk"

    elif uranium_value < 10:
        return "Moderate Risk"

    else:
        return "High Risk"


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "message": "Groundwater Quality ML API is running"
    }


# =====================================================
# MONITORING DATA
# =====================================================

@app.get("/monitoring-data")
def monitoring_data():

    return {
        "total_samples": int(len(dataset)),

        "total_wells": int(
            dataset["Well_ID"].nunique()
        ),

        "average_uranium_ppm": round(
            float(dataset["Uranium_ppm"].mean()),
            3
        ),

        "maximum_uranium_ppm": round(
            float(dataset["Uranium_ppm"].max()),
            3
        ),

        "data": dataset.to_dict(
            orient="records"
        )
    }


# =====================================================
# ML PREDICTION
# =====================================================

@app.post("/predict")
def predict(data: WaterQualityInput):

    input_data = pd.DataFrame(
        [[
            data.Distance_km,
            data.pH,
            data.TDS,
            data.Conductivity,
            data.Hydraulic_K,
            data.Temperature_C,
            data.Dissolved_Oxygen,
            data.Nitrate,
            data.Rainfall_mm,
            data.Aquifer_Depth_m,
            data.Soil_Permeability,
            data.Porosity
        ]],
        columns=features
    )


    prediction = model.predict(
        input_data
    )[0]


    risk = classify_risk(
        prediction
    )


    return {

        "predicted_uranium_ppm": round(
            float(prediction),
            3
        ),

        "risk": risk
    }