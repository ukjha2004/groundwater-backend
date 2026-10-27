"""
Train the uranium prediction model and save it with its validation metrics.

Run from this folder:
    python train_model.py

IMPORTANT: run it with the same scikit-learn version that is pinned in
requirements.txt (1.7.2), otherwise the saved .pkl may not load on Render.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_validate

DATASET = "groundwater_monitoring_dataset_3000.csv"
OUTPUT = "groundwater_quality_model_final.pkl"

FEATURES = [
    "Distance_km",
    "pH",
    "TDS",
    "Conductivity",
    "Hydraulic_K",
    "Temperature_C",
    "Dissolved_Oxygen",
    "Nitrate",
    "Rainfall_mm",
    "Aquifer_Depth_m",
    "Soil_Permeability",
    "Porosity",
]

data = pd.read_csv(DATASET)
X = data[FEATURES]
y = data["Uranium_ppm"]
groups = data["Well_ID"]

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

# ---- Validation: 5-fold Group CV, so test wells are never seen in training
scores = cross_validate(
    clone(model),
    X,
    y,
    groups=groups,
    cv=GroupKFold(n_splits=5),
    scoring={
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    },
)

metrics = {
    "validation": "5-Fold Group Cross-Validation (grouped by Well_ID)",
    "r2_mean": float(np.mean(scores["test_r2"])),
    "r2_std": float(np.std(scores["test_r2"])),
    "mae_mean": float(-np.mean(scores["test_mae"])),
    "rmse_mean": float(-np.mean(scores["test_rmse"])),
}

# ---- Final model on all data
model.fit(X, y)

joblib.dump(
    {"model": model, "features": FEATURES, "metrics": metrics},
    OUTPUT,
)

print("Validation results (5-fold Group CV):")
print(f"  R²   = {metrics['r2_mean']:.3f} ± {metrics['r2_std']:.3f}")
print(f"  MAE  = {metrics['mae_mean']:.3f} ppm")
print(f"  RMSE = {metrics['rmse_mean']:.3f} ppm")
print(f"Model saved as {OUTPUT}")
