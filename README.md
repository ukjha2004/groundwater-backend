# Groundwater Quality ML API

FastAPI backend for the UDAAN UG Fellowship project
**Machine Learning Approaches for Water Quality Testing** (TEXMiN).
Serves a Gradient Boosting model that predicts uranium concentration from
12 groundwater and hydrogeological parameters.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status check (used by the website to detect cold starts) |
| GET | `/summary` | Dataset headline numbers |
| GET | `/latest-wells` | Latest observation and prototype risk for each well |
| GET | `/analytics` | Aggregated data for charts |
| GET | `/model-info` | Group-CV metrics, feature importance, input ranges, risk thresholds |
| POST | `/predict` | Prediction; inputs outside the training range return HTTP 422 |
| GET | `/monitoring-data` | Full dataset (legacy, ~1.3 MB) |

Interactive docs: `/docs`.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python train_model.py          # retrain + save metrics into the .pkl
uvicorn main:app --reload      # http://localhost:8000/docs
```

## Deploy (Render)

- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Python version: `.python-version` (3.10.8)
- Optional env var `ALLOWED_ORIGINS` (comma-separated) to allow extra frontend URLs.
  The Vercel site and `localhost:5173` are allowed by default.

**Important:** train the `.pkl` with the same scikit-learn version pinned in
`requirements.txt` (1.7.2), or it may not load on Render.

## Limitations

Synthetic/hypothetical dataset; not field-validated. Risk thresholds
(5 / 10 ppm) are project-defined prototypes, not regulatory limits.
