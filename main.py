from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from typing import List

app = FastAPI(title="Breast Cancer Ensemble API", version="1.0")

# -----------------------
# Load Model on Startup
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "voting_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load feature names used during training (header of processed CSV)
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "processed_breast_cancer.csv"
try:
    feature_names = pd.read_csv(PROCESSED_PATH, nrows=0).columns.tolist()
except Exception:
    feature_names = None

# -----------------------
# Request / Response Schemas
# -----------------------
class PredictionRequest(BaseModel):
    features: List[float]   # One row of feature values in correct order


class PredictionResponse(BaseModel):
    prediction: float    


# -----------------------
# Health Check
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------
# Predict Endpoint
# -----------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Validate feature length when feature names are available
        if feature_names is not None and len(request.features) != len(feature_names):
            raise ValueError(f"expected {len(feature_names)} features, got {len(request.features)}")

        df = pd.DataFrame([request.features], columns=feature_names) if feature_names is not None else pd.DataFrame([request.features])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][int(pred)]

        return {
            "prediction": int(pred),
            "probability": float(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------
# Batch Prediction
# -----------------------
class BatchRequest(BaseModel):
    samples: List[List[float]]


@app.post("/predict-batch")
def predict_batch(request: BatchRequest):
    try:
        if feature_names is not None:
            for i, s in enumerate(request.samples):
                if len(s) != len(feature_names):
                    raise ValueError(f"sample {i} expected {len(feature_names)} features, got {len(s)}")
            df = pd.DataFrame(request.samples, columns=feature_names)
        else:
            df = pd.DataFrame(request.samples)
        preds = model.predict(df)
        probs = model.predict_proba(df).max(axis=1)

        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
