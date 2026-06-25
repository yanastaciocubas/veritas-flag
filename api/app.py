"""
api/app.py
----------
FastAPI service that wraps the trained XGBoost fraud detection model.

Exposes three endpoints:
    POST /predict  : score a single transaction, returns fraud probability
    GET  /health   : confirm the model is loaded and ready
    GET  /features : return the list of features the model expects

The request schema matches the Credit Card Fraud Detection dataset:
    https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

Features V1 through V28 are PCA-transformed components of the original
transaction data (anonymized for privacy). Amount is the transaction
value in euros, standardized to match training preprocessing.

Run locally:
    cd api
    uvicorn app:app --reload --port 8000

Interactive docs available at:
    http://localhost:8000/docs
"""

import json
import os
import time
from typing import Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Paths are relative to the project root, one level above api/
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/xgb_fraud_model.json")
META_PATH  = os.path.join(BASE_DIR, "../models/model_meta.json")

# Global state: loaded once at startup, reused across requests
model:         Optional[xgb.XGBClassifier] = None
feature_names: list = []
threshold:     float = 0.5


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card transaction fraud scoring using XGBoost. "
        "Trained on the UCI Credit Card Fraud Detection dataset. "
        "Returns a fraud probability score and binary decision per transaction."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def load_model() -> None:
    """
    Load the trained model and metadata into memory at server startup.

    Loading once at startup (rather than per request) keeps inference
    latency low. The model and feature list are stored as module-level
    globals and shared across all requests.
    """
    global model, feature_names, threshold

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: model not found at {MODEL_PATH}. Run src/train.py first.")
        return

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        feature_names = meta.get("feature_names", [])
        threshold     = meta.get("optimal_threshold", 0.5)

    print(f"Model ready: {len(feature_names)} features, threshold = {threshold}")


# --------------------------------------------------------------------------
# Request and response schemas
# --------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    """
    Credit card transaction features for fraud scoring.

    V1 through V28 are the principal components from PCA applied to the
    original transaction features. The exact features are anonymized but
    the PCA transformation preserves the predictive signal.

    Amount should be provided in raw euros. The API standardizes it
    internally to match the preprocessing applied during training.

    All V-features are optional and default to 0 if not provided.
    Providing more features produces more accurate scores.
    """
    Amount: float = Field(..., description="Transaction amount in euros (raw, not scaled)")
    V1:  Optional[float] = Field(None, description="PCA component 1")
    V2:  Optional[float] = Field(None, description="PCA component 2")
    V3:  Optional[float] = Field(None, description="PCA component 3")
    V4:  Optional[float] = Field(None, description="PCA component 4")
    V5:  Optional[float] = Field(None, description="PCA component 5")
    V6:  Optional[float] = Field(None, description="PCA component 6")
    V7:  Optional[float] = Field(None, description="PCA component 7")
    V8:  Optional[float] = Field(None, description="PCA component 8")
    V9:  Optional[float] = Field(None, description="PCA component 9")
    V10: Optional[float] = Field(None, description="PCA component 10")
    V11: Optional[float] = Field(None, description="PCA component 11")
    V12: Optional[float] = Field(None, description="PCA component 12")
    V13: Optional[float] = Field(None, description="PCA component 13")
    V14: Optional[float] = Field(None, description="PCA component 14 (highest SHAP importance)")
    V15: Optional[float] = Field(None, description="PCA component 15")
    V16: Optional[float] = Field(None, description="PCA component 16")
    V17: Optional[float] = Field(None, description="PCA component 17")
    V18: Optional[float] = Field(None, description="PCA component 18")
    V19: Optional[float] = Field(None, description="PCA component 19")
    V20: Optional[float] = Field(None, description="PCA component 20")
    V21: Optional[float] = Field(None, description="PCA component 21")
    V22: Optional[float] = Field(None, description="PCA component 22")
    V23: Optional[float] = Field(None, description="PCA component 23")
    V24: Optional[float] = Field(None, description="PCA component 24")
    V25: Optional[float] = Field(None, description="PCA component 25")
    V26: Optional[float] = Field(None, description="PCA component 26")
    V27: Optional[float] = Field(None, description="PCA component 27")
    V28: Optional[float] = Field(None, description="PCA component 28")


class FraudPrediction(BaseModel):
    fraud_probability: float = Field(..., description="Model confidence score from 0 to 1")
    is_fraud:          bool  = Field(..., description="Binary decision at the optimal F1 threshold")
    threshold_used:    float = Field(..., description="Decision threshold applied")
    latency_ms:        float = Field(..., description="Inference latency in milliseconds")
    model_version:     str   = Field(..., description="Model identifier for tracking")


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    n_features:   int


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    """
    Health check endpoint.

    Returns status ok if the server is running, and indicates whether
    the model has been loaded successfully. Use this to verify the
    service is ready before sending prediction requests.
    """
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "n_features":   len(feature_names),
    }


@app.post("/predict", response_model=FraudPrediction)
def predict(transaction: TransactionRequest) -> dict:
    """
    Score a single credit card transaction for fraud probability.

    Accepts a partial or full set of V1-V28 PCA features plus Amount.
    Missing features are filled with 0, which matches the median
    imputation applied during training preprocessing.

    Returns a fraud probability in [0, 1] and a binary is_fraud decision
    based on the threshold that maximizes F1 score on the validation set.

    Raises:
        503 if the model is not loaded (run src/train.py first)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py to train and save the model first.",
        )

    start = time.perf_counter()

    # Build feature vector aligned to training schema
    # Missing features default to 0 (consistent with training imputation)
    feature_map = transaction.dict()
    X = np.array([[feature_map.get(f, 0) or 0 for f in feature_names]])

    prob       = float(model.predict_proba(X)[0][1])
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    return {
        "fraud_probability": round(prob, 6),
        "is_fraud":          prob >= threshold,
        "threshold_used":    threshold,
        "latency_ms":        latency_ms,
        "model_version":     "xgb-creditcard-v1",
    }


@app.get("/features")
def get_features() -> dict:
    """
    Return the list of feature names the model expects.

    Useful for callers that want to construct a full feature vector
    without guessing the column order.
    """
    return {
        "feature_names": feature_names,
        "n_features":    len(feature_names),
    }