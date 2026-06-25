"""
predict.py
----------
Standalone inference script for batch or single-transaction scoring.

Useful for testing the trained model locally without spinning up the
FastAPI server. Loads the saved model and metadata from models/ and
runs predictions against the creditcard.csv feature schema.

Usage:
    python predict.py --single
    python predict.py --csv_path ../data/creditcard.csv
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_PATH = "../models/xgb_fraud_model.json"
META_PATH  = "../models/model_meta.json"


def load_model_and_meta() -> tuple:
    """
    Load the trained XGBoost model and its metadata from disk.

    The metadata file contains feature names and the optimal decision
    threshold computed during evaluation. Both are required for inference.

    Returns:
        model: loaded XGBoost classifier
        meta:  dict with feature_names, optimal_threshold, and training params

    Raises:
        FileNotFoundError if the model file does not exist
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    with open(META_PATH) as f:
        meta = json.load(f)

    print(f"Model loaded: {meta.get('n_features')} features, "
          f"threshold = {meta.get('optimal_threshold', 0.5)}")
    return model, meta


def predict_batch(
    csv_path: str,
    output_path: str = "../outputs/predictions.csv",
) -> pd.DataFrame:
    """
    Score all transactions in a CSV file and save results to disk.

    Aligns input columns to the training feature schema, fills any
    missing columns with 0, and reports per-transaction latency.

    Args:
        csv_path:    path to input CSV (same schema as creditcard.csv)
        output_path: where to save scored results

    Returns:
        DataFrame with columns: fraud_probability, is_fraud
    """
    model, meta = load_model_and_meta()
    feature_names = meta["feature_names"]
    threshold = meta.get("optimal_threshold", 0.5)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} transactions from {csv_path}")

    # Align to training feature schema, fill unknown columns with 0
    X = df.reindex(columns=feature_names, fill_value=0).values

    start = time.perf_counter()
    proba = model.predict_proba(X)[:, 1]
    elapsed = time.perf_counter() - start

    results = pd.DataFrame({
        "fraud_probability": proba.round(6),
        "is_fraud": (proba >= threshold).astype(int),
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)

    n = len(results)
    print(f"Scored {n:,} transactions in {elapsed*1000:.1f}ms "
          f"({elapsed/n*1000:.3f}ms per transaction)")
    print(f"Flagged {results['is_fraud'].sum():,} as fraudulent "
          f"({results['is_fraud'].mean()*100:.3f}%)")
    print(f"Results saved to {output_path}")

    return results


def predict_single(transaction: dict) -> dict:
    """
    Score a single transaction and return the result as a dict.

    This function mirrors what the FastAPI /predict endpoint does
    internally. Useful for quick testing and debugging.

    Args:
        transaction: dict mapping feature names to values.
                     Any missing features default to 0.

    Returns:
        dict with fraud_probability, is_fraud, threshold_used, latency_ms
    """
    model, meta = load_model_and_meta()
    feature_names = meta["feature_names"]
    threshold = meta.get("optimal_threshold", 0.5)

    # Build feature vector aligned to training schema
    X = np.array([[transaction.get(f, 0) or 0 for f in feature_names]])

    start = time.perf_counter()
    prob = float(model.predict_proba(X)[0][1])
    latency_ms = (time.perf_counter() - start) * 1000

    result = {
        "fraud_probability": round(prob, 6),
        "is_fraud":          prob >= threshold,
        "threshold_used":    threshold,
        "latency_ms":        round(latency_ms, 3),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score transactions using the trained fraud detection model."
    )
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="Path to a CSV file to score in batch mode."
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Score a single hardcoded example transaction."
    )
    args = parser.parse_args()

    if args.single:
        # Example transaction: high amount, first-time card, scaled features
        # V14 is the most predictive feature in this dataset per SHAP analysis
        example = {
            "V1":     -2.3,   # strong negative SHAP contribution
            "V2":      1.8,
            "V3":     -1.9,
            "V4":      0.5,
            "V14":    -3.1,   # most predictive feature for fraud
            "V17":    -2.5,
            "Amount":  2500.0, # large transaction amount, standardized at inference
        }
        predict_single(example)

    elif args.csv_path:
        predict_batch(args.csv_path)

    else:
        print("Provide --csv_path or --single flag.")
        print("Example: python predict.py --single")