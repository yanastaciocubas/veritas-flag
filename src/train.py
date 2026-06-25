"""
train.py
--------
Trains an XGBoost classifier on the Credit Card Fraud Detection dataset.

Uses scale_pos_weight as a complement to SMOTE for handling the severe
class imbalance (0.17% fraud rate). Early stopping is evaluated on
PR-AUC rather than accuracy or ROC-AUC, which are misleading metrics
when positives are this rare.

Usage:
    python train.py                          # uses ../data/creditcard.csv
    python train.py ../data/creditcard.csv   # explicit path
"""

import xgboost as xgb
import json
import os
import sys
from datetime import datetime
from preprocess import preprocess

MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_fraud_model.json")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")


def get_scale_pos_weight(y_train) -> float:
    """
    Compute XGBoost's scale_pos_weight parameter.

    Formula: count(negative class) / count(positive class)
    For this dataset that is roughly 577, meaning each fraud sample
    is weighted 577x more than a legitimate transaction during training.

    This is applied on top of SMOTE for two layers of imbalance correction:
    SMOTE resamples at the data level, scale_pos_weight adjusts at the loss level.

    Args:
        y_train: training labels after SMOTE (will be ~50/50 if SMOTE applied)

    Returns:
        scale_pos_weight as a float
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = round(neg / pos, 2)
    print(f"scale_pos_weight: {spw} (neg={neg:,}, pos={pos:,})")
    return spw


def train(
    csv_path: str = "../data/creditcard.csv",
    apply_smote: bool = True,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> tuple:
    """
    Full training pipeline:
        1. Preprocess data (scale Amount, stratified split, SMOTE)
        2. Train XGBoost with early stopping on PR-AUC
        3. Save model and metadata to disk

    Args:
        csv_path:         path to creditcard.csv
        apply_smote:      whether to apply SMOTE oversampling
        n_estimators:     maximum number of boosting rounds
        max_depth:        maximum tree depth
        learning_rate:    step size shrinkage
        subsample:        fraction of samples per tree
        colsample_bytree: fraction of features per tree
        random_state:     random seed for reproducibility

    Returns:
        model, X_test, y_test, feature_names
    """
    print("=" * 55)
    print("FRAUD DETECTION PIPELINE - TRAINING")
    print("=" * 55)

    X_train, X_test, y_train, y_test, feature_names = preprocess(
        csv_path=csv_path,
        apply_smote=apply_smote,
        random_state=random_state,
    )

    spw = get_scale_pos_weight(y_train)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=spw,
        eval_metric="aucpr",         # PR-AUC: right metric for rare fraud events
        early_stopping_rounds=50,    # stop if no improvement for 50 rounds
        random_state=random_state,
        n_jobs=-1,
        verbosity=1,
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    print(f"\nBest iteration: {model.best_iteration}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")

    # Save metadata: feature names and threshold are needed at inference time
    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "dataset": "creditcard.csv (UCI / Kaggle Credit Card Fraud)",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "best_iteration": int(model.best_iteration),
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "apply_smote": apply_smote,
            "random_state": random_state,
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {META_PATH}")

    return model, X_test, y_test, feature_names


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "../data/creditcard.csv"
    train(csv_path=csv)