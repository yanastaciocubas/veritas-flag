"""
evaluate.py
-----------
Evaluates model performance and generates SHAP explainability plots.

Focuses on metrics appropriate for imbalanced fraud detection:
    - PR-AUC (more informative than ROC-AUC when positives are rare)
    - F1, Precision, Recall at the optimal decision threshold
    - Confusion matrix
    - SHAP feature importance (global bar plot and beeswarm)

Outputs are saved to outputs/plots/ as PNG files.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

matplotlib.use("Agg")  # headless rendering, no display required

PLOTS_DIR = "../outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def find_optimal_threshold(y_true, y_proba) -> float:
    """
    Find the classification threshold that maximizes F1 score.

    The default threshold of 0.5 is rarely optimal for imbalanced datasets.
    For fraud data with 0.17% positive rate, the optimal threshold is
    typically much lower, often around 0.2 to 0.4.

    Args:
        y_true:  ground truth labels (0 or 1)
        y_proba: predicted probabilities for the positive class

    Returns:
        threshold as a float
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    print(f"Optimal threshold: {best_threshold:.4f}  (F1 = {f1_scores[best_idx]:.4f})")
    return best_threshold


def evaluate(
    model: xgb.XGBClassifier,
    X_test,
    y_test,
    feature_names: list,
    save_plots: bool = True,
) -> dict:
    """
    Run the full evaluation suite on the held-out test set.

    Steps:
        1. Predict fraud probabilities
        2. Compute ROC-AUC and PR-AUC
        3. Find the F1-optimal decision threshold
        4. Print classification report (precision, recall, F1 per class)
        5. Save PR curve, confusion matrix, and SHAP plots

    Args:
        model:         trained XGBoost classifier
        X_test:        held-out feature matrix
        y_test:        held-out labels
        feature_names: list of feature column names
        save_plots:    whether to save plots to disk (default True)

    Returns:
        dict with roc_auc, pr_auc, optimal_threshold, f1_fraud
    """
    print("\n" + "=" * 55)
    print("EVALUATION")
    print("=" * 55)

    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}  (primary metric for imbalanced fraud detection)")

    threshold = find_optimal_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    results = {
        "roc_auc":           round(roc_auc, 4),
        "pr_auc":            round(pr_auc, 4),
        "optimal_threshold": round(threshold, 4),
        "f1_fraud":          round(f1_score(y_test, y_pred), 4),
    }

    if save_plots:
        _plot_pr_curve(y_test, y_proba, pr_auc)
        _plot_confusion_matrix(y_test, y_pred)
        _plot_shap(model, X_test, feature_names)

    return results


def _plot_pr_curve(y_test, y_proba, pr_auc: float) -> None:
    """
    Save a Precision-Recall curve to outputs/plots/pr_curve.png.

    PR curves are more informative than ROC curves for imbalanced datasets
    because they focus on the minority class (fraud) rather than rewarding
    correct predictions on the majority class (legitimate transactions).
    """
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(8, 5))
    plt.plot(recalls, precisions, color="#e63946", lw=2,
             label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: Fraud Detection")
    plt.legend(loc="upper right")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "pr_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"PR curve saved to {path}")


def _plot_confusion_matrix(y_test, y_pred) -> None:
    """
    Save a confusion matrix to outputs/plots/confusion_matrix.png.

    Rows are actual labels, columns are predicted labels:
        True Negatives  | False Positives
        False Negatives | True Positives

    For fraud detection, False Negatives (missed fraud) are the most
    costly error. The matrix makes this tradeoff visible.
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {path}")


def _plot_shap(model, X_test, feature_names: list) -> None:
    """
    Generate and save two SHAP plots to outputs/plots/.

    shap_importance.png: global feature importance ranked by mean absolute
    SHAP value. Shows which features matter most across the full test set.

    shap_beeswarm.png: beeswarm plot showing both the magnitude and direction
    of each feature's impact. Red points push toward fraud, blue toward legit.

    Uses a 2000-sample subset for speed. SHAP TreeExplainer is exact for
    tree-based models so no approximation error is introduced by subsampling.

    Args:
        model:         trained XGBoost classifier
        X_test:        held-out feature matrix
        feature_names: list of feature column names
    """
    print("\nComputing SHAP values (this may take a moment)...")

    # Subsample for speed: 2000 samples is enough for stable global explanations
    n = min(2000, len(X_test))
    X_sample = X_test[:n]
    if hasattr(X_sample, "values"):
        X_sample = X_sample.values

    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Bar plot: global feature importance
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=20,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance (Top 20)")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "shap_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP importance plot saved to {path}")

    # Beeswarm plot: feature impact direction
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.title("SHAP Beeswarm: Feature Impact on Fraud Score")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "shap_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP beeswarm plot saved to {path}")