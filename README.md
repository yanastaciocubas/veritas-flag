# veritas-flag

> "Veritas" is Latin for truth. This project is about finding it in transaction data.

A real-time fraud detection system built from scratch. 284,807 credit card transactions. 492 of them fraud. The model's job is to find those 492 without crying wolf on the other 284,315.

This is not a tutorial project, but a full pipeline: raw data in, fraud score out, live API running. Built because I wanted to understand how these systems actually work, not just how they look in a notebook.

---

## What this is

A machine learning pipeline that scores credit card transactions for fraud probability in real time. You send it a transaction. It tells you whether to be worried.

Under the hood: XGBoost trained on PCA-transformed transaction features, SMOTE to handle the severe class imbalance, SHAP to explain every decision, and a FastAPI service that responds in under 100ms.

The dataset is the [UCI Credit Card Fraud Detection dataset](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv), one of the most studied fraud datasets in existence. 0.17% of transactions are fraud. That tiny number is where all the interesting problems live.

---

## Why I built this

False negatives let fraud through. False positives block real people from their money. Both are failures, and they pull in opposite directions.

I wanted to build something that takes that tension seriously. Not just "maximize accuracy" (useless on imbalanced data) but: what is the right metric, what is the right threshold, and how do you explain the decision to a human analyst who has to act on it?

That is what this project is about.

---

## Architecture

```
creditcard.csv
284,807 transactions
492 fraud cases (0.17%)
        |
        v
+---------------------+
|   preprocess.py     |  Drop Time, standardize Amount,
|                     |  stratified 80/20 split
+--------+------------+
         |
         v
+---------------------+
|   SMOTE             |  Synthetic oversampling on training set only.
|                     |  Never touches the test set. No leakage.
+--------+------------+
         |
         v
+---------------------+
|   XGBoost           |  Early stopping on PR-AUC.
|                     |  scale_pos_weight for extra imbalance correction.
+--------+------------+
         |
    +----+----+
    v         v
Evaluate    FastAPI
SHAP +      /predict
metrics     endpoint
```

---

## The decisions that actually matter

**Why PR-AUC and not ROC-AUC**

ROC-AUC looks great on fraud data because getting the negatives right is easy when 99.83% of transactions are legitimate. PR-AUC only cares about how well you find the fraud. It is harder to game and more honest about what the model is actually doing.

**Why SMOTE plus scale_pos_weight**

Two layers of imbalance correction. SMOTE resamples at the data level, creating synthetic fraud examples in feature space. scale_pos_weight adjusts at the gradient level during training. Together they give the model every chance to learn from 492 examples without overfitting to them.

**Why threshold tuning**

The default 0.5 threshold is almost never right for imbalanced data. This pipeline finds the threshold that maximizes F1 on the validation set. For this dataset it lands around 0.23, meaning the model flags a transaction as fraud when it is only 23% confident. That sounds low until you remember how rare fraud is.

**Why SHAP**

Because "the model said so" is not good enough. Every prediction gets decomposed into per-feature contributions. V14 pushed this score up by 0.4. Amount pushed it down by 0.1. That is what a fraud analyst actually needs to investigate.

---

## Tech stack

| What | How |
|------|-----|
| Model | XGBoost 2.0 |
| Imbalance | SMOTE via imbalanced-learn |
| Explainability | SHAP TreeExplainer |
| API | FastAPI + Uvicorn |
| Data | pandas, numpy, scikit-learn |
| Serialization | XGBoost native JSON |

---

## Setup

```bash
git clone https://github.com/yanastaciocubas/veritas-flag.git
cd veritas-flag
pip install -r requirements.txt
```

Get the data (one command, no account needed):

```bash
mkdir -p data && cd data
curl -O https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

Check it downloaded correctly:

```bash
wc -l creditcard.csv   # should print 284808
```

---

## Usage

**Train:**

```bash
cd src
python train.py
```

You will see PR-AUC climb across boosting rounds and land somewhere around 0.85. Early stopping kicks in when it stops improving. The model and its metadata save to `models/`.

**Evaluate and generate SHAP plots:**

```python
from train import train
from evaluate import evaluate

model, X_test, y_test, feature_names = train()
results = evaluate(model, X_test, y_test, feature_names)
```

Four plots land in `outputs/plots/`: PR curve, confusion matrix, SHAP importance bar chart, SHAP beeswarm.

**Score a single transaction:**

```bash
cd src
python predict.py --single
```

**Run the API:**

```bash
cd api
uvicorn app:app --reload --port 8000
```

Docs at `http://localhost:8000/docs`.

---

## API

**POST /predict**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Amount": 2500.00,
    "V14": -3.1,
    "V4": 0.5,
    "V17": -2.5
  }'
```

```json
{
  "fraud_probability": 0.912847,
  "is_fraud": true,
  "threshold_used": 0.2341,
  "latency_ms": 1.83,
  "model_version": "xgb-creditcard-v1"
}
```

**GET /health** confirms the model is loaded and ready

**GET /features** returns the list of features the model expects

---

## What the SHAP analysis found

The most important feature is V14. Strongly negative V14 values are the clearest signal of fraud in this dataset. V14 is a PCA component of the original transaction data, so we cannot know exactly what it encodes, but the model is very sure about it.

V4, V11, V12, V10, and V17 round out the top features. Their SHAP directions are consistent: certain PCA components push toward fraud, others toward legitimate. The beeswarm plot makes this visible.

Amount matters but less than you might expect. Large amounts elevate risk, and so do suspiciously small amounts (a common pattern when testing a stolen card). But the PCA features dominate.

---

## Model performance

On 20% held-out test data, stratified to preserve the 0.17% fraud rate:

| Metric | Value |
|--------|-------|
| PR-AUC | ~0.85 |
| ROC-AUC | ~0.97 |
| F1 (fraud class) | ~0.80 |
| Optimal threshold | ~0.23 |

Results vary slightly between runs because SMOTE is stochastic. Set `random_state=42` everywhere to reproduce exactly.

---

## What I would build next

Rolling features via Kafka and Flink: average spend per card over the last hour, number of unique merchants in the last day. Time-window aggregates are where the real signal lives in production systems but they need streaming infrastructure.

Graph features: fraud rings show up as dense subgraphs in card-device-IP networks. A simple co-occurrence matrix gets you part of the way there before you need a GNN.

Model monitoring: the fraud distribution shifts as attackers adapt. Threshold and drift alerts, automatic retraining triggers.

Docker and Railway deployment: one command to get the API live with a public URL.

---

## Project structure

```
veritas-flag/
├── README.md
├── requirements.txt
├── .gitignore
├── data/               (git-ignored: run the curl command in Setup)
├── models/             (git-ignored: regenerate with train.py)
├── outputs/
│   └── plots/          (SHAP plots and evaluation charts)
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── api/
    └── app.py
```

---
