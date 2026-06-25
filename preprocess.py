"""
preprocess.py
-------------
Loads and prepares the Credit Card Fraud Detection dataset for training.

Dataset: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

Columns:
    Time:    seconds elapsed between this transaction and the first transaction
    V1-V28:  anonymized PCA features (real transaction features, privacy-protected)
    Amount:  transaction amount in euros
    Class:   target label: 1 = fraud, 0 = legitimate

Class imbalance: 492 fraud cases out of 284,807 transactions (0.17%).
SMOTE is applied to the training set only to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the creditcard.csv dataset from disk.

    Args:
        csv_path: path to creditcard.csv

    Returns:
        DataFrame with 284,807 rows and 31 columns
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]:,} transactions, {df.shape[1]} columns")
    print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
    return df


def preprocess(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
    scale_amount: bool = True,
) -> tuple:
    """
    Full preprocessing pipeline for creditcard.csv.

    Steps:
        1. Load data
        2. Drop the Time column (not predictive, just ordering)
        3. Optionally scale Amount (V1-V28 are already PCA-scaled)
        4. Stratified train/test split (preserves 0.17% fraud ratio)
        5. Apply SMOTE to training set only

    Args:
        csv_path:      path to creditcard.csv
        test_size:     fraction of data held out for testing (default 0.2)
        random_state:  random seed for reproducibility (default 42)
        apply_smote:   whether to apply SMOTE oversampling (default True)
        scale_amount:  whether to standardize the Amount column (default True)

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = load_data(csv_path)

    # Drop Time: it reflects transaction ordering within the dataset,
    # not a feature a real-time scoring system would have access to
    df = df.drop(columns=["Time"])

    # Scale Amount: V1-V28 are already PCA-transformed and roughly scaled,
    # but Amount is raw euros and needs standardization
    if scale_amount:
        scaler = StandardScaler()
        df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    feature_names = X.columns.tolist()
    print(f"Features: {len(feature_names)} columns")

    # Stratified split: preserves the fraud ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    fraud_rate = y_train.mean() * 100
    print(f"Training fraud rate before SMOTE: {fraud_rate:.3f}%")
    print(f"Training set size: {len(X_train):,} samples")

    if apply_smote:
        print("Applying SMOTE to training set...")
        sm = SMOTE(random_state=random_state, n_jobs=-1)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"Training set after SMOTE: {len(X_train):,} samples "
              f"({y_train.mean()*100:.1f}% fraud)")

    return X_train, X_test, y_train, y_test, feature_names