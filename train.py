import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEPLOY_FEATURES = [
    "age",
    "gender",
    "daily_gaming_hours",
    "weekly_sessions",
    "years_gaming",
    "competitive_rank",
    "online_friends",
    "microtransactions_spending",
    "screen_time_total",
    "sleep_hours",
    "stress_level",
    "depression_score",
]

TARGET_SOURCE_COL = "addiction_level"
TARGET_COL = "high_addiction"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantile", type=float, default=0.80)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    files = list(Path(train_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {train_dir}")
    csv_path = str(files[0])

    df = pd.read_csv(csv_path, low_memory=False)

    # Validate required columns
    required = set(DEPLOY_FEATURES + [TARGET_SOURCE_COL])
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create special target exactly like your Final Project
    threshold = float(df[TARGET_SOURCE_COL].quantile(args.quantile))
    df[TARGET_COL] = (df[TARGET_SOURCE_COL] >= threshold).astype(int)

    X = df[DEPLOY_FEATURES].copy()
    y = df[TARGET_COL].copy()

    # Column types
    cat_cols = ["gender"]
    num_cols = [c for c in DEPLOY_FEATURES if c not in cat_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )

    lr = LogisticRegression(max_iter=1000)

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", lr)
    ])

    # 70/15/15 split (stratified) — same as your notebook
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=args.random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.random_state, stratify=y_temp
    )

    model.fit(X_train, y_train)

    # Quick accuracy (kept lightweight inside training job)
    val_acc = float((model.predict(X_val) == y_val).mean())
    test_acc = float((model.predict(X_test) == y_test).mean())

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)

    # Save for inference container
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

    with open(os.path.join(model_dir, "deploy_features.json"), "w") as f:
        json.dump(DEPLOY_FEATURES, f, indent=2)

    with open(os.path.join(model_dir, "high_addiction_threshold.txt"), "w") as f:
        f.write(str(threshold))

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(
            {"val_accuracy": val_acc, "test_accuracy": test_acc, "quantile": args.quantile, "threshold": threshold},
            f,
            indent=2
        )

    print("Saved model to:", model_dir)
    print("Threshold:", threshold)
    print("Val acc:", val_acc, "Test acc:", test_acc)


if __name__ == "__main__":
    main()
