import json
import os
import joblib
import pandas as pd

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

THRESHOLD = 0.50

def model_fn(model_dir: str):
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)

    if isinstance(payload, dict) and "instances" in payload:
        instances = payload["instances"]
    else:
        instances = payload

    if isinstance(instances, dict):
        instances = [instances]

    if not isinstance(instances, list) or not all(isinstance(x, dict) for x in instances):
        raise ValueError("JSON must be a dict, a list of dicts, or {'instances': [..]}")

    df = pd.DataFrame(instances)

    # Ensure expected columns exist
    for c in DEPLOY_FEATURES:
        if c not in df.columns:
            df[c] = None

    return df[DEPLOY_FEATURES]

def predict_fn(input_data, model):
    proba = model.predict_proba(input_data)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return {"probabilities": proba.tolist(), "predictions": pred.tolist(), "threshold": THRESHOLD}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
