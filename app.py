import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(APP_DIR, "artifacts")

FEATURES_PATH = os.path.join(ARTIFACT_DIR, "deploy_features.json")
BOUNDS_PATH = os.path.join(ARTIFACT_DIR, "input_bounds.json")
THRESH_PATH = os.path.join(ARTIFACT_DIR, "high_addiction_threshold.txt")

MODEL_JOBLIB_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
MODEL_PKL_PATH = os.path.join(ARTIFACT_DIR, "deploy_model.pkl")

PRED_THRESHOLD = 0.50

app = Flask(__name__)

MODEL = None
DEPLOY_FEATURES: List[str] = []
BOUNDS: Dict[str, Any] = {}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_artifacts() -> None:
    global MODEL, DEPLOY_FEATURES, BOUNDS

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing {FEATURES_PATH}")
    DEPLOY_FEATURES = _load_json(FEATURES_PATH)

    if os.path.exists(BOUNDS_PATH):
        BOUNDS = _load_json(BOUNDS_PATH)
    else:
        BOUNDS = {}

    if os.path.exists(MODEL_JOBLIB_PATH):
        MODEL = joblib.load(MODEL_JOBLIB_PATH)
        return

    if os.path.exists(MODEL_PKL_PATH):
        MODEL = joblib.load(MODEL_PKL_PATH)
        return

    raise FileNotFoundError(f"Missing model file. Expected {MODEL_JOBLIB_PATH} or {MODEL_PKL_PATH}")


def _coerce_instances(payload: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - {"instances":[{...},{...}]}
      - [{...},{...}]
      - {...}
    """
    if isinstance(payload, dict) and "instances" in payload:
        instances = payload["instances"]
    else:
        instances = payload

    if isinstance(instances, dict):
        return [instances]
    if isinstance(instances, list) and all(isinstance(x, dict) for x in instances):
        return instances

    raise ValueError("Payload must be a dict, list of dicts, or {'instances': [..]}.")


def _normalize_gender(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return s


def validate_instances(instances: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Implements the same “limiters” concept from your notebook:
    - required feature presence
    - numeric min/max bounds from input_bounds.json
    - allowed gender values from input_bounds.json
    """
    errors: List[str] = []
    out_rows: List[Dict[str, Any]] = []

    feat_bounds = (BOUNDS.get("features") or {}) if isinstance(BOUNDS, dict) else {}

    allowed_gender = None
    if "gender" in feat_bounds and isinstance(feat_bounds["gender"], dict):
        allowed_gender = feat_bounds["gender"].get("allowed")
        if isinstance(allowed_gender, list):
            allowed_gender = set(str(x) for x in allowed_gender)

    for i, row in enumerate(instances):
        # required fields
        missing = [f for f in DEPLOY_FEATURES if f not in row]
        if missing:
            errors.append(f"row[{i}] missing required fields: {missing}")
            continue

        clean: Dict[str, Any] = {}

        for f in DEPLOY_FEATURES:
            v = row.get(f)

            if f == "gender":
                g = _normalize_gender(v)
                if allowed_gender is not None and g not in allowed_gender:
                    errors.append(f"row[{i}].gender='{g}' not in allowed={sorted(list(allowed_gender))}")
                clean[f] = g
                continue

            try:
                num = float(v)
            except Exception:
                errors.append(f"row[{i}].{f}='{v}' is not numeric")
                continue

            if not np.isfinite(num):
                errors.append(f"row[{i}].{f} is not finite")
                continue

            b = feat_bounds.get(f)
            if isinstance(b, dict) and "min" in b and "max" in b:
                lo = float(b["min"])
                hi = float(b["max"])
                if num < lo or num > hi:
                    errors.append(f"row[{i}].{f}={num} outside [{lo}, {hi}]")
            clean[f] = num

        out_rows.append(clean)

    return out_rows, errors


@app.before_first_request
def _startup():
    load_artifacts()


@app.get("/")
def root():
    return jsonify({"status": "ok", "service": "final-project"})


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        instances = _coerce_instances(payload)

        clean_rows, errors = validate_instances(instances)
        if errors:
            return jsonify({"error": "Input validation failed", "details": errors}), 400

        df = pd.DataFrame(clean_rows, columns=DEPLOY_FEATURES)

        proba = MODEL.predict_proba(df)[:, 1]
        pred = (proba >= PRED_THRESHOLD).astype(int)

        return jsonify(
            {
                "probabilities": proba.tolist(),
                "predictions": pred.tolist(),
                "threshold": PRED_THRESHOLD,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))