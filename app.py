import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.join(APP_DIR, "artifacts"))

FEATURES_PATH = os.path.join(ARTIFACT_DIR, "deploy_features.json")
BOUNDS_PATH = os.path.join(ARTIFACT_DIR, "input_bounds.json")
THRESH_PATH = os.path.join(ARTIFACT_DIR, "high_addiction_threshold.txt")

MODEL_JOBLIB_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
MODEL_PKL_PATH = os.path.join(ARTIFACT_DIR, "deploy_model.pkl")

PRED_THRESHOLD = float(os.environ.get("PRED_THRESHOLD", "0.5"))

app = Flask(__name__)

MODEL: Any = None
DEPLOY_FEATURES: List[str] = []
BOUNDS: Dict[str, Any] = {}

_ARTIFACTS_READY = False
_ARTIFACTS_ERROR: Optional[str] = None


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_artifacts() -> None:
    global MODEL, DEPLOY_FEATURES, BOUNDS, _ARTIFACTS_READY, _ARTIFACTS_ERROR

    if _ARTIFACTS_READY:
        return

    try:
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f"Missing {FEATURES_PATH}")

        DEPLOY_FEATURES = _load_json(FEATURES_PATH)
        if not isinstance(DEPLOY_FEATURES, list) or not all(isinstance(x, str) for x in DEPLOY_FEATURES):
            raise ValueError("deploy_features.json must be a JSON list of strings")

        if os.path.exists(BOUNDS_PATH):
            BOUNDS = _load_json(BOUNDS_PATH)
        else:
            BOUNDS = {}

        if os.path.exists(MODEL_JOBLIB_PATH):
            MODEL = joblib.load(MODEL_JOBLIB_PATH)
            logger.info("Loaded model: %s", MODEL_JOBLIB_PATH)
        elif os.path.exists(MODEL_PKL_PATH):
            MODEL = joblib.load(MODEL_PKL_PATH)
            logger.info("Loaded model: %s", MODEL_PKL_PATH)
        else:
            raise FileNotFoundError(f"Missing model file. Expected {MODEL_JOBLIB_PATH} or {MODEL_PKL_PATH}")

        if os.path.exists(THRESH_PATH):
            logger.info("high_addiction_threshold.txt: %s", _load_text(THRESH_PATH))

        _ARTIFACTS_READY = True
        _ARTIFACTS_ERROR = None

    except Exception as e:
        _ARTIFACTS_ERROR = str(e)
        logger.exception("Artifact load failed: %s", _ARTIFACTS_ERROR)


@app.before_request
def _ensure_artifacts_loaded():
    if not _ARTIFACTS_READY and _ARTIFACTS_ERROR is None:
        load_artifacts()


def _coerce_instances(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and "instances" in payload:
        instances = payload["instances"]
    else:
        instances = payload

    if isinstance(instances, dict):
        return [instances]

    if isinstance(instances, list) and all(isinstance(x, dict) for x in instances):
        return instances

    raise ValueError("Payload must be a dict, list of dicts, or {'instances': [..]}.")


def _get_feature_bounds() -> Dict[str, Any]:
    if isinstance(BOUNDS, dict):
        fb = BOUNDS.get("features", {})
        if isinstance(fb, dict):
            return fb
    return {}


def _allowed_gender() -> Optional[set]:
    fb = _get_feature_bounds()
    g = fb.get("gender")
    if isinstance(g, dict) and isinstance(g.get("allowed"), list):
        return set(str(x).strip() for x in g["allowed"])
    return None


def validate_instances(instances: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    cleaned: List[Dict[str, Any]] = []
    fb = _get_feature_bounds()
    allowed_gender = _allowed_gender()

    for i, row in enumerate(instances):
        row_errors: List[str] = []
        missing = [f for f in DEPLOY_FEATURES if f not in row]
        if missing:
            row_errors.append(f"missing required fields: {missing}")

        out: Dict[str, Any] = {}

        for f in DEPLOY_FEATURES:
            v = row.get(f, None)

            if f == "gender":
                g = "" if v is None else str(v).strip()
                if allowed_gender is not None and g not in allowed_gender:
                    row_errors.append(f"gender='{g}' not in allowed={sorted(list(allowed_gender))}")
                out[f] = g
                continue

            try:
                num = float(v)
            except Exception:
                row_errors.append(f"{f}='{v}' is not numeric")
                continue

            if not np.isfinite(num):
                row_errors.append(f"{f} is not finite")
                continue

            b = fb.get(f)
            if isinstance(b, dict) and "min" in b and "max" in b:
                lo = float(b["min"])
                hi = float(b["max"])
                if num < lo or num > hi:
                    row_errors.append(f"{f}={num} outside [{lo}, {hi}]")

            out[f] = num

        if row_errors:
            errors.append(f"row[{i}] " + "; ".join(row_errors))
        else:
            cleaned.append(out)

    return cleaned, errors


def _predict(df: pd.DataFrame) -> Dict[str, Any]:
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(df)[:, 1]
        pred = (proba >= PRED_THRESHOLD).astype(int)
        return {"probabilities": proba.tolist(), "predictions": pred.tolist(), "threshold": PRED_THRESHOLD}

    if hasattr(MODEL, "decision_function"):
        scores = MODEL.decision_function(df)
        proba = 1 / (1 + np.exp(-scores))
        pred = (proba >= PRED_THRESHOLD).astype(int)
        return {"probabilities": proba.tolist(), "predictions": pred.tolist(), "threshold": PRED_THRESHOLD}

    preds = MODEL.predict(df)
    return {"predictions": [int(x) for x in preds], "threshold": PRED_THRESHOLD}


@app.get("/")
def root():
    if _ARTIFACTS_ERROR:
        return jsonify({"status": "error", "detail": _ARTIFACTS_ERROR}), 500

    allowed = ["Female", "Male", "Other"]
    try:
        gset = _allowed_gender()
        if gset:
            allowed = sorted(list(gset))
    except Exception:
        pass

    try:
        return render_template("index.html", allowed_genders=allowed)
    except Exception:
        return jsonify({"service": "ana680-final", "status": "ok"}), 200


@app.get("/health")
def health():
    if _ARTIFACTS_ERROR:
        return jsonify({"status": "error", "detail": _ARTIFACTS_ERROR}), 500
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    if _ARTIFACTS_ERROR:
        return jsonify({"error": "Artifacts not loaded", "detail": _ARTIFACTS_ERROR}), 500

    try:
        payload = request.get_json(force=True)
        instances = _coerce_instances(payload)

        clean_rows, errors = validate_instances(instances)
        if errors:
            return jsonify({"error": "Input validation failed", "details": errors}), 400

        df = pd.DataFrame(clean_rows, columns=DEPLOY_FEATURES)
        out = _predict(df)
        return jsonify(out), 200

    except Exception as e:
        logger.exception("Prediction error: %s", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)