"""
AWS Lambda entry: JSON body {"features": [[...], ...], "feature_names": optional}.

Container image CMD: lambda_function.lambda_handler
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

_MODEL: xgb.Booster | None = None
_FEATURE_NAMES: list[str] | None = None
_THRESHOLD: float = 0.5


def _load_artifacts() -> None:
    global _MODEL, _FEATURE_NAMES, _THRESHOLD
    if _MODEL is not None:
        return
    base = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))
    model_path = base / "xgboost_gunshot.json"
    thr_path = base / "threshold.json"
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    meta = json.loads(thr_path.read_text(encoding="utf-8"))
    _MODEL = booster
    _FEATURE_NAMES = list(meta["feature_names"])
    _THRESHOLD = float(meta["threshold"])


def lambda_handler(event, context):
    _load_artifacts()
    assert _MODEL is not None and _FEATURE_NAMES is not None

    if isinstance(event.get("body"), str):
        body = json.loads(event["body"])
    else:
        body = event

    t0 = time.perf_counter()
    feature_names = body.get("feature_names") or _FEATURE_NAMES
    rows = body["features"]
    X = np.asarray(rows, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "error": "features must be 2D [batch, n_features]",
                    "expected_n_features": len(feature_names),
                    "got_shape": list(X.shape),
                }
            ),
        }

    dmat = xgb.DMatrix(X, feature_names=feature_names)
    proba = _MODEL.predict(dmat)
    pred = (proba >= _THRESHOLD).astype(int)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out = {
        "proba": [float(p) for p in proba],
        "pred": [int(p) for p in pred],
        "threshold": _THRESHOLD,
        "inference_ms": round(elapsed_ms, 4),
        "n_rows": int(X.shape[0]),
    }
    return {"statusCode": 200, "body": json.dumps(out)}
