"""Batch inference: load XGBoost JSON + threshold, score features_lagged (or stdin JSON)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_model(model_path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def predict_batch(
    booster: xgb.Booster,
    feature_names: list[str],
    X: np.ndarray,
) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=feature_names)
    return booster.predict(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "xgboost_gunshot.json")
    ap.add_argument("--threshold", type=Path, default=ROOT / "models" / "threshold.json")
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "features_lagged.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "predictions.json")
    args = ap.parse_args()

    thr_meta = json.loads(args.threshold.read_text(encoding="utf-8"))
    feature_names = thr_meta["feature_names"]
    thr = float(thr_meta["threshold"])

    df = pd.read_parquet(args.features)
    X = df[feature_names].to_numpy(dtype=np.float32)

    t0 = time.perf_counter()
    booster = load_model(args.model)
    proba = predict_batch(booster, feature_names, X)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pred = (proba >= thr).astype(int)
    out = {
        "threshold": thr,
        "n_rows": len(df),
        "inference_ms_for_batch": round(elapsed_ms, 3),
        "rows_per_ms": round(len(df) / max(elapsed_ms, 1e-9), 1),
        "times": df["t"].astype(float).round(2).tolist(),
        "proba": [float(p) for p in proba],
        "pred": [int(p) for p in pred],
        "y_true": df["y"].astype(int).tolist() if "y" in df.columns else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", args.out, f"({elapsed_ms:.1f} ms for {len(df)} rows)")


if __name__ == "__main__":
    main()
