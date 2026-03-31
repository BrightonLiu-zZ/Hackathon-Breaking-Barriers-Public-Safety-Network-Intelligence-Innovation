"""
Pre-compute per-timestep XGBoost inference for the Day 1 demo window.

Reads  results/out/features_demo_window.json  (built by build_demo_features_json.py)
Writes results/out/xgb_scores_demo_window.json:

  {
    "threshold": 0.7415,
    "rows": {
      "25580.0": {"proba": 0.0123, "pred": 0, "inference_ms": 1.2},
      "25620.0": {"proba": 0.8734, "pred": 1, "inference_ms": 1.1},
      ...
    }
  }

The browser loads this so the precomputed fallback path can show realistic
per-row latency numbers even when the local inference server is offline.

Run from repo root: python scripts/build_demo_scores_json.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    feat_path  = ROOT / "results" / "out" / "features_demo_window.json"
    model_path = ROOT / "models" / "xgboost_56day.json"
    thr_path   = ROOT / "models" / "threshold_56day.json"
    out_path   = ROOT / "results" / "out" / "xgb_scores_demo_window.json"

    # Load model + threshold
    print(f"Loading model: {model_path}")
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    thr_meta = json.loads(thr_path.read_text(encoding="utf-8"))
    threshold: float = float(thr_meta["threshold"])
    feature_names: list[str] = thr_meta["feature_names"]
    print(f"  threshold={threshold:.6f}  features={len(feature_names)}")

    # Load feature window
    print(f"Loading features: {feat_path}")
    raw = json.loads(feat_path.read_text(encoding="utf-8"))
    rows_in: dict[str, list[float]] = raw["rows"]
    print(f"  {len(rows_in)} timesteps in demo window")

    # Score each row individually (mimics single-row lambda calls)
    rows_out: dict[str, dict] = {}
    for t_str, row in rows_in.items():
        X = np.asarray([row], dtype=np.float32)
        t0 = time.perf_counter()
        dmat = xgb.DMatrix(X, feature_names=feature_names)
        proba = float(booster.predict(dmat)[0])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rows_out[t_str] = {
            "proba":        round(proba, 6),
            "pred":         int(proba >= threshold),
            "inference_ms": round(elapsed_ms, 3),
        }

    n_alerts = sum(1 for v in rows_out.values() if v["pred"] == 1)
    print(f"  {n_alerts} alert timesteps out of {len(rows_out)}")

    out = {"threshold": threshold, "rows": rows_out}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
