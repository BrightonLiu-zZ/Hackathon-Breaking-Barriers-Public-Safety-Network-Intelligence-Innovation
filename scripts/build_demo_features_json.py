"""
Extract feature rows for the demo time window from features_lagged_56day.parquet.

The browser loads this JSON so it can send the correct feature vector to the
local inference server each time the slider moves to a new frame.

Output: results/out/features_demo_window.json
  {
    "feature_names": [...],
    "rows": {
      "25580.0": [0.12, 1.3, ...],
      "25581.0": [...],
      ...
    }
  }

Run from repo root: python scripts/build_demo_features_json.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    meta_path = ROOT / "data" / "sim_metadata.json"
    feat_path = ROOT / "data" / "features_lagged_56day.parquet"
    thr_path  = ROOT / "models" / "threshold_56day.json"
    out_path  = ROOT / "results" / "out" / "features_demo_window.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    thr_meta = json.loads(thr_path.read_text(encoding="utf-8"))
    feature_names: list[str] = thr_meta["feature_names"]

    # Demo window bounds (same logic as build_demo_geojson.py)
    starts = meta["gunshot_event_starts_s"]
    dur    = float(meta["event_duration_s"])
    pad    = 35.0
    t0     = starts[0] - pad
    t1     = starts[0] + dur + pad

    print(f"Loading {feat_path} ...")
    df = pd.read_parquet(feat_path)
    window = df[(df["t"] >= t0) & (df["t"] <= t1)].copy()
    print(f"  Total feature rows: {len(df):,}  Demo window rows: {len(window)}")

    if window.empty:
        print("WARNING: no feature rows found in demo window — check t range vs parquet contents")

    rows: dict[str, list[float]] = {}
    for _, row in window.iterrows():
        key = str(round(float(row["t"]), 1))
        rows[key] = [round(float(row[f]), 6) for f in feature_names]

    out = {"feature_names": feature_names, "rows": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"Wrote {out_path}  ({len(rows)} time steps)")


if __name__ == "__main__":
    main()
