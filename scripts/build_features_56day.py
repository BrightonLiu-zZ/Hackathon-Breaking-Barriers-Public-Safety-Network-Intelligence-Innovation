"""
Build lagged spatio-temporal features from expanded_gunshot_sim_56day.csv.

No mask column in the 56-day dataset, so the standard build_feature_table()
pipeline can be called directly.

Outputs:
  data/features_lagged_56day.parquet
  data/features_lagged_56day.csv
  data/features_lagged_56day_meta.json

Run from repo root: python scripts/build_features_56day.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gunshot_ml.features import build_feature_table

MICRO_BATCH_S = 1.0
LAG_STEPS = 3


def main() -> None:
    csv_path = ROOT / "data" / "expanded_gunshot_sim_56day.csv"
    print(f"Reading {csv_path} ...")

    df, feature_names = build_feature_table(csv_path, micro_batch_s=MICRO_BATCH_S, lag_steps=LAG_STEPS)

    n_pos = int(df["y"].sum())
    pos_rate = float(n_pos / len(df)) if len(df) > 0 else 0.0
    print(f"Rows: {len(df):,}  positives (y=1): {n_pos}")
    print(f"Positive rate: {pos_rate:.6f}")

    out_parquet = ROOT / "data" / "features_lagged_56day.parquet"
    out_csv     = ROOT / "data" / "features_lagged_56day.csv"
    out_meta    = ROOT / "data" / "features_lagged_56day_meta.json"

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    meta = {
        "feature_names": feature_names,
        "micro_batch_s": MICRO_BATCH_S,
        "lag_steps": LAG_STEPS,
        "n_rows": len(df),
        "n_positive": n_pos,
        "positive_rate": pos_rate,
        "n_days": 56,
        "day_duration_s": 43200.0,
        "temporal_split": {
            "train_days": "1-39",
            "val_days": "40-47",
            "test_days": "48-56",
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {out_parquet}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_meta}")


if __name__ == "__main__":
    main()
