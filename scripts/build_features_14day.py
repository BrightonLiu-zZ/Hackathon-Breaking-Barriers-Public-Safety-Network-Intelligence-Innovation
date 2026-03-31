"""
Build lagged spatio-temporal features from expanded_gunshot_sim_14day.csv.

Differences from build_features_lagged.py:
  - Propagates the `mask` column through micro-batching (max aggregation per bin)
  - Outputs features_lagged_14day.parquet / .csv / _meta.json

Run from repo root: python scripts/build_features_14day.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gunshot_ml.features import (
    BASE_FEATURE_COLS,
    add_lagged_features,
    aggregate_native_frames,
    expanded_csv_to_per_phone_df,
    microbatch_aggregate,
)

MICRO_BATCH_S = 1.0
LAG_STEPS = 3


def build_feature_table_14day(
    csv_path: Path,
    micro_batch_s: float = MICRO_BATCH_S,
    lag_steps: int = LAG_STEPS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full pipeline for 14-day CSV:
      1. Compute mask per micro-batch bin (max = if any raw row in bin is masked)
      2. Run standard feature pipeline on data without mask column
      3. Merge mask back onto feature table by t_bin
    """
    raw = pd.read_csv(csv_path)

    # ── Step 1: mask per 1-second bin ────────────────────────────────────────
    raw["t_bin"] = np.floor(raw["t"] / micro_batch_s) * micro_batch_s
    mask_per_bin = (
        raw.groupby("t_bin")["mask"]
        .max()
        .reset_index()
        .rename(columns={"t_bin": "t", "mask": "mask"})
    )

    # ── Step 2: feature pipeline (strip mask so it doesn't interfere) ────────
    raw_clean = raw.drop(columns=["mask", "t_bin"])
    per_phone = expanded_csv_to_per_phone_df(raw_clean)
    native = aggregate_native_frames(per_phone)
    batched = microbatch_aggregate(native, micro_batch_s=micro_batch_s)
    full = add_lagged_features(batched, lag_steps=lag_steps)
    full = full.dropna().reset_index(drop=True)

    # ── Step 3: merge mask (t column in `full` == t_bin after microbatch) ────
    full = full.merge(mask_per_bin, on="t", how="left")
    full["mask"] = full["mask"].fillna(0).astype(int)

    # ── Feature name list (same schema as original) ───────────────────────────
    feature_names: list[str] = []
    for c in BASE_FEATURE_COLS:
        if c in full.columns:
            feature_names.append(c)
        for L in range(1, lag_steps + 1):
            lc = f"{c}_lag{L}"
            if lc in full.columns:
                feature_names.append(lc)

    return full, feature_names


def main() -> None:
    csv_path = ROOT / "data" / "expanded_gunshot_sim_14day.csv"
    print(f"Reading {csv_path} ...")

    df, feature_names = build_feature_table_14day(csv_path)

    n_pos = int(df["y"].sum())
    n_mask = int(df["mask"].sum())
    pos_rate = float(n_pos / len(df)) if len(df) > 0 else 0.0

    print(f"Rows: {len(df):,}  positives (y=1): {n_pos}  masked rows: {n_mask}")
    print(f"Positive rate (unmasked positive bars only): {pos_rate:.6f}")

    out_parquet = ROOT / "data" / "features_lagged_14day.parquet"
    out_csv = ROOT / "data" / "features_lagged_14day.csv"
    out_meta = ROOT / "data" / "features_lagged_14day_meta.json"

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    meta = {
        "feature_names": feature_names,
        "micro_batch_s": MICRO_BATCH_S,
        "lag_steps": LAG_STEPS,
        "n_rows": len(df),
        "n_positive": n_pos,
        "n_masked": n_mask,
        "positive_rate": pos_rate,
        "n_days": 14,
        "day_duration_s": 43200.0,
        "temporal_split": {
            "train_days": "1-10",
            "val_days": "11-12",
            "test_days": "13-14",
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {out_parquet}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_meta}")


if __name__ == "__main__":
    main()
