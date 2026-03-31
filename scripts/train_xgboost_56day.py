"""
Train XGBoost on 56-day features with temporal split.

Uses train_xgboost_14day() from gunshot_ml.train with 56-day split parameters:
  train = days 1-39, val = days 40-47, test = days 48-56

Outputs (all in models/, do NOT overwrite existing model files):
  models/xgboost_56day.json
  models/threshold_56day.json
  models/metrics_56day.json
  models/evaluation_report_56day.md
  models/pr_curve_56day.png

Run from repo root: python scripts/train_xgboost_56day.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gunshot_ml.train import save_training_report, train_xgboost_14day


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=ROOT / "data" / "features_lagged_56day.parquet")
    ap.add_argument("--meta", type=Path,
                    default=ROOT / "data" / "features_lagged_56day_meta.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    feature_names = meta["feature_names"]
    df = pd.read_parquet(args.features)

    n_pos = int(df["y"].sum())
    print(f"Loaded {len(df):,} rows  positives={n_pos}")

    # Reuse 14-day training function with 56-day split parameters
    result = train_xgboost_14day(
        df, feature_names,
        n_train_days=39,
        n_val_days=8,
        n_test_days=9,
        day_s=43200.0,
    )
    model = result["model"]

    si = result["split_info"]
    print(f"Split — train: {si['train_rows']} rows / {si['train_positives']} pos  "
          f"val: {si['val_rows']} rows / {si['val_positives']} pos  "
          f"test: {si['test_rows']} rows / {si['test_positives']} pos")
    print(f"Best iteration: {result['best_iteration']}")
    print(f"PR-AUC  val={result['pr_auc_val']:.4f}  test={result['pr_auc_test']:.4f}")
    print(f"Threshold (F2): {result['threshold']:.6f}")
    print(f"Val  @ threshold: {result['threshold_report_val']}")
    print(f"Test confusion:   {result['test_confusion']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    json_path = args.out_dir / "xgboost_56day.json"
    model.save_model(str(json_path))
    print(f"Wrote {json_path}")

    # ── Threshold ─────────────────────────────────────────────────────────────
    thr_path = args.out_dir / "threshold_56day.json"
    thr_path.write_text(
        json.dumps({
            "threshold": result["threshold"],
            "threshold_report_val": result["threshold_report_val"],
            "feature_names": feature_names,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {thr_path}")

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = args.out_dir / "evaluation_report_56day.md"
    save_training_report(result, report_path)
    print(f"Wrote {report_path}")

    # ── PR curve ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    if result["y_val"].sum() > 0:
        prec, rec, _ = precision_recall_curve(result["y_val"], result["proba_val"])
        ax.plot(rec, prec, label=f"PR-AUC={result['pr_auc_val']:.4f}")
        ax.axvline(
            result["threshold_report_val"].get("tp", 0) / max(result["y_val"].sum(), 1),
            ls="--", color="gray", alpha=0.5, label="F2 threshold recall",
        )
    else:
        ax.text(0.5, 0.5, "No positives in validation split",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall (validation) — 56-day model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    pr_png = args.out_dir / "pr_curve_56day.png"
    fig.tight_layout()
    fig.savefig(pr_png, dpi=120)
    plt.close(fig)
    print(f"Wrote {pr_png}")

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    metrics = {
        "pr_auc_val": result["pr_auc_val"],
        "pr_auc_test": result["pr_auc_test"],
        "pr_curve_auc_val": result["pr_curve_auc_val"],
        "test_confusion": result["test_confusion"],
        "threshold": result["threshold"],
        "best_iteration": result["best_iteration"],
        "split_info": result["split_info"],
    }
    metrics_path = args.out_dir / "metrics_56day.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
