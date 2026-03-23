"""Train XGBoost on features_lagged, save model + PR/FN report + optional ONNX."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gunshot_ml.train import save_training_report, train_xgboost_pr


def try_export_onnx(model, feature_names: list[str], out_path: Path) -> bool:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("skl2onnx not installed; skip ONNX. pip install skl2onnx onnxruntime")
        return False
    n = len(feature_names)
    initial_types = [("float_input", FloatTensorType([None, n]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        options={id(model): {"zipmap": False}},
        target_opset=12,
    )
    out_path.write_bytes(onnx_model.SerializeToString())
    print("Wrote ONNX", out_path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "features_lagged.parquet")
    ap.add_argument("--meta", type=Path, default=ROOT / "data" / "features_lagged_meta.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    feature_names = meta["feature_names"]
    df = pd.read_parquet(args.features)

    result = train_xgboost_pr(df, feature_names)
    model = result["model"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "xgboost_gunshot.json"
    model.save_model(str(json_path))
    print("Wrote", json_path)

    thr_path = args.out_dir / "threshold.json"
    thr_path.write_text(
        json.dumps(
            {
                "threshold": result["threshold"],
                "threshold_report_val": result["threshold_report_val"],
                "feature_names": feature_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = args.out_dir / "evaluation_report.md"
    save_training_report(result, report_path)
    print("Wrote", report_path)

    # PR curve (validation)
    fig, ax = plt.subplots(figsize=(6, 4))
    from sklearn.metrics import precision_recall_curve

    prec, rec, _ = precision_recall_curve(result["y_val"], result["proba_val"])
    ax.plot(rec, prec, label=f"PR-AUC={result['pr_auc_val']:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall (validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    pr_png = args.out_dir / "pr_curve_val.png"
    fig.tight_layout()
    fig.savefig(pr_png, dpi=120)
    plt.close(fig)
    print("Wrote", pr_png)

    try:
        try_export_onnx(model, feature_names, args.out_dir / "gunshot_xgb.onnx")
    except Exception as e:
        print("ONNX export skipped:", e)

    metrics = {
        "pr_auc_val": result["pr_auc_val"],
        "pr_auc_test": result["pr_auc_test"],
        "pr_curve_auc_val": result["pr_curve_auc_val"],
        "test_confusion": result["test_confusion"],
        "threshold": result["threshold"],
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
