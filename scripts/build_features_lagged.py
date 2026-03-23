"""Build data/features_lagged.parquet from expanded_gunshot_sim.csv."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gunshot_ml.features import build_feature_table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--expanded",
        type=Path,
        default=ROOT / "data" / "expanded_gunshot_sim.csv",
        help="Path to expanded_gunshot_sim.csv",
    )
    p.add_argument("--micro-batch-s", type=float, default=1.0)
    p.add_argument("--lag-steps", type=int, default=3)
    args = p.parse_args()

    df, feature_names = build_feature_table(
        args.expanded,
        micro_batch_s=args.micro_batch_s,
        lag_steps=args.lag_steps,
    )
    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "features_lagged.parquet"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(out_dir / "features_lagged.csv", index=False)
    meta = {
        "feature_names": feature_names,
        "micro_batch_s": args.micro_batch_s,
        "lag_steps": args.lag_steps,
        "n_rows": len(df),
        "positive_rate": float(df["y"].mean()),
    }
    (out_dir / "features_lagged_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", parquet_path, "rows=", len(df), "features=", len(feature_names))
    print("positive_rate", meta["positive_rate"])


if __name__ == "__main__":
    main()
