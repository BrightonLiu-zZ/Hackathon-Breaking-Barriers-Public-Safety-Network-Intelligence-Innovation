"""Build results/out/gunshot_points_all.geojson (demo time window) + out/xgb_alerts.json for static HUD."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    meta = json.loads((ROOT / "data" / "sim_metadata.json").read_text(encoding="utf-8"))
    starts = meta["gunshot_event_starts_s"]
    dur = float(meta["event_duration_s"])
    pad = 35.0
    t0 = starts[0] - pad
    t1 = starts[0] + dur + pad

    df = pd.read_csv(ROOT / "data" / "expanded_gunshot_sim.csv")
    df = df[(df["t"] >= t0) & (df["t"] <= t1)].copy()

    features = []
    for _, row in df.iterrows():
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
                "properties": {"phone_id": str(row["phone_id"]), "t": float(row["t"])},
            }
        )
    fc = {"type": "FeatureCollection", "features": features}

    out_dir = ROOT / "results" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    geo_path = out_dir / "gunshot_points_all.geojson"
    geo_path.write_text(json.dumps(fc, separators=(",", ":")), encoding="utf-8")

    pred_path = ROOT / "results" / "predictions.json"
    if pred_path.exists():
        pr = json.loads(pred_path.read_text(encoding="utf-8"))
        alert_times = [float(t) for t, p in zip(pr["times"], pr["pred"]) if p == 1 and t0 <= float(t) <= t1]
        alert_times = sorted(set(alert_times))
    else:
        alert_times = []

    alert_meta = {
        "demo_start_s": t0,
        "demo_end_s": t1,
        "gunshot_event_start_s": starts[0],
        "xgb_alert_times_s": alert_times,
    }
    (out_dir / "xgb_alerts.json").write_text(json.dumps(alert_meta, indent=2), encoding="utf-8")
    print("Wrote", geo_path, "features=", len(features))
    print("Wrote", out_dir / "xgb_alerts.json", "n_alerts=", len(alert_times))


if __name__ == "__main__":
    main()
