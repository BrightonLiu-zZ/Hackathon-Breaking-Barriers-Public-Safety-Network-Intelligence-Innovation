# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hackathon context and judge expectations

The project targets a **public-safety / campus** story: detect **tail-risk** crowd behavior (simulated gunshot aftermath) from synthetic location streams.

### Judge rubric (`suggestions_from_judge.txt`)

1. **Real-time anomaly detection** for simulated tail-risk events, with a **0.5–1s micro-batching** style pipeline to process synthetic streams efficiently.
2. **Lagged spatio-temporal features** (in this repo implemented in **Python**: aggregation + sliding / lagged columns; aligns with “stateless backend receives fixed-shape feature rows”).
3. **XGBoost** classifier; **threshold tuning using PR-style metrics** to handle **imbalanced** data and emphasize **low False Negatives** where possible.
4. **AWS Lambda** for inference (latency is a design goal; sub-100ms is illustrative, not a hard SLA in this repo).
5. **S3**-hosted front end with **Amazon Location Service** for an interactive map (“dashboard” = static site + ALS basemap, not a separate AWS product).

### Clarifications the team uses

- **“LLM-generated”** in the judge text refers to **simulation / notebook code that was authored with LLM assistance**, not to a live LLM producing the data stream at runtime.
- The **public demo** can show a **precomputed** time window and alerts on the map; **full real-time** scoring in the browser is optional. **Inference code** must still exist and run **locally** and/or on **Lambda**.
- The current pipeline uses a **single XGBoost** path. Legacy **Model A + Model B fusion** is **not** the active design (see `legacy/`).

---

## Project overview

**End-to-end flow:** synthetic crowd GPS (rare positive `is_gunshot` windows) → **1s micro-batch** frame features with **lags** → **XGBoost** → PR/FN-oriented **evaluation artifacts** → **local batch inference** → **demo GeoJSON + alert JSON** → **MapLibre + ALS** map (`results/index.html`), optionally on **S3**.

---

## Development commands

```bash
# Install dependencies (Python 3.10+)
pip install -r requirements.txt

# Full pipeline (repo root, in order)
python scripts/generate_expanded_dataset.py   # simulation CSV + sim_metadata.json
python scripts/build_features_lagged.py       # features_lagged.parquet + meta
python scripts/train_xgboost.py               # model + threshold + reports
python scripts/infer_local.py                 # results/predictions.json
python scripts/build_demo_geojson.py          # map assets under results/out/

# Map demo locally (avoid file:// — use http://)
cd results && python -m http.server 8000
# Open http://localhost:8000/index.html

# Optional: Lambda container
docker build -f lambda_inference/Dockerfile -t gunshot-lambda .
```

---

## Pipeline (data flow)

```
scripts/generate_expanded_dataset.py
  → data/expanded_gunshot_sim.csv    # per-phone lat/lon + is_gunshot
  → data/sim_metadata.json           # event start time(s) for demo cropping

scripts/build_features_lagged.py  (gunshot_ml.features.build_feature_table)
  → data/features_lagged.parquet
  → data/features_lagged.csv
  → data/features_lagged_meta.json

scripts/train_xgboost.py  (gunshot_ml.train)
  → models/xgboost_gunshot.json
  → models/threshold.json
  → models/evaluation_report.md
  → models/pr_curve_val.png
  → models/metrics.json

scripts/infer_local.py
  → results/predictions.json

scripts/build_demo_geojson.py
  → results/out/gunshot_points_all.geojson
  → results/out/xgb_alerts.json
```

Rough scales (regenerate after changing `T_END` / seeds): expanded CSV on the order of **hundreds of thousands** of rows; micro-batch feature table on the order of **tens of thousands** of rows with **very low** positive rate.

---

## Important paths and roles

| Path | Role |
|------|------|
| `suggestions_from_judge.txt` | Source-of-truth bullet list of judge expectations |
| `README.md` | Human-facing overview, AWS roles, reproduction, legacy notes |
| `requirements.txt` | Python dependencies for scripts and training |
| `scripts/generate_expanded_dataset.py` | CLI for the same simulation as `notebooks/create_expanded_dataset.ipynb` |
| `scripts/build_features_lagged.py` | Builds lagged micro-batch features from expanded CSV |
| `scripts/train_xgboost.py` | Trains XGBoost, writes model + threshold + PR/FN reports |
| `scripts/infer_local.py` | Loads Booster + threshold; scores full feature table → `predictions.json` |
| `scripts/build_demo_geojson.py` | Crops demo time window; writes GeoJSON + `xgb_alerts.json` for the HUD |
| `gunshot_ml/features.py` | Feature engineering (meters, per-frame aggregates, 1s bins, lags) |
| `gunshot_ml/train.py` | Stratified splits, XGBoost fit, threshold selection, metrics |
| `models/xgboost_gunshot.json` | Serialized model for local + Lambda |
| `models/threshold.json` | Decision threshold + **feature_names** (must match inference columns) |
| `models/evaluation_report.md` | Markdown report for judges (PR-AUC, confusion, etc.) |
| `results/index.html` | Map UI: MapLibre + Cognito Identity Pool + ALS style; reads `./out/*` |
| `lambda_inference/lambda_function.py` | Lambda handler: JSON batch in → probabilities / preds out |
| `lambda_inference/Dockerfile` | Container image bundling handler + copied `models/*` |
| `legacy/get_anomaly_dataset.py` | **Deprecated** Model A + B fusion script (reference only) |
| `notebooks/create_expanded_dataset.ipynb` | Notebook twin of dataset generator |
| `notebooks/train_xgboost_pr.ipynb` | Thin notebook that shells to training scripts |

---

## Architecture notes

### Core package: `gunshot_ml/`

- **`features.py`**: `expanded_csv_to_per_phone_df` → `aggregate_native_frames` → `microbatch_aggregate` (~1s) → `add_lagged_features` → `build_feature_table`.
- **`train.py`**: Stratified train/val/test (so rare positives appear in each split); `scale_pos_weight` for imbalance; threshold rule favors **low FN** on validation when possible; exports metrics and PR curve plot via `scripts/train_xgboost.py`.

### Visualization

- **`results/index.html`**: Loads `out/gunshot_points_all.geojson` and `out/xgb_alerts.json`; shows an XGBoost alert banner when the scrubbed time is near a precomputed alert time.

### AWS (conceptual roles)

| Service | Role in this project |
|--------|----------------------|
| **S3** | Static hosting for `results/` (HTML + GeoJSON + JSON). Inference does **not** run on S3. |
| **Lambda** | Optional: same model JSON + threshold as local inference. |
| **Amazon Location Service** | Basemap / style for MapLibre (`style-descriptor` in `index.html`). |
| **Cognito Identity Pool** | Short-lived credentials for the browser to use ALS (configured in `index.html`). |

---

## Design decisions (for contributors)

- **Micro-batching (~1s)**: Aligns with judge wording; training and inference share the same feature schema.
- **Stratified splits**: Pure time splits often leave validation/test with **zero** positives when only one short event exists in a long simulation.
- **Native XGBoost JSON in Lambda**: Avoids fragile ONNX export for XGBoost 2.x; `Dockerfile` copies `models/xgboost_gunshot.json` and `models/threshold.json`.
- **Demo assets are precomputed**: The S3-facing demo can highlight a window where XGBoost “fires” without requiring live inference from the static site.
