# Public Safety Network Intelligence — Gunshot Anomaly Detection

End-to-end pipeline: **synthetic crowd simulation** (rare gunshot windows, 56-day corpus) → **lagged spatio-temporal frame features** with **1 s micro-batching** → **XGBoost** classifier with **PR / FN-focused threshold tuning** → **local inference server** (Lambda-compatible) → **S3 static site** with MapLibre + **Amazon Location Service** basemap.

---

## Live demo

**S3 static site (precomputed):** [gunshot-demo-site.s3-website-us-west-2.amazonaws.com](http://gunshot-demo-site.s3-website-us-west-2.amazonaws.com/)

- Scrub the time slider to replay the Day 1 gunshot event
- Red alert banner appears when the precomputed XGBoost score fires (`prob ≥ 0.74`)
- Latency shown as `~1.x ms (precomputed)` — matches real single-row inference time

**Real-time inference demo (local server):**

<!-- TODO: upload real_time_inference.mp4 and replace this comment with:
![Real-time inference demo](real_time_inference.mp4)
or link to it: [Watch demo video](real_time_inference.mp4)
-->

> Video coming soon — run locally with `python server/app.py` to see live inference.

---

## What each AWS piece does

| Service | Role |
|--------|------|
| **S3** | Hosts `index.html` + `out/*.geojson` / JSON sidecars. No inference runs in S3. |
| **Lambda** | Optional HTTP inference: `lambda_inference/lambda_function.py` loads `models/xgboost_56day.json` + `threshold_56day.json`, scores a batch of feature rows. **Local server** (`server/app.py`) mirrors this interface exactly for demos. |
| **Amazon Location Service** | MapLibre basemap (`style-descriptor`). Cognito Identity Pool in `index.html` supplies temporary credentials. |

---

## Quick reproduction

```bash
pip install -r requirements.txt

# 1. Simulate 56-day crowd corpus (train days 1-39 / val 40-47 / test 48-56)
python scripts/generate_56day_dataset.py

# 2. Build lagged 1s micro-batch features
python scripts/build_features_56day.py

# 3. Train XGBoost — temporal split, F2-score threshold
python scripts/train_xgboost_56day.py

# 4. Batch inference → predictions.json
python scripts/infer_local.py

# 5. Build demo map assets (Day 1 event window)
python scripts/build_demo_geojson.py
python scripts/build_demo_features_json.py
python scripts/build_demo_scores_json.py
```

Key outputs:

| Path | Description |
|------|-------------|
| `data/expanded_gunshot_sim_56day.csv` | Per-phone positions + `is_gunshot` labels (generated, git-ignored) |
| `data/features_lagged_56day.parquet` | Feature matrix with lags (generated, git-ignored) |
| `models/xgboost_56day.json` | Serialized XGBoost model |
| `models/threshold_56day.json` | Decision threshold + `feature_names` |
| `models/evaluation_report_56day.md` | PR-AUC, confusion matrix, threshold report |
| `models/pr_curve_56day.png` | Validation PR curve |
| `results/predictions.json` | Batch scores |
| `results/out/gunshot_points_all.geojson` | Demo-window map data |
| `results/out/xgb_alerts.json` | Alert times for HUD |
| `results/out/features_demo_window.json` | Per-timestep feature rows (for live browser inference) |
| `results/out/xgb_scores_demo_window.json` | Per-timestep precomputed scores + latency |

---

## View the map locally

```bash
# Terminal 1 — static file server
cd results && python -m http.server 8000
# Open http://localhost:8000/index.html

# Terminal 2 (optional) — live inference server (Lambda-compatible)
python server/app.py
# Server auto-detected by browser; status dot turns green
```

The browser pings `http://localhost:8001/health` on load. If the server is up, the slider calls it for real-time XGBoost inference; otherwise it falls back to precomputed alerts automatically.

---

## Model & evaluation (56-day corpus)

- **Corpus:** 56 simulated days; each day has one 10-bar (~10 s) gunshot event at a randomized time, with mild confounding noise (mini-dispersal false alarms on ~40% of days, randomized crowd pop 85–100).
- **Split:** temporal — train days 1–39, val 40–47, test 48–56. No data leakage.
- **Features:** 5 base features × 4 lags = 20 columns (`outward_fraction`, `mean_outward_speed_mps`, `crowd_count`, `net_radial_flow_mps`, `near_center_fraction_5m`).
- **Threshold rule:** F2-score maximisation on validation (emphasises recall / low FN).
- **Validation results:** `tn=138144, fp=16, fn=2, tp=78` — PR-AUC ≈ 0.83.
- Full report: `models/evaluation_report_56day.md`; curve: `models/pr_curve_56day.png`.

---

## Local inference server

`server/app.py` is a zero-dependency (stdlib only) HTTP server that mirrors the AWS Lambda interface:

```
POST http://localhost:8001/2015-03-31/functions/function/invocations
GET  http://localhost:8001/health
```

Request / response JSON is identical to `lambda_inference/lambda_function.py`. Run it before opening the map for real-time scoring.

```bash
python server/app.py                          # default port 8001, 56-day model
python server/app.py --port 8002 --model models/xgboost_56day.json
```

Terminal output per inference call:
```
[⚠  ALERT]  n=  1  max_prob=0.8734  threshold=0.7415  8.3 ms
```

---

## S3 static deploy

Upload these files to your bucket (static website hosting enabled):

```
index.html
out/gunshot_points_all.geojson
out/xgb_alerts.json
out/features_demo_window.json
out/xgb_scores_demo_window.json
```

Keep the same relative paths so `./out/…` resolves from `index.html`. Reuse the existing Identity Pool + ALS map IDs in `index.html`.

---

## Lambda (container image)

```bash
docker build -f lambda_inference/Dockerfile -t gunshot-lambda .
```

Push to **Amazon ECR**, create a Lambda from the image. Request body:

```json
{
  "features": [[0.0, 0.1, ...]],
  "feature_names": ["outward_fraction", "outward_fraction_lag1", "..."]
}
```

`feature_names` can be omitted if it matches `threshold_56day.json`.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/create_expanded_dataset.ipynb` | Interactive twin of `scripts/generate_56day_dataset.py` |
| `notebooks/train_xgboost_pr.ipynb` | Calls training scripts, shows PR curves |

---

## Legacy

`legacy/get_anomaly_dataset.py` — original Model A + B fusion script. **Not part of the current pipeline.**

---

## License & attribution

Simulated data and derived assets are for research / demo purposes. Map rendering: MapLibre + Amazon Location Service. Classifier: XGBoost.
