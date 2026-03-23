# Gunshot pattern — XGBoost + ALS map (hackathon)

End-to-end pipeline: **synthetic crowd simulation** (rare gunshot window) → **lagged spatio-temporal frame features** with **1s micro-batching** → **XGBoost** classifier → **PR curve / FN-focused threshold report** → optional **AWS Lambda** inference (native XGBoost JSON) → **S3 static site** with MapLibre + **Amazon Location Service** basemap.

**Live demo (existing):** [S3 static site](http://gunshot-demo-site.s3-website-us-west-2.amazonaws.com/) — replace contents with `results/` after you build assets.

---

## What each AWS piece does here

| Service | Role in this project |
|--------|----------------------|
| **S3** | Hosts `index.html`, `results/out/*.geojson`, and JSON sidecars. No inference runs in S3—only static files. |
| **Lambda** | Optional HTTP inference: load `models/xgboost_gunshot.json` + `threshold.json`, score a batch of feature rows. See `lambda_inference/`. |
| **Amazon Location Service** | Map **basemap** for MapLibre (`style-descriptor`). Cognito **Identity Pool** in `results/index.html` supplies temporary credentials (already configured). |

---

## Quick reproduction (repo root)

```bash
pip install -r requirements.txt
python scripts/generate_expanded_dataset.py
python scripts/build_features_lagged.py
python scripts/train_xgboost.py
python scripts/infer_local.py
python scripts/build_demo_geojson.py
```

Outputs:

| Path | Description |
|------|-------------|
| `data/expanded_gunshot_sim.csv` | Per-phone positions; `is_gunshot` labels rare windows |
| `data/sim_metadata.json` | Gunshot event start time(s) for demos |
| `data/features_lagged.parquet` | Training/inference feature matrix + `y` |
| `models/evaluation_report.md` | PR-AUC, threshold, confusion matrix |
| `models/pr_curve_val.png` | Validation PR curve |
| `models/xgboost_gunshot.json` | Serialized model |
| `models/threshold.json` | Threshold + `feature_names` |
| `results/predictions.json` | Batch scores from `infer_local.py` |
| `results/out/gunshot_points_all.geojson` | Demo-window map data |
| `results/out/xgb_alerts.json` | Times where the model fires (for HUD) |

### View the map locally

```bash
cd results
python -m http.server 8000
```

Open `http://localhost:8000/index.html` (file:// breaks CORS/fetch).

### S3 static deploy

Upload the **`results/`** folder (including `out/`) to your bucket with **static website hosting** enabled. Keep the same paths so `./out/gunshot_points_all.geojson` resolves. Reuse the existing **Identity Pool** + **Amazon Location Service** map IDs in `index.html` unless you create new resources.

---

## Model & evaluation

- **Classifier:** single `XGBClassifier` on lagged frame features (no legacy Model A + Model B fusion).
- **Labels:** `y` derived from simulation `is_gunshot` at the **micro-batch** level (max within each 1s bin).
- **Split:** **stratified** train/val/test so each split contains positives (pure time splits often empty for a single rare event).
- **Threshold:** validation rule prefers **zero false negatives** when possible (`min_positive_score_minus_epsilon`), then falls back to scanning thresholds.
- **Metrics:** see `models/evaluation_report.md` and `models/metrics.json`.

---

## Lambda (container image)

From repo root (Docker required):

```bash
docker build -f lambda_inference/Dockerfile -t gunshot-lambda .
```

Push to **Amazon ECR**, create a **Lambda** from the image. The image sets `CMD ["lambda_function.lambda_handler"]`. Request body (JSON):

```json
{
  "features": [[0.0, 0.1, ...]],
  "feature_names": ["outward_fraction", ...]
}
```

`feature_names` can be omitted if it matches `threshold.json`. ONNX export is optional and often unnecessary; the image uses **XGBoost + JSON** for compatibility.

---

## Legacy (Model A + B)

The old fusion script lives in `legacy/get_anomaly_dataset.py` and is **not** part of the current pipeline.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/create_expanded_dataset.ipynb` | Same simulation as `scripts/generate_expanded_dataset.py` |
| `notebooks/train_xgboost_pr.ipynb` | Calls training scripts |
| `notebooks/create_model_A_B_features.ipynb` | **Legacy** feature recipe (per-phone + Model B columns); superseded by `gunshot_ml/features.py` |

---

## License & attribution

Simulated data and derived assets are for research/demo. Map rendering: MapLibre + Amazon Location Service. XGBoost for the classifier.
