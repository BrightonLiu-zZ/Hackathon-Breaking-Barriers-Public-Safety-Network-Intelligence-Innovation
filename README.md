# Gunshot-Pattern

A small pipeline that simulates a crowd, extracts features, detects “evacuation-like” motion (**Model B**) and “surprising movement vs prediction” (**Model A**), fuses both triggers, and writes the rows used by the final animated viz.

## TL; DR
Our demo (using ALS for the map background, served on AWS site) for viz of a gunshot case on happening on a campus detected by our models: http://gunshot-demo-site.s3-website-us-west-2.amazonaws.com/

## Pipeline Overview

**Goal:** Raise a final alert only when both are true:

- **Trigger A (Model A):** Many people move very differently from a short-term prediction (residual anomaly).
- **Trigger B (Model B):** A large share of people move outward from a center fast enough (evacuation pattern).

**Viz Input:** Timestamps judged anomalous (A **AND** B) and the corresponding rows from `data/expanded_gunshot_sim.csv`, saved as `data/gunshot_anomaly.csv`. These are converted into GeoJSON frames that animate in `results/index.html`.

---

## Key Concepts

### Model A (per person → frame)
- Predictors per device at time *t*:  
  `x_t_m, y_t_m, vx_t_mps, vy_t_mps`.
- Predict next position via constant-velocity; compute **residual** (meters).
- A device is “surprised” if residual > its **own** 99th percentile (from normal times).
- If a big **fraction** of active devices are surprised for ≥ a few seconds → **Trigger A = ON**.

### Model B (per frame)
- Predictors per timestamp:  
  `outward_fraction`, `mean_outward_speed_mps`.
- Tiny XGBoost outputs `P(evac)`.  
  If `P(evac) ≥ cutoff` for ≥ **M** consecutive ticks → **Trigger B = ON**.

### Fusion
- **Final anomaly** when **A AND B** are both true at (or within ±1 tick of) the same time.

---

## End-to-End Steps

> If you just want the viz quickly, see **Quick Start**.

### 1) Generate / Re-generate the simulated day (optional)
- `notebooks/create_expanded_dataset.ipynb` → `data/expanded_gunshot_sim.csv`  
  *(Already provided for convenience.)*

### 2) Create model predictors (features)
- `notebooks/create_model_A_B_features.ipynb` →  
  - `data/modelA_predictors.csv` with:  
    `phone_id, t, x_t_m, y_t_m, vx_t_mps, vy_t_mps`
  - `data/modelB_predictors.csv` with:  
    `t, outward_fraction, mean_outward_speed_mps`  
  *(Both already included.)*

### 3) Train / Inspect Models (optional but recommended)
- `notebooks/modelA.ipynb`: explore residuals, choose thresholds, view Trigger A over time.
- `notebooks/modelB.ipynb`: train tiny local XGBoost, choose probability cutoff, apply persistence.  
  Final cell writes `modelB_classifier_outputs_sagemaker.csv` (local file with columns `t, proba, pred_persist, label`).  
  > Name is kept for compatibility; it does **not** require SageMaker.

### 4) Fuse triggers and write anomaly rows
Run:
```bash
python src/get_anomaly_dataset.py
```

This will:

1. Read `data/modelA_predictors.csv`, compute **Trigger A**.
2. Read `data/modelB_predictors.csv` and either:

   * use `modelB_classifier_outputs_sagemaker.csv` if present, **or**
   * apply a simple two-threshold rule (fallback).
3. Fuse **A AND B** → anomaly timestamps.
4. Filter `data/expanded_gunshot_sim.csv` to those times and save **`data/gunshot_anomaly.csv`**.

### 5) Build viz assets & view animation

* `notebooks/create_necessary_files_for_ALS.ipynb` → produces:

  * `results/gunshot_points_all.geojson`
  * `results/frames/` + `results/frames_index.json`
  * `results/gunshot_clean.csv`
* Serve locally to avoid CORS:

```bash
cd results
python -m http.server 8000
```

Open: `http://localhost:8000/index.html`

---

## Quick Start

1. (Optional) Install basics:

   ```bash
   pip install pandas numpy xgboost scikit-learn
   ```
2. Fuse triggers & write anomaly rows:

   ```bash
   python src/get_anomaly_dataset.py
   ```
3. Build viz assets:

   * Run `notebooks/create_necessary_files_for_ALS.ipynb`
4. Serve the viz:

   ```bash
   cd results
   python -m http.server 8000
   ```

   Open `http://localhost:8000/index.html`.

---

## Data & Feature Schemas

### `data/expanded_gunshot_sim.csv`

* `phone_id` — unique device ID
* `t` — seconds since start (2.5s cadence)
* `lat`, `lon` — WGS84 coordinates
* `is_gunshot` — 1 inside the scripted window, else 0

### `data/modelA_predictors.csv` (per device, per time)

* `phone_id`, `t`
* `x_t_m`, `y_t_m` — meters in a local planar frame (centered on the area)
* `vx_t_mps`, `vy_t_mps` — per-axis velocities (m/s)

### `data/modelB_predictors.csv` (per frame)

* `t`
* `outward_fraction` — share of people moving outward from the center (and above a speed threshold)
* `mean_outward_speed_mps` — average speed (m/s) among outward movers

### `data/gunshot_anomaly.csv` (output for viz)

* Subset of rows from `expanded_gunshot_sim.csv` at anomaly timestamps
* Includes Trigger A/B flags for traceability

---

## Tuning Dials (safe defaults)

Edit at the top of `src/get_anomaly_dataset.py`:

**Model A**

* `RESIDUAL_PERCENTILE = 0.99` (per-device residual threshold)
* `MIN_FRACTION_SURPRISED = 0.35` (frame-level surprised fraction)
* `PERSIST_TICKS_A = 2` (consecutive ticks ≈ 5 s)

**Model B** (rule fallback when classifier output isn’t present)

* `B_FRAC = 0.65` (min outward fraction)
* `B_SPEED = 2.3` (m/s, min speed for outward movers)
* `PERSIST_TICKS_B = 2`

**Fusion**

* A simple **AND** at the same tick (allow ±1 tick if needed).

> Fewer false alerts → raise thresholds or persistence.
> Earlier detection → lower thresholds or persistence.

---

## Common Pitfalls

* **No anomalies appear**
  Lower `MIN_FRACTION_SURPRISED` (A) or Model B cutoff / `B_FRAC` / `B_SPEED`. Confirm feature notebooks ran.

* **Too many anomalies**
  Raise thresholds or increase `PERSIST_TICKS_*` from 2 → 3.

* **Viz won’t load frames**
  Always open `results/index.html` via `python -m http.server` (not as a file URL).

---

## License & Attribution

* Simulated data and derived assets for research/demo purposes.
* Map rendering via MapLibre (`results/index.html`).
* XGBoost is used for the tiny frame-level classifier (Model B).

---

## Maintainer Notes

* Keep **feature recipe constants** (e.g., speed threshold defining an “outward mover”) and **persistence windows** together for quick tuning across venues (size, density, device cadence).