import os
import numpy as np
import pandas as pd

A_PATH = r"C:\Hackathon\gunshot\modelA_predictors.csv"
B_PATH = r"C:\Hackathon\gunshot\modelB_predictors.csv"
EXPANDED_PATH = r"C:\Hackathon\gunshot\expanded_gunshot_sim.csv"
OUT_PATH = r"C:\Hackathon\gunshot\gunshot_anomaly.csv"

DT_DEFAULT_S = 2.5
RESIDUAL_PERCENTILE = 0.99     # per-device residual threshold
MIN_FRACTION_SURPRISED = 0.35  # fraction of devices above threshold at time t
PERSIST_TICKS_A = 2            # require 2 consecutive ticks (~5s)
# Trigger B (rule only; ignored if SageMaker output is present)
B_FRAC = 0.65
B_SPEED = 2.3                  # m/s
PERSIST_TICKS_B = 2            # ~5s

def apply_persistence(df_time_bool, persist_ticks):
    df_time_bool = df_time_bool.sort_values("t").copy()
    roll = df_time_bool["flag_raw"].rolling(window=persist_ticks, min_periods=persist_ticks).sum()
    df_time_bool["flag_persist"] = (roll >= persist_ticks).astype(int)
    return df_time_bool[["t","flag_persist"]]

# Trigger A: residual anomaly from Model A predictors
A = pd.read_csv(A_PATH).sort_values(["phone_id","t"]).reset_index(drop=True)

# Align each row to its next step per phone
A["x_next"] = A.groupby("phone_id")["x_t_m"].shift(-1)
A["y_next"] = A.groupby("phone_id")["y_t_m"].shift(-1)
A["dt_s"]   = A.groupby("phone_id")["t"].diff().shift(-1)
A["dt_s"]   = A["dt_s"].where(A["dt_s"] > 0, DT_DEFAULT_S)
A = A.dropna(subset=["x_next","y_next"]).reset_index(drop=True)

# Constant-velocity prediction and residual (meters)
A["x_pred_next"] = A["x_t_m"] + A["vx_t_mps"] * A["dt_s"]
A["y_pred_next"] = A["y_t_m"] + A["vy_t_mps"] * A["dt_s"]
A["residual_m"]  = np.hypot(A["x_next"] - A["x_pred_next"], A["y_next"] - A["y_pred_next"])

# Per-device residual threshold (99th percentile), then per-row flag
thr = A.groupby("phone_id")["residual_m"].quantile(RESIDUAL_PERCENTILE).rename("res_thr")
A = A.merge(thr, on="phone_id", how="left")
A["flag"] = (A["residual_m"] > A["res_thr"]).astype(int)

# Aggregate to frame and apply persistence
per_t_A = (A.groupby("t")
             .agg(active=("phone_id","count"), surprised=("flag","sum"))
             .reset_index())
per_t_A["frac_surprised"] = per_t_A["surprised"] / per_t_A["active"]
per_t_A["flag_raw"] = (per_t_A["frac_surprised"] >= MIN_FRACTION_SURPRISED).astype(int)
per_t_A = apply_persistence(per_t_A[["t","flag_raw"]], PERSIST_TICKS_A).rename(columns={"flag_persist":"triggerA"})

# Trigger B: classifier output if present, else simple rule ----------
if os.path.exists(B_SAGEMAKER_OUTPUT):
    # Use SageMaker outputs (expects columns: t, pred_persist)
    B_full = pd.read_csv(B_SAGEMAKER_OUTPUT)
    if "pred_persist" not in B_full.columns:
        raise ValueError(f"{B_SAGEMAKER_OUTPUT} found but missing 'pred_persist' column.")
    per_t_B = B_full[["t","pred_persist"]].copy().rename(columns={"pred_persist":"triggerB"})
else:
    # Rule with 2 features
    B = pd.read_csv(B_PATH).sort_values("t").reset_index(drop=True)
    B["flag_raw"] = ((B["outward_fraction"] >= B_FRAC) &
                     (B["mean_outward_speed_mps"] >= B_SPEED)).astype(int)
    per_t_B = apply_persistence(B[["t","flag_raw"]], PERSIST_TICKS_B).rename(columns={"flag_persist":"triggerB"})

fused = per_t_A.merge(per_t_B, on="t", how="inner")
fused["fused"] = ((fused["triggerA"] == 1) & (fused["triggerB"] == 1)).astype(int)
anomaly_times = fused.loc[fused["fused"] == 1, "t"].round(1).unique()

print(f"Found {len(anomaly_times)} anomaly timestamps after fusion.")

EXP = pd.read_csv(EXPANDED_PATH)
# Ensure same rounding/format
if "t" in EXP.columns:
    t_col = "t"
elif "timestamp" in EXP.columns:
    t_col = "timestamp"
else:
    raise ValueError("expanded_gunshot_sim.csv must contain a time column named 't' or 'timestamp'.")

EXP["_t_round"] = EXP[t_col].round(1)
out = EXP[EXP["_t_round"].isin(anomaly_times)].drop(columns=["_t_round"])

# (Optional) attach which trigger columns for traceability
out = out.merge(fused[["t","triggerA","triggerB"]].rename(columns={"t": t_col}),
                on=t_col, how="left")

out.to_csv(OUT_PATH, index=False)
print(f"Saved {len(out)} rows to {OUT_PATH}")
