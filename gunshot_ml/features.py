"""
Build lagged spatio-temporal frame features from expanded_gunshot_sim.csv,
then aggregate into micro-batch windows (default 1.0s) for XGBoost.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

CENTER_LON = -121.7493613
CENTER_LAT = 38.5411082
DT_DEFAULT_S = 2.5
SPEED_THRESH_MPS = 2.0
NEAR_CENTER_R_M = 5.0

BASE_FEATURE_COLS = [
    "outward_fraction",
    "mean_outward_speed_mps",
    "crowd_count",
    "net_radial_flow_mps",
    "near_center_fraction_5m",
]


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111_132.954 - 559.822 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_per_deg_lon = (math.pi / 180) * 6_378_137.0 * math.cos(lat)
    return m_per_deg_lat, m_per_deg_lon


M_PER_DEG_LAT, M_PER_DEG_LON = meters_per_degree(CENTER_LAT)


def expanded_csv_to_per_phone_df(df: pd.DataFrame) -> pd.DataFrame:
    """Lat/lon -> local meters; velocities; outward flags (same recipe as legacy Model B notebook)."""
    df = df.sort_values(["phone_id", "t"]).reset_index(drop=True)
    df["x_t_m"] = (df["lon"] - CENTER_LON) * M_PER_DEG_LON
    df["y_t_m"] = (df["lat"] - CENTER_LAT) * M_PER_DEG_LAT

    df["dt_s"] = df.groupby("phone_id")["t"].diff()
    df["dt_s"] = df["dt_s"].where(df["dt_s"] > 0, DT_DEFAULT_S)

    df["dx_m"] = df.groupby("phone_id")["x_t_m"].diff().fillna(0.0)
    df["dy_m"] = df.groupby("phone_id")["y_t_m"].diff().fillna(0.0)
    df["vx_t_mps"] = df["dx_m"] / df["dt_s"]
    df["vy_t_mps"] = df["dy_m"] / df["dt_s"]

    df["speed_mps"] = np.sqrt(df["vx_t_mps"] ** 2 + df["vy_t_mps"] ** 2)
    r = np.sqrt(df["x_t_m"] ** 2 + df["y_t_m"] ** 2)
    r_safe = r.replace(0, np.nan)
    df["radial_mps"] = (df["vx_t_mps"] * df["x_t_m"] + df["vy_t_mps"] * df["y_t_m"]) / r_safe
    df["radial_mps"] = df["radial_mps"].fillna(0.0)
    df["outward_flag"] = (df["radial_mps"] > 0) & (df["speed_mps"] >= SPEED_THRESH_MPS)
    df["near_center"] = r <= NEAR_CENTER_R_M
    return df


def aggregate_native_frames(per_phone: pd.DataFrame) -> pd.DataFrame:
    """One row per native timestep t (DT_DEFAULT_S cadence)."""
    grp = per_phone.groupby("t", as_index=False)
    frame_counts = grp["phone_id"].count().rename(columns={"phone_id": "crowd_count"})
    outward = grp["outward_flag"].sum().rename(columns={"outward_flag": "outward_n"})
    near_c = grp["near_center"].mean().rename(columns={"near_center": "near_center_fraction_5m"})
    agg = frame_counts.merge(outward, on="t").merge(near_c, on="t")
    agg["outward_fraction"] = agg["outward_n"] / agg["crowd_count"].replace(0, np.nan)
    agg["outward_fraction"] = agg["outward_fraction"].fillna(0.0)

    mean_out = (
        per_phone[per_phone["outward_flag"]]
        .groupby("t")["speed_mps"]
        .mean()
        .reindex(agg["t"])
        .fillna(0.0)
        .reset_index(drop=True)
    )
    agg["mean_outward_speed_mps"] = mean_out

    net_radial = grp["radial_mps"].mean().rename(columns={"radial_mps": "net_radial_flow_mps"})
    agg = agg.merge(net_radial, on="t")

    label_t = per_phone.groupby("t")["is_gunshot"].max().reset_index(name="y")
    agg = agg.merge(label_t, on="t", how="left")
    agg["y"] = agg["y"].fillna(0).astype(int)
    agg = agg.drop(columns=["outward_n"], errors="ignore")
    return agg.sort_values("t").reset_index(drop=True)


def microbatch_aggregate(native: pd.DataFrame, micro_batch_s: float = 1.0) -> pd.DataFrame:
    """Bucket native frames into fixed-width time bins (simulates 0.5–1s micro-batching)."""
    native = native.copy()
    native["t_bin"] = np.floor(native["t"] / micro_batch_s) * micro_batch_s
    mean_cols = [c for c in native.columns if c not in ("t", "t_bin", "y")]
    agg_dict = {c: "mean" for c in mean_cols}
    agg_dict["y"] = "max"
    batched = native.groupby("t_bin", as_index=False).agg(agg_dict)
    batched = batched.rename(columns={"t_bin": "t"})
    return batched.sort_values("t").reset_index(drop=True)


def add_lagged_features(df: pd.DataFrame, lag_steps: int = 3) -> pd.DataFrame:
    """Lags on base spatio-temporal aggregates (stateless-friendly: same schema at inference)."""
    df = df.sort_values("t").reset_index(drop=True)
    for col in BASE_FEATURE_COLS:
        if col not in df.columns:
            continue
        for L in range(1, lag_steps + 1):
            df[f"{col}_lag{L}"] = df[col].shift(L)
    return df


def build_feature_table(
    expanded_path: str | Path,
    *,
    micro_batch_s: float = 1.0,
    lag_steps: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """Full pipeline: expanded CSV -> native frames -> micro-batches -> lags. Returns (df, feature_names)."""
    expanded_path = Path(expanded_path)
    raw = pd.read_csv(expanded_path)
    per_phone = expanded_csv_to_per_phone_df(raw)
    native = aggregate_native_frames(per_phone)
    batched = microbatch_aggregate(native, micro_batch_s=micro_batch_s)
    full = add_lagged_features(batched, lag_steps=lag_steps)
    full = full.dropna().reset_index(drop=True)

    feature_names: list[str] = []
    for c in BASE_FEATURE_COLS:
        if c in full.columns:
            feature_names.append(c)
        for L in range(1, lag_steps + 1):
            lc = f"{c}_lag{L}"
            if lc in full.columns:
                feature_names.append(lc)
    return full, feature_names
