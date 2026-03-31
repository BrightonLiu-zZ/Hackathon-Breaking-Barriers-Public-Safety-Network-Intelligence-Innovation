"""Stratified train/val/test split, XGBoost, PR metrics, FN-aware threshold."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb


def stratified_train_val_test(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split so rare positives appear in train/val/test (needed when positives << 1%).

    Pure time-based splits often yield empty val/test positives for a single short event;
    use this for PR/FN evaluation. Document temporal validation separately if needed.
    """
    y = df["y"].to_numpy()
    if y.sum() < 3 or (1 - y).sum() < 3:
        raise ValueError("Need at least 3 positives and 3 negatives for stratified split.")
    idx = np.arange(len(df))
    idx_temp, idx_test, y_temp, y_test = train_test_split(
        idx, y, test_size=test_fraction, stratify=y, random_state=random_state
    )
    val_rel = val_fraction / (1.0 - test_fraction)
    idx_train, idx_val, _, _ = train_test_split(
        idx_temp, y_temp, test_size=val_rel, stratify=y_temp, random_state=random_state
    )
    train = df.iloc[idx_train].copy()
    val = df.iloc[idx_val].copy()
    test = df.iloc[idx_test].copy()
    return train, val, test


def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return 1.0
    return float(neg / pos)


def pick_threshold_min_fn(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    max_fp: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Prefer **zero false negatives** on validation when any positive exists: set threshold
    just below the lowest score on positive validation rows, then minimize FP among ties.

    If that is impossible (e.g. positives scored below negatives), fall back to scanning
    thresholds minimizing (FN, FP).
    """
    pos_mask = y_true == 1
    if pos_mask.any():
        min_pos = float(proba[pos_mask].min())
        thr_high_recall = max(0.0, min_pos - 1e-5)
        pred = (proba >= thr_high_recall).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        if fn == 0:
            rep = {
                "threshold": float(thr_high_recall),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "rule": "min_positive_score_minus_epsilon",
            }
            if max_fp is None or fp <= max_fp:
                return thr_high_recall, rep

    thresholds = np.unique(np.concatenate([[0.0], np.sort(proba), [1.0]]))
    best: dict[str, Any] | None = None
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        if max_fp is not None and fp > max_fp:
            continue
        key = (fn, fp, -thr)
        if best is None or key < best["_sort"]:
            best = {
                "_sort": key,
                "threshold": float(thr),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "rule": "scan_minimize_fn_then_fp",
            }
    assert best is not None
    del best["_sort"]
    return best["threshold"], best


def train_xgboost_pr(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> dict[str, Any]:
    train_df, val_df, test_df = stratified_train_val_test(
        df,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    X_tr = train_df[feature_names].to_numpy(dtype=np.float32)
    y_tr = train_df["y"].to_numpy()
    X_val = val_df[feature_names].to_numpy(dtype=np.float32)
    y_val = val_df["y"].to_numpy()
    X_te = test_df[feature_names].to_numpy(dtype=np.float32)
    y_te = test_df["y"].to_numpy()

    spw = compute_scale_pos_weight(y_tr)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        random_state=random_state,
        eval_metric="aucpr",
        tree_method="hist",
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    proba_val = model.predict_proba(X_val)[:, 1]
    proba_te = model.predict_proba(X_te)[:, 1]

    pr_auc_val = average_precision_score(y_val, proba_val)
    pr_auc_te = average_precision_score(y_te, proba_te)

    prec, rec, _ = precision_recall_curve(y_val, proba_val)
    pr_curve_auc = auc(rec, prec)

    thr, thr_report = pick_threshold_min_fn(y_val, proba_val)

    pred_te = (proba_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred_te, labels=[0, 1]).ravel()

    return {
        "model": model,
        "feature_names": feature_names,
        "threshold": thr,
        "threshold_report_val": thr_report,
        "pr_auc_val": float(pr_auc_val),
        "pr_auc_test": float(pr_auc_te),
        "pr_curve_auc_val": float(pr_curve_auc),
        "scale_pos_weight": spw,
        "test_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "y_test": y_te,
        "proba_test": proba_te,
        "y_val": y_val,
        "proba_val": proba_val,
    }


# ── 14-day model: temporal split + F2 threshold + regularised XGBoost ────────

def temporal_split_by_day(
    df: pd.DataFrame,
    *,
    n_train_days: int = 10,
    n_val_days: int = 2,
    n_test_days: int = 2,
    day_s: float = 43200.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split by absolute t column (seconds).
    train = days 1–n_train, val = next n_val, test = remaining n_test.
    """
    train_end = n_train_days * day_s
    val_end = (n_train_days + n_val_days) * day_s
    train = df[df["t"] < train_end].copy()
    val = df[(df["t"] >= train_end) & (df["t"] < val_end)].copy()
    test = df[df["t"] >= val_end].copy()
    return train, val, test


def pick_threshold_f2(
    y_true: np.ndarray,
    proba: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """
    Select threshold maximising F2-score (β=2, recall-weighted) on the given split.
    Falls back to 0.5 if no positives or degenerate curve.
    """
    if y_true.sum() == 0:
        return 0.5, {"threshold": 0.5, "tn": int((y_true == 0).sum()),
                     "fp": 0, "fn": 0, "tp": 0, "rule": "no_positives_fallback"}

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve appends a sentinel (prec=1, rec=0) with no threshold
    p, r = prec_arr[:-1], rec_arr[:-1]
    beta2 = 4.0   # β=2 → β²=4
    denom = beta2 * p + r
    f2 = np.where(denom > 0, (1 + beta2) * p * r / denom, 0.0)

    if f2.max() == 0:
        thr = 0.5
    else:
        thr = float(thresholds[int(np.argmax(f2))])

    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return thr, {
        "threshold": float(thr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "rule": "f2_score_maximisation",
    }


def train_xgboost_14day(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    n_train_days: int = 10,
    n_val_days: int = 2,
    n_test_days: int = 2,
    day_s: float = 43200.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train XGBoost on 14-day data.

    Key differences from train_xgboost_pr():
    - Temporal split (days 1-10 / 11-12 / 13-14) instead of stratified random split
    - Masked rows (mask=1) excluded from all splits before fitting
    - Regularised hyperparameters (depth=3, reg_lambda=2, min_child_weight=5)
    - Early stopping on val aucpr (up to 500 trees)
    - Threshold chosen by F2-score (β=2) on validation, not by min-FN guarantee
    """
    train_df, val_df, test_df = temporal_split_by_day(
        df,
        n_train_days=n_train_days,
        n_val_days=n_val_days,
        n_test_days=n_test_days,
        day_s=day_s,
    )

    # Drop masked rows (bars 2-10 of each event)
    for split in ("mask",):
        if split in train_df.columns:
            train_df = train_df[train_df["mask"] == 0].copy()
            val_df = val_df[val_df["mask"] == 0].copy()
            test_df = test_df[test_df["mask"] == 0].copy()
            break

    X_tr = train_df[feature_names].to_numpy(dtype=np.float32)
    y_tr = train_df["y"].to_numpy()
    X_val = val_df[feature_names].to_numpy(dtype=np.float32)
    y_val = val_df["y"].to_numpy()
    X_te = test_df[feature_names].to_numpy(dtype=np.float32)
    y_te = test_df["y"].to_numpy()

    spw = compute_scale_pos_weight(y_tr)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.1,
        scale_pos_weight=spw,
        random_state=random_state,
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=20,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    proba_val = model.predict_proba(X_val)[:, 1]
    proba_te = model.predict_proba(X_te)[:, 1]

    pr_auc_val = average_precision_score(y_val, proba_val) if y_val.sum() > 0 else float("nan")
    pr_auc_te = average_precision_score(y_te, proba_te) if y_te.sum() > 0 else float("nan")

    if y_val.sum() > 0:
        prec_v, rec_v, _ = precision_recall_curve(y_val, proba_val)
        pr_curve_auc_val = float(auc(rec_v, prec_v))
    else:
        pr_curve_auc_val = float("nan")

    thr, thr_report = pick_threshold_f2(y_val, proba_val)

    pred_te = (proba_te >= thr).astype(int)
    if y_te.sum() > 0:
        tn, fp, fn, tp = confusion_matrix(y_te, pred_te, labels=[0, 1]).ravel()
    else:
        tn = int((y_te == 0).sum()); fp = fn = tp = 0

    return {
        "model": model,
        "feature_names": feature_names,
        "threshold": thr,
        "threshold_report_val": thr_report,
        "pr_auc_val": float(pr_auc_val),
        "pr_auc_test": float(pr_auc_te),
        "pr_curve_auc_val": float(pr_curve_auc_val),
        "scale_pos_weight": spw,
        "test_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "y_test": y_te,
        "proba_test": proba_te,
        "y_val": y_val,
        "proba_val": proba_val,
        "best_iteration": getattr(model, "best_iteration", model.n_estimators),
        "split_info": {
            "train_positives": int(y_tr.sum()),
            "val_positives": int(y_val.sum()),
            "test_positives": int(y_te.sum()),
            "train_rows": len(y_tr),
            "val_rows": len(y_val),
            "test_rows": len(y_te),
        },
    }


def save_training_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# XGBoost gunshot — PR / FN report",
        "",
        f"- PR-AUC (validation, average_precision_score): **{result['pr_auc_val']:.6f}**",
        f"- PR-AUC (test): **{result['pr_auc_test']:.6f}**",
        f"- PR curve integral (validation, trapz on precision-recall curve): **{result['pr_curve_auc_val']:.6f}**",
        f"- `scale_pos_weight` (neg/pos on train): **{result['scale_pos_weight']:.4f}**",
        "",
        "## Threshold (chosen on validation to minimize FN, then FP)",
        "",
        f"- **threshold** = `{result['threshold']:.6f}`",
        "",
        "### Validation at chosen threshold",
        "",
        json.dumps(result["threshold_report_val"], indent=2),
        "",
        "### Test set confusion matrix (tn, fp, fn, tp)",
        "",
        json.dumps(result["test_confusion"], indent=2),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
