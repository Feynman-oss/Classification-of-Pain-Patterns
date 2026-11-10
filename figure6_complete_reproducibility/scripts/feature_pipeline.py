"""Feature generation used by the two released Figure 6 model branches.

Branch A creates 27 DFF features and nine RC features from one prepared 10-s
segment. Branch B starts from the per-window table, fits the 32-level quantizer
on the training events only, adds the RC lookup features, and aggregates each
event with five statistics to obtain the 195-feature model input.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

DFF27 = [
    "points", "mean", "std", "min", "max", "range", "median", "auc", "slope",
    "rate_mean", "rate_abs_mean", "rate_std", "rate_min", "rate_max", "rate_range",
    "rate_abs_max", "rate_pos_ratio", "rate_neg_ratio", "peak_count", "mean_abs",
    "slope_abs", "cv", "range_per_std", "rate_energy", "dynamic_index", "asym_rate",
    "peak_density",
]
RC9 = [
    "rc_binary_shape_hrs_read", "rc_binary_shape_hrs_peak", "rc_binary_shape_lrs_read",
    "rc_binary_shape_lrs_peak", "rc_binary_shape_hrs_delta", "rc_binary_shape_lrs_delta",
    "rc_binary_shape_read_gap", "rc_binary_shape_peak_gap", "rc_binary_shape_energy",
]
AGGREGATION_STATS = ["mean", "std", "min", "max", "median"]
WINDOW_SECONDS = 10.0
RESAMPLE_POINTS = 300


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    return 0.0 if np.std(a) <= 1e-12 or np.std(b) <= 1e-12 else float(np.corrcoef(a, b)[0, 1])


def resample(time_s: np.ndarray, signal: np.ndarray) -> np.ndarray:
    order = np.argsort(time_s)
    t, x = np.asarray(time_s)[order], np.asarray(signal)[order]
    t, unique = np.unique(t, return_index=True)
    x = x[unique]
    grid = np.linspace(t[0], t[-1], RESAMPLE_POINTS)
    return np.interp(grid, t, x)


def dff27(time_s: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    x = resample(time_s, signal)
    dt = WINDOW_SECONDS / RESAMPLE_POINTS
    diff = np.diff(x)
    rate = diff / dt
    std = float(np.std(x, ddof=1))
    q10, q25, q75, q90 = np.quantile(x, [0.10, 0.25, 0.75, 0.90])
    peak_threshold = float(np.mean(x) + std)
    peaks = int(((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > peak_threshold)).sum())
    base = {
        "points": len(x), "mean": np.mean(x), "std": std, "min": np.min(x), "max": np.max(x),
        "range": np.ptp(x), "median": np.median(x), "auc": np.trapezoid(x, dx=dt),
        "slope": (x[-1] - x[0]) / WINDOW_SECONDS, "rate_mean": np.mean(rate),
        "rate_abs_mean": np.mean(np.abs(rate)), "rate_std": np.std(rate, ddof=1),
        "rate_min": np.min(rate), "rate_max": np.max(rate), "rate_range": np.ptp(rate),
        "rate_abs_max": np.max(np.abs(rate)), "rate_pos_ratio": np.mean(rate > 0),
        "rate_neg_ratio": np.mean(rate < 0), "peak_count": peaks,
    }
    base.update({
        "mean_abs": abs(base["mean"]), "slope_abs": abs(base["slope"]),
        "cv": std / (abs(base["mean"]) + 1e-12), "range_per_std": base["range"] / (std + 1e-12),
        "rate_energy": np.mean(rate**2), "dynamic_index": base["range"] * base["rate_abs_mean"],
        "asym_rate": base["rate_pos_ratio"] - base["rate_neg_ratio"],
        "peak_density": peaks / len(x),
    })
    return {k: float(base[k]) for k in DFF27}


def rc9_from_code(code: str, response: pd.DataFrame) -> dict[str, float]:
    row = response.set_index("code").loc[str(code).zfill(5)]
    hr, hp, lr, lp = (float(row[c]) for c in ["hrs_read", "hrs_peak", "lrs_read", "lrs_peak"])
    values = [hr, hp, lr, lp, hp - hr, lp - lr, hr - lr, hp - lp, abs(hr) + abs(hp) + abs(lr) + abs(lp)]
    return dict(zip(RC9, values))


def add_fold_rc_and_aggregate(
    windows: pd.DataFrame, response: pd.DataFrame, train_events: set[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = windows[windows.event_id.isin(train_events)]
    z_min = float(train.rc_scalar_mean_plus_change.min())
    z_max = float(train.rc_scalar_mean_plus_change.max())
    if z_max <= z_min:
        z_max = z_min + 1e-12
    out = windows.copy()
    out["rc_quantized_level"] = np.clip(
        np.rint(31 * (out.rc_scalar_mean_plus_change - z_min) / (z_max - z_min)), 0, 31
    ).astype(int)
    out["rc_code"] = out.rc_quantized_level.map(lambda v: format(int(v), "05b"))
    lookup = response.assign(code=response.code.astype(str).str.zfill(5)).set_index("code")
    for c in ["hrs_read", "hrs_peak", "lrs_read", "lrs_peak"]:
        out[f"rc_{c}"] = out.rc_code.map(lookup[c]).astype(float)
    out["rc_hrs_delta"] = out.rc_hrs_peak - out.rc_hrs_read
    out["rc_lrs_delta"] = out.rc_lrs_peak - out.rc_lrs_read
    out["rc_read_gap"] = out.rc_hrs_read - out.rc_lrs_read
    out["rc_peak_gap"] = out.rc_hrs_peak - out.rc_lrs_peak
    out["rc_energy"] = out[["rc_hrs_read", "rc_hrs_peak", "rc_lrs_read", "rc_lrs_peak"]].abs().sum(axis=1)
    metadata = ["event_id", "target"]
    features = [c for c in out if c.startswith(("dff_", "rate_", "peak_", "rc_")) and c != "rc_code"]
    agg = out.groupby(metadata)[features].agg(AGGREGATION_STATS)
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    return agg.reset_index().fillna(0), {"z_min": z_min, "z_max": z_max}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", help="CSV with time_s,dff for the 27+9 branch")
    ap.add_argument("--code", default="00000")
    ap.add_argument("--windows", help="Prepared per-window CSV for the 195-feature branch")
    ap.add_argument("--response-table", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    response = pd.read_csv(args.response_table, dtype={"code": str})
    if args.segment:
        segment = pd.read_csv(args.segment)
        row = {**dff27(segment.time_s.to_numpy(), segment.dff.to_numpy()), **rc9_from_code(args.code, response)}
        pd.DataFrame([row]).to_csv(args.output, index=False)
    elif args.windows:
        windows = pd.read_csv(args.windows)
        table, _ = add_fold_rc_and_aggregate(windows, response, set(windows.event_id))
        table.to_csv(args.output, index=False)
    else:
        ap.error("provide --segment or --windows")


if __name__ == "__main__":
    main()
