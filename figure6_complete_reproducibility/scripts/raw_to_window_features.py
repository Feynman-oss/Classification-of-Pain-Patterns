"""Convert released raw 10-s fluorescence events into window-level features.

Accepts a directory, CSV file, or the ``103_raw_10s_segments_only.zip`` Release
asset. It preserves event labels, resamples each window to 300 equally spaced
points, and writes the window-level quantities used by the Figure 6 workflow.
Quantizer limits are fitted later by the grouped training scripts.
"""
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

N_POINTS = 300
WINDOW_SECONDS = 10.0


def _read_csvs(source: Path) -> list[pd.DataFrame]:
    if source.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(source) as zf:
                zf.extractall(tmp)
            return [pd.read_csv(path) for path in sorted(Path(tmp).rglob("*.csv"))]
    files = sorted(source.rglob("*.csv")) if source.is_dir() else [source]
    return [pd.read_csv(path) for path in files]


def _windows(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if "window_index" in frame.columns:
        return [part.copy() for _, part in frame.groupby("window_index", sort=True)]
    return [frame]


def _features(frame: pd.DataFrame) -> dict[str, float | int | str]:
    event_id = str(frame["event_id"].iloc[0])
    target = int(frame["target"].iloc[0])
    label = frame["class_label"].iloc[0] if "class_label" in frame else ("SNI" if target else "Sham")
    time_col = "time_s" if "time_s" in frame else "time_event_s"
    time_s, signal = frame[time_col].to_numpy(float), frame["dff_raw"].to_numpy(float)
    order = np.argsort(time_s)
    time_s, signal = time_s[order], signal[order]
    time_s, unique = np.unique(time_s, return_index=True)
    signal = signal[unique]
    x = np.interp(np.linspace(time_s[0], time_s[-1], N_POINTS), time_s, signal)
    dt = WINDOW_SECONDS / N_POINTS
    rate = np.diff(x) / dt
    std = float(np.std(x, ddof=1))
    threshold = float(np.mean(x) + std)
    peaks = int(((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > threshold)).sum())
    ac1 = 0.0 if np.std(x[:-1]) <= 1e-12 or np.std(x[1:]) <= 1e-12 else float(np.corrcoef(x[:-1], x[1:])[0, 1])
    q10, q25, q75, q90 = np.quantile(x, [0.10, 0.25, 0.75, 0.90])
    return {
        "event_id": event_id, "target": target, "class_label": label,
        "dff_mean": float(np.mean(x)), "dff_std": std, "dff_min": float(np.min(x)), "dff_max": float(np.max(x)),
        "dff_range": float(np.ptp(x)), "dff_median": float(np.median(x)), "dff_q10": float(q10),
        "dff_q25": float(q25), "dff_q75": float(q75), "dff_q90": float(q90), "dff_iqr": float(q75 - q25),
        "dff_rms": float(np.sqrt(np.mean(x ** 2))), "dff_auc": float(np.trapezoid(x, dx=dt)),
        "dff_slope": float((x[-1] - x[0]) / WINDOW_SECONDS), "dff_skew": float(skew(x, bias=True)),
        "dff_kurtosis": float(kurtosis(x, bias=True)), "dff_autocorr1": ac1, "rate_mean": float(np.mean(rate)),
        "rate_abs_mean": float(np.mean(np.abs(rate))), "rate_std": float(np.std(rate, ddof=1)),
        "rate_min": float(np.min(rate)), "rate_max": float(np.max(rate)), "rate_range": float(np.ptp(rate)),
        "rate_abs_max": float(np.max(np.abs(rate))), "rate_positive_ratio": float(np.mean(rate > 0)),
        "rate_negative_ratio": float(np.mean(rate < 0)), "peak_count": peaks, "peak_density": float(peaks / N_POINTS),
        "rc_scalar_mean_plus_change": float(np.mean(x) + np.mean(np.abs(np.diff(x)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Raw-event directory, CSV, or ZIP archive")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for frame in _read_csvs(args.input):
        for window_number, window in enumerate(_windows(frame)):
            row = _features(window)
            row["window"] = window_number
            rows.append(row)
    columns = ["event_id", "target", "class_label", "window"] + [c for c in rows[0] if c not in {"event_id", "target", "class_label", "window"}]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[columns].sort_values(["event_id", "window"]).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} windows from {len(set(r['event_id'] for r in rows))} events to {args.output}")


if __name__ == "__main__":
    main()
