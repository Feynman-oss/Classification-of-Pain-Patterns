"""Process one prepared 10-s fluorescence segment.

This package intentionally starts from a prepared 10-s segment. It does not
read raw animal files or perform animal/session/trial splitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


N_POINTS = 300


def load_segment(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV containing `time_s` and `dff` columns."""
    frame = pd.read_csv(path)
    required = {"time_s", "dff"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame[["time_s", "dff"]].replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.sort_values("time_s").drop_duplicates("time_s")
    if len(frame) < 2:
        raise ValueError("At least two finite samples are required")
    return frame.time_s.to_numpy(float), frame.dff.to_numpy(float)


def resample_10s(time_s: np.ndarray, dff: np.ndarray) -> np.ndarray:
    """Linearly resample one 10-s segment to 300 equally spaced points."""
    if time_s[-1] - time_s[0] <= 0:
        raise ValueError("time_s must be strictly increasing")
    target = np.linspace(time_s[0], time_s[-1], N_POINTS)
    return np.interp(target, time_s, dff)


def encode_segment(x: np.ndarray, z_min: float, z_max: float) -> dict[str, object]:
    """Compute the paper scalar and map it to a five-bit code.

    z_min and z_max must be supplied from the training partition. They are not
    estimated from the segment being processed.
    """
    if z_max <= z_min:
        raise ValueError("z_max must be larger than z_min")
    first_difference = np.diff(x)
    z = float(np.mean(x) + np.mean(np.abs(first_difference)))
    q = int(np.clip(np.rint(31.0 * (z - z_min) / (z_max - z_min)), 0, 31))
    return {
        "z": z,
        "quantized_level": q,
        "code": format(q, "05b"),
        "n_points": int(x.size),
    }


def load_response_table(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"code": str})
    required = {"code", "hrs_read", "hrs_peak", "lrs_read", "lrs_peak"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing response-table columns: {sorted(missing)}")
    table["code"] = table["code"].str.zfill(5)
    if set(table["code"]) != {format(i, "05b") for i in range(32)}:
        raise ValueError("The response table must contain exactly 00000–11111")
    return table.set_index("code")


def process_segment(input_csv: str | Path, response_csv: str | Path, z_min: float, z_max: float) -> dict[str, object]:
    time_s, dff = load_segment(input_csv)
    x = resample_10s(time_s, dff)
    encoded = encode_segment(x, z_min, z_max)
    response = load_response_table(response_csv).loc[encoded["code"]]
    return {**encoded, **response.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared 10-s CSV with time_s,dff")
    parser.add_argument("--response-table", required=True, help="Measured 32-code response CSV")
    parser.add_argument("--z-min", required=True, type=float, help="Training-partition lower encoding limit")
    parser.add_argument("--z-max", required=True, type=float, help="Training-partition upper encoding limit")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()
    pd.DataFrame([process_segment(args.input, args.response_table, args.z_min, args.z_max)]).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
