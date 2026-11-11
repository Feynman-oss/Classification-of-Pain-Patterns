"""Verify and replay the numerical values reported in Figures 6f, 6g and S12.

This is a source-data verification entry point. It checks the frozen numerical
tables used to draw the manuscript figures and deliberately keeps them separate
from the optional retraining scripts, whose results can vary with implementation
details and model-library versions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

EXPECTED_FIG6F = {"RC only": 0.875, "DFF only": 0.9166666666666666, "DFF + RC": 0.9305555555555556}
EXPECTED_FIG6G = {"RC_only": 0.7184466019417476, "DFF_only": 0.8543689320388349, "DFF_plus_RC": 0.8640776699029126}


def check_close(actual: float, expected: float, name: str) -> None:
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    data = args.data_dir

    fig6f = pd.read_csv(data / "fig6f_reported_accuracy.csv")
    if len(fig6f) != 3:
        raise AssertionError(f"Fig. 6f source table has {len(fig6f)} rows, expected 3")
    for name, expected in EXPECTED_FIG6F.items():
        check_close(float(fig6f.loc[fig6f.feature_set == name, "accuracy"].iloc[0]), expected, f"Fig. 6f {name}")

    fig6g = pd.read_csv(data / "nested_grouped_metrics.csv")
    if len(fig6g) != 3:
        raise AssertionError(f"Fig. 6g source table has {len(fig6g)} rows, expected 3")
    for name, expected in EXPECTED_FIG6G.items():
        check_close(float(fig6g.loc[fig6g.feature_set == name, "accuracy"].iloc[0]), expected, f"Fig. 6g {name}")

    s12 = pd.read_csv(data / "figs12_81_configurations.csv")
    if len(s12) != 81:
        raise AssertionError(f"Fig. S12 source table has {len(s12)} rows, expected 81")
    if s12[["min_samples_leaf", "max_features", "rc_group"]].drop_duplicates().shape[0] != 81:
        raise AssertionError("Fig. S12 source table does not contain 81 unique configurations")

    print("Reported-result verification passed")
    print(f"Fig. 6f: {dict(zip(fig6f.feature_set, fig6f.accuracy))}")
    print(f"Fig. 6g: {dict(zip(fig6g.feature_set, fig6g.accuracy))}")
    print(f"Fig. S12 configurations: {len(s12)}")


if __name__ == "__main__":
    main()
