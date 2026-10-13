"""Deterministic inference using the released ExtraTrees artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="extratrees_artifacts/rc_gain_models.joblib")
    parser.add_argument("--input", required=True, help="CSV containing one or more feature rows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--branch", choices=["dff", "fusion", "rc"], default="fusion")
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    if args.branch == "dff":
        model = artifact["dff_model"]
        features = artifact["dff_features"]
        threshold = artifact["thresholds"]["dff_selected"]
    elif args.branch == "rc":
        model = artifact["rc_model"]
        features = artifact["rc_features"]
        threshold = artifact["thresholds"]["fusion_selected"]
    else:
        model = artifact["fusion_model"]
        features = artifact["dff_features"] + artifact["rc_features"]
        threshold = artifact["thresholds"]["fusion_selected"]

    table = pd.read_csv(args.input)
    missing = [column for column in features if column not in table.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    probability = model.predict_proba(table.loc[:, features])[:, 1]
    prediction = (probability >= threshold).astype(int)
    result = pd.DataFrame(
        {
            "predicted_class": prediction,
            "predicted_label": ["SNI" if value else "sham" for value in prediction],
            "probability_sni": probability,
            "threshold": threshold,
        }
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
