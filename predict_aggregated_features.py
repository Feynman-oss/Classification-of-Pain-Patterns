"""Run the released animal-level classifier on one anonymized feature row."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_artifacts/best_dff_rc_gradient_boosting_model.joblib")
    parser.add_argument("--input", required=True, help="CSV with one or more rows of aggregated features")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    expected = list(artifact["feature_columns"])
    table = pd.read_csv(args.input)
    missing = [column for column in expected if column not in table.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    x = table.loc[:, expected]
    probability = artifact["model"].predict_proba(x)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    result = pd.DataFrame(
        {
            "predicted_class": prediction,
            "predicted_label": [artifact["class_mapping"][str(value)] for value in prediction],
            "probability_sni": probability,
        }
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
