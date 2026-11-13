from __future__ import annotations

import argparse
import importlib.metadata as md
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "numpy": "2.3.4",
    "pandas": "2.3.3",
    "scipy": "1.16.2",
    "scikit-learn": "1.9.0",
    "matplotlib": "3.10.7",
    "joblib": "1.5.3",
    "openpyxl": "3.1.5",
}


def run(command: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True, env={**__import__("os").environ, "PYTHONWARNINGS": "error", "MPLBACKEND": "Agg"})


def main(full: bool) -> None:
    warnings.simplefilter("error")
    actual = {name: md.version(name) for name in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError(f"Environment mismatch: expected {EXPECTED}, obtained {actual}")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        run([sys.executable, "process_10s_segment.py", "--input", "example_10s_input.csv", "--response-table", "device_response_table.csv", "--z-min", "-0.019259", "--z-max", "0.128833", "--output", str(out / "segment.csv")])
        run([sys.executable, "predict_aggregated_features.py", "--model", "model_artifacts/best_dff_rc_gradient_boosting_model.joblib", "--input", "example_aggregated_features.csv", "--output", str(out / "aggregated.csv")])
        run([sys.executable, "predict_extratrees.py", "--model", "extratrees_artifacts/rc_gain_models.joblib", "--input", "example_extratrees_features.csv", "--output", str(out / "extratrees.csv"), "--branch", "fusion"])
        got = pd.read_csv(out / "aggregated.csv").iloc[0]
        assert int(got["predicted_class"]) == 0
        assert abs(float(got["probability_sni"]) - 0.0037948470431724045) < 1e-15
        got = pd.read_csv(out / "extratrees.csv").iloc[0]
        assert int(got["predicted_class"]) == 0
        assert float(got["probability_sni"]) == 0.0

    run([sys.executable, "fig6_reproducibility/validate_public_package.py"])
    if full:
        figure_root = ROOT / "figure6_complete_reproducibility"
        run([sys.executable, "scripts/verify_reported_results.py", "--data-dir", str(figure_root / "data")], figure_root)
        with tempfile.TemporaryDirectory() as td:
            run([
                sys.executable, "scripts/statistics_and_figures.py",
                "--data-dir", str(figure_root / "data"),
                "--output-dir", str(Path(td) / "reported_results"),
            ], figure_root)
    print("Clean-environment validation passed without warnings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    main(args.full)
