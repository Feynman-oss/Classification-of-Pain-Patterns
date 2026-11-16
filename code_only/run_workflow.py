"""Run the complete local training workflow after data are placed in the contract layout."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "scripts_training_validation"


def run(script: str) -> None:
    path = TRAIN / script
    subprocess.run([sys.executable, str(path)], cwd=TRAIN, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-reservoir", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    if not args.skip_reservoir:
        run("run_rc_reservoir_validation.py")
    run("scan_equal_duration_rc_windows.py")
    if not args.skip_baseline:
        run("run_requested_dff_rc_gradient_boosting.py")
    run("classify_sham_sni_with_rc.py")
    print("Training workflow completed; outputs are written below scripts_training_validation/.")


if __name__ == "__main__":
    main()
