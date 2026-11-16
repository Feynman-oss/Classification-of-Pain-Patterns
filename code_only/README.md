# Fig. 6 and S12–S13 code-only workflow

This package contains code only. It does not include animal recordings, device-response data, trained models, predictions, or figure source tables. Those materials must be supplied separately under the directory layout below.

## Required data layout

Place the following under `scripts_training_validation/`:

```text
scripts_training_validation/
  RC.xlsx
  1/*control.xlsx and *pain.xlsx
  2/*control.xlsx and *pain.xlsx
  3/*control.xlsx and *pain.xlsx
  all_folders_mixed_results/all_folders_1s_pseudo_labels.csv
```

The Excel files must retain the sheet names and columns expected by the parsers (`Overall DFF` and/or `Detail DFF`). The device table must contain all 32 five-bit codes in the `RC.xlsx` sheet used by the scripts.

## Run the training workflow

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python run_workflow.py
```

To skip the reservoir-only stage when its pseudo-label input is unavailable:

```bash
python run_workflow.py --skip-reservoir
```

The workflow performs reservoir validation, scans equal-duration windows, generates DFF+RC features, and runs the paired Sham/SNI classifier. Outputs are written into subdirectories under `scripts_training_validation/`.

## Individual stages

- `scripts_training_validation/classify_sham_sni_with_rc.py`: paired trace loading, 1-s window features, reservoir lookup, grouped validation, and model artifacts.
- `scripts_training_validation/scan_equal_duration_rc_windows.py`: 1-, 2-, 5- and 10-s window construction and validation.
- `scripts_training_validation/run_requested_dff_rc_gradient_boosting.py`: 10-s DFF/RC feature construction, train-only quantization, model comparison, and predictions.
- `scripts_training_validation/run_rc_reservoir_validation.py`: reservoir-only validation from the pseudo-label input and `RC.xlsx`.
- `scripts_plotting/`: figure-generation scripts for the main RC/DFF figures and supporting panels.
- `scripts_inference/`: deterministic processing and inference for prepared 10-s inputs.

## Reproducibility boundary

The workflow is deterministic conditional on the supplied data files, their trace pairing, the partition metadata, the pinned environment, and the random seeds in the scripts. The code does not reconstruct missing biological recordings or device measurements. Generated figures and metrics should be checked against the supplied source data and manuscript-specific preprocessing choices.
