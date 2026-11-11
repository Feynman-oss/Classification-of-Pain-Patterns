# Figure 6, Figure S12 and Figure S13 reproducibility package

This directory replaces the earlier inference-only archive. It separates two model branches that were previously conflated:

1. **195-feature event-aggregated branch.** Each prepared 10-s window yields 29 signal-derived quantities (including the scalar used for 32-level encoding) plus ten mapped/derived RC quantities. Five event-level statistics (mean, standard deviation, minimum, maximum and median) produce 195 ordered features. The 32-level quantizer is fitted inside each training fold.
2. **27 DFF + 9 selected RC feature branch.** The released ExtraTrees model uses the exact 27 DFF features and the nine `binary_shape` RC features listed in `models/extratrees_model_metadata.json`. The public matrix also retains the other RC groups needed for the 81-configuration sensitivity analysis.

## Contents

- `scripts/feature_pipeline.py`: 10-s resampling, 27 DFF feature extraction, five-bit/32-code RC lookup, nine RC features, fold-specific quantization and 195-feature aggregation.
- `scripts/raw_to_window_features.py`: converts the released raw 10-s event CSVs or ZIP archive into the window-level feature table used by the downstream workflows.
- `scripts/fragment_windows_as_events.py`: converts an existing window-feature table so that every 10-s fragment receives a unique event ID; no animal or subject grouping is used.
- `scripts/nested_grouped_training.py`: complete 5-fold outer/4-fold inner event-disjoint training, inner-fold threshold selection and out-of-fold prediction.
- `scripts/train_window_extratrees.py`: deterministic ExtraTrees training, train-only threshold selection and the 9 RC groups × 9 ExtraTrees settings sensitivity analysis.
- `scripts/statistics_and_figures.py`: paired event bootstrap, exact McNemar test, ROC curves, integer and row-normalized confusion matrices, and scripts for Fig. 6f, Fig. 6g, Fig. S12 and Fig. S13.
- `scripts/verify_reported_results.py`: checks the frozen numerical source tables against the accuracies and 81 configurations reported in the manuscript.
- `data/processed_10s_window_features_103_events.csv`: de-identified prepared features for 103 independent events and 136 complete 10-s windows.
- `data/processed_window_features_27dff_all_rc.csv`: de-identified fixed-split table for the 27+9 and 81-configuration analyses.
- `data/nested_grouped_oof_predictions.csv`: paired out-of-fold predictions for RC-only, DFF-only and DFF+RC.
- `data/nested_grouped_fold_settings.csv`: thresholds and quantizer limits fitted in each fold.
- `data/fig6f_reported_accuracy.csv`, `data/nested_grouped_metrics.csv`, and `data/figs12_81_configurations.csv`: numerical source data for the reported panels.
- `models/`: serialized estimators, exact feature order, quantizer metadata, class mapping, seeds and expected outputs.

## Reproduction

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python scripts/raw_to_window_features.py \
  --input 103_raw_10s_segments_only.zip \
  --output outputs/raw_recomputed_window_features.csv
.venv/Scripts/python scripts/fragment_windows_as_events.py \
  --input outputs/raw_recomputed_window_features.csv \
  --output outputs/fragment_events.csv
.venv/Scripts/python run_all.py \
  --windows outputs/fragment_events.csv
```

On Linux/macOS, replace `.venv/Scripts/` with `.venv/bin/`.

Outputs are written to `outputs/`:

- `nested_recomputed/`: model metrics, fold settings and OOF predictions recalculated from processed 10-s features;
- `window_recomputed/`: 27+9/all-RC ExtraTrees results and a recalculated 81-cell audit;
- `reported_results/`: Fig. 6f, Fig. 6g, Fig. S12, Fig. S13, ROC curves, paired bootstrap intervals and the exact McNemar test.

The figure-regeneration script uses the archived reported OOF predictions and source tables so that the manuscript figures can be regenerated deterministically. Run `verify_reported_results.py` first to verify the reported numerical values. The training scripts independently recalculate the workflows from the released processed features; retraining is an audit and can differ from the frozen report tables because of model-version, threshold, and historical partition details.

## Validation design

- Event-disjoint branch: 103 independent events, five outer folds, four inner folds, seed `20260712`.
- Quantizer fitting and threshold selection occur inside the training partition of each fold.
- Fixed window branch: 167 training windows and 72 held-out windows, seed `20260706`.
- Class mapping: `0 = Sham`, `1 = SNI`.
- Reported accuracy is calculated over held-out events for Fig. 6g and over held-out 10-s windows for Fig. 6f.

## Exact reported-result replay

```bash
python scripts/verify_reported_results.py
python scripts/statistics_and_figures.py \
  --data-dir data \
  --output-dir outputs/reported_results
```

The exact reported values are stored in `data/fig6f_reported_accuracy.csv`,
`data/nested_grouped_metrics.csv`, and `data/figs12_81_configurations.csv`.
These source tables are used only to reproduce the published figures and are
kept separate from the retraining outputs.

See `DATA_ACCESS.md` for the boundary between the public de-identified package and restricted raw fluorescence recordings.

