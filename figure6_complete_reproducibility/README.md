# Exact Figure 6 reproducibility package

This directory is the deterministic replay package for the numerical results reported in Figures 6f, 6g, S12 and S13. It uses the frozen numerical source tables associated with the manuscript.

## Run the verification

From this directory, run:

```bash
python scripts/verify_reported_results.py
python scripts/statistics_and_figures.py \
  --data-dir data \
  --output-dir outputs/reported_results
```

`verify_reported_results.py` checks the reported accuracies and confirms that the S12 table contains all 81 configurations. `statistics_and_figures.py` regenerates the reported bar charts, ROC curve, confusion matrices, S12 heatmaps, bootstrap interval, and exact McNemar result.

## Frozen source tables

- `data/fig6f_reported_accuracy.csv`: the three reported Fig. 6f accuracies.
- `data/nested_grouped_metrics.csv`: the three reported Fig. 6g accuracies and confusion counts.
- `data/figs12_81_configurations.csv`: the 81 reported S12 configurations.
- `data/nested_grouped_oof_predictions.csv`: the out-of-fold predictions used for Fig. 6g statistics and S13.
- `data/fixed_validation_predictions.csv`: the stored Fig. 6f predictions.

The reported values reproduced by the package are 87.50%, 91.67% and 93.06% for Fig. 6f, and 71.84%, 85.44% and 86.41% for Fig. 6g. The package does not include an alternative retraining entry point that could silently produce different values from the frozen manuscript results.

## Reproducibility scope

This package reproduces the published numerical source tables and figures. It does not claim that a fresh model fit will necessarily return the same values. The 103-trace/136-window feature files and the raw 10-s trace archive are provided separately for data inspection and independent methodological analysis. They are not used by the deterministic figure replay command above.

The exact software environment is specified by the repository lock files and `ENVIRONMENT.md`. Raw non-deidentified animal recordings and the original behavioral acquisition data are not publicly distributed.
