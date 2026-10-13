# 10-s fluorescence-segment processing

This repository contains the public code for processing an already prepared
10-s fluorescence segment. It implements the signal transformation used in
the manuscript:

1. load a prepared segment with `time_s` and `dff` columns;
2. linearly resample the segment to 300 equally spaced points;
3. calculate `z = mean(dff) + mean(abs(diff(dff)))`;
4. quantize `z` to 32 levels using `z_min` and `z_max` supplied from the
   training partition;
5. represent the level as a most-significant-bit-first five-bit code; and
6. map the code to the measured four-channel device-response table.

The repository intentionally does **not** contain raw animal photometry data,
animal/session/trial splitting code, or institution-specific raw-file
organization. The input to the public workflow is one prepared 10-s segment.
The `z_min` and `z_max` arguments must be calculated from the authorized
training partition and must not be estimated from the held-out segment.

## Input format

`example_10s_input.csv` illustrates the required input schema:

```text
time_s,dff
0.000,0.0123
0.034,0.0128
...
```

The segment must contain finite timestamps and fluorescence values. The code
sorts timestamps, removes duplicate timestamps, and linearly interpolates to
300 points between the first and last timestamp.

## Run

```bash
python process_10s_segment.py \
  --input example_10s_input.csv \
  --response-table device_response_table.csv \
  --z-min -0.019259 \
  --z-max 0.128833 \
  --output example_output.csv
```

The output contains the scalar encoding value, quantized level, five-bit code,
and the four mapped response channels: `hrs_read`, `hrs_peak`, `lrs_read`, and
`lrs_peak`.

## Released animal-level model artifact

The primary animal-level model reported in the manuscript is released as
`model_artifacts/best_dff_rc_gradient_boosting_model.joblib`. It is a
scikit-learn `GradientBoostingClassifier` with `random_state=20260712`,
`n_estimators=100`, `learning_rate=0.1`, `max_depth=3`,
`min_samples_leaf=1`, `subsample=1.0`, `max_features=None`, and
`loss=log_loss`. Class `0` is `sham` and class `1` is `SNI`.

The artifact contains the fixed order of all 195 aggregated features, the
training quantizer, the 10-s window and 300-point resampling settings, and the
reservoir encoding formula. The complete feature order and metadata are also
provided in `model_artifacts/model_metadata.json`.

To run deterministic inference on an anonymized aggregated feature row:

```bash
python predict_aggregated_features.py \
  --model model_artifacts/best_dff_rc_gradient_boosting_model.joblib \
  --input example_aggregated_features.csv \
  --output example_prediction_reproduced.csv
```

The expected output is stored in `example_prediction.csv`.

## Released ExtraTrees artifact

The ExtraTrees artifact requested for reproducibility is provided in
`extratrees_artifacts/rc_gain_models.joblib`. The fusion branch contains an
ExtraTrees classifier with 1,200 trees, `class_weight="balanced"`,
`min_samples_leaf=1`, `max_features="sqrt"`, and `random_state=20260706`.
The fixed DFF and reservoir feature orders, selected reservoir group,
classification threshold, and class mapping are recorded in
`extratrees_artifacts/model_metadata.json`.

Run deterministic inference with:

```bash
python predict_extratrees.py \
  --model extratrees_artifacts/rc_gain_models.joblib \
  --input example_extratrees_features.csv \
  --output example_extratrees_prediction_reproduced.csv \
  --branch fusion
```

The expected result is provided in `example_extratrees_prediction.csv`.
No additional neural network was used in this study.

## Reproducibility

The workflow requires Python 3.10 or newer. NumPy and pandas support segment
processing; scikit-learn and joblib support deterministic model inference. The
measured response table is included because it is a device calibration
artifact rather than animal data. Raw animal recordings and
animal/session/trial splitting code are not included.
