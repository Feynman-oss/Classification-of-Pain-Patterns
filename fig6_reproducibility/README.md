# Figure 6 reproducibility package

This directory documents and validates the publicly releasable part of the Figure 6 workflow.
The application used an offline, hardware-calibrated lookup table: a prepared 10-s event is
encoded as a five-bit code, the corresponding response vector is retrieved from the measured
32-code table, and the resulting features are passed to the released ExtraTrees estimator.

The public package does not contain restricted raw animal photometry. It therefore supports
deterministic processing of a de-identified/prepared 10-s event and deterministic model
inference. It does not claim that the restricted biological dataset can be regenerated without
controlled access.

## Public files

- `../process_10s_segment.py`: resampling, scalar encoding, five-bit code generation, and
  lookup-table mapping.
- `../predict_extratrees.py`: inference with the released ExtraTrees artifact.
- `../device_response_table.csv`: the 32-code hardware-calibration table used by the example.
- `../example_10s_input.csv`: a prepared 10-s input example.
- `../example_extratrees_features.csv`: the feature row used by the deterministic model example.
- `partition_template.csv`: schema for anonymized event-level partition metadata. The template
  contains no biological identifiers or restricted data.
- `validate_public_package.py`: checks the response table, example schema, metadata, and expected
  prediction files.

## Scope and limitations

The physical device was measured before the classification analysis; biological events were not
streamed to the device in real time. The response table is consequently an offline
hardware-calibrated transformation. Raw animal recordings, institution-specific file paths, and
restricted animal/session metadata are not redistributed. The exact restricted-data access
procedure is described in the manuscript's Data Availability statement.

## Run the public example

From this directory's parent package directory:

```bash
python process_10s_segment.py --input example_10s_input.csv \
  --response-table device_response_table.csv \
  --z-min -0.019259 --z-max 0.128833 \
  --output example_output_reproduced.csv

python predict_extratrees.py \
  --model extratrees_artifacts/rc_gain_models.joblib \
  --input example_extratrees_features.csv \
  --output example_extratrees_prediction_reproduced.csv \
  --branch fusion

python fig6_reproducibility/validate_public_package.py
```

The expected prediction is stored in `example_extratrees_prediction.csv`. The validation script
compares the generated schema and prediction with the released expected output.
