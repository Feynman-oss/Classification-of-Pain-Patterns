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

## Reproducibility

The workflow requires Python 3.10 or newer, NumPy, and pandas. The measured
response table is included because it is a device calibration artifact rather
than animal data. Random seeds are not used by the deterministic segment
processing step.
