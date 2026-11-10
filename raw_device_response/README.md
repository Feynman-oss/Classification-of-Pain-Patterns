# Raw device-response calibration data

This directory contains the raw current-voltage-time exports used to construct the nominal
32-code, four-channel response table. The calibration is a single-device, single-batch,
single-acquisition dataset. It does not provide device-to-device, batch-to-batch, or repeated-run
statistics.

## Files

- `hrs/00001.csv` to `hrs/11111.csv`: positive-polarity HRS-mode records.
- `lrs/00001.csv` to `lrs/11111.csv`: negative-polarity LRS-mode records.
- `00000` is the defined zero-input reference `[0, 0, 0, 0]`; no physical pulse file was
  generated for this code.
- `extract_response_table.py`: reconstructs the nominal four-channel table from the raw files.
- `manifest_sha256.csv`: SHA-256 checksums for every released raw CSV.

The HRS `01111.csv` released here corresponds to the acquisition named `01111-1.csv` in the
original working directory because that acquisition was used for the published lookup-table row.

## Acquisition sequence inferred from the exported source-voltage column

Each CSV contains 66 samples from 0.0005 to 1.3005 s. The nominal sampling interval is 20 ms.
The five-bit code occupies the first 0.40 s. A binary 1 is represented by a 40 ms pulse of +1 V
(HRS mode) or -1 V (LRS mode), with a 40 ms zero-voltage interval between bit positions. Binary 0
is represented by 0 V. After the code, the sequence contains 40 ms at 0 V, a fixed 40 ms probe
pulse of +1 V or -1 V, 40 ms at 0 V, a 400 ms read segment at +0.1 V or -0.1 V, and a final
400 ms zero-voltage segment.

The peak channel is the current at the final sample of the fixed +/-1 V probe pulse (0.4605 s).
The readout channel is the current at the first sample of the +/-0.1 V read segment (0.5205 s).
The table values may be rounded relative to the exported raw current.

Measurements were acquired in the dark at room temperature using a Keithley B2902B connected to
an LNP-3RT probe station. The exported CSV does not contain a compliance-current field or the
external state-initialization command. Those settings must be reported from the instrument method
or laboratory record and cannot be reconstructed from these raw exports alone.
