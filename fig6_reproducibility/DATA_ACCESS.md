# Data access and reproducibility boundary

The public repository contains code, model metadata, the measured 32-code device-response
table, a prepared 10-s example, and anonymized schema files. Raw animal photometry and identifying
animal/session/trial metadata are restricted and are not redistributed in the public repository.

For a controlled-access request, the data provider should supply the de-identified event table,
the corresponding source-group mapping, and the partition manifest. The public processing code
expects a prepared 10-s event with `time_s` and `dff` columns and applies training-derived
encoding limits. No held-out event should be used to estimate preprocessing or quantization
parameters.

The public example is intended to verify software installation, five-bit encoding, response-table
lookup, feature ordering, and deterministic ExtraTrees inference. It is not a substitute for the
restricted biological source data and does not by itself reproduce group-level biological plots.
