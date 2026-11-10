# Data access and de-identification

The public tables contain de-identified, processed event features and out-of-fold predictions. Original animal/session names were replaced by sequential `event_###` identifiers. The public release contains 103 independent event identifiers for the event-disjoint analysis.

Raw fluorescence recordings are not redistributed because the authors' data-use restrictions do not permit unrestricted public release. Qualified researchers may request controlled access from the corresponding author, subject to the applicable institutional and data-use requirements. The restriction applies to the raw fluorescence files, not to the code, model artifacts, measured 32-code device table, de-identified processed features, fold assignments, out-of-fold predictions, or figure source data released here.

`deidentification_note.json` records the number of released events and windows. The public tables are sufficient to rerun model fitting from processed features, threshold selection, nested event-disjoint validation, paired bootstrap analysis, McNemar testing, ROC analysis, confusion matrices, and the reported figure panels. Regenerating the processed features from the non-public raw fluorescence files requires controlled access.
