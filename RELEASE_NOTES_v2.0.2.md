# Classification of Pain Patterns v2.0.2

This patch release is the canonical software package associated with the revised manuscript. It supersedes v2.0.1 by pinning the complete computational environment and adding warning-as-error clean-environment validation.

## Added in v2.0.2

- exact direct dependencies in `requirements.txt`;
- a complete transitive lock in `requirements-lock.txt`;
- Python 3.12.10 and Conda metadata in `environment.yml`;
- a reproducible Linux container in `Dockerfile`;
- operating-system and validation documentation in `ENVIRONMENT.md`;
- machine-readable validation results in `CLEAN_ENVIRONMENT_VALIDATION.json`; and
- `validate_clean_environment.py`, which verifies package versions, model loading, deterministic examples, public-package integrity, and the reported event-level metrics with warnings treated as errors.

Raw animal photometry recordings are not redistributed. See `fig6_reproducibility/DATA_ACCESS.md`.
