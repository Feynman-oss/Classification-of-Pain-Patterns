# Reproducible software environment

## Validated host environment

- Operating system: Microsoft Windows 10, version 10.0.19045 SP0, 64-bit
- Python: CPython 3.12.10, 64-bit (MSC v.1943)
- Package environment: `requirements-lock.txt`
- Model serialization version: scikit-learn 1.9.0
- Validation policy: `PYTHONWARNINGS=error`

## Environment files

- `requirements.txt`: exact direct dependencies.
- `requirements-lock.txt`: exact direct and transitive Python dependencies.
- `environment.yml`: Conda environment with Python and package versions pinned.
- `Dockerfile`: Linux container based on `python:3.12.10-slim-bookworm`.
- `validate_clean_environment.py`: checks versions, loads both serialized model branches, runs all expected examples and optionally reruns the complete Figure 6/S12/S13 workflow.

## Commands

Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip==26.3
.venv\Scripts\python -m pip install -r requirements-lock.txt
$env:PYTHONWARNINGS='error'
.venv\Scripts\python validate_clean_environment.py --full
```

Docker:

```bash
docker build -t pain-patterns:v2.0.2 .
docker run --rm pain-patterns:v2.0.2
```

## Clean-environment result

The complete workflow was rerun in a newly created Python 3.12.10 virtual environment on 10 November 2026. No warning was emitted while loading either serialized model or running the examples. The event-disjoint workflow reproduced DFF-only accuracy 0.854369 and DFF+RC accuracy 0.864078 over 103 events. The deterministic model examples reproduced the archived predictions exactly.
