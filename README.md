# Classification of Pain Patterns reproducibility package

This clean package contains only the materials approved for public release:

- `code_only/`: training, feature-processing, validation, inference, and plotting code;
- `RC/`: the device-response table and RC.xlsx;
- `figure_replay/`: frozen numerical source tables and deterministic Figure 6/S12/S13 replay scripts;
- `environment/`: pinned environment and container files.

No animal traces, raw fluorescence recordings, animal identifiers, or other protected biological data are included.

## Exact reported-result replay

```bash
python figure_replay/scripts/verify_reported_results.py --data-dir figure_replay/data
python figure_replay/scripts/statistics_and_figures.py --data-dir figure_replay/data --output-dir figure_replay/outputs
```

This reproduces the frozen numerical source tables and figures reported in the manuscript. The frozen tables are intentionally separated from fresh retraining outputs.

## Full local training workflow

The code-only workflow is in `code_only/`. It requires the protected or controlled-access data to be placed according to `code_only/README.md`, including the paired trace workbooks and pseudo-label inputs. No such data are bundled here.

```bash
cd code_only
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt
python run_workflow.py
```

The original animal data can only be accessed under the applicable institutional and animal-ethics data-governance procedure.
