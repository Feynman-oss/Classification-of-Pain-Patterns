"""Run the public Figure 6/S12/S13 reproducibility workflow."""
from pathlib import Path
import subprocess, sys

ROOT=Path(__file__).resolve().parent
PY=sys.executable
commands=[
 [PY,str(ROOT/'scripts/nested_grouped_training.py'),'--windows',str(ROOT/'data/processed_10s_window_features_103_events.csv'),'--response-table',str(ROOT/'data/device_response_table.csv'),'--output-dir',str(ROOT/'outputs/nested_recomputed')],
 [PY,str(ROOT/'scripts/train_window_extratrees.py'),'--features',str(ROOT/'data/processed_window_features_27dff_all_rc.csv'),'--output-dir',str(ROOT/'outputs/window_recomputed')],
 [PY,str(ROOT/'scripts/statistics_and_figures.py'),'--data-dir',str(ROOT/'data'),'--output-dir',str(ROOT/'outputs/reported_results')],
]
for command in commands:
    print(' '.join(command),flush=True); subprocess.run(command,check=True,cwd=ROOT)
print('All outputs written under',ROOT/'outputs')
