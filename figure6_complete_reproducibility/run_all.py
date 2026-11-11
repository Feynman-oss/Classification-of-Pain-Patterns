"""Run the public Figure 6/S12/S13 reproducibility workflow."""
from pathlib import Path
import argparse
import subprocess, sys

ROOT=Path(__file__).resolve().parent
PY=sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--windows', default=str(ROOT/'data/processed_10s_window_features_103_events.csv'), help='Window-feature CSV; can be generated from raw_to_window_features.py')
parser.add_argument('--features', default=str(ROOT/'data/processed_window_features_27dff_all_rc.csv'))
args = parser.parse_args()
commands=[
 [PY,str(ROOT/'scripts/nested_grouped_training.py'),'--windows',args.windows,'--response-table',str(ROOT/'data/device_response_table.csv'),'--output-dir',str(ROOT/'outputs/nested_recomputed')],
 [PY,str(ROOT/'scripts/train_window_extratrees.py'),'--features',args.features,'--output-dir',str(ROOT/'outputs/window_recomputed')],
 [PY,str(ROOT/'scripts/statistics_and_figures.py'),'--data-dir',str(ROOT/'data'),'--output-dir',str(ROOT/'outputs/reported_results')],
]
for command in commands:
    print(' '.join(command),flush=True); subprocess.run(command,check=True,cwd=ROOT)
print('All outputs written under',ROOT/'outputs')
