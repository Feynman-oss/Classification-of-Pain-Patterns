"""Validate only the public, deterministic Figure 6 example package.

This script deliberately does not read or infer restricted biological data.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    table = pd.read_csv(ROOT / "device_response_table.csv", dtype={"code": str})
    expected_codes = {format(i, "05b") for i in range(32)}
    actual_codes = set(table["code"].str.zfill(5))
    if actual_codes != expected_codes:
        raise SystemExit("device_response_table.csv must contain codes 00000-11111")
    required = {"code", "hrs_read", "hrs_peak", "lrs_read", "lrs_peak"}
    if not required.issubset(table.columns):
        raise SystemExit(f"Missing response-table columns: {sorted(required-set(table.columns))}")

    example = pd.read_csv(ROOT / "example_10s_input.csv")
    if not {"time_s", "dff"}.issubset(example.columns):
        raise SystemExit("example_10s_input.csv must contain time_s and dff")

    metadata_path = ROOT / "extratrees_artifacts" / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("classifier") != "sklearn.ensemble.ExtraTreesClassifier":
        raise SystemExit("Unexpected classifier in ExtraTrees metadata")
    if metadata.get("class_mapping") != {"0": "sham", "1": "SNI"}:
        raise SystemExit("Unexpected class mapping")

    partition = pd.read_csv(Path(__file__).with_name("partition_template.csv"))
    required_partition = {"event_id", "source_group_id", "recording_id", "session_id", "trial_id", "partition"}
    if not required_partition.issubset(partition.columns):
        raise SystemExit("partition_template.csv has an incomplete schema")
    print("Public Figure 6 package validation passed.")

if __name__ == "__main__":
    main()
