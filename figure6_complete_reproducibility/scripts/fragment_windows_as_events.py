"""Treat each already extracted 10-s window as one independent event.

This transformation does not use animal, session, or subject identifiers. It
only creates a unique event identifier for each 10-s fragment so that the
existing grouped training code cannot place two windows from the same source
event into different folds.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    required = {"event_id", "window", "target"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = table.copy()
    out["source_event_id"] = out["event_id"].astype(str)
    out["source_window"] = out["window"].astype(int)
    out["event_id"] = out.apply(lambda r: f"{r['source_event_id']}_segment_{int(r['source_window']):03d}", axis=1)
    out["window"] = 0
    if out["event_id"].duplicated().any():
        raise ValueError("Fragment event IDs are not unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} independent 10-s fragment events to {args.output}")


if __name__ == "__main__":
    main()
