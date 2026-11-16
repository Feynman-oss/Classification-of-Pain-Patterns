from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from train_pain_classifier import extract_windows
from validate_folder3_detail import read_detail_trails
from run_rc_reservoir_validation import add_reservoir_features
from optimize_sham_sni_from_scratch import add_more_features, aggregate_samples


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "sham_sni_window_length_scan"
RANDOM_STATE = 20260706

CORE_FEATURES = [
    "points",
    "mean",
    "std",
    "min",
    "max",
    "range",
    "median",
    "auc",
    "slope",
    "rate_mean",
    "rate_abs_mean",
    "rate_std",
    "rate_min",
    "rate_max",
    "rate_range",
    "rate_abs_max",
    "rate_pos_ratio",
    "rate_neg_ratio",
    "peak_count",
    "mean_abs",
    "slope_abs",
    "cv",
    "range_per_std",
    "rate_energy",
    "dynamic_index",
    "asym_rate",
    "peak_density",
]


def read_detail_trails_with_window(path: Path, window_seconds: float) -> pd.DataFrame:
    # Reuse the 1-s reader, then aggregate adjacent 1-s windows for 2/5/10 s.
    # This keeps folder-3 Detail DFF parsing stable while changing the window length.
    base = read_detail_trails(path)
    if window_seconds == 1:
        return base
    rows = []
    group_cols = ["file", "pair_id", "condition", "trail", "sample_id"]
    base["big_window"] = np.floor(base["window"] / window_seconds).astype(int)
    numeric = [
        c for c in base.columns
        if c not in set(group_cols + ["window", "big_window"]) and pd.api.types.is_numeric_dtype(base[c])
    ]
    for keys, group in base.groupby(group_cols + ["big_window"]):
        if len(group) < 1:
            continue
        record = dict(zip(group_cols + ["window"], keys))
        for col in numeric:
            record[col] = group[col].mean()
        record["points"] = group["points"].sum()
        rows.append(record)
    return pd.DataFrame(rows)


def load_pain_windows(window_seconds: float) -> pd.DataFrame:
    frames = []
    for source in ["1", "2"]:
        for path in sorted((ROOT / source).glob("*pain.xlsx")):
            frame = extract_windows(path, split="all", label="pain", window_seconds=window_seconds, baseline_normalize=False)
            frame["source"] = source
            frame["model_type"] = "sham"
            frame["target"] = 0
            frame["pair_id"] = f"{source}/{path.stem.removesuffix('pain')}"
            frame["sample_id"] = f"{source}/{path.name}"
            frames.append(frame)

    for path in sorted((ROOT / "3").glob("*pain.xlsx")):
        frame = read_detail_trails_with_window(path, window_seconds)
        if frame.empty:
            continue
        stem = path.stem.lower()
        model_type = "sni" if "sni" in stem else "sham"
        frame["source"] = "3"
        frame["model_type"] = model_type
        frame["target"] = 1 if model_type == "sni" else 0
        frame["pair_id"] = "3/" + frame["pair_id"].astype(str)
        frame["sample_id"] = "3/" + frame["sample_id"].astype(str)
        frame["label"] = "pain"
        frame["y"] = 1
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False).replace([np.inf, -np.inf], np.nan).fillna(0)


def metrics(y, pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_sni": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def eval_random(df: pd.DataFrame, features: list[str], level: str, window_seconds: float, feature_set: str) -> list[dict]:
    train_idx, val_idx = train_test_split(
        np.arange(len(df)), test_size=0.30, random_state=RANDOM_STATE, stratify=df["target"]
    )
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]
    models = {
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=1200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=1200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    rows = []
    for model_name, model in models.items():
        model.fit(train[features], train["target"])
        pred = model.predict(val[features])
        m = metrics(val["target"], pred)
        rows.append(
            {
                "window_seconds": window_seconds,
                "level": level,
                "validation_type": "random_70_30",
                "feature_set": feature_set,
                "model": model_name,
                "n_total": int(len(df)),
                "n_train": int(len(train)),
                "n_validation": int(len(val)),
                "sham_total": int((df["target"] == 0).sum()),
                "sni_total": int((df["target"] == 1).sum()),
                **{f"validation_{k}": v for k, v in m.items()},
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    counts = []
    for window_seconds in [2.0, 5.0, 10.0]:
        data = add_more_features(load_pain_windows(window_seconds))
        dummy_train, _ = train_test_split(
            np.arange(len(data)), test_size=0.30, random_state=RANDOM_STATE, stratify=data["target"]
        )
        data, rc_features = add_reservoir_features(data, dummy_train)
        data.to_csv(OUT_DIR / f"sham_sni_{window_seconds:g}s_windows.csv", index=False, encoding="utf-8-sig")
        counts.append(
            {
                "window_seconds": window_seconds,
                "total_windows": len(data),
                "sham_windows": int((data["target"] == 0).sum()),
                "sni_windows": int((data["target"] == 1).sum()),
                "samples": int(data["sample_id"].nunique()),
            }
        )
        feature_sets = {
            "DFF": CORE_FEATURES,
            "RC": rc_features,
            "DFF_RC": CORE_FEATURES + rc_features,
        }
        for name, features in feature_sets.items():
            all_rows.extend(eval_random(data, features, "window", window_seconds, name))
            sample_df = aggregate_samples(data, features)
            sample_features = [
                c for c in sample_df.columns
                if c not in {"sample_id", "source", "model_type", "target"} and pd.api.types.is_numeric_dtype(sample_df[c])
            ]
            all_rows.extend(eval_random(sample_df, sample_features, "sample_aggregated", window_seconds, name))

    summary = pd.DataFrame(all_rows).sort_values(
        ["validation_balanced_accuracy", "validation_accuracy"], ascending=False
    )
    count_df = pd.DataFrame(counts)
    summary.to_csv(OUT_DIR / "window_length_scan_summary.csv", index=False, encoding="utf-8-sig")
    count_df.to_csv(OUT_DIR / "window_length_counts.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "window_length_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print("Window length scan completed.")
    print(f"Results: {OUT_DIR}")
    print(count_df.to_string(index=False))
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
