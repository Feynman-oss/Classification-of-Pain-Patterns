from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

import classify_sham_sni_with_rc as data_source


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "sham_sni_equal_duration_rc_windows"
WINDOW_LENGTHS = [1.0, 2.0, 5.0, 10.0]
SAMPLES_PER_SECOND = 30
RANDOM_STATE = 20260711


def resample_segment(values: np.ndarray, times_s: np.ndarray, start: float, stop: float, points: int) -> np.ndarray | None:
    mask = (times_s >= start) & (times_s < stop)
    x = values[mask]
    t = times_s[mask]
    if len(x) < 3:
        return None
    target = np.linspace(start, stop, points, endpoint=False)
    return np.interp(target, t, x, left=x[0], right=x[-1])


def compact_stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(x, float)
    diff = np.diff(x)
    q10, q25, q75, q90 = np.quantile(x, [0.10, 0.25, 0.75, 0.90])
    out = {
        f"{prefix}mean": float(np.mean(x)),
        f"{prefix}std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        f"{prefix}min": float(np.min(x)),
        f"{prefix}max": float(np.max(x)),
        f"{prefix}range": float(np.ptp(x)),
        f"{prefix}median": float(np.median(x)),
        f"{prefix}q10": float(q10),
        f"{prefix}q25": float(q25),
        f"{prefix}q75": float(q75),
        f"{prefix}q90": float(q90),
        f"{prefix}iqr": float(q75 - q25),
        f"{prefix}first": float(x[0]),
        f"{prefix}last": float(x[-1]),
        f"{prefix}last_first": float(x[-1] - x[0]),
        f"{prefix}diff_mean": float(np.mean(diff)) if len(diff) else 0.0,
        f"{prefix}diff_std": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
        f"{prefix}diff_abs_mean": float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        f"{prefix}diff_abs_max": float(np.max(np.abs(diff))) if len(diff) else 0.0,
    }
    for bin_number, indices in enumerate(np.array_split(np.arange(len(x)), 5)):
        out[f"{prefix}timebin{bin_number}_mean"] = float(np.mean(x[indices]))
    return out


def map_binary_words(bits: np.ndarray, rc: pd.DataFrame) -> dict[str, np.ndarray]:
    usable = len(bits) - len(bits) % 5
    words = np.asarray(bits[:usable], int).reshape(-1, 5)
    codes = ["".join(str(int(v)) for v in word) for word in words]
    return {
        short: rc.loc[codes, column].to_numpy(float)
        for column, short in data_source.RC_COLUMNS.items()
    }


def segment_features(control: np.ndarray, pain: np.ndarray, rc: pd.DataFrame) -> dict[str, float]:
    control_median = float(np.median(control))
    control_mad = float(np.median(np.abs(control - control_median)))
    scale = max(1.4826 * control_mad, 1e-9)
    cz = (control - control_median) / scale
    pz = (pain - control_median) / scale

    direct_control = compact_stats(cz, "")
    direct_pain = compact_stats(pz, "")
    features: dict[str, float] = {}
    for key in direct_control:
        features[f"dff_control__{key}"] = direct_control[key]
        features[f"dff_pain__{key}"] = direct_pain[key]
        features[f"dff_delta__{key}"] = direct_pain[key] - direct_control[key]

    for mode, c_bits, p_bits in [
        ("amplitude", cz > 0.0, pz > 0.0),
        ("high_amplitude", cz > 1.0, pz > 1.0),
        ("direction", np.r_[False, np.diff(cz) > 0], np.r_[False, np.diff(pz) > 0]),
    ]:
        c_channels = map_binary_words(c_bits, rc)
        p_channels = map_binary_words(p_bits, rc)
        for channel in c_channels:
            c_stats = compact_stats(c_channels[channel], "")
            p_stats = compact_stats(p_channels[channel], "")
            for stat in c_stats:
                root = f"rc_{mode}_{channel}__{stat}"
                features[f"{root}_control"] = c_stats[stat]
                features[f"{root}_pain"] = p_stats[stat]
                features[f"{root}_delta"] = p_stats[stat] - c_stats[stat]
    return features


def build_window_table(window_seconds: float) -> pd.DataFrame:
    rc = data_source.read_rc_table()
    traces, _ = data_source.load_traces()
    pairs: dict[str, dict[str, data_source.Trace]] = {}
    for trace in traces:
        pairs.setdefault(trace.recording_id, {})[trace.condition] = trace
    rows = []
    points = max(30, int(round(window_seconds * SAMPLES_PER_SECOND)))
    for recording_id, pair in sorted(pairs.items()):
        if set(pair) != {"control", "pain"}:
            continue
        control_trace, pain_trace = pair["control"], pair["pain"]
        ct = (control_trace.timestamps_ns - control_trace.timestamps_ns[0]) / 1e9
        pt = (pain_trace.timestamps_ns - pain_trace.timestamps_ns[0]) / 1e9
        common_duration = min(float(ct[-1]), float(pt[-1]))
        complete_windows = int(np.floor(common_duration / window_seconds + 1e-9))
        for window in range(complete_windows):
            start, stop = window * window_seconds, (window + 1) * window_seconds
            control = resample_segment(control_trace.values, ct, start, stop, points)
            pain = resample_segment(pain_trace.values, pt, start, stop, points)
            if control is None or pain is None:
                continue
            rows.append(
                {
                    "recording_id": recording_id,
                    "biological_id": control_trace.biological_id,
                    "source": control_trace.source,
                    "model_type": control_trace.model_type,
                    "target": control_trace.target,
                    "window_seconds": window_seconds,
                    "window": window,
                    **segment_features(control, pain, rc),
                }
            )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)


def models() -> dict[str, object]:
    return {
        "extra_sqrt_leaf2": ExtraTreesClassifier(
            n_estimators=250, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "extra_half_leaf5": ExtraTreesClassifier(
            n_estimators=250, max_features=0.5, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "random_forest_sqrt_leaf2": RandomForestClassifier(
            n_estimators=250, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def evaluate_window_setting(table: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    excluded = {"recording_id", "biological_id", "source", "model_type", "target", "window_seconds", "window"}
    rc_features = [c for c in table if c.startswith("rc_")]
    dff_features = [c for c in table if c.startswith("dff_")]
    sets = {"RC_only": rc_features, "DFF_plus_RC": dff_features + rc_features}

    animal_labels = table.groupby("biological_id", as_index=False)["target"].first()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(table, table["target"], groups=table["biological_id"]))
    rows, predictions = [], []
    for set_name, features in sets.items():
        for model_name, model in models().items():
            probability = np.zeros(len(table), float)
            fold_number = np.zeros(len(table), int)
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                train = table.iloc[train_idx]
                test = table.iloc[test_idx]
                fitted = model.__class__(**model.get_params())
                per_animal_count = train.groupby("biological_id")["biological_id"].transform("size")
                weights = compute_sample_weight("balanced", train["target"]) / per_animal_count.to_numpy(float)
                fitted.fit(train[features], train["target"], sample_weight=weights)
                probability[test_idx] = fitted.predict_proba(test[features])[:, 1]
                fold_number[test_idx] = fold
            window_pred = (probability >= 0.5).astype(int)
            window_accuracy = float(accuracy_score(table["target"], window_pred))
            pred_table = table[["recording_id", "biological_id", "source", "model_type", "target", "window_seconds", "window"]].copy()
            pred_table["probability_sni"] = probability
            pred_table["fold"] = fold_number
            animal_pred = (
                pred_table.groupby(["biological_id", "source", "model_type", "target", "window_seconds"], as_index=False)
                .agg(probability_sni=("probability_sni", "mean"), windows=("window", "count"), fold=("fold", "first"))
            )
            animal_pred["predicted_model"] = np.where(animal_pred["probability_sni"] >= 0.5, "sni", "sham")
            animal_pred["correct"] = animal_pred["predicted_model"].eq(animal_pred["model_type"])
            y = animal_pred["target"].to_numpy(int)
            pred = (animal_pred["probability_sni"].to_numpy(float) >= 0.5).astype(int)
            cm = confusion_matrix(y, pred, labels=[0, 1])
            rows.append(
                {
                    "window_seconds": float(table["window_seconds"].iloc[0]),
                    "feature_set": set_name,
                    "model": model_name,
                    "animals": len(animal_pred),
                    "windows": len(table),
                    "animal_accuracy": float(accuracy_score(y, pred)),
                    "animal_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                    "animal_roc_auc": float(roc_auc_score(y, animal_pred["probability_sni"])),
                    "animal_f1_sni": float(f1_score(y, pred, zero_division=0)),
                    "window_accuracy_secondary": window_accuracy,
                    "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
                }
            )
            animal_pred["feature_set"] = set_name
            animal_pred["model"] = model_name
            predictions.append(animal_pred)
    return rows, predictions


def fit_final(table: pd.DataFrame, feature_set: str, model_name: str) -> dict:
    rc_features = [c for c in table if c.startswith("rc_")]
    dff_features = [c for c in table if c.startswith("dff_")]
    features = rc_features if feature_set == "RC_only" else dff_features + rc_features
    model = models()[model_name]
    per_animal_count = table.groupby("biological_id")["biological_id"].transform("size")
    weights = compute_sample_weight("balanced", table["target"]) / per_animal_count.to_numpy(float)
    model.fit(table[features], table["target"], sample_weight=weights)
    return {"model": model, "feature_columns": features, "feature_set": feature_set, "window_seconds": float(table["window_seconds"].iloc[0]), "samples_per_second": SAMPLES_PER_SECOND}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, all_predictions, tables = [], [], {}
    for window_seconds in WINDOW_LENGTHS:
        table = build_window_table(window_seconds)
        tables[window_seconds] = table
        rows, predictions = evaluate_window_setting(table)
        all_rows.extend(rows)
        all_predictions.extend(predictions)
        table.to_csv(OUT_DIR / f"{window_seconds:g}s_window_features.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(all_rows).sort_values(
        ["animal_balanced_accuracy", "animal_accuracy", "animal_roc_auc"], ascending=False
    )
    predictions = pd.concat(all_predictions, ignore_index=True)
    best = summary.iloc[0]
    best_predictions = predictions[
        (predictions["window_seconds"] == best["window_seconds"])
        & (predictions["feature_set"] == best["feature_set"])
        & (predictions["model"] == best["model"])
    ].copy()
    artifact = fit_final(tables[float(best["window_seconds"])], str(best["feature_set"]), str(best["model"]))
    joblib.dump(artifact, OUT_DIR / "best_equal_duration_rc_model.joblib")
    summary.to_csv(OUT_DIR / "window_length_scan_summary.csv", index=False, encoding="utf-8-sig")
    best_predictions.to_csv(OUT_DIR / "best_animal_predictions.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT_DIR / "equal_duration_rc_window_report.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="window_scan", index=False)
        best_predictions.to_excel(writer, sheet_name="best_predictions", index=False)
        predictions.to_excel(writer, sheet_name="all_predictions", index=False)
        pd.DataFrame(
            {
                "rule": [
                    "Each folder-3 Detail DFF trail is one animal.",
                    "Each folder-1/2 control-pain file pair is one animal.",
                    "Only complete equal-duration windows are retained.",
                    "All windows from one animal remain in the same CV fold.",
                    "Source, original point count and total duration are excluded from model features.",
                ]
            }
        ).to_excel(writer, sheet_name="validation_rules", index=False)
    report = {
        "best_result": best.to_dict(),
        "window_lengths_seconds": WINDOW_LENGTHS,
        "samples_per_second": SAMPLES_PER_SECOND,
        "validation": "5-fold stratified group CV by animal; animal probability is the mean of its window probabilities",
        "acquisition_metadata_used": False,
    }
    with open(OUT_DIR / "equal_duration_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(summary.to_string(index=False))
    print("\nBest")
    print(best.to_string())
    print("\nOutputs", OUT_DIR)


if __name__ == "__main__":
    main()
