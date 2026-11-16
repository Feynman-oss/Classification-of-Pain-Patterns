from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import classify_sham_sni_with_rc as data_source
from scan_equal_duration_rc_windows import resample_segment


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "requested_dff_rc_gradient_boosting"
WINDOW_SECONDS = 10.0
RESAMPLE_POINTS = 300
RANDOM_STATE = 20260712


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def dff_features(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, float)
    dt = WINDOW_SECONDS / RESAMPLE_POINTS
    diff = np.diff(x)
    rate = diff / dt
    abs_rate = np.abs(rate)
    q10, q25, q75, q90 = np.quantile(x, [0.10, 0.25, 0.75, 0.90])
    threshold = float(np.mean(x) + np.std(x, ddof=1))
    peaks = int(((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > threshold)).sum())
    return {
        "dff_mean": float(np.mean(x)),
        "dff_std": float(np.std(x, ddof=1)),
        "dff_min": float(np.min(x)),
        "dff_max": float(np.max(x)),
        "dff_range": float(np.ptp(x)),
        "dff_median": float(np.median(x)),
        "dff_q10": float(q10),
        "dff_q25": float(q25),
        "dff_q75": float(q75),
        "dff_q90": float(q90),
        "dff_iqr": float(q75 - q25),
        "dff_rms": float(np.sqrt(np.mean(x**2))),
        "dff_auc": float(np.mean(x) * WINDOW_SECONDS),
        "dff_slope": float((x[-1] - x[0]) / WINDOW_SECONDS),
        "dff_skew": float(skew(x, bias=False)),
        "dff_kurtosis": float(kurtosis(x, bias=False)),
        "dff_autocorr1": safe_corr(x[:-1], x[1:]),
        "rate_mean": float(np.mean(rate)),
        "rate_abs_mean": float(np.mean(abs_rate)),
        "rate_std": float(np.std(rate, ddof=1)),
        "rate_min": float(np.min(rate)),
        "rate_max": float(np.max(rate)),
        "rate_range": float(np.ptp(rate)),
        "rate_abs_max": float(np.max(abs_rate)),
        "rate_positive_ratio": float(np.mean(rate > 0)),
        "rate_negative_ratio": float(np.mean(rate < 0)),
        "peak_count": peaks,
        "peak_density": float(peaks / len(x)),
        # Requested scalar used for 32-level/5-bit RC input.
        "rc_scalar_mean_plus_change": float(np.mean(x) + np.mean(np.abs(diff))),
    }


def build_windows() -> pd.DataFrame:
    traces, _ = data_source.load_traces()
    rows = []
    for trace in traces:
        if trace.condition != "pain":
            continue
        time_s = (trace.timestamps_ns - trace.timestamps_ns[0]) / 1e9
        complete = int(np.floor(float(time_s[-1]) / WINDOW_SECONDS + 1e-9))
        for window in range(complete):
            start, stop = window * WINDOW_SECONDS, (window + 1) * WINDOW_SECONDS
            x = resample_segment(trace.values, time_s, start, stop, RESAMPLE_POINTS)
            if x is None:
                continue
            rows.append(
                {
                    "biological_id": trace.biological_id,
                    "source": str(trace.source),
                    "model_type": trace.model_type,
                    "target": trace.target,
                    "window": window,
                    **dff_features(x),
                }
            )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)


def add_fold_rc_features(
    windows: pd.DataFrame,
    train_animals: set[str],
    rc: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = windows[windows["biological_id"].isin(train_animals)]
    z_min = float(train["rc_scalar_mean_plus_change"].min())
    z_max = float(train["rc_scalar_mean_plus_change"].max())
    if z_max <= z_min:
        z_max = z_min + 1e-12
    value = windows["rc_scalar_mean_plus_change"].to_numpy(float)
    q = np.clip(np.rint(31.0 * (value - z_min) / (z_max - z_min)), 0, 31).astype(int)
    codes = pd.Series([format(int(v), "05b") for v in q], index=windows.index)
    out = windows.copy()
    out["rc_quantized_level"] = q
    out["rc_code"] = codes
    for source, short in data_source.RC_COLUMNS.items():
        out[f"rc_{short}"] = codes.map(rc[source]).astype(float)
    out["rc_hrs_delta"] = out["rc_hrs_peak"] - out["rc_hrs_read"]
    out["rc_lrs_delta"] = out["rc_lrs_peak"] - out["rc_lrs_read"]
    out["rc_read_gap"] = out["rc_hrs_read"] - out["rc_lrs_read"]
    out["rc_peak_gap"] = out["rc_hrs_peak"] - out["rc_lrs_peak"]
    out["rc_energy"] = sum(out[f"rc_{short}"].abs() for short in data_source.RC_COLUMNS.values())
    return out, {"z_min": z_min, "z_max": z_max}


def feature_columns(table: pd.DataFrame, set_name: str) -> list[str]:
    dff = [
        c for c in table
        if c.startswith(("dff_", "rate_", "peak_")) and c != "rc_scalar_mean_plus_change"
    ]
    rc = [c for c in table if c.startswith("rc_") and c not in {"rc_code", "rc_scalar_mean_plus_change"}]
    if set_name == "DFF_only":
        return dff
    if set_name == "RC_only":
        return rc
    if set_name == "DFF_plus_RC":
        return dff + rc
    raise KeyError(set_name)


def aggregate_animals(windows: pd.DataFrame) -> pd.DataFrame:
    metadata = ["biological_id", "source", "model_type", "target"]
    features = [c for c in windows if c.startswith(("dff_", "rate_", "peak_", "rc_")) and c != "rc_code"]
    animal = windows.sort_values("window").groupby(metadata)[features].agg(["mean", "std", "min", "max", "median"])
    animal.columns = [f"{feature}__{stat}" for feature, stat in animal.columns]
    return animal.reset_index().replace([np.inf, -np.inf], np.nan).fillna(0).copy()


def gb_models() -> dict[str, GradientBoostingClassifier]:
    return {
        "GB_default": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "GB_shallow": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=1,
            min_samples_leaf=3, subsample=0.8, random_state=RANDOM_STATE,
        ),
        "GB_regularized": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.03, max_depth=2,
            min_samples_leaf=4, subsample=0.8, max_features="sqrt", random_state=RANDOM_STATE,
        ),
        "GB_deeper": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=3,
            min_samples_leaf=2, subsample=0.8, random_state=RANDOM_STATE,
        ),
    }


def metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    pred = (probability >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "f1_sni": float(f1_score(y, pred, zero_division=0)),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = build_windows()
    rc = data_source.read_rc_table()
    metadata = ["biological_id", "source", "model_type", "target"]
    animals = windows[metadata].drop_duplicates().sort_values("biological_id").reset_index(drop=True)
    y_animals = animals["target"].to_numpy(int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(animals, y_animals))

    results, prediction_tables, fold_quantizers = [], [], []
    for validation_level in ["animal_aggregated", "window_vote_to_animal"]:
        for set_name in ["DFF_only", "RC_only", "DFF_plus_RC"]:
            for model_name in gb_models():
                probability = np.zeros(len(animals), float)
                fold_number = np.zeros(len(animals), int)
                for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                    train_animals = set(animals.iloc[train_idx]["biological_id"])
                    test_animals = set(animals.iloc[test_idx]["biological_id"])
                    encoded, quantizer = add_fold_rc_features(windows, train_animals, rc)
                    fold_quantizers.append({"fold": fold, **quantizer})
                    model = gb_models()[model_name]
                    if validation_level == "animal_aggregated":
                        table = aggregate_animals(encoded)
                        table = animals.merge(table, on=metadata, validate="one_to_one")
                        if set_name == "DFF_only":
                            features = [c for c in table if c.startswith(("dff_", "rate_", "peak_"))]
                        elif set_name == "RC_only":
                            features = [c for c in table if c.startswith("rc_")]
                        else:
                            features = [c for c in table if c.startswith(("dff_", "rate_", "peak_", "rc_"))]
                        model.fit(table.iloc[train_idx][features], y_animals[train_idx])
                        probability[test_idx] = model.predict_proba(table.iloc[test_idx][features])[:, 1]
                    else:
                        features = feature_columns(encoded, set_name)
                        train_windows = encoded[encoded["biological_id"].isin(train_animals)]
                        test_windows = encoded[encoded["biological_id"].isin(test_animals)]
                        count = train_windows.groupby("biological_id")["biological_id"].transform("size")
                        weights = 1.0 / count.to_numpy(float)
                        model.fit(train_windows[features], train_windows["target"], sample_weight=weights)
                        window_probability = model.predict_proba(test_windows[features])[:, 1]
                        animal_probability = (
                            test_windows.assign(probability_sni=window_probability)
                            .groupby("biological_id")["probability_sni"].mean()
                        )
                        probability[test_idx] = animals.iloc[test_idx]["biological_id"].map(animal_probability).to_numpy(float)
                    fold_number[test_idx] = fold
                result_metrics = metrics(y_animals, probability)
                results.append(
                    {
                        "validation_level": validation_level,
                        "feature_set": set_name,
                        "model": model_name,
                        "animals": len(animals),
                        "pain_windows": len(windows),
                        **result_metrics,
                    }
                )
                pred = animals.copy()
                pred["probability_sni"] = probability
                pred["predicted_model"] = np.where(probability >= 0.5, "sni", "sham")
                pred["correct"] = pred["predicted_model"].eq(pred["model_type"])
                pred["fold"] = fold_number
                pred["validation_level"] = validation_level
                pred["feature_set"] = set_name
                pred["model"] = model_name
                prediction_tables.append(pred)

    summary = pd.DataFrame(results).sort_values(["balanced_accuracy", "accuracy", "roc_auc"], ascending=False)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    best = summary.iloc[0]
    best_predictions = predictions[
        (predictions["validation_level"] == best["validation_level"])
        & (predictions["feature_set"] == best["feature_set"])
        & (predictions["model"] == best["model"])
    ].copy()

    # Fit a final artifact with all-data quantization for later inference.
    encoded, final_quantizer = add_fold_rc_features(windows, set(animals["biological_id"]), rc)
    if best["validation_level"] == "animal_aggregated":
        final_table = animals.merge(aggregate_animals(encoded), on=metadata, validate="one_to_one")
        if best["feature_set"] == "DFF_only":
            final_features = [c for c in final_table if c.startswith(("dff_", "rate_", "peak_"))]
        elif best["feature_set"] == "RC_only":
            final_features = [c for c in final_table if c.startswith("rc_")]
        else:
            final_features = [c for c in final_table if c.startswith(("dff_", "rate_", "peak_", "rc_"))]
        final_y = final_table["target"]
    else:
        final_table = encoded
        final_features = feature_columns(encoded, str(best["feature_set"]))
        final_y = encoded["target"]
    final_model = gb_models()[str(best["model"])]
    final_model.fit(final_table[final_features], final_y)
    artifact = {
        "model": final_model,
        "feature_columns": final_features,
        "quantizer": final_quantizer,
        "window_seconds": WINDOW_SECONDS,
        "resample_points": RESAMPLE_POINTS,
        "rc_formula": "Mean(DFF) + Mean(abs(diff(DFF)))",
        "validation_level": str(best["validation_level"]),
        "feature_set": str(best["feature_set"]),
        "input_condition": "pain_only",
        "uses_batch_metadata": False,
    }
    joblib.dump(artifact, OUT_DIR / "best_dff_rc_gradient_boosting_model.joblib")
    windows.to_csv(OUT_DIR / "pain_10s_dff_window_features.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "dff_rc_gradient_boosting_summary.csv", index=False, encoding="utf-8-sig")
    best_predictions.to_csv(OUT_DIR / "best_dff_rc_gradient_boosting_predictions.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT_DIR / "dff_rc_gradient_boosting_report.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="model_comparison", index=False)
        best_predictions.to_excel(writer, sheet_name="best_predictions", index=False)
        predictions.to_excel(writer, sheet_name="all_predictions", index=False)
        pd.DataFrame(fold_quantizers).drop_duplicates().sort_values("fold").to_excel(writer, sheet_name="fold_quantizers", index=False)
        pd.DataFrame(
            {"method": [
                "Pain-state DFF only; complete 10 s windows resampled to 300 points.",
                "DFF time-domain and rate features are extracted per window.",
                "RC scalar = Mean(DFF) + Mean(abs(diff(DFF))).",
                "Training-fold min/max maps the scalar to q=0..31 and a five-bit code.",
                "The code is mapped to four measured RC.xlsx channels plus channel interactions.",
                "Gradient boosting is evaluated with animal-level five-fold validation.",
                "No source, duration, raw point count, filename or control feature is used.",
            ]}
        ).to_excel(writer, sheet_name="method", index=False)
    report = {
        "best_result": best.to_dict(),
        "requested_method": True,
        "rc_formula": "Mean(DFF) + Mean(abs(diff(DFF)))",
        "input_condition": "pain_only",
        "uses_batch_metadata": False,
        "validation": "animal-level five-fold CV; min/max quantizer fitted on each training fold only",
    }
    with open(OUT_DIR / "dff_rc_gradient_boosting_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(summary.to_string(index=False))
    print("\nBest")
    print(best.to_string())
    print("\nOutputs", OUT_DIR)


if __name__ == "__main__":
    main()
