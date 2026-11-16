from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "all_folders_mixed_results" / "all_folders_1s_pseudo_labels.csv"
RC_PATH = ROOT / "RC.xlsx"
OUT_DIR = ROOT / "rc_reservoir_results"
RANDOM_STATE = 20260628


CORE_FEATURES = [
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
    "points",
]

RELATIVE_FEATURES = [
    "control_relative_z_mean",
    "control_relative_z_std",
    "control_relative_z_range",
    "control_relative_z_slope",
    "control_relative_z_rate_abs_mean",
    "control_relative_z_rate_std",
    "control_relative_z_rate_abs_max",
    "control_relative_z_peak_count",
]


def read_rc() -> pd.DataFrame:
    rc = pd.read_excel(RC_PATH, sheet_name=0)
    rc = rc.drop(columns=[c for c in rc.columns if str(c).startswith("Unnamed")], errors="ignore")
    rc = rc.dropna(how="all").copy()
    rc["编码"] = rc["编码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    return rc.set_index("编码")


def quantile_bins(values: pd.Series, n_bins: int) -> np.ndarray:
    edges = values.quantile(np.linspace(0, 1, n_bins + 1)[1:-1]).to_numpy(float)
    return np.unique(edges)


def bin_code(values: pd.Series, edges: np.ndarray, n_bins: int) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values.to_numpy(float), side="right"), 0, n_bins - 1)


def binary_feature_code(df: pd.DataFrame, train_df: pd.DataFrame, features: list[str]) -> pd.Series:
    bits = []
    for feature in features:
        threshold = float(train_df[feature].median())
        bits.append((df[feature].to_numpy(float) > threshold).astype(int))
    matrix = np.column_stack(bits)
    return pd.Series(["".join(str(int(v)) for v in row) for row in matrix], index=df.index)


def quantile_pair_code(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_a: str,
    bins_a: int,
    feature_b: str,
    bins_b: int,
) -> pd.Series:
    edges_a = quantile_bins(train_df[feature_a], bins_a)
    edges_b = quantile_bins(train_df[feature_b], bins_b)
    a = bin_code(df[feature_a], edges_a, bins_a)
    b = bin_code(df[feature_b], edges_b, bins_b)
    combined = a * bins_b + b
    return pd.Series([format(int(x), "05b") for x in combined], index=df.index)


def quantile_single_code(df: pd.DataFrame, train_df: pd.DataFrame, feature: str) -> pd.Series:
    edges = quantile_bins(train_df[feature], 32)
    b = bin_code(df[feature], edges, 32)
    return pd.Series([format(int(x), "05b") for x in b], index=df.index)


def append_rc_response(out: pd.DataFrame, rc: pd.DataFrame, code: pd.Series, prefix: str) -> None:
    code = code.astype(str).str.zfill(5)
    out[f"{prefix}_code"] = code
    channel_map = {
        "HRS 读取": "hrs_read",
        "HRS 峰值": "hrs_peak",
        "LRS 读取": "lrs_read",
        "LRS 峰值": "lrs_peak",
    }
    for source, short in channel_map.items():
        out[f"{prefix}_{short}"] = code.map(rc[source]).astype(float)
    out[f"{prefix}_hrs_delta"] = out[f"{prefix}_hrs_peak"] - out[f"{prefix}_hrs_read"]
    out[f"{prefix}_lrs_delta"] = out[f"{prefix}_lrs_peak"] - out[f"{prefix}_lrs_read"]
    out[f"{prefix}_read_gap"] = out[f"{prefix}_hrs_read"] - out[f"{prefix}_lrs_read"]
    out[f"{prefix}_peak_gap"] = out[f"{prefix}_hrs_peak"] - out[f"{prefix}_lrs_peak"]
    out[f"{prefix}_energy"] = (
        out[f"{prefix}_hrs_read"].abs()
        + out[f"{prefix}_hrs_peak"].abs()
        + out[f"{prefix}_lrs_read"].abs()
        + out[f"{prefix}_lrs_peak"].abs()
    )


def add_reservoir_features(df: pd.DataFrame, train_idx: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    rc = read_rc()
    out = df.copy()
    train_df = out.iloc[train_idx]

    encoders = {
        "rc_amp_rate": quantile_pair_code(out, train_df, "mean", 8, "rate_abs_mean", 4),
        "rc_amp_slope": quantile_pair_code(out, train_df, "mean", 8, "slope", 4),
        "rc_var_rate": quantile_pair_code(out, train_df, "std", 8, "rate_std", 4),
        "rc_range_peak": quantile_pair_code(out, train_df, "range", 8, "peak_count", 4),
        "rc_mean32": quantile_single_code(out, train_df, "mean"),
        "rc_rate32": quantile_single_code(out, train_df, "rate_abs_mean"),
        "rc_binary_shape": binary_feature_code(out, train_df, ["mean", "std", "range", "slope", "rate_abs_mean"]),
        "rc_binary_dynamics": binary_feature_code(out, train_df, ["rate_mean", "rate_std", "rate_abs_max", "rate_pos_ratio", "peak_count"]),
    }
    for prefix, code in encoders.items():
        append_rc_response(out, rc, code, prefix)

    rc_features = [
        c
        for c in out.columns
        if c.startswith("rc_") and not c.endswith("_code")
    ]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0), rc_features


def optimize_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (proba >= t).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def result_metrics(y_true: pd.Series, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def evaluate_split(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, validation_type: str) -> tuple[list[dict], pd.DataFrame]:
    data, rc_features = add_reservoir_features(df, train_idx)
    train = data.iloc[train_idx]
    val = data.iloc[val_idx]

    feature_sets = {
        "DFF_core": CORE_FEATURES,
        "RC_all_modes": rc_features,
        "DFF_core_plus_RC": CORE_FEATURES + rc_features,
        "DFF_relative": CORE_FEATURES + RELATIVE_FEATURES,
        "DFF_relative_plus_RC": CORE_FEATURES + RELATIVE_FEATURES + rc_features,
    }
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=1400,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=1400,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "rbf_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("svc", SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
            ]
        ),
    }

    results = []
    best_predictions = None
    best_score = -1.0
    for set_name, features in feature_sets.items():
        for model_name, model in models.items():
            model.fit(train[features], train["pseudo_stimulus"])
            train_proba = model.predict_proba(train[features])[:, 1]
            threshold = optimize_threshold(train["pseudo_stimulus"], train_proba)
            train_pred = (train_proba >= threshold).astype(int)
            val_proba = model.predict_proba(val[features])[:, 1]
            val_pred = (val_proba >= threshold).astype(int)
            train_metrics = result_metrics(train["pseudo_stimulus"], train_pred)
            val_metrics = result_metrics(val["pseudo_stimulus"], val_pred)
            row = {
                "validation_type": validation_type,
                "feature_set": set_name,
                "model": model_name,
                "threshold_from_train": threshold,
                "total_windows": int(len(data)),
                "train_windows": int(len(train)),
                "validation_windows": int(len(val)),
                "total_groups": int(data["group_id"].nunique()),
                "train_groups": int(train["group_id"].nunique()),
                "validation_groups": int(val["group_id"].nunique()),
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_confusion_matrix": train_metrics["confusion_matrix"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "validation_precision": val_metrics["precision"],
                "validation_recall": val_metrics["recall"],
                "validation_confusion_matrix": val_metrics["confusion_matrix"],
            }
            results.append(row)
            rank_score = val_metrics["accuracy"] + 0.2 * val_metrics["balanced_accuracy"]
            if rank_score > best_score:
                best_score = rank_score
                best_predictions = val[
                    ["source", "pair_id", "sample_id", "condition", "window", "stimulus_score", "pseudo_stimulus"]
                ].copy()
                best_predictions["predicted_probability"] = val_proba
                best_predictions["predicted_stimulus"] = val_pred
                best_predictions["correct"] = best_predictions["pseudo_stimulus"].eq(best_predictions["predicted_stimulus"])
                best_predictions["validation_type"] = validation_type
                best_predictions["feature_set"] = set_name
                best_predictions["model"] = model_name

    assert best_predictions is not None
    return results, best_predictions


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan).fillna(0)

    random_train, random_val = train_test_split(
        np.arange(len(df)),
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["pseudo_stimulus"],
    )
    group_split = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    group_train, group_val = next(group_split.split(df, df["pseudo_stimulus"], groups=df["group_id"]))

    all_results = []
    best_predictions = []
    for validation_type, train_idx, val_idx in [
        ("window_random_70_30", random_train, random_val),
        ("sample_grouped_70_30", group_train, group_val),
    ]:
        results, predictions = evaluate_split(df, train_idx, val_idx, validation_type)
        all_results.extend(results)
        best_predictions.append(predictions)

    summary = pd.DataFrame(all_results).sort_values(
        ["validation_accuracy", "validation_balanced_accuracy"], ascending=False
    )
    summary.to_csv(OUT_DIR / "rc_reservoir_validation_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(best_predictions, ignore_index=True).to_csv(
        OUT_DIR / "best_rc_reservoir_predictions.csv", index=False, encoding="utf-8-sig"
    )
    with open(OUT_DIR / "rc_reservoir_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("RC reservoir validation completed.")
    print(f"Results: {OUT_DIR}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
