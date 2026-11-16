from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RC_PATH = ROOT / "RC.xlsx"
OUT_DIR = ROOT / "sham_sni_rc_classification"
RANDOM_STATE = 20260711
WINDOW_SECONDS = 1.0

RC_COLUMNS = {
    "HRS 读取": "hrs_read",
    "HRS 峰值": "hrs_peak",
    "LRS 读取": "lrs_read",
    "LRS 峰值": "lrs_peak",
}
ENCODING_FEATURES = ["mean", "std", "range", "slope", "rate_abs_mean"]
SUMMARY_STATS = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q25",
    "q75",
    "iqr",
    "first",
    "last",
    "last_minus_first",
    "trend",
    "diff_abs_mean",
    "diff_std",
]


@dataclass
class Trace:
    recording_id: str
    biological_id: str
    source: str
    model_type: str
    target: int
    condition: str
    values: np.ndarray
    timestamps_ns: np.ndarray


def read_rc_table() -> pd.DataFrame:
    rc = pd.read_excel(RC_PATH, sheet_name="Sheet1", dtype={"编码": str})
    rc = rc.drop(columns=[c for c in rc.columns if str(c).startswith("Unnamed")], errors="ignore")
    rc = rc.dropna(how="all").copy()
    required = ["编码", *RC_COLUMNS]
    missing = [c for c in required if c not in rc.columns]
    if missing:
        raise ValueError(f"RC.xlsx 缺少列: {missing}")
    rc["编码"] = rc["编码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    for col in RC_COLUMNS:
        rc[col] = pd.to_numeric(rc[col], errors="raise")
    rc = rc.set_index("编码")
    expected = {format(i, "05b") for i in range(32)}
    if set(rc.index) != expected:
        raise ValueError("RC.xlsx 必须包含 00000–11111 的全部 32 个五位编码且不能重复")
    return rc


def _clean_trace(values, timestamps) -> tuple[np.ndarray, np.ndarray] | None:
    frame = pd.DataFrame({"value": values, "timestamp": timestamps})
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().sort_values("timestamp")
    frame = frame.drop_duplicates("timestamp")
    if len(frame) < 3:
        return None
    return frame["value"].to_numpy(float), frame["timestamp"].to_numpy(float)


def _overall_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(path, sheet_name="Overall DFF")
    if "Mean" not in df or "Timestamp(ns)" not in df:
        raise ValueError(f"{path} 的 Overall DFF 缺少 Mean 或 Timestamp(ns)")
    cleaned = _clean_trace(df["Mean"], df["Timestamp(ns)"])
    if cleaned is None:
        raise ValueError(f"{path} 的有效 DFF 数据不足")
    return cleaned


def _detail_traces(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    df = pd.read_excel(path, sheet_name="Detail DFF")
    columns = list(df.columns)
    traces: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, col in enumerate(columns):
        match = re.match(r"Trail\s+(\d+)-", str(col))
        if not match:
            continue
        timestamp_col = next(
            (c for c in columns[idx + 1 : min(idx + 9, len(columns))] if str(c).startswith("Timestamp(ns)")),
            None,
        )
        if timestamp_col is None:
            continue
        cleaned = _clean_trace(df[col], df[timestamp_col])
        if cleaned is not None:
            traces[int(match.group(1))] = cleaned
    return traces


def load_traces() -> tuple[list[Trace], list[dict]]:
    traces: list[Trace] = []
    audit: list[dict] = []

    # Folders 1 and 2: each control-pain file pair is one biological animal.
    for source in ["1", "2"]:
        pain_files = sorted((ROOT / source).glob("*pain.xlsx"))
        for pain_path in pain_files:
            base = pain_path.stem.removesuffix("pain")
            control_path = pain_path.with_name(base + "control.xlsx")
            if not control_path.exists():
                audit.append({"status": "skipped", "reason": "missing control", "file": str(pain_path.relative_to(ROOT))})
                continue
            animal_match = re.match(r"(animal\d+)-", base, flags=re.I)
            if not animal_match:
                audit.append({"status": "skipped", "reason": "cannot parse animal", "file": str(pain_path.relative_to(ROOT))})
                continue
            recording_id = f"{source}/{base.rstrip('-')}"
            biological_id = recording_id
            for condition, path in [("control", control_path), ("pain", pain_path)]:
                values, times = _overall_trace(path)
                traces.append(Trace(recording_id, biological_id, source, "sham", 0, condition, values, times))
            audit.append({"status": "included", "source": source, "model_type": "sham", "biological_id": biological_id, "recording_id": recording_id})

    # Folder 3: per user definition, every Detail DFF trail is one biological animal.
    for pain_path in sorted((ROOT / "3").glob("*pain.xlsx")):
        stem = pain_path.stem.lower()
        if "-sni-" in stem:
            model_type, target = "sni", 1
        elif "-sham-" in stem:
            model_type, target = "sham", 0
        else:
            continue
        base = pain_path.stem.removesuffix("pain")
        control_path = pain_path.with_name(base + "control.xlsx")
        if not control_path.exists():
            audit.append({"status": "skipped", "reason": "missing control", "file": str(pain_path.relative_to(ROOT))})
            continue
        animal_number = stem.split("-")[0]
        pain_traces = _detail_traces(pain_path)
        control_traces = _detail_traces(control_path)
        common = sorted(set(pain_traces) & set(control_traces))
        for trail in common:
            recording_id = f"3/{model_type}-animal{animal_number}/trail{trail}"
            biological_id = recording_id
            for condition, data in [("control", control_traces[trail]), ("pain", pain_traces[trail])]:
                traces.append(Trace(recording_id, biological_id, "3", model_type, target, condition, data[0], data[1]))
            audit.append({"status": "included", "source": "3", "model_type": model_type, "biological_id": biological_id, "recording_id": recording_id})
        for trail in sorted(set(pain_traces) ^ set(control_traces)):
            audit.append({"status": "skipped", "reason": "unpaired trail", "file": base, "trail": trail})

    return traces, audit


def extract_windows(trace: Trace) -> pd.DataFrame:
    t = (trace.timestamps_ns - trace.timestamps_ns[0]) / 1e9
    window_index = np.floor(t / WINDOW_SECONDS).astype(int)
    rows = []
    for window in np.unique(window_index):
        mask = window_index == window
        x = trace.values[mask]
        tw = t[mask]
        if len(x) < 3:
            continue
        dt = np.diff(tw)
        dx = np.diff(x)
        valid = dt > 0
        rate = dx[valid] / dt[valid] if valid.any() else np.array([0.0])
        duration = max(float(tw[-1] - tw[0]), 1e-9)
        rows.append(
            {
                "recording_id": trace.recording_id,
                "biological_id": trace.biological_id,
                "source": trace.source,
                "model_type": trace.model_type,
                "target": trace.target,
                "condition": trace.condition,
                "window": int(window),
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)),
                "range": float(np.max(x) - np.min(x)),
                "slope": float((x[-1] - x[0]) / duration),
                "rate_abs_mean": float(np.mean(np.abs(rate))),
            }
        )
    return pd.DataFrame(rows)


def fit_encoding_thresholds(windows: pd.DataFrame, train_groups: set[str]) -> dict[str, float]:
    train = windows[windows["biological_id"].isin(train_groups)]
    return {feature: float(train[feature].median()) for feature in ENCODING_FEATURES}


def encode_windows(windows: pd.DataFrame, rc: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = windows.copy()
    bit_columns = []
    for feature in ENCODING_FEATURES:
        bit_col = f"bit_{feature}"
        out[bit_col] = (out[feature] > thresholds[feature]).astype(int)
        bit_columns.append(bit_col)
    out["rc_code"] = out[bit_columns].astype(str).agg("".join, axis=1)
    for rc_col, short in RC_COLUMNS.items():
        out[short] = out["rc_code"].map(rc[rc_col]).astype(float)
    return out


def summarize_sequence(values: pd.Series) -> dict[str, float]:
    x = values.to_numpy(float)
    if len(x) == 0:
        return {name: 0.0 for name in SUMMARY_STATS}
    q25, q75 = np.quantile(x, [0.25, 0.75])
    diff = np.diff(x)
    if len(x) > 1:
        trend = float(np.polyfit(np.arange(len(x), dtype=float), x, 1)[0])
    else:
        trend = 0.0
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "first": float(x[0]),
        "last": float(x[-1]),
        "last_minus_first": float(x[-1] - x[0]),
        "trend": trend,
        "diff_abs_mean": float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        "diff_std": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
    }


def build_recording_features(encoded: pd.DataFrame) -> pd.DataFrame:
    metadata = ["recording_id", "biological_id", "source", "model_type", "target", "condition"]
    rows = []
    for keys, group in encoded.sort_values("window").groupby(metadata, sort=False):
        row = dict(zip(metadata, keys))
        for channel in RC_COLUMNS.values():
            for stat, value in summarize_sequence(group[channel]).items():
                row[f"{channel}__{stat}"] = value
        rows.append(row)
    condition_table = pd.DataFrame(rows)
    index_cols = ["recording_id", "biological_id", "source", "model_type", "target"]
    feature_cols = [c for c in condition_table.columns if c not in index_cols + ["condition"]]
    control = condition_table[condition_table["condition"] == "control"].set_index(index_cols)
    pain = condition_table[condition_table["condition"] == "pain"].set_index(index_cols)
    common = control.index.intersection(pain.index)
    result = pd.DataFrame(index=common).reset_index()
    derived: dict[str, np.ndarray] = {}
    for col in feature_cols:
        c = control.loc[common, col].to_numpy(float)
        p = pain.loc[common, col].to_numpy(float)
        derived[f"control__{col}"] = c
        derived[f"pain__{col}"] = p
        derived[f"delta__{col}"] = p - c
    result = pd.concat([result, pd.DataFrame(derived)], axis=1)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def candidate_feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "rc_delta": [c for c in table if c.startswith("delta__")],
        "rc_pain_plus_delta": [c for c in table if c.startswith(("pain__", "delta__"))],
        "rc_control_pain_delta": [c for c in table if c.startswith(("control__", "pain__", "delta__"))],
    }


def make_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _animal_balanced_accuracy(table: pd.DataFrame, probabilities: np.ndarray) -> float:
    animal = table[["biological_id", "target"]].copy()
    animal["probability_sni"] = probabilities
    animal = animal.groupby("biological_id", as_index=False).agg(target=("target", "first"), probability_sni=("probability_sni", "mean"))
    return float(balanced_accuracy_score(animal["target"], (animal["probability_sni"] >= 0.5).astype(int)))


def tune_candidate(train_table: pd.DataFrame) -> tuple[str, float, pd.DataFrame]:
    group_counts = train_table.groupby("target")["biological_id"].nunique()
    n_splits = int(min(3, group_counts.min()))
    if n_splits < 2:
        raise ValueError("训练集中每一类至少需要 2 个生物学个体")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    candidates = []
    for set_name, features in candidate_feature_sets(train_table).items():
        for c_value in [0.01, 0.1, 1.0, 10.0]:
            fold_scores = []
            for inner_train, inner_val in splitter.split(train_table, train_table["target"], train_table["biological_id"]):
                model = make_model(c_value)
                model.fit(train_table.iloc[inner_train][features], train_table.iloc[inner_train]["target"])
                proba = model.predict_proba(train_table.iloc[inner_val][features])[:, 1]
                fold_scores.append(_animal_balanced_accuracy(train_table.iloc[inner_val], proba))
            candidates.append({"feature_set": set_name, "C": c_value, "inner_animal_balanced_accuracy": float(np.mean(fold_scores))})
    ranking = pd.DataFrame(candidates).sort_values(
        ["inner_animal_balanced_accuracy", "C", "feature_set"], ascending=[False, True, True]
    )
    best = ranking.iloc[0]
    return str(best["feature_set"]), float(best["C"]), ranking


def metric_dict(y_true, probability) -> dict:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pred = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_sham": int((y == 0).sum()),
        "n_sni": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity_sni": float(recall_score(y, pred, zero_division=0)),
        "specificity_sham": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "precision_sni": float(precision_score(y, pred, zero_division=0)),
        "f1_sni": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def stratified_bootstrap_ci(animal_predictions: pd.DataFrame, iterations: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    sham = animal_predictions[animal_predictions["target"] == 0]
    sni = animal_predictions[animal_predictions["target"] == 1]
    metrics = {"balanced_accuracy": [], "sensitivity_sni": [], "specificity_sham": [], "roc_auc": []}
    for _ in range(iterations):
        sample = pd.concat(
            [
                sham.iloc[rng.integers(0, len(sham), len(sham))],
                sni.iloc[rng.integers(0, len(sni), len(sni))],
            ],
            ignore_index=True,
        )
        row = metric_dict(sample["target"], sample["probability_sni"])
        for key in metrics:
            metrics[key].append(row[key])
    return pd.DataFrame(
        [
            {"metric": key, "ci95_low": float(np.quantile(values, 0.025)), "ci95_high": float(np.quantile(values, 0.975))}
            for key, values in metrics.items()
        ]
    )


def run_nested_logo(windows: pd.DataFrame, rc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = sorted(windows["biological_id"].unique())
    prediction_rows = []
    tuning_rows = []
    for held_out in groups:
        train_groups = set(groups) - {held_out}
        thresholds = fit_encoding_thresholds(windows, train_groups)
        table = build_recording_features(encode_windows(windows, rc, thresholds))
        train_table = table[table["biological_id"].isin(train_groups)].copy()
        test_table = table[table["biological_id"] == held_out].copy()
        feature_set, c_value, ranking = tune_candidate(train_table)
        features = candidate_feature_sets(table)[feature_set]
        model = make_model(c_value)
        model.fit(train_table[features], train_table["target"])
        probabilities = model.predict_proba(test_table[features])[:, 1]
        for (_, row), probability in zip(test_table.iterrows(), probabilities):
            prediction_rows.append(
                {
                    "recording_id": row["recording_id"],
                    "biological_id": row["biological_id"],
                    "source": row["source"],
                    "model_type": row["model_type"],
                    "target": int(row["target"]),
                    "probability_sni": float(probability),
                    "predicted_model": "sni" if probability >= 0.5 else "sham",
                    "correct": bool((probability >= 0.5) == bool(row["target"])),
                    "outer_held_out": held_out,
                    "selected_feature_set": feature_set,
                    "selected_C": c_value,
                }
            )
        best_inner = ranking.iloc[0].to_dict()
        tuning_rows.append({"outer_held_out": held_out, **thresholds, **best_inner})
    recording_predictions = pd.DataFrame(prediction_rows)
    animal_predictions = (
        recording_predictions.groupby(["biological_id", "source", "model_type", "target"], as_index=False)
        .agg(probability_sni=("probability_sni", "mean"), n_recordings=("recording_id", "count"))
    )
    animal_predictions["predicted_model"] = np.where(animal_predictions["probability_sni"] >= 0.5, "sni", "sham")
    animal_predictions["correct"] = animal_predictions["model_type"].eq(animal_predictions["predicted_model"])
    return recording_predictions, animal_predictions, pd.DataFrame(tuning_rows)


def fit_final_model(windows: pd.DataFrame, rc: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    all_groups = set(windows["biological_id"].unique())
    thresholds = fit_encoding_thresholds(windows, all_groups)
    table = build_recording_features(encode_windows(windows, rc, thresholds))
    feature_set, c_value, ranking = tune_candidate(table)
    features = candidate_feature_sets(table)[feature_set]
    model = make_model(c_value)
    model.fit(table[features], table["target"])
    artifact = {
        "purpose": "research-only sham vs SNI classification from paired DFF traces using measured RC.xlsx responses",
        "window_seconds": WINDOW_SECONDS,
        "encoding_features_in_bit_order": ENCODING_FEATURES,
        "encoding_thresholds": thresholds,
        "rc_columns": RC_COLUMNS,
        "summary_stats": SUMMARY_STATS,
        "selected_feature_set": feature_set,
        "feature_columns": features,
        "C": c_value,
        "classifier": model,
        "random_state": RANDOM_STATE,
    }
    return artifact, ranking


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rc = read_rc_table()
    traces, audit = load_traces()
    windows = pd.concat([extract_windows(trace) for trace in traces], ignore_index=True)
    windows = windows.replace([np.inf, -np.inf], np.nan).dropna(subset=ENCODING_FEATURES)

    recording_predictions, animal_predictions, tuning = run_nested_logo(windows, rc)
    recording_metrics = metric_dict(recording_predictions["target"], recording_predictions["probability_sni"])
    animal_metrics = metric_dict(animal_predictions["target"], animal_predictions["probability_sni"])
    ci = stratified_bootstrap_ci(animal_predictions)

    artifact, final_ranking = fit_final_model(windows, rc)
    joblib.dump(artifact, OUT_DIR / "final_rc_classifier.joblib")

    audit_df = pd.DataFrame(audit)
    data_audit = (
        recording_predictions.groupby(["source", "model_type"], as_index=False)
        .agg(biological_animals=("biological_id", "nunique"), paired_recordings=("recording_id", "nunique"))
    )
    summary = pd.DataFrame(
        [
            {"evaluation_level": "biological_animal (primary)", **animal_metrics},
            {"evaluation_level": "paired_recording (secondary; clustered within animal)", **recording_metrics},
        ]
    )
    limitations = pd.DataFrame(
        {
            "item": [
                "Validation design",
                "Biological sample size",
                "Batch/source confounding",
                "Repeated measurements",
                "Intended use",
            ],
            "detail": [
                "Nested leave-one-biological-animal-out; RC thresholds are fitted only on each outer training fold.",
                f"{animal_metrics['n']} animals total: {animal_metrics['n_sham']} sham and {animal_metrics['n_sni']} SNI.",
                "All SNI animals are from folder 3. Folder 3 sham is retained as a same-source negative control, but source and model remain partly confounded.",
                "Trails/repeats are never split across folds; primary metrics average probabilities within each biological animal.",
                "Research analysis only; external animals/batches are required before biological or diagnostic claims.",
            ],
        }
    )

    recording_predictions.to_csv(OUT_DIR / "recording_level_predictions.csv", index=False, encoding="utf-8-sig")
    animal_predictions.to_csv(OUT_DIR / "animal_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "classification_metrics.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(OUT_DIR / "animal_metric_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(OUT_DIR / "one_second_dff_windows.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_DIR / "sham_sni_rc_classification_report.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="metrics", index=False)
        ci.to_excel(writer, sheet_name="animal_metric_CI", index=False)
        animal_predictions.to_excel(writer, sheet_name="animal_predictions", index=False)
        recording_predictions.to_excel(writer, sheet_name="recording_predictions", index=False)
        data_audit.to_excel(writer, sheet_name="data_audit", index=False)
        tuning.to_excel(writer, sheet_name="outer_fold_settings", index=False)
        final_ranking.to_excel(writer, sheet_name="final_model_selection", index=False)
        audit_df.to_excel(writer, sheet_name="input_audit", index=False)
        limitations.to_excel(writer, sheet_name="limitations", index=False)

    report = {
        "method": {
            "classification_target": "sham=0, SNI=1",
            "input": "paired control and pain DFF traces",
            "reservoir": "measured 5-bit/4-channel responses from RC.xlsx",
            "encoding_bit_order": ENCODING_FEATURES,
            "validation": "nested leave-one-biological-animal-out",
            "decision_threshold": 0.5,
            "random_state": RANDOM_STATE,
        },
        "animal_metrics": animal_metrics,
        "recording_metrics": recording_metrics,
        "animal_metric_ci95": ci.to_dict(orient="records"),
        "final_model": {k: v for k, v in artifact.items() if k != "classifier"},
        "limitations": limitations.to_dict(orient="records"),
    }
    with open(OUT_DIR / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Completed sham vs SNI RC classification")
    print(data_audit.to_string(index=False))
    print(summary.to_string(index=False))
    print("\nAnimal predictions:")
    print(animal_predictions.to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
