"""Nested event-disjoint validation for DFF-only, RC-only and DFF+RC models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_pipeline import add_fold_rc_and_aggregate

SEED = 20260712
FEATURE_SETS = ["DFF_only", "RC_only", "DFF_plus_RC"]


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    scored = []
    for threshold in np.linspace(0.10, 0.90, 161):
        pred = probability >= threshold
        scored.append((balanced_accuracy_score(y, pred), accuracy_score(y, pred), -abs(threshold - .5), threshold))
    return float(max(scored)[-1])


def features(table: pd.DataFrame, name: str) -> list[str]:
    dff = [c for c in table if c.startswith(("dff_", "rate_", "peak_"))]
    rc = [c for c in table if c.startswith("rc_")]
    return dff if name == "DFF_only" else rc if name == "RC_only" else dff + rc


def model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=SEED)


def evaluate(name: str, windows: pd.DataFrame, response: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    events = windows[["event_id", "target"]].drop_duplicates().sort_values("event_id").reset_index(drop=True)
    y = events.target.to_numpy(int)
    probability = np.zeros(len(events)); prediction = np.zeros(len(events), int); fold_id = np.zeros(len(events), int)
    fold_rows = []
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (outer_train, outer_test) in enumerate(outer.split(events, y), 1):
        train_events = events.iloc[outer_train].reset_index(drop=True)
        train_y = train_events.target.to_numpy(int)
        inner_probability = np.zeros(len(train_events))
        inner = StratifiedKFold(4, shuffle=True, random_state=SEED + fold)
        for inner_train, inner_val in inner.split(train_events, train_y):
            ids = set(train_events.iloc[inner_train].event_id)
            table, _ = add_fold_rc_and_aggregate(windows, response, ids)
            table = train_events.merge(table, on=["event_id", "target"], validate="one_to_one")
            cols = features(table, name)
            fitted = model().fit(table.iloc[inner_train][cols], train_y[inner_train])
            inner_probability[inner_val] = fitted.predict_proba(table.iloc[inner_val][cols])[:, 1]
        threshold = choose_threshold(train_y, inner_probability)
        ids = set(events.iloc[outer_train].event_id)
        table, quantizer = add_fold_rc_and_aggregate(windows, response, ids)
        table = events.merge(table, on=["event_id", "target"], validate="one_to_one")
        cols = features(table, name)
        fitted = model().fit(table.iloc[outer_train][cols], y[outer_train])
        p = fitted.predict_proba(table.iloc[outer_test][cols])[:, 1]
        probability[outer_test] = p
        prediction[outer_test] = p >= threshold
        fold_id[outer_test] = fold
        fold_rows.append({"feature_set": name, "outer_fold": fold, "threshold": threshold, **quantizer, "n_test": len(outer_test)})
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    metric = {
        "feature_set": name, "n_events": len(events), "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction), "roc_auc": roc_auc_score(y, probability),
        "f1_sni": f1_score(y, prediction), "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    pred = events.copy(); pred["probability_sni"] = probability; pred["predicted_class"] = prediction
    pred["correct"] = prediction == y; pred["outer_fold"] = fold_id; pred["feature_set"] = name
    return metric, pred, pd.DataFrame(fold_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--response-table", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    windows = pd.read_csv(args.windows)
    response = pd.read_csv(args.response_table, dtype={"code": str})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics, predictions, settings = [], [], []
    for name in FEATURE_SETS:
        m, p, s = evaluate(name, windows, response); metrics.append(m); predictions.append(p); settings.append(s)
    pd.DataFrame(metrics).to_csv(out / "nested_metrics.csv", index=False)
    pd.concat(predictions).to_csv(out / "nested_oof_predictions.csv", index=False)
    pd.concat(settings).to_csv(out / "nested_fold_settings.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps({"seed": SEED, "outer_folds": 5, "inner_folds": 4,
        "threshold_grid": [0.10, 0.90, 0.005], "quantizer_fitted_within_training_fold": True}, indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
