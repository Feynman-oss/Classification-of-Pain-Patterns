from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import optimize_train_only_rc_gain as opt
from scan_sham_sni_window_lengths import CORE_FEATURES


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "train_only_rc_gain_optimization"


def wilson(k, n, z=1.959963984540054):
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return centre-half, centre+half


def evaluate(name, y, probability, threshold, status):
    pred = probability >= threshold
    cm = confusion_matrix(y, pred, labels=[0, 1])
    lo, hi = wilson(int(pred.eq(y).sum()) if isinstance(pred, pd.Series) else int((pred == y).sum()), len(y))
    return {
        "method": name, "status": status, "threshold": threshold,
        "accuracy": accuracy_score(y, pred), "accuracy_ci95_low": lo, "accuracy_ci95_high": hi,
        "balanced_accuracy": balanced_accuracy_score(y, pred), "roc_auc": roc_auc_score(y, probability),
        "f1_sni": f1_score(y, pred), "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }, pred.astype(int)


def main():
    data = pd.read_csv(opt.DATA).replace([np.inf, -np.inf], np.nan).fillna(0)
    train_idx, validation_idx = train_test_split(np.arange(len(data)), test_size=0.30, random_state=opt.SEED, stratify=data.target)
    train = data.iloc[train_idx].reset_index(drop=True)
    validation = data.iloc[validation_idx].reset_index(drop=True)
    y_train = train.target.to_numpy(int)
    y_val = validation.target.to_numpy(int)
    groups = opt.rc_feature_groups(opt.rc_columns(data))
    rc_all = groups["all_rc"]
    rc_amp_slope = groups["amp_slope"]

    search = pd.read_csv(OUT / "train_only_paired_search.csv")
    chosen = search[(search.min_samples_leaf == 4) & (search.max_features.astype(str) == "0.5") & (search.rc_group == "amp_slope")].iloc[0]

    historical_dff = ExtraTreesClassifier(n_estimators=1200, class_weight="balanced", random_state=opt.SEED, n_jobs=-1).fit(train[CORE_FEATURES], y_train)
    historical_fusion = ExtraTreesClassifier(n_estimators=1200, class_weight="balanced", random_state=opt.SEED, n_jobs=-1).fit(train[CORE_FEATURES + rc_all], y_train)
    optimized_dff = opt.model((4, 0.5), trees=700).fit(train[CORE_FEATURES], y_train)
    optimized_fusion = opt.model((4, 0.5), trees=700).fit(train[CORE_FEATURES + rc_amp_slope], y_train)

    models = [
        ("Prespecified DFF", historical_dff, CORE_FEATURES, 0.5, "prespecified reproducibility control"),
        ("Prespecified DFF + all RC", historical_fusion, CORE_FEATURES + rc_all, 0.5, "prespecified reproducibility control"),
        ("Exploratory optimized DFF", optimized_dff, CORE_FEATURES, float(chosen.dff_threshold), "exploratory after configuration audit"),
        ("Exploratory optimized DFF + amp_slope RC", optimized_fusion, CORE_FEATURES + rc_amp_slope, float(chosen.fusion_threshold), "exploratory after configuration audit"),
    ]
    rows, predictions = [], validation[["sample_id", "source", "model_type", "target", "window"]].copy()
    pred_cache = {}
    for name, fitted, features, threshold, status in models:
        probability = fitted.predict_proba(validation[features])[:, 1]
        row, pred = evaluate(name, y_val, probability, threshold, status)
        rows.append(row)
        key = name.lower().replace(" ", "_").replace("+", "plus")
        predictions[f"p_{key}"] = probability
        predictions[f"pred_{key}"] = pred
        predictions[f"correct_{key}"] = pred == y_val
        pred_cache[name] = pred

    metrics = pd.DataFrame(rows)
    comparisons = []
    for label, before, after in [
        ("prespecified", "Prespecified DFF", "Prespecified DFF + all RC"),
        ("exploratory_optimized", "Exploratory optimized DFF", "Exploratory optimized DFF + amp_slope RC"),
    ]:
        a, b = pred_cache[before], pred_cache[after]
        gain = accuracy_score(y_val, b) - accuracy_score(y_val, a)
        improved = int(((a != y_val) & (b == y_val)).sum())
        worsened = int(((a == y_val) & (b != y_val)).sum())
        discordant = improved + worsened
        p = 1.0 if discordant == 0 else float(binomtest(min(improved, worsened), discordant, 0.5).pvalue)
        comparisons.append({
            "comparison": label, "dff_accuracy": accuracy_score(y_val, a), "dff_rc_accuracy": accuracy_score(y_val, b),
            "gain_percentage_points": 100*gain, "rc_fixed_errors": improved, "rc_introduced_errors": worsened,
            "mcnemar_exact_p": p,
        })
    comparisons = pd.DataFrame(comparisons)

    metrics.to_csv(OUT / "final_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(OUT / "final_rc_gain_comparisons.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "final_candidate_predictions.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "final_rc_gain_report.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        comparisons.to_excel(writer, sheet_name="rc_gain", index=False)
        predictions.to_excel(writer, sheet_name="predictions", index=False)
    joblib.dump({
        "historical_dff": historical_dff, "historical_fusion": historical_fusion,
        "optimized_dff": optimized_dff, "optimized_fusion": optimized_fusion,
        "dff_features": CORE_FEATURES, "all_rc_features": rc_all, "amp_slope_rc_features": rc_amp_slope,
        "optimized_thresholds": {"dff": float(chosen.dff_threshold), "fusion": float(chosen.fusion_threshold)},
        "fixed_split_random_state": opt.SEED,
    }, OUT / "final_rc_gain_models.joblib")
    report = {"metrics": metrics.to_dict(orient="records"), "comparisons": comparisons.to_dict(orient="records"), "validation_selection_warning": "The optimized candidate was chosen after a configuration sensitivity audit and requires independent confirmation."}
    (OUT / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
