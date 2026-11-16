from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from scan_sham_sni_window_lengths import CORE_FEATURES


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "train_only_rc_gain_optimization"
DATA = ROOT / "sham_sni_window_length_scan" / "sham_sni_10s_windows.csv"
SEED = 20260706
RESPONSE_SUFFIXES = [
    "hrs_read", "hrs_peak", "lrs_read", "lrs_peak", "hrs_delta",
    "lrs_delta", "read_gap", "peak_gap", "energy",
]


def rc_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data if c.startswith("rc_") and not c.endswith("_code") and pd.api.types.is_numeric_dtype(data[c])]


def rc_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"all_rc": columns}
    for col in columns:
        mode = col
        for suffix in RESPONSE_SUFFIXES:
            token = "_" + suffix
            if col.endswith(token):
                mode = col[: -len(token)]
                break
        groups.setdefault(mode.removeprefix("rc_"), []).append(col)
    return groups


def specs():
    return [(leaf, mf) for leaf in [1, 2, 4] for mf in ["sqrt", 0.5, 1.0]]


def model(spec, trees=200, seed=SEED):
    leaf, max_features = spec
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def choose_threshold(y, probability):
    rows = []
    for threshold in np.arange(0.20, 0.801, 0.01):
        prediction = probability >= threshold
        rows.append((accuracy_score(y, prediction), balanced_accuracy_score(y, prediction), -abs(threshold - 0.5), float(threshold)))
    return max(rows)[-1]


def metric_row(name, y, probability, threshold, selection_basis):
    prediction = (probability >= threshold).astype(int)
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    return {
        "method": name,
        "selection_basis": selection_basis,
        "threshold": threshold,
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "roc_auc": roc_auc_score(y, probability),
        "f1_sni": f1_score(y, prediction, zero_division=0),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def oof_probability(x, y, spec, cv, trees=200):
    probability = np.zeros(len(y), float)
    for fold, (fit, hold) in enumerate(cv.split(x, y), 1):
        fitted = model(spec, trees=trees, seed=SEED + fold).fit(x.iloc[fit], y[fit])
        probability[hold] = fitted.predict_proba(x.iloc[hold])[:, 1]
    return probability


def main():
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(DATA).replace([np.inf, -np.inf], np.nan).fillna(0)
    train_idx, validation_idx = train_test_split(
        np.arange(len(data)), test_size=0.30, random_state=SEED, stratify=data.target,
    )
    train = data.iloc[train_idx].reset_index(drop=True)
    validation = data.iloc[validation_idx].reset_index(drop=True)
    y_train = train.target.to_numpy(int)
    y_validation = validation.target.to_numpy(int)
    rc_all = rc_columns(data)
    groups = rc_feature_groups(rc_all)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    search_rows = []
    cache: dict[tuple, np.ndarray] = {}
    for spec in specs():
        dff_probability = oof_probability(train[CORE_FEATURES], y_train, spec, cv)
        dff_threshold = choose_threshold(y_train, dff_probability)
        dff_prediction = dff_probability >= dff_threshold
        dff_accuracy = accuracy_score(y_train, dff_prediction)
        dff_balanced = balanced_accuracy_score(y_train, dff_prediction)
        cache[(spec, "DFF")] = dff_probability
        for group_name, group_columns in groups.items():
            features = CORE_FEATURES + group_columns
            fused_probability = oof_probability(train[features], y_train, spec, cv)
            fused_threshold = choose_threshold(y_train, fused_probability)
            fused_prediction = fused_probability >= fused_threshold
            row = {
                "min_samples_leaf": spec[0], "max_features": str(spec[1]), "rc_group": group_name,
                "n_rc_features": len(group_columns),
                "dff_threshold": dff_threshold,
                "dff_oof_accuracy": dff_accuracy,
                "dff_oof_balanced_accuracy": dff_balanced,
                "fusion_threshold": fused_threshold,
                "fusion_oof_accuracy": accuracy_score(y_train, fused_prediction),
                "fusion_oof_balanced_accuracy": balanced_accuracy_score(y_train, fused_prediction),
                "fusion_oof_roc_auc": roc_auc_score(y_train, fused_probability),
            }
            row["oof_gain"] = row["fusion_oof_accuracy"] - row["dff_oof_accuracy"]
            search_rows.append(row)
            cache[(spec, group_name)] = fused_probability

    search = pd.DataFrame(search_rows)
    eligible = search[search.oof_gain > 0].copy()
    ranking_source = eligible if not eligible.empty else search
    selected = ranking_source.sort_values(
        ["fusion_oof_accuracy", "fusion_oof_balanced_accuracy", "oof_gain", "fusion_oof_roc_auc"],
        ascending=False,
    ).iloc[0]
    selected_spec = (int(selected.min_samples_leaf), "sqrt" if selected.max_features == "sqrt" else float(selected.max_features))
    selected_group = str(selected.rc_group)
    selected_rc = groups[selected_group]
    dff_oof = cache[(selected_spec, "DFF")]
    fusion_oof = cache[(selected_spec, selected_group)]
    rc_oof = oof_probability(train[selected_rc], y_train, selected_spec, cv)

    # Train-only selection of a soft blend and an uncertainty gate.
    blend_candidates = []
    for alpha in np.arange(0.0, 1.51, 0.05):
        probability = np.clip(dff_oof + alpha * (fusion_oof - dff_oof), 0, 1)
        threshold = choose_threshold(y_train, probability)
        prediction = probability >= threshold
        blend_candidates.append((accuracy_score(y_train, prediction), balanced_accuracy_score(y_train, prediction), alpha, threshold))
    blend_accuracy, blend_balanced, blend_alpha, blend_threshold = max(blend_candidates)

    gate_candidates = []
    for width in np.arange(0.05, 0.451, 0.025):
        probability = np.where(np.abs(dff_oof - 0.5) <= width, fusion_oof, dff_oof)
        threshold = choose_threshold(y_train, probability)
        prediction = probability >= threshold
        gate_candidates.append((accuracy_score(y_train, prediction), balanced_accuracy_score(y_train, prediction), -width, width, threshold))
    gate_accuracy, gate_balanced, _, gate_width, gate_threshold = max(gate_candidates)

    # Cross-fitted second-level model prevents fitting and evaluating the stack on the same meta rows.
    meta_x = np.column_stack([dff_oof, fusion_oof, rc_oof, fusion_oof - dff_oof])
    meta_oof = np.zeros(len(y_train), float)
    for fold, (fit, hold) in enumerate(cv.split(meta_x, y_train), 1):
        meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED + fold)
        meta.fit(meta_x[fit], y_train[fit])
        meta_oof[hold] = meta.predict_proba(meta_x[hold])[:, 1]
    meta_threshold = choose_threshold(y_train, meta_oof)
    meta_final = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED).fit(meta_x, y_train)

    # Final base models fitted only on the 167 training windows.
    dff_model = model(selected_spec, trees=1200).fit(train[CORE_FEATURES], y_train)
    fusion_model = model(selected_spec, trees=1200).fit(train[CORE_FEATURES + selected_rc], y_train)
    rc_model = model(selected_spec, trees=1200).fit(train[selected_rc], y_train)
    p_dff = dff_model.predict_proba(validation[CORE_FEATURES])[:, 1]
    p_fusion = fusion_model.predict_proba(validation[CORE_FEATURES + selected_rc])[:, 1]
    p_rc = rc_model.predict_proba(validation[selected_rc])[:, 1]
    p_blend = np.clip(p_dff + blend_alpha * (p_fusion - p_dff), 0, 1)
    p_gate = np.where(np.abs(p_dff - 0.5) <= gate_width, p_fusion, p_dff)
    p_stack = meta_final.predict_proba(np.column_stack([p_dff, p_fusion, p_rc, p_fusion - p_dff]))[:, 1]

    results = [
        metric_row("DFF selected ExtraTrees", y_validation, p_dff, float(selected.dff_threshold), "train-only 5-fold OOF"),
        metric_row("DFF + RC selected ExtraTrees", y_validation, p_fusion, float(selected.fusion_threshold), "train-only 5-fold OOF"),
        metric_row("DFF + RC soft blend", y_validation, p_blend, blend_threshold, "train-only 5-fold OOF"),
        metric_row("DFF + RC uncertainty gate", y_validation, p_gate, gate_threshold, "train-only 5-fold OOF"),
        metric_row("DFF + RC stacked correction", y_validation, p_stack, meta_threshold, "cross-fitted train-only meta OOF"),
    ]

    # Prespecified historical ExtraTrees comparison retained as an exact reproducibility control.
    historical_dff = ExtraTreesClassifier(n_estimators=1200, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(train[CORE_FEATURES], y_train)
    historical_fusion = ExtraTreesClassifier(n_estimators=1200, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(train[CORE_FEATURES + rc_all], y_train)
    p_hist_dff = historical_dff.predict_proba(validation[CORE_FEATURES])[:, 1]
    p_hist_fusion = historical_fusion.predict_proba(validation[CORE_FEATURES + rc_all])[:, 1]
    results.extend([
        metric_row("Historical DFF ExtraTrees", y_validation, p_hist_dff, 0.5, "prespecified 20260706 split"),
        metric_row("Historical DFF + all RC ExtraTrees", y_validation, p_hist_fusion, 0.5, "prespecified 20260706 split"),
    ])
    metrics = pd.DataFrame(results)

    predictions = validation[["sample_id", "source", "model_type", "target", "window"]].copy()
    probabilities = {
        "dff_selected": p_dff, "fusion_selected": p_fusion, "soft_blend": p_blend,
        "uncertainty_gate": p_gate, "stacked_correction": p_stack,
        "historical_dff": p_hist_dff, "historical_dff_all_rc": p_hist_fusion,
    }
    thresholds = {
        "dff_selected": float(selected.dff_threshold), "fusion_selected": float(selected.fusion_threshold),
        "soft_blend": float(blend_threshold), "uncertainty_gate": float(gate_threshold),
        "stacked_correction": float(meta_threshold), "historical_dff": 0.5, "historical_dff_all_rc": 0.5,
    }
    for name, probability in probabilities.items():
        predictions[f"p_{name}"] = probability
        predictions[f"pred_{name}"] = (probability >= thresholds[name]).astype(int)
        predictions[f"correct_{name}"] = predictions[f"pred_{name}"].eq(predictions.target)

    search.to_csv(OUT / "train_only_paired_search.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "fixed_validation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "fixed_validation_predictions.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "rc_gain_optimization_report.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="validation_metrics", index=False)
        predictions.to_excel(writer, sheet_name="validation_predictions", index=False)
        search.sort_values(["fusion_oof_accuracy", "oof_gain"], ascending=False).to_excel(writer, sheet_name="train_only_search", index=False)
    artifact = {
        "dff_model": dff_model, "fusion_model": fusion_model, "rc_model": rc_model, "meta_model": meta_final,
        "dff_features": CORE_FEATURES, "rc_features": selected_rc, "selected_rc_group": selected_group,
        "selected_spec": selected_spec, "thresholds": thresholds, "blend_alpha": float(blend_alpha), "gate_width": float(gate_width),
        "uses_validation_labels_for_selection": False, "validation_type": "fixed window-level random 70/30",
    }
    joblib.dump(artifact, OUT / "rc_gain_models.joblib")
    report = {
        "selected_spec": [selected_spec[0], selected_spec[1]], "selected_rc_group": selected_group,
        "selected_rc_features": len(selected_rc), "blend_alpha": float(blend_alpha), "gate_width": float(gate_width),
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("selected", selected_spec, selected_group, len(selected_rc))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
