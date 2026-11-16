from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import optimize_train_only_rc_gain as opt
from scan_sham_sni_window_lengths import CORE_FEATURES


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "train_only_rc_gain_optimization"


def parse_spec(row):
    max_features = "sqrt" if str(row.max_features) == "sqrt" else float(row.max_features)
    return int(row.min_samples_leaf), max_features


def main():
    data = pd.read_csv(opt.DATA).replace([np.inf, -np.inf], np.nan).fillna(0)
    train_idx, validation_idx = train_test_split(
        np.arange(len(data)), test_size=0.30, random_state=opt.SEED, stratify=data.target,
    )
    train = data.iloc[train_idx].reset_index(drop=True)
    validation = data.iloc[validation_idx].reset_index(drop=True)
    y_train = train.target.to_numpy(int)
    y_validation = validation.target.to_numpy(int)
    groups = opt.rc_feature_groups(opt.rc_columns(data))
    search = pd.read_csv(OUT / "train_only_paired_search.csv")
    dff_probability = {}
    rows = []
    for _, candidate in search.iterrows():
        spec = parse_spec(candidate)
        if spec not in dff_probability:
            fitted = opt.model(spec, trees=700).fit(train[CORE_FEATURES], y_train)
            dff_probability[spec] = fitted.predict_proba(validation[CORE_FEATURES])[:, 1]
        p_dff = dff_probability[spec]
        rc = groups[str(candidate.rc_group)]
        fused = opt.model(spec, trees=700).fit(train[CORE_FEATURES + rc], y_train)
        p_fusion = fused.predict_proba(validation[CORE_FEATURES + rc])[:, 1]
        pred_dff = p_dff >= float(candidate.dff_threshold)
        pred_fusion = p_fusion >= float(candidate.fusion_threshold)
        dff_acc = accuracy_score(y_validation, pred_dff)
        fusion_acc = accuracy_score(y_validation, pred_fusion)
        rows.append({
            "min_samples_leaf": spec[0], "max_features": str(spec[1]), "rc_group": candidate.rc_group,
            "n_rc_features": len(rc), "train_oof_gain": candidate.oof_gain,
            "dff_validation_accuracy": dff_acc,
            "fusion_validation_accuracy": fusion_acc,
            "validation_gain": fusion_acc - dff_acc,
            "fusion_validation_balanced_accuracy": balanced_accuracy_score(y_validation, pred_fusion),
            "fusion_validation_roc_auc": roc_auc_score(y_validation, p_fusion),
            "meets_accuracy_and_gain": bool(fusion_acc >= 0.90 and fusion_acc > dff_acc),
        })
    audit = pd.DataFrame(rows).sort_values(
        ["meets_accuracy_and_gain", "fusion_validation_accuracy", "validation_gain", "train_oof_gain"],
        ascending=False,
    )
    audit.to_csv(OUT / "all_config_sensitivity_audit.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "all_config_sensitivity_audit.xlsx", engine="openpyxl") as writer:
        audit.to_excel(writer, sheet_name="all_configs", index=False)
        audit[audit.meets_accuracy_and_gain].to_excel(writer, sheet_name="meets_both_conditions", index=False)
    print("configs", len(audit), "meet both", int(audit.meets_accuracy_and_gain.sum()))
    print(audit.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
