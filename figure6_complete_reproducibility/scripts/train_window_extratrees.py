"""Window-level ExtraTrees training, threshold selection, and 81-cell sensitivity audit."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from feature_pipeline import DFF27, RC9

SEED = 20260706
GROUPS = ["all_rc", "amp_rate", "amp_slope", "var_rate", "range_peak", "mean32", "rate32", "binary_shape", "binary_dynamics"]


def rc_columns(table: pd.DataFrame, group: str) -> list[str]:
    all_rc = [c for c in table if c.startswith("rc_") and not c.endswith("_code")]
    return all_rc if group == "all_rc" else [c for c in all_rc if c.startswith(f"rc_{group}_")]


def classifier(leaf=1, max_features="sqrt") -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=1200, min_samples_leaf=leaf, max_features=max_features,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)


def threshold_from_training(x: pd.DataFrame, y: pd.Series) -> float:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(classifier(), x, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    candidates = []
    for t in np.arange(.20, .801, .01):
        pred = p >= t
        candidates.append((balanced_accuracy_score(y, pred), accuracy_score(y, pred), -abs(t-.5), t))
    return float(max(candidates)[-1])


def metric(name, y, p, threshold):
    pred = p >= threshold; cm = confusion_matrix(y, pred, labels=[0, 1])
    return {"feature_set": name, "threshold": threshold, "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred), "roc_auc": roc_auc_score(y, p),
            "tn": cm[0,0], "fp": cm[0,1], "fn": cm[1,0], "tp": cm[1,1]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--features", required=True); ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(); data = pd.read_csv(args.features); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train = data[data.partition == "train"].copy(); test = data[data.partition == "validation"].copy()
    y_train, y_test = train.target, test.target
    all_rc = rc_columns(data, "all_rc")
    feature_sets = {"DFF_only": DFF27, "RC_only": all_rc, "DFF_plus_RC": DFF27 + all_rc,
                    "DFF_plus_selected_9RC": DFF27 + RC9}
    rows=[]; predictions=test[["event_id","target","window"]].copy()
    for name, cols in feature_sets.items():
        t = threshold_from_training(train[cols], y_train)
        fitted = classifier().fit(train[cols], y_train); p = fitted.predict_proba(test[cols])[:,1]
        rows.append(metric(name,y_test,p,t)); predictions[f"probability_{name}"]=p; predictions[f"prediction_{name}"]=(p>=t).astype(int)
    pd.DataFrame(rows).to_csv(out/"window_level_metrics.csv",index=False); predictions.to_csv(out/"window_level_predictions.csv",index=False)

    # Nine RC groups crossed with 3 leaf sizes and 3 max-feature settings = 81 configurations.
    sensitivity=[]
    for group in GROUPS:
        rc=rc_columns(data,group)
        for leaf in [1,2,4]:
            for mf in ["sqrt",.5,1.0]:
                fitted=classifier(leaf,mf).fit(train[DFF27+rc],y_train); p=fitted.predict_proba(test[DFF27+rc])[:,1]
                r=metric(group,y_test,p,.5); r.update({"rc_group":group,"n_rc_features":len(rc),"min_samples_leaf":leaf,"max_features":mf})
                sensitivity.append(r)
    pd.DataFrame(sensitivity).to_csv(out/"figs12_81_recomputed.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": main()
