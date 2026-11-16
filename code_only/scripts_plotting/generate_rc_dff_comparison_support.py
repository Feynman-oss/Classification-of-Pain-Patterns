from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

from run_rc_reservoir_validation import (
    CORE_FEATURES,
    DATA_PATH,
    RANDOM_STATE,
    RELATIVE_FEATURES,
    add_reservoir_features,
    optimize_threshold,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "supporting_dff_rc_comparison"
FIG_DIR = OUT_DIR / "figures"
RAW_DIR = OUT_DIR / "source_data"

COL = {
    "DFF-only": "#4C78A8",
    "RC-only": "#845EC2",
    "DFF + RC": "#D55E00",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 600,
    }
)
sns.set_theme(style="whitegrid", rc={"grid.color": "#E8E8E8", "axes.edgecolor": "#333333"})


def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def evaluate(train: pd.DataFrame, val: pd.DataFrame, features: list[str], name: str) -> tuple[dict, pd.DataFrame]:
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(train[features], train["pseudo_stimulus"])
    train_proba = model.predict_proba(train[features])[:, 1]
    threshold = optimize_threshold(train["pseudo_stimulus"], train_proba)
    val_proba = model.predict_proba(val[features])[:, 1]
    pred = (val_proba >= threshold).astype(int)
    cm = confusion_matrix(val["pseudo_stimulus"], pred, labels=[0, 1])
    result = {
        "feature_set": name,
        "model": "gradient boosting",
        "threshold": threshold,
        "accuracy": accuracy_score(val["pseudo_stimulus"], pred),
        "balanced_accuracy": balanced_accuracy_score(val["pseudo_stimulus"], pred),
        "positive_recall": recall_score(val["pseudo_stimulus"], pred, zero_division=0),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    pred_df = val[["source", "pair_id", "sample_id", "condition", "window", "pseudo_stimulus"]].copy()
    pred_df["feature_set"] = name
    pred_df["predicted_probability"] = val_proba
    pred_df["predicted_stimulus"] = pred
    pred_df["correct"] = pred_df["pseudo_stimulus"].eq(pred)
    return result, pred_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan).fillna(0)
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["pseudo_stimulus"],
    )
    data, rc_features = add_reservoir_features(df, train_idx)
    train = data.iloc[train_idx]
    val = data.iloc[val_idx]

    dff_features = CORE_FEATURES + RELATIVE_FEATURES
    # Use the same RC configuration as the best main-text DFF+RC run:
    # mean+rate encoding with all RC response and derived channels.
    rc_features_best = [
        "rc_amp_rate_hrs_read",
        "rc_amp_rate_hrs_peak",
        "rc_amp_rate_lrs_read",
        "rc_amp_rate_lrs_peak",
        "rc_amp_rate_hrs_delta",
        "rc_amp_rate_lrs_delta",
        "rc_amp_rate_read_gap",
        "rc_amp_rate_peak_gap",
        "rc_amp_rate_energy",
    ]

    comparisons = {
        "DFF-only": dff_features,
        "RC-only": rc_features_best,
        "DFF + RC": dff_features + rc_features_best,
    }

    rows = []
    predictions = []
    for name, features in comparisons.items():
        result, pred_df = evaluate(train, val, features, name)
        result["feature_count"] = len(features)
        rows.append(result)
        predictions.append(pred_df)

    results = pd.DataFrame(rows)
    results["accuracy_percent"] = results["accuracy"] * 100
    results["balanced_accuracy_percent"] = results["balanced_accuracy"] * 100
    results["positive_recall_percent"] = results["positive_recall"] * 100
    results.to_csv(RAW_DIR / "dff_only_rc_only_dff_plus_rc_metrics.csv", index=False, encoding="utf-8-sig")
    pred_all = pd.concat(predictions, ignore_index=True)
    pred_all.to_csv(RAW_DIR / "dff_only_rc_only_dff_plus_rc_predictions.csv", index=False, encoding="utf-8-sig")

    metric_long = results.melt(
        id_vars="feature_set",
        value_vars=["accuracy_percent", "balanced_accuracy_percent", "positive_recall_percent"],
        var_name="metric",
        value_name="value",
    )
    metric_long["metric"] = metric_long["metric"].map(
        {
            "accuracy_percent": "Accuracy",
            "balanced_accuracy_percent": "Balanced accuracy",
            "positive_recall_percent": "Positive recall",
        }
    )
    metric_long.to_csv(RAW_DIR / "comparison_barplot_source_data.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], left=0.08, right=0.98, top=0.90, bottom=0.10, hspace=0.55, wspace=0.38)

    ax = fig.add_subplot(gs[0, :])
    sns.barplot(data=metric_long, x="feature_set", y="value", hue="metric", palette=["#8FB3D9", "#F0A36E", "#845EC2"], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=7, padding=2)
    ax.set_ylim(0, 105)
    ax.set_title("DFF-only, RC-only and fused DFF + RC readouts")
    ax.set_xlabel("")
    ax.set_ylabel("%")
    ax.legend(title="", loc="lower left", ncol=3, fontsize=7)
    panel(ax, "a")

    for i, row in results.iterrows():
        ax_cm = fig.add_subplot(gs[1, i])
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        pd.DataFrame(cm, index=["true neg.", "true pos."], columns=["pred neg.", "pred pos."]).to_csv(
            RAW_DIR / f"confusion_matrix_{row['feature_set'].replace(' ', '_').replace('+', 'plus')}.csv",
            encoding="utf-8-sig",
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=sns.light_palette(COL[row["feature_set"]], as_cmap=True),
            cbar=False,
            linewidths=0.6,
            linecolor="white",
            xticklabels=["neg.", "pos."],
            yticklabels=["neg.", "pos."],
            ax=ax_cm,
        )
        ax_cm.set_title(row["feature_set"])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True" if i == 0 else "")
        panel(ax_cm, chr(ord("b") + i))

    fig.suptitle("Fusing memristor RC states with DFF features improves the readout over either feature source alone", x=0.02, ha="left", fontsize=10, fontweight="bold")
    save(fig, "supporting_dff_rc_feature_source_comparison")
    plt.close(fig)

    print("DFF/RC comparison supporting figure generated.")
    print(results[["feature_set", "accuracy", "balanced_accuracy", "positive_recall", "tn", "fp", "fn", "tp"]].to_string(index=False))
    print(f"Figures: {FIG_DIR}")
    print(f"Source data: {RAW_DIR}")


if __name__ == "__main__":
    main()
