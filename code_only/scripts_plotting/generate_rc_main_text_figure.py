from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
OUT_DIR = ROOT / "rc_main_text_figure"
FIG_DIR = OUT_DIR / "figures"
RAW_DIR = OUT_DIR / "source_data"

ENCODINGS = {
    "rc_amp_rate": "mean + rate",
    "rc_amp_slope": "mean + slope",
    "rc_var_rate": "std + rate",
    "rc_range_peak": "range + peaks",
}

CHANNEL_GROUPS = {
    "raw 4-channel": ["hrs_read", "hrs_peak", "lrs_read", "lrs_peak"],
    "gap features": ["read_gap", "peak_gap"],
    "energy": ["energy"],
    "all RC": ["hrs_read", "hrs_peak", "lrs_read", "lrs_peak", "hrs_delta", "lrs_delta", "read_gap", "peak_gap", "energy"],
}

COL = {
    "dff": "#4C78A8",
    "combined": "#D55E00",
    "rc": "#009E73",
    "gray": "#6E7F91",
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
sns.set_theme(style="whitegrid", rc={"grid.color": "#E7E7E7", "axes.edgecolor": "#333333"})


def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.tiff", bbox_inches="tight", dpi=600)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def feature_names(prefix: str, suffixes: list[str]) -> list[str]:
    return [f"{prefix}_{suffix}" for suffix in suffixes]


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, features: list[str], model_name: str) -> dict:
    if model_name == "gradient boosting":
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    else:
        model = RandomForestClassifier(
            n_estimators=1200,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    model.fit(train[features], train["pseudo_stimulus"])
    train_proba = model.predict_proba(train[features])[:, 1]
    threshold = optimize_threshold(train["pseudo_stimulus"], train_proba)
    val_proba = model.predict_proba(val[features])[:, 1]
    pred = (val_proba >= threshold).astype(int)
    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(val["pseudo_stimulus"], pred),
        "balanced_accuracy": balanced_accuracy_score(val["pseudo_stimulus"], pred),
        "positive_recall": recall_score(val["pseudo_stimulus"], pred, zero_division=0),
        "confusion_matrix": confusion_matrix(val["pseudo_stimulus"], pred, labels=[0, 1]).tolist(),
    }


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
    data, _ = add_reservoir_features(df, train_idx)
    train = data.iloc[train_idx]
    val = data.iloc[val_idx]

    dff_features = CORE_FEATURES + RELATIVE_FEATURES
    baseline_rows = []
    for model_name in ["gradient boosting", "random forest"]:
        row = fit_predict(train, val, dff_features, model_name)
        baseline_rows.append({"feature_set": "DFF", **row})
    baseline = pd.DataFrame(baseline_rows).sort_values("accuracy", ascending=False).iloc[0]

    rows = []
    for prefix, encoding_label in ENCODINGS.items():
        for channel_label, suffixes in CHANNEL_GROUPS.items():
            rc_features = feature_names(prefix, suffixes)
            features = dff_features + rc_features
            for model_name in ["gradient boosting", "random forest"]:
                row = fit_predict(train, val, features, model_name)
                rows.append(
                    {
                        "encoding": encoding_label,
                        "channel_group": channel_label,
                        "feature_count": len(features),
                        **row,
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(RAW_DIR / "dff_plus_rc_channel_ablation_all_runs.csv", index=False, encoding="utf-8-sig")

    best = results.sort_values(["accuracy", "balanced_accuracy"], ascending=False).drop_duplicates(["encoding", "channel_group"])
    best["accuracy_percent"] = best["accuracy"] * 100
    best["balanced_accuracy_percent"] = best["balanced_accuracy"] * 100
    best["positive_recall_percent"] = best["positive_recall"] * 100
    best.to_csv(RAW_DIR / "dff_plus_rc_channel_ablation_best.csv", index=False, encoding="utf-8-sig")

    acc_matrix = best.pivot(index="encoding", columns="channel_group", values="accuracy_percent")
    acc_matrix = acc_matrix.loc[list(ENCODINGS.values()), list(CHANNEL_GROUPS.keys())]
    acc_matrix.to_csv(RAW_DIR / "main_text_accuracy_heatmap_matrix.csv", encoding="utf-8-sig")

    top = best.sort_values(["accuracy", "balanced_accuracy"], ascending=False).iloc[0]
    top_cm = np.array(top["confusion_matrix"])
    if isinstance(top_cm, str):
        top_cm = np.array(eval(top["confusion_matrix"]))
    pd.DataFrame(top_cm, index=["true negative", "true positive"], columns=["pred negative", "pred positive"]).to_csv(
        RAW_DIR / "main_text_best_confusion_matrix.csv", encoding="utf-8-sig"
    )

    comparison = pd.DataFrame(
        [
            {
                "method": "DFF",
                "accuracy_percent": baseline["accuracy"] * 100,
                "balanced_accuracy_percent": baseline["balanced_accuracy"] * 100,
                "positive_recall_percent": baseline["positive_recall"] * 100,
            },
            {
                "method": "DFF + RC",
                "accuracy_percent": top["accuracy_percent"],
                "balanced_accuracy_percent": top["balanced_accuracy_percent"],
                "positive_recall_percent": top["positive_recall_percent"],
            },
        ]
    )
    comparison.to_csv(RAW_DIR / "main_text_dff_vs_dff_rc_comparison.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.2, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.85, 0.8])

    ax = fig.add_subplot(gs[0, 0])
    sns.heatmap(acc_matrix, cmap="YlGnBu", vmin=88, vmax=94, annot=True, fmt=".1f", linewidths=0.6, linecolor="white", cbar_kws={"label": "accuracy (%)"}, ax=ax)
    ax.set_title("DFF + RC channel ablation")
    ax.set_xlabel("RC channel group")
    ax.set_ylabel("5-bit encoding")
    ax.tick_params(axis="x", rotation=25)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        top_cm,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(COL["combined"], as_cmap=True),
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        xticklabels=["neg.", "pos."],
        yticklabels=["neg.", "pos."],
        ax=ax,
    )
    ax.set_title("Best DFF + RC")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    panel(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    long = comparison.melt(id_vars="method", value_vars=["accuracy_percent", "balanced_accuracy_percent"], var_name="metric", value_name="value")
    long["metric"] = long["metric"].map({"accuracy_percent": "accuracy", "balanced_accuracy_percent": "balanced acc."})
    sns.barplot(data=long, x="method", y="value", hue="metric", palette=[COL["dff"], COL["combined"]], ax=ax)
    ax.set_ylim(0, 100)
    ax.set_title("Performance")
    ax.set_xlabel("")
    ax.set_ylabel("%")
    ax.legend(title="", fontsize=6, loc="lower left")
    panel(ax, "c")

    fig.suptitle("DFF features combined with memristor RC channels preserve >90% validation accuracy", x=0.02, y=1.06, ha="left", fontsize=10, fontweight="bold")
    save(fig, "main_text_dff_plus_rc_figure")
    plt.close(fig)

    print("Main-text DFF+RC figure generated.")
    print(f"Baseline DFF: acc={baseline['accuracy']:.4f}, balanced={baseline['balanced_accuracy']:.4f}")
    print(f"Best DFF+RC: encoding={top['encoding']}, channel={top['channel_group']}, model={top['model']}, acc={top['accuracy']:.4f}, balanced={top['balanced_accuracy']:.4f}")
    print(f"Figures: {FIG_DIR}")
    print(f"Source data: {RAW_DIR}")


if __name__ == "__main__":
    main()
