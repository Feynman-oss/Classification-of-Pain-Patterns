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
    DATA_PATH,
    RANDOM_STATE,
    add_reservoir_features,
    optimize_threshold,
)


ROOT = Path(__file__).resolve().parent
RC_PATH = ROOT / "RC.xlsx"
OUT_DIR = ROOT / "rc_channel_ablation_figures"
FIG_DIR = OUT_DIR / "figures"
RAW_DIR = OUT_DIR / "source_data"

ENCODINGS = [
    "rc_amp_rate",
    "rc_amp_slope",
    "rc_var_rate",
    "rc_range_peak",
    "rc_mean32",
    "rc_rate32",
    "rc_binary_shape",
    "rc_binary_dynamics",
]

CHANNEL_GROUPS = {
    "HRS read": ["hrs_read"],
    "HRS peak": ["hrs_peak"],
    "LRS read": ["lrs_read"],
    "LRS peak": ["lrs_peak"],
    "read gap": ["read_gap"],
    "peak gap": ["peak_gap"],
    "energy": ["energy"],
    "all 4 raw": ["hrs_read", "hrs_peak", "lrs_read", "lrs_peak"],
    "all derived": ["hrs_delta", "lrs_delta", "read_gap", "peak_gap", "energy"],
    "all RC": ["hrs_read", "hrs_peak", "lrs_read", "lrs_peak", "hrs_delta", "lrs_delta", "read_gap", "peak_gap", "energy"],
}

COL = {
    "green": "#009E73",
    "orange": "#D55E00",
    "blue": "#4C78A8",
    "gray": "#6E7F91",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
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
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def channel_features(prefix: str, suffixes: list[str]) -> list[str]:
    return [f"{prefix}_{suffix}" for suffix in suffixes]


def evaluate(train: pd.DataFrame, val: pd.DataFrame, features: list[str], model_name: str) -> dict:
    if model_name == "random forest":
        model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
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

    rows = []
    for encoding in ENCODINGS:
        for group, suffixes in CHANNEL_GROUPS.items():
            features = channel_features(encoding, suffixes)
            for model_name in ["gradient boosting", "random forest"]:
                out = evaluate(train, val, features, model_name)
                rows.append(
                    {
                        "encoding": encoding.replace("rc_", ""),
                        "channel_group": group,
                        "feature_count": len(features),
                        **out,
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(RAW_DIR / "rc_channel_ablation_summary.csv", index=False, encoding="utf-8-sig")

    # Best model per encoding/channel group keeps the heatmap readable.
    best = summary.sort_values(["accuracy", "balanced_accuracy"], ascending=False).drop_duplicates(
        ["encoding", "channel_group"]
    )
    best.to_csv(RAW_DIR / "rc_channel_ablation_best_per_group.csv", index=False, encoding="utf-8-sig")

    acc_pivot = best.pivot(index="encoding", columns="channel_group", values="accuracy") * 100
    bal_pivot = best.pivot(index="encoding", columns="channel_group", values="balanced_accuracy") * 100
    recall_pivot = best.pivot(index="encoding", columns="channel_group", values="positive_recall") * 100
    acc_pivot.to_csv(RAW_DIR / "heatmap_accuracy_matrix.csv", encoding="utf-8-sig")
    bal_pivot.to_csv(RAW_DIR / "heatmap_balanced_accuracy_matrix.csv", encoding="utf-8-sig")
    recall_pivot.to_csv(RAW_DIR / "heatmap_positive_recall_matrix.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), constrained_layout=True)
    for ax, matrix, title, label in [
        (axes[0], acc_pivot, "Accuracy (%)", "a"),
        (axes[1], bal_pivot, "Balanced accuracy (%)", "b"),
        (axes[2], recall_pivot, "Positive recall (%)", "c"),
    ]:
        sns.heatmap(matrix, cmap="viridis", annot=True, fmt=".1f", cbar=True, linewidths=0.35, linecolor="white", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("RC channel group")
        ax.set_ylabel("5-bit encoding")
        ax.tick_params(axis="x", rotation=45)
        panel(ax, label)
    fig.suptitle("RC channel and encoding ablation for reservoir readout", x=0.02, y=1.05, ha="left", fontsize=9, weight="bold")
    save(fig, "rc_channel_encoding_ablation_heatmaps")
    plt.close(fig)

    rc = pd.read_excel(RC_PATH, sheet_name=0)
    rc = rc.drop(columns=[c for c in rc.columns if str(c).startswith("Unnamed")], errors="ignore").dropna(how="all")
    rc["code"] = rc["编码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    rc_renamed = rc.rename(
        columns={
            "HRS 读取": "HRS read",
            "HRS 峰值": "HRS peak",
            "LRS 读取": "LRS read",
            "LRS 峰值": "LRS peak",
        }
    )
    response = rc_renamed.set_index("code")[["HRS read", "HRS peak", "LRS read", "LRS peak"]]
    response.to_csv(RAW_DIR / "rc_four_channel_response_matrix.csv", encoding="utf-8-sig")

    fig2, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    sns.heatmap(response, cmap="mako", ax=axes[0], cbar_kws={"label": "RC response"})
    axes[0].set_title("Four-channel RC response map")
    axes[0].set_xlabel("channel")
    axes[0].set_ylabel("5-bit state")
    panel(axes[0], "a")

    best_top = summary.sort_values(["accuracy", "balanced_accuracy"], ascending=False).head(12).copy()
    best_top["run"] = best_top["encoding"] + "\n" + best_top["channel_group"] + "\n" + best_top["model"]
    best_top.to_csv(RAW_DIR / "top_rc_channel_runs.csv", index=False, encoding="utf-8-sig")
    sns.barplot(data=best_top, y="run", x="accuracy", color=COL["green"], ax=axes[1])
    axes[1].set_xlim(0, 1)
    axes[1].axvline(0.9, color="#555555", lw=0.8, ls="--")
    axes[1].set_title("Top RC-only channel runs")
    axes[1].set_xlabel("accuracy")
    axes[1].set_ylabel("")
    panel(axes[1], "b")
    fig2.suptitle("Memristor response states and RC-only readout performance", x=0.02, y=1.05, ha="left", fontsize=9, weight="bold")
    save(fig2, "rc_response_map_and_top_runs")
    plt.close(fig2)

    print("RC channel ablation figures generated.")
    print(f"Figures: {FIG_DIR}")
    print(f"Source data: {RAW_DIR}")
    print(best_top[["encoding", "channel_group", "model", "accuracy", "balanced_accuracy", "positive_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
