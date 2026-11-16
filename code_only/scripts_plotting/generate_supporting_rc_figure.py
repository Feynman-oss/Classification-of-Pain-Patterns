from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "supporting_rc_figure"
FIG_DIR = OUT_DIR / "figures"
RAW_DIR = OUT_DIR / "source_data"

RC_TABLE = ROOT / "rc_channel_ablation_figures" / "source_data" / "rc_four_channel_response_matrix.csv"
TRACE = ROOT / "rc_fig3_trace" / "source_data" / "fig3_representative_dff_rc_trace.csv"
BITS = ROOT / "rc_fig3_trace" / "source_data" / "fig3_5bit_code_matrix.csv"
HEATMAP = ROOT / "rc_main_text_figure" / "source_data" / "main_text_accuracy_heatmap_matrix.csv"
CM = ROOT / "rc_main_text_figure" / "source_data" / "main_text_best_confusion_matrix.csv"
PERF = ROOT / "rc_main_text_figure" / "source_data" / "main_text_dff_vs_dff_rc_comparison.csv"

PURPLE = "#845EC2"
BLUE = "#4C78A8"
ORANGE = "#D55E00"
GRAY = "#6E7F91"


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
    ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rc = pd.read_csv(RC_TABLE)
    trace = pd.read_csv(TRACE)
    bits = pd.read_csv(BITS)
    heatmap = pd.read_csv(HEATMAP, index_col=0)
    cm = pd.read_csv(CM, index_col=0)
    perf = pd.read_csv(PERF)

    rc.to_csv(RAW_DIR / "supporting_panel_a_rc_response_matrix.csv", index=False, encoding="utf-8-sig")
    trace.to_csv(RAW_DIR / "supporting_panel_b_dff_trace.csv", index=False, encoding="utf-8-sig")
    bits.to_csv(RAW_DIR / "supporting_panel_c_5bit_code.csv", index=False, encoding="utf-8-sig")
    heatmap.to_csv(RAW_DIR / "supporting_panel_d_accuracy_heatmap.csv", encoding="utf-8-sig")
    cm.to_csv(RAW_DIR / "supporting_panel_e_confusion_matrix.csv", encoding="utf-8-sig")
    perf.to_csv(RAW_DIR / "supporting_panel_f_performance.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(8.2, 10.0))
    outer = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.55, 1.2, 1.35, 1.25],
        width_ratios=[1.15, 1.0],
        left=0.08,
        right=0.98,
        bottom=0.06,
        top=0.94,
        hspace=0.62,
        wspace=0.32,
    )

    # a: four RC response channels, larger and readable.
    gs_a = outer[0, :].subgridspec(2, 2, hspace=0.45, wspace=0.24)
    channels = ["HRS read", "HRS peak", "LRS read", "LRS peak"]
    labels = ["SMP1", "SMP2", "SMP3", "SMP4"]
    colors = [BLUE, "#F39B7F", BLUE, "#F39B7F"]
    for i, (ch, lab, color) in enumerate(zip(channels, labels, colors)):
        ax = fig.add_subplot(gs_a[i // 2, i % 2])
        ax.plot(range(len(rc)), rc[ch], color=PURPLE, lw=1.1, alpha=0.65)
        ax.scatter(range(len(rc)), rc[ch], color=color, s=14, zorder=3)
        ax.set_title(lab, loc="right", pad=2, fontsize=9)
        ax.set_xlabel("5-bit state index" if i >= 2 else "")
        ax.set_ylabel("Current (a.u.)" if i % 2 == 0 else "")
        if i == 0:
            panel(ax, "a")

    # b: representative DFF trace.
    ax_b = fig.add_subplot(outer[1, 0])
    ax_b.plot(trace["relative_window"], trace["mean"], color=BLUE, lw=1.5)
    pos = trace[trace["pseudo_stimulus"] == 1]
    ax_b.scatter(pos["relative_window"], pos["mean"], color=ORANGE, s=24, zorder=3, label="positive window")
    for x in pos["relative_window"]:
        ax_b.axvspan(x - 0.45, x + 0.45, color=ORANGE, alpha=0.10, lw=0)
    ax_b.set_title("Representative mouse DFF windows")
    ax_b.set_xlabel("1-s window")
    ax_b.set_ylabel("Mean DFF")
    ax_b.legend(loc="best", fontsize=6)
    panel(ax_b, "b")

    # c: 5-bit code, readable labels and purple mapping.
    ax_c = fig.add_subplot(outer[1, 1])
    bit_cols = [c for c in bits.columns if c != "relative_window"]
    sns.heatmap(
        bits[bit_cols].T,
        cmap=sns.color_palette(["#F1F1F4", PURPLE], as_cmap=True),
        cbar=False,
        linewidths=0.25,
        linecolor="white",
        ax=ax_c,
    )
    ax_c.set_title("5-bit pulse encoding")
    ax_c.set_xlabel("1-s window")
    ax_c.set_ylabel("")
    ax_c.set_yticklabels(bit_cols, rotation=0)
    ax_c.set_xticks([])
    panel(ax_c, "c")

    # d: DFF + RC channel ablation heatmap.
    ax_d = fig.add_subplot(outer[2, :])
    sns.heatmap(
        heatmap,
        cmap="YlGnBu",
        vmin=88,
        vmax=94,
        annot=True,
        fmt=".1f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax_d,
    )
    ax_d.set_title("DFF + RC channel ablation")
    ax_d.set_xlabel("RC channel group")
    ax_d.set_ylabel("5-bit encoding")
    ax_d.tick_params(axis="x", rotation=20)
    ax_d.tick_params(axis="y", rotation=0)
    panel(ax_d, "d")

    # e: confusion matrix.
    ax_e = fig.add_subplot(outer[3, 0])
    sns.heatmap(
        cm,
        cmap=sns.light_palette(ORANGE, as_cmap=True),
        annot=True,
        fmt="d",
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        ax=ax_e,
    )
    ax_e.set_title("Best DFF + RC readout")
    ax_e.set_xlabel("Predicted")
    ax_e.set_ylabel("True")
    panel(ax_e, "e")

    # f: performance comparison.
    ax_f = fig.add_subplot(outer[3, 1])
    perf_long = perf.melt(
        id_vars="method",
        value_vars=["accuracy_percent", "balanced_accuracy_percent"],
        var_name="metric",
        value_name="value",
    )
    perf_long["metric"] = perf_long["metric"].map(
        {"accuracy_percent": "Accuracy", "balanced_accuracy_percent": "Balanced acc."}
    )
    sns.barplot(data=perf_long, x="method", y="value", hue="metric", palette=[BLUE, ORANGE], ax=ax_f)
    ax_f.set_ylim(0, 100)
    ax_f.set_title("Validation performance")
    ax_f.set_xlabel("")
    ax_f.set_ylabel("%")
    ax_f.legend(title="", fontsize=6, loc="lower left")
    panel(ax_f, "f")

    fig.suptitle(
        "Supporting analysis of memristor RC mapping for DFF-window classification",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    save(fig, "supporting_rc_mapping_and_readout")
    plt.close(fig)

    print("Supporting RC figure generated.")
    print(f"Figures: {FIG_DIR}")
    print(f"Source data: {RAW_DIR}")


if __name__ == "__main__":
    main()
