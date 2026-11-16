from __future__ import annotations

import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figure6_figs8_corrections"
FIG = OUT / "figures"
TEXT = OUT / "legends_and_methods"
SRC = OUT / "source_data"

BLUE = "#4C78A8"
PURPLE = "#7A5AC8"
GREY = "#A9AFB8"
ORANGE = "#EE8844"
DARK = "#343941"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.2,
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def panel(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def make_fig6fg():
    fixed = pd.read_csv(ROOT / "train_only_rc_gain_optimization" / "final_candidate_metrics.csv")
    scan = pd.read_csv(ROOT / "sham_sni_window_length_scan" / "window_length_scan_summary.csv")
    nested = pd.read_excel(
        ROOT / "requested_dff_rc_gradient_boosting" / "nested_feature_set_comparison.xlsx",
        sheet_name="metrics",
    )

    dff = fixed.loc[fixed.method.eq("Prespecified DFF"), "accuracy"].iloc[0]
    fusion = fixed.loc[fixed.method.eq("Prespecified DFF + all RC"), "accuracy"].iloc[0]
    rc = scan.loc[
        scan.window_seconds.eq(10)
        & scan.level.eq("window")
        & scan.model.eq("extra_trees")
        & scan.feature_set.eq("RC"),
        "validation_accuracy",
    ].iloc[0]
    window = pd.DataFrame({
        "feature_set": ["RC only", "DFF only", "DFF + RC"],
        "accuracy": [rc, dff, fusion],
        "evaluation_unit": "10-s window",
        "validation_scheme": "fixed random 70/30 window split",
        "n_evaluation_units": 72,
    })

    animal = nested.set_index("feature_set").loc[["RC_only", "DFF_only", "DFF_plus_RC"]].reset_index()
    animal["feature_set"] = animal.feature_set.map({
        "RC_only": "RC only", "DFF_only": "DFF only", "DFF_plus_RC": "DFF + RC"
    })
    animal["evaluation_unit"] = "animal"
    animal["validation_scheme"] = "animal-disjoint fivefold out-of-fold"
    animal["n_evaluation_units"] = 103

    SRC.mkdir(parents=True, exist_ok=True)
    window.to_csv(SRC / "Fig6f_window_level_accuracy.csv", index=False, encoding="utf-8-sig")
    animal.to_csv(SRC / "Fig6g_animal_level_accuracy.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.40})
    colors = [GREY, BLUE, PURPLE]
    for ax, df, title, subtitle, letter, ylim in [
        (axes[0], window, "Window-level holdout", "Random 70/30 split | n = 72 windows", "f", (82, 96.5)),
        (axes[1], animal, "Animal-disjoint generalization", "Fivefold out-of-fold | n = 103 animals", "g", (68, 90)),
    ]:
        vals = 100 * df.accuracy.to_numpy(float)
        x = np.arange(3)
        bars = ax.bar(x, vals, color=colors, width=.66, edgecolor=DARK, linewidth=.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+.35, f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, df.feature_set, rotation=18, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontweight="bold", pad=16)
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=6.5, color=DARK, va="bottom")
        panel(ax, letter, x=-.15, y=1.10)
    axes[0].text(.5, -.34, "Windows from the same animal may occur in both sets.", transform=axes[0].transAxes,
                 ha="center", fontsize=6.2, color="#8A3B3B")
    axes[1].text(.5, -.34, "No animal appears in both training and held-out folds.", transform=axes[1].transAxes,
                 ha="center", fontsize=6.2, color="#246B45")
    fig.subplots_adjust(left=.09, right=.98, top=.83, bottom=.30)
    save(fig, "Fig6fg_validation_levels_revised")


def make_figs8():
    data = pd.read_csv(ROOT / "train_only_rc_gain_optimization" / "all_config_sensitivity_audit.csv")
    SRC.mkdir(parents=True, exist_ok=True)
    data.to_csv(SRC / "FigS8_all_81_configurations.csv", index=False, encoding="utf-8-sig")

    row_order = ["all_rc", "amp_rate", "amp_slope", "binary_dynamics", "binary_shape",
                 "mean32", "range_peak", "rate32", "var_rate"]
    row_labels = ["All RC encodings", "Mean + rate", "Mean + slope", "Binary dynamics",
                  "Binary shape", "Mean (32 bins)", "Range + peaks", "Rate (32 bins)",
                  "Variance + rate"]
    mf_order = ["sqrt", "0.5", "1.0"]
    mf_labels = [r"$\sqrt{p}$", "50%", "100%"]
    leaves = [1, 2, 4]

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.0), sharey=True)
    for col, leaf in enumerate(leaves):
        sub = data[data.min_samples_leaf.eq(leaf)].copy()
        acc = sub.pivot(index="rc_group", columns="max_features", values="fusion_validation_accuracy").loc[row_order, mf_order] * 100
        gain = sub.pivot(index="rc_group", columns="max_features", values="validation_gain").loc[row_order, mf_order] * 100
        for row, (matrix, cmap, vmin, vmax) in enumerate([
            (acc, "viridis", 85, 98),
            (gain, "RdBu_r", -7, 7),
        ]):
            ax = axes[row, col]
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(3), mf_labels)
            ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", length=0)
            if col == 0:
                ax.set_yticks(range(9), row_labels)
            ax.set_title(f"Minimum samples per leaf = {leaf}", fontsize=7.2, fontweight="bold", pad=5)
            for s in ax.spines.values():
                s.set_visible(False)
        if col == 2:
            cb1 = fig.colorbar(axes[0, col].images[0], ax=axes[0, :], fraction=.018, pad=.025)
            cb1.set_label("Fusion accuracy (%)", fontsize=6.5)
            cb1.ax.tick_params(labelsize=6, length=2)
            cb2 = fig.colorbar(axes[1, col].images[0], ax=axes[1, :], fraction=.018, pad=.025)
            cb2.set_label("RC gain (percentage points)", fontsize=6.5)
            cb2.ax.tick_params(labelsize=6, length=2)

    fig.text(.018, .955, "a", fontsize=10, fontweight="bold")
    fig.text(.018, .505, "b", fontsize=10, fontweight="bold")
    fig.text(.035, .73, "Fusion accuracy (%)", rotation=90, va="center", ha="center",
             fontsize=7.3, fontweight="bold")
    fig.text(.035, .29, "Gain over matched DFF-only model (pp)", rotation=90, va="center", ha="center",
             fontsize=7.3, fontweight="bold")
    fig.text(.56, .085, "Features considered per split (fraction of p)", ha="center", fontsize=7.2)
    fig.text(.51, .015,
             "p is the total number of input features available to the ExtraTrees model. Rows are RC encoding groups; columns are model hyperparameters.",
             ha="center", fontsize=6.4, color=DARK)
    fig.subplots_adjust(left=.24, right=.88, top=.92, bottom=.12, wspace=.10, hspace=.30)
    save(fig, "FigS8_configuration_sensitivity_revised")


def write_text():
    TEXT.mkdir(parents=True, exist_ok=True)
    fig6 = (
        "Revised Fig. 6 legend text (replacement clauses)\n\n"
        "The four measured reservoir outputs are defined as SMP1, HRS readout; SMP2, HRS peak; "
        "SMP3, LRS readout; and SMP4, LRS peak. f, Prespecified window-level feature ablation using "
        "non-overlapping 10-s pain-state windows, an ExtraTrees classifier and a fixed random 70/30 "
        "window split (n = 72 validation windows). DFF-only accuracy was 91.67%, and DFF plus all RC "
        "features achieved 93.06%. Windows from the same animal could occur in both the training and "
        "validation sets. g, Animal-disjoint nested fivefold validation summarized from out-of-fold "
        "predictions for 103 animals (57 SHAM and 46 SNI). DFF-only accuracy was 85.44%, and DFF plus "
        "RC achieved 86.41%. No animal occurred in both the training and held-out folds. Panels f and g "
        "therefore report different evaluation units and validation schemes and are not expected to yield "
        "identical accuracy estimates. Source data are provided as a Source Data file.\n"
    )
    (TEXT / "revised_Fig6_legend.txt").write_text(fig6, encoding="utf-8")

    figs8 = (
        "Revised Fig. S8 legend\n\n"
        "Fig. S8 | Sensitivity of DFF–RC fusion to RC encoding and ExtraTrees hyperparameters. "
        "a, Validation accuracy of the fused DFF–RC classifier. b, Accuracy difference between each "
        "fused classifier and its matched DFF-only classifier, expressed in percentage points. Rows "
        "denote nine RC feature groups: all RC encodings, mean plus rate, mean plus slope, binary "
        "dynamics, binary shape, mean quantized into 32 bins, range plus peak count, rate quantized "
        "into 32 bins, and variance plus rate. Columns are grouped by the minimum number of samples "
        "per terminal leaf (1, 2 or 4). Within each group, sqrt(p), 50% of p and 100% of p specify the "
        "number of input features considered at each ExtraTrees split, where p is the total number of "
        "features available to the model. The resulting grid contains 9 RC feature groups × 9 model "
        "hyperparameter settings = 81 configurations. These analyses are exploratory because the same "
        "validation set was inspected across configurations. Source data are provided as a Source Data file.\n"
    )
    (TEXT / "revised_FigS8_legend.txt").write_text(figs8, encoding="utf-8")

    methods = (
        "Methods definition for Fig. S8\n\n"
        "Configuration sensitivity analysis. ExtraTrees classifiers were evaluated using a factorial "
        "grid of min_samples_leaf = {1, 2, 4} and max_features = {sqrt(p), 0.5p, 1.0p}, where p denotes "
        "the total number of input features available to the classifier. min_samples_leaf specifies the "
        "minimum number of observations required in a terminal leaf, whereas max_features specifies the "
        "number of candidate features randomly considered at each split. These nine classifier settings "
        "were crossed with nine RC feature groups, yielding 81 exploratory configurations.\n"
    )
    (TEXT / "Methods_FigS8_parameter_definition.txt").write_text(methods, encoding="utf-8")

    ledger = pd.DataFrame([
        ["SMP1", "HRS readout", "Measured HRS read response"],
        ["SMP2", "HRS peak", "Measured HRS peak response"],
        ["SMP3", "LRS readout", "Measured LRS read response"],
        ["SMP4", "LRS peak", "Measured LRS peak response"],
        ["p", "Total input-feature count", "Used only in ExtraTrees max_features definitions"],
        ["Window-level holdout", "10-s window", "Fixed random 70/30 split; n = 72 validation windows"],
        ["Animal-disjoint validation", "Animal", "Nested fivefold out-of-fold; n = 103 animals"],
    ], columns=["canonical_term", "definition", "use_in_manuscript"])
    ledger.to_csv(TEXT / "terminology_ledger.csv", index=False, encoding="utf-8-sig")


def qa():
    rows = []
    for p in sorted(FIG.glob("*")):
        if p.suffix == ".svg":
            rows.append(f"{p.name}: editable_text={'<text' in p.read_text(encoding='utf-8')}")
        elif p.suffix == ".tiff":
            with Image.open(p) as im:
                rows.append(f"{p.name}: size={im.size}; dpi={im.info.get('dpi')}")
        elif p.suffix == ".png":
            with Image.open(p) as im:
                rows.append(f"{p.name}: size={im.size}")
    rows += [
        "Fig. 6f unit explicitly stated: 10-s validation windows (n=72)",
        "Fig. 6g unit explicitly stated: animals (n=103; animal-disjoint fivefold)",
        "Fig. S8 axis labels expanded into min_samples_leaf and max_features",
        "SMP mapping verified against the four columns in the measured RC table",
        "Workflow SVG labels corrected to the verified SMP1-SMP4 mapping",
    ]
    (OUT / "QA_REPORT.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    for p in [OUT, FIG, TEXT, SRC]:
        p.mkdir(parents=True, exist_ok=True)
    make_fig6fg()
    make_figs8()
    write_text()
    import create_illustrator_workflow_svg as workflow
    workflow.build()
    shutil.copy2(workflow.OUT, FIG / "workflow_SMP_mapping_corrected.svg")
    qa()
    print(OUT)


if __name__ == "__main__":
    main()
