from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", str((__import__("pathlib").Path(__file__).resolve().parent / ".mplconfig")))

import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import classify_sham_sni_with_rc as data_source
from scan_equal_duration_rc_windows import resample_segment


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "manuscript_main_figure_rc_sham_sni"
SRC = OUT / "source_data"
SEED = 20260706

SHAM = "#4C78A8"
SNI = "#D96B36"
RC_COLOR = "#7A5AC8"
RC_DARK = "#57429A"
NEUTRAL = "#B8BDC7"
NEUTRAL_DARK = "#5D6470"
GAIN = "#2E8B57"


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.2
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7


def panel_label(ax, label: str, x=-0.13, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")


def all_complete_windows():
    traces, _ = data_source.load_traces()
    records = []
    for trace in traces:
        if trace.condition != "pain":
            continue
        time_s = (trace.timestamps_ns - trace.timestamps_ns[0]) / 1e9
        n_complete = int(np.floor(float(time_s[-1]) / 10.0 + 1e-9))
        for window in range(n_complete):
            x = resample_segment(trace.values, time_s, 10.0 * window, 10.0 * (window + 1), 300)
            if x is None:
                continue
            records.append({
                "biological_id": trace.biological_id,
                "source": str(trace.source),
                "model_type": trace.model_type,
                "target": int(trace.target),
                "window": window,
                "signal": np.asarray(x, float),
            })
    return records


def representative_records(records):
    z = np.asarray([np.mean(r["signal"]) + np.mean(np.abs(np.diff(r["signal"]))) for r in records])
    y = np.asarray([r["target"] for r in records])
    train, _ = train_test_split(np.arange(len(records)), test_size=0.30, random_state=SEED, stratify=y)
    z_min, z_max = float(z[train].min()), float(z[train].max())
    q_all = np.clip(np.rint(31.0 * (z - z_min) / (z_max - z_min)), 0, 31).astype(int)
    selected = {}
    target_code = {"sham": 3, "sni": 5}
    for label in ["sham", "sni"]:
        candidates = [r for r, q in zip(records, q_all) if r["source"] == "3" and r["model_type"] == label and q == target_code[label]]
        feats = np.asarray([
            [np.mean(r["signal"]), np.std(r["signal"]), np.ptp(r["signal"]), np.mean(np.abs(np.diff(r["signal"])))]
            for r in candidates
        ])
        scale = np.std(feats, axis=0)
        scale[scale < 1e-12] = 1.0
        distance = np.sum(((feats - np.median(feats, axis=0)) / scale) ** 2, axis=1)
        selected[label] = candidates[int(np.argmin(distance))]
    return selected, z_min, z_max


def encode_records(selected, z_min, z_max):
    rc_table = data_source.read_rc_table()
    rows = []
    for label, record in selected.items():
        value = float(np.mean(record["signal"]) + np.mean(np.abs(np.diff(record["signal"]))))
        q = int(np.clip(np.rint(31.0 * (value - z_min) / (z_max - z_min)), 0, 31))
        code = f"{q:05b}"
        rc = rc_table.loc[code]
        rows.append({
            "model_type": label,
            "biological_id": record["biological_id"],
            "window": record["window"],
            "z_mean_plus_change": value,
            "q": q,
            "code": code,
            **{data_source.RC_COLUMNS[col]: float(rc[col]) for col in data_source.RC_COLUMNS},
        })
    return pd.DataFrame(rows), z_min, z_max


def performance_tables():
    summary = pd.read_csv(ROOT / "sham_sni_window_length_scan" / "window_length_scan_summary.csv")
    ten = summary[(summary.window_seconds == 10.0) & (summary.level == "window")].copy()
    gb = ten[ten.model == "gradient_boosting"].copy()
    order = ["RC", "DFF", "DFF_RC"]
    gb["feature_set"] = pd.Categorical(gb.feature_set, order, ordered=True)
    ablation = gb.sort_values("feature_set")[["feature_set", "validation_accuracy", "validation_balanced_accuracy", "validation_f1_sni"]]
    ablation["feature_label"] = ablation.feature_set.astype(str).map({"RC": "RC only", "DFF": "DFF only", "DFF_RC": "DFF + RC"})

    paired = ten[ten.feature_set.isin(["DFF", "DFF_RC"])].pivot(index="model", columns="feature_set", values="validation_accuracy").reset_index()
    paired["model_label"] = paired.model.map({"extra_trees": "ExtraTrees", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting"})
    paired["gain_percentage_points"] = 100.0 * (paired.DFF_RC - paired.DFF)
    paired = paired.sort_values("model_label")

    best = ten[(ten.feature_set == "DFF_RC") & (ten.model == "gradient_boosting")].iloc[0]
    cm = np.asarray(ast.literal_eval(best.validation_confusion_matrix), dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    return ablation, paired, cm, cm_norm


def draw_figure(selected, encoding, ablation, paired, cm_norm):
    fig = plt.figure(figsize=(7.15, 5.7), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.42)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.36)
    trace_grid = top[0].subgridspec(2, 1, hspace=0.12)
    ax_sham = fig.add_subplot(trace_grid[0])
    ax_sni = fig.add_subplot(trace_grid[1], sharex=ax_sham)
    code_grid = top[1].subgridspec(2, 1, height_ratios=[0.9, 1.0], hspace=0.38)
    ax_bits = fig.add_subplot(code_grid[0])
    ax_rc = fig.add_subplot(code_grid[1])

    time = np.linspace(0, 10, 300, endpoint=False)
    for ax, label, color in [(ax_sham, "sham", SHAM), (ax_sni, "sni", SNI)]:
        signal = selected[label]["signal"]
        ax.plot(time, signal, color=color, lw=1.15)
        ax.axhline(0, color="#D9DCE2", lw=0.55, zorder=0)
        ax.text(0.012, 0.83, label.upper(), transform=ax.transAxes, color=color, fontweight="bold", fontsize=7.5)
        ax.set_ylabel("DFF")
        ax.margins(x=0)
        ax.tick_params(length=2.5)
    ax_sham.tick_params(labelbottom=False)
    ax_sni.set_xlabel("Time (s)")
    ax_sham.set_title("Representative pain-state 10-s signals", loc="left", pad=5, fontweight="bold")
    panel_label(ax_sham, "a", x=-0.11, y=1.08)

    bit_matrix = np.asarray([[int(v) for v in code] for code in encoding.code])
    bit_cmap = LinearSegmentedColormap.from_list("bit", ["#F1F2F5", RC_COLOR])
    ax_bits.imshow(bit_matrix, cmap=bit_cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(2):
        for j in range(5):
            ax_bits.text(j, i, str(bit_matrix[i, j]), ha="center", va="center", fontsize=7,
                         color="white" if bit_matrix[i, j] else NEUTRAL_DARK, fontweight="bold")
    ax_bits.set_xticks(range(5), [r"$b_1$", r"$b_2$", r"$b_3$", r"$b_4$", r"$b_5$"])
    ax_bits.set_yticks(range(2), ["SHAM", "SNI"])
    ax_bits.tick_params(length=0)
    for spine in ax_bits.spines.values():
        spine.set_visible(False)
    ax_bits.set_title("5-bit input code", loc="left", pad=5, fontweight="bold")
    panel_label(ax_bits, "b", x=-0.22, y=1.08)

    rc_cols = ["hrs_read", "hrs_peak", "lrs_read", "lrs_peak"]
    rc_values = encoding[rc_cols].to_numpy(float)
    scale = np.max(np.abs(rc_values), axis=0)
    scale[scale < 1e-12] = 1
    rc_scaled = rc_values / scale
    rc_cmap = LinearSegmentedColormap.from_list("rc", ["#E7EEF6", "#FFFFFF", "#E5DDF5"])
    ax_rc.imshow(rc_scaled, cmap=rc_cmap, vmin=-1, vmax=1, aspect="auto")
    ax_rc.set_xticks(range(4), ["HRS\nread", "HRS\npeak", "LRS\nread", "LRS\npeak"])
    ax_rc.set_yticks(range(2), ["SHAM", "SNI"])
    ax_rc.tick_params(length=0)
    for spine in ax_rc.spines.values():
        spine.set_visible(False)
    ax_rc.set_title("Measured RC state (channel-normalized)", loc="left", pad=5, fontweight="bold")

    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.12, 0.88, 1.38], wspace=0.68)
    ax_ab = fig.add_subplot(bottom[0])
    ax_cm = fig.add_subplot(bottom[1])
    ax_gain = fig.add_subplot(bottom[2])

    values = 100 * ablation.validation_accuracy.to_numpy(float)
    labels = ablation.feature_label.tolist()
    colors = [NEUTRAL, SHAM, RC_COLOR]
    x = np.arange(3)
    bars = ax_ab.bar(x, values, width=0.68, color=colors, edgecolor="#40444B", linewidth=0.65)
    for bar, value in zip(bars, values):
        ax_ab.text(bar.get_x() + bar.get_width()/2, value + 0.35, f"{value:.1f}", ha="center", va="bottom", fontsize=6.8)
    ax_ab.set_xticks(x, labels, rotation=24, ha="right")
    ax_ab.set_ylabel("Accuracy (%)")
    lower = max(0, 5 * np.floor(np.min(values) / 5) - 3)
    ax_ab.set_ylim(lower, 98)
    ax_ab.set_title("Feature ablation", loc="left", pad=5, fontweight="bold")
    panel_label(ax_ab, "c")

    im = ax_cm.imshow(cm_norm, cmap="Purples", vmin=0, vmax=1, aspect="equal")
    ax_cm.set_xticks([0, 1], ["SHAM", "SNI"])
    ax_cm.set_yticks([0, 1], ["SHAM", "SNI"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.tick_params(length=0)
    for spine in ax_cm.spines.values():
        spine.set_visible(False)
    ax_cm.set_title("Best 10-s confusion\nstructure", loc="left", pad=5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.045, ticks=[0, 0.5, 1])
    cbar.set_label("Fraction", fontsize=6.2)
    cbar.ax.tick_params(labelsize=6, length=2)
    panel_label(ax_cm, "d", x=-0.22)

    preferred = ["ExtraTrees", "Gradient boosting", "Random forest"]
    model_order = [name for name in preferred if name in set(paired.model_label)]
    model_order += [name for name in paired.model_label if name not in model_order]
    paired = paired.set_index("model_label").loc[model_order].reset_index()
    y = np.arange(len(paired))[::-1]
    dff = 100 * paired.DFF.to_numpy(float)
    fused = 100 * paired.DFF_RC.to_numpy(float)
    for yi, left, right in zip(y, dff, fused):
        ax_gain.plot([left, right], [yi, yi], color="#A7ACB5", lw=1.2, zorder=1)
    ax_gain.scatter(dff, y, color=SHAM, s=25, label="DFF only", zorder=3, edgecolor="white", linewidth=0.5)
    ax_gain.scatter(fused, y, color=RC_COLOR, s=25, label="DFF + RC", zorder=4, edgecolor="white", linewidth=0.5)
    for yi, delta, right in zip(y, paired.gain_percentage_points, fused):
        ax_gain.text(max(right, dff[len(y)-1-yi]) + 0.22, yi, f"{delta:+.1f} pp", va="center",
                     fontsize=6.2, color=GAIN if delta > 0 else NEUTRAL_DARK)
    short_labels = {"Gradient boosting": "GB", "Prespecified all RC": "All RC (fixed)", "Optimized amp+slope RC": "Amp+slope RC"}
    ax_gain.set_yticks(y, [short_labels.get(name, name) for name in model_order])
    ax_gain.set_xlabel("Accuracy (%)")
    xmin = min(float(np.min(dff)), float(np.min(fused))) - 1.0
    xmax = max(float(np.max(dff)), float(np.max(fused))) + 1.5
    ax_gain.set_xlim(xmin, xmax)
    ax_gain.set_title("Effect of adding RC", loc="left", pad=5, fontweight="bold")
    ax_gain.legend(loc="center right", fontsize=6.1, handletextpad=0.35)
    panel_label(ax_gain, "e", x=-0.20)

    fig.text(0.50, 0.485, "Fixed 10-s window-level random 70/30 validation  |  n = 72 validation windows",
             ha="center", va="center", color=NEUTRAL_DARK, fontsize=6.5)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.96, bottom=0.09)
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    SRC.mkdir(exist_ok=True)
    records = all_complete_windows()
    selected, z_min, z_max = representative_records(records)
    encoding, z_min, z_max = encode_records(selected, z_min, z_max)
    ablation, paired, cm, cm_norm = performance_tables()

    signal_rows = []
    time = np.linspace(0, 10, 300, endpoint=False)
    for label, record in selected.items():
        for t, value in zip(time, record["signal"]):
            signal_rows.append({"model_type": label, "biological_id": record["biological_id"], "window": record["window"], "time_s": t, "dff": value})
    pd.DataFrame(signal_rows).to_csv(SRC / "panel_a_representative_signals.csv", index=False, encoding="utf-8-sig")
    encoding.assign(quantizer_z_min=z_min, quantizer_z_max=z_max).to_csv(SRC / "panel_b_5bit_and_rc_mapping.csv", index=False, encoding="utf-8-sig")
    distribution_rows = []
    for record in records:
        value = float(np.mean(record["signal"]) + np.mean(np.abs(np.diff(record["signal"]))))
        q = int(np.clip(np.rint(31.0 * (value - z_min) / (z_max - z_min)), 0, 31))
        distribution_rows.append({"source": record["source"], "model_type": record["model_type"], "biological_id": record["biological_id"], "window": record["window"], "q": q, "code": f"{q:05b}"})
    pd.DataFrame(distribution_rows).to_csv(SRC / "panel_b_code_distribution.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(SRC / "panel_c_feature_ablation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm, index=["true_sham", "true_sni"], columns=["pred_sham", "pred_sni"]).to_csv(SRC / "panel_d_confusion_counts.csv", encoding="utf-8-sig")
    pd.DataFrame(cm_norm, index=["true_sham", "true_sni"], columns=["pred_sham", "pred_sni"]).to_csv(SRC / "panel_d_confusion_row_fraction.csv", encoding="utf-8-sig")
    paired.to_csv(SRC / "panel_e_rc_gain_by_model.csv", index=False, encoding="utf-8-sig")

    fig = draw_figure(selected, encoding, ablation, paired, cm_norm)
    base = OUT / "Fig_main_sham_sni_10s_memristor_RC"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    legend = """Fig. X | Memristor reservoir encoding supports window-level discrimination of SHAM and SNI pain-state signals. a, Representative complete 10-s DFF signals from SHAM and SNI animals within the same acquisition source. b, The scalar mean DFF plus mean absolute change was quantized to one of 32 levels, converted to a 5-bit input, and mapped through the four measured RC.xlsx response channels; channel-normalized values are shown for visualization. c, Gradient-boosting feature ablation in the fixed 10-s window-level random 70/30 split (n = 167 training and 72 validation windows). No error bars are shown because this panel reports one prespecified split. d, Row-normalized confusion heatmap for the best 10-s DFF + RC gradient-boosting classifier (accuracy, 94.44%; balanced accuracy, 95.45%); cell counts are intentionally omitted. e, Paired DFF-only and DFF + RC accuracies across three classifiers under the same split. Adding RC improved ExtraTrees by 1.39 percentage points and did not change the other two classifiers. This window-level analysis permits windows from the same animal in training and validation and therefore does not estimate performance on unseen animals. Source data are provided with the figure.\n"""
    (OUT / "figure_legend.txt").write_text(legend, encoding="utf-8")
    qa_notes = """Figure QA notes
- Core conclusion: 10-s pain-state DFF windows can be converted into 5-bit inputs and measured memristor RC states for SHAM/SNI window-level classification.
- Archetype: asymmetric mixed-modality figure.
- Backend: Python/matplotlib only.
- Final canvas: 7.15 x 5.7 inches (approximately 182 x 145 mm).
- Validation: fixed window-level random 70/30 split; 167 training and 72 validation windows; one seed (20260706).
- Representative-signal control: SHAM and SNI examples come from source 3. Codes q=3 and q=5 were common, non-extreme codes; the medoid signal within each code-specific class subset was selected.
- Confusion heatmap: row normalized; cell numbers intentionally omitted; raw counts retained in source data.
- Statistics: no uncertainty interval is available for the single fixed split, so no error bars or inferential test are shown.
- Integrity: the figure states that RC gain is model-dependent; it does not claim a gradient-boosting gain where none exists.
- Generalization limitation: windows from the same animal may occur in both training and validation; this is not unseen-animal performance.
"""
    (OUT / "QA_notes.txt").write_text(qa_notes, encoding="utf-8")
    print(base)
    print(encoding[["model_type", "biological_id", "q", "code"]].to_string(index=False))
    print(ablation[["feature_label", "validation_accuracy"]].to_string(index=False))
    print(paired[["model_label", "DFF", "DFF_RC", "gain_percentage_points"]].to_string(index=False))


if __name__ == "__main__":
    main()
