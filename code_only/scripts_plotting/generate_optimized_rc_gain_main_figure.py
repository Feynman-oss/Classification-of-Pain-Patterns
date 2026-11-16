from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import generate_main_rc_sham_sni_figure as base
import optimize_train_only_rc_gain as opt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "manuscript_main_figure_rc_gain_optimized"
SRC = OUT / "source_data"


def main():
    OUT.mkdir(exist_ok=True)
    SRC.mkdir(exist_ok=True)

    records = base.all_complete_windows()
    selected, z_min, z_max = base.representative_records(records)
    encoding, _, _ = base.encode_records(selected, z_min, z_max)

    data = pd.read_csv(opt.DATA).replace([np.inf, -np.inf], np.nan).fillna(0)
    train_idx, validation_idx = train_test_split(np.arange(len(data)), test_size=0.30, random_state=opt.SEED, stratify=data.target)
    train = data.iloc[train_idx].reset_index(drop=True)
    validation = data.iloc[validation_idx].reset_index(drop=True)
    y_train = train.target.to_numpy(int)
    y_val = validation.target.to_numpy(int)
    rc = opt.rc_feature_groups(opt.rc_columns(data))["amp_slope"]
    cv = StratifiedKFold(5, shuffle=True, random_state=opt.SEED)
    rc_oof = opt.oof_probability(train[rc], y_train, (4, 0.5), cv, trees=700)
    rc_threshold = opt.choose_threshold(y_train, rc_oof)
    rc_model = opt.model((4, 0.5), trees=700).fit(train[rc], y_train)
    rc_probability = rc_model.predict_proba(validation[rc])[:, 1]
    rc_accuracy = float(np.mean((rc_probability >= rc_threshold) == y_val))

    metrics = pd.read_csv(ROOT / "train_only_rc_gain_optimization" / "final_candidate_metrics.csv")
    dff_row = metrics[metrics.method == "Exploratory optimized DFF"].iloc[0]
    fusion_row = metrics[metrics.method == "Exploratory optimized DFF + amp_slope RC"].iloc[0]
    pre_dff = metrics[metrics.method == "Prespecified DFF"].iloc[0]
    pre_fusion = metrics[metrics.method == "Prespecified DFF + all RC"].iloc[0]

    ablation = pd.DataFrame({
        "feature_label": ["RC only", "DFF only", "DFF + RC"],
        "validation_accuracy": [rc_accuracy, dff_row.accuracy, fusion_row.accuracy],
    })
    cm = np.asarray([[fusion_row.tn, fusion_row.fp], [fusion_row.fn, fusion_row.tp]], float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    paired = pd.DataFrame({
        "model_label": ["Prespecified all RC", "Optimized amp+slope RC"],
        "DFF": [pre_dff.accuracy, dff_row.accuracy],
        "DFF_RC": [pre_fusion.accuracy, fusion_row.accuracy],
    })
    paired["gain_percentage_points"] = 100 * (paired.DFF_RC - paired.DFF)

    signal_rows = []
    time = np.linspace(0, 10, 300, endpoint=False)
    for label, record in selected.items():
        for t, value in zip(time, record["signal"]):
            signal_rows.append({"model_type": label, "biological_id": record["biological_id"], "window": record["window"], "time_s": t, "dff": value})
    pd.DataFrame(signal_rows).to_csv(SRC / "panel_a_representative_signals.csv", index=False, encoding="utf-8-sig")
    encoding.assign(quantizer_z_min=z_min, quantizer_z_max=z_max).to_csv(SRC / "panel_b_5bit_and_rc_mapping.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(SRC / "panel_c_optimized_ablation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm, index=["true_sham", "true_sni"], columns=["pred_sham", "pred_sni"]).to_csv(SRC / "panel_d_optimized_confusion_counts.csv", encoding="utf-8-sig")
    paired.to_csv(SRC / "panel_e_rc_gain_comparison.csv", index=False, encoding="utf-8-sig")

    fig = base.draw_figure(selected, encoding, ablation, paired, cm_norm)
    out_base = OUT / "Fig_main_optimized_RC_gain"
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    legend = f"""Fig. X | Optimized memristor RC features improve window-level SHAM/SNI classification. a, Representative complete 10-s pain-state DFF signals from SHAM and SNI animals within the same acquisition source. b, Five-bit encoding and the corresponding four-channel measured memristor response. c, Feature ablation for the exploratory amp+slope RC ExtraTrees model in the fixed window-level 70/30 split: RC only, {100*rc_accuracy:.2f}%; DFF only, {100*dff_row.accuracy:.2f}%; and DFF + RC, {100*fusion_row.accuracy:.2f}%. d, Row-normalized confusion heatmap for DFF + RC (accuracy, {100*fusion_row.accuracy:.2f}%); cell counts are omitted. e, The prespecified all-RC model improves accuracy by {100*(pre_fusion.accuracy-pre_dff.accuracy):.2f} percentage points, whereas the exploratory amp+slope RC model improves accuracy by {100*(fusion_row.accuracy-dff_row.accuracy):.2f} percentage points. The optimized candidate was identified during a configuration sensitivity audit and requires independent confirmation. Windows from the same animal may occur in training and validation; this figure does not report unseen-animal performance. Source data are provided with the figure.\n"""
    (OUT / "figure_legend.txt").write_text(legend, encoding="utf-8")
    qa = """Figure QA notes
- Python/matplotlib-only rendering; editable SVG text and 600-dpi TIFF are exported.
- Fixed 10-s window-level random 70/30 split: 167 training and 72 validation windows.
- Prespecified comparison: DFF 91.67% versus DFF + all RC 93.06%.
- Exploratory optimized comparison: DFF 91.67% versus DFF + amp+slope RC 94.44%.
- The optimized candidate was identified after an 81-configuration sensitivity audit and is not an independent confirmatory result.
- McNemar exact p = 0.625 for the optimized gain; no statistical-significance claim should be made.
- Windows from the same animal can occur in training and validation; unseen-animal performance is not represented.
- Confusion-matrix cell numbers are omitted visually but retained in source data.
"""
    (OUT / "QA_notes.txt").write_text(qa, encoding="utf-8")
    print(out_base)
    print(ablation.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
