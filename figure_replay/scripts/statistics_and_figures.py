"""Recompute paired statistics and regenerate Fig. 6f, 6g, S12 and S13."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve

SEED = 20261110


def bootstrap_pair(table: pd.DataFrame, n=10000):
    wide = table.pivot(index="event_id", columns="feature_set", values=["target", "predicted_model", "probability_sni"])
    ids = wide.index.to_numpy(); rng = np.random.default_rng(SEED); rows=[]
    target = wide[("target","DFF_only")].to_numpy(int)
    dff = (wide[("predicted_model","DFF_only")].to_numpy() == "sni").astype(int)
    fused = (wide[("predicted_model","DFF_plus_RC")].to_numpy() == "sni").astype(int)
    for _ in range(n):
        take=rng.integers(0,len(ids),len(ids)); y=target[take]; a=dff[take]; b=fused[take]
        rows.append((accuracy_score(y,a),accuracy_score(y,b),accuracy_score(y,b)-accuracy_score(y,a),
                     balanced_accuracy_score(y,a),balanced_accuracy_score(y,b)))
    b=pd.DataFrame(rows,columns=["dff_accuracy","fused_accuracy","paired_accuracy_gain","dff_balanced_accuracy","fused_balanced_accuracy"])
    return b.quantile([.025,.5,.975]).rename_axis("quantile").reset_index(), target, dff, fused


def save_figures(data: Path, out: Path):
    out.mkdir(parents=True,exist_ok=True)
    # Fig. 6f: reported fixed-split values.
    f=pd.read_csv(data/"fig6f_reported_accuracy.csv")
    fig,ax=plt.subplots(figsize=(4.2,3.2)); ax.bar(f.feature_set,f.accuracy*100,color=["#6c5ce7","#4c78a8","#2a9d8f"])
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0,100); ax.set_title("Fixed 10-s window validation")
    for i,v in enumerate(f.accuracy*100): ax.text(i,v+1,f"{v:.2f}",ha="center",fontsize=8)
    fig.tight_layout(); fig.savefig(out/"Fig6f_reproduced.png",dpi=300); plt.close(fig)

    pred=pd.read_csv(data/"nested_grouped_oof_predictions.csv")
    met=pd.read_csv(data/"nested_grouped_metrics.csv")
    order=["RC_only","DFF_only","DFF_plus_RC"]; met=met.set_index("feature_set").loc[order].reset_index()
    fig,ax=plt.subplots(figsize=(4.2,3.2)); ax.bar(met.feature_set,met.accuracy*100,color=["#6c5ce7","#4c78a8","#2a9d8f"])
    ax.set_ylabel("Event-disjoint accuracy (%)"); ax.set_ylim(0,100); ax.tick_params(axis='x',rotation=15)
    for i,v in enumerate(met.accuracy*100): ax.text(i,v+1,f"{v:.2f}",ha="center",fontsize=8)
    fig.tight_layout(); fig.savefig(out/"Fig6g_reproduced.png",dpi=300); plt.close(fig)

    # ROC curves from pooled out-of-fold probabilities.
    fig,ax=plt.subplots(figsize=(4,3.5))
    for name,color in [("DFF_only","#4c78a8"),("DFF_plus_RC","#2a9d8f")]:
        s=pred[pred.feature_set==name]; fpr,tpr,_=roc_curve(s.target,s.probability_sni)
        ax.plot(fpr,tpr,label=f"{name} (AUC={roc_auc_score(s.target,s.probability_sni):.3f})",color=color)
    ax.plot([0,1],[0,1],'--',color='0.6'); ax.set(xlabel="False-positive rate",ylabel="True-positive rate")
    ax.legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(out/"ROC_reproduced.png",dpi=300); plt.close(fig)

    # Fig. S13 confusion matrices.
    fig,axes=plt.subplots(1,2,figsize=(7,3))
    for ax,name in zip(axes,["DFF_only","DFF_plus_RC"]):
        s=pred[pred.feature_set==name]; yhat=(s.predicted_model=="sni").astype(int); cm=confusion_matrix(s.target,yhat,labels=[0,1]); norm=cm/cm.sum(axis=1,keepdims=True)
        ax.imshow(norm,vmin=0,vmax=1,cmap="Purples"); ax.set_title(name); ax.set_xticks([0,1],["Sham","SNI"]); ax.set_yticks([0,1],["Sham","SNI"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(2):
            for j in range(2): ax.text(j,i,f"{cm[i,j]}\n({norm[i,j]:.3f})",ha="center",va="center",color="white" if norm[i,j]>.55 else "black")
    fig.tight_layout(); fig.savefig(out/"FigS13_reproduced.png",dpi=300); plt.close(fig)

    # Fig. S12, two 9x9 panels.
    s=pd.read_csv(data/"figs12_81_configurations.csv"); groups=list(dict.fromkeys(s.rc_group)); settings=s[["min_samples_leaf","max_features"]].drop_duplicates().copy(); settings["label"]=settings.apply(lambda r:f"leaf={r.min_samples_leaf}\nmax={r.max_features}",axis=1)
    s["setting"]=s.apply(lambda r:f"leaf={r.min_samples_leaf}\nmax={r.max_features}",axis=1)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,value,title in zip(axes,["fusion_validation_accuracy","validation_gain"],["Fused accuracy","Accuracy gain over DFF"]):
        mat=s.pivot(index="rc_group",columns="setting",values=value).reindex(index=groups,columns=settings.label)
        im=ax.imshow(mat*100,aspect="auto",cmap="viridis"); ax.set_xticks(range(len(mat.columns)),mat.columns,rotation=45,ha="right",fontsize=7); ax.set_yticks(range(len(mat.index)),mat.index,fontsize=8); ax.set_title(title); fig.colorbar(im,ax=ax,label="Percentage points")
    fig.tight_layout(); fig.savefig(out/"FigS12_reproduced.png",dpi=300); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--bootstrap",type=int,default=10000)
    a=ap.parse_args(); data=Path(a.data_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    pred=pd.read_csv(data/"nested_grouped_oof_predictions.csv")
    ci,y,dff,fused=bootstrap_pair(pred,a.bootstrap); ci.to_csv(out/"paired_event_bootstrap_ci.csv",index=False)
    dff_ok=dff==y; fused_ok=fused==y; b=int((dff_ok & ~fused_ok).sum()); c=int((~dff_ok & fused_ok).sum())
    mcnemar=pd.DataFrame([{"dff_correct_fused_wrong":b,"dff_wrong_fused_correct":c,"exact_two_sided_p":binomtest(min(b,c),b+c,.5).pvalue if b+c else 1.0}])
    mcnemar.to_csv(out/"mcnemar_exact.csv",index=False)
    save_figures(data,out); print(ci.to_string(index=False)); print(mcnemar.to_string(index=False))


if __name__=="__main__": main()
