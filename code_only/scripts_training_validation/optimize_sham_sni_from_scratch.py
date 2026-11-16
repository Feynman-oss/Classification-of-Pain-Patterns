from __future__ import annotations
import pandas as pd

def add_more_features(data: pd.DataFrame) -> pd.DataFrame:
    out=data.copy()
    if "mean" in out: out["mean_abs"]=out["mean"].abs()
    if "std" in out: out["cv"]=out["std"]/(out["mean"].abs()+1e-12)
    return out.fillna(0)

def aggregate_samples(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    group=[c for c in ["sample_id","pair_id","target","model_type","source"] if c in data.columns]
    if not group: group=["sample_id"] if "sample_id" in data else []
    out=data.groupby(group)[features].agg(["mean","std","min","max","median"])
    out.columns=[f"{a}__{b}" for a,b in out.columns]
    return out.reset_index().fillna(0)
