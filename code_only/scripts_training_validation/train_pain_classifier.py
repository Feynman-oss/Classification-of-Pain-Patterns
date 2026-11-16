from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def extract_windows(path: Path, split: str = "all", label: str = "pain", window_seconds: float = 1.0, baseline_normalize: bool = False) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Overall DFF")
    values = pd.to_numeric(df["Mean"], errors="coerce")
    times = pd.to_numeric(df["Timestamp(ns)"], errors="coerce")
    x = pd.DataFrame({"value": values, "time": times}).dropna().sort_values("time")
    if x.empty:
        return pd.DataFrame()
    t = (x.time.to_numpy(float) - float(x.time.iloc[0])) / 1e9
    v = x.value.to_numpy(float)
    rows=[]
    for w in range(int(np.floor(t[-1]/window_seconds))):
        m=(t>=w*window_seconds)&(t<(w+1)*window_seconds)
        if m.sum()<3: continue
        y=v[m]; tt=t[m]; dt=np.diff(tt); dv=np.diff(y); rate=dv[dt>0]/dt[dt>0] if np.any(dt>0) else np.array([0.])
        rows.append({"file":path.name,"sample_id":path.stem,"pair_id":path.stem,"window":w,"label":label,"y":1,"points":len(y),"mean":np.mean(y),"std":np.std(y,ddof=1),"min":np.min(y),"max":np.max(y),"range":np.ptp(y),"median":np.median(y),"auc":np.trapezoid(y,dx=window_seconds/max(len(y),1)),"slope":(y[-1]-y[0])/window_seconds,"rate_mean":np.mean(rate),"rate_abs_mean":np.mean(np.abs(rate)),"rate_std":np.std(rate,ddof=1),"rate_min":np.min(rate),"rate_max":np.max(rate),"rate_range":np.ptp(rate),"rate_abs_max":np.max(np.abs(rate)),"rate_pos_ratio":np.mean(rate>0),"rate_neg_ratio":np.mean(rate<0),"peak_count":int(((y[1:-1]>y[:-2])&(y[1:-1]>y[2:])&(y[1:-1]>np.mean(y)+np.std(y,ddof=1))).sum())})
    return pd.DataFrame(rows)
