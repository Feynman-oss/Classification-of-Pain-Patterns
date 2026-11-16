from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd


def read_detail_trails(path: Path) -> pd.DataFrame:
    df=pd.read_excel(path,sheet_name="Detail DFF"); cols=list(df.columns); rows=[]
    for i,col in enumerate(cols):
        m=re.match(r"Trail\s+(\d+)-",str(col))
        if not m: continue
        tc=next((c for c in cols[i+1:min(i+9,len(cols))] if str(c).startswith("Timestamp(ns)")),None)
        if tc is None: continue
        x=pd.DataFrame({"value":pd.to_numeric(df[col],errors="coerce"),"time":pd.to_numeric(df[tc],errors="coerce")}).dropna().sort_values("time")
        if len(x)<3: continue
        t=(x.time.to_numpy(float)-float(x.time.iloc[0]))/1e9; v=x.value.to_numpy(float); n=int(np.floor(t[-1]))
        for w in range(n):
            q=(t>=w)&(t<w+1)
            if q.sum()<3: continue
            y=v[q]; rows.append({"file":path.name,"pair_id":path.stem,"sample_id":path.stem,"trail":int(m.group(1)),"window":w,"points":len(y),"mean":np.mean(y),"std":np.std(y,ddof=1),"min":np.min(y),"max":np.max(y),"range":np.ptp(y),"slope":y[-1]-y[0],"rate_abs_mean":np.mean(np.abs(np.diff(y))),"label":"pain","y":1})
    return pd.DataFrame(rows)
