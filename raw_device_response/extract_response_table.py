from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

def extract(path: Path, polarity: int):
    df = pd.read_csv(path)
    required = {"CH1 Source", "CH1 Current", "CH1 Time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    probe = df.loc[(df["CH1 Source"] - polarity * 1.0).abs() < 1e-9]
    read = df.loc[(df["CH1 Source"] - polarity * 0.1).abs() < 1e-9]
    if probe.empty or read.empty:
        raise ValueError(f"{path}: probe/read segment not found")
    peak_current = float(probe.iloc[-1]["CH1 Current"])
    read_current = float(read.iloc[0]["CH1 Current"])
    return read_current, peak_current

rows = [{"code":"00000", "hrs_read":0.0, "hrs_peak":0.0, "lrs_read":0.0, "lrs_peak":0.0}]
for number in range(1, 32):
    code = f"{number:05b}"
    hrs_read, hrs_peak = extract(ROOT / "hrs" / f"{code}.csv", 1)
    lrs_read, lrs_peak = extract(ROOT / "lrs" / f"{code}.csv", -1)
    rows.append({"code":code, "hrs_read":hrs_read, "hrs_peak":hrs_peak,
                 "lrs_read":lrs_read, "lrs_peak":lrs_peak})

out = pd.DataFrame(rows)
out.to_csv(ROOT / "device_response_table_reconstructed.csv", index=False)
print(out.to_string(index=False))
