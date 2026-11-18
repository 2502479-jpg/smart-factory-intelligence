"""Create SYNTHETIC Bosch-like train_numeric/date/categorical CSVs (aligned rows) for smoke tests only.
    python make_synthetic_bosch_files.py <out_dir>"""
import sys, os
import numpy as np, pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1); n = 6000
stations = [(l, s) for l in range(4) for s in range(l * 3, l * 3 + 3)]      # 12 stations on 4 lines
visit = rng.random((n, len(stations))) < 0.6
y = (rng.random(n) < 0.03).astype(int)
num, dat, cat = {}, {}, {}
k = 0
for j, (l, s) in enumerate(stations):
    for _ in range(3):
        k += 1
        v = rng.normal(size=n) + 0.8 * y * (j == 2); v[~visit[:, j]] = np.nan
        num[f"L{l}_S{s}_F{k}"] = v
    k += 1; t = 100 + rng.random(n) * 5 + j; t[~visit[:, j]] = np.nan; dat[f"L{l}_S{s}_D{k}"] = t
    k += 1
    codes = np.where(rng.random(n) < 0.2 + 0.6 * y * (j == 5), "T2", "T1").astype(object)
    codes[~visit[:, j]] = np.nan; cat[f"L{l}_S{s}_F{k}"] = codes
ids = np.arange(1, n + 1)
pd.concat([pd.Series(ids, name="Id"), pd.DataFrame(num), pd.Series(y, name="Response")], axis=1).to_csv(f"{out}/train_numeric.csv", index=False)
pd.concat([pd.Series(ids, name="Id"), pd.DataFrame(dat)], axis=1).to_csv(f"{out}/train_date.csv", index=False)
pd.concat([pd.Series(ids, name="Id"), pd.DataFrame(cat)], axis=1).to_csv(f"{out}/train_categorical.csv", index=False)
print("wrote synthetic files to", out, "defects:", int(y.sum()))
