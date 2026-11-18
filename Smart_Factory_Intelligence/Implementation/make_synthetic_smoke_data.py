"""Create a SYNTHETIC Bosch-like CSV for smoke-testing the code only. Results from it mean nothing scientifically.
    python make_synthetic_smoke_data.py synthetic_train_numeric.csv
"""
import sys
import numpy as np, pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_train_numeric.csv"
rng = np.random.default_rng(0); n, f = 30_000, 80
X = rng.normal(size=(n, f)).astype("float32")
y = (rng.random(n) < (0.006 + 0.02 * (X[:, 0] > 2))).astype(int)
X[:, 3] += y * 1.5
X[rng.random((n, f)) < np.linspace(0.1, 0.9, f)] = np.nan
df = pd.DataFrame(X, columns=[f"L{i % 4}_S{i // 10}_F{i}" for i in range(f)])
df.insert(0, "Id", np.arange(n)); df["Response"] = y
df.to_csv(out, index=False); print("wrote", out, "defective:", int(y.sum()))
