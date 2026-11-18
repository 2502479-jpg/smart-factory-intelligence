"""Dataset loading and auditing for the Bosch Production Line Performance numeric training file."""
import json, os
import numpy as np
import pandas as pd
from config import EXPECTED

SETUP_HELP = ("Download train_numeric.csv from "
              "https://www.kaggle.com/competitions/bosch-production-line-performance/data "
              "(unzip it) and pass its path with --data, e.g. --data Data/train_numeric.csv")


def load_bosch(path, nrows=None):
    """Return (X, y). X = numeric predictors (float32), y = 'Response' (0/1)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}\n{SETUP_HELP}")
    df = pd.read_csv(path, nrows=nrows, dtype=np.float32)
    missing = [c for c in ("Id", "Response") if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns {missing} not found. Is this train_numeric.csv?\n{SETUP_HELP}")
    y = df["Response"].astype(int)
    X = df.drop(columns=["Id", "Response"])
    return X, y


def audit_dataset(X, y, out_dir, full_run=True):
    """Print/save dimensions, class distribution and missingness; warn if they differ from the paper."""
    n_def = int(y.sum())
    info = {
        "records": int(len(X)), "columns": int(X.shape[1] + 2),
        "identifier_columns": 1, "target_column": "Response",
        "numeric_predictors": int(X.shape[1]),
        "defective": n_def, "non_defective": int(len(y) - n_def),
        "defect_prevalence_percent": round(100 * n_def / len(y), 4),
        "missing_numeric_cells": int(X.isna().sum().sum()),
    }
    print("Dataset audit:", json.dumps(info, indent=2))
    if full_run:
        for key, ek in (("records", "records"), ("columns", "columns"),
                        ("defective", "defective"), ("non_defective", "non_defective")):
            if info[key] != EXPECTED[ek]:
                print(f"WARNING: {key}={info[key]:,} but the paper documents {EXPECTED[ek]:,}.")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dataset_audit.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info
