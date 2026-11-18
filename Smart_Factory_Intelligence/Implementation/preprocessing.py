"""Feature selection, stratified partitioning and imputation."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from config import SEED, MAX_MISSING_FRACTION, EXPECTED


def select_features(X, max_missing=MAX_MISSING_FRACTION):
    """Keep numeric features with at most `max_missing` fraction of missing values.
    Missing values are NOT globally replaced by zero (Boost handles NaN natively;
    Decision Tree / Random Forest use median imputation fitted on their training sample)."""
    keep = X.columns[X.isna().mean() <= max_missing]
    print(f"Selected {len(keep)} of {X.shape[1]} numeric features (<= {max_missing:.0%} missing).")
    return X[keep]


def split_data(X, y, full_run=True):
    """Stratified 70/15/15 split, random_state=42.
    Two-step split: 30% held out, then halved. On the full Bosch file this yields exactly
    828,622 / 177,562 / 177,563 rows, matching the paper."""
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=SEED)
    print(f"Split sizes  train={len(ytr):,}  validation={len(yva):,}  test={len(yte):,}")
    if full_run and (len(ytr), len(yva), len(yte)) != (EXPECTED["train_rows"], EXPECTED["val_rows"], EXPECTED["test_rows"]):
        print("WARNING: split sizes differ from the paper (828,622 / 177,562 / 177,563).")
    return Xtr, Xva, Xte, ytr, yva, yte


def fit_median_imputer(X_sample):
    return SimpleImputer(strategy="median").fit(X_sample)
