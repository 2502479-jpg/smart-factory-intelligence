"""KS / PSI drift statistics and the ORIGINAL controlled shift (+1 training std on non-missing values)."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from config import SEED
from evaluation import evaluate


def psi(reference, current, bins=10, eps=1e-6):
    """Population Stability Index with bin edges taken from reference-distribution quantiles."""
    ref, cur = reference[~np.isnan(reference)], current[~np.isnan(current)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift_table(X_reference, X_current, features, ref_n=100_000):
    """KS statistic, KS p-value (may underflow to 0.0 -> read as 'extremely small') and PSI per feature."""
    ref = X_reference.sample(n=min(ref_n, len(X_reference)), random_state=SEED)
    rows = []
    for f in features:
        a, b = ref[f].dropna().values, X_current[f].dropna().values
        stat, p = ks_2samp(a, b)
        rows.append({"feature": f, "ks_statistic": float(stat), "ks_pvalue": float(p), "psi": psi(a, b)})
    return pd.DataFrame(rows)


def shift_constant(X, features, deltas):
    """ORIGINAL experiment: add a constant (here 1 training std) to non-missing values (NaN stays NaN)."""
    out = X.copy()
    out[features] = out[features] + deltas
    return out


def original_shift_experiment(boost, threshold, Xtr, Xte, yte, p_test_original, features, std_multiplier=1.0):
    """Evaluate the original Boost model on the shifted test set + feature-level KS/PSI."""
    deltas = std_multiplier * Xtr[features].std(skipna=True).values
    Xte_shift = shift_constant(Xte, features, deltas)
    p_shift = boost.predict_proba(Xte_shift)[:, 1]
    table = pd.DataFrame({"original_test": evaluate(yte, p_test_original, threshold),
                          "shifted_test": evaluate(yte, p_shift, threshold)})
    drift = feature_drift_table(Xtr, Xte_shift, features).sort_values("psi", ascending=False)
    return table, drift, Xte_shift, deltas, p_shift
