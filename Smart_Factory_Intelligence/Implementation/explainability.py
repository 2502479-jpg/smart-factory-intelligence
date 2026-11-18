"""SHAP TreeExplainer for the Boost model."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import SEED, TOP_K_FEATURES


def shap_analysis(model, Xtr, out_tables, out_graphs, n_sample=20_000, top_k=TOP_K_FEATURES):
    """Compute SHAP on a random sample of training rows (the paper does not state the size; default 20,000).
    Saves: mean-|SHAP| table (all features + top-k) and a SHAP summary plot. Returns the top-k Series."""
    import shap  # imported lazily so that other modules work without it
    sample = Xtr.sample(n=min(n_sample, len(Xtr)), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xtr.columns, name="mean_abs_shap").sort_values(ascending=False)
    imp.to_csv(os.path.join(out_tables, "shap_all_features.csv"))
    top = imp.head(top_k)
    top.to_csv(os.path.join(out_tables, f"shap_top{top_k}.csv"))
    plt.figure()
    shap.summary_plot(sv, sample, max_display=top_k, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_graphs, "shap_summary_plot.png"), dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"SHAP computed on {len(sample):,} training rows; top feature: {top.index[0]}")
    return top
