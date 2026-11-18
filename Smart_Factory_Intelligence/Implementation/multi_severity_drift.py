"""
EXTENSION (not part of the original paper):
Multi-Severity Feature Drift Analysis and Model Adaptation.

Location-scale drift applied to the top-20 SHAP-ranked features:
    shifted = ref_mean + scale * (x - ref_mean) + location * ref_std
Reference mean/std come from the TRAINING partition (NaN-aware) and are used for every condition.
Only non-missing values are transformed (NaN stays NaN).

For each condition we record KS and PSI (per feature) and Precision/Recall/F1/ROC-AUC/PR-AUC/MCC for
  * the ORIGINAL Boost model (trained once on unshifted data, original validation threshold), and
  * a RETRAINED Boost model (trained on the equally shifted training data, threshold re-picked on the
    shifted validation data).
The "No Drift" condition is the identity transform, so the retrained model equals the original there
and is not retrained (documented, saves compute).

Standalone usage (needs the saved model + SHAP table written by main.py):
    python multi_severity_drift.py --data Data/train_numeric.csv --results ../Results
"""
import argparse, json, os
import numpy as np
import pandas as pd
from config import SEVERITY_CONDITIONS, TOP_K_FEATURES, METRICS
from evaluation import evaluate, pick_threshold
from drift_detection import feature_drift_table
from model_training import make_boost
import plotting


def location_scale(X, features, ref_mean, ref_std, scale, location):
    out = X.copy()
    ref_mean, ref_std = np.asarray(ref_mean), np.asarray(ref_std)
    out[features] = ref_mean + scale * (out[features] - ref_mean) + location * ref_std   # NaN propagates
    return out


def run_extension(boost, thr_original, Xtr, Xva, Xte, ytr, yva, yte, features, results_dir):
    tables = os.path.join(results_dir, "Tables"); graphs = os.path.join(results_dir, "Graphs")
    exp = os.path.join(results_dir, "Experimental_Results")
    for d in (tables, graphs, exp): os.makedirs(d, exist_ok=True)

    ref_mean = Xtr[features].mean(skipna=True).values.astype("float64")
    ref_std = Xtr[features].std(skipna=True).values.astype("float64")
    rows, drift_rows = [], []
    for cond, p in SEVERITY_CONDITIONS.items():
        print(f"[extension] condition: {cond} (scale={p['scale']}, location={p['location']})")
        Xte_c = location_scale(Xte, features, ref_mean, ref_std, p["scale"], p["location"])
        # drift statistics versus the (unshifted) training reference
        dt = feature_drift_table(Xtr, Xte_c, features)
        dt.insert(0, "condition", cond); drift_rows.append(dt)
        # original model
        p_orig = boost.predict_proba(Xte_c)[:, 1]
        rows.append({"condition": cond, "model": "original", **p, **evaluate(yte, p_orig, thr_original)})
        # retrained model
        if p["scale"] == 1.0 and p["location"] == 0.0:
            rows.append({"condition": cond, "model": "retrained", **p, **evaluate(yte, p_orig, thr_original),
                         "note": "identity transform: retrained == original (not retrained)"})
            continue
        Xtr_c = location_scale(Xtr, features, ref_mean, ref_std, p["scale"], p["location"])
        Xva_c = location_scale(Xva, features, ref_mean, ref_std, p["scale"], p["location"])
        model = make_boost(ytr)
        model.fit(Xtr_c, ytr, eval_set=[(Xva_c, yva)], verbose=False)
        thr = pick_threshold(yva, model.predict_proba(Xva_c)[:, 1])
        rows.append({"condition": cond, "model": "retrained", **p, **evaluate(yte, model.predict_proba(Xte_c)[:, 1], thr)})

    res = pd.DataFrame(rows); stats = pd.concat(drift_rows, ignore_index=True)
    res.to_csv(os.path.join(tables, "multi_severity_performance.csv"), index=False)
    stats.to_csv(os.path.join(tables, "multi_severity_feature_drift.csv"), index=False)
    summ = stats.groupby("condition", sort=False).agg(mean_ks=("ks_statistic", "mean"), mean_psi=("psi", "mean"),
                                                       median_psi=("psi", "median"),
                                                       features_psi_gt_0_25=("psi", lambda s: int((s > .25).sum()))).reset_index()
    summ.to_csv(os.path.join(tables, "multi_severity_drift_summary.csv"), index=False)
    with open(os.path.join(exp, "multi_severity_settings.json"), "w") as f:
        json.dump({"features": list(features), "conditions": SEVERITY_CONDITIONS,
                   "reference_mean": dict(zip(features, ref_mean)), "reference_std": dict(zip(features, ref_std))}, f, indent=2)

    ext = os.path.join(graphs, "Extension_Multi_Severity"); os.makedirs(ext, exist_ok=True)
    for i, m in enumerate(["recall", "f1", "pr_auc", "roc_auc", "mcc"], 1):
        plotting.plot_severity_metric(res, m, os.path.join(ext, f"Ext{i}_severity_vs_{m}.png"))
    plotting.plot_psi_by_severity(stats, os.path.join(ext, "Ext6_psi_by_severity.png"))
    plotting.plot_severity_overview(res, os.path.join(ext, "Ext7_original_vs_retrained_by_severity.png"))
    return res, stats


def _standalone():
    import xgboost as xgb
    from data_loader import load_bosch
    from preprocessing import select_features, split_data
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--results", default="../Results")
    ap.add_argument("--nrows", type=int, default=None)
    a = ap.parse_args()
    exp = os.path.join(a.results, "Experimental_Results")
    mp, meta = os.path.join(exp, "boost_model.json"), os.path.join(exp, "run_metadata.json")
    shap_csv = os.path.join(a.results, "Tables", f"shap_top{TOP_K_FEATURES}.csv")
    for p in (mp, meta, shap_csv):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} not found - run main.py first (it saves the model, metadata and SHAP table).")
    boost = xgb.XGBClassifier(); boost.load_model(mp)
    thr = json.load(open(meta))["boost_threshold"]
    features = list(pd.read_csv(shap_csv, index_col=0).index)
    X, y = load_bosch(a.data, a.nrows)
    X = select_features(X)
    Xtr, Xva, Xte, ytr, yva, yte = split_data(X, y, full_run=a.nrows is None)
    run_extension(boost, thr, Xtr, Xva, Xte, ytr, yva, yte, features, a.results)


if __name__ == "__main__":
    _standalone()
