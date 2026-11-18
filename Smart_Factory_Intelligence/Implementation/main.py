"""
Run the complete study:  python main.py --data <path to train_numeric.csv> --results ../Results

Stages: dataset audit -> feature selection -> split -> baseline models -> SHAP -> original drift & retraining
        -> NEW multi-severity extension. All tables/graphs are written from the actual computed results.
Full Bosch run needs roughly 16 GB+ RAM and can take hours; use --nrows N for a quick (non-reproducing) test.
"""
import argparse, json, os, sys
import pandas as pd
from config import TOP_K_FEATURES, SEED, RECORDED_THRESHOLDS
from data_loader import load_bosch, audit_dataset
from preprocessing import select_features, split_data
from model_training import train_baseline
from evaluation import save_table
from explainability import shap_analysis
from drift_detection import original_shift_experiment
from retraining import retraining_experiment
from multi_severity_drift import run_extension
import plotting


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to Bosch train_numeric.csv")
    ap.add_argument("--results", default="../Results", help="output folder (Graphs/, Tables/, Experimental_Results/)")
    ap.add_argument("--nrows", type=int, default=None, help="read only the first N rows (quick test; not the paper setting)")
    ap.add_argument("--shap_sample", type=int, default=20_000, help="training rows used for SHAP (paper does not state this)")
    ap.add_argument("--dt_rf_sample", type=int, default=250_000)
    ap.add_argument("--use_recorded_thresholds", action="store_true", help="use the paper's thresholds instead of validation-F1 thresholds")
    ap.add_argument("--skip_extension", action="store_true", help="run only the original paper implementation")
    ap.add_argument("--top_k", type=int, default=TOP_K_FEATURES)
    a = ap.parse_args()

    full = a.nrows is None
    tables, graphs = os.path.join(a.results, "Tables"), os.path.join(a.results, "Graphs")
    exp = os.path.join(a.results, "Experimental_Results")
    orig_g = os.path.join(graphs, "Original_Study")
    for d in (tables, graphs, exp, orig_g): os.makedirs(d, exist_ok=True)
    if not full:
        print(f"NOTE: --nrows {a.nrows} given: results will NOT reproduce the paper.")

    X, y = load_bosch(a.data, a.nrows)
    audit_dataset(X, y, exp, full_run=full)
    X = select_features(X)
    Xtr, Xva, Xte, ytr, yva, yte = split_data(X, y, full_run=full)

    # --- original study ---
    res, probs, boost = train_baseline(Xtr, Xva, Xte, ytr, yva, yte, a.dt_rf_sample, a.use_recorded_thresholds)
    save_table(pd.DataFrame(res).T, os.path.join(tables, "model_comparison.csv"))
    plotting.plot_model_comparison(res, os.path.join(orig_g, "model_comparison.png"))
    plotting.plot_confusion_matrices(res, orig_g)
    thr = res["Boost"]["threshold"]
    boost.save_model(os.path.join(exp, "boost_model.json"))
    json.dump({"boost_threshold": thr, "seed": SEED, "n_features": int(X.shape[1]),
               "features": list(X.columns), "full_run": full}, open(os.path.join(exp, "run_metadata.json"), "w"), indent=2)

    top = shap_analysis(boost, Xtr, tables, orig_g, a.shap_sample, a.top_k)
    plotting.plot_shap_bar(top, os.path.join(orig_g, "shap_top_features.png"))
    features = list(top.index)

    shifted_table, drift, Xte_shift, deltas, p_shift = original_shift_experiment(boost, thr, Xtr, Xte, yte, probs["Boost"], features)
    save_table(shifted_table, os.path.join(tables, "original_vs_shifted.csv"))
    drift.to_csv(os.path.join(tables, "feature_drift_original_shift.csv"), index=False)
    plotting.plot_drift(drift, os.path.join(orig_g, "feature_drift_ks_psi.png"), "Feature-Level Drift (+1 std synthetic shift)")
    plotting.plot_two_condition(shifted_table, ["Original test", "Shifted test"], ["#0072B2", "#D55E00"],
                                "Original vs Shifted Test (Boost)", os.path.join(orig_g, "original_vs_shifted.png"))

    retr, check = retraining_experiment(Xtr, Xva, ytr, yva, yte, Xte_shift, features, deltas, p_shift, thr, probs["Boost"], exp)
    save_table(retr, os.path.join(tables, "original_vs_retrained.csv"))
    plotting.plot_two_condition(retr, ["Original model", "Retrained model"], ["#D55E00", "#009E73"],
                                "Original vs Retrained Model on Shifted Test", os.path.join(orig_g, "original_vs_retrained.png"))
    print("Retraining equivalence check:", check)

    # --- NEW EXTENSION ---
    if not a.skip_extension:
        run_extension(boost, thr, Xtr, Xva, Xte, ytr, yva, yte, features, a.results)
    print("Done. Outputs in", os.path.abspath(a.results))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")
