"""
Implementation of the BASE research paper (Zdravevski et al., 2026) + the NEW multi-severity drift extension.

  python run_reference_paper.py --numeric Data/train_numeric.csv --date Data/train_date.csv \
         --categorical Data/train_categorical.csv --results ../Results/Reference_Paper [--nrows N] [--no_extension]

Steps: station/line/path numeric+date aggregation -> out-of-fold WoE nominal compression (16 features) ->
80/20 stratified split -> remove constant/duplicate features -> oversample training set -> benchmark models ->
(extension) SHAP top-20 on XGBoost, then multi-severity drift on those engineered features.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from reference_features import build_reference_features
from reference_benchmark import drop_constant_and_duplicate, evaluate_all

SEED = 42


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--numeric", required=True); ap.add_argument("--date", required=True); ap.add_argument("--categorical", required=True)
    ap.add_argument("--results", default="../Results/Reference_Paper")
    ap.add_argument("--nrows", type=int, default=None, help="quick test on the first N rows (not the paper setting)")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names to train")
    ap.add_argument("--no_extension", action="store_true"); ap.add_argument("--shap_sample", type=int, default=20_000)
    a = ap.parse_args()
    for p in (a.numeric, a.date, a.categorical):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: file not found: {p}\nDownload train_numeric/date/categorical.csv from the Kaggle Bosch competition.")
    tables, graphs, exp = (os.path.join(a.results, d) for d in ("Tables", "Graphs", "Experimental_Results"))
    for d in (tables, graphs, exp): os.makedirs(d, exist_ok=True)

    X, y, train_mask = build_reference_features(a.numeric, a.date, a.categorical, a.nrows, seed=SEED)
    print(f"engineered features: {X.shape[1]}  parts: {len(y):,}  defect rate: {y.mean():.4%}")
    Xtr, Xte = X[train_mask].reset_index(drop=True), X[~train_mask].reset_index(drop=True)
    ytr, yte = y[train_mask], y[~train_mask]
    Xtr, Xte = drop_constant_and_duplicate(Xtr, Xte)
    print(f"final input features after removing constant/duplicate columns: {Xtr.shape[1]}")
    json.dump({"engineered_features": int(X.shape[1]), "final_features": int(Xtr.shape[1]), "train_rows": int(len(ytr)),
               "test_rows": int(len(yte)), "defect_rate": float(y.mean()), "feature_names": list(Xtr.columns)},
              open(os.path.join(exp, "reference_run_metadata.json"), "w"), indent=2)

    table, imp, fitted = evaluate_all(Xtr, ytr, Xte, yte, a.models)
    table.to_csv(os.path.join(tables, "reference_model_comparison.csv")); print(table.round(4))

    if not a.no_extension and "XGBoost" in fitted:
        import model_training, evaluation, explainability, multi_severity_drift
        # validation split carved from the 80% training part (needed for thresholds / retraining in the extension)
        Xa, Xv, ya, yv = train_test_split(Xtr, pd.Series(ytr), test_size=0.25, stratify=ytr, random_state=SEED)
        Xa, Xv = Xa.reset_index(drop=True), Xv.reset_index(drop=True); ya, yv = ya.reset_index(drop=True), yv.reset_index(drop=True)
        yte_s = pd.Series(yte)
        boost = model_training.make_boost(ya); boost.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        thr = evaluation.pick_threshold(yv, boost.predict_proba(Xv)[:, 1])
        top = explainability.shap_analysis(boost, Xa, tables, graphs, a.shap_sample, 20)
        multi_severity_drift.run_extension(boost, thr, Xa, Xv, Xte, ya, yv, yte_s, list(top.index), a.results)
    print("Done:", os.path.abspath(a.results))


if __name__ == "__main__":
    main()
