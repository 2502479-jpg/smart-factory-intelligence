"""Model definitions (hyper-parameters from the paper) and baseline training."""
import numpy as np
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from config import SEED, DT_RF_TRAIN_SAMPLE, RECORDED_THRESHOLDS
from evaluation import pick_threshold, evaluate
from preprocessing import fit_median_imputer


def make_decision_tree():
    return DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, class_weight="balanced", random_state=SEED)


def make_random_forest():
    return RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=10, max_features="sqrt",
                                  class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)


def make_boost(y_train):
    """XGBoost ("Boost" in the paper). scale_pos_weight = negatives/positives (~171.09 on the full train split)."""
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, objective="binary:logistic", eval_metric="aucpr",
                             tree_method="hist", scale_pos_weight=spw, random_state=SEED, n_jobs=-1)


def train_baseline(Xtr, Xva, Xte, ytr, yva, yte, sample_n=DT_RF_TRAIN_SAMPLE, use_recorded_thresholds=False):
    """Train the three classifiers. Returns (results dict, test-probability dict, Boost model)."""
    results, probs = {}, {}
    idx = Xtr.sample(n=min(sample_n, len(Xtr)), random_state=SEED).index
    Xs, ys = Xtr.loc[idx], ytr.loc[idx]
    imp = fit_median_imputer(Xs)
    Xs_i, Xva_i, Xte_i = imp.transform(Xs), imp.transform(Xva), imp.transform(Xte)

    for name, model in (("Decision Tree", make_decision_tree()), ("Random Forest", make_random_forest())):
        print(f"Training {name} on {len(ys):,} rows ...")
        model.fit(Xs_i, ys)
        thr = RECORDED_THRESHOLDS[name] if use_recorded_thresholds else pick_threshold(yva, model.predict_proba(Xva_i)[:, 1])
        p = model.predict_proba(Xte_i)[:, 1]
        results[name] = {"train_rows": int(len(ys)), **evaluate(yte, p, thr)}
        probs[name] = p

    print(f"Training Boost on {len(ytr):,} rows ...")
    boost = make_boost(ytr)
    boost.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    thr = RECORDED_THRESHOLDS["Boost"] if use_recorded_thresholds else pick_threshold(yva, boost.predict_proba(Xva)[:, 1])
    p = boost.predict_proba(Xte)[:, 1]
    results["Boost"] = {"train_rows": int(len(ytr)), **evaluate(yte, p, thr)}
    probs["Boost"] = p
    return results, probs, boost
