"""Benchmark of the BASE paper: random oversampling of the training set, model comparison, Table-4-style metrics."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score, average_precision_score,
                             matthews_corrcoef, accuracy_score)

SEED = 42


def drop_constant_and_duplicate(Xtr, *others):
    keep = [c for c in Xtr.columns if Xtr[c].nunique(dropna=False) > 1]
    Xtr = Xtr[keep]
    Xtr = Xtr.loc[:, ~Xtr.T.duplicated()]      # duplicated columns (training split)
    cols = list(Xtr.columns)
    return (Xtr, *[o[cols] for o in others])


def random_oversample(X, y, seed=SEED):
    """Replicate minority (defect) rows in the TRAINING set until classes are balanced; test set untouched."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    extra = rng.choice(pos, size=max(len(neg) - len(pos), 0), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra]); rng.shuffle(idx)
    return X[idx], y[idx]


def models():
    """Hyper-parameters from the base paper's Table 3 where scikit-learn has an equivalent."""
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, max_features="log2", random_state=SEED, n_jobs=-1),
        # INTERP: Spark GBT -> sklearn HistGradientBoosting (much faster on >1M rows), same depth/learning rate
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=10, learning_rate=0.2, random_state=SEED),
        "XGBoost": xgb.XGBClassifier(max_depth=15, learning_rate=0.3, min_child_weight=5, colsample_bytree=0.8,
                                     subsample=0.8, tree_method="hist", random_state=SEED, n_jobs=-1),
        # INTERP: Spark regParam 0.01 ~ sklearn C = 1/0.01
        "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=100.0, random_state=SEED, max_iter=5000)),   # INTERP: scaling added
        "Multilayer Perceptron": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), solver="sgd",
                                               learning_rate_init=0.1, max_iter=50, random_state=SEED)),   # INTERP: scaling added
    }   # Factorization Machines and TabNet are NOT implemented here (no scikit-learn / extra deep-learning dependency)


def _score(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def evaluate_all(Xtr, ytr, Xte, yte, selected=None):
    imp = SimpleImputer(strategy="mean").fit(Xtr)            # paper: mean imputation fitted on training data
    A, B = imp.transform(Xtr), imp.transform(Xte)
    A, ya = random_oversample(A, ytr.values if hasattr(ytr, "values") else ytr)
    rows, fitted = {}, {}
    for name, m in models().items():
        if selected and name not in selected: continue
        print("Training", name, "...")
        m.fit(A, ya)
        s = _score(m, B)
        pred = m.predict(B)
        # Spark's 'weighted' metrics (likely what the base paper's Table 4 reports: recall ~ accuracy) AND minority-class metrics
        pw, rw, fw, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
        p1, r1, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        rows[name] = {"precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
                      "precision_defect": p1, "recall_defect": r1, "f1_defect": f1,
                      "accuracy": accuracy_score(yte, pred), "auc_roc": roc_auc_score(yte, s),
                      "pr_auc": average_precision_score(yte, s), "mcc": matthews_corrcoef(yte, pred)}
        fitted[name] = m
    return pd.DataFrame(rows).T, imp, fitted
