"""ORIGINAL retraining experiment: constant shift applied to train/validation/test, retrain, compare."""
import json, os
import numpy as np
import pandas as pd
from evaluation import evaluate, pick_threshold
from drift_detection import shift_constant
from model_training import make_boost


def retraining_experiment(Xtr, Xva, ytr, yva, yte, Xte_shift, features, deltas,
                          p_shift_original, thr_original, p_test_original, out_dir):
    Xtr_s, Xva_s = shift_constant(Xtr, features, deltas), shift_constant(Xva, features, deltas)
    retrained = make_boost(ytr)
    retrained.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], verbose=False)
    thr_new = pick_threshold(yva, retrained.predict_proba(Xva_s)[:, 1])
    p_new = retrained.predict_proba(Xte_shift)[:, 1]
    table = pd.DataFrame({"original_model_on_shifted_test": evaluate(yte, p_shift_original, thr_original),
                          "retrained_model_on_shifted_test": evaluate(yte, p_new, thr_new)})
    # Near-equivalence check (limitation): tree splits are invariant to a constant shift applied
    # consistently to training and test data, so the retrained model is expected to behave like the original.
    check = {"mean_abs_probability_difference": float(np.mean(np.abs(p_test_original - p_new))),
             "max_abs_probability_difference": float(np.max(np.abs(p_test_original - p_new))),
             "identical_binary_predictions": bool(np.array_equal(p_test_original >= thr_original, p_new >= thr_new)),
             "note": "original model on original test vs retrained model on the correspondingly shifted test"}
    with open(os.path.join(out_dir, "retraining_equivalence_check.json"), "w") as f:
        json.dump(check, f, indent=2)
    return table, check
