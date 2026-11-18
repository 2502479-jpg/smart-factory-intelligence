"""Metrics, threshold selection and result tables."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef, confusion_matrix,
                             precision_recall_curve)


def pick_threshold(y_val, p_val):
    """Threshold that maximises F1 on VALIDATION predictions. The test set is never used here."""
    prec, rec, th = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(th[int(np.argmax(f1))])


def evaluate(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),   # average precision, as in the paper
            "mcc": matthews_corrcoef(y, pred),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def save_table(df, path):
    df.to_csv(path)
    print("saved", path)
