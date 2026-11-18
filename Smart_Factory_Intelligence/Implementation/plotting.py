"""Publication-style figures generated from RESULT TABLES (never from hard-coded numbers)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from config import METRICS

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
DPI = 300
METRIC_LABEL = {"precision": "Precision", "recall": "Recall", "f1": "F1-score",
                "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "mcc": "MCC"}


def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def small_multiples(values, series, colors, title, path, metrics=METRICS):
    """One panel per metric (own y-scale) so small and large metrics are both readable.
    values: {metric: [value per series]}"""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for ax, m in zip(axes.ravel(), metrics):
        v = values[m]
        bars = ax.bar(range(len(series)), v, color=colors[:len(series)], edgecolor="black", linewidth=.4)
        top = max(v) * 1.2 if max(v) > 0 else 1
        ax.set_ylim(0, top)
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x + top * .015, f"{x:.4f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(series))); ax.set_xticklabels(series, fontsize=7, rotation=15)
        ax.set_title(METRIC_LABEL[m], fontsize=9); ax.yaxis.grid(True, alpha=.3, linewidth=.5); ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96)); _save(fig, path)


def plot_model_comparison(results, path):
    names = list(results)
    vals = {m: [results[n][m] for n in names] for m in METRICS}
    small_multiples(vals, names, ["#bdbdbd", "#6baed6", "#08306b"], "Held-out Test Performance by Model", path)


def plot_confusion_matrices(results, out_dir):
    for name, r in results.items():
        v = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        fig, ax = plt.subplots(figsize=(3.5, 3.1))
        ax.imshow(v, cmap="Blues", norm=LogNorm(vmin=max(v.min(), 1), vmax=v.max()))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(v[i, j])}", ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if v[i, j] > v.max() * 0.02 else "black")
        lab = ["Non-defective (0)", "Defective (1)"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(lab); ax.set_yticks([0, 1]); ax.set_yticklabels(lab, rotation=90, va="center")
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        ax.set_title(f"{name} (threshold {r['threshold']:.2f})", fontsize=9.5); ax.tick_params(length=0)
        _save(fig, os.path.join(out_dir, f"confusion_matrix_{name.replace(' ', '_')}.png"))


def plot_shap_bar(top, path):
    top = top.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    bars = ax.barh(top.index, top.values, color="#4a6fa5", edgecolor="black", linewidth=.4)
    for b, v in zip(bars, top.values):
        ax.text(v + top.max() * .01, b.get_y() + b.get_height() / 2, f"{v:.6f}", va="center", fontsize=7)
    ax.set_xlim(0, top.max() * 1.18); ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title(f"Top {len(top)} Features by Mean Absolute SHAP Value", fontweight="bold")
    ax.xaxis.grid(True, alpha=.3, linewidth=.5); ax.set_axisbelow(True); _save(fig, path)


def plot_drift(drift, path, title="Feature-Level Drift"):
    d = drift.reset_index(drop=True); y = np.arange(len(d))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 6.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.15], "wspace": .05})
    a.barh(y, d.ks_statistic, color="#0072B2", edgecolor="black", linewidth=.4)
    for yy, v in zip(y, d.ks_statistic): a.text(v + .015, yy, f"{v:.6f}", va="center", fontsize=6)
    a.set_xlim(0, 1.28); a.set_xticks([0, .25, .5, .75, 1]); a.set_xlabel("KS statistic (linear axis)"); a.set_title("(a) KS statistic", fontsize=9)
    b.barh(y, d.psi, color="#D55E00", edgecolor="black", linewidth=.4)
    b.set_xscale("log"); b.set_xlim(min(d.psi.min() / 3, 1e-4), d.psi.max() * 300)
    for yy, v in zip(y, d.psi): b.text(v * 1.25, yy, f"{v:.6f}", va="center", fontsize=6)
    b.axvline(.10, color="black", ls=":", label="PSI 0.10 (heuristic)"); b.axvline(.25, color="black", ls="--", label="PSI 0.25 (heuristic)")
    b.set_xlabel("PSI (logarithmic axis)"); b.set_title("(b) Population Stability Index", fontsize=9)
    b.legend(loc="lower right", fontsize=6.5, frameon=True, fancybox=False, edgecolor="black")
    a.set_yticks(y); a.set_yticklabels(d.feature, fontsize=7.5); b.tick_params(axis="y", length=0)
    fig.suptitle(title, fontsize=10, fontweight="bold"); fig.subplots_adjust(left=.2, right=.98, top=.93, bottom=.08)
    _save(fig, path)


def plot_two_condition(table, labels, colors, title, path):
    vals = {m: [table[c][m] for c in table.columns] for m in METRICS}
    small_multiples(vals, labels, colors, title, path)


# ---------------------------------------------------------------- extension plots
def plot_severity_metric(results_df, metric, path):
    """Line plot: metric versus drift severity, original vs retrained model."""
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    order = list(dict.fromkeys(results_df["condition"]))
    for model, col, mk in (("original", "#D55E00", "o"), ("retrained", "#009E73", "s")):
        s = results_df[results_df.model == model].set_index("condition").loc[order, metric]
        ax.plot(order, s.values, marker=mk, color=col, label=f"{model.capitalize()} model")
        for x, v in zip(order, s.values): ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_xlabel("Drift severity"); ax.set_ylabel(METRIC_LABEL[metric])
    ax.set_title(f"Drift Severity versus {METRIC_LABEL[metric]}", fontweight="bold", fontsize=10)
    ax.yaxis.grid(True, alpha=.3, linewidth=.5); ax.legend(frameon=False, fontsize=8); _save(fig, path)


def plot_psi_by_severity(stats_df, path):
    """Mean PSI per severity (bars) with every feature's PSI as a point; log axis."""
    order = list(dict.fromkeys(stats_df["condition"]))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    means = [stats_df[stats_df.condition == c].psi.mean() for c in order]
    ax.bar(order, means, color="#D55E00", alpha=.55, edgecolor="black", linewidth=.5, label="Mean PSI (20 features)")
    for i, c in enumerate(order):
        pts = stats_df[stats_df.condition == c].psi.values
        ax.scatter(np.full(len(pts), i) + np.random.default_rng(0).uniform(-.18, .18, len(pts)), pts, s=9, color="black", zorder=3,
                   label="Individual features" if i == 0 else None)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.axhline(.10, color="black", ls=":", lw=.9, label="PSI 0.10 (heuristic)"); ax.axhline(.25, color="black", ls="--", lw=.9, label="PSI 0.25 (heuristic)")
    ax.set_xlabel("Drift severity"); ax.set_ylabel("PSI (symlog axis)")
    ax.set_title("PSI Across Drift Severity Levels", fontweight="bold", fontsize=10); ax.legend(frameon=False, fontsize=7); _save(fig, path)


def plot_severity_overview(results_df, path):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.2))
    order = list(dict.fromkeys(results_df["condition"]))
    for ax, m in zip(axes.ravel(), METRICS):
        for model, col, mk in (("original", "#D55E00", "o"), ("retrained", "#009E73", "s")):
            s = results_df[results_df.model == model].set_index("condition").loc[order, m]
            ax.plot(order, s.values, marker=mk, color=col, label=model.capitalize())
        ax.set_title(METRIC_LABEL[m], fontsize=9); ax.tick_params(axis="x", labelsize=7, rotation=20); ax.yaxis.grid(True, alpha=.3, linewidth=.5)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Original vs Retrained Model Across Drift Severities", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96)); _save(fig, path)
