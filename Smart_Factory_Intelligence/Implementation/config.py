"""Central configuration: seeds, documented dataset facts and model hyper-parameters (from the paper)."""
SEED = 42
MAX_MISSING_FRACTION = 0.50          # features with more than 50% missing values are dropped
TOP_K_FEATURES = 20                  # SHAP-ranked features used in all drift experiments
DT_RF_TRAIN_SAMPLE = 250_000         # Decision Tree / Random Forest training sample (paper limitation)

# Facts documented in the paper (used only to WARN if the loaded data differ).
EXPECTED = {"records": 1_183_747, "columns": 970, "defective": 6_879,
            "non_defective": 1_176_868, "selected_features": 156,
            "train_rows": 828_622, "val_rows": 177_562, "test_rows": 177_563}

# Thresholds recorded in the paper. The pipeline re-derives thresholds on VALIDATION data;
# these values are only used with --use-recorded-thresholds or for comparison.
RECORDED_THRESHOLDS = {"Decision Tree": 0.95, "Random Forest": 0.58, "Boost": 0.86}

# Multi-severity extension: shifted = ref_mean + scale*(x - ref_mean) + location*ref_std
SEVERITY_CONDITIONS = {
    "No Drift": {"scale": 1.00, "location": 0.00},
    "Mild":     {"scale": 1.25, "location": 0.25},
    "Moderate": {"scale": 1.50, "location": 0.50},
    "Strong":   {"scale": 1.75, "location": 0.75},
}
METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
