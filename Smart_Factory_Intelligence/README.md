# Smart Factory Intelligence (Intro to AI)

**Smart Factory Intelligence: Manufacturing Defect Prediction, Explainability, and Controlled Data-Drift Adaptation**
Author: Muhammad Areeb Waqas, Air University, Islamabad, Pakistan. Dataset: Bosch Production Line Performance.

## Two implementations
- **A. Base paper pipeline** (`Implementation/run_reference_paper.py`, `reference_features.py`, `reference_benchmark.py`):
  station/line/path numeric + date aggregates, out-of-fold Weight-of-Evidence compression of the nominal columns
  (16 features), 80/20 stratified split, constant/duplicate feature removal, random oversampling of the training set, and a
  benchmark of Decision Tree, Random Forest, Gradient Boosting, XGBoost, Linear SVM and MLP (hyper-parameters from the
  paper's Table 3), followed by the **new drift extension** on the engineered features. Needs `train_numeric.csv`,
  `train_date.csv` and `train_categorical.csv`.
- **B. Derived study** (`Implementation/main.py` and the other modules): the author's numeric-feature study with
  70/15/15 split, SHAP, KS/PSI drift and retraining (see sections below).

### Deviations from the base paper (not hidden)
Pandas/scikit-learn instead of Apache Spark; **Factorization Machines and TabNet are not implemented**; no 5x3 cross-validated
hyper-parameter search (Table 3 values used directly); Spark GBT replaced by scikit-learn HistGradientBoosting; the paper's
numerical feature count (178) cannot be derived exactly from its textual description, so this code reports the count it produces;
the 'path' level of the WoE summary and some other details are interpretations (marked `INTERP` in the code).
Scaling was added for SVM/MLP. Results will therefore not match the paper's Table 4 exactly.

### Observations about the base paper (for your report)
- Its Table 4 precision/recall/F1 (e.g. recall 0.9975 with MCC 0.79) look like *weighted-average* metrics; the code reports both
  weighted and defect-class metrics.
- The text says positive WoE means higher defect risk, but the printed formula ln(%non-events/%events) gives the opposite sign;
  the code follows the formula.

## Research summary (derived study)
Compares a Decision Tree, Random Forest and Boost (XGBoost) on the numeric Bosch features (0.5811% defective),
explains Boost with SHAP, measures controlled feature drift with KS and PSI, and tests an initial retraining
experiment. The recorded results and their limitations are in `Research_Paper/Updated_Research_Paper.docx`.
Decision Tree and Random Forest were trained on 250,000 rows, Boost on all 828,622 (disclosed limitation).

## New addition (extension)
**Multi-Severity Feature Drift Analysis and Model Adaptation**: four drift severities (location-scale transform of
the top-20 SHAP features), KS/PSI and metrics for original vs retrained models. See `NEW_ADDITION.txt`.
**Status:** implemented and smoke-tested on synthetic data only; **not yet run on the real dataset, so no extension
results are reported.**

## Structure
```
Research_Paper/   Base_Paper_Zdravevski_2026.pdf (paper being implemented), Original_Research_Paper.docx (derived study), Updated_Research_Paper.docx, Research_Paper.pdf
Implementation/   main.py, data_loader.py, preprocessing.py, model_training.py, evaluation.py,
                  explainability.py, drift_detection.py, retraining.py, multi_severity_drift.py,
                  plotting.py, config.py, regenerate_paper_figures.py, make_synthetic_smoke_data.py, requirements.txt
Results/          Graphs/, Tables/, Experimental_Results/
NEW_ADDITION.txt  README.md  Verification_Report.md  .gitignore  LICENSE.txt
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r Implementation/requirements.txt
```
**Dataset (not included).** Download `train_numeric.csv` from
https://www.kaggle.com/competitions/bosch-production-line-performance/data (Kaggle login and accepting the
competition rules are required), unzip it, and place it e.g. at `Data/train_numeric.csv`.

## Run
```bash
cd Implementation
# A. base paper pipeline + extension (3 Kaggle files)
python run_reference_paper.py --numeric ../Data/train_numeric.csv --date ../Data/train_date.csv --categorical ../Data/train_categorical.csv --results ../Results/Reference_Paper
# B. derived study + extension
python main.py --data ../Data/train_numeric.csv --results ../Results             # full study + extension
python main.py --data ../Data/train_numeric.csv --results ../Results --skip_extension   # original paper only
python multi_severity_drift.py --data ../Data/train_numeric.csv --results ../Results    # extension only (after main.py)
python main.py --data ../Data/train_numeric.csv --results ../Results_test --nrows 200000   # quick test, NOT the paper setting
python regenerate_paper_figures.py ../Results/Graphs/Recorded_Paper_Readings           # figures from recorded readings
```
Smoke test without the dataset: `python make_synthetic_bosch_files.py /tmp/sb && python run_reference_paper.py --numeric /tmp/sb/train_numeric.csv --date /tmp/sb/train_date.csv --categorical /tmp/sb/train_categorical.csv --results ../Results_smoke`; or for B: `python make_synthetic_smoke_data.py s.csv && python main.py --data s.csv --results ../Results_smoke --nrows 30000 --shap_sample 3000 --dt_rf_sample 10000`
(synthetic output is meaningless scientifically; delete it afterwards).

**Compute:** the full file is 1,183,747 x 970 values; plan for roughly 16 GB+ RAM and a long run (several model
fits including SHAP and three retrainings for the extension). Nothing is silently subsampled unless you pass `--nrows`.

## Outputs
`Tables/` (model comparison, SHAP, drift, original-vs-shifted, retrained, multi-severity CSVs), `Graphs/Original_Study/`,
`Graphs/Extension_Multi_Severity/`, `Experimental_Results/` (audit, saved Boost model, metadata, equivalence check).

## Documented choices and assumptions
- Split: stratified 70/15/15, `random_state=42` (two-step); on the full data this gives exactly 828,622 / 177,562 / 177,563 rows.
  The paper's original saved split indices were not available.
- Thresholds are re-selected on validation data (maximum F1); the paper's recorded thresholds (0.95 / 0.58 / 0.86) can be
  forced with `--use_recorded_thresholds`. Values may differ slightly from the paper.
- SHAP uses a random training sample (default 20,000 rows; the paper does not state the size).
- The original retraining experiment is implemented as a +1 training-std constant shift on the top-20 features (the paper
  does not give the exact constant), so its numbers may differ from the recorded ones.
- ROC/PR curve plots are not produced from recorded AUC values (would be fabricated).

## Upload to GitHub (not done automatically)
```bash
git init
git add .
git commit -m "Smart Factory Intelligence - Intro to AI"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
Add the instructor: repository **Settings, Collaborators, Add people**, enter Sir Ali Javed's GitHub username or
`alipirzada@gmail.com`, then send the invitation. GitHub shows the invite as pending until he accepts.
