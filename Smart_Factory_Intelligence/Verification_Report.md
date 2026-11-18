# Implementation verification report
Built: 18 Nov 2026 in a sandbox WITHOUT the Bosch dataset and WITHOUT GitHub access.

## Verified (actually performed)
- All `Implementation/*.py` files compile and import (checked below in the packaging step).
- `main.py` ran end to end on a 30,000-row SYNTHETIC file: audit, feature selection, split, three models, SHAP (tables + summary plot),
  original shift + KS/PSI, original retraining + equivalence check, and the multi-severity extension (tables + 7 graphs).
- `multi_severity_drift.py` standalone mode ran using the model/metadata saved by `main.py`.
- Split logic: sklearn's split arithmetic for n = 1,183,747 gives train 828,622 / validation 177,562 / test 177,563 (matches the paper).
- Recorded-readings figures (8 PNG, 300 DPI) and tables: 36 plotted series asserted equal to the supplied readings.
- Updated paper DOCX passes the DOCX validator; PDF converted from it; ZIP extraction tested.
- No raw dataset or synthetic data is in the repository.

- Base-paper pipeline (`run_reference_paper.py`) ran end to end on SYNTHETIC aligned numeric/date/categorical files, including
  out-of-fold WoE, oversampling, the six models, SHAP and the multi-severity extension.

## NOT performed (do not claim)
- Factorization Machines and TabNet from the base paper are not implemented.
- No run on the real Bosch dataset (either pipeline): paper numbers were not re-derived by this code, and NO multi-severity results exist.
- ROC and Precision-Recall curves not created (no probabilities available).
- Nothing pushed to GitHub; no collaborator invitation sent.
- The earlier ZIP/code from previous chats was the only prior implementation; no other files were supplied in this chat.

## Observation from the synthetic smoke test (not a scientific result)
With the increasing linear transform, the retrained model matched the No-Drift baseline exactly, consistent with the invariance of
tree models to monotone feature transforms; this is documented as an expected limitation in the paper and NEW_ADDITION.txt.
