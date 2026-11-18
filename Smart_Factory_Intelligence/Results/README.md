# Results folder
- `Graphs/Recorded_Paper_Readings/` and `Tables/Recorded_Paper_Readings/` - figures and tables drawn from the
  EXACT readings recorded in the paper (script: `Implementation/regenerate_paper_figures.py`). These are the
  figures used in the paper. ROC and Precision-Recall curves are NOT included: the original prediction
  probabilities are not available and curves must not be reconstructed from AUC values.
- Running `Implementation/main.py` writes NEW results here from an actual run:
  `Graphs/Original_Study/`, `Graphs/Extension_Multi_Severity/`, `Tables/*.csv`, `Experimental_Results/*`.
  (Not present until you run it on the full Bosch data.)
