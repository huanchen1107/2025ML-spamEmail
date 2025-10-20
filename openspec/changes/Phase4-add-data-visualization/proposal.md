## Why
We want clear, reproducible visual reports for the spam classifier: data distribution, token patterns, and model performance. This aids understanding and presentation for Phase 4.

## What Changes
- Add a visualization CLI `scripts/visualize_spam.py` to generate:
  - Class distribution (bar chart) from cleaned dataset
  - Top token frequency charts for spam vs ham
  - Confusion matrix on held-out test
  - ROC and Precision-Recall curves
  - Threshold sweep table/plot (Precision/Recall/F1 vs threshold)
- Save figures and tables under `reports/visualizations/`.
- Document usage in README.

## Impact
- New dependencies: `matplotlib`, `seaborn` (add to requirements.txt)
- New script: `scripts/visualize_spam.py`
- New folder: `reports/visualizations/` for outputs (git-ignored if large)

## Out of Scope
- Interactive dashboards and web apps
- SHAP/feature attribution beyond token frequency
