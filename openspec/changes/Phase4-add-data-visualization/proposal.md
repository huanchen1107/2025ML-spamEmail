## Why
We want clear, reproducible visual reports for the spam classifier: data distribution, token patterns, and model performance. This aids understanding and presentation for Phase 4.

## What Changes
- Add a visualization CLI `scripts/visualize_spam.py` to generate:
  - Class distribution (bar chart) from cleaned dataset
  - Top token frequency charts for spam vs ham
  - Confusion matrix on held-out test
  - ROC and Precision-Recall curves
  - Threshold sweep table/plot (Precision/Recall/F1 vs threshold)
- Add a Streamlit dashboard `app/streamlit_app.py` to explore the same visuals interactively and provide a live inference text box that returns label + probability with a threshold slider.
- Save figures and tables under `reports/visualizations/`.
- Document usage in README.

## Impact
- New dependencies: `matplotlib`, `seaborn`, `streamlit` (add to requirements.txt)
- New script: `scripts/visualize_spam.py`
- New app: `app/streamlit_app.py`
- New folder: `reports/visualizations/` for outputs (git-ignored if large)

## Out of Scope
- Advanced dashboards (multi-page apps, authentication, or persistent backends)
- SHAP/feature attribution beyond token frequency
