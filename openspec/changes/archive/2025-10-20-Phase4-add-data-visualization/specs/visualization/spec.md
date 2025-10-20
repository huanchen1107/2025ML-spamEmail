## ADDED Requirements

### Requirement: Data Visualization for Spam Classifier
The system MUST provide reproducible visual artifacts to describe data distribution, token patterns, and model performance.

#### Scenario: Class distribution
- WHEN `python scripts/visualize_spam.py --class-dist --input datasets/processed/sms_spam_clean.csv --label-col col_0 --outdir reports/visualizations`
- THEN a bar chart of class counts is saved under `reports/visualizations/` (PNG)

#### Scenario: Token frequency (top-N per class)
- WHEN `python scripts/visualize_spam.py --token-freq --input datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col text_clean --outdir reports/visualizations`
- THEN bar charts of top tokens for spam and ham are saved (PNGs)

#### Scenario: Confusion matrix
- GIVEN trained artifacts exist in `models/`
- WHEN `python scripts/visualize_spam.py --confusion-matrix --models-dir models --input datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col text_clean --outdir reports/visualizations`
- THEN a confusion matrix image is saved (PNG)

#### Scenario: ROC and PR curves
- GIVEN trained artifacts exist
- WHEN running with `--roc` and `--pr`
- THEN ROC and Precision-Recall plots are saved (PNGs)

#### Scenario: Threshold sweep
- GIVEN trained artifacts exist
- WHEN `--threshold-sweep` is provided
- THEN a CSV table and optional plot of threshold vs Precision/Recall/F1 is saved under `reports/visualizations/`
