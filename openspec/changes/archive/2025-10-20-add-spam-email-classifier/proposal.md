## Why
We need a machine learning solution to classify emails as spam or not spam for coursework. The goal is a simple, reproducible baseline that trains locally, evaluates with clear metrics, and provides a small CLI for inference.

## What Changes
- Add a new capability `spam-classifier` under OpenSpec.
- Separate dataset preprocessing from model training. Provide a standalone preprocessing step that reads raw CSVs in `datasets/`, applies deterministic text normalization, and writes cleaned CSVs to `datasets/processed/`.
- Implement a minimal training pipeline using scikit-learn (TF-IDF + Logistic Regression) over the cleaned dataset.
- Persist trained artifacts under `models/` and expose a simple prediction CLI.
- Document how to run preprocessing, training, and inference locally with Python.

## Impact
- New Python runtime dependencies: `scikit-learn`, `pandas`, `numpy`, `joblib` (optional: `scipy` if not already present).
- Adds scripts: `scripts/preprocess_emails.py`, `scripts/train_spam_classifier.py`, `scripts/predict_spam.py`.
- Adds model artifacts in `models/` (git-ignored if large).
- No breaking changes to existing notebooks; optional integration demo notebook could be added.

## Out of Scope
- Cloud deployment or APIs
- Advanced model architectures (transformers, deep learning)
- Labeling or data collection tooling

## Preprocessing (Separated Stage)
Goal: produce a cleaned, deterministic text column for modeling, stored in `datasets/processed/*.csv`.

Inputs
- Raw CSV in `datasets/` (default: `datasets/sms_spam_no_header.csv`)
- Configurable columns via flags: `--label-col`, `--text-col`

Operations (deterministic)
- Lowercase normalization
- Trim and collapse whitespace
- Replace URLs, emails, and phone numbers with special tokens (`<URL>`, `<EMAIL>`, `<PHONE>`) using regex
- Replace numbers with `<NUM>` token (optional flag `--keep-numbers` to bypass)
- Strip surrounding punctuation; preserve intra-word apostrophes/hyphens
- Optional stopword removal flag (`--remove-stopwords`) — off by default to keep baseline simple

Outputs
- Cleaned CSV written to `datasets/processed/<name>_clean.csv`
- Column names preserved; text column becomes normalized (e.g., `text_clean` if `--output-text-col` provided)

CLI (to be implemented)
- `python scripts/preprocess_emails.py --input datasets/sms_spam_no_header.csv --output datasets/processed/sms_spam_clean.csv --label-col label --text-col text`

Rationale
- Clear separation of concerns: data cleaning is explicit and reproducible
- Enables faster iteration on feature engineering without altering raw data
