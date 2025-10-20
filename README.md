# Spam Email Classifier (AIoT-DA2025 HW3)

A simple, reproducible pipeline to classify messages/emails as spam or ham using scikit‑learn and OpenSpec.

- Preprocessing report: docs/PREPROCESSING.md
- OpenSpec change proposal: openspec/changes/add-spam-email-classifier/

## Setup

```
# In a fresh virtual environment (recommended)
pip install -r requirements.txt
```

## Data

- Raw dataset (headerless 2-column CSV): `datasets/sms_spam_no_header.csv`
- Cleaned dataset (generated): `datasets/processed/sms_spam_clean.csv`

## Commands

Preprocess (saves per-step outputs, optional):
```
python scripts/preprocess_emails.py \
  --input datasets/sms_spam_no_header.csv \
  --output datasets/processed/sms_spam_clean.csv \
  --no-header --label-col-index 0 --text-col-index 1 \
  --output-text-col text_clean \
  --save-step-columns \
  --steps-out-dir datasets/processed/steps
```

Train:
```
python scripts/train_spam_classifier.py \
  --input datasets/processed/sms_spam_clean.csv \
  --label-col col_0 --text-col text_clean
```

Predict (single text):
```
python scripts/predict_spam.py --text "Free entry in 2 a wkly comp to win cash"
```

Predict (batch CSV):
```
python scripts/predict_spam.py \
  --input datasets/processed/sms_spam_clean.csv \
  --text-col text_clean \
  --output predictions.csv
```

## Notes
- Artifacts are saved to `models/` for reuse (vectorizer, model, label mapping).
- See docs/PREPROCESSING.md for detailed step-by-step preprocessing with examples.
- OpenSpec usage: `openspec validate add-spam-email-classifier --strict`
