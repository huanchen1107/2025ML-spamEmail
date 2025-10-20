# Final Report — Spam Email Classifier (Phase 1 → Phase 4)

This report summarizes the end‑to‑end work completed across four phases of the project, from baseline spam/ham classification to improved model quality and an interactive Streamlit dashboard with visual analytics.

- Repository: `huanchen1107/2025ML-spamEmail`
- Demo (Streamlit Cloud): https://2025spamemail.streamlit.app/

## Contents
- Phase 1 — Baseline Classifier (Preprocess → Train → Predict)
- Phase 2 — Improve Recall (tunable pipeline)
- Phase 3 — Restore Precision with High Recall (recommended settings)
- Phase 4 — Visualization + Streamlit App (dashboard, live inference)
- Artifacts and Directory Map
- Reproducibility and Validation

---

## Phase 1 — Baseline Classifier
Goal: build a simple, reproducible binary classifier (spam vs ham) with a lightweight feature pipeline and clear metrics.

Key deliverables
- Preprocessing script: `scripts/preprocess_emails.py`
- Training script: `scripts/train_spam_classifier.py`
- Prediction script: `scripts/predict_spam.py`
- Cleaned dataset: `datasets/processed/sms_spam_clean.csv` (from `datasets/sms_spam_no_header.csv`)
- Model artifacts: `models/` (`spam_tfidf_vectorizer.joblib`, `spam_logreg_model.joblib`, `spam_label_mapping.json`)

Preprocessing (deterministic and idempotent)
- Normalize text: lowercase, collapse whitespace
- Mask special patterns: URLs→`<URL>`, emails→`<EMAIL>`, phones→`<PHONE>`, digits→`<NUM>` (optional)
- Strip punctuation (preserve word chars, spaces, special tokens)
- Optional stopword removal (off by default)
- Outputs (optional): column‑wise step results and per‑step CSVs under `datasets/processed/steps/`

Run
```bash
# Clean raw CSV (headerless: 1st col=label, 2nd col=text), keep a clean column `text_clean`
python scripts/preprocess_emails.py \
  --input datasets/sms_spam_no_header.csv \
  --output datasets/processed/sms_spam_clean.csv \
  --no-header --label-col-index 0 --text-col-index 1 \
  --output-text-col text_clean --save-step-columns \
  --steps-out-dir datasets/processed/steps
```

Model and features
- TF‑IDF (`TfidfVectorizer(stop_words="english")`) over unigrams → sparse matrix
- Logistic Regression (`LogisticRegression(max_iter=1000)`)
- Train/test split: 80/20, `random_state=42`, stratify by label

Training
```bash
python scripts/train_spam_classifier.py \
  --input datasets/processed/sms_spam_clean.csv \
  --label-col col_0 --text-col text_clean
```

Prediction
```bash
# Single text
python scripts/predict_spam.py --text "Free entry in 2 a wkly comp to win cash"

# Batch CSV
python scripts/predict_spam.py \
  --input datasets/processed/sms_spam_clean.csv \
  --text-col text_clean \
  --output predictions.csv
```

Observations (typical baseline)
- Accuracy ≈ 0.97–0.98, Precision high, Recall typically lower (≈ 0.85) on the held‑out split.

---

## Phase 2 — Improve Recall
Goal: increase Recall (catch more spam) while keeping Precision acceptable.

Enhancements
- New training flags in `scripts/train_spam_classifier.py`:
  - `--class-weight {none,balanced}`
  - `--ngram-range min,max` (e.g., `1,2` for unigrams+bigrams)
  - `--min-df` term‑frequency cutoff
  - `--sublinear-tf` enable TF sublinear scaling
  - `--eval-threshold` probability threshold used during evaluation

Tuned example
```bash
python scripts/train_spam_classifier.py \
  --input datasets/processed/sms_spam_clean.csv \
  --label-col col_0 --text-col text_clean \
  --class-weight balanced --ngram-range 1,2 \
  --min-df 2 --sublinear-tf --eval-threshold 0.40
```

Outcome (example)
- Accuracy: 0.9731  • Precision: 0.8521  • Recall: 0.9664  • F1: 0.9057
- Recall target achieved (≥ 0.93), Precision decreased due to threshold/weighting.

Archived change: `2025-10-20-Phase2-improve-spam-recall`

---

## Phase 3 — Restore Precision with High Recall
Goal: achieve Precision ≥ 0.90 while maintaining Recall ≥ 0.93.

Sweep and recommended settings
- Candidate sweeps over `class_weight`, `ngram_range`, `min_df`, `C`, and `eval_threshold`.
- Recommended (best trade‑off in runs):
```bash
python scripts/train_spam_classifier.py \
  --input datasets/processed/sms_spam_clean.csv \
  --label-col col_0 --text-col text_clean \
  --class-weight balanced \
  --ngram-range 1,2 \
  --min-df 2 \
  --sublinear-tf \
  --C 2.0 \
  --eval-threshold 0.50
```

Observed (held‑out)
- Accuracy: 0.9848  • Precision: 0.9231  • Recall: 0.9664  • F1: 0.9443

Archived change: `2025-10-20-Phase3-improve-spam-precision`

---

## Phase 4 — Visualization + Streamlit App
Goals: produce reproducible visual reports and an interactive dashboard with live inference.

CLI visualizations — `scripts/visualize_spam.py`
- Class distribution bar chart (cleaned data)
- Token frequency bar charts (top‑N; spam vs ham)
- Confusion matrix (test set)
- ROC and Precision–Recall curves
- Threshold sweep (CSV + plot)
- Outputs: `reports/visualizations/` (git‑ignored locally)

Streamlit dashboard — `app/streamlit_app.py`
- Sidebar: dataset/column pickers, models dir, test size, seed, threshold
- Data Overview: class distribution, token replacement counts
- Top Tokens by Class: bar charts for ham/spam tokens
- Model Performance (test): confusion matrix, ROC, PR, threshold sweep table
- Live Inference:
  - Normalizes input (URL/EMAIL/PHONE masks, `<NUM>`), matching `text_clean`
  - Returns label + spam probability; probability bar with threshold marker
  - “Use spam example” / “Use ham example” buttons to auto‑fill text
- Batch Predict (upload CSV): choose a text column, optional normalization, preview + download predictions

Run locally
```bash
streamlit run app/streamlit_app.py
# If needed: --server.address 127.0.0.1 --server.port 8505
```

Deploy to Streamlit Cloud
- Repo: `huanchen1107/2025ML-spamEmail` (models included)
- Main file: `app/streamlit_app.py`
- Result: https://2025spamemail.streamlit.app/

Archived change: `2025-10-20-Phase4-add-data-visualization`

---

## Artifacts and Directory Map
- `datasets/` – raw CSVs; `datasets/processed/` – cleaned data; `datasets/processed/steps/` – per‑step files
- `models/` – TF‑IDF vectorizer, Logistic Regression model, label mapping (committed for cloud deploy)
- `scripts/` – CLI tools (`preprocess_emails.py`, `train_spam_classifier.py`, `predict_spam.py`, `visualize_spam.py`)
- `app/` – Streamlit app (`streamlit_app.py`)
- `openspec/` – project instructions and spec/change system
  - `openspec/specs/` – live specs (spam‑classifier, visualization)
  - `openspec/changes/archive/` – archived changes
- `reports/visualizations/` – generated figures/CSVs (ignored in git by default)

---

## Reproducibility and Validation
- Deterministic preprocessing; optional per‑step outputs and CSVs
- Seeded train/test split; explicit thresholds for evaluation
- Validation metrics logged at train time; visualization scripts produce charts and tables
- Recommended configuration (Phase 3) documented in README for consistent results
- Streamlit app normalizes live input to align with training `text_clean`

---

## Appendix — Quick Commands
```bash
# Preprocess
python scripts/preprocess_emails.py \
  --input datasets/sms_spam_no_header.csv \
  --output datasets/processed/sms_spam_clean.csv \
  --no-header --label-col-index 0 --text-col-index 1 \
  --output-text-col text_clean

# Train (recommended Phase 3)
python scripts/train_spam_classifier.py \
  --input datasets/processed/sms_spam_clean.csv \
  --label-col col_0 --text-col text_clean \
  --class-weight balanced --ngram-range 1,2 --min-df 2 --sublinear-tf --C 2.0 --eval-threshold 0.50

# Predict (single)
python scripts/predict_spam.py --text "You won a prize! Claim now!"

# Visualizations (saved under reports/visualizations/)
python scripts/visualize_spam.py --class-dist --input datasets/processed/sms_spam_clean.csv --label-col col_0
python scripts/visualize_spam.py --token-freq --input datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col text_clean --topn 20
python scripts/visualize_spam.py --confusion-matrix --roc --pr --models-dir models --input datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col text_clean
python scripts/visualize_spam.py --threshold-sweep --models-dir models --input datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col text_clean

# Streamlit (dashboard)
streamlit run app/streamlit_app.py
```

---

All four phases are archived; live specs reflect the final behavior for classification and visualization. The project is ready for local demos and cloud deployment.

