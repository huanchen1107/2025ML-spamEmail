## 1. Implementation
- [x] 1.1 Create `scripts/preprocess_emails.py`:
      - Inputs: `--input` (raw CSV in `datasets/`), `--output` (CSV in `datasets/processed/`), `--label-col`, `--text-col`, optional `--output-text-col` (default: `text`), flags `--keep-numbers`, `--remove-stopwords`.
      - Deterministic ops: lowercase, collapse whitespace, regex tokenization for URLs (`<URL>`), emails (`<EMAIL>`), phones (`<PHONE>`), optional numbers to `<NUM>`, strip surrounding punctuation.
      - Ensure `datasets/processed/` exists; write cleaned CSV preserving column order; do not modify label semantics.
      - Smoke test: run on `datasets/sms_spam_no_header.csv` to produce `datasets/processed/sms_spam_clean.csv`.
- [x] 1.2 Create `scripts/train_spam_classifier.py` with pipeline:
      - Load dataset (default: `datasets/processed/sms_spam_clean.csv` if present, else fall back to `datasets/sms_spam_no_header.csv`).
      - Columns: use `--label-col` and `--text-col` (default to those in the file; support `--text-col text_clean` when using cleaned data).
      - Split train/test (e.g., 80/20), seed 42.
      - Vectorize with `TfidfVectorizer(stop_words="english")`.
      - Train `LogisticRegression(max_iter=1000)`.
      - Evaluate: accuracy, precision, recall, F1 (print to stdout).
      - Save artifacts: `models/spam_tfidf_vectorizer.joblib`, `models/spam_logreg_model.joblib`.
- [x] 1.3 Create `scripts/predict_spam.py`:
      - Load artifacts.
      - Predict label + probability for input text via `--text` or batch CSV via `--input` and write `--output` CSV.
      - Exit code 0 and clear stdout messages.
- [x] 1.4 Add `README` section with end-to-end usage examples (preprocess -> train -> predict).

## 2. Quality
- [x] 2.1 Ensure training completes <1 minute on typical laptop.
- [ ] 2.2 Meet metrics on held-out test: Accuracy >= 0.95, Precision >= 0.93, Recall >= 0.93 (dataset-dependent).
- [x] 2.3 Add basic argument validation and helpful error messages.
- [x] 2.4 Preprocessing is deterministic/idempotent: repeated runs on the same input yield identical output.

## 3. Ops
- [x] 3.1 Add `models/` to `.gitignore` if not already present.
- [x] 3.2 Create `datasets/processed/` (keep empty `.gitkeep` if needed) and document that processed CSVs can be regenerated.
- [x] 3.3 Document dependency install (`pip install -r requirements.txt`).

