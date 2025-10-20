# Preprocessing Report: SMS/Email Spam Dataset

## Overview
This report documents the text preprocessing pipeline used to prepare raw SMS/email data for spam classification. The pipeline is deterministic and idempotent: the same input produces the same output. It reads raw CSVs from `datasets/` and writes cleaned CSVs and optional per‑step files to `datasets/processed/`.

- Entry script: `scripts/preprocess_emails.py`
- Default output (final): `datasets/processed/sms_spam_clean.csv`
- Optional step outputs: `datasets/processed/steps/` (one CSV per step)
- Final cleaned column name: by default the same as the input text column; can be set via `--output-text-col` (e.g., `text_clean`).

## Dataset
- Typical raw file: `datasets/sms_spam_no_header.csv`
- Headerless format (2 columns):
  - `col_0` → label (e.g., `spam`/`ham`)
  - `col_1` → raw message text

## Pipeline Steps (in order)
Each step transforms the text and is also available as an intermediate column when using `--save-step-columns`. When `--steps-out-dir` is provided, a CSV is saved after every step.

1) Lowercase Normalization (`text_lower`)
- Description: convert all characters to lowercase.
- Before: `"Ok lar... Joking wif u oni..."`
- After: `"ok lar... joking wif u oni..."`

2) Mask Contacts (`text_contacts_masked`)
- Description: replace URLs, emails, and phone numbers with tokens.
- Tokens: `<URL>`, `<EMAIL>`, `<PHONE>`
- Example email → token: `"Contact me at test@example.com"` → `"contact me at <EMAIL>"`
- Example URL → token: `"Visit https://example.com"` → `"visit <URL>"`
- Example phone → token: `"Call +1 415-555-1212"` → `"call <PHONE>"`

3) Numbers Handling (`text_numbers`)
- Description: replace digit sequences with `<NUM>` unless `--keep-numbers` is used.
- Before: `"Text FA to 87121 to receive entry"`
- After: `"text fa to <NUM> to receive entry"`

4) Strip Punctuation (`text_stripped`)
- Description: remove punctuation; keep word characters, spaces, and special tokens (`<>`).
- Before: `"crazy.. available only in bugis"`
- After: `"crazy   available only in bugis"`

5) Whitespace Normalization (`text_whitespace`)
- Description: collapse repeated spaces to a single space and trim.
- Before: `"go until jurong point  crazy   available"`
- After: `"go until jurong point crazy available"`

6) Stopword Removal (Optional) (`text_stopwords_removed`)
- Description: remove English stopwords (scikit‑learn ENGLISH_STOP_WORDS) when `--remove-stopwords` is provided; off by default.
- Before: `"this is only a test of the system"`
- After: `"test system"`

Final cleaned text (`text_clean`)
- The final column is taken from `text_stopwords_removed` (or from `text_whitespace` if no stopword removal is applied). The name defaults to the input text column, or to `--output-text-col` if provided.

## Example: Before vs After
Sample from `datasets/sms_spam_no_header.csv` → `datasets/processed/sms_spam_clean.csv`:

- Raw: `"spam","Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 ..."`
- Cleaned (`text_clean`): `"free entry in <NUM> a wkly comp to win fa cup final tkts <NUM>st may <NUM> text fa to <NUM> ..."`

Another raw line:
- Raw: `"ham","Ok lar... Joking wif u oni..."`
- Cleaned: `"ok lar joking wif u oni"`

## How To Reproduce
Run preprocessing with intermediate artifacts saved:

```
python scripts/preprocess_emails.py \
  --input datasets/sms_spam_no_header.csv \
  --output datasets/processed/sms_spam_clean.csv \
  --no-header --label-col-index 0 --text-col-index 1 \
  --output-text-col text_clean \
  --save-step-columns \
  --steps-out-dir datasets/processed/steps
```

Outputs:
- Final CSV with step columns: `datasets/processed/sms_spam_clean.csv`
- Per‑step CSVs: `datasets/processed/steps/`
  - `sms_spam_clean_01_text_lower.csv`
  - `sms_spam_clean_02_text_contacts_masked.csv`
  - `sms_spam_clean_03_text_numbers.csv`
  - `sms_spam_clean_04_text_stripped.csv`
  - `sms_spam_clean_05_text_whitespace.csv`
  - `sms_spam_clean_06_text_stopwords_removed.csv`

## Notes & Guarantees
- Deterministic/idempotent: same input → same output.
- Label semantics are preserved; only the text column is transformed.
- Headerless input gets temporary column names (e.g., `col_0`, `col_1`).
- Performance: linear in number of characters; suitable for CPU‑only laptops.

## Related Files
- Script: `scripts/preprocess_emails.py`
- Cleaned dataset (default): `datasets/processed/sms_spam_clean.csv`
- Step outputs (optional): `datasets/processed/steps/`

## From Preprocessing to Classification

### Feature Extraction (TF‑IDF)
- Purpose: convert cleaned text into numeric feature vectors for modeling.
- Implementation: `sklearn.feature_extraction.text.TfidfVectorizer(stop_words="english")` in `scripts/train_spam_classifier.py`.
- Behavior:
  - Tokenizes into unigrams (single tokens like `free`, `entry`, `<URL>`, `<NUM>`).
  - Builds a vocabulary from the training set only.
  - Computes tf‑idf weights per document and L2‑normalizes rows.
- Output: a sparse matrix `X` with shape `[n_samples, vocab_size]`, where each column corresponds to a token and each row is a document.

### Train/Test Split
- Split: 80/20 hold‑out using `train_test_split(..., test_size=0.2, random_state=42, stratify=labels)`.
- Why: evaluates generalization on unseen messages with preserved class balance.

### Classifier
- Model: `sklearn.linear_model.LogisticRegression(max_iter=1000)`.
- Rationale: strong, fast baseline for high‑dimensional sparse text features.
- Options to tune (if needed): `class_weight="balanced"`, regularization `C`, n‑grams for TF‑IDF.

### Metrics
- Reported on test set: Accuracy, Precision, Recall, F1.
- Targets (as per spec, dataset‑dependent): Accuracy ≥ 0.95; Precision/Recall ≥ 0.93.
- Threshold: default decision threshold is 0.5; lower (e.g., 0.4) to increase Recall at some Precision cost.

### Artifacts
- Saved to `models/` after training:
  - `spam_tfidf_vectorizer.joblib` — vocabulary and IDF weights.
  - `spam_logreg_model.joblib` — trained classifier.
  - `spam_label_mapping.json` — mapping of numeric classes back to labels (e.g., 1→`spam`, 0→`ham`).
- Purpose: enable consistent, reproducible inference without retraining.

### Inference
- Single text:
  - `python scripts/predict_spam.py --text "Free entry in 2 a wkly comp to win cash"`
- Batch CSV:
  - `python scripts/predict_spam.py --input datasets/processed/sms_spam_clean.csv --text-col text_clean --output predictions.csv`

## Detailed Walkthrough: Clean Text -> Features -> Classification

This walkthrough shows how a cleaned message becomes numeric features and how the classifier makes a decision.

### Example message (before/after)
- Raw (from datasets/sms_spam_no_header.csv):
  - "spam","Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
- Cleaned (text_clean):
  - "free entry in <NUM> a wkly comp to win fa cup final tkts <NUM>st may <NUM> text fa to <NUM> to receive entry question std txt rate t c s apply <NUM>over<NUM> s"

The cleaning step preserves informative tokens (free, win, <NUM>, wkly, tkts) and removes punctuation/case that add noise.

### 1) Tokenization and vocabulary
- The TF-IDF vectorizer splits the cleaned text into tokens (unigrams). Example tokens:
  - [free, entry, in, <NUM>, a, wkly, comp, to, win, fa, cup, final, tkts, <NUM>st, may, <NUM>, text, fa, to, <NUM>, receive, entry, question, std, txt, rate, t, c, s, apply, <NUM>over<NUM>, s]
- Over the training set, a vocabulary maps tokens to column indices, e.g.:
  - {"free": 1021, "entry": 874, "win": 4321, "<NUM>": 77, "wkly": 5603, ...}
  - Note: indices depend on the actual training data.

### 2) TF-IDF feature vector
- For one message, tf counts token frequency; idf down-weights tokens common across the corpus; rows are L2-normalized.
- Illustrative non-zero weights for the example (not exact):
  - free: 0.42, entry: 0.23, win: 0.31, <NUM>: 0.18, wkly: 0.29, comp: 0.22, tkts: 0.26, text: 0.10
- Interpretation:
  - Higher values -> token is relatively important for this message and discriminative overall (high idf).
  - The vector is sparse: only tokens present in the message have non-zero entries; all others are 0.

### 3) Train/test split
- We use 80/20 with random_state=42 and stratify=labels, so spam/ham ratio is preserved.
- The vectorizer is fit on the training texts only, then applied to the test texts.

### 4) Logistic regression decision
- The model learns weights w and bias b. For a feature vector x: z = w·x + b, prob(spam) = 1/(1 + exp(-z)).
- Qualitatively: tokens like free/win/cash/claim/<NUM> often push z upward (spam), while ok/thanks/see/tonight might push z downward (ham).
- Default threshold is 0.5. You can change it at predict time (e.g., --threshold 0.4) to trade precision vs recall.

### 5) Example classification
- Command: python scripts/predict_spam.py --text "Free entry in 2 a wkly comp to win cash"
- Example output: label=spam    prob=0.78
- Meaning: at threshold 0.5 this is classified as spam. Lower/higher thresholds change the decision boundary.

### 6) Test-set metrics (held-out)
- We report Accuracy, Precision, Recall, F1 on the 20% test split.
- Example (varies by split/options):
  - Accuracy: 0.9776
  - Precision: 0.9769
  - Recall: 0.8523
  - F1: 0.9104
- Improving recall: lower threshold (e.g., 0.4), set class_weight="balanced", or include bi-grams in TF-IDF (ngram_range=(1,2)).

### 7) Batch inference
- Command: python scripts/predict_spam.py --input datasets/processed/sms_spam_clean.csv --text-col text_clean --output predictions.csv
- Output columns: pred_label (spam|ham), pred_prob (0..1).

### 8) Reproducibility via artifacts
- Saved artifacts under models/:
  - spam_tfidf_vectorizer.joblib (vocabulary + IDF)
  - spam_logreg_model.joblib (weights)
  - spam_label_mapping.json (numeric->text labels)
- Using these ensures future predictions use the same feature mapping and decision boundary.

## Appendix: Toy TF-IDF Example (exact numbers)

This appendix computes TF-IDF exactly as scikit-learn does by default:
- IDF formula: idf(t) = ln((1 + N) / (1 + df(t))) + 1, where N is number of documents and df(t) is the document frequency of token t.
- Row normalization: each TF-IDF row is L2-normalized (divide by sqrt(sum of squares)).

Toy corpus (3 short docs):
- D1: "free win cash"
- D2: "free free entry now"
- D3: "see you tonight"

Document frequencies (df) and IDF (N = 3):
- df(free)=2  → idf(free) = ln((1+3)/(1+2)) + 1 = ln(4/3) + 1 ≈ 0.287682 + 1 = 1.287682
- df(win)=1   → idf(win)  = ln(4/2) + 1 = ln(2) + 1 ≈ 0.693147 + 1 = 1.693147
- df(cash)=1  → idf(cash) = 1.693147
- df(entry)=1 → idf(entry)= 1.693147
- df(now)=1   → idf(now)  = 1.693147
- df(see)=1   → idf(see)  = 1.693147
- df(you)=1   → idf(you)  = 1.693147
- df(tonight)=1 → idf(tonight)= 1.693147

Term frequencies (tf) and pre-normalized TF-IDF (tf * idf):
- D1 tokens: free(1), win(1), cash(1)
  - TF-IDF: free=1.287682, win=1.693147, cash=1.693147
  - L2 norm = sqrt(1.287682^2 + 1.693147^2 + 1.693147^2)
             ≈ sqrt(1.658 + 2.868 + 2.868) ≈ sqrt(7.394) ≈ 2.720
  - Normalized vector (non-zero entries):
    - free ≈ 1.287682 / 2.720 ≈ 0.473
    - win  ≈ 1.693147 / 2.720 ≈ 0.623
    - cash ≈ 1.693147 / 2.720 ≈ 0.623

- D2 tokens: free(2), entry(1), now(1)
  - TF-IDF: free=2*1.287682=2.575364, entry=1.693147, now=1.693147
  - L2 norm = sqrt(2.575364^2 + 1.693147^2 + 1.693147^2)
             ≈ sqrt(6.635 + 2.868 + 2.868) ≈ sqrt(12.371) ≈ 3.519
  - Normalized vector:
    - free ≈ 2.575364 / 3.519 ≈ 0.732
    - entry ≈ 1.693147 / 3.519 ≈ 0.481
    - now  ≈ 1.693147 / 3.519 ≈ 0.481

- D3 tokens: see(1), you(1), tonight(1)
  - TF-IDF: each = 1.693147
  - L2 norm = sqrt(3 * 1.693147^2) = sqrt(3 * 2.868) = sqrt(8.604) ≈ 2.934
  - Normalized vector:
    - see ≈ 1.693147 / 2.934 ≈ 0.577
    - you ≈ 1.693147 / 2.934 ≈ 0.577
    - tonight ≈ 1.693147 / 2.934 ≈ 0.577

Interpretation:
- Tokens unique to fewer documents (higher idf) tend to have larger weights when they appear.
- The final feature vectors are sparse and L2-normalized rows, exactly matching scikit-learn defaults used in this project.
- Optional: `--threshold 0.4` to trade Precision for higher Recall.

### End‑to‑End Commands
- Preprocess:
  - `python scripts/preprocess_emails.py --input datasets/sms_spam_no_header.csv --output datasets/processed/sms_spam_clean.csv --no-header --label-col-index 0 --text-col-index 1 --output-text-col text_clean`
- Train:
  - `python scripts/train_spam_classifier.py --input datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col text_clean`
- Predict (single):
  - `python scripts/predict_spam.py --text "You won a prize. Claim now!"`
- Predict (batch):
  - `python scripts/predict_spam.py --input datasets/processed/sms_spam_clean.csv --text-col text_clean --output predictions.csv`
