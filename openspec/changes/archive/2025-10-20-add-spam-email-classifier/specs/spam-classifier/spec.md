## ADDED Requirements

### Requirement: Spam Email Classification
The system MUST classify text messages/emails as `spam` or `ham` using a lightweight, CPU-friendly ML pipeline.

#### Scenario: Train baseline classifier
- WHEN `python scripts/train_spam_classifier.py` runs with the default dataset
- THEN it saves `models/spam_tfidf_vectorizer.joblib` and `models/spam_logreg_model.joblib`
- AND it prints Accuracy, Precision, Recall, and F1 to stdout
- AND Accuracy >= 0.95, Precision >= 0.93, Recall >= 0.93 on the held-out test set

#### Scenario: Predict single text
- GIVEN trained artifacts exist
- WHEN `python scripts/predict_spam.py --text "Free entry in 2 a wkly comp to win cash"` runs
- THEN it outputs a label (`spam`|`ham`) and a probability in stdout

#### Scenario: Batch predict from CSV
- GIVEN trained artifacts exist
- WHEN `python scripts/predict_spam.py --input datasets/some_messages.csv --output predictions.csv`
- THEN it writes a CSV including original text and predicted label/probability to `predictions.csv`

#### Scenario: Reproducible training
- WHEN training is run with `--seed 42` on the same dataset
- THEN it produces consistent metrics within a small tolerance (e.g., +/- 0.5 percentage points)

#### Scenario: Runtime constraints
- WHEN training and inference run on a CPU-only laptop
- THEN training completes in under 2 minutes and single-text inference is sub-100ms

#### Scenario: Dependencies
- WHEN setting up the environment
- THEN the solution uses Python >= 3.9 and the following libraries: scikit-learn, pandas, numpy, joblib
