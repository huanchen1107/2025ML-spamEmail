## Why
After recall tuning, Recall is about 0.97 but Precision is about 0.85. We need to regain Precision to >= 0.90 while keeping Recall >= 0.93.

## What Changes
- Tune evaluation threshold and TF-IDF params (min_df, n-grams) and regularization (C) to raise Precision.
- Provide a recommended configuration and document trade-offs.

## Impact
- No new flags required (existing training flags suffice). Update README guidance.

## Out of Scope
- Dataset relabeling or external data collection.

## Recommended Configuration (approved)
- class_weight: balanced
- ngram_range: 1,2 (unigrams + bigrams)
- min_df: 2
- sublinear_tf: true
- C: 2.0
- eval_threshold: 0.50 (evaluation threshold)

Command
```
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

Observed Metrics (held-out test)
- Accuracy: 0.9848
- Precision: 0.9231
- Recall: 0.9664
- F1: 0.9443
