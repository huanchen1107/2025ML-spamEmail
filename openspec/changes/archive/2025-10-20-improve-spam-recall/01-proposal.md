## Why
Recall on the spam class (~0.85) is below the target (≥ 0.93). We want to catch more spam with acceptable precision.

## What Changes
- Add training options to improve recall: class weighting, bi-gram features, min-df filtering, sublinear TF, and configurable eval threshold.
- Retrain with tuned settings and document metrics.

## Impact
- scripts/train_spam_classifier.py: new flags (`--class-weight`, `--ngram-range`, `--min-df`, `--sublinear-tf`, `--eval-threshold`).
- docs: update guidance on recall/precision trade-offs.

## Out of Scope
- Deep learning models; dataset expansion.
