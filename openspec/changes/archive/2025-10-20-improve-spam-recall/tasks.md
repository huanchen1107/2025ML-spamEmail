## 1. Implementation
- [x] Add flags to training script: `--class-weight (balanced|none)`, `--ngram-range 1,2`, `--min-df`, `--sublinear-tf`, `--eval-threshold`.
- [x] Retrain with: class_weight=balanced, ngram_range=(1,2), min_df=2, sublinear_tf on, eval threshold=0.40.
- [x] Save updated metrics and compare against baseline.

## 2. Validation
- [x] Ensure Recall ≥ 0.93 on held-out split, Precision remains reasonable (≥ 0.90 preferred).
- [x] Document final command lines in README.

