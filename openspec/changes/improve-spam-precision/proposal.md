## Why
After recall tuning, Recall ≈ 0.97 but Precision ≈ 0.85. We need to regain Precision to ≥ 0.90 while keeping Recall ≥ 0.93.

## What Changes
- Tune evaluation threshold and TF-IDF params (min_df, n-grams) and regularization (C) to raise Precision.
- Provide a recommended configuration and document trade-offs.

## Impact
- No new flags required (existing training flags suffice). Update README guidance.

## Out of Scope
- Dataset relabeling or external data collection.
