## 1. Implementation
- [ ] Create `scripts/visualize_spam.py` with subcommands/flags:
      - Inputs: `--input` cleaned CSV, `--label-col`, `--text-col`, `--models-dir`, `--outdir` (default: reports/visualizations)
      - Plots: `--class-dist`, `--token-freq`, `--confusion-matrix`, `--roc`, `--pr`, `--threshold-sweep`
      - Token freq: top-N per class (default: 20), simple tokenization on cleaned text
      - Use `ConfusionMatrixDisplay`, `RocCurveDisplay`, `PrecisionRecallDisplay`
- [ ] Add `matplotlib` and `seaborn` to `requirements.txt`
- [ ] Create `reports/visualizations/` and write outputs with timestamped filenames

## 2. Validation
- [ ] Run visualizations on current cleaned dataset and trained model artifacts
- [ ] Ensure images (PNG) are produced without errors and are readable
- [ ] Add README section with example commands and sample images list
