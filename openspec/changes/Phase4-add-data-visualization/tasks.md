## 1. Implementation
- [x] Create `scripts/visualize_spam.py` with subcommands/flags:
      - Inputs: `--input` cleaned CSV, `--label-col`, `--text-col`, `--models-dir`, `--outdir` (default: reports/visualizations)
      - Plots: `--class-dist`, `--token-freq`, `--confusion-matrix`, `--roc`, `--pr`, `--threshold-sweep`
      - Token freq: top-N per class (default: 20), simple tokenization on cleaned text
      - Use `ConfusionMatrixDisplay`, `RocCurveDisplay`, `PrecisionRecallDisplay`
- [x] Add `matplotlib`, `seaborn`, and `streamlit` to `requirements.txt`
- [ ] Create `reports/visualizations/` and write outputs with timestamped filenames

## 2. Validation
- [x] Run visualizations on current cleaned dataset and trained model artifacts
- [x] Ensure images (PNG) are produced without errors and are readable
- [ ] Add README section with example commands and sample images list

- [x] Add Streamlit app `app/streamlit_app.py` with:
      - Dataset/model selectors, label/text column pickers
      - Class distribution, top tokens by class
      - Confusion matrix, ROC/PR (requires artifacts)
      - Threshold slider with live metrics table
      - Live inference text box that returns label + probability
      - Notes to run: `streamlit run app/streamlit_app.py`
## 3. Live Inference (separated)
- [x] 3.1 Add input text box to accept a message and return predicted label + spam probability.
- [x] 3.2 Add associated visualization for the single prediction:
      - Probability bar/gauge with threshold marker
      - Optional token-based context (e.g., highlight presence of top spam/ham tokens)
- [x] 3.3 Document this feature in README with a screenshot and usage notes.


