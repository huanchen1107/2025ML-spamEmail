import argparse
import json
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd


def load_artifacts(models_dir: str):
    vec_path = os.path.join(models_dir, "spam_tfidf_vectorizer.joblib")
    model_path = os.path.join(models_dir, "spam_logreg_model.joblib")
    meta_path = os.path.join(models_dir, "spam_label_mapping.json")
    if not (os.path.exists(vec_path) and os.path.exists(model_path)):
        raise FileNotFoundError("Model artifacts not found. Train first.")
    vec = joblib.load(vec_path)
    clf = joblib.load(model_path)
    positive = "spam"
    negative = "ham"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                positive = meta.get("positive", positive)
                negative = meta.get("negative", negative)
        except Exception:
            pass
    return vec, clf, positive, negative


def predict_text(text: str, vec, clf, positive_label: str, negative_label: str, threshold: float = 0.5):
    X = vec.transform([str(text)])
    proba = float(clf.predict_proba(X)[:, 1][0])
    label = positive_label if proba >= threshold else negative_label
    return label, proba


def main() -> int:
    p = argparse.ArgumentParser(description="Predict spam/ham using trained artifacts")
    p.add_argument("--text", help="Single input text to classify")
    p.add_argument("--input", help="CSV file with a text column to batch predict")
    p.add_argument("--text-col", default="text", help="Text column name for batch input")
    p.add_argument("--output", help="Output CSV path for batch predictions")
    p.add_argument("--models-dir", default=os.path.join("models"))
    p.add_argument("--threshold", type=float, default=0.5)

    args = p.parse_args()

    vec, clf, positive_label, negative_label = load_artifacts(args.models_dir)

    if args.text:
        label, proba = predict_text(args.text, vec, clf, positive_label, negative_label, args.threshold)
        print(f"label={label}\tprob={proba:.4f}")
        return 0

    if args.input:
        if not args.output:
            print("--output is required for batch prediction", file=sys.stderr)
            return 2
        df = pd.read_csv(args.input)
        if args.text_col not in df.columns:
            print(f"Text column '{args.text_col}' not found in input CSV", file=sys.stderr)
            return 2
        X = vec.transform(df[args.text_col].astype(str).fillna(""))
        proba = clf.predict_proba(X)[:, 1]
        labels = np.where(proba >= args.threshold, positive_label, negative_label)
        out = df.copy()
        out["pred_label"] = labels
        out["pred_prob"] = proba
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"Wrote predictions to: {args.output}")
        return 0

    print("Provide either --text or --input for batch mode", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

