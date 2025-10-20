import argparse
import os
import sys
import time
from collections import Counter
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    f1_score,
)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_outdir(path: str) -> None:
    os.makedirs(path or ".", exist_ok=True)


def load_artifacts(models_dir: str):
    vec_path = os.path.join(models_dir, "spam_tfidf_vectorizer.joblib")
    model_path = os.path.join(models_dir, "spam_logreg_model.joblib")
    meta_path = os.path.join(models_dir, "spam_label_mapping.json")
    if not (os.path.exists(vec_path) and os.path.exists(model_path)):
        raise FileNotFoundError("Model artifacts not found. Train first.")
    vec = joblib.load(vec_path)
    clf = joblib.load(model_path)
    pos, neg = "spam", "ham"
    if os.path.exists(meta_path):
        try:
            import json

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                pos = meta.get("positive", pos)
                neg = meta.get("negative", neg)
        except Exception:
            pass
    return vec, clf, pos, neg


def split_xy(df: pd.DataFrame, label_col: str, text_col: str, test_size: float, seed: int):
    y_text = df[label_col].astype(str).str.lower()
    # Map textual labels to 0/1 using common convention
    # try spam/ham first, else infer most frequent as negative
    classes = sorted(y_text.unique())
    if set(["spam", "ham"]).issubset(set(classes)):
        y = (y_text == "spam").astype(int).values
    else:
        # fallback: sort by frequency and pick last as positive
        freq = y_text.value_counts()
        neg_label = freq.index[0]
        y = (y_text != neg_label).astype(int).values
    X = df[text_col].astype(str).fillna("")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Xtr, Xte, ytr, yte


def plot_class_dist(df: pd.DataFrame, label_col: str, outdir: str) -> str:
    ensure_outdir(outdir)
    ts = timestamp()
    counts = df[label_col].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="Blues_d")
    plt.title("Class Distribution")
    plt.xlabel(label_col)
    plt.ylabel("count")
    fname = os.path.join(outdir, f"class_distribution_{ts}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def top_tokens(texts: pd.Series, topn: int) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for t in texts:
        counter.update(str(t).split())
    return counter.most_common(topn)


def plot_token_freq(df: pd.DataFrame, label_col: str, text_col: str, outdir: str, topn: int = 20) -> List[str]:
    ensure_outdir(outdir)
    ts = timestamp()
    paths: List[str] = []
    for label, sub in df.groupby(label_col):
        top = top_tokens(sub[text_col].astype(str), topn)
        if not top:
            continue
        toks, freqs = zip(*top)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(freqs), y=list(toks), palette="viridis")
        plt.title(f"Top {topn} tokens for class {label}")
        plt.xlabel("frequency")
        plt.ylabel("token")
        fname = os.path.join(outdir, f"token_freq_{label}_{ts}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        paths.append(fname)
    return paths


def plot_confusion_roc_pr(
    df: pd.DataFrame,
    label_col: str,
    text_col: str,
    models_dir: str,
    outdir: str,
    test_size: float,
    seed: int,
) -> List[str]:
    ensure_outdir(outdir)
    vec, clf, pos_label, neg_label = load_artifacts(models_dir)
    Xtr, Xte, ytr, yte = split_xy(df, label_col, text_col, test_size, seed)
    Xte_vec = vec.transform(Xte)
    y_proba = clf.predict_proba(Xte_vec)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    ts = timestamp()
    paths: List[str] = []

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(yte, y_pred, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix (test)")
    fname_cm = os.path.join(outdir, f"confusion_matrix_{ts}.png")
    plt.tight_layout()
    plt.savefig(fname_cm)
    plt.close(fig)
    paths.append(fname_cm)

    # ROC
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(yte, y_proba, ax=ax)
    ax.set_title("ROC Curve (test)")
    fname_roc = os.path.join(outdir, f"roc_curve_{ts}.png")
    plt.tight_layout()
    plt.savefig(fname_roc)
    plt.close(fig)
    paths.append(fname_roc)

    # PR
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(yte, y_proba, ax=ax)
    ax.set_title("Precision-Recall (test)")
    fname_pr = os.path.join(outdir, f"pr_curve_{ts}.png")
    plt.tight_layout()
    plt.savefig(fname_pr)
    plt.close(fig)
    paths.append(fname_pr)

    return paths


def threshold_sweep(
    df: pd.DataFrame,
    label_col: str,
    text_col: str,
    models_dir: str,
    outdir: str,
    test_size: float,
    seed: int,
    start: float = 0.3,
    stop: float = 0.8,
    step: float = 0.05,
) -> Tuple[str, str]:
    ensure_outdir(outdir)
    vec, clf, _, _ = load_artifacts(models_dir)
    Xtr, Xte, ytr, yte = split_xy(df, label_col, text_col, test_size, seed)
    y_proba = clf.predict_proba(vec.transform(Xte))[:, 1]
    rows = []
    thr = start
    while thr <= (stop + 1e-9):
        y_pred = (y_proba >= thr).astype(int)
        prec = precision_score(yte, y_pred, zero_division=0)
        rec = recall_score(yte, y_pred, zero_division=0)
        f1 = f1_score(yte, y_pred, zero_division=0)
        rows.append({"threshold": round(thr, 3), "precision": prec, "recall": rec, "f1": f1})
        thr += step
    df_out = pd.DataFrame(rows)
    ts = timestamp()
    csv_path = os.path.join(outdir, f"threshold_sweep_{ts}.csv")
    plot_path = os.path.join(outdir, f"threshold_sweep_{ts}.png")
    df_out.to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(df_out["threshold"], df_out["precision"], label="precision")
    plt.plot(df_out["threshold"], df_out["recall"], label="recall")
    plt.plot(df_out["threshold"], df_out["f1"], label="f1")
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title("Threshold vs Precision/Recall/F1 (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return csv_path, plot_path


def main() -> int:
    p = argparse.ArgumentParser(description="Generate visual reports for spam classifier")
    p.add_argument("--input", required=True, help="Cleaned CSV path")
    p.add_argument("--label-col", required=True)
    p.add_argument("--text-col", default="text_clean")
    p.add_argument("--models-dir", default=os.path.join("models"))
    p.add_argument("--outdir", default=os.path.join("reports", "visualizations"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # Which visuals
    p.add_argument("--class-dist", action="store_true")
    p.add_argument("--token-freq", action="store_true")
    p.add_argument("--confusion-matrix", action="store_true")
    p.add_argument("--roc", action="store_true")
    p.add_argument("--pr", action="store_true")
    p.add_argument("--threshold-sweep", action="store_true")
    p.add_argument("--topn", type=int, default=20, help="Top-N tokens per class for token-freq")

    args = p.parse_args()

    df = pd.read_csv(args.input)
    outputs: List[str] = []

    if args.class_dist:
        outputs.append(plot_class_dist(df, args.label_col, args.outdir))

    if args.token_freq:
        outputs.extend(plot_token_freq(df, args.label_col, args.text_col, args.outdir, topn=args.topn))

    # For model-based visuals, ensure artifacts exist
    if args.confusion_matrix or args.roc or args.pr:
        paths = plot_confusion_roc_pr(
            df,
            args.label_col,
            args.text_col,
            args.models_dir,
            args.outdir,
            args.test_size,
            args.seed,
        )
        outputs.extend(paths)

    if args.threshold_sweep:
        csv_path, plot_path = threshold_sweep(
            df,
            args.label_col,
            args.text_col,
            args.models_dir,
            args.outdir,
            args.test_size,
            args.seed,
        )
        outputs.extend([csv_path, plot_path])

    if not outputs:
        print("No visualization flags provided. Use --class-dist/--token-freq/--confusion-matrix/--roc/--pr/--threshold-sweep.", file=sys.stderr)
        return 2

    print("Generated outputs:")
    for pth in outputs:
        print(pth)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

