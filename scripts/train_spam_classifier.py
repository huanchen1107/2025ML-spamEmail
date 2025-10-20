import argparse
import json
import os
import sys
from typing import Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def find_default_dataset() -> str:
    processed = os.path.join("datasets", "processed", "sms_spam_clean.csv")
    raw = os.path.join("datasets", "sms_spam_no_header.csv")
    return processed if os.path.exists(processed) else raw


def read_dataset(path: str, no_header: bool) -> pd.DataFrame:
    if no_header:
        return pd.read_csv(path, header=None)
    return pd.read_csv(path)


def resolve_columns(df: pd.DataFrame, label_col: Optional[str], text_col: Optional[str],
                    label_idx: Optional[int], text_idx: Optional[int]) -> Tuple[str, str]:
    if label_col is None and label_idx is not None:
        label_col = f"col_{label_idx}"
        df.rename(columns={label_idx: label_col}, inplace=True)
    if text_col is None and text_idx is not None:
        text_col = f"col_{text_idx}"
        df.rename(columns={text_idx: text_col}, inplace=True)
    if label_col is None or text_col is None:
        raise ValueError("Must provide --label-col/--text-col or their index variants when no header is present")
    return label_col, text_col


def map_labels(series: pd.Series, positive_label: Optional[str], negative_label: Optional[str]) -> Tuple[np.ndarray, dict]:
    vals = series.astype(str).str.lower().values
    mapping = {}
    if positive_label and negative_label:
        pos = positive_label.lower()
        neg = negative_label.lower()
        y = np.array([1 if v == pos else 0 for v in vals])
        mapping = {1: positive_label, 0: negative_label}
        return y, mapping
    unique = sorted(set(vals))
    if {"spam", "ham"}.issubset(set(unique)):
        y = np.array([1 if v == "spam" else 0 for v in vals])
        mapping = {1: "spam", 0: "ham"}
    elif set(unique).issubset({"0", "1"}):
        y = np.array([int(v) for v in vals])
        mapping = {1: "spam", 0: "ham"}
    else:
        first = list(unique)
        if len(first) != 2:
            raise ValueError(f"Unsupported label set: {unique}. Provide --positive-label/--negative-label")
        pos_val = first[1]
        y = np.array([1 if v == pos_val else 0 for v in vals])
        mapping = {1: pos_val, 0: first[0]}
    return y, mapping


def main() -> int:
    p = argparse.ArgumentParser(description="Train a spam classifier (TF-IDF + LogisticRegression)")
    p.add_argument("--input", default=find_default_dataset(), help="Input CSV path (clean preferred)")
    p.add_argument("--no-header", action="store_true", help="Treat CSV as having no header")
    p.add_argument("--label-col", help="Label column name when CSV has header")
    p.add_argument("--text-col", help="Text column name when CSV has header")
    p.add_argument("--label-col-index", type=int, help="Label column index when no header")
    p.add_argument("--text-col-index", type=int, help="Text column index when no header")
    p.add_argument("--positive-label", help="Explicit positive class label (e.g., spam)")
    p.add_argument("--negative-label", help="Explicit negative class label (e.g., ham)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--models-dir", default=os.path.join("models"))
    p.add_argument("--steps-out-dir", default=None, help="Optional directory to save CSVs for each training sub-step under datasets/processed/steps style")
    p.add_argument("--topk", type=int, default=10, help="Top-K TF-IDF tokens to export per document when saving steps")
    # recall-improving knobs
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    p.add_argument("--ngram-range", default="1,1", help="n-gram range as 'min,max' (e.g., '1,2' for unigrams+bigrams)")
    p.add_argument("--min-df", type=int, default=1, help="Ignore terms with document frequency < min-df")
    p.add_argument("--sublinear-tf", action="store_true", help="Use sublinear tf scaling in TF-IDF")
    p.add_argument("--eval-threshold", type=float, default=0.5, help="Threshold for converting probabilities to labels during evaluation")

    args = p.parse_args()

    df = read_dataset(args.input, args.no_header)
    label_col, text_col = resolve_columns(df, args.label_col, args.text_col, args.label_col_index, args.text_col_index)

    y, mapping = map_labels(df[label_col], args.positive_label, args.negative_label)
    X_text = df[text_col].astype(str).fillna("")

    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    # parse ngram range
    try:
        ngram_parts = [int(x.strip()) for x in args.ngram_range.split(",")]
        if len(ngram_parts) != 2:
            raise ValueError
        ngram_tuple = (ngram_parts[0], ngram_parts[1])
    except Exception:
        raise ValueError("--ngram-range must be like '1,2'")

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_tuple,
        min_df=args.min_df,
        sublinear_tf=args.sublinear_tf,
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    class_weight = None if args.class_weight == "none" else "balanced"
    clf = LogisticRegression(max_iter=args.max_iter, C=args.C, class_weight=class_weight)
    clf.fit(Xtr, y_train)

    y_proba = clf.predict_proba(Xte)[:, 1]
    # allow custom threshold for evaluation
    y_pred = (y_proba >= args.eval_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

    # Optionally export step CSVs for downstream reporting
    if args.steps_out_dir:
        os.makedirs(args.steps_out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        # 07: cleaned text snapshot (label + text)
        out07 = os.path.join(args.steps_out_dir, f"{base_name}_07_text_clean.csv")
        df[[label_col, text_col]].to_csv(out07, index=False)
        # 08/09: train/test splits
        import pandas as _pd
        _train_df = _pd.DataFrame({label_col: y_train, text_col: X_train})
        _test_df = _pd.DataFrame({label_col: y_test, text_col: X_test})
        out08 = os.path.join(args.steps_out_dir, f"{base_name}_08_train_split.csv")
        out09 = os.path.join(args.steps_out_dir, f"{base_name}_09_test_split.csv")
        _train_df.to_csv(out08, index=False)
        _test_df.to_csv(out09, index=False)
        # 10: vocabulary with idf
        vocab_items: List[Tuple[str, int]] = sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])
        # Align idf_ by feature index
        tokens = [tok for tok, idx in vocab_items]
        indices = [idx for tok, idx in vocab_items]
        idf_vals = vec.idf_[indices]
        _vdf = _pd.DataFrame({"token": tokens, "index": indices, "idf": idf_vals})
        out10 = os.path.join(args.steps_out_dir, f"{base_name}_10_vocab.csv")
        _vdf.to_csv(out10, index=False)
        # 11/12: top-K tf-idf tokens per doc (train/test)
        def topk_for_matrix(X, k: int):
            rows = []
            # iterate rows
            for i in range(X.shape[0]):
                row = X.getrow(i)
                if row.nnz == 0:
                    rows.append({"doc_index": i, "top_tokens": ""})
                    continue
                data = row.data
                idxs = row.indices
                order = data.argsort()[::-1]
                top = order[: min(k, len(order))]
                items = []
                for j in top:
                    tok_idx = idxs[j]
                    # reverse map: index -> token
                    # We can index tokens list by tok_idx since it's aligned to indices
                    tok = tokens[tok_idx] if tok_idx < len(tokens) else str(tok_idx)
                    items.append(f"{tok}:{data[j]:.4f}")
                rows.append({"doc_index": i, "top_tokens": "; ".join(items)})
            return _pd.DataFrame(rows)

        out11 = os.path.join(args.steps_out_dir, f"{base_name}_11_tfidf_train_top.csv")
        out12 = os.path.join(args.steps_out_dir, f"{base_name}_12_tfidf_test_top.csv")
        topk_for_matrix(Xtr, args.topk).to_csv(out11, index=False)
        topk_for_matrix(Xte, args.topk).to_csv(out12, index=False)
        # 13: metrics
        out13 = os.path.join(args.steps_out_dir, f"{base_name}_13_metrics.csv")
        _pd.DataFrame([
            {"metric": "accuracy", "value": acc},
            {"metric": "precision", "value": prec},
            {"metric": "recall", "value": rec},
            {"metric": "f1", "value": f1},
        ]).to_csv(out13, index=False)
        # 14: test predictions with text/labels
        out14 = os.path.join(args.steps_out_dir, f"{base_name}_14_test_predictions.csv")
        # map back labels to strings
        pos_label = mapping.get(1, "spam")
        neg_label = mapping.get(0, "ham")
        pred_labels = np.where(y_proba >= 0.5, pos_label, neg_label)
        true_labels = np.where(y_test == 1, pos_label, neg_label)
        _pd.DataFrame({
            text_col: X_test.values,
            "true_label": true_labels,
            "pred_label": pred_labels,
            "pred_prob": y_proba,
        }).to_csv(out14, index=False)

    os.makedirs(args.models_dir, exist_ok=True)
    vec_path = os.path.join(args.models_dir, "spam_tfidf_vectorizer.joblib")
    model_path = os.path.join(args.models_dir, "spam_logreg_model.joblib")
    meta_path = os.path.join(args.models_dir, "spam_label_mapping.json")
    joblib.dump(vec, vec_path)
    joblib.dump(clf, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"positive": mapping.get(1, "spam"), "negative": mapping.get(0, "ham")}, f)
    print(f"Saved: {vec_path}")
    print(f"Saved: {model_path}")
    print(f"Saved: {meta_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
