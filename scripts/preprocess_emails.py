import argparse
import os
import re
import sys
import pandas as pd
from typing import Optional, Dict


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b")


def normalize_text(text: str, keep_numbers: bool = False, remove_stopwords: bool = False,
                   stopwords: Optional[set] = None) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    t = text.lower()
    t = URL_RE.sub("<URL>", t)
    t = EMAIL_RE.sub("<EMAIL>", t)
    t = PHONE_RE.sub("<PHONE>", t)
    if not keep_numbers:
        t = re.sub(r"\d+", "<NUM>", t)
    t = re.sub(r"[^\w\s<>]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if remove_stopwords and stopwords:
        tokens = [tok for tok in t.split() if tok not in stopwords]
        t = " ".join(tokens)
    return t


def resolve_columns(df: pd.DataFrame, label_col: Optional[str], text_col: Optional[str],
                    label_idx: Optional[int], text_idx: Optional[int]) -> tuple[str, str]:
    if label_col is None and label_idx is not None:
        label_col = f"col_{label_idx}"
        df.rename(columns={label_idx: label_col}, inplace=True)
    if text_col is None and text_idx is not None:
        text_col = f"col_{text_idx}"
        df.rename(columns={text_idx: text_col}, inplace=True)
    if label_col is None or text_col is None:
        raise ValueError("Must provide --label-col/--text-col or their index variants when no header is present")
    return label_col, text_col


def main() -> int:
    p = argparse.ArgumentParser(description="Preprocess email/text dataset for spam classification")
    p.add_argument("--input", required=True, help="Input CSV path (raw)")
    p.add_argument("--output", required=True, help="Output CSV path (cleaned)")
    p.add_argument("--label-col", help="Label column name when CSV has header")
    p.add_argument("--text-col", help="Text column name when CSV has header")
    p.add_argument("--label-col-index", type=int, help="Label column index when CSV has no header")
    p.add_argument("--text-col-index", type=int, help="Text column index when CSV has no header")
    p.add_argument("--no-header", action="store_true", help="Treat CSV as having no header")
    p.add_argument("--output-text-col", default=None, help="Name for cleaned text column; defaults to text column name")
    p.add_argument("--keep-numbers", action="store_true", help="Keep raw numbers instead of replacing with <NUM>")
    p.add_argument("--remove-stopwords", action="store_true", help="Remove english stopwords from cleaned text")
    p.add_argument("--save-step-columns", action="store_true", help="Also include intermediate step columns in the final output CSV")
    p.add_argument("--steps-out-dir", default=None, help="Optional directory to save a CSV after each preprocessing step")

    args = p.parse_args()

    in_path = args.input
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if args.no_header:
        df = pd.read_csv(in_path, header=None)
    else:
        df = pd.read_csv(in_path)

    if args.remove_stopwords:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
            stopwords = set(SW)
        except Exception:
            stopwords = set()
    else:
        stopwords = set()

    label_col, text_col = resolve_columns(
        df,
        label_col=args.label_col,
        text_col=args.text_col,
        label_idx=args.label_col_index,
        text_idx=args.text_col_index,
    )

    out_text_col = args.output_text_col or text_col

    # Sequential, named steps for transparency and optional saving
    def step_lower(s: str) -> str:
        return str(s).lower()

    def step_mask_contacts(s: str) -> str:
        s2 = URL_RE.sub("<URL>", s)
        s2 = EMAIL_RE.sub("<EMAIL>", s2)
        s2 = PHONE_RE.sub("<PHONE>", s2)
        return s2

    def step_numbers(s: str) -> str:
        return s if args.keep_numbers else re.sub(r"\d+", "<NUM>", s)

    def step_strip_punct(s: str) -> str:
        return re.sub(r"[^\w\s<>]", " ", s)

    def step_whitespace(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def step_stopwords(s: str) -> str:
        if args.remove_stopwords and stopwords:
            tokens = [tok for tok in s.split() if tok not in stopwords]
            return " ".join(tokens)
        return s

    # Compute each step as a new Series
    base = df[text_col].astype(str)
    steps: Dict[str, pd.Series] = {}
    steps["text_lower"] = base.apply(step_lower)
    steps["text_contacts_masked"] = steps["text_lower"].apply(step_mask_contacts)
    steps["text_numbers"] = steps["text_contacts_masked"].apply(step_numbers)
    steps["text_stripped"] = steps["text_numbers"].apply(step_strip_punct)
    steps["text_whitespace"] = steps["text_stripped"].apply(step_whitespace)
    steps["text_stopwords_removed"] = steps["text_whitespace"].apply(step_stopwords)

    # Final cleaned output
    cleaned = steps["text_stopwords_removed"]
    df[out_text_col] = cleaned

    # Optionally include intermediate columns in the final output CSV
    if args.save_step_columns:
        for name, series in steps.items():
            # Avoid collision if out_text_col matches one of the step names
            if name == out_text_col:
                df[f"{name}_step"] = series
            else:
                df[name] = series

    # Optionally write per-step CSVs
    if args.steps_out_dir:
        os.makedirs(args.steps_out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(out_path))[0]
        for i, (name, series) in enumerate(steps.items(), start=1):
            tmp = df.copy()
            tmp[name] = series
            step_path = os.path.join(args.steps_out_dir, f"{base_name}_{i:02d}_{name}.csv")
            tmp.to_csv(step_path, index=False)

    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned dataset to: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
