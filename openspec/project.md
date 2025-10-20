# Project Context

## Purpose
Build a simple, reproducible machine‑learning solution to classify messages/emails as spam or not spam for AIoT‑DA2025 HW3. Provide a baseline pipeline, clear metrics, and easy local workflows for training and inference.

## Tech Stack
- Python 3.9–3.12 (Windows)
- Jupyter Notebooks (exploration under `sources/`)
- scikit‑learn, pandas, numpy, scipy, joblib (optional: nltk for tokenization in notebooks)
- Node.js LTS and OpenSpec CLI for spec‑driven workflow
- PowerShell scripts (Windows‑friendly commands)

## Project Conventions
- Code Style: PEP8; prefer type hints; small functions; avoid hidden globals
- CLI Scripts: `argparse` with `--help`; explicit `--input/--output` paths; deterministic `--seed`
- Project Layout:
  - `datasets/` raw CSVs (e.g., `sms_spam_no_header.csv`)
  - `scripts/` training and inference CLIs
  - `models/` saved artifacts (`.joblib`), can be git‑ignored if large
  - `sources/` exploratory notebooks
  - `openspec/` specs and change proposals
- Reproducibility: fixed seed (e.g., 42), capture library versions in `requirements.txt`

## Architecture Patterns
- Lightweight, single‑file scripts over frameworks
- TF‑IDF vectorizer + linear classifier (Logistic Regression baseline)
- Separate training and prediction scripts; model artifacts saved via joblib

## Testing Strategy
- Hold‑out split (e.g., 80/20) with fixed seed; report Accuracy, Precision, Recall, F1
- Quick smoke tests: run prediction on a few known spam/ham examples
- Optional unit tests for text cleaning/tokenization if added

## Git Workflow
- Main branch for stable work; feature branches optional
- Changes proposed via OpenSpec under `openspec/changes/<change-id>/`
- Commit messages: short, imperative; reference change id when applicable (e.g., `feat(spam-classifier): add training script [add-spam-email-classifier]`)

## Domain Context
- Goal: binary classification: `spam` vs `ham`
- Typical pitfalls: class imbalance; overfitting with small datasets; noisy tokens (URLs, numbers)
- Datasets present: `datasets/sms_spam_no_header.csv` (and others in `datasets/`)

## Important Constraints
- CPU‑only; training under ~2 minutes; low memory footprint
- Offline‑friendly; no external API calls required to run
- Deterministic results (seeded), suitable for grading and reproducibility

## External Dependencies
- Node.js + OpenSpec CLI (installed globally) for managing specs/changes
- Python packages via pip (scikit‑learn, pandas, numpy, scipy, joblib)
