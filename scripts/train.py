"""End-to-end training pipeline.

Stages (as required):
  data load -> preprocess -> feature engineering -> train ALL models ->
  compare -> save best ensemble -> print full evaluation report.

Models compared:
  1. Baseline : TF-IDF + Logistic Regression (text only)
  2. Tabular  : XGBoost on 25 engineered features (+ optional SMOTE)
  3. Text     : the ensemble's text branch (TF-IDF+LR, or DistilBERT if enabled)
  4. Ensemble : stacked meta-learner over (2) and (3)   <-- promoted artifact

Run:
  python scripts/train.py                 # synthetic fallback data
  python scripts/train.py --data data/fake_job_postings.csv
  python scripts/train.py --use-bert      # if torch+transformers installed
  python scripts/train.py --smote
"""
from __future__ import annotations

import sys

# IMPORTANT (Windows): torch's c10.dll fails to initialize (WinError 1114) if it
# is imported AFTER scikit-learn/xgboost, due to conflicting bundled OpenMP
# runtimes. When BERT is requested we must load torch FIRST, before any import
# that pulls in sklearn/xgboost. This is a no-op on Linux/macOS and when torch is
# absent.
if "--use-bert" in sys.argv:
    try:
        import torch  # noqa: F401
    except Exception:
        pass

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Make the repo root importable when run as `python scripts/train.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobguard.config import settings  # noqa: E402
from jobguard.dataset import load_dataset  # noqa: E402
from jobguard.ensemble import EnsembleModel  # noqa: E402
from jobguard.evaluation import (best_threshold_for_recall,  # noqa: E402
                                 classification_report_text, evaluate)


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _baseline_tfidf_lr(train_corpus, y_train, test_corpus):
    """Model 1: the simplest reasonable baseline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                                  max_features=30000)),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced",
                                  solver="liblinear")),
    ])
    pipe.fit(train_corpus, y_train)
    return pipe.predict_proba(test_corpus)[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the JobGuard fraud ensemble.")
    ap.add_argument("--data", default=None, help="Path to EMSCAD/Kaggle CSV.")
    ap.add_argument("--synthetic-n", type=int, default=4000)
    ap.add_argument("--use-bert", action="store_true", help="Use DistilBERT text branch.")
    ap.add_argument("--smote", action="store_true", help="Apply SMOTE to tabular branch.")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", default=None, help="Output bundle path.")
    ap.add_argument("--bert-epochs", type=int, default=2, help="DistilBERT fine-tune epochs.")
    ap.add_argument("--bert-max-len", type=int, default=192, help="DistilBERT token length.")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Subsample the dataset to at most N rows (0 = all). Handy "
                         "for CPU BERT training.")
    ap.add_argument("--n-splits", type=int, default=0,
                    help="OOF folds (0 = auto: 5 normally, 3 with --use-bert).")
    args = ap.parse_args()

    t0 = time.time()

    # ---------------------------------------------------------------- Stage 1
    _print_header("STAGE 1/5  Load data")
    postings, labels, source = load_dataset(args.data, synthetic_n=args.synthetic_n)
    labels = np.asarray(labels, dtype=int)

    # Optional stratified subsample (keeps the fraud rate) for fast CPU BERT runs.
    if args.max_rows and args.max_rows < len(labels):
        from sklearn.model_selection import train_test_split as _sub
        keep, _ = _sub(np.arange(len(labels)), train_size=args.max_rows,
                       stratify=labels, random_state=42)
        postings = [postings[i] for i in keep]
        labels = labels[keep]
        source += f" [subsampled to {len(labels)}]"

    print(f"Source          : {source}")
    print(f"Total postings  : {len(labels)}")
    print(f"Fraudulent      : {int(labels.sum())} ({labels.mean():.2%})")
    print(f"Genuine         : {int((labels == 0).sum())}")

    # ---------------------------------------------------------------- Stage 2
    _print_header("STAGE 2/5  Train/test split (stratified)")
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    tr_idx, te_idx = train_test_split(
        idx, test_size=args.test_size, stratify=labels, random_state=42
    )
    train_p = [postings[i] for i in tr_idx]
    test_p = [postings[i] for i in te_idx]
    y_train, y_test = labels[tr_idx], labels[te_idx]
    print(f"Train: {len(train_p)}  (fraud {int(y_train.sum())})")
    print(f"Test : {len(test_p)}  (fraud {int(y_test.sum())})")

    train_corpus = [p.combined_text() for p in train_p]
    test_corpus = [p.combined_text() for p in test_p]

    # ---------------------------------------------------------------- Stage 3
    _print_header("STAGE 3/5  Train & compare models")
    results = []

    print("[1/4] Baseline: TF-IDF + Logistic Regression ...")
    base_prob = _baseline_tfidf_lr(train_corpus, y_train, test_corpus)
    results.append(evaluate("Baseline TF-IDF+LR", y_test, base_prob))

    # Select text branch (BERT optional).
    text_branch = None
    default_splits = 5
    if args.use_bert:
        from jobguard.bert_branch import BertTextBranch, bert_available

        if bert_available(verbose=True):
            print(f"      DistilBERT available -> fine-tuning (epochs={args.bert_epochs}, "
                  f"max_len={args.bert_max_len}). This is slow on CPU.")
            text_branch = BertTextBranch(epochs=args.bert_epochs, max_len=args.bert_max_len)
            default_splits = 3  # fewer folds: each fold is a full fine-tune
        else:
            print("      torch/transformers not installed -> falling back to TF-IDF branch.")

    n_splits = args.n_splits or default_splits
    print(f"[2/4] + [3/4] + [4/4] Training stacked ensemble (tabular + text), "
          f"{n_splits}-fold OOF ...")
    model = EnsembleModel(text_branch=text_branch, use_smote=args.smote, n_splits=n_splits)
    model.fit(train_p, y_train)

    # Per-branch test probabilities (reuse fitted base models).
    X_tab_test = EnsembleModel._tabular_matrix(test_p)
    p_tab = model.xgb.predict_proba(X_tab_test)[:, 1]
    p_text = model.text_branch.predict_proba(test_corpus)
    p_ens = model.predict_fraud_proba(test_p)

    results.append(evaluate("XGBoost (tabular)", y_test, p_tab))
    results.append(evaluate(f"Text branch ({model.metadata['text_branch']})", y_test, p_text))
    results.append(evaluate("Ensemble (stacked)", y_test, p_ens))

    # ---------------------------------------------------------------- Stage 4
    _print_header("STAGE 4/5  Evaluation report (positive class = FAKE)")
    print(f"{'MODEL':<28} {'metrics @ threshold 0.5'}")
    print("-" * 72)
    for r in results:
        print(r.as_row())

    ensemble_result = results[-1]
    print("\nConfusion matrix — Ensemble (rows=true [REAL,FAKE], cols=pred):")
    for row in ensemble_result.confusion:
        print("   ", row)

    print("\nClassification report — Ensemble @ 0.5:")
    print(classification_report_text(y_test, p_ens))

    # Recall-optimized operating point for the FAKE class.
    fraud_thr = best_threshold_for_recall(y_test, p_ens, min_precision=0.5)
    print(f"Recall-optimized FAKE threshold (precision>=0.50): {fraud_thr:.3f}")
    print("Classification report — Ensemble @ recall-optimized threshold:")
    print(classification_report_text(y_test, p_ens, threshold=fraud_thr))

    # ---------------------------------------------------------------- Stage 5
    _print_header("STAGE 5/5  Persist artifacts")
    out_path = Path(args.out) if args.out else settings.model_bundle_path
    model.metadata["operating_threshold_fraud"] = float(fraud_thr)
    model.metadata["trained_at"] = datetime.now(timezone.utc).isoformat()
    model.metadata["data_source"] = source
    model.save(out_path)

    # Model registry entry (append-only JSONL).
    registry = settings.models_dir / "registry.jsonl"
    entry = {
        "timestamp": model.metadata["trained_at"],
        "bundle": str(out_path.name),
        "data_source": source,
        "n_samples": int(len(labels)),
        "fraud_rate": float(labels.mean()),
        "ensemble_pr_auc": ensemble_result.pr_auc,
        "ensemble_recall": ensemble_result.recall,
        "ensemble_precision": ensemble_result.precision,
        "operating_threshold_fraud": float(fraud_thr),
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    print(f"Saved ensemble bundle -> {out_path}")
    print(f"Appended registry entry -> {registry}")
    print(f"\nDone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
