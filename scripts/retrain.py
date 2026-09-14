"""Retraining with the feedback loop + guarded promotion.

Flow:
  1. Load the base dataset (EMSCAD CSV or synthetic fallback).
  2. Pull accumulated user feedback from the database and convert each correction
     into a labeled training row.
  3. Train a fresh ensemble on the combined corpus.
  4. Evaluate PR-AUC on the FAKE class (held-out split).
  5. Promote the new model ONLY if its PR-AUC beats the current production model's
     recorded PR-AUC (a guardrail against regressions). Otherwise keep it as a
     candidate. Either way, append a model-registry entry with metrics + timestamp.

Run:
  python scripts/retrain.py
  python scripts/retrain.py --data data/fake_job_postings.csv
  python scripts/retrain.py --force     # promote regardless of comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobguard.config import settings  # noqa: E402
from jobguard.dataset import load_dataset  # noqa: E402
from jobguard.ensemble import EnsembleModel  # noqa: E402
from jobguard.evaluation import evaluate  # noqa: E402
from jobguard.parser import parse_raw_text  # noqa: E402


def _load_feedback_rows():
    """Pull (JobPosting, label) pairs from the feedback table. Empty if no DB."""
    try:
        from sqlalchemy import select

        # Import backend DB models lazily so this script has no hard web dep.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
        from app.database import Feedback, SessionLocal  # type: ignore

        postings, labels = [], []
        with SessionLocal() as db:
            for fb in db.execute(select(Feedback)).scalars():
                if not fb.input_text:
                    continue
                postings.append(parse_raw_text(fb.input_text))
                labels.append(int(fb.user_label))
        return postings, labels
    except Exception as exc:
        print(f"[feedback] none loaded ({type(exc).__name__}: {exc})")
        return [], []


def _current_best_pr_auc() -> float:
    """Highest PR-AUC among previously PROMOTED registry entries."""
    registry = settings.models_dir / "registry.jsonl"
    if not registry.exists():
        return 0.0
    best = 0.0
    for line in registry.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("promoted", True):  # train.py entries count as promoted
            best = max(best, float(entry.get("ensemble_pr_auc", 0.0)))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain with feedback and guard promotion.")
    ap.add_argument("--data", default=None)
    ap.add_argument("--synthetic-n", type=int, default=4000)
    ap.add_argument("--force", action="store_true", help="Promote regardless of PR-AUC.")
    ap.add_argument("--smote", action="store_true")
    args = ap.parse_args()

    print("Loading base dataset ...")
    postings, labels, source = load_dataset(args.data, synthetic_n=args.synthetic_n)

    fb_postings, fb_labels = _load_feedback_rows()
    print(f"Feedback rows incorporated: {len(fb_labels)}")
    postings = list(postings) + fb_postings
    labels = list(labels) + fb_labels
    labels = np.asarray(labels, dtype=int)

    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=0.2, stratify=labels, random_state=7)
    train_p = [postings[i] for i in tr]
    test_p = [postings[i] for i in te]
    y_test = labels[te]

    print("Training candidate ensemble ...")
    model = EnsembleModel(use_smote=args.smote)
    model.fit(train_p, labels[tr])

    p_ens = model.predict_fraud_proba(test_p)
    result = evaluate("candidate", y_test, p_ens)
    print(f"Candidate: {result.as_row()}")

    current_best = _current_best_pr_auc()
    improved = result.pr_auc >= current_best
    promote = improved or args.force
    print(f"Current production PR-AUC: {current_best:.4f} | candidate: {result.pr_auc:.4f} "
          f"| {'PROMOTE' if promote else 'KEEP CANDIDATE'}")

    ts = datetime.now(timezone.utc).isoformat()
    model.metadata.update({
        "trained_at": ts, "data_source": source,
        "feedback_rows": len(fb_labels), "retrained": True,
    })

    if promote:
        out_path = settings.model_bundle_path
    else:
        out_path = settings.models_dir / f"candidate_{ts.replace(':', '').replace('-', '')[:15]}.joblib"
    model.save(out_path)

    registry = settings.models_dir / "registry.jsonl"
    with registry.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "timestamp": ts,
            "bundle": out_path.name,
            "data_source": source,
            "feedback_rows": len(fb_labels),
            "ensemble_pr_auc": result.pr_auc,
            "ensemble_recall": result.recall,
            "ensemble_precision": result.precision,
            "promoted": bool(promote),
        }) + "\n")

    print(f"Saved -> {out_path}")
    print(f"Registry updated -> {registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
