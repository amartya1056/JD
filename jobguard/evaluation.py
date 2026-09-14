"""Evaluation helpers tuned for class imbalance.

On a 4%-fraud dataset, a model that predicts "REAL" for everything scores 96%
accuracy while catching zero scams. So we lead with PR-AUC and FAKE-class recall
(the cost of a missed scam >> the cost of a false alarm) and print a full
classification report + confusion matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalResult:
    name: str
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    confusion: list[list[int]]
    threshold: float

    def as_row(self) -> str:
        return (
            f"{self.name:<28} P={self.precision:.3f} R={self.recall:.3f} "
            f"F1={self.f1:.3f} PR-AUC={self.pr_auc:.3f} ROC-AUC={self.roc_auc:.3f}"
        )


def evaluate(name: str, y_true, y_prob, threshold: float = 0.5) -> EvalResult:
    """Compute metrics for the FRAUD (positive) class at a decision threshold."""
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    pr_auc = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    roc_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return EvalResult(name, float(p), float(r), float(f1), float(pr_auc),
                      float(roc_auc), cm, threshold)


def classification_report_text(y_true, y_prob, threshold: float = 0.5) -> str:
    from sklearn.metrics import classification_report

    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return classification_report(
        y_true, y_pred, target_names=["REAL", "FAKE"], digits=3, zero_division=0
    )


def best_threshold_for_recall(y_true, y_prob, min_precision: float = 0.5) -> float:
    """Pick the lowest threshold that still keeps precision >= min_precision.

    This operationalizes "optimize recall on FAKE while keeping precision
    reasonable": we accept more false alarms to miss fewer scams, but not so many
    that the tool cries wolf.
    """
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thr)+1 precision/recall points.
    best = 0.5
    best_recall = -1.0
    for i, t in enumerate(thr):
        if prec[i] >= min_precision and rec[i] > best_recall:
            best_recall = rec[i]
            best = float(t)
    return best
