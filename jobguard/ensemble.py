"""The stacked ensemble: XGBoost (tabular) + text branch, combined by a meta-LR.

Pipeline at fit time:
  1. Engineer 25 named tabular features per posting  -> X_tab
  2. Build combined-text corpus                       -> corpus
  3. Get *out-of-fold* fraud probabilities from each base model (no leakage)
  4. Train a logistic-regression meta-learner on [p_tab, p_text]
  5. Refit both base models on the full data for deployment

At predict time we return not just the fused probability but the individual
branch probabilities and the tabular feature vector, so the explainability layer
can attribute the decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .features import (FEATURE_NAMES, JobPosting, extract_features,
                       features_to_vector)
from .text_branch import TfidfTextBranch


@dataclass
class BranchScores:
    """Per-posting breakdown returned by the ensemble for explainability."""

    p_fraud: float
    p_tabular: float
    p_text: float
    tabular_vector: list[float] = field(default_factory=list)


class EnsembleModel:
    """Trainable, serializable fraud classifier."""

    def __init__(self, text_branch=None, use_smote: bool = False, random_state: int = 42,
                 n_splits: int = 5):
        self.random_state = random_state
        self.use_smote = use_smote
        # Fewer folds when the text branch is an expensive fine-tune (DistilBERT):
        # each fold trains the branch once, so 5 folds = 6 fine-tunes on CPU.
        self.n_splits = n_splits
        self.text_branch = text_branch if text_branch is not None else TfidfTextBranch()
        self.xgb = None
        self.meta = None
        self.feature_names = list(FEATURE_NAMES)
        self.metadata: dict = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _tabular_matrix(postings: list[JobPosting]) -> np.ndarray:
        return np.asarray(
            [features_to_vector(extract_features(p)) for p in postings], dtype=np.float64
        )

    def _make_xgb(self, y: np.ndarray):
        from xgboost import XGBClassifier

        n_pos = max(1, int(y.sum()))
        n_neg = max(1, int(len(y) - n_pos))
        # scale_pos_weight handles imbalance analytically (neg/pos ratio).
        return XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="aucpr", scale_pos_weight=n_neg / n_pos,
            tree_method="hist", random_state=self.random_state, n_jobs=-1,
        )

    # -------------------------------------------------------------------- fit
    def fit(self, postings: list[JobPosting], labels: list[int]) -> "EnsembleModel":
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        y = np.asarray(labels, dtype=int)
        X_tab = self._tabular_matrix(postings)
        corpus = [p.combined_text() for p in postings]

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # --- Out-of-fold probabilities for the tabular branch ---
        oof_tab = self._oof_tabular(X_tab, y, skf)

        # --- Out-of-fold probabilities for the text branch ---
        oof_text = self._oof_text(corpus, y, skf)

        # --- Meta-learner on stacked OOF predictions ---
        meta_X = np.column_stack([oof_tab, oof_text])
        self.meta = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.meta.fit(meta_X, y)

        # --- Refit base models on ALL data for deployment ---
        self.xgb = self._fit_tabular_full(X_tab, y)
        self.text_branch.fit(corpus, y)

        self.metadata = {
            "n_samples": int(len(y)),
            "n_fraud": int(y.sum()),
            "fraud_rate": float(y.mean()),
            "text_branch": type(self.text_branch).__name__,
            "use_smote": self.use_smote,
            "meta_coef": self.meta.coef_.tolist(),
        }
        return self

    def _oof_tabular(self, X_tab, y, skf) -> np.ndarray:
        """OOF fraud probs from XGBoost, optionally SMOTE-balanced per fold."""
        oof = np.zeros(len(y))
        for tr, va in skf.split(X_tab, y):
            Xtr, ytr = X_tab[tr], y[tr]
            if self.use_smote:
                Xtr, ytr = self._smote(Xtr, ytr)
            clf = self._make_xgb(ytr)
            clf.fit(Xtr, ytr)
            oof[va] = clf.predict_proba(X_tab[va])[:, 1]
        return oof

    def _oof_text(self, corpus, y, skf) -> np.ndarray:
        oof = np.zeros(len(y))
        corpus = np.asarray(corpus, dtype=object)
        for tr, va in skf.split(corpus, y):
            branch = self._fresh_text_branch()  # new instance, SAME hyperparameters
            branch.fit(list(corpus[tr]), list(y[tr]))
            oof[va] = branch.predict_proba(list(corpus[va]))
        return oof

    def _fresh_text_branch(self):
        """A new, unfitted text branch preserving the configured hyperparameters.

        Prefers the branch's own `fresh()` factory (BERT needs its epochs/max_len
        carried over); falls back to a default-constructed instance.
        """
        if hasattr(self.text_branch, "fresh"):
            return self.text_branch.fresh()
        return type(self.text_branch)()

    def _fit_tabular_full(self, X_tab, y):
        Xtr, ytr = (self._smote(X_tab, y) if self.use_smote else (X_tab, y))
        clf = self._make_xgb(ytr)
        clf.fit(Xtr, ytr)
        return clf

    def _smote(self, X, y):
        """Balance the minority class. Falls back silently if imblearn absent."""
        try:
            from imblearn.over_sampling import SMOTE

            k = min(5, max(1, int(np.sum(y == 1)) - 1))
            return SMOTE(random_state=self.random_state, k_neighbors=k).fit_resample(X, y)
        except Exception:
            return X, y

    # ---------------------------------------------------------------- predict
    def score(self, posting: JobPosting) -> BranchScores:
        """Return fused + per-branch fraud probabilities for one posting."""
        vec = features_to_vector(extract_features(posting))
        X_tab = np.asarray([vec], dtype=np.float64)
        p_tab = float(self.xgb.predict_proba(X_tab)[:, 1][0])
        p_text = float(self.text_branch.predict_proba([posting.combined_text()])[0])
        p_fraud = float(self.meta.predict_proba([[p_tab, p_text]])[:, 1][0])
        return BranchScores(p_fraud=p_fraud, p_tabular=p_tab, p_text=p_text, tabular_vector=vec)

    def predict_fraud_proba(self, postings: list[JobPosting]) -> np.ndarray:
        """Batch fraud probability (used during evaluation)."""
        X_tab = self._tabular_matrix(postings)
        p_tab = self.xgb.predict_proba(X_tab)[:, 1]
        p_text = self.text_branch.predict_proba([p.combined_text() for p in postings])
        meta_X = np.column_stack([p_tab, p_text])
        return self.meta.predict_proba(meta_X)[:, 1]

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Heavy text branches (BERT) persist their own directory first.
        if hasattr(self.text_branch, "save"):
            self.text_branch.save(path.parent / (path.stem + "_textbranch"))
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "EnsembleModel":
        import joblib

        return joblib.load(path)
