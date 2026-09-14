"""Text-classification branches (pluggable).

Both branches implement the same tiny interface so the ensemble is agnostic to
which one is used:

    fit(corpus: list[str], labels: list[int]) -> self
    predict_proba(corpus: list[str]) -> np.ndarray   # P(fraud) per row
    top_phrases(text: str, k) -> list[(phrase, weight)]  # for explanations

`TfidfTextBranch` is pure scikit-learn — always available, fast, joblib-safe,
and the project's default. `BertTextBranch` (in `bert_branch.py`) is a drop-in
upgrade activated when torch + transformers are installed.
"""
from __future__ import annotations

import numpy as np

from .preprocessing import clean_text


class TfidfTextBranch:
    """TF-IDF (word + char n-grams) followed by class-weighted Logistic Regression.

    Char n-grams catch obfuscated scam tokens (e.g. "w1re transfer") that word
    features miss. Class weights handle the ~4% fraud imbalance without SMOTE on
    the (large, sparse) text matrix.
    """

    def fresh(self) -> "TfidfTextBranch":
        """Return a new, unfitted branch with the same configuration."""
        return TfidfTextBranch(max_features=self.max_features)

    def __init__(self, max_features: int = 30000):
        self.max_features = max_features
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.word_vec = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=2,
            max_features=max_features, dtype=np.float32,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3,
            max_features=max_features, dtype=np.float32,
        )
        self.clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", C=4.0, solver="liblinear",
        )
        self._fitted = False

    # --- helpers ---
    @staticmethod
    def _clean(corpus: list[str]) -> list[str]:
        return [clean_text(t) for t in corpus]

    def _transform(self, cleaned: list[str]):
        from scipy.sparse import hstack

        return hstack([self.word_vec.transform(cleaned), self.char_vec.transform(cleaned)]).tocsr()

    # --- interface ---
    def fit(self, corpus: list[str], labels: list[int]) -> "TfidfTextBranch":
        cleaned = self._clean(corpus)
        Xw = self.word_vec.fit_transform(cleaned)
        Xc = self.char_vec.fit_transform(cleaned)
        from scipy.sparse import hstack

        X = hstack([Xw, Xc]).tocsr()
        self.clf.fit(X, labels)
        self._fitted = True
        return self

    def predict_proba(self, corpus: list[str]) -> np.ndarray:
        X = self._transform(self._clean(corpus))
        return self.clf.predict_proba(X)[:, 1]

    def top_phrases(self, text: str, k: int = 8) -> list[tuple[str, float]]:
        """Return the word n-grams that pushed this text toward FRAUD.

        Uses the linear model's coefficients weighted by the document's TF-IDF
        values — a fast, faithful local explanation for a linear text model
        (this is exactly what LIME approximates for linear classifiers).
        """
        if not self._fitted:
            return []
        cleaned = clean_text(text)
        row = self.word_vec.transform([cleaned])  # 1 x V
        # Coefficients for the word block are the first V entries of clf.coef_.
        v = len(self.word_vec.get_feature_names_out())
        coefs = self.clf.coef_[0][:v]
        contributions = row.multiply(coefs).tocoo()
        names = self.word_vec.get_feature_names_out()
        scored = [(names[j], float(val)) for j, val in zip(contributions.col, contributions.data)]
        # Most positive contributions = most fraud-indicative.
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(p, w) for p, w in scored[:k] if w > 0]
