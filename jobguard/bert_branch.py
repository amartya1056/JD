"""Optional DistilBERT text branch (drop-in upgrade for TfidfTextBranch).

This module is imported lazily and ONLY when `torch` + `transformers` are
installed and BERT is requested. It fine-tunes DistilBERT for binary
fraud classification and exposes the same interface as `TfidfTextBranch`, so the
ensemble code does not change.

Trade-off note: DistilBERT typically lifts recall on subtle scams at the cost of
~1-2 GB of dependencies and much slower training/inference. The project ships
with the TF-IDF branch as default and treats this as a v2 upgrade, exactly as
recommended for a solo build.
"""
from __future__ import annotations

import os
from pathlib import Path

# torch, xgboost and scikit-learn each ship their own OpenMP runtime. On Windows,
# loading torch AFTER the others can abort with "WinError 1114 / c10.dll
# initialization routine failed" because the duplicate-runtime guard trips. This
# flag lets the runtimes coexist. Must be set BEFORE torch is imported; we do it
# at module import, which happens before any lazy `import torch` inside the
# methods below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


def bert_available(verbose: bool = False) -> bool:
    """True if the heavy stack is importable. Cheap, no model download."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        if verbose:
            import sys
            print(f"[bert_available] import failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        return False


class BertTextBranch:
    """Fine-tuned DistilBERT sequence classifier.

    Serialization: the HF model/tokenizer are saved to a sibling directory next
    to the ensemble bundle (torch weights don't belong inside a joblib pickle).
    `__getstate__`/`__setstate__` keep the branch joblib-compatible by storing
    only a directory reference.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", max_len: int = 192,
                 epochs: int = 2, batch_size: int = 16, save_dir: str | None = None):
        self.model_name = model_name
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_dir = save_dir  # populated on save()
        self._model = None
        self._tokenizer = None

    def fresh(self) -> "BertTextBranch":
        """New, unfitted branch carrying the SAME hyperparameters (used per OOF
        fold). Crucially preserves epochs/max_len so folds match the final fit."""
        return BertTextBranch(
            model_name=self.model_name, max_len=self.max_len,
            epochs=self.epochs, batch_size=self.batch_size,
        )

    # --- lazy model loaders ---
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        src = self.save_dir or self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(src)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            src, num_labels=2
        )
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def fit(self, corpus: list[str], labels: list[int]) -> "BertTextBranch":
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, get_linear_schedule_with_warmup)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        class _DS(Dataset):
            def __init__(self, texts, ys, tok, max_len):
                self.enc = tok(list(texts), truncation=True, padding="max_length",
                               max_length=max_len, return_tensors="pt")
                self.y = torch.tensor(ys, dtype=torch.long)

            def __len__(self):
                return len(self.y)

            def __getitem__(self, i):
                return ({k: v[i] for k, v in self.enc.items()}, self.y[i])

        ds = _DS(corpus, labels, tokenizer, self.max_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Class weights to counter imbalance (mirrors the tabular branch).
        n_pos = max(1, sum(labels))
        n_neg = max(1, len(labels) - n_pos)
        weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
        total_steps = len(loader) * self.epochs
        sched = get_linear_schedule_with_warmup(optim, 0, total_steps)

        model.train()
        for _ in range(self.epochs):
            for batch, y in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                y = y.to(device)
                optim.zero_grad()
                out = model(**batch)
                loss = loss_fn(out.logits, y)
                loss.backward()
                optim.step()
                sched.step()

        model.eval()
        self._model, self._tokenizer, self._device = model, tokenizer, device
        return self

    def predict_proba(self, corpus: list[str]) -> np.ndarray:
        import torch

        self._ensure_loaded()
        probs = []
        for i in range(0, len(corpus), self.batch_size):
            chunk = corpus[i : i + self.batch_size]
            enc = self._tokenizer(chunk, truncation=True, padding=True,
                                  max_length=self.max_len, return_tensors="pt").to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
        return np.concatenate(probs) if probs else np.zeros(0)

    def top_phrases(self, text: str, k: int = 8) -> list[tuple[str, float]]:
        """Attention-free saliency: mask each token and measure the drop in the
        fraud probability. Model-faithful and requires no extra libraries."""
        import torch

        self._ensure_loaded()
        words = text.split()
        if not words:
            return []
        base = float(self.predict_proba([text])[0])
        scores: list[tuple[str, float]] = []
        for idx, w in enumerate(words[:120]):  # cap for latency
            masked = words.copy()
            masked[idx] = self._tokenizer.mask_token or "[MASK]"
            drop = base - float(self.predict_proba([" ".join(masked)])[0])
            if drop > 0:
                scores.append((w, drop))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # --- joblib serialization: keep only a directory reference ---
    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._ensure_loaded()
        self._model.save_pretrained(directory)
        self._tokenizer.save_pretrained(directory)
        self.save_dir = str(directory)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_tokenizer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
