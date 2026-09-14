"""Text cleaning and normalization.

Design goals:
- Deterministic and dependency-light by default (pure stdlib + regex), so the
  training pipeline runs on a bare install.
- Optionally upgrades to NLTK lemmatization + stopword removal when that data is
  available (see `ensure_nltk()`), giving better TF-IDF features in production.

Both the training pipeline and the live API call `clean_text()`, guaranteeing
identical normalization at fit and inference time.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache

# --- Static regexes (compiled once) ---
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z][a-z]+")

# Minimal built-in stopword list (fallback when NLTK is unavailable).
_BASIC_STOPWORDS = frozenset(
    "a an the and or but if then else for to of in on at by with from as is are "
    "was were be been being this that these those it its we you they he she i our "
    "your their his her not no do does did have has had will would can could".split()
)


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities (job boards embed lots of markup).

    Non-breaking spaces (&nbsp; -> \\xa0) are normalized to regular spaces, since
    job boards litter salary/location lines with them.
    """
    text = html.unescape(text or "").replace("\xa0", " ")
    return _HTML_TAG_RE.sub(" ", text)


@lru_cache(maxsize=1)
def _nltk_tools():
    """Lazily load NLTK lemmatizer + stopwords. Returns None if unavailable.

    Cached so we pay the import/load cost at most once per process.
    """
    try:
        from nltk.corpus import stopwords  # type: ignore
        from nltk.stem import WordNetLemmatizer  # type: ignore

        # Touch the corpora to confirm the data is actually downloaded.
        sw = set(stopwords.words("english"))
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")  # forces wordnet load; raises if missing
        return lem, sw
    except Exception:
        return None


def clean_text(text: str, *, lemmatize: bool = True) -> str:
    """Full normalization used for TF-IDF / embedding inputs.

    Steps: strip HTML -> unescape -> drop URLs/emails -> lowercase ->
    remove punctuation -> tokenize -> stopword removal -> optional lemmatize.
    """
    if not text:
        return ""
    text = strip_html(text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()

    tokens = text.split()
    tools = _nltk_tools() if lemmatize else None
    if tools is not None:
        lemmatizer, stops = tools
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stops and len(t) > 1]
    else:
        tokens = [t for t in tokens if t not in _BASIC_STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def raw_tokens(text: str) -> list[str]:
    """Lowercased alphabetic tokens *before* stopword removal.

    Used by phrase-highlighting so we can map suspicious spans back onto the
    original wording the user pasted.
    """
    return _TOKEN_RE.findall((text or "").lower())


def ensure_nltk() -> bool:
    """Download the NLTK corpora needed for lemmatization. Returns success.

    Call this once from a setup script (not on the hot path). Safe to re-run.
    """
    try:
        import nltk  # type: ignore

        for pkg in ("wordnet", "omw-1.4", "stopwords"):
            nltk.download(pkg, quiet=True)
        _nltk_tools.cache_clear()
        return _nltk_tools() is not None
    except Exception:
        return False
