"""Unit tests for text preprocessing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobguard.preprocessing import clean_text, raw_tokens, strip_html


def test_strip_html_removes_tags_and_unescapes():
    assert "hello world" in strip_html("<p>hello&nbsp;world</p>").lower()
    assert "<" not in strip_html("<div><b>x</b></div>")


def test_clean_text_lowercases_and_drops_punctuation():
    out = clean_text("Hello, WORLD!!! Visit https://x.com or a@b.com now")
    assert out == out.lower()
    # URL and email should be stripped out entirely.
    assert "http" not in out
    assert "@" not in out


def test_clean_text_removes_stopwords():
    # 'the', 'and', 'is' are stopwords and should not survive.
    out = clean_text("the salary is good and the role is remote")
    tokens = out.split()
    assert "the" not in tokens
    assert "and" not in tokens


def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""  # type: ignore[arg-type]


def test_raw_tokens_preserves_words_before_stopword_removal():
    toks = raw_tokens("Send BANK details Now")
    assert "bank" in toks
    assert "now" in toks  # stopword, but raw_tokens keeps it
