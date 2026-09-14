"""Engineered tabular feature extraction.

This module turns a structured job posting into a fixed-length, *named* numeric
vector. Named features matter: SHAP explanations reference these names, and we
map each one to plain English for the UI.

A single function, `extract_features(posting)`, is the contract shared by
training and serving. `FEATURE_NAMES` defines the column order and must stay in
sync with the vector `extract_features` returns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .keywords import FREE_EMAIL_DOMAINS, URGENCY_WORDS, find_suspicious_phrases

# Deterministic feature order. Changing this requires retraining.
FEATURE_NAMES: list[str] = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "has_salary_range",
    "salary_span_ratio",
    "salary_too_high_flag",
    "has_company_profile",
    "has_requirements",
    "has_benefits",
    "description_word_count",
    "description_char_count",
    "title_word_count",
    "all_caps_ratio",
    "exclamation_count",
    "urgency_word_ratio",
    "suspicious_phrase_count",
    "external_link_count",
    "email_present",
    "free_email_flag",
    "email_domain_mismatch",
    "has_employment_type",
    "has_required_experience",
    "has_required_education",
    "digit_ratio",
    "uppercase_word_ratio",
]

# Human-readable labels for the UI's SHAP panel.
FEATURE_LABELS: dict[str, str] = {
    "telecommuting": "Remote/telecommuting flag",
    "has_company_logo": "Company logo present",
    "has_questions": "Application screening questions present",
    "has_salary_range": "Salary range provided",
    "salary_span_ratio": "Width of the salary range",
    "salary_too_high_flag": "Salary looks too good to be true",
    "has_company_profile": "Company profile provided",
    "has_requirements": "Requirements section present",
    "has_benefits": "Benefits section present",
    "description_word_count": "Description length (words)",
    "description_char_count": "Description length (characters)",
    "title_word_count": "Title length (words)",
    "all_caps_ratio": "Proportion of ALL-CAPS text",
    "exclamation_count": "Number of exclamation marks",
    "urgency_word_ratio": "Density of urgency/hype words",
    "suspicious_phrase_count": "Count of known scam phrases",
    "external_link_count": "Number of external links",
    "email_present": "Contact email present",
    "free_email_flag": "Uses a free email provider (gmail, etc.)",
    "email_domain_mismatch": "Email domain differs from company",
    "has_employment_type": "Employment type specified",
    "has_required_experience": "Required experience specified",
    "has_required_education": "Required education specified",
    "digit_ratio": "Proportion of digits in text",
    "uppercase_word_ratio": "Proportion of fully-uppercase words",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
# Salary like "$50,000 - $70,000" or "50000-70000" or "40k-60k".
# Matches "$50,000 - $70,000", "50000-70000", "40k-60k", "90000 to 110000".
_SALARY_RANGE_RE = re.compile(
    r"(\$?\s?\d{2,3}(?:,?\d{3})?(?:\.\d+)?\s?k?)\s*(?:-|–|to)\s*(\$?\s?\d{2,3}(?:,?\d{3})?(?:\.\d+)?\s?k?)",
    re.IGNORECASE,
)


@dataclass
class JobPosting:
    """Normalized job posting. Any missing field is None/empty, never crashes.

    This is the single structured type flowing through the whole system: the
    scraper produces it, training rows are converted to it, and the API accepts
    it. Optional fields mirror EMSCAD's columns.
    """

    title: str = ""
    company_profile: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""
    company: str = ""
    location: str = ""
    salary_range: str = ""
    employment_type: str = ""
    required_experience: str = ""
    required_education: str = ""
    telecommuting: Optional[int] = None
    has_company_logo: Optional[int] = None
    has_questions: Optional[int] = None
    source_url: str = ""

    def combined_text(self) -> str:
        """Concatenate the text columns the way EMSCAD analyses do."""
        parts = [self.title, self.company_profile, self.description, self.requirements, self.benefits]
        return "\n".join(p for p in parts if p)


def _parse_salary(salary: str) -> Optional[tuple[float, float]]:
    """Extract (low, high) salary numbers, handling 'k' shorthand. None if absent."""
    if not salary:
        return None
    m = _SALARY_RANGE_RE.search(salary)
    if not m:
        return None

    def to_num(s: str) -> float:
        s = s.strip().lower().replace("$", "").replace(",", "").replace(" ", "")
        mult = 1000.0 if s.endswith("k") else 1.0
        s = s.rstrip("k")
        try:
            return float(s) * mult
        except ValueError:
            return 0.0

    low, high = to_num(m.group(1)), to_num(m.group(2))
    if low <= 0 or high <= 0:
        return None
    return (min(low, high), max(low, high))


def _all_caps_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase. Scam posts SHOUT."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def _uppercase_word_ratio(text: str) -> float:
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if not words:
        return 0.0
    caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    return caps / len(words)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def _company_tokens(company: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", company.lower()) if len(t) > 2}


def extract_features(posting: JobPosting) -> dict[str, float]:
    """Compute the full engineered feature dict for one posting.

    Returns a mapping name->value in `FEATURE_NAMES` order-compatible form.
    Pure and deterministic; no network calls (domain-age is handled in the
    metadata module so training stays offline/fast).
    """
    text = posting.combined_text()
    words = text.split()
    n_words = max(len(words), 1)

    salary = _parse_salary(posting.salary_range)
    has_salary = salary is not None
    salary_span_ratio = 0.0
    salary_too_high = 0.0
    if salary:
        low, high = salary
        salary_span_ratio = (high - low) / high if high else 0.0
        # "Too good to be true": annualized pay above a generous ceiling, or a
        # range so wide it's meaningless. Tunable heuristic, not a hard rule.
        if high >= 300_000 or (high >= 150_000 and salary_span_ratio > 0.8):
            salary_too_high = 1.0

    # Urgency density = urgency tokens / total tokens.
    lower_words = [w.strip(".,!?").lower() for w in words]
    urgency_hits = sum(1 for w in lower_words if w in URGENCY_WORDS)
    urgency_ratio = urgency_hits / n_words

    # Email + domain-mismatch signal.
    email_match = _EMAIL_RE.search(text)
    email_present = 1.0 if email_match else 0.0
    free_email = 0.0
    domain_mismatch = 0.0
    if email_match:
        domain = email_match.group(2).lower()
        if domain in FREE_EMAIL_DOMAINS:
            free_email = 1.0
        comp_tokens = _company_tokens(posting.company)
        if comp_tokens:
            domain_root = domain.split(".")[0]
            # Mismatch if no company token appears in the email domain root.
            if not any(tok in domain_root or domain_root in tok for tok in comp_tokens):
                domain_mismatch = 1.0

    feats: dict[str, float] = {
        "telecommuting": float(posting.telecommuting or 0),
        "has_company_logo": float(posting.has_company_logo or 0),
        "has_questions": float(posting.has_questions or 0),
        "has_salary_range": float(has_salary),
        "salary_span_ratio": float(salary_span_ratio),
        "salary_too_high_flag": float(salary_too_high),
        "has_company_profile": float(bool(posting.company_profile.strip())),
        "has_requirements": float(bool(posting.requirements.strip())),
        "has_benefits": float(bool(posting.benefits.strip())),
        "description_word_count": float(len(posting.description.split())),
        "description_char_count": float(len(posting.description)),
        "title_word_count": float(len(posting.title.split())),
        "all_caps_ratio": _all_caps_ratio(text),
        "exclamation_count": float(text.count("!")),
        "urgency_word_ratio": float(urgency_ratio),
        "suspicious_phrase_count": float(len(find_suspicious_phrases(text))),
        "external_link_count": float(len(_URL_RE.findall(text))),
        "email_present": email_present,
        "free_email_flag": free_email,
        "email_domain_mismatch": domain_mismatch,
        "has_employment_type": float(bool(posting.employment_type.strip())),
        "has_required_experience": float(bool(posting.required_experience.strip())),
        "has_required_education": float(bool(posting.required_education.strip())),
        "digit_ratio": _digit_ratio(text),
        "uppercase_word_ratio": _uppercase_word_ratio(text),
    }
    return feats


def features_to_vector(feats: dict[str, float]) -> list[float]:
    """Project a feature dict onto the canonical `FEATURE_NAMES` order."""
    return [float(feats.get(name, 0.0)) for name in FEATURE_NAMES]
