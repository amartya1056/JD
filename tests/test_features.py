"""Unit tests for engineered feature extraction and parsing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobguard.features import (FEATURE_NAMES, JobPosting, extract_features,
                               features_to_vector)
from jobguard.keywords import find_suspicious_phrases
from jobguard.parser import parse_raw_text


def test_feature_vector_matches_feature_names_length():
    p = JobPosting(title="Engineer", description="Build things.")
    feats = extract_features(p)
    vec = features_to_vector(feats)
    assert len(vec) == len(FEATURE_NAMES)
    # Every declared feature must be produced.
    for name in FEATURE_NAMES:
        assert name in feats


def test_suspicious_phrase_and_salary_signals():
    p = JobPosting(
        title="Easy money",
        description="No experience needed. Send bank account details and pay a "
                    "registration fee. Wire transfer via Western Union.",
        salary_range="$300,000 - $600,000",
        company="",
    )
    feats = extract_features(p)
    assert feats["suspicious_phrase_count"] >= 3
    assert feats["salary_too_high_flag"] == 1.0
    assert feats["has_salary_range"] == 1.0


def test_salary_parsing_variants():
    for salary in ["$50,000 - $70,000", "50000-70000", "40k-60k", "90000 to 110000"]:
        feats = extract_features(JobPosting(description="x", salary_range=salary))
        assert feats["has_salary_range"] == 1.0, salary


def test_email_domain_mismatch_and_free_email():
    p = JobPosting(
        company="Northwind Traders",
        description="Apply at recruiter@gmail.com today.",
    )
    feats = extract_features(p)
    assert feats["email_present"] == 1.0
    assert feats["free_email_flag"] == 1.0
    assert feats["email_domain_mismatch"] == 1.0


def test_find_suspicious_phrases_deduplicates_by_reason():
    hits = find_suspicious_phrases("bank details and bank account details here")
    reasons = [r for _p, r in hits]
    assert len(reasons) == len(set(reasons))


def test_parser_extracts_sections():
    text = (
        "Senior Analyst\n"
        "Company: Acme\n"
        "Salary: $90,000 - $110,000\n"
        "About us:\nAcme is a great firm.\n"
        "Requirements:\nBachelor degree.\n"
        "Benefits:\nHealth insurance.\n"
    )
    p = parse_raw_text(text)
    assert p.company == "Acme"
    assert "90,000" in p.salary_range
    assert p.company_profile != ""
    assert p.requirements != ""
    assert p.benefits != ""
