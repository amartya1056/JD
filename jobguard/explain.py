"""Explainability: turn model internals into plain-English evidence.

Three explanation sources feed the UI:
  1. Rule-based flags      — deterministic reasons from keywords/metadata/salary
  2. SHAP on the XGBoost   — which tabular features moved the score, in words
  3. Text-branch saliency  — the phrases in the description that look fraudulent

Everything here maps raw numbers onto human sentences; the API never surfaces
feature indices or coefficients to the end user.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import (FEATURE_LABELS, FEATURE_NAMES, JobPosting,
                       extract_features)
from .keywords import find_suspicious_phrases

# SHAP explainer is cached per-model so we build it once.
_SHAP_CACHE: dict[int, object] = {}


@dataclass
class Explanation:
    flagged_reasons: list[str] = field(default_factory=list)
    top_suspicious_phrases: list[str] = field(default_factory=list)
    shap_top_features: list[dict] = field(default_factory=list)


def _shap_top_features(model, tabular_vector: list[float], k: int = 6) -> list[dict]:
    """Return the top tabular features pushing THIS posting toward fraud.

    Uses SHAP's TreeExplainer when available (exact for tree models). Falls back
    to a global-importance x local-value heuristic if `shap` is not installed, so
    the API never crashes for lack of an optional dependency.
    """
    x = np.asarray(tabular_vector, dtype=np.float64).reshape(1, -1)
    try:
        import shap  # type: ignore

        key = id(model)
        explainer = _SHAP_CACHE.get(key)
        if explainer is None:
            explainer = shap.TreeExplainer(model)
            _SHAP_CACHE[key] = explainer
        sv = explainer.shap_values(x)
        # Binary XGBoost -> a single array of shape (1, n_features).
        vals = np.asarray(sv)[0] if not isinstance(sv, list) else np.asarray(sv[-1])[0]
        contributions = list(zip(FEATURE_NAMES, vals))
    except Exception:
        # Fallback: global importance * (deviation of this value from 0).
        try:
            importances = model.feature_importances_
        except Exception:
            importances = np.ones(len(FEATURE_NAMES))
        contributions = [
            (name, float(importances[i]) * float(tabular_vector[i]))
            for i, name in enumerate(FEATURE_NAMES)
        ]

    # Rank by absolute impact; report direction in words.
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)
    out: list[dict] = []
    for name, val in contributions[:k]:
        if abs(val) < 1e-9:
            continue
        out.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "impact": round(float(val), 4),
            "direction": "raises fraud risk" if val > 0 else "lowers fraud risk",
        })
    return out


def build_flagged_reasons(posting: JobPosting, feats: dict[str, float]) -> list[str]:
    """Deterministic, high-precision reasons — independent of the ML model.

    These always fire on known scam patterns even if the model is uncertain,
    giving the user concrete, checkable evidence.
    """
    reasons: list[str] = []
    text = posting.combined_text()

    # 1. Known scam phrases (each already carries a human reason).
    for _phrase, reason in find_suspicious_phrases(text):
        reasons.append(reason)

    # 2. Metadata gaps. Logo/questions are only observable from a scraped listing
    # (structured source), never from raw pasted text — so we flag them only when
    # the field was actually observed as 0, not when it is unknown (None).
    if posting.has_company_logo == 0:
        reasons.append("No company logo — common in fraudulent listings")
    if not feats.get("has_company_profile"):
        reasons.append("No company profile or 'about us' information provided")
    if feats.get("free_email_flag"):
        reasons.append("Contact uses a free email provider instead of a company domain")
    if feats.get("email_domain_mismatch"):
        reasons.append("Contact email domain doesn't match the company name")

    # 3. Salary anomalies.
    if feats.get("salary_too_high_flag"):
        reasons.append("Advertised salary is abnormally high for the role")

    # 4. Style signals.
    if feats.get("all_caps_ratio", 0) > 0.30:
        reasons.append("Excessive use of ALL-CAPS text")
    if feats.get("exclamation_count", 0) >= 5:
        reasons.append("Unusually high number of exclamation marks")
    if feats.get("urgency_word_ratio", 0) > 0.05:
        reasons.append("Heavy use of urgency/hype language")
    if feats.get("description_word_count", 0) < 20:
        reasons.append("Job description is suspiciously short on detail")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique = []
    for r in reasons:
        if r not in seen:
            unique.append(r)
            seen.add(r)
    return unique


def build_explanation(model, posting: JobPosting, tabular_vector: list[float]) -> Explanation:
    """Assemble the full explanation payload for one posting."""
    feats = extract_features(posting)
    reasons = build_flagged_reasons(posting, feats)

    # Suspicious phrases: union of model-driven salient words and known scam
    # phrases actually present in the text (the latter are higher-precision).
    phrases: list[str] = []
    try:
        for phrase, _w in model.text_branch.top_phrases(posting.combined_text(), k=8):
            phrases.append(phrase)
    except Exception:
        pass
    for phrase, _reason in find_suspicious_phrases(posting.combined_text()):
        if phrase not in phrases:
            phrases.append(phrase)

    shap_feats = _shap_top_features(model.xgb, tabular_vector) if model.xgb is not None else []

    return Explanation(
        flagged_reasons=reasons,
        top_suspicious_phrases=phrases[:10],
        shap_top_features=shap_feats,
    )


def safety_advice(verdict: str) -> str:
    """Verdict-specific guidance. Always non-committal about certainty."""
    common = (
        "This is an automated, probabilistic assessment — not a definitive "
        "judgment. Always verify independently."
    )
    if verdict == "FAKE":
        return (
            "Treat this posting with strong caution. Never send money, pay any "
            "fee, or share bank details, your SSN, or ID documents to apply. "
            "Verify the company through its official website and independent "
            "sources before responding. " + common
        )
    if verdict == "SUSPICIOUS":
        return (
            "Some signals are concerning. Do not share financial or personal "
            "identification information until you have confirmed the employer is "
            "legitimate. Contact the company through official channels. " + common
        )
    return (
        "No strong fraud signals were detected, but stay alert: legitimate "
        "employers never ask for payment or bank details to apply. " + common
    )
