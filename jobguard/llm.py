"""Groq-powered natural-language explanations.

The ML ensemble makes the decision; the LLM only *narrates* it. We pass the
model's structured outputs (verdict, probabilities, flagged reasons, top
features, suspicious phrases) and ask for a concise, grounded explanation plus
practical next steps. The prompt forbids the model from overriding the verdict or
inventing facts not present in the signals — this keeps the LLM an explainer, not
a second (unvalidated) classifier.

Design choices:
- Uses only the stdlib `urllib` (no extra dependency) to call Groq's
  OpenAI-compatible endpoint.
- Fails OPEN: any error (no key, timeout, bad response) returns None and the API
  simply omits the AI narrative — the deterministic explanation always remains.
- `reasoning_effort=low` because gpt-oss models emit separate reasoning tokens;
  we only want the final prose.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .config import settings

_SYSTEM_PROMPT = (
    "You are JobGuard's safety explainer. A machine-learning ensemble has ALREADY "
    "classified a job posting as REAL, FAKE, or SUSPICIOUS. Your job is to explain "
    "that verdict to a non-technical job seeker in clear, calm language and give "
    "practical guidance.\n\n"
    "STRICT RULES:\n"
    "1. NEVER contradict or change the provided verdict. Explain it as given.\n"
    "2. Only use the signals provided. Do NOT invent company names, facts, or "
    "details that are not in the input.\n"
    "3. Always frame the result as a probabilistic assessment, not a certainty.\n"
    "4. Never tell the user something is definitely safe; advise independent "
    "verification.\n"
    "5. Be concise: 2-4 short paragraphs, plain English, no markdown headers."
)


@dataclass
class LLMExplanation:
    summary: str
    model: str


def _build_user_prompt(payload: dict) -> str:
    reasons = payload.get("flagged_reasons") or []
    phrases = (payload.get("explanations") or {}).get("top_suspicious_phrases") or []
    shap = (payload.get("explanations") or {}).get("shap_top_features") or []
    shap_lines = [f"- {f.get('label')}: {f.get('direction')}" for f in shap[:6]]
    parsed = payload.get("parsed") or {}

    return (
        f"VERDICT: {payload.get('verdict')}\n"
        f"Risk score: {payload.get('risk_score')}/100 "
        f"(probability genuine {payload.get('probability_real')}, "
        f"probability fake {payload.get('probability_fake')}).\n"
        f"Model confidence in this verdict: {payload.get('confidence')}.\n\n"
        f"Parsed posting — title: {parsed.get('title') or 'n/a'}; "
        f"company: {parsed.get('company') or 'n/a'}; "
        f"salary: {parsed.get('salary_range') or 'n/a'}.\n\n"
        f"Flagged reasons (deterministic signals):\n"
        + ("\n".join(f"- {r}" for r in reasons) if reasons else "- (none)")
        + "\n\nTop tabular features that moved the score:\n"
        + ("\n".join(shap_lines) if shap_lines else "- (none)")
        + "\n\nSuspicious phrases detected in the text: "
        + (", ".join(phrases) if phrases else "(none)")
        + "\n\nWrite the explanation now: first explain WHY the posting received "
        "this verdict based on the signals above, then give the job seeker 2-3 "
        "concrete next steps. Remember it is a probabilistic estimate."
    )


def generate_explanation(payload: dict) -> Optional[LLMExplanation]:
    """Return a grounded natural-language explanation, or None on any failure."""
    if not settings.use_llm_explanation or not settings.groq_api_key:
        return None

    body = json.dumps({
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(payload)},
        ],
        "temperature": 0.3,
        "max_tokens": 700,
        "reasoning_effort": "low",  # gpt-oss: keep only final prose
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{settings.groq_base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
            # Groq sits behind Cloudflare, which blocks the default python-urllib
            # User-Agent (HTTP 403, error 1010). A normal UA passes.
            "User-Agent": "JobGuard/1.0 (+https://example.com)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=settings.groq_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (data["choices"][0]["message"].get("content") or "").strip()
        if not content:
            return None
        return LLMExplanation(summary=content, model=settings.groq_model)
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError):
        # Fail open — the deterministic explanation is always still returned.
        return None
