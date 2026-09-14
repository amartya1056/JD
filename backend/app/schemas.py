"""Pydantic request/response models — the API's typed contract.

Validation here is the first line of defense: it rejects malformed input before
it reaches the model, and documents the response shape in the OpenAPI schema
(auto-served at /docs).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalyzeRequest(BaseModel):
    url: Optional[str] = Field(None, description="URL of the job posting to scrape.")
    text: Optional[str] = Field(None, description="Raw pasted job posting text.")

    @model_validator(mode="after")
    def _one_of(self) -> "AnalyzeRequest":
        has_url = bool(self.url and self.url.strip())
        has_text = bool(self.text and len(self.text.strip()) >= 20)
        if not has_url and not has_text:
            raise ValueError("Provide either a valid 'url' or at least 20 characters of 'text'.")
        return self


class ShapFeature(BaseModel):
    feature: str
    label: str
    impact: float
    direction: str


class Explanations(BaseModel):
    top_suspicious_phrases: list[str] = []
    shap_top_features: list[ShapFeature] = []


class AnalyzeResponse(BaseModel):
    submission_id: int
    verdict: Literal["REAL", "FAKE", "SUSPICIOUS"]
    confidence: float
    risk_score: int
    probability_real: float
    probability_fake: float
    flagged_reasons: list[str]
    explanations: Explanations
    advice: str
    branch_scores: dict[str, float]
    scrape_method: Optional[str] = None
    parsed: dict[str, Any] = {}
    # Optional Groq-generated narrative: {"summary": str, "model": str}.
    # Null when the LLM is disabled or unavailable (deterministic fields remain).
    ai_explanation: Optional[dict[str, str]] = None


class FeedbackRequest(BaseModel):
    submission_id: int
    # True label according to the user: "fake" or "real".
    correct_label: Literal["fake", "real"]


class FeedbackResponse(BaseModel):
    ok: bool
    message: str


class HealthResponse(BaseModel):
    # `model_` is a Pydantic-protected prefix; opt out so these field names work.
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_metadata: dict[str, Any] = {}
