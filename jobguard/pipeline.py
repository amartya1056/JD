"""End-to-end inference pipeline: JobPosting -> verdict payload.

Loaded once at API startup and cached in memory. `analyze()` is the single
callable the API, the demo script, and the tests all share.

Verdict policy (on p_real = probability the posting is genuine):
    p_real >= real_threshold (0.80)  -> REAL
    p_real <  fake_threshold (0.50)  -> FAKE
    otherwise                        -> SUSPICIOUS
This directly implements the confidence-thresholding rule from the spec.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import (VERDICT_FAKE, VERDICT_REAL, VERDICT_SUSPICIOUS, settings)
from .ensemble import EnsembleModel
from .explain import build_explanation, safety_advice
from .features import JobPosting


@dataclass
class AnalysisResult:
    verdict: str
    confidence: float          # confidence IN THE VERDICT (0-1)
    risk_score: int            # 0-100, higher = more likely fake
    probability_real: float
    probability_fake: float
    flagged_reasons: list[str] = field(default_factory=list)
    explanations: dict = field(default_factory=dict)
    advice: str = ""
    branch_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class InferencePipeline:
    """Wraps a trained EnsembleModel with verdict + explanation logic."""

    def __init__(self, model: EnsembleModel):
        self.model = model

    @classmethod
    def load(cls, bundle_path: str | Path | None = None) -> "InferencePipeline":
        path = Path(bundle_path) if bundle_path else settings.model_bundle_path
        if not path.exists():
            raise FileNotFoundError(
                f"No model bundle at {path}. Run `python scripts/train.py` first."
            )
        return cls(EnsembleModel.load(path))

    def _verdict(self, p_real: float) -> tuple[str, float]:
        """Map p_real to a label and a verdict-confidence."""
        if p_real >= settings.real_threshold:
            # Confidence scales how far above the REAL threshold we are.
            conf = 0.5 + 0.5 * (p_real - settings.real_threshold) / (1 - settings.real_threshold)
            return VERDICT_REAL, min(conf, 0.99)
        if p_real < settings.fake_threshold:
            conf = 0.5 + 0.5 * (settings.fake_threshold - p_real) / settings.fake_threshold
            return VERDICT_FAKE, min(conf, 0.99)
        # In the uncertain band: confidence is deliberately modest.
        span = settings.real_threshold - settings.fake_threshold
        mid = settings.fake_threshold + span / 2
        conf = 0.55 - 0.10 * abs(p_real - mid) / (span / 2)
        return VERDICT_SUSPICIOUS, max(0.40, conf)

    def analyze(self, posting: JobPosting) -> AnalysisResult:
        """Run the full pipeline on one structured posting."""
        scores = self.model.score(posting)
        p_fraud = scores.p_fraud
        p_real = 1.0 - p_fraud

        verdict, confidence = self._verdict(p_real)
        risk_score = int(round(p_fraud * 100))

        explanation = build_explanation(self.model, posting, scores.tabular_vector)

        return AnalysisResult(
            verdict=verdict,
            confidence=round(confidence, 4),
            risk_score=risk_score,
            probability_real=round(p_real, 4),
            probability_fake=round(p_fraud, 4),
            flagged_reasons=explanation.flagged_reasons,
            explanations={
                "top_suspicious_phrases": explanation.top_suspicious_phrases,
                "shap_top_features": explanation.shap_top_features,
            },
            advice=safety_advice(verdict),
            branch_scores={
                "tabular_model_fraud_prob": round(scores.p_tabular, 4),
                "text_model_fraud_prob": round(scores.p_text, 4),
            },
        )
