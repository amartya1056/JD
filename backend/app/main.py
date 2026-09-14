"""FastAPI application: /analyze, /feedback, /health.

The trained ensemble is loaded ONCE at startup and cached in memory (an
InferencePipeline held in app state), so per-request latency is just feature
extraction + a few model calls — no disk I/O.

Cross-cutting concerns wired here: CORS (locked to configured origins), rate
limiting (slowapi), request validation (pydantic), and graceful scraper
fallback.

NOTE: we deliberately do NOT `from __future__ import annotations` here — FastAPI
resolves route type hints at import time, and stringized annotations break its
dependency injection for the request/response models.
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Serving a DistilBERT ensemble requires torch, which on Windows must be imported
# BEFORE scikit-learn/xgboost (else c10.dll init fails — WinError 1114). If the
# configured bundle looks like a BERT model, or BERT serving is opted in, preload
# torch here before any jobguard import pulls in sklearn/xgboost. Cheap and safe;
# skipped when torch is absent or the default TF-IDF model is used.
if "bert" in os.getenv("JOBGUARD_MODEL_BUNDLE", "").lower() or os.getenv("JOBGUARD_SERVE_BERT", "").lower() in {"1", "true", "on"}:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401
    except Exception:
        pass

# Make the shared `jobguard` package importable when running from /backend.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from slowapi import Limiter  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from jobguard.config import settings  # noqa: E402
from jobguard.llm import generate_explanation  # noqa: E402
from jobguard.parser import parse_raw_text  # noqa: E402
from jobguard.pipeline import InferencePipeline  # noqa: E402
from jobguard.scraper import scrape  # noqa: E402

from .database import Feedback, Submission, get_session, init_db  # noqa: E402
from .schemas import (AnalyzeRequest, AnalyzeResponse, FeedbackRequest,  # noqa: E402
                      FeedbackResponse, HealthResponse)

limiter = Limiter(key_func=get_remote_address)

# Holds the singleton pipeline; populated on startup.
_state: dict[str, object] = {"pipeline": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown. Load DB schema and the model bundle once."""
    init_db()
    try:
        _state["pipeline"] = InferencePipeline.load()
        print(f"[startup] Model loaded from {settings.model_bundle_path}")
    except FileNotFoundError as exc:
        # Boot anyway so /health can report the problem; /analyze will 503.
        _state["pipeline"] = None
        print(f"[startup] WARNING: {exc}")
    yield
    _state.clear()


app = FastAPI(
    title="JobGuard API",
    version="1.0.0",
    description="Detect fraudulent job postings with an explainable ML ensemble.",
    lifespan=lifespan,
)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please slow down."})


def _pipeline() -> InferencePipeline:
    p = _state.get("pipeline")
    if p is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Model not loaded. Run scripts/train.py.")
    return p  # type: ignore[return-value]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    p = _state.get("pipeline")
    meta = getattr(p.model, "metadata", {}) if p else {}
    return HealthResponse(status="ok", model_loaded=p is not None, model_metadata=meta)


@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
def analyze(request: Request, body: AnalyzeRequest, db: Session = Depends(get_session)) -> AnalyzeResponse:
    """Scrape/parse -> run the ensemble -> persist -> return the verdict."""
    pipeline = _pipeline()
    scrape_method = None

    # Resolve input into a structured posting.
    if body.url and body.url.strip():
        result = scrape(body.url.strip())
        if not result.ok or result.posting is None:
            # Graceful fallback: tell the client to paste text.
            from fastapi import HTTPException

            raise HTTPException(status_code=422, detail=result.error)
        posting = result.posting
        scrape_method = result.method
        input_text = posting.combined_text()
    else:
        input_text = (body.text or "").strip()
        posting = parse_raw_text(input_text)

    analysis = pipeline.analyze(posting)

    # Persist the submission (retraining corpus + audit trail).
    sub = Submission(
        source_url=(body.url or "")[:2048],
        input_text=input_text[:20000],
        verdict=analysis.verdict,
        risk_score=analysis.risk_score,
        probability_fake=analysis.probability_fake,
        flagged_reasons=analysis.flagged_reasons,
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)

    payload = analysis.to_dict()
    payload["submission_id"] = sub.id
    payload["scrape_method"] = scrape_method
    payload["parsed"] = {
        "title": posting.title,
        "company": posting.company,
        "location": posting.location,
        "salary_range": posting.salary_range,
    }

    # Groq narrative explanation grounded in the model's own signals. Fails open:
    # if the LLM is off/unreachable, ai_explanation stays None and every
    # deterministic field is still returned.
    llm = generate_explanation(payload)
    payload["ai_explanation"] = {"summary": llm.summary, "model": llm.model} if llm else None

    return AnalyzeResponse(**payload)


@app.post("/feedback", response_model=FeedbackResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
def feedback(request: Request, body: FeedbackRequest, db: Session = Depends(get_session)) -> FeedbackResponse:
    """Record a user's correction for the retraining loop."""
    sub = db.get(Submission, body.submission_id)
    if sub is None:
        return FeedbackResponse(ok=False, message="Unknown submission_id.")

    user_label = 1 if body.correct_label == "fake" else 0
    predicted_fake = sub.verdict == "FAKE"
    is_correct = int(predicted_fake == bool(user_label))

    db.add(Feedback(
        submission_id=sub.id,
        predicted_verdict=sub.verdict,
        user_label=user_label,
        is_correct=is_correct,
        input_text=sub.input_text,
    ))
    db.commit()
    return FeedbackResponse(ok=True, message="Thank you — your feedback will improve the model.")
