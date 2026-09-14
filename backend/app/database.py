"""Database setup (SQLAlchemy 2.0).

Defaults to a local SQLite file so the project runs with zero infra. Set
`JOBGUARD_DATABASE_URL` to a Postgres DSN (e.g. postgresql+psycopg://user:pass@db/jobguard)
to use the docker-compose Postgres service instead.

Stores only what the feedback loop needs: the analyzed text, the verdict, and any
user correction. No personal user identifiers are collected (ethical guardrail).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, DateTime, Float, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, sessionmaker)

from jobguard.config import settings

# SQLite needs a special flag for multithreaded FastAPI access.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, echo=False, future=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class Submission(Base):
    """One /analyze call and its verdict."""

    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    source_url: Mapped[str] = mapped_column(String(2048), default="")
    # Truncated text kept for retraining; not tied to any user identity.
    input_text: Mapped[str] = mapped_column(Text, default="")
    verdict: Mapped[str] = mapped_column(String(16), default="")
    risk_score: Mapped[int] = mapped_column(Integer, default=0)
    probability_fake: Mapped[float] = mapped_column(Float, default=0.0)
    flagged_reasons: Mapped[list] = mapped_column(JSON, default=list)


class Feedback(Base):
    """A user's correction of a verdict — the retraining signal."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    submission_id: Mapped[int] = mapped_column(Integer, index=True)
    predicted_verdict: Mapped[str] = mapped_column(String(16), default="")
    # What the user says the true label is: 1 = fake/fraudulent, 0 = genuine.
    user_label: Mapped[int] = mapped_column(Integer, default=0)
    is_correct: Mapped[int] = mapped_column(Integer, default=0)
    input_text: Mapped[str] = mapped_column(Text, default="")


def init_db() -> None:
    """Create tables if they don't exist. Idempotent; called at startup."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """FastAPI dependency yielding a scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
