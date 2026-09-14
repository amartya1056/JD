"""Central configuration, driven by environment variables.

Every path and threshold the rest of the system needs is resolved here, so the
application can be reconfigured at deploy time without touching code (a core
twelve-factor principle). Defaults are chosen so the project runs out of the box
on a fresh checkout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = two levels up from this file (jobguard/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency).

    Reads KEY=VALUE lines from `path` into os.environ WITHOUT overriding values
    already set in the real environment (env wins over file, the usual
    precedence). Silently ignores a missing file. Keeps secrets like the Groq API
    key out of source code and out of the process invocation.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# Load repo-root .env first, then a backend-local .env, before Settings is built.
_load_dotenv(REPO_ROOT / ".env")
_load_dotenv(REPO_ROOT / "backend" / ".env")


def _env_path(var: str, default: Path) -> Path:
    raw = os.getenv(var)
    return Path(raw).resolve() if raw else default


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot. Read once, injected everywhere."""

    # --- Filesystem layout ---
    repo_root: Path = REPO_ROOT
    models_dir: Path = field(default_factory=lambda: _env_path("JOBGUARD_MODELS_DIR", REPO_ROOT / "models"))
    data_dir: Path = field(default_factory=lambda: _env_path("JOBGUARD_DATA_DIR", REPO_ROOT / "data"))

    # Name of the promoted (production) model artifact bundle.
    model_bundle_name: str = os.getenv("JOBGUARD_MODEL_BUNDLE", "ensemble_latest.joblib")

    # --- Verdict thresholds (probability that the posting is REAL) ---
    # Interpretation: p_real >= real_threshold  -> REAL
    #                 p_real <  fake_threshold  -> FAKE
    #                 otherwise                 -> SUSPICIOUS
    real_threshold: float = float(os.getenv("JOBGUARD_REAL_THRESHOLD", "0.80"))
    fake_threshold: float = float(os.getenv("JOBGUARD_FAKE_THRESHOLD", "0.50"))

    # --- Ensemble weighting (tabular vs. text-transformer branch) ---
    # Weighted-average fallback used when no meta-learner is trained.
    tabular_weight: float = float(os.getenv("JOBGUARD_TABULAR_WEIGHT", "0.6"))
    text_weight: float = float(os.getenv("JOBGUARD_TEXT_WEIGHT", "0.4"))

    # --- Feature toggles ---
    use_bert: bool = os.getenv("JOBGUARD_USE_BERT", "auto").lower() not in {"0", "false", "off"}

    # --- Scraper ---
    scrape_timeout: int = int(os.getenv("JOBGUARD_SCRAPE_TIMEOUT", "15"))
    user_agent: str = os.getenv(
        "JOBGUARD_USER_AGENT",
        "Mozilla/5.0 (compatible; JobGuardBot/1.0; +https://example.com/bot)",
    )

    # --- Database (feedback loop) ---
    database_url: str = os.getenv("JOBGUARD_DATABASE_URL", "sqlite:///./jobguard.db")

    # --- API ---
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in os.getenv("JOBGUARD_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()
    )
    rate_limit_per_minute: int = int(os.getenv("JOBGUARD_RATE_LIMIT", "30"))

    # --- Groq LLM (natural-language explanations) ---
    # The key is read from the environment ONLY (never committed). Loaded from a
    # gitignored .env by jobguard.env_loading at import time.
    groq_api_key: str = os.getenv("JOBGUARD_GROQ_API_KEY", "")
    groq_model: str = os.getenv("JOBGUARD_GROQ_MODEL", "openai/gpt-oss-20b")
    groq_base_url: str = os.getenv("JOBGUARD_GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_timeout: float = float(os.getenv("JOBGUARD_GROQ_TIMEOUT", "12"))
    # Master toggle; the feature also silently no-ops when no key is present.
    use_llm_explanation: bool = os.getenv("JOBGUARD_USE_LLM", "on").lower() not in {"0", "false", "off"}

    @property
    def model_bundle_path(self) -> Path:
        return self.models_dir / self.model_bundle_name


# Module-level singleton. Import this everywhere.
settings = Settings()

# Verdict label constants — imported by API, UI serialization, and tests.
VERDICT_REAL = "REAL"
VERDICT_FAKE = "FAKE"
VERDICT_SUSPICIOUS = "SUSPICIOUS"
