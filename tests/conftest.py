"""Shared pytest fixtures + test isolation.

We point the app at a throwaway SQLite database BEFORE any backend module imports
`settings`, so tests never touch a real database. Env is the injection point
because jobguard.config builds an immutable Settings snapshot at import time.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# Isolated DB + permissive CORS/rate limit for tests. Must be set before imports.
_TMP_DB = Path(tempfile.gettempdir()) / "jobguard_test.db"
os.environ.setdefault("JOBGUARD_DATABASE_URL", f"sqlite:///{_TMP_DB.as_posix()}")
os.environ.setdefault("JOBGUARD_RATE_LIMIT", "1000")
# Never call the external Groq LLM during tests (keeps them fast, offline,
# deterministic, and free). The endpoint falls back to deterministic fields.
os.environ.setdefault("JOBGUARD_USE_LLM", "off")
