"""API tests for /health, /analyze, /feedback — with the scraper mocked.

These require a trained model bundle at models/ensemble_latest.joblib. If it is
missing the whole module is skipped with a clear message (run scripts/train.py).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobguard.config import settings

pytestmark = pytest.mark.skipif(
    not settings.model_bundle_path.exists(),
    reason="No trained model bundle. Run `python scripts/train.py` first.",
)

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import main as main_module  # noqa: E402
from jobguard.features import JobPosting  # noqa: E402
from jobguard.scraper import ScrapeResult  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # `with` triggers the lifespan handler -> model loads once for the module.
    with TestClient(main_module.app) as c:
        yield c


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_analyze_flags_obvious_scam(client):
    text = (
        "URGENT!! Work from home and earn $5000/week. No experience needed. "
        "Pay a registration fee and send your bank account details. Commission "
        "paid by wire transfer via Western Union. Contact us on Telegram."
    )
    r = client.post("/analyze", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in {"FAKE", "SUSPICIOUS"}
    assert body["risk_score"] >= 50
    assert body["submission_id"] > 0
    assert len(body["flagged_reasons"]) > 0
    assert "advice" in body and body["advice"]


def test_analyze_requires_input(client):
    r = client.post("/analyze", json={})
    assert r.status_code == 422  # pydantic validation error


def test_analyze_with_url_uses_scraper(client, monkeypatch):
    """URL path should call the scraper; we mock it so there's no network."""
    def fake_scrape(url):
        return ScrapeResult(
            ok=True,
            posting=JobPosting(
                title="Data Engineer",
                company="Contoso",
                company_profile="A cloud company.",
                description="Build data pipelines. Collaborate with analytics teams.",
                requirements="3 years experience.",
                benefits="Health insurance.",
                salary_range="$120,000 - $150,000",
                has_company_logo=1,
            ),
            method="static+jsonld",
        )

    monkeypatch.setattr(main_module, "scrape", fake_scrape)
    r = client.post("/analyze", json={"url": "https://example.com/job/1"})
    assert r.status_code == 200
    body = r.json()
    assert body["scrape_method"] == "static+jsonld"
    assert body["parsed"]["company"] == "Contoso"


def test_analyze_url_scrape_failure_returns_422(client, monkeypatch):
    def failing_scrape(url):
        return ScrapeResult(ok=False, error="blocked — please paste the text")

    monkeypatch.setattr(main_module, "scrape", failing_scrape)
    r = client.post("/analyze", json={"url": "https://blocked.example.com"})
    assert r.status_code == 422
    assert "paste" in r.json()["detail"].lower()


def test_feedback_records_correction(client):
    # First create a submission to reference.
    r = client.post("/analyze", json={"text": "Some job posting text that is long enough."})
    sub_id = r.json()["submission_id"]
    fb = client.post("/feedback", json={"submission_id": sub_id, "correct_label": "fake"})
    assert fb.status_code == 200
    assert fb.json()["ok"] is True
