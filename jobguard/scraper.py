"""Job-posting scraper: static (requests+BeautifulSoup) with a JS fallback.

Extraction strategy, most reliable first:
  1. schema.org JobPosting JSON-LD  — embedded by most ATS/job boards (Greenhouse,
     Lever, Workday, Indeed). Structured and trustworthy.
  2. Open Graph / meta tags + heuristic body text.
  3. Playwright render (optional) — only if the static fetch looks JS-gated and
     Playwright is installed.

Everything degrades gracefully: on any failure we return a ScrapeResult with
`ok=False` and a message telling the caller to fall back to pasted text. The
scraper never raises to the API layer.

Domain age (a fraud signal) is looked up via python-whois when available. It is
optional because reliable WHOIS often needs a paid API; we note the trade-off and
skip it silently otherwise.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from .config import settings
from .features import JobPosting


@dataclass
class ScrapeResult:
    ok: bool
    posting: JobPosting | None = None
    method: str = ""
    error: str = ""
    domain_age_days: int | None = None


_JOB_SALARY_KEYS = ("baseSalary", "estimatedSalary")


def _clean_html_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()


def _extract_jsonld_jobposting(html: str) -> dict | None:
    """Find a schema.org JobPosting object in any JSON-LD <script> block."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        # Some sites wrap objects in an @graph list.
        for c in list(candidates):
            if isinstance(c, dict) and "@graph" in c and isinstance(c["@graph"], list):
                candidates.extend(c["@graph"])
        for c in candidates:
            if isinstance(c, dict) and str(c.get("@type", "")).lower() == "jobposting":
                return c
    return None


def _salary_from_jsonld(node: dict) -> str:
    for key in _JOB_SALARY_KEYS:
        sal = node.get(key)
        if isinstance(sal, dict):
            value = sal.get("value", sal)
            if isinstance(value, dict):
                lo = value.get("minValue")
                hi = value.get("maxValue")
                if lo and hi:
                    cur = sal.get("currency", "")
                    return f"{cur} {lo} - {hi}".strip()
                if value.get("value"):
                    return str(value["value"])
    return ""


def _posting_from_jsonld(node: dict, url: str) -> JobPosting:
    org = node.get("hiringOrganization") or {}
    company = org.get("name", "") if isinstance(org, dict) else str(org)
    loc = node.get("jobLocation") or {}
    if isinstance(loc, list):
        loc = loc[0] if loc else {}
    address = loc.get("address", {}) if isinstance(loc, dict) else {}
    location = ""
    if isinstance(address, dict):
        location = ", ".join(
            str(address.get(k, "")) for k in ("addressLocality", "addressRegion", "addressCountry")
            if address.get(k)
        )
    return JobPosting(
        title=node.get("title", ""),
        company=company,
        company_profile=(org.get("description", "") if isinstance(org, dict) else ""),
        description=_clean_html_text(node.get("description", "")) if node.get("description") else "",
        requirements=_clean_html_text(node.get("qualifications", "") or node.get("experienceRequirements", "") or "") if isinstance(node.get("qualifications", ""), str) else "",
        benefits=_clean_html_text(node.get("jobBenefits", "")) if isinstance(node.get("jobBenefits", ""), str) else "",
        location=location,
        salary_range=_salary_from_jsonld(node),
        employment_type=(", ".join(node["employmentType"]) if isinstance(node.get("employmentType"), list) else node.get("employmentType", "")),
        required_education=str(node.get("educationRequirements", "") or "")[:120],
        required_experience=str(node.get("experienceRequirements", "") or "")[:120],
        has_company_logo=1 if (isinstance(org, dict) and org.get("logo")) else 0,
        source_url=url,
    )


def _looks_js_gated(text: str) -> bool:
    """Heuristic: very little body text usually means client-side rendering."""
    return len(text.split()) < 40


def _domain_age_days(url: str) -> int | None:
    """Best-effort domain age via python-whois. Optional dependency.

    Trade-off note: free WHOIS is rate-limited and inconsistent across TLDs. For
    production accuracy a paid API (e.g. WhoisXML) is recommended. We fail silent.
    """
    try:
        import whois  # type: ignore
        from datetime import datetime

        domain = urlparse(url).netloc.split(":")[0]
        info = whois.whois(domain)
        created = info.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            return max(0, (datetime.now() - created).days)
    except Exception:
        return None
    return None


def _static_fetch(url: str) -> str:
    import requests

    resp = requests.get(
        url, timeout=settings.scrape_timeout,
        headers={"User-Agent": settings.user_agent, "Accept-Language": "en-US,en;q=0.9"},
    )
    resp.raise_for_status()
    return resp.text


def _playwright_fetch(url: str) -> str | None:
    """Render a JS-heavy page if Playwright is installed. Optional."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=settings.user_agent)
            page.goto(url, timeout=settings.scrape_timeout * 1000, wait_until="networkidle")
            html = page.content()
            browser.close()
            return html
    except Exception:
        return None


def scrape(url: str) -> ScrapeResult:
    """Fetch and parse a posting URL. Never raises."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ScrapeResult(ok=False, error="Invalid URL. Provide an http(s) link or paste the text.")

    # 1. Static fetch.
    try:
        html = _static_fetch(url)
    except Exception as exc:  # network error, 404, paywall, etc.
        return ScrapeResult(
            ok=False,
            error=f"Could not fetch the page ({type(exc).__name__}). "
                  "The site may block bots or require login — please paste the text instead.",
        )

    method = "static"
    node = _extract_jsonld_jobposting(html)
    body_text = _clean_html_text(html)

    # 2. If static looks JS-gated and no structured data, try Playwright.
    if node is None and _looks_js_gated(body_text):
        rendered = _playwright_fetch(url)
        if rendered:
            html = rendered
            method = "playwright"
            node = _extract_jsonld_jobposting(html)
            body_text = _clean_html_text(html)

    if node is not None:
        posting = _posting_from_jsonld(node, url)
        # Ensure the text branch has content even if description was thin.
        if len(posting.description.split()) < 20:
            posting.description = (posting.description + "\n" + body_text).strip()
        method += "+jsonld"
    else:
        if _looks_js_gated(body_text):
            return ScrapeResult(
                ok=False,
                error="The page appears to require JavaScript or login and no job "
                      "data was found. Please paste the posting text instead.",
            )
        # 3. Heuristic fallback from raw text.
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")
        posting = JobPosting(title=title[:140], description=body_text, source_url=url)

    return ScrapeResult(ok=True, posting=posting, method=method,
                        domain_age_days=_domain_age_days(url))
