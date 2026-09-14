"""Parse free-form pasted job text into a structured JobPosting.

When a user pastes raw text (rather than a URL), we still want the salary,
company, and location signals. This uses light regex heuristics on common
label patterns ("Salary:", "Company:", ...) and falls back to putting the whole
blob in `description` so the text branch always has something to work with.
"""
from __future__ import annotations

import re

from .features import JobPosting

_LABELS = {
    "title": re.compile(r"^\s*(?:job\s*title|title|position|role)\s*[:\-]\s*(.+)$", re.I | re.M),
    "company": re.compile(r"^\s*(?:company|employer|organization)\s*[:\-]\s*(.+)$", re.I | re.M),
    "location": re.compile(r"^\s*(?:location|based in|city)\s*[:\-]\s*(.+)$", re.I | re.M),
    "salary_range": re.compile(r"^\s*(?:salary|pay|compensation|wage)\s*[:\-]\s*(.+)$", re.I | re.M),
    "employment_type": re.compile(r"^\s*(?:employment\s*type|job\s*type)\s*[:\-]\s*(.+)$", re.I | re.M),
    "required_experience": re.compile(r"^\s*(?:required\s*experience|experience)\s*[:\-]\s*(.+)$", re.I | re.M),
    "required_education": re.compile(r"^\s*(?:required\s*education|education)\s*[:\-]\s*(.+)$", re.I | re.M),
}
# Inline salary pattern for when there's no explicit "Salary:" label.
_INLINE_SALARY = re.compile(
    r"(\$?\s?\d{2,3}(?:,?\d{3})?(?:\.\d+)?\s?k?\s*(?:-|–|to)\s*\$?\s?\d{2,3}(?:,?\d{3})?(?:\.\d+)?\s?k?)",
    re.I,
)

# Section headers we can split a pasted posting on, so structured fields
# (company_profile / requirements / benefits) populate like the training data.
_SECTION_HEADERS = {
    "company_profile": re.compile(r"^\s*(?:about\s*us|about\s*the\s*company|company\s*profile|who\s*we\s*are)\s*[:\-]?\s*$", re.I),
    "requirements": re.compile(r"^\s*(?:requirements?|qualifications?|what\s*you'?ll\s*need|must\s*have|skills)\s*[:\-]?\s*$", re.I),
    "benefits": re.compile(r"^\s*(?:benefits?|perks|what\s*we\s*offer|compensation\s*and\s*benefits)\s*[:\-]?\s*$", re.I),
    "description": re.compile(r"^\s*(?:the\s*role|responsibilities|job\s*description|description|what\s*you'?ll\s*do|about\s*the\s*role)\s*[:\-]?\s*$", re.I),
}


def _split_sections(text: str) -> dict[str, str]:
    """Group lines under recognized section headers. Unheadered text -> 'body'."""
    sections: dict[str, list[str]] = {"body": []}
    current = "body"
    for line in text.splitlines():
        header_hit = None
        for name, rx in _SECTION_HEADERS.items():
            if rx.match(line):
                header_hit = name
                break
        if header_hit:
            current = header_hit
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def parse_raw_text(text: str) -> JobPosting:
    """Best-effort structured extraction from pasted text."""
    text = (text or "").strip()
    fields: dict[str, str] = {}
    for name, rx in _LABELS.items():
        m = rx.search(text)
        if m:
            fields[name] = m.group(1).strip()

    # Title fallback: first non-empty line if not explicitly labeled.
    if "title" not in fields:
        for line in text.splitlines():
            if line.strip():
                fields["title"] = line.strip()[:140]
                break

    # Salary fallback: first inline currency range.
    if "salary_range" not in fields:
        m = _INLINE_SALARY.search(text)
        if m:
            fields["salary_range"] = m.group(1).strip()

    # Split into structured sections so metadata features fire correctly.
    sections = _split_sections(text)
    # Description = explicit 'the role' section if present, else the leftover body.
    description = sections.get("description") or sections.get("body") or text

    return JobPosting(
        title=fields.get("title", ""),
        company=fields.get("company", ""),
        location=fields.get("location", ""),
        salary_range=fields.get("salary_range", ""),
        employment_type=fields.get("employment_type", ""),
        required_experience=fields.get("required_experience", ""),
        required_education=fields.get("required_education", ""),
        company_profile=sections.get("company_profile", ""),
        requirements=sections.get("requirements", ""),
        benefits=sections.get("benefits", ""),
        # Full text still drives the text branch via combined_text().
        description=description,
        # Metadata we can't observe from raw text stays unknown (None -> 0 in
        # features), which is itself a mild fraud signal (missing info).
        telecommuting=None,
        has_company_logo=None,
        has_questions=None,
    )
