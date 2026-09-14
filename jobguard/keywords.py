"""Curated fraud-signal lexicons.

These lists encode domain knowledge about employment-scam tactics. They power
two things: (1) engineered boolean/count features fed to the tabular model, and
(2) the human-readable `flagged_reasons` shown in the UI. Each phrase maps to a
plain-English reason so explanations never leak raw feature names to the user.

Sources of these heuristics: FTC job-scam guidance, EMSCAD fraud analyses, and
common reshipping / money-mule / advance-fee patterns.
"""
from __future__ import annotations

# Phrase -> user-facing reason. Matched case-insensitively as substrings on the
# cleaned text. Keep phrases lowercase.
SUSPICIOUS_PHRASES: dict[str, str] = {
    # Money / financial requests (strongest scam signal)
    "wire transfer": "Mentions wiring money — a hallmark of employment scams",
    "western union": "References Western Union transfers, commonly used in scams",
    "moneygram": "References MoneyGram transfers, commonly used in scams",
    "bank account details": "Asks for bank account details up front",
    "bank details": "Asks for bank details up front",
    "personal bank": "Requests access to a personal bank account",
    "credit card details": "Asks for credit card details",
    "registration fee": "Requires a registration/processing fee to apply",
    "processing fee": "Requires an up-front processing fee",
    "application fee": "Charges an application fee (legitimate jobs never do)",
    "training fee": "Requires payment for training",
    "startup fee": "Requires an up-front startup fee",
    "pay a fee": "Requires the applicant to pay a fee",
    "upfront payment": "Requests an up-front payment from the applicant",
    "cashier's check": "Involves cashier's checks, common in overpayment scams",
    "money order": "Involves money orders, common in overpayment scams",
    "process payments": "Asks you to process payments (money-mule pattern)",
    "receive and forward": "Reshipping / package-forwarding scam pattern",
    "reship": "Reshipping scam pattern",

    # Too-good-to-be-true / urgency
    "no experience needed": "Promises income with no experience required",
    "no experience necessary": "Promises income with no experience required",
    "earn money fast": "Promises fast, easy money",
    "quick money": "Promises quick money",
    "unlimited earning": "Promises unlimited earnings",
    "work from home and earn": "Generic work-from-home earnings pitch",
    "be your own boss": "Uses generic 'be your own boss' recruitment language",
    "immediate start": "Pressures an immediate start",
    "start today": "Pressures you to start today",
    "limited positions": "Manufactures urgency with 'limited positions'",
    "apply now": "High-pressure 'apply now' language",
    "guaranteed income": "Guarantees income, which no real job can",
    "guaranteed job": "Guarantees a job outright",
    "100% guaranteed": "Makes a 100% guarantee",

    # Identity harvesting
    "social security number": "Requests your Social Security number before hiring",
    "ssn": "Requests your SSN before hiring",
    "passport copy": "Requests a passport copy up front",
    "driver's license copy": "Requests a copy of your driver's license up front",

    # Communication red flags
    "telegram": "Directs hiring to Telegram, common in scams",
    "whatsapp only": "Conducts hiring only over WhatsApp",
    "google hangouts": "Conducts interviews over Google Hangouts (classic scam tell)",
    "text only": "Insists on text-only communication",
}

# Words that, when frequent, signal hype/urgency. Used for a density feature.
URGENCY_WORDS: frozenset[str] = frozenset({
    "urgent", "immediately", "now", "hurry", "fast", "quick", "instant",
    "limited", "act", "today", "asap", "guaranteed", "exclusive", "amazing",
    "incredible", "unbelievable", "free", "bonus", "cash", "!!!",
})

# Free public email providers. A "company" using one is a metadata red flag.
FREE_EMAIL_DOMAINS: frozenset[str] = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "mail.com", "protonmail.com", "gmx.com", "yandex.com",
    "live.com", "msn.com", "ymail.com", "rocketmail.com", "zoho.com",
})


def find_suspicious_phrases(text: str) -> list[tuple[str, str]]:
    """Return (matched_phrase, human_reason) pairs present in `text`.

    Case-insensitive substring match. Deduplicated by reason so the UI never
    shows the same advice twice (e.g. 'bank details' and 'bank account details').
    """
    lowered = text.lower()
    hits: list[tuple[str, str]] = []
    seen_reasons: set[str] = set()
    for phrase, reason in SUSPICIOUS_PHRASES.items():
        if phrase in lowered and reason not in seen_reasons:
            hits.append((phrase, reason))
            seen_reasons.add(reason)
    return hits
