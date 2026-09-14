"""Dataset loading: real EMSCAD/Kaggle CSV, with a synthetic fallback.

The production dataset is EMSCAD (a.k.a. the Kaggle "Real or Fake Job Posting
Prediction" set, file `fake_job_postings.csv`, ~18k rows, ~4% fraudulent).
`scripts/download_data.py` fetches it. When it is absent we generate a
statistically similar synthetic corpus so the whole pipeline runs on a fresh
checkout — with correlated fraud signals, not random noise, so the model has
real structure to learn.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

from .features import JobPosting

# --- Kaggle/EMSCAD column names -> JobPosting fields ---
_COLUMN_MAP = {
    "title": "title",
    "company_profile": "company_profile",
    "description": "description",
    "requirements": "requirements",
    "benefits": "benefits",
    "salary_range": "salary_range",
    "location": "location",
    "employment_type": "employment_type",
    "required_experience": "required_experience",
    "required_education": "required_education",
}
_INT_COLUMNS = {"telecommuting", "has_company_logo", "has_questions"}


def _row_to_posting(row: dict[str, str]) -> JobPosting:
    kwargs: dict[str, object] = {}
    for col, field in _COLUMN_MAP.items():
        kwargs[field] = (row.get(col) or "").strip()
    for col in _INT_COLUMNS:
        raw = (row.get(col) or "").strip()
        kwargs[col] = int(raw) if raw.isdigit() else 0
    # EMSCAD has no dedicated company-name column; fall back to industry text.
    kwargs["company"] = (row.get("department") or row.get("industry") or "").strip()
    return JobPosting(**kwargs)  # type: ignore[arg-type]


def load_emscad_csv(path: str | Path) -> tuple[list[JobPosting], list[int]]:
    """Load the real dataset. Returns (postings, labels) where 1 == fraudulent."""
    path = Path(path)
    postings: list[JobPosting] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            postings.append(_row_to_posting(row))
            labels.append(int((row.get("fraudulent") or "0").strip() or 0))
    return postings, labels


# ----------------------------------------------------------------------------
# Synthetic generator (fallback)
# ----------------------------------------------------------------------------

# Realistic titles — multi-word, with role qualifiers and acronyms (ML, UX, SRE),
# like real postings. Short/simple titles were teaching the model that longer
# titles = fraud, which flagged legitimate senior roles.
_LEGIT_TITLES = [
    "Senior Software Engineer, Backend Platform",
    "Research Scientist, Machine Learning",
    "Staff Data Engineer, Analytics Platform",
    "Product Manager, Growth",
    "Software Engineer, Distributed Systems",
    "Machine Learning Engineer, LLM Training",
    "Senior Product Designer (UX)",
    "Engineering Manager, Infrastructure",
    "Data Analyst, Business Intelligence",
    "Research Engineer, Reinforcement Learning",
    "Registered Nurse, ICU",
    "Financial Analyst, FP&A",
    "Marketing Manager, Demand Generation",
    "Customer Success Manager, Enterprise",
    "Site Reliability Engineer (SRE)",
    "Accountant, General Ledger",
]
_SCAM_TITLES = [
    "Work From Home Data Entry", "Personal Assistant Needed Urgent",
    "Online Rebate Processor", "Package Handling Agent", "Payment Processor",
    "Mystery Shopper", "Home-Based Typist", "Immediate Hire - No Experience",
]

# Detailed, acronym-rich legit paragraphs. These COMPOSE into long descriptions
# (real postings run 500–17,000+ characters). Crucially, the presence of length
# and acronyms must NOT signal fraud — real jobs look exactly like this.
_LEGIT_INTRO = [
    "About the role: We are looking for an experienced professional to join our "
    "team and help build the next generation of our platform. You will work "
    "cross-functionally with engineering, product, design, and data science to "
    "ship features that delight customers and move key business metrics.",
    "We are hiring a thoughtful, driven individual to take ownership of a core "
    "area of our product. This is a high-impact role with significant autonomy, "
    "a clear growth path, and the opportunity to shape both the technology and "
    "the team around it.",
    "Our mission is to build reliable, trustworthy systems at scale. In this role "
    "you will partner with senior leaders to define strategy, set priorities, and "
    "deliver results in a fast-moving, collaborative environment.",
]
_LEGIT_RESP = [
    "You will design, build, and operate large-scale distributed systems serving "
    "millions of requests per day. You will own services end to end, from API "
    "design through deployment on AWS and Kubernetes, and drive improvements in "
    "reliability, latency, and cost across the platform.",
    "You will train and evaluate large language models (LLMs), build data "
    "pipelines for pretraining and fine-tuning, and collaborate with researchers "
    "to turn experimental ideas into production ML systems. You will profile GPU "
    "workloads, optimize throughput, and contribute to our internal training "
    "frameworks in PyTorch.",
    "You will lead the roadmap for a core product area, partnering with "
    "engineering, design, and analytics to ship features customers love. You will "
    "define KPIs, run A/B tests, analyze results in SQL, and communicate findings "
    "to senior stakeholders across the organization.",
    "You will develop and maintain CI/CD pipelines, improve observability with "
    "metrics, logs, and traces, and champion engineering best practices including "
    "code review, testing, and incident response. You will mentor junior engineers "
    "and help raise the bar for technical excellence.",
    "You will analyze large datasets to surface actionable insights, build "
    "dashboards and models, and partner with business teams to inform decisions. "
    "You will write clean, well-tested code in Python and SQL and present your "
    "work clearly to both technical and non-technical audiences.",
    "You will own the reliability of critical services, define SLOs, reduce toil "
    "through automation, and participate in an on-call rotation. You will work "
    "closely with product engineering to design systems that are secure, scalable, "
    "and resilient by default.",
]
_LEGIT_ABOUT = [
    "Our team values rigorous thinking, clear written communication, and a bias "
    "toward action. We work in small, autonomous groups and invest heavily in "
    "mentorship, documentation, and engineering excellence. We are a hybrid "
    "workplace with offices in San Francisco, New York, and London.",
    "We are a diverse, mission-driven organization that cares deeply about doing "
    "the right thing for our customers and each other. We offer a supportive, "
    "inclusive culture, generous learning budgets, and real opportunities for "
    "career growth.",
]
_LEGIT_REQ = [
    "Requirements: 5+ years of professional software engineering experience, "
    "strong proficiency in Python, Go, or Java, and a solid understanding of data "
    "structures, algorithms, and system design. Experience with cloud platforms "
    "(AWS, GCP, or Azure) and CI/CD pipelines is expected.",
    "You should have a BS or MS in Computer Science or a related field, hands-on "
    "experience with PyTorch or TensorFlow, and familiarity with GPUs, CUDA, and "
    "distributed training. Publications at NeurIPS, ICML, or ICLR are a plus but "
    "not required.",
    "Qualifications: Bachelor's degree and 3+ years of relevant experience, "
    "excellent analytical and communication skills, and the ability to work both "
    "independently and within a cross-functional team. Proficiency with SQL and "
    "modern BI tools is strongly preferred.",
    "You bring a proven track record in a similar role, strong stakeholder "
    "management, and the ability to translate ambiguous problems into clear, "
    "measurable outcomes. Experience in a regulated or high-growth environment is "
    "a plus.",
]
_LEGIT_BENEFITS = [
    "We offer competitive compensation including equity, comprehensive health, "
    "dental, and vision insurance, a generous 401(k) match, flexible paid time "
    "off, paid parental leave, and an annual learning and development budget. "
    "Relocation assistance is available.",
    "Benefits include medical, dental, and vision coverage, a wellness stipend, "
    "commuter benefits, and a supportive hybrid work policy. We are an equal "
    "opportunity employer and celebrate a diverse, inclusive workforce.",
    "Health, dental, and vision insurance. 401(k) matching. Paid parental leave "
    "and a professional development budget.",
]
# Short neutral descriptions — kept for a small fraction of SHORT legit posts and
# for the blur logic, so length does not correlate with the label.
_NEUTRAL_DESC = [
    "We are hiring a remote associate to support our operations team. Flexible "
    "hours and weekly pay. Apply with your resume today.",
    "Seeking a motivated individual for an administrative role. Duties include "
    "scheduling, data entry, and correspondence. Training provided.",
    "Join our customer support team working from home. Competitive pay and a "
    "supportive environment. Send your application to get started.",
    "Our organization is looking for a detail-oriented individual to manage day "
    "to day operations and drive continuous improvement in a hybrid environment.",
]
_SCAM_DESC = [
    "Earn money fast working from home! No experience needed. Guaranteed income "
    "of $5000 per week. Immediate start, apply now! Send your bank account "
    "details to receive your first payment. Limited positions available!",
    "Be your own boss! Process payments and receive and forward packages from "
    "home. We will wire transfer your commission via Western Union. A small "
    "registration fee is required to activate your account. Contact us on Telegram.",
    "Unlimited earning potential! Quick money guaranteed. Simply provide your "
    "social security number and a copy of your driver's license to get started "
    "today. Pay a one-time processing fee and start earning immediately!",
]
# Filler used to pad a fraction of scams to longer lengths, so that "long" does
# not imply "legitimate". The scam phrases remain the real signal.
_SCAM_FILLER = [
    "This is a once in a lifetime opportunity! Act now, positions are filling "
    "fast! Do not miss out on this amazing chance to earn from home!",
    "No interview required. Simply pay the small activation fee and start today. "
    "We will send your payment via wire transfer once you provide your details.",
    "Work whenever you want and earn unlimited cash. Refer your friends for even "
    "bigger bonuses. Contact our agent on WhatsApp or Telegram right away!",
]
_LOCATIONS = ["US, NY, New York", "US, CA, San Francisco", "GB, LND, London",
              "US, TX, Austin", "CA, ON, Toronto", "", "US, , Remote"]
_EMP_TYPES = ["Full-time", "Part-time", "Contract", ""]
_EXP = ["Mid-Senior level", "Entry level", "Associate", ""]
_EDU = ["Bachelor's Degree", "Master's Degree", "High School or equivalent", ""]


def _compose_legit_description(rng: random.Random) -> str:
    """Assemble a realistic legit description spanning a WIDE length range.

    Real postings run from a few hundred to 17,000+ characters. Building to a
    random target length (with paragraph repetition allowed) covers that whole
    range, so the model stops treating length as a fraud signal — the bug that
    flagged a real 17k-char Anthropic posting as fake.
    """
    if rng.random() < 0.12:
        return rng.choice(_NEUTRAL_DESC)  # a few genuinely short legit posts
    target = rng.choice([600, 900, 1400, 2200, 3500, 6000, 10000, 15000])
    pool = _LEGIT_RESP + _LEGIT_ABOUT + _LEGIT_REQ + _LEGIT_BENEFITS
    blocks = [rng.choice(_LEGIT_INTRO)]
    while len("\n\n".join(blocks)) < target:
        blocks.append(rng.choice(pool))
    return "\n\n".join(blocks)


def _compose_scam_description(rng: random.Random) -> str:
    """Scam text: always carries scam phrases, but its LENGTH varies (some long,
    padded with filler) so length never implies legitimacy."""
    desc = rng.choice(_SCAM_DESC)
    if rng.random() < 0.6:
        desc += " Contact us at hr.recruiter2024@gmail.com to apply."
    if rng.random() < 0.35:  # a third of scams are padded long
        target = rng.choice([2000, 4000, 8000])
        while len(desc) < target:
            desc += "\n\n" + rng.choice(_SCAM_DESC + _SCAM_FILLER)
    return desc


def _legit_posting(rng: random.Random) -> JobPosting:
    title = rng.choice(_LEGIT_TITLES)
    company = rng.choice(["Acme Corp", "Globex", "Initech", "Umbrella Health",
                          "Northwind Traders", "Stark Industries", "Vantage Labs",
                          "Meridian Systems"])
    low = rng.choice([45, 60, 70, 85, 95, 120, 150, 180])
    high = low + rng.choice([15, 20, 25, 30, 40, 60])
    # Genuine posts vary a lot in completeness — many real (ATS-hosted) listings
    # keep everything inside one long description and expose no separate company
    # profile / requirements / benefits fields. Modeling that stops the classifier
    # from treating a "missing section" as strong fraud evidence.
    return JobPosting(
        title=title,
        company=company,
        company_profile=(f"{company} is an established organization with a strong "
                         "reputation and a diverse, inclusive culture."
                         if rng.random() < 0.45 else ""),
        description=_compose_legit_description(rng),
        requirements=rng.choice(_LEGIT_REQ) if rng.random() < 0.5 else "",
        benefits=rng.choice(_LEGIT_BENEFITS) if rng.random() < 0.45 else "",
        location=rng.choice(_LOCATIONS),
        salary_range=(f"${low},000 - ${high},000" if rng.random() < 0.6 else ""),
        employment_type=rng.choice(_EMP_TYPES),
        required_experience=rng.choice(_EXP),
        required_education=rng.choice(_EDU),
        telecommuting=rng.choice([0, 0, 0, 1]),
        has_company_logo=rng.choice([1, 1, 1, 0]),  # usually present
        has_questions=rng.choice([1, 0]),
    )


def _scam_posting(rng: random.Random) -> JobPosting:
    title = rng.choice(_SCAM_TITLES)
    # Scam "companies" are vague or missing; often free email.
    company = rng.choice(["", "Global Opportunities", "Home Careers LLC", "HR Dept"])
    # Inflated / too-good salary sometimes present.
    salary = rng.choice([
        "$8,000 - $15,000", "$200,000 - $500,000", "", "$5,000 - $9,000",
    ])
    desc = _compose_scam_description(rng)
    return JobPosting(
        title=title,
        company=company,
        company_profile="" if rng.random() < 0.7 else "We are a fast growing team.",
        description=desc,
        requirements="" if rng.random() < 0.6 else "No experience necessary.",
        benefits="" if rng.random() < 0.8 else "Guaranteed weekly pay.",
        location=rng.choice(_LOCATIONS),
        salary_range=salary,
        employment_type=rng.choice(_EMP_TYPES),  # often blank
        required_experience=rng.choice(["", "", "Entry level"]),
        required_education=rng.choice(["", "", "High School or equivalent"]),
        telecommuting=rng.choice([1, 1, 0]),  # scams love "remote"
        has_company_logo=rng.choice([0, 0, 0, 1]),  # usually missing
        has_questions=rng.choice([0, 0, 1]),
    )


def _blur(posting: JobPosting, rng: random.Random, toward_scam: bool) -> JobPosting:
    """Inject class overlap so the task is realistically hard (not separable).

    A genuine post may occasionally look sketchy (missing logo, urgent tone) and
    a scam may look polished (corporate email, has logo). Without this, every
    model scores a meaningless 1.000 because the label is trivially decodable.
    """
    if toward_scam:  # make a genuine post look suspicious (false-alarm risk)
        if rng.random() < 0.5:
            posting.has_company_logo = 0
        if rng.random() < 0.6:
            posting.description += (
                " Immediate start, apply now! Limited positions available, "
                "act fast for this amazing opportunity."
            )
        if rng.random() < 0.4:
            posting.salary_range = "$120,000 - $260,000"
        if rng.random() < 0.3:
            posting.company_profile = ""
    else:  # make a scam look legitimate (hardest cases: benign text + weak metadata)
        if rng.random() < 0.5:
            posting.has_company_logo = 1
        if rng.random() < 0.5:
            posting.company = rng.choice(["Bluewave Solutions", "Meridian Group",
                                          "Cedar & Co", "Vantage Partners"])
            posting.company_profile = "An established firm serving clients nationwide."
        # The crucial move: often replace scam text with fully benign wording, so
        # the text branch can't catch it and the model must lean on metadata.
        if rng.random() < 0.7:
            # Fully sanitize EVERY text field from the shared/neutral pools so no
            # token leaks the label. These rows become text-identical to genuine
            # posts; only metadata (telecommuting, missing fields, has_questions,
            # email domain) can flag them -> forces the ensemble to earn its keep.
            posting.title = rng.choice(_LEGIT_TITLES)
            posting.description = rng.choice(_NEUTRAL_DESC)
            posting.requirements = rng.choice(_LEGIT_REQ + [""])
            posting.benefits = rng.choice(_LEGIT_BENEFITS + [""])
            posting.salary_range = rng.choice(["", "$40,000 - $55,000", "$50,000 - $65,000"])
            # ~20% become "perfect mimics": legit-looking metadata too, so they
            # are near-irreducible. This keeps the ensemble's recall realistically
            # below 100% instead of a suspiciously perfect score.
            if rng.random() < 0.2:
                posting.has_company_logo = 1
                posting.has_questions = 1
                posting.employment_type = "Full-time"
                posting.required_experience = "Mid-Senior level"
                posting.required_education = "Bachelor's Degree"
                posting.telecommuting = 0
                posting.company_profile = "An established firm serving clients nationwide."
    return posting


def generate_synthetic(
    n: int = 4000, fraud_rate: float = 0.045, seed: int = 42, overlap: float = 0.55
) -> tuple[list[JobPosting], list[int]]:
    """Create an imbalanced synthetic corpus with correlated fraud signals.

    The default ~4.5% fraud rate mirrors EMSCAD, so class-imbalance handling
    (SMOTE, class weights, PR-AUC) is genuinely exercised. `overlap` controls how
    often a posting is blurred toward the other class, keeping the task hard.
    """
    rng = random.Random(seed)
    n_fraud = max(1, int(round(n * fraud_rate)))
    postings: list[JobPosting] = []
    labels: list[int] = []
    for i in range(n):
        is_fraud = i < n_fraud
        post = _scam_posting(rng) if is_fraud else _legit_posting(rng)
        if rng.random() < overlap:
            post = _blur(post, rng, toward_scam=not is_fraud)
        postings.append(post)
        labels.append(1 if is_fraud else 0)
    # Shuffle so fraud isn't all at the front.
    combined = list(zip(postings, labels))
    rng.shuffle(combined)
    postings, labels = map(list, zip(*combined))
    return postings, labels  # type: ignore[return-value]


def load_dataset(
    csv_path: str | Path | None = None, synthetic_n: int = 4000
) -> tuple[list[JobPosting], list[int], str]:
    """Unified entry point. Returns (postings, labels, source_description)."""
    if csv_path and Path(csv_path).exists():
        postings, labels = load_emscad_csv(csv_path)
        return postings, labels, f"EMSCAD CSV ({Path(csv_path).name}, {len(labels)} rows)"
    postings, labels = generate_synthetic(synthetic_n)
    return postings, labels, f"synthetic fallback ({len(labels)} rows)"
