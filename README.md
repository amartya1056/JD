<div align="center">

# 🛡️ JobGuard

### Explainable Machine-Learning Detection of Fraudulent Job Postings

Paste a job posting URL or its text — JobGuard scrapes it, extracts three parallel
signal categories, runs a stacked ML ensemble, and returns a **verdict**
(`REAL` / `SUSPICIOUS` / `FAKE`) with a confidence score, plain-English flagged
reasons, SHAP feature attributions, highlighted suspicious phrases, and an
AI-written analyst summary.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Vite-61DAFB?logo=react&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-ensemble-EB5E28)
![DistilBERT](https://img.shields.io/badge/DistilBERT-optional-FFBF00?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

---

## Table of Contents

- [What it does](#what-it-does)
- [Why it matters](#why-it-matters)
- [Key features](#key-features)
- [System architecture](#system-architecture)
- [How the ML works](#how-the-ml-works)
- [The dataset (EMSCAD)](#the-dataset-emscad)
- [Model performance](#model-performance)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Training the model](#training-the-model)
- [Running the API](#running-the-api)
- [Running the frontend](#running-the-frontend)
- [Docker](#docker-deployment)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Explainability](#explainability)
- [AI analyst summary (Groq)](#ai-analyst-summary-groq)
- [Fine-tuned DistilBERT branch](#fine-tuned-distilbert-branch)
- [Feedback loop & continuous learning](#feedback-loop--continuous-learning)
- [Testing](#testing)
- [Windows notes](#windows-notes)
- [Limitations & ethics](#limitations--ethical-considerations)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## What it does

Employment scams cost job seekers money and expose them to identity theft.
JobGuard is an end-to-end platform that estimates whether a job posting is
genuine or fraudulent and — crucially — **explains why** in language a
non-technical user understands.

A user pastes a URL (or the raw posting text). JobGuard:

1. **Scrapes / parses** the posting into a structured record (title, company,
   salary, description, metadata).
2. **Extracts three signal categories** in parallel — NLP text, salary anomalies,
   and company/metadata red flags.
3. **Runs a stacked ensemble** (XGBoost on engineered features + a text
   classifier) fused by a logistic-regression meta-learner.
4. **Applies confidence thresholds** to produce a `REAL` / `SUSPICIOUS` / `FAKE`
   verdict with a 0–100 risk index.
5. **Explains the decision** with SHAP (tabular), phrase saliency (text), a
   deterministic rule layer, and an LLM-written narrative.
6. **Learns over time** — user corrections are stored and folded into a guarded
   retraining pipeline.

## Why it matters

On a dataset where only ~5% of postings are fraudulent, **accuracy is a trap**: a
model that always predicts "REAL" scores ~95% accuracy while catching *zero*
scams. JobGuard is therefore optimized and evaluated on **PR-AUC and recall for
the FAKE class**, because missing a scam is far more harmful than a false alarm.
Every verdict is presented as a *probability, not a certainty*, and the UI always
advises independent verification.

---

## Key features

| Category | Highlights |
|----------|-----------|
| **Detection** | Stacked ensemble (XGBoost + TF-IDF/LogReg text branch), out-of-fold meta-learner, optional fine-tuned DistilBERT text branch |
| **Signals** | 25 named engineered features across text, salary and metadata; scam-phrase lexicon; salary-anomaly and email-domain-mismatch detectors |
| **Explainability** | SHAP feature attributions, text-model phrase saliency, deterministic rule flags, and a Groq LLM "Analyst Summary" grounded in the model's own signals |
| **Scraping** | Static (`requests` + BeautifulSoup) with schema.org JSON-LD extraction, optional Playwright JS fallback, optional WHOIS domain age; graceful failure to "paste text" |
| **API** | Async FastAPI — `/analyze`, `/feedback`, `/health` — with CORS, rate limiting, input validation and OpenAPI docs |
| **UI** | React + Vite + Tailwind single-page app: premium "trust report" design, color-coded verdict card, risk gauge, highlighted phrases, feedback |
| **Learning** | Every submission + correction persisted; `retrain.py` retrains and **only promotes** a new model if PR-AUC improves |
| **Ops** | Env-driven config, model registry (`registry.jsonl`), Dockerfiles + `docker-compose` (Postgres + backend + frontend) |

---

## System architecture

```
 User input (URL / pasted text)
   │
   ▼
 ┌───────────────────────┐   scrape (JSON-LD → HTML → Playwright)  OR  parse raw text
 │  Scraper / Parser     │   → structured JobPosting
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐   clean · tokenize · stopword removal · lemmatize
 │  Preprocessing        │
 └───────────┬───────────┘
             ▼
 ┌───────────────────────────────────────────────────────────────┐
 │  Feature extraction (3 parallel branches)                      │
 │  (a) NLP text     → TF-IDF (word + char n-grams)  [+DistilBERT]│
 │  (b) Salary       → range parsing, "too good to be true" flags │
 │  (c) Metadata     → logo, profile, email domain, links, caps   │
 └───────────┬───────────────────────────────────────────────────┘
             ▼
 ┌───────────────────────┐   XGBoost (tabular) ─┐
 │  Stacked Ensemble     │                      ├─► Logistic-Regression meta-learner
 │                       │   Text branch ───────┘        (out-of-fold stacking)
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐   p_real ≥ 0.80 → REAL · < 0.50 → FAKE · else SUSPICIOUS
 │  Verdict thresholding │   → risk index 0–100
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐   SHAP (tabular) + phrase saliency (text)
 │  Explainability       │   + deterministic rule flags + Groq LLM narrative
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐   verdict · reasons · advice   (React verdict card)
 │  Verdict display      │
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐   user "report wrong verdict" → Postgres/SQLite
 │  Feedback loop        │   → retrain.py (guarded promotion by PR-AUC)
 └───────────────────────┘
```

The core ML + scraping + explainability logic lives in one installable package,
**`jobguard/`**, imported by both the training scripts and the API. This
guarantees **train/serve parity** — identical preprocessing and features at fit
time and inference time.

---

## How the ML works

### Three signal branches

**(a) Text (NLP).** The combined text (`title + company_profile + description +
requirements + benefits`) is cleaned and vectorized with **TF-IDF word (1–2 gram)
and character (3–5 gram) features**. Character n-grams catch obfuscated scam
tokens (`w1re transfer`). A class-weighted Logistic Regression classifies it.
An optional **fine-tuned DistilBERT** branch is a drop-in replacement behind the
same interface.

**(b) Salary.** Salary strings (`$50,000 - $70,000`, `40k-60k`, `90000 to 110000`)
are parsed into ranges. Features encode presence, span width, and a
"too-good-to-be-true" flag (inflated ceilings or absurdly wide ranges).

**(c) Company / metadata.** 25 named engineered features (see
`jobguard/features.py`) including: `has_company_logo`, `has_company_profile`,
`telecommuting`, `has_questions`, ALL-CAPS ratio, exclamation count, urgency-word
density, suspicious-phrase count, external-link count, free-email flag, and
**email-domain-mismatch** (recruiter's email domain vs. the company name).

### The stacked ensemble

```
XGBoost(25 engineered features)  ──►  p_tabular ─┐
                                                 ├──►  LogReg meta-learner  ──►  p_fraud
Text branch (TF-IDF+LR / DistilBERT) ──► p_text ─┘
```

The base models' probabilities are combined by a logistic-regression
**meta-learner** trained on **out-of-fold** predictions (via `cross_val_predict`)
so the meta-learner never sees a base model's prediction on data that model was
trained on — this prevents the overconfidence/leakage that naive stacking causes.

### Why XGBoost gets only 25 named features

Keeping the tabular model's input space small and *named* is a deliberate
explainability choice: SHAP values map directly to human sentences like *"Salary
looks too good to be true"* instead of `tfidf_dim_4471`. The high-dimensional text
signal lives in its own branch, explained separately via phrase saliency.

### Class imbalance

EMSCAD is ~5% fraud. JobGuard handles this with **class weights**
(`class_weight="balanced"`, XGBoost `scale_pos_weight`) and supports **SMOTE**
(`imbalanced-learn`) on the tabular branch via `--smote`. Evaluation leads with
**PR-AUC** and **FAKE-class recall**, never raw accuracy.

### Verdict thresholding

On `p_real = 1 − p_fraud`:

| Condition | Verdict |
|-----------|---------|
| `p_real ≥ 0.80` | 🟢 **REAL** |
| `p_real < 0.50` | 🔴 **FAKE** |
| otherwise | 🟡 **SUSPICIOUS** |

Thresholds are configurable via env vars (`JOBGUARD_REAL_THRESHOLD`,
`JOBGUARD_FAKE_THRESHOLD`).

---

## The dataset (EMSCAD)

JobGuard trains on the **Employment Scam Aegean Dataset (EMSCAD)** — the standard
benchmark for this task, also published on Kaggle as *Real / Fake Job Posting
Prediction* (`fake_job_postings.csv`).

- **17,880** real-world job postings
- **866 fraudulent (4.84%)**, 17,014 genuine — severely imbalanced
- Text columns (title, company profile, description, requirements, benefits) plus
  categorical/boolean metadata (telecommuting, logo, questions, employment type,
  required experience/education)

Download it:

```bash
# Option A — direct (no credentials):
curl -L -o data/fake_job_postings.csv \
  https://raw.githubusercontent.com/Erfaniaa/fake-job-posting-detection/master/dataset.csv

# Option B — Kaggle API (needs ~/.kaggle/kaggle.json):
python scripts/download_data.py
```

If the CSV is absent, the pipeline automatically falls back to a **synthetic
generator** so the project still runs on a fresh checkout — but real metrics
require the real dataset.

---

## Model performance

Trained on the full EMSCAD dataset (80/20 stratified split, positive class =
FAKE). Metrics below are on the held-out 20% test set.

<!-- METRICS_TABLE -->
_(populated by `scripts/train.py` — see the training log / `models/registry.jsonl`)_

> **Read accuracy in context.** Because only ~5% of postings are fraudulent, a
> trivial "always REAL" classifier already scores ~95% accuracy. The numbers that
> actually measure fraud-catching ability are **PR-AUC** and **FAKE-class recall**.
> JobGuard also reports **balanced accuracy** (the mean of per-class recall), which
> is not inflated by the majority class.

Full evaluation (confusion matrix, per-class classification report at both the
default 0.5 threshold and a recall-optimized threshold) is printed by `train.py`
and can be reproduced with:

```bash
python scripts/train.py --data data/fake_job_postings.csv
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11, FastAPI (async), Uvicorn |
| **ML** | scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP; optional HuggingFace Transformers (DistilBERT), NLTK |
| **Scraping** | requests + BeautifulSoup + lxml (static & JSON-LD), Playwright (JS fallback), python-whois (domain age) |
| **LLM** | Groq (OpenAI-compatible API) for the natural-language Analyst Summary |
| **Frontend** | React 18, Vite, TailwindCSS |
| **Storage** | SQLAlchemy 2.0 — SQLite by default, PostgreSQL in Docker |
| **Serving** | Model loaded once at startup, cached in memory |
| **Ops** | Dockerfiles + docker-compose, model registry, env-driven config |

---

## Repository structure

```
jobguard/                    # shared core library (ML + scraping + explainability)
├── config.py                # env-driven settings + .env loader
├── keywords.py              # scam-phrase lexicons → human reasons
├── preprocessing.py         # text cleaning (optional NLTK lemmatization)
├── features.py              # 25 named engineered features + JobPosting type
├── dataset.py               # EMSCAD loader + synthetic fallback generator
├── text_branch.py           # TF-IDF + LogReg text classifier
├── bert_branch.py           # OPTIONAL fine-tuned DistilBERT (same interface)
├── ensemble.py              # stacked ensemble (out-of-fold meta-learner)
├── evaluation.py            # PR-AUC / recall / accuracy metrics
├── explain.py               # SHAP + flagged reasons + phrase saliency
├── llm.py                   # Groq LLM analyst summary (grounded, fails open)
├── scraper.py               # static + JSON-LD + Playwright fallback
├── parser.py                # raw text → structured JobPosting
└── pipeline.py              # InferencePipeline: posting → verdict payload

backend/app/                 # FastAPI service
├── main.py                  # /analyze /feedback /health, CORS, rate limiting
├── schemas.py               # Pydantic request/response models
└── database.py              # SQLAlchemy models (Submission, Feedback)

frontend/                    # React + Vite + Tailwind SPA
├── src/App.jsx              # page: header, analyzer, verdict, footer
└── src/components/          # VerdictCard, ConfidenceGauge, ShapPanel,
                             # HighlightedText, AiAnalystPanel

scripts/                     # train.py, retrain.py, download_data.py, demo.py
data/samples/                # one obvious fake + one genuine posting
models/                      # saved artifacts + registry.jsonl
tests/                       # preprocessing, features, /analyze (mocked scraper)
docker-compose.yml           # db + backend + frontend
```

---

## Quick start

**Prerequisites:** Python 3.11+, Node 18+.

```bash
# 1. Clone
git clone https://github.com/<your-username>/jobguard.git
cd jobguard

# 2. Python deps
python -m pip install -r requirements.txt

# 3. Get the real dataset (or skip — a synthetic fallback runs automatically)
curl -L -o data/fake_job_postings.csv \
  https://raw.githubusercontent.com/Erfaniaa/fake-job-posting-detection/master/dataset.csv

# 4. Train (saves models/ensemble_latest.joblib + a registry entry)
python scripts/train.py --data data/fake_job_postings.csv

# 5. Verify on the seed samples
python scripts/demo.py

# 6. Start the API
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# 7. Start the UI (separate terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Open http://localhost:5173, click **"scam sample"**, and hit **Analyze**.

---

## Training the model

`scripts/train.py` loads data → preprocesses → engineers features → trains and
**compares four models** → saves the best ensemble → prints a full evaluation
report and appends a `models/registry.jsonl` entry.

| Model trained | Description |
|---------------|-------------|
| Baseline | TF-IDF + Logistic Regression (text only) |
| XGBoost (tabular) | 25 engineered features, imbalance-aware |
| Text branch | TF-IDF word+char + LR (or DistilBERT with `--use-bert`) |
| **Ensemble** | stacked meta-learner over the two branches → **promoted** |

**Flags:**

```
--data PATH          EMSCAD/Kaggle CSV (else synthetic fallback)
--synthetic-n N      synthetic corpus size when no CSV
--smote              apply SMOTE to the tabular branch
--use-bert           use the DistilBERT text branch (needs requirements-bert.txt)
--bert-epochs N      DistilBERT fine-tune epochs (default 2)
--bert-max-len N     DistilBERT token length (default 128 in CLI)
--max-rows N         subsample the dataset (handy for CPU BERT runs)
--n-splits N         out-of-fold folds (default 5; 3 with --use-bert)
--out PATH           output bundle path
```

---

## Running the API

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

The trained ensemble is loaded **once at startup** and cached in memory.
Interactive OpenAPI docs are served at **http://localhost:8000/docs**.

---

## Running the frontend

```bash
cd frontend
npm install
npm run dev        # dev server on :5173, proxies /api → :8000
npm run build      # production build to dist/
```

Configure the API base with `frontend/.env` (see `frontend/.env.example`). In dev,
`vite.config.js` proxies `/api` to the backend.

---

## Docker deployment

```bash
python scripts/train.py --data data/fake_job_postings.csv   # ensure a model exists
docker compose up --build
# Frontend → http://localhost:8080   API → http://localhost:8000
```

`docker-compose.yml` runs **Postgres + the FastAPI backend + an nginx-served React
build**. The backend reads `JOBGUARD_DATABASE_URL` (Postgres in compose, SQLite
locally). The `models/` directory is mounted so you can retrain on the host and
restart without rebuilding.

---

## Configuration

All configuration is environment-driven (`jobguard/config.py`), loaded from a
gitignored `.env`. Copy `.env.example` to `.env` and adjust.

| Variable | Default | Purpose |
|----------|---------|---------|
| `JOBGUARD_MODELS_DIR` | `./models` | Where model bundles live |
| `JOBGUARD_MODEL_BUNDLE` | `ensemble_latest.joblib` | Promoted model filename |
| `JOBGUARD_REAL_THRESHOLD` | `0.80` | `p_real` ≥ this → REAL |
| `JOBGUARD_FAKE_THRESHOLD` | `0.50` | `p_real` < this → FAKE |
| `JOBGUARD_DATABASE_URL` | `sqlite:///./jobguard.db` | Feedback DB (Postgres in Docker) |
| `JOBGUARD_CORS_ORIGINS` | `localhost:5173,3000` | Allowed CORS origins |
| `JOBGUARD_RATE_LIMIT` | `30` | Requests/min per IP |
| `JOBGUARD_GROQ_API_KEY` | _(empty)_ | Groq key for Analyst Summary (server-side only) |
| `JOBGUARD_GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model id |
| `JOBGUARD_USE_LLM` | `on` | Toggle the LLM narrative |
| `KMP_DUPLICATE_LIB_OK` | — | Set `TRUE` on Windows when using DistilBERT |

---

## API reference

### `POST /analyze`

Body: `{ "url": "..." }` **or** `{ "text": "..." }` (at least one).

```json
{
  "submission_id": 42,
  "verdict": "FAKE",
  "confidence": 0.99,
  "risk_score": 100,
  "probability_real": 0.001,
  "probability_fake": 0.999,
  "flagged_reasons": ["Asks for bank account details up front", "..."],
  "explanations": {
    "top_suspicious_phrases": ["wire transfer", "registration fee"],
    "shap_top_features": [
      {"feature": "salary_too_high_flag", "label": "Salary looks too good to be true",
       "impact": 0.34, "direction": "raises fraud risk"}
    ]
  },
  "advice": "Treat this posting with strong caution. Never send money ...",
  "branch_scores": {"tabular_model_fraud_prob": 0.99, "text_model_fraud_prob": 0.99},
  "scrape_method": "static+jsonld",
  "parsed": {"title": "...", "company": "...", "salary_range": "..."},
  "ai_explanation": {"summary": "The system flagged ...", "model": "openai/gpt-oss-20b"}
}
```

Scraping failures (paywall, JS-gated, 404, bot-block) return **HTTP 422** with a
message asking the user to paste the text instead.

### `POST /feedback`

`{ "submission_id": 42, "correct_label": "fake" | "real" }` — records a user
correction for retraining.

### `GET /health`

Returns model-loaded status and the promoted model's metadata.

---

## Explainability

Explanations are surfaced in plain English, never as raw feature indices:

- **SHAP** (`TreeExplainer` on the XGBoost tabular model) shows which named
  features pushed *this* posting toward fraud or genuineness. Falls back to a
  global-importance heuristic if `shap` is not installed.
- **Text saliency** — the linear text model's coefficients weighted by the
  document's TF-IDF values highlight the exact phrases driving the prediction
  (what LIME approximates for linear models). The DistilBERT branch uses
  token-masking saliency instead.
- **Deterministic rule flags** — a high-precision layer that always fires on known
  scam patterns (fee requests, wire transfers, identity harvesting, free-email
  contacts, salary anomalies), independent of the ML model.

---

## AI analyst summary (Groq)

Each `/analyze` response can include a natural-language explanation generated by a
Groq-hosted LLM (`jobguard/llm.py`). The LLM only **narrates** the ML verdict — it
is fed the model's structured signals and **forbidden by prompt** from changing
the verdict or inventing facts. It **fails open**: with no key or an unreachable
API, `ai_explanation` is simply `null` and every deterministic field remains.

```bash
# .env (gitignored, server-side only — never exposed to the browser):
JOBGUARD_GROQ_API_KEY=gsk_...        # from https://console.groq.com/keys
JOBGUARD_GROQ_MODEL=openai/gpt-oss-20b
```

---

## Fine-tuned DistilBERT branch

The DistilBERT branch is a real, drop-in text classifier behind the same interface
as the TF-IDF branch:

```bash
pip install -r requirements-bert.txt      # torch + transformers (~2 GB)
python scripts/train.py --use-bert --max-rows 2000 --bert-epochs 2 \
        --out models/ensemble_bert.joblib
# Serve it:
JOBGUARD_MODEL_BUNDLE=ensemble_bert.joblib uvicorn backend.app.main:app
```

Trade-off: DistilBERT is data-hungry and slow on CPU; on small subsamples it does
not beat TF-IDF, but on the full dataset with a GPU it typically lifts recall.
The default TF-IDF branch needs none of the heavy stack.

---

## Feedback loop & continuous learning

Every submission and every user correction is persisted (SQLite by default,
Postgres in Docker). `scripts/retrain.py`:

1. loads EMSCAD + accumulated feedback,
2. retrains the ensemble,
3. evaluates PR-AUC on a held-out split,
4. **promotes the new model only if its PR-AUC ≥ the current production model's**
   (`--force` to override), and
5. appends a model-registry entry (`models/registry.jsonl`) with metrics +
   timestamp.

---

## Testing

```bash
python -m pytest
```

Covers preprocessing, feature extraction / parsing, and the `/analyze`,
`/feedback`, `/health` endpoints (the scraper is mocked and the LLM disabled, so
tests need no network and hit no external APIs). Endpoint tests skip automatically
if no trained model bundle is present.

---

## Windows notes

`torch`, `xgboost`, and `scikit-learn` each bundle an OpenMP runtime. On Windows,
importing `torch` **after** the others aborts with `WinError 1114 — c10.dll
initialization failed`. JobGuard handles this by (a) preloading `torch` first in
`train.py` (with `--use-bert`) and in the API when a BERT bundle is configured, and
(b) setting `KMP_DUPLICATE_LIB_OK=TRUE`. If you hit the error running a custom
script, set that env var and import `torch` before `sklearn`/`xgboost`.

---

## Limitations & ethical considerations

- **Probabilistic, not definitive.** JobGuard outputs a probability and always
  shows its confidence and reasoning. The UI advises independent verification and
  never tells a user something is guaranteed safe.
- **Scraping is fragile.** Many job sites (LinkedIn, Indeed) block bots or require
  login. The "paste the text" fallback is the reliable path.
- **Trained on English EMSCAD.** Performance on other languages, regions, or
  posting styles not represented in EMSCAD will differ; retrain on representative
  data for production use.
- **Privacy.** Only what the feedback loop needs is stored — the posting text and
  the verdict/correction. No personal user identifiers are collected.
- **WHOIS domain age** uses the free `python-whois` (rate-limited, inconsistent
  across TLDs); a paid API is recommended for production accuracy.
- **The LLM narrates, it does not decide.** The ML ensemble makes the verdict; the
  LLM only explains it and is prompt-constrained against overriding it.

---

## Roadmap

- [ ] Train DistilBERT on full EMSCAD with a GPU and stack it into production
- [ ] Company-registry / domain-age enrichment via a paid API
- [ ] Batch analysis and a browser extension
- [ ] Active-learning prioritization of feedback for retraining
- [ ] Multilingual support

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Acknowledgements

- **EMSCAD** — the Employment Scam Aegean Dataset (University of the Aegean),
  distributed on Kaggle as *Real / Fake Job Posting Prediction*.
- **scikit-learn, XGBoost, imbalanced-learn, SHAP, HuggingFace Transformers**, and
  **FastAPI** — the open-source shoulders this project stands on.

> ⚠️ **Disclaimer:** JobGuard is a decision-support tool, not a guarantee. Always
> verify an employer independently, and never send money or personal financial
> information to apply for a job.
