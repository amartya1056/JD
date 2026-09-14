# 🛡️ JobGuard — Fake Job Posting Detector

An end-to-end, explainable ML platform that decides whether a job posting is
**REAL**, **FAKE**, or **SUSPICIOUS**. Paste a URL or the raw text; JobGuard
scrapes/parses it, extracts three parallel signal categories (NLP text, salary
anomalies, company/metadata), runs a **stacked ensemble** (XGBoost on engineered
features + a text branch), and returns a verdict with a confidence score,
plain-English flagged reasons, SHAP feature attributions, and highlighted
suspicious phrases — plus a feedback loop for continuous retraining.

> ⚠️ **Ethical note:** JobGuard produces a *probabilistic* estimate, never legal
> certainty. It always shows confidence and advises independent verification, and
> stores only the minimum needed for the feedback loop (no personal identifiers).

---

## Architecture

```
User input (URL / pasted text)
  → Scraper/Parser        (jobguard/scraper.py, jobguard/parser.py)
  → Preprocessing         (jobguard/preprocessing.py: clean, tokenize, lemmatize)
  → Feature extraction    (3 branches)
       (a) Text NLP        TF-IDF word+char n-grams (+ optional DistilBERT)
       (b) Salary signals  range parsing, "too good to be true" flags
       (c) Metadata        logo, profile, email domain mismatch, free email, ...
  → Ensemble              XGBoost (tabular) + text branch → LR meta-learner
  → Verdict thresholding   p_real ≥ .80 REAL · < .50 FAKE · else SUSPICIOUS
  → Explainability         SHAP (tabular) + phrase saliency (text)
  → Verdict display        score, label, reasons, advice   (React UI)
  → Feedback loop          Postgres/SQLite → retrain.py (guarded promotion)
```

Everything the training pipeline and the API share lives in one installable
package, **`jobguard/`**, guaranteeing train/serve parity.

```
├── jobguard/            # shared core library (ML + scraping + explainability)
│   ├── config.py        # env-driven settings singleton
│   ├── keywords.py      # scam-phrase lexicons → human reasons
│   ├── preprocessing.py # text cleaning (optional NLTK lemmatization)
│   ├── features.py      # 25 named engineered tabular features + JobPosting type
│   ├── text_branch.py   # TF-IDF + LogisticRegression text classifier
│   ├── bert_branch.py   # OPTIONAL fine-tuned DistilBERT (same interface)
│   ├── ensemble.py      # stacked ensemble (OOF meta-learner)
│   ├── evaluation.py    # PR-AUC / recall-focused metrics
│   ├── explain.py       # SHAP + flagged reasons + phrase saliency
│   ├── scraper.py       # static + JSON-LD + Playwright fallback
│   ├── parser.py        # raw-text → structured JobPosting
│   └── pipeline.py      # InferencePipeline: posting → verdict payload
├── backend/app/         # FastAPI: /analyze /feedback /health
├── frontend/            # React + Vite + Tailwind SPA
├── scripts/             # train.py, retrain.py, download_data.py, demo.py
├── data/samples/        # one obvious fake + one genuine posting
├── models/              # saved artifacts + registry.jsonl
├── tests/               # preprocessing, features, /analyze (mocked scraper)
├── requirements.txt     # core deps   (requirements-bert.txt = optional BERT)
└── docker-compose.yml   # db + backend + frontend
```

---

## Quick start (local, no Docker)

Requires **Python 3.11+** and **Node 18+**.

```bash
# 1. Install the Python stack
python -m pip install -r requirements.txt

# 2. Train a model (uses the synthetic fallback dataset out of the box)
python scripts/train.py

# 3. (verify) run inference on the seed samples
python scripts/demo.py

# 4. Start the API (loads the model once at startup)
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# 5. In another terminal, start the UI
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api → :8000)
```

Open http://localhost:5173, click **Try a sample scam**, and hit **Analyze**.

### Optional: better lemmatization

```bash
python -c "from jobguard.preprocessing import ensure_nltk; print(ensure_nltk())"
```

Without this, preprocessing uses a lightweight built-in stopword list (still fully
functional).

---

## Using the real EMSCAD dataset

The synthetic fallback lets everything run immediately, but the real benchmark is
**EMSCAD** (the Kaggle *Real or Fake Job Posting* set, ~18k rows, ~4% fraud).

```bash
# Option A: Kaggle API (needs ~/.kaggle/kaggle.json)
python scripts/download_data.py

# Option B: manually download fake_job_postings.csv into ./data/ then:
python scripts/train.py --data data/fake_job_postings.csv
```

Synthetic-data metrics are **optimistic** (the generator's fraud signals are
cleaner than reality). On real EMSCAD expect roughly **PR-AUC 0.85–0.95** and
FAKE-class recall **0.6–0.85**, depending on the operating threshold.

---

## Training & evaluation

`python scripts/train.py` trains and compares four models, then saves the best
ensemble + a registry entry:

| Model | What it is |
|-------|------------|
| Baseline | TF-IDF + Logistic Regression (text only) |
| XGBoost (tabular) | 25 engineered features, class-imbalance aware |
| Text branch | TF-IDF word+char n-grams + LR (or DistilBERT with `--use-bert`) |
| **Ensemble** | stacked meta-learner over the two branches → **promoted** |

Flags: `--smote` (SMOTE on the tabular branch), `--use-bert` (DistilBERT text
branch, needs `requirements-bert.txt`), `--data PATH`, `--synthetic-n N`,
`--max-rows N` (subsample), `--bert-epochs`, `--bert-max-len`, `--n-splits`.

### Fine-tuned DistilBERT branch (real)

```bash
pip install -r requirements-bert.txt      # torch + transformers (~2 GB)
# Train a DistilBERT ensemble to a SEPARATE bundle (keeps TF-IDF as production):
python scripts/train.py --use-bert --max-rows 900 --bert-epochs 2 \
        --bert-max-len 128 --out models/ensemble_bert.joblib
```

The DistilBERT branch (`jobguard/bert_branch.py`) implements the *same* interface
as the TF-IDF branch, so the stacked ensemble is unchanged — it just fine-tunes
DistilBERT for the text signal and stacks it with XGBoost. Saliency is computed
by token-masking (model-faithful, no extra deps).

To **serve** the BERT model, point the API at it:
```bash
JOBGUARD_MODEL_BUNDLE=ensemble_bert.joblib uvicorn backend.app.main:app
```

> **Windows note (important):** torch's `c10.dll` fails to initialize (WinError
> 1114) if imported *after* scikit-learn/xgboost, due to conflicting bundled
> OpenMP runtimes. The code preloads torch first (in `train.py`, and in the API
> when a BERT bundle is configured) and sets `KMP_DUPLICATE_LIB_OK=TRUE`. On CPU,
> fine-tuning is slow — subsample with `--max-rows` for quick runs; use a GPU and
> the full dataset for real training. On this small synthetic set, DistilBERT is
> competitive with but does not beat TF-IDF (transformers need more data); on real
> EMSCAD the transformer branch typically lifts recall.

### AI "Analyst Summary" (Groq)

Every `/analyze` response can include a natural-language explanation generated by
a Groq-hosted LLM (`jobguard/llm.py`). The LLM only **narrates** the ML verdict —
it is fed the model's structured signals and forbidden by prompt from changing the
verdict or inventing facts. It **fails open**: if the key is missing or Groq is
unreachable, `ai_explanation` is simply `null` and all deterministic fields remain.

Setup (server-side only, never exposed to the browser):
```bash
# In .env (gitignored):
JOBGUARD_GROQ_API_KEY=gsk_...          # from https://console.groq.com/keys
JOBGUARD_GROQ_MODEL=openai/gpt-oss-20b
```

**Why these metrics:** on a 4%-fraud problem, accuracy is meaningless (predict
"REAL" always → 96% accuracy, 0 scams caught). The pipeline reports **Precision,
Recall, F1, PR-AUC, ROC-AUC, and a confusion matrix**, and optimizes for high
FAKE-class recall while keeping precision reasonable (missing a scam costs more
than a false alarm).

---

## API

`POST /analyze` — body `{ "url": "..." }` **or** `{ "text": "..." }`

```json
{
  "verdict": "FAKE",
  "confidence": 0.99,
  "risk_score": 100,
  "probability_real": 0.001,
  "probability_fake": 0.999,
  "flagged_reasons": ["Asks for bank account details up front", "..."],
  "explanations": {
    "top_suspicious_phrases": ["wire transfer", "registration fee"],
    "shap_top_features": [{"label": "Salary looks too good to be true",
                            "direction": "raises fraud risk", "impact": 0.34}]
  },
  "advice": "Treat this posting with strong caution. Never send money ...",
  "branch_scores": {"tabular_model_fraud_prob": 0.99, "text_model_fraud_prob": 0.99},
  "ai_explanation": {"summary": "The system flagged this posting as almost "
                     "certainly a scam ...", "model": "openai/gpt-oss-20b"}
}
```

`POST /feedback` — `{ "submission_id": 1, "correct_label": "fake" | "real" }`
`GET  /health`   — model status + metadata. Interactive docs at `/docs`.

Scraping failures (paywall / JS / 404 / bot-block) return **422** with a message
telling the user to paste the text instead. CORS, rate limiting (slowapi), and
pydantic validation are all wired in.

---

## Feedback loop & continuous learning

Every submission and every user correction is persisted (SQLite by default,
Postgres in Docker). `scripts/retrain.py`:

1. loads the base dataset + accumulated feedback,
2. retrains the ensemble,
3. evaluates PR-AUC on a held-out split,
4. **promotes the new model only if PR-AUC ≥ the current production model's**
   (use `--force` to override), and
5. appends a registry entry (`models/registry.jsonl`) with metrics + timestamp.

---

## Docker

```bash
python scripts/train.py            # ensure models/ensemble_latest.joblib exists
docker compose up --build
# Frontend → http://localhost:8080   API → http://localhost:8000
```

Compose runs Postgres + the FastAPI backend + an nginx-served React build.

---

## Tests

```bash
python -m pytest
```

Covers preprocessing, feature extraction / parsing, and the `/analyze`,
`/feedback`, `/health` endpoints (the scraper is mocked, so tests need no
network). The endpoint tests skip automatically if no model bundle is present.

---

## Implementation trade-offs (called out honestly)

- **DistilBERT is optional.** The default text branch is TF-IDF + LR — fast and
  dependency-light. Install `requirements-bert.txt` and pass `--use-bert` for the
  transformer branch. Both share one interface (`jobguard/text_branch.py`).
- **WHOIS domain age** uses the free `python-whois` (rate-limited, inconsistent
  across TLDs). A paid API would be more reliable; the code fails silent without it.
- **JS-rendered pages** need Playwright (`pip install playwright && playwright
  install chromium`). Without it, JS-gated pages fall back to "paste the text".
- **Synthetic data** is a stand-in for EMSCAD so the repo runs on checkout; its
  metrics are optimistic. Use the real dataset for meaningful numbers.
```
