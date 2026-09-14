# Deploying JobGuard

JobGuard has two halves that deploy to two different places:

| Half | Best platform | Why |
|------|---------------|-----|
| **React frontend** (`frontend/`) | **Vercel** | Static Vite build — Vercel's sweet spot |
| **Python ML backend** (`backend/`) | **Render / Railway / Fly.io** (Docker) | XGBoost + scikit-learn + SHAP unzip to ~250 MB and load a model at startup — too heavy for Vercel serverless functions |

> **Do not try to deploy the Python backend on Vercel serverless.** The ML
> dependencies exceed Vercel's 250 MB function limit and cold-start model loading
> is slow. Use a container host for the API; use Vercel for the UI.

---

## 1. Deploy the frontend to Vercel

In the Vercel "New Project" flow, import your GitHub repo, then set:

| Setting | Value |
|---------|-------|
| **Root Directory** | `frontend`  ← **this is the key step** |
| **Framework Preset** | `Vite` (auto-detected once Root Directory is `frontend`) |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |
| **Install Command** | `npm install` |

> The screenshot showing `pip install -r requirements.txt` means Vercel is looking
> at the **repo root** (a Python project). Set **Root Directory = `frontend`** and
> those boxes flip to the npm values above (a `frontend/vercel.json` is included so
> they're pre-filled and SPA routing works).

**Environment variable** (Project → Settings → Environment Variables):

| Name | Value |
|------|-------|
| `VITE_API_BASE` | `https://<your-backend>.onrender.com` (the API URL from step 2) |

Vite bakes this in at build time, so **redeploy the frontend after you set it**.
Without it, the app calls `/api` on the Vercel domain, which has no backend.

---

## 2. Deploy the backend to Render (Docker)

The repo includes `render.yaml` and ships the trained model
(`models/ensemble_latest.joblib`, ~3 MB), so the API works on first deploy.

1. Push the repo to GitHub (make sure `models/ensemble_latest.joblib` is committed
   — the `.gitignore` already whitelists it: `git add -f models/ensemble_latest.joblib`).
2. In Render: **New + → Blueprint** → select your repo. It builds
   `backend/Dockerfile`.
3. Set environment variables in the Render dashboard:
   - `JOBGUARD_CORS_ORIGINS` = your Vercel URL, e.g. `https://jobguard.vercel.app`
   - `JOBGUARD_GROQ_API_KEY` = your Groq key (leave the Analyst Summary off by
     omitting it)
4. Copy the resulting URL (e.g. `https://jobguard-api.onrender.com`) and put it in
   the frontend's `VITE_API_BASE`, then redeploy the frontend.

**Alternatives:** the same `backend/Dockerfile` deploys unchanged to **Railway**
(New → Deploy from Repo → Dockerfile) or **Fly.io** (`fly launch`). All inject
`$PORT`, which the Dockerfile honors.

**Durable feedback (optional):** on the free tier SQLite resets on redeploy. Add a
managed Postgres and set `JOBGUARD_DATABASE_URL` to
`postgresql+psycopg://user:pass@host:5432/db`.

---

## 3. Wire the two together (checklist)

- [ ] Backend deployed; `GET https://<backend>/health` returns `{"status":"ok"}`.
- [ ] `VITE_API_BASE` on Vercel = the backend URL; frontend redeployed.
- [ ] `JOBGUARD_CORS_ORIGINS` on the backend = the Vercel URL (exact origin, no
      trailing slash).
- [ ] Open the Vercel URL, click **scam sample**, **Analyze** → verdict renders.

---

## Full-stack alternative: one box with Docker Compose

If you'd rather run everything on a single VM (DigitalOcean, EC2, etc.):

```bash
docker compose up --build
# Frontend → :8080   API → :8000   (Postgres included)
```

This is the simplest way to get the whole system (frontend + API + Postgres)
running together, and avoids the two-platform wiring entirely.
