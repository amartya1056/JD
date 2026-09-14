import { useState } from "react";
import { analyze } from "./api.js";
import VerdictCard from "./components/VerdictCard.jsx";

const FAKE_SAMPLE =
  "URGENT!! Work from home and earn $5,000/week — NO EXPERIENCE NEEDED!!! To start today, pay a one-time registration fee and send your bank account details. You will process payments and receive and forward packages. Commission paid by wire transfer via Western Union. Contact our recruiter on Telegram.";

const REAL_SAMPLE =
  "Senior Software Engineer — Backend Platform\nCompany: Northwind Traders\nLocation: Austin, TX (Hybrid)\nSalary: $130,000 - $165,000\nEmployment type: Full-time\n\nThe role:\nYou will design, build and maintain scalable backend services powering our supply-chain platform, collaborate with product and data teams, and mentor junior engineers.\n\nRequirements:\nBachelor's degree in Computer Science and 5+ years building production backend systems (Python, Go or Java). Experience with distributed systems and cloud infrastructure.\n\nBenefits:\nHealth, dental and vision insurance. 401(k) matching. Paid parental leave.";

function ShieldMark({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <path d="M16 2.5 27 6v8.5c0 7-4.7 12.4-11 15-6.3-2.6-11-8-11-15V6l11-3.5z"
            fill="var(--ink)" />
      <path d="M16 2.5 27 6v8.5c0 7-4.7 12.4-11 15-6.3-2.6-11-8-11-15V6l11-3.5z"
            stroke="var(--gold)" strokeWidth="0.8" />
      <path d="m11 16 3.4 3.4L21 12.8" stroke="var(--gold-soft)" strokeWidth="2.2"
            strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

const SIGNALS = [
  {
    t: "Language analysis",
    d: "TF-IDF and a fine-tuned transformer read the wording for scam patterns, urgency and pressure tactics.",
    icon: "M4 7h16M4 12h10M4 17h7",
  },
  {
    t: "Compensation signals",
    d: "Salary ranges are parsed and checked for anomalies — the 'too good to be true' pay that flags fraud.",
    icon: "M12 3v18M8 7h6a3 3 0 0 1 0 6H8m0 0h8",
  },
  {
    t: "Metadata & identity",
    d: "Company profile, logo, email domain and contact channels are cross-checked for the tells of a fake employer.",
    icon: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 20a8 8 0 0 1 16 0",
  },
];

function ScanningLoader() {
  return (
    <div className="animate-fade rounded-2xl border border-line bg-card p-10 shadow-md">
      <div className="mx-auto flex max-w-sm flex-col items-center text-center">
        <div className="relative h-20 w-20 overflow-hidden rounded-2xl border border-line bg-paper-2">
          <div className="flex h-full items-center justify-center">
            <ShieldMark size={40} />
          </div>
          <div
            className="absolute inset-x-0 h-8"
            style={{
              background: "linear-gradient(var(--gold-soft), transparent)",
              opacity: 0.5,
              animation: "scanline 1.4s ease-in-out infinite",
            }}
          />
        </div>
        <p className="mt-5 font-display text-lg text-ink">Analyzing the posting</p>
        <p className="mt-1 text-sm text-muted">
          Extracting signals · running the ensemble · drafting the explanation
        </p>
      </div>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [analyzedText, setAnalyzedText] = useState("");

  async function onAnalyze(e) {
    e.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const data = await analyze(mode === "url" ? { url } : { text });
      setResult(data);
      setAnalyzedText(mode === "url" ? "" : text);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const canSubmit = mode === "url" ? url.trim().length > 4 : text.trim().length >= 20;

  return (
    <div className="min-h-screen">
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="sticky top-0 z-20 border-b border-line/70 bg-paper/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-5 py-3.5">
          <div className="flex items-center gap-2.5">
            <ShieldMark />
            <div className="leading-tight">
              <div className="font-display text-lg tracking-tightish text-ink">JobGuard</div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-muted">
                Employment Fraud Intelligence
              </div>
            </div>
          </div>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="hidden text-sm text-ink-soft underline-offset-4 hover:text-ink hover:underline sm:block"
          >
            API reference
          </a>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-5 pb-24">
        {/* ── Hero ──────────────────────────────────────────────── */}
        <section className="pt-14 text-center sm:pt-20">
          <span className="inline-flex items-center gap-2 rounded-full border border-line bg-paper-2 px-3.5 py-1 text-xs tracking-wide text-ink-soft">
            <span className="h-1.5 w-1.5 rounded-full bg-real" />
            Ensemble ML · SHAP explainability · AI summaries
          </span>
          <h1 className="mx-auto mt-6 max-w-3xl font-display text-5xl font-extrabold leading-[1.03] tracking-tight text-ink sm:text-6xl">
            Know if a job offer is
            <span className="relative mx-2 inline-block text-gold">
              real
              <svg className="absolute -bottom-2 left-0 w-full" height="8" viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true">
                <path d="M1 5 Q 25 1 50 4 T 99 3" stroke="var(--gold-soft)" strokeWidth="2" fill="none" strokeLinecap="round" />
              </svg>
            </span>
            before you apply.
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-[15px] leading-relaxed text-ink-soft">
            Paste a job posting URL or its text. JobGuard scores it for fraud with an
            explainable model ensemble and tells you, in plain language, exactly why.
          </p>
        </section>

        {/* ── Analyzer ──────────────────────────────────────────── */}
        <section className="mx-auto mt-10 max-w-2xl">
          <form onSubmit={onAnalyze} className="rounded-2xl border border-line bg-card p-4 shadow-md sm:p-5">
            <div className="mb-3 flex items-center justify-between">
              <div className="inline-flex rounded-lg border border-line bg-paper-2 p-1" role="tablist">
                {[["text", "Paste text"], ["url", "Job URL"]].map(([m, label]) => (
                  <button
                    key={m}
                    type="button"
                    role="tab"
                    aria-selected={mode === m}
                    onClick={() => setMode(m)}
                    className={`rounded-md px-4 py-1.5 text-sm font-medium transition ${
                      mode === m ? "bg-ink text-paper shadow-sm" : "text-muted hover:text-ink"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {mode === "text" && (
                <div className="flex gap-3 text-xs">
                  <button type="button" onClick={() => setText(FAKE_SAMPLE)}
                          className="text-muted underline-offset-2 hover:text-fake hover:underline">
                    scam sample
                  </button>
                  <button type="button" onClick={() => setText(REAL_SAMPLE)}
                          className="text-muted underline-offset-2 hover:text-real hover:underline">
                    genuine sample
                  </button>
                </div>
              )}
            </div>

            {mode === "url" ? (
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://careers.example.com/job/12345"
                className="w-full rounded-xl border border-line bg-paper-2 px-4 py-3.5 text-sm text-ink placeholder:text-muted focus:border-gold focus:outline-none"
                aria-label="Job posting URL"
              />
            ) : (
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={7}
                placeholder="Paste the full job posting here…"
                className="w-full resize-y rounded-xl border border-line bg-paper-2 px-4 py-3.5 text-sm leading-relaxed text-ink placeholder:text-muted focus:border-gold focus:outline-none"
                aria-label="Job posting text"
              />
            )}

            <div className="mt-3 flex items-center justify-between gap-3">
              <span className="text-xs text-muted">
                {mode === "url" ? "Blocked by the site? Paste the text instead." : "We store only what's needed to improve the model."}
              </span>
              <button
                type="submit"
                disabled={!canSubmit || loading}
                className="group inline-flex items-center gap-2 rounded-xl bg-ink px-6 py-3 text-sm font-semibold text-paper shadow-sm transition hover:shadow-md disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? "Analyzing…" : "Analyze posting"}
                {!loading && (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                       className="transition group-hover:translate-x-0.5">
                    <path d="M5 12h14m-6-6 6 6-6 6" stroke="currentColor" strokeWidth="2"
                          strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
              </button>
            </div>
          </form>

          {error && (
            <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-fake/30 bg-fake-wash px-4 py-3 text-sm text-fake">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="mt-0.5 shrink-0">
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.8" />
                <path d="M12 8v4m0 4h.01" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
              </svg>
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* ── Result / loading / trust strip ────────────────────── */}
        <section className="mx-auto mt-8 max-w-3xl">
          {loading && <ScanningLoader />}

          {result && !loading && <VerdictCard result={result} sourceText={analyzedText} />}

          {!result && !loading && (
            <div className="mt-8 animate-fade">
              <p className="mb-5 text-center text-xs uppercase tracking-[0.2em] text-muted">
                Three signal categories, one verdict
              </p>
              <div className="grid gap-4 sm:grid-cols-3">
                {SIGNALS.map((s) => (
                  <div key={s.t} className="rounded-2xl border border-line bg-card p-5 shadow-sm">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-gold-wash text-gold">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                        <path d={s.icon} stroke="currentColor" strokeWidth="1.8"
                              strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </span>
                    <h3 className="mt-3.5 font-display text-lg text-ink">{s.t}</h3>
                    <p className="mt-1.5 text-[13px] leading-relaxed text-ink-soft">{s.d}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </main>

      {/* ── Footer ──────────────────────────────────────────────── */}
      <footer className="border-t border-line/70 bg-paper-2/60">
        <div className="mx-auto max-w-5xl px-5 py-8 text-center">
          <p className="mx-auto max-w-2xl text-xs leading-relaxed text-muted">
            JobGuard produces a <span className="text-ink-soft">probabilistic estimate, not legal
            certainty</span>. It always shows its confidence and reasoning. Verify any employer
            independently, and never send money or personal financial information to apply.
          </p>
        </div>
      </footer>
    </div>
  );
}
