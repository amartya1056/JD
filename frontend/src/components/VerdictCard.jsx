import { useState } from "react";
import ConfidenceGauge from "./ConfidenceGauge.jsx";
import HighlightedText from "./HighlightedText.jsx";
import ShapPanel from "./ShapPanel.jsx";
import AiAnalystPanel from "./AiAnalystPanel.jsx";
import { sendFeedback } from "../api.js";

const THEME = {
  REAL: { color: "var(--real)", wash: "var(--real-wash)", label: "Likely Genuine", tag: "LOW RISK" },
  FAKE: { color: "var(--fake)", wash: "var(--fake-wash)", label: "Likely Fraudulent", tag: "HIGH RISK" },
  SUSPICIOUS: { color: "var(--amber)", wash: "var(--amber-wash)", label: "Proceed With Caution", tag: "ELEVATED RISK" },
};

function StatChip({ label, value }) {
  return (
    <div className="rounded-xl border border-line bg-paper-2 px-3.5 py-2.5">
      <div className="text-[10px] uppercase tracking-widest text-muted">{label}</div>
      <div className="mt-0.5 font-display text-lg tnum text-ink">{value}</div>
    </div>
  );
}

export default function VerdictCard({ result, sourceText }) {
  const [feedbackState, setFeedbackState] = useState("idle");
  const t = THEME[result.verdict] || THEME.SUSPICIOUS;

  async function report(label) {
    setFeedbackState("sending");
    try {
      await sendFeedback({ submissionId: result.submission_id, correctLabel: label });
      setFeedbackState("done");
    } catch {
      setFeedbackState("error");
    }
  }

  return (
    <div className="animate-rise space-y-5">
      {/* ── Verdict header ─────────────────────────────────────── */}
      <div
        className="relative overflow-hidden rounded-2xl border bg-card shadow-md"
        style={{ borderColor: t.color + "55" }}
      >
        <div className="absolute inset-y-0 left-0 w-1.5" style={{ background: t.color }} />
        <div className="grid gap-6 p-6 sm:p-8 md:grid-cols-[1fr_auto] md:items-center">
          <div>
            <span
              className="inline-flex items-center gap-2 rounded-full px-3 py-1 text-[11px] font-bold tracking-widest"
              style={{ background: t.wash, color: t.color }}
            >
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: t.color }} />
              {t.tag}
            </span>
            <h2 className="mt-3 font-display text-4xl leading-none tracking-tightish text-ink">
              {t.label}
            </h2>
            <p className="mt-3 max-w-md text-sm text-ink-soft">{result.advice}</p>

            <div className="mt-5 grid grid-cols-3 gap-2.5">
              <StatChip label="Confidence" value={`${Math.round(result.confidence * 100)}%`} />
              <StatChip label="P(genuine)" value={`${Math.round(result.probability_real * 100)}%`} />
              <StatChip label="Verdict" value={result.verdict} />
            </div>
          </div>
          <ConfidenceGauge riskScore={result.risk_score} color={t.color} />
        </div>
      </div>

      {/* ── AI analyst summary (Groq) ──────────────────────────── */}
      <AiAnalystPanel explanation={result.ai_explanation} />

      {/* ── Flagged reasons ────────────────────────────────────── */}
      {result.flagged_reasons?.length > 0 && (
        <div className="rounded-2xl border border-line bg-card p-6 shadow-sm sm:p-7">
          <h3 className="mb-4 font-display text-lg text-ink">
            Flagged signals
            <span className="ml-2 text-sm font-normal text-muted">
              ({result.flagged_reasons.length})
            </span>
          </h3>
          <ul className="grid gap-x-8 gap-y-2.5 sm:grid-cols-2">
            {result.flagged_reasons.map((r, i) => (
              <li key={i} className="flex items-start gap-2.5 text-[13.5px] text-ink-soft">
                <svg width="16" height="16" viewBox="0 0 24 24" className="mt-0.5 shrink-0"
                     style={{ color: t.color }} fill="none" aria-hidden="true">
                  <path d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"
                        stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span>{r}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Decision factors + suspicious phrases ──────────────── */}
      <div className="grid gap-5 lg:grid-cols-2">
        <div className="rounded-2xl border border-line bg-card p-6 shadow-sm sm:p-7">
          <ShapPanel features={result.explanations?.shap_top_features} />
        </div>
        {sourceText && (
          <div className="rounded-2xl border border-line bg-card p-6 shadow-sm sm:p-7">
            <h3 className="mb-1 font-display text-lg text-ink">In the listing</h3>
            <p className="mb-4 text-xs text-muted">Phrases the language model weighted</p>
            <div className="scroll-fine max-h-64 overflow-y-auto rounded-xl border border-line bg-paper-2 p-4">
              <HighlightedText text={sourceText} phrases={result.explanations?.top_suspicious_phrases} />
            </div>
          </div>
        )}
      </div>

      {/* ── Model transparency + feedback ──────────────────────── */}
      <div className="rounded-2xl border border-line bg-card p-6 shadow-sm sm:p-7">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-muted">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: "var(--ink)" }} />
            Metadata model {Math.round((result.branch_scores?.tabular_model_fraud_prob ?? 0) * 100)}% fraud
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: "var(--gold)" }} />
            Language model {Math.round((result.branch_scores?.text_model_fraud_prob ?? 0) * 100)}% fraud
          </span>
          {result.scrape_method && <span>Source · {result.scrape_method}</span>}
        </div>

        <div className="mt-5 border-t border-line pt-5">
          {feedbackState === "done" ? (
            <p className="text-sm text-real">✓ Thank you — your correction was recorded for retraining.</p>
          ) : (
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-sm text-ink-soft">Disagree with this verdict?</span>
              <button
                onClick={() => report("real")}
                disabled={feedbackState === "sending"}
                className="rounded-lg border border-line px-3.5 py-1.5 text-sm text-ink transition hover:border-real hover:text-real disabled:opacity-50"
              >
                It's genuine
              </button>
              <button
                onClick={() => report("fake")}
                disabled={feedbackState === "sending"}
                className="rounded-lg border border-line px-3.5 py-1.5 text-sm text-ink transition hover:border-fake hover:text-fake disabled:opacity-50"
              >
                It's fake
              </button>
              {feedbackState === "error" && <span className="text-sm text-fake">Couldn't send — retry.</span>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
