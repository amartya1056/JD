// Plain-English rendering of the SHAP tabular attribution. Diverging bars grow
// from a centre baseline — right/red raises fraud risk, left/green lowers it —
// so the direction of each factor is legible at a glance.
export default function ShapPanel({ features }) {
  if (!features || features.length === 0) return null;
  const maxImpact = Math.max(...features.map((f) => Math.abs(f.impact)), 1e-6);

  return (
    <div>
      <h3 className="mb-1 font-display text-lg text-ink">Decision factors</h3>
      <p className="mb-4 text-xs text-muted">
        How each measured attribute pushed the assessment
      </p>
      <ul className="space-y-3">
        {features.map((f) => {
          const pct = Math.min(50, (Math.abs(f.impact) / maxImpact) * 50);
          const raises = f.direction.includes("raises");
          return (
            <li key={f.feature}>
              <div className="mb-1 flex items-center justify-between text-[13px]">
                <span className="text-ink-soft">{f.label}</span>
                <span
                  className="text-[11px] font-semibold tracking-wide"
                  style={{ color: raises ? "var(--fake)" : "var(--real)" }}
                >
                  {raises ? "RAISES RISK" : "LOWERS RISK"}
                </span>
              </div>
              {/* Diverging bar around a centre line */}
              <div className="relative h-2.5 w-full rounded-full bg-line/70">
                <div className="absolute left-1/2 top-0 h-full w-px bg-ink/15" />
                <div
                  className="absolute top-0 h-full rounded-full"
                  style={{
                    width: `${pct}%`,
                    left: raises ? "50%" : `${50 - pct}%`,
                    background: raises ? "var(--fake)" : "var(--real)",
                    transition: "width 0.7s cubic-bezier(0.16,1,0.3,1)",
                  }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
