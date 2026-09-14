// Elegant semicircular risk gauge. Pure SVG, tabular numerals in the display
// serif. The arc is verdict-tinted; a faint tick scale adds an instrument feel.
export default function ConfidenceGauge({ riskScore, color }) {
  const clamped = Math.max(0, Math.min(100, riskScore));
  const R = 78;
  const CIRC = Math.PI * R;
  const offset = CIRC * (1 - clamped / 100);

  // Minor tick marks around the arc for a measured, instrument look.
  const ticks = Array.from({ length: 11 }, (_, i) => {
    const a = Math.PI - (i / 10) * Math.PI; // 180° → 0°
    const x1 = 100 + Math.cos(a) * (R + 10);
    const y1 = 100 - Math.sin(a) * (R + 10);
    const x2 = 100 + Math.cos(a) * (R + 16);
    const y2 = 100 - Math.sin(a) * (R + 16);
    return { x1, y1, x2, y2, key: i };
  });

  return (
    <div className="flex flex-col items-center" aria-label={`Risk score ${clamped} of 100`}>
      <svg width="200" height="128" viewBox="0 0 200 128">
        {ticks.map((t) => (
          <line
            key={t.key}
            x1={t.x1}
            y1={t.y1}
            x2={t.x2}
            y2={t.y2}
            stroke="var(--line)"
            strokeWidth="1.5"
          />
        ))}
        {/* Track */}
        <path
          d="M 22 100 A 78 78 0 0 1 178 100"
          fill="none"
          stroke="var(--line)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Value */}
        <path
          d="M 22 100 A 78 78 0 0 1 178 100"
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={CIRC}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1s cubic-bezier(0.16,1,0.3,1)" }}
        />
        <text
          x="100"
          y="90"
          textAnchor="middle"
          className="font-display tnum"
          style={{ fontSize: 42, fontWeight: 600, fill: "var(--ink)" }}
        >
          {clamped}
        </text>
        <text
          x="100"
          y="112"
          textAnchor="middle"
          style={{ fontSize: 11, letterSpacing: "0.12em", fill: "var(--muted)" }}
        >
          RISK INDEX
        </text>
      </svg>
    </div>
  );
}
