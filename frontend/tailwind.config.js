/** @type {import('tailwindcss').Config} */
// Colors are given as literal hex (mirroring the CSS variables in index.css) so
// Tailwind can parse them and support opacity modifiers like `text-paper/90`.
// The matching CSS variables remain the source of truth for inline styles/SVG.
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Archivo"', '"Manrope"', "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ['"Manrope"', "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        paper: "#f4f1ea",
        "paper-2": "#fbfaf6",
        card: "#ffffff",
        ink: "#0e1c2b",
        "ink-soft": "#33465a",
        muted: "#6c7b89",
        line: "#e6e0d4",
        "line-cool": "#e4e9ef",
        gold: "#b0843f",
        "gold-soft": "#cba55f",
        "gold-wash": "#f6efe0",
        real: "#1f7a54",
        "real-wash": "#e8f3ec",
        amber: "#a9761b",
        "amber-wash": "#f7efdd",
        fake: "#b12f2c",
        "fake-wash": "#f7e7e4",
      },
      boxShadow: {
        sm: "var(--shadow-sm)",
        md: "var(--shadow-md)",
        lg: "var(--shadow-lg)",
      },
      letterSpacing: {
        tightish: "-0.015em",
      },
    },
  },
  plugins: [],
};
