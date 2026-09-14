// Renders the posting text with suspicious phrases underlined in the accent
// colour — the text-model saliency shown in situ so the user sees exactly which
// wording the model reacted to.
function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export default function HighlightedText({ text, phrases }) {
  if (!text) return null;
  const clean = (phrases || []).filter((p) => p && p.trim().length > 1);

  let nodes = [text];
  clean.forEach((phrase, idx) => {
    const re = new RegExp(`(${escapeRegExp(phrase)})`, "ig");
    nodes = nodes.flatMap((node) => {
      if (typeof node !== "string") return [node];
      const parts = node.split(re);
      return parts.map((part, i) =>
        re.test(part) && part.toLowerCase() === phrase.toLowerCase() ? (
          <mark
            key={`${idx}-${i}-${part}`}
            className="rounded-[3px] px-0.5"
            style={{
              background: "var(--gold-wash)",
              color: "var(--ink)",
              boxShadow: "inset 0 -0.55em 0 rgba(203,165,95,0.35)",
            }}
          >
            {part}
          </mark>
        ) : (
          part
        )
      );
    });
  });

  return (
    <p className="whitespace-pre-wrap text-[13.5px] leading-relaxed text-ink-soft">
      {nodes}
    </p>
  );
}
