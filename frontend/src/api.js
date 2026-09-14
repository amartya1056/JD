// Thin API client. In dev, requests go to /api/* and Vite proxies them to the
// backend (see vite.config.js). In production set VITE_API_BASE to the API URL.
const BASE = import.meta.env.VITE_API_BASE || "/api";

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    // FastAPI validation errors arrive as {detail: [...]} or {detail: "..."}.
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d) => d.msg).join("; ")
      : data.detail || `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return data;
}

export function analyze({ url, text }) {
  return post("/analyze", { url: url || null, text: text || null });
}

export function sendFeedback({ submissionId, correctLabel }) {
  return post("/feedback", {
    submission_id: submissionId,
    correct_label: correctLabel,
  });
}
