/*
  Turning one result into HTML. Pure: every function takes a result object and returns a
  string, touching no element and fetching nothing.

  It is a separate file because there are two front ends. The live app posts to /api/ask
  and renders what comes back; the public demo (docs/demo/) renders recordings of the same
  shape, served as static files with no Python behind them. Both must draw an answer, a
  trajectory and a cost table identically, or the demo stops being evidence of what the
  real thing does.

  Shape of `result` is `rpsg.web.service.AskResult` as JSON — the response body of
  /api/ask, and exactly what `scripts/record_demo.py` writes into recordings.json.
*/
"use strict";

window.RPSG = (() => {
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

  // The synthesis prompt does not ask for markdown, but the models emit it anyway, and
  // literal `**` in an answer reads as a typo. Only the three inline forms that actually
  // show up are handled — this is not a markdown parser, and it must never become one.
  //
  // Applied strictly after `escapeHtml`, so the only tags in the result are the ones
  // these three rules put there. Reversing that order would make model output executable.
  const renderInline = (escaped) => escaped
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");

  // `[paper:<id>]` is what the synthesizer's handles are rewritten back to before the
  // answer is stored, so it is the form every arm's answer arrives in.
  const renderAnswer = (text) =>
    renderInline(escapeHtml(text)).replace(/\[paper:([^\]]+)\]/g,
      (_, id) => `<span class="cite">${escapeHtml(id)}</span>`);

  const fmtInt = (n) => Number(n).toLocaleString();

  const card = (title, inner, extraClass) =>
    `<div class="card${extraClass ? " " + extraClass : ""}"><h2>${title}</h2>${inner}</div>`;

  function renderTrajectory(t) {
    const parts = [];
    if (t.planner_failed) {
      parts.push('<p class="error" style="margin-top:0">The planner failed — this degraded ' +
                 "to a single retrieval on the original question.</p>");
    }
    const subs = t.sub_questions || [];
    parts.push(`<p style="margin:0">Plan: ${subs.length} sub-question(s)</p>`);
    if (subs.length) {
      parts.push(`<ol class="plan">${subs.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ol>`);
    }
    if (t.plan_reasoning) {
      parts.push(`<p class="hint">reasoning: ${escapeHtml(t.plan_reasoning)}</p>`);
    }
    if (t.graph_hints && t.graph_hints.length) {
      parts.push(`<p class="hint">graph hints: ${escapeHtml(t.graph_hints.slice(0, 6).join(", "))}</p>`);
    }
    const bits = [
      `retrievals used: ${t.retrievals_used || 0}`,
      `refused: ${t.retrievals_refused || 0}`,
      `anchor: ${t.anchor_used ? "on" : "off"}`,
    ];
    if (t.critique_ran) {
      bits.push(`critique: ${(t.gaps || []).length} gap(s), ` +
                `${(t.critique_added_papers || []).length} paper(s) added`);
    } else {
      bits.push("critique: disabled");
    }
    parts.push(`<div class="meta">${bits.map((b) => `<span>${escapeHtml(b)}</span>`).join("")}</div>`);
    if (t.critique_assessment) {
      parts.push(`<p class="hint">assessment: ${escapeHtml(t.critique_assessment)}</p>`);
    }
    return card("Trajectory", parts.join(""));
  }

  function renderUsage(rows, elapsed) {
    const body = rows.length
      ? "<table><thead><tr><th>model</th><th>calls</th><th>in</th><th>out</th>" +
        "<th>cached</th><th>cost</th></tr></thead><tbody>" +
        rows.map((r) => `<tr><td>${escapeHtml(r.model)}</td><td class="num">${fmtInt(r.calls)}</td>` +
          `<td class="num">${fmtInt(r.input_tokens)}</td><td class="num">${fmtInt(r.output_tokens)}</td>` +
          `<td class="num">${fmtInt(r.cached_input_tokens)}</td>` +
          // Not "$0.00": an unpriced model is unknown cost, not free.
          `<td class="num">${r.cost_usd === null ? "n/a" : "$" + r.cost_usd.toFixed(4)}</td></tr>`).join("") +
        "</tbody></table>"
      : '<p class="hint" style="margin:0">No LLM calls — retrieval only.</p>';
    const note = rows.some((r) => r.cost_usd === null)
      ? '<p class="hint">cost "n/a": add rates under models.pricing in configs/settings.yaml.</p>'
      : "";
    return card("This question", body + note +
      `<div class="meta"><span>${Number(elapsed).toFixed(1)}s</span></div>`);
  }

  /** The whole result as HTML. Returns a string; the caller decides where it goes. */
  function renderResult(d) {
    const out = [];

    if (d.answer !== null && d.answer !== undefined) {
      const papers = d.cited_paper_ids.length
        ? `<ul class="papers">${d.cited_paper_ids.map((p) => `<li>${escapeHtml(p)}</li>`).join("")}</ul>`
        : '<p class="hint">The answer cited no paper.</p>';
      out.push(card(`Answer — ${escapeHtml(d.arm)}`,
        `<div class="answer">${renderAnswer(d.answer)}</div>` +
        `<p class="hint" style="margin-bottom:0">grounded on ${d.cited_paper_ids.length} paper(s)</p>` +
        papers));
    }

    if (d.trace && Object.keys(d.trace).length) out.push(renderTrajectory(d.trace));

    if (d.hits && d.hits.length) {
      const rows = d.hits.map((h) =>
        '<div class="hitrow"><div class="hitmeta">' +
        `<span class="score">${h.score >= 0 ? "+" : ""}${h.score.toFixed(3)}</span>` +
        `<span class="tag">${escapeHtml(h.section_type)}</span>` +
        `<span>${escapeHtml(h.paper_id)}</span></div>` +
        `<div class="excerpt">${escapeHtml(h.excerpt)}…</div></div>`).join("");
      out.push(card(`Retrieved — ${d.hits.length} chunk(s)`, `<div class="scroll">${rows}</div>`));
    }

    if (d.evidence) {
      out.push(card(`Evidence sent to the model — ${fmtInt(d.evidence.length)} chars`,
        `<div class="scroll"><pre class="evidence">${escapeHtml(d.evidence)}</pre></div>`));
    }

    out.push(renderUsage(d.usage, d.elapsed_s));
    return out.join("");
  }

  return { escapeHtml, renderAnswer, fmtInt, card, renderTrajectory, renderUsage, renderResult };
})();
