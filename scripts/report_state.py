"""Produce a readable report of what the pipeline has actually built.

    python scripts/report_state.py                      # -> reports/state-<utc>.md
    python scripts/report_state.py --format html        # styled, printable to PDF
    python scripts/report_state.py -o /tmp/state.md

Everything is recomputed from the artifacts on disk — no numbers are hardcoded — so this
stays honest as the corpus grows and can be re-run after any stage. Needs no API keys and
makes no network calls.

Missing artifacts are reported as missing rather than raising, so it is safe to run
part-way through a pipeline.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from rpsg.config import get_settings
from rpsg.extraction.schema import EDGE_TIER, NODE_TIER, EdgeType, NodeType
from rpsg.ingestion.chunking import DROP_SECTION_TYPES

Rows = list[tuple[str, str]]
Section = tuple[str, str, Rows]  # (heading, note, rows)


def _n(value: float | int) -> str:
    return f"{value:,}" if isinstance(value, int) else f"{value:,.2f}"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _corpus(s) -> Section:
    papers = _jsonl(s.paths.data_external / "papers.jsonl")
    pdfs = list((s.paths.data_raw / "pdfs").glob("*.pdf"))
    parsed = list((s.paths.data_interim / "sections").glob("*.json"))
    rows: Rows = [
        ("metadata records", _n(len(papers))),
        ("PDFs downloaded", _n(len(pdfs))),
        ("papers parsed to sections", _n(len(parsed))),
    ]
    if papers:
        with_abstract = sum(1 for p in papers if p.get("abstract"))
        rows.append(("with abstract", f"{with_abstract:,} ({with_abstract / len(papers):.0%})"))
        tier_a_only = len(papers) - len(parsed)
        rows.append(
            (
                "Tier-A only (no full text)",
                f"{tier_a_only:,} — metadata + citations, no Method/Problem/Claim",
            )
        )
    note = (
        "Papers without a downloadable PDF contribute Tier-A metadata and citation edges "
        "but no semantic nodes, so the corpus has two classes of paper."
    )
    return "Corpus", note, rows


def _sections(s) -> Section:
    files = sorted((s.paths.data_interim / "sections").glob("*.json"))
    if not files:
        return "Sections", "Stage 02 has not run.", []
    per, types, no_route = [], collections.Counter(), 0
    for f in files:
        secs = json.loads(f.read_text())
        per.append(len(secs))
        counts = collections.Counter(x["section_type"] for x in secs)
        types.update(counts)
        if not (
            counts.get("conclusion") or counts.get("discussion") or counts.get("limitations")
        ):
            no_route += 1
    total = sum(per)
    rows: Rows = [
        ("total sections", _n(total)),
        ("per paper", f"median {statistics.median(per):.0f}, min {min(per)}, max {max(per)}"),
    ]
    for stype, count in types.most_common():
        flag = "  (dropped before chunking + extraction)" if stype in DROP_SECTION_TYPES else ""
        rows.append((f"type: {stype}", f"{count:,} ({count / total:.1%}){flag}"))
    rows.append(
        (
            "papers with no Limitation route",
            f"{no_route}/{len(files)} ({no_route / len(files):.0%})",
        )
    )
    dropped = sum(types[t] for t in DROP_SECTION_TYPES)
    rows.append(("extraction calls this corpus costs", _n(total - dropped)))
    note = (
        "`section_type` selects which node and edge types extraction asks for "
        "(`rpsg.extraction.prompts`). Only `conclusion`, `discussion`, and `limitations` "
        "route `Limitation`, so papers with none of them cannot produce one."
    )
    return "Sections", note, rows


def _chunks(s) -> Section:
    rows_data = _jsonl(s.paths.data_interim / "chunks.jsonl")
    if not rows_data:
        return "Chunks", "Stage 03 has not run.", []
    lens = sorted(len(r["text"]) for r in rows_data)
    n = len(lens)
    by_corpus = collections.Counter(r["corpus"] for r in rows_data)
    rows: Rows = [
        ("total chunks", _n(n)),
        ("by corpus", ", ".join(f"{k}={v:,}" for k, v in by_corpus.most_common())),
        ("length: median", f"{statistics.median(lens):,.0f} chars"),
        ("length: min / max", f"{lens[0]:,} / {lens[-1]:,} chars"),
        ("under 300 chars", f"{sum(1 for x in lens if x < 300):,} "
                            f"({sum(1 for x in lens if x < 300) / n:.1%})"),
    ]
    note = (
        f"Similarity is damped for chunks under {s.retrieval.length_damping_chars} chars at "
        "search time (`availability` exempt), because short text embeds near the corpus "
        "centroid and over-scores against every query."
    )
    return "Chunks", note, rows


def _extraction(s) -> Section:
    recs = _jsonl(s.paths.data_processed / "extractions.jsonl")
    if not recs:
        return "Extraction (Tier B/C)", "Stage 04 has not run.", []
    nodes = [x for r in recs for x in r["nodes"]]
    edges = [x for r in recs for x in r["edges"]]
    npapers = len(recs)
    rows: Rows = [
        ("papers extracted", _n(npapers)),
        ("nodes / edges", f"{len(nodes):,} / {len(edges):,}"),
        ("nodes per paper", f"{len(nodes) / npapers:.0f}"),
    ]
    ntypes = collections.Counter(x["type"] for x in nodes)
    for t in [x.value for x in NodeType]:
        count = ntypes.get(t, 0)
        if not count:
            continue
        covered = len({r["paper_id"] for r in recs for x in r["nodes"] if x["type"] == t})
        rows.append((f"node: {t}", f"{count:,} in {covered}/{npapers} papers"))
    etypes = collections.Counter(x["type"] for x in edges)
    for t in [x.value for x in EdgeType]:
        if etypes.get(t):
            rows.append((f"edge: {t}", _n(etypes[t])))
    tiers: collections.Counter = collections.Counter()
    for x in nodes:
        tiers[NODE_TIER[NodeType(x["type"])].value] += 1
    etiers: collections.Counter = collections.Counter()
    for x in edges:
        etiers[EDGE_TIER[EdgeType(x["type"])].value] += 1
    for tier in sorted(set(tiers) | set(etiers)):
        rows.append(
            (f"tier {tier}", f"{tiers.get(tier, 0):,} nodes, {etiers.get(tier, 0):,} edges")
        )
    rows.append(("papers yielding 0 nodes", _n(sum(1 for r in recs if not r["nodes"]))))
    below_n = sum(1 for x in nodes if x["confidence"] < s.extraction.min_node_confidence)
    below_e = sum(1 for x in edges if x["confidence"] < s.extraction.min_edge_confidence)
    rows.append(
        (
            "below confidence gate",
            f"{below_n} nodes, {below_e} edges (expected 0; gates "
            f"{s.extraction.min_node_confidence}/{s.extraction.min_edge_confidence})",
        )
    )
    note = (
        "Tier A never appears here — Paper/Author/Venue and `cites` come from Semantic "
        "Scholar in stage 05, not from the LLM. `refutes`/`undercuts` are near-empty by "
        "construction: extraction runs per-section within one paper, so it cannot see "
        "cross-paper contradictions."
    )
    return "Extraction (Tier B/C)", note, rows


def _citations(s) -> Section:
    papers = _jsonl(s.paths.data_external / "papers.jsonl")
    if not papers:
        return "Citation graph (Tier A)", "Stage 01 has not run.", []
    ids = {p["paperId"] for p in papers}
    refs = [r.get("paperId") for p in papers for r in (p.get("references") or [])]
    refs = [r for r in refs if r]
    inside = sum(1 for r in refs if r in ids)
    n = len(papers)
    linked = sum(
        1
        for p in papers
        if any(r.get("paperId") in ids for r in (p.get("references") or []))
    )
    d = inside / n if n else 0.0
    rows: Rows = [
        ("papers", _n(n)),
        ("reference entries", f"{len(refs):,} (~{len(refs) / n:.0f}/paper)" if n else "0"),
        ("pointing inside the corpus", f"{inside:,} ({inside / len(refs):.2%})" if refs else "0"),
        ("in-corpus out-degree", f"{d:.2f} citations/paper"),
        ("papers with >=1 outgoing cite", f"{linked}/{n} ({linked / n:.0%})" if n else "0"),
        ("2-hop reach (approx d + d^2)", f"~{d + d * d:.0f} papers"),
        ("traversable for an ablation?", "yes" if d >= 2.5 else "NO — too sparse"),
    ]
    note = (
        "Relevance search returns papers *about* a topic, not papers that cite each other. "
        "Co-citation expansion (`--expand-citations`) is what makes this traversable; see "
        "the finding section of the README."
    )
    return "Citation graph (Tier A)", note, rows


def _stores(s) -> Section:
    rows: Rows = []
    for label, path in (
        ("vector index", s.paths.vector_index),
        ("vector metadata", s.paths.vector_index.with_suffix(".meta.jsonl")),
        ("graph (Kuzu)", s.paths.kuzu_db),
    ):
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*")) if path.is_dir() \
                else path.stat().st_size
            rows.append((label, f"{size / 1e6:,.1f} MB"))
        else:
            rows.append((label, "not built"))
    if s.paths.kuzu_db.exists():
        try:
            from rpsg.stores.graph_store import KuzuGraphStore

            g = KuzuGraphStore(str(s.paths.kuzu_db))
            nodes = g.query("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"]
            edges = g.query("MATCH ()-[e:REL]->() RETURN count(e) AS c")[0]["c"]
            rows.append(("graph contents", f"{nodes:,} nodes, {edges:,} edges"))
            for r in g.query(
                "MATCH ()-[e:REL]->() RETURN e.type AS t, count(*) AS c ORDER BY c DESC"
            ):
                rows.append((f"  edge: {r['t']}", _n(r["c"])))
        except Exception as exc:  # noqa: BLE001 - a report must not fail on an inspection
            rows.append(("graph query failed", str(exc)[:100]))
    return "Stores", "", rows


def _eval_state(s) -> Section:
    rows: Rows = []
    for name in ("queries", "extraction_gold", "repro_gold"):
        path = s.paths.eval_gold / f"{name}.jsonl"
        recs = _jsonl(path)
        placeholder = "PLACEHOLDER" in path.read_text() if path.exists() else False
        detail = f"{len(recs)} record(s)"
        if placeholder:
            detail += "  — CONTAINS PLACEHOLDERS, metrics not meaningful"
        rows.append((f"gold/{name}.jsonl", detail))
    runs = [p for p in s.paths.eval_runs.glob("*") if p.is_dir()]
    rows.append(("scored runs", f"{len(runs)}" if runs else "none"))
    note = (
        "The Iteration-1 exit criterion is a scored `vector_fulltext` run against a real "
        "gold set. Placeholder ids make `must_cite_recall` and `citation_precision` "
        "meaningless, and no driver runs `rpsg.eval.calibration`, so judge scores are "
        "uncalibrated."
    )
    return "Evaluation state", note, rows


def _config(s) -> Section:
    rows: Rows = [
        ("extraction model", s.models.extraction_model),
        ("judge / synthesis model", f"{s.models.judge_model} / {s.models.synthesis_model}"),
        ("embeddings", f"{s.embeddings.model_name} (dim {s.embeddings.dim})"),
        ("confidence gates", f"nodes >={s.extraction.min_node_confidence}, "
                             f"edges >={s.extraction.min_edge_confidence}"),
        ("extraction workers", _n(s.extraction.max_workers)),
        ("length damping", f"{s.retrieval.length_damping_chars} chars"),
        ("chunk target", f"{s.chunking.target_tokens} tokens, "
                         f"{s.chunking.overlap_tokens} overlap"),
        ("pricing configured", "yes" if s.models.pricing else "no — costs report as n/a"),
    ]
    return "Configuration", "", rows


def render_markdown(sections: list[Section], generated: str) -> str:
    out = [
        "# RPSG — pipeline state report",
        "",
        f"Generated {generated} by `scripts/report_state.py`. Every figure is recomputed "
        "from the artifacts on disk.",
        "",
    ]
    for heading, note, rows in sections:
        out += [f"## {heading}", ""]
        if note:
            out += [f"> {note}", ""]
        if not rows:
            out += ["_no data_", ""]
            continue
        out += ["| | |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in rows]
        out += [""]
    return "\n".join(out)


def render_html(sections: list[Section], generated: str) -> str:
    css = (
        "body{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "max-width:56rem;margin:3rem auto;padding:0 1.5rem;color:#1a1a1a}"
        "h1{font-size:1.7rem;border-bottom:2px solid #eee;padding-bottom:.4rem}"
        "h2{font-size:1.15rem;margin-top:2.2rem;color:#0b5}"
        "blockquote{margin:.6rem 0;padding:.6rem .9rem;background:#f6f8fa;"
        "border-left:3px solid #ccc;color:#444;font-size:.92rem}"
        "table{border-collapse:collapse;width:100%;margin:.6rem 0}"
        "td{border-bottom:1px solid #eee;padding:.35rem .5rem;vertical-align:top}"
        "td:first-child{color:#555;width:45%}"
        "code{background:#f6f8fa;padding:.1rem .3rem;border-radius:3px}"
        "@media print{body{margin:0;max-width:none}h2{page-break-after:avoid}}"
    )
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>RPSG — pipeline state report</title>",
        f"<style>{css}</style>",
        "<h1>RPSG — pipeline state report</h1>",
        f"<p>Generated {generated} by <code>scripts/report_state.py</code>. "
        "Every figure is recomputed from the artifacts on disk.</p>",
    ]
    for heading, note, rows in sections:
        parts.append(f"<h2>{heading}</h2>")
        if note:
            parts.append(f"<blockquote>{note}</blockquote>")
        if not rows:
            parts.append("<p><em>no data</em></p>")
            continue
        body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        parts.append(f"<table>{body}</table>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["md", "html"], default="md")
    ap.add_argument("-o", "--out", type=Path, help="output path (default reports/state-<utc>.*)")
    args = ap.parse_args()

    s = get_settings()
    sections = [
        _corpus(s),
        _sections(s),
        _chunks(s),
        _extraction(s),
        _citations(s),
        _stores(s),
        _eval_state(s),
        _config(s),
    ]
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    text = (
        render_markdown(sections, stamp)
        if args.format == "md"
        else render_html(sections, stamp)
    )

    out = args.out
    if out is None:
        slug = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = Path("reports") / f"state-{slug}.{args.format}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}  ({len(text):,} chars)")
    if args.format == "html":
        print("  open it and print-to-PDF for a PDF copy")


if __name__ == "__main__":
    main()
