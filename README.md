# Research-Paper-Synthesis-Graph (RPSG)

An agentic, **typed Graph-RAG** system for synthesizing relational answers across public
literature (ArXiv, Semantic Scholar) and an internal corpus. The core bet is a *typed*
knowledge graph (Paper / Method / Problem / Dataset / Claim / Limitation, plus a
reproducibility layer) rather than a citation network or vector-only RAG — so a
researcher can ask *"what methods were tried on problem X, which were limited by Y, and
what's still open?"*

> **This repository is the Phase-1 (Iteration 1) spine + evaluation scaffold.**
> It builds the ingestion → chunk → extract → store pipeline and — first — the evaluation
> harness. The exit criterion for Iteration 1 is: a **vector-over-full-text baseline scored
> end-to-end by a calibrated LLM judge.** The agentic planner–critic loop and the four
> extensions come in Iterations 2–3.

[![CI](https://github.com/jask7075/Research-Paper-Synthesis-Graph/actions/workflows/ci.yml/badge.svg)](https://github.com/jask7075/Research-Paper-Synthesis-Graph/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Status

**The pipeline runs end to end on a real corpus. The exit criterion is not yet met** — it
requires a scored run against a real gold query set, and the gold set is still being
written (the committed one holds placeholder paper ids). Stated plainly because the
distinction matters: the system works, but nothing here is a measured *claim* yet.

Built and verified by running it:

| | |
|---|---|
| corpus | 317 papers (Semantic Scholar metadata), 249 with full text |
| sections / chunks | 5,229 sections → 9,913 chunks |
| typed graph | 23,460 nodes, 10,660 edges (Kuzu) |
| citation layer | 2,333 `cites` edges, in-corpus out-degree 7.64 |
| vector index | 9,913 chunks, SPECTER embeddings (faiss) |
| extraction cost | $5.60 total, ~2¢/paper (`gpt-5.4-nano`, 4,800 calls) |
| query cost | ~$0.005 (`ask.py`, retrieve + synthesise) |
| tests | 25 passing; ruff + mypy clean in CI |

Ask it something with [`scripts/ask.py`](scripts/ask.py); inspect what the parser sees with
[`scripts/inspect_pdf.py`](scripts/inspect_pdf.py); run the whole thing with
[`scripts/run_pipeline.py`](scripts/run_pipeline.py).

**Outstanding for Iteration 1**

- **Gold query set** — in progress. Until then `scripts/06_run_eval.py` runs but its
  deterministic metrics are meaningless (`must_cite_recall` scores against placeholder ids).
- **Judge calibration** — `rpsg.eval.calibration` is implemented and unit-tested; no
  driver runs it, so judge scores are uncalibrated and should not be quoted.
- **`extraction_gold` / `repro_gold`** — schemas exist by example, unpopulated. So every
  statement about extraction *quality* below is an inspection, not a measurement.

**Known limitations, measured**

- **No entity resolution.** Node ids are slugified surface names, so `random circuits` and
  `Random Parameterized Quantum Circuits (RPQCs)` remain distinct nodes. Identical strings
  do collapse (605 `Method` duplicates merged on store), which is partial and accidental.
  Iteration 2.
- **`refutes` / `undercuts` are near-empty** (4 and 24 edges across 249 papers). This is
  structural, not a tuning gap: extraction runs per-section within one paper, and a paper
  rarely refutes itself. Cross-paper contradictions need a second pass over extracted
  claims — Iteration 2, with entity resolution.
- **`Hardware` is sparse** (8 nodes / 6 papers). Most quantum-computing papers simply do
  not report hardware specifications, so extension #4's ceiling is set by the literature
  rather than by extraction.
- **52% of sections type as `other`.** Most are genuine domain subsections ("The role of
  measurements") that no keyword scheme will classify; they fall back to the default node
  types. GROBID would improve this and is preferred when reachable — the bundled fallback
  exists because GROBID ships amd64-only and cannot spawn `pdfalto` under Apple-Silicon
  emulation.

**Next (Iteration 2):** entity resolution for Method/Problem nodes, the typed-graph
retrieval system, and the citation-graph ablation.

## Design principles (why the code is shaped this way)

1. **Eval-first.** The harness (`rpsg.eval`) is built and calibrated against a *placeholder*
   system before the real system exists. If you can't score a baseline, you can't claim
   the typed graph earns its complexity.
2. **Tiered schema.** Metadata (Tier A, from APIs) is cheap and high-precision; semantic
   nodes (Tier B) and relational edges (Tier C, e.g. `refutes`/`undercuts`) are expensive
   and noisy. The hard tier must never *block* the system. See `rpsg.extraction.schema`.
3. **Curated vs. staged layers.** Offline, reviewed extractions are `curated`; anything the
   agent writes at query time is `staged` with provenance and is never auto-merged. Metrics
   run against `curated`.
4. **Build the base corpus with an API model.** Corpus extraction is a *one-time offline
   batch* — the quality gap between a local 8B and a frontier-small API model is exactly
   your system's ceiling. Local models are reserved for query-time inference.
5. **Portability by interface.** Graph and vector stores sit behind `rpsg.stores.base`;
   chat models sit behind `rpsg.llm.ChatClient`. Provider is inferred from the model id,
   so switching LLM vendor is a one-line edit in `configs/settings.yaml` and Phase 2
   (Neo4j AuraDB + Qdrant) is a config swap, not a rewrite.

## Layout

```
configs/            YAML config (settings, baselines, judge)
data/               raw → interim → processed  (git-ignored; cookiecutter-ds convention)
src/rpsg/
  config.py         pydantic-settings; single source of runtime config
  ingestion/        Semantic Scholar / ArXiv fetch, PDF→sections, section-aware chunking
  llm/              provider-neutral ChatClient (OpenAI / Anthropic) + token accounting
  extraction/       frozen tiered schema + prompts + API-based extractor
  stores/           GraphStore / VectorStore interfaces + Kuzu / local adapters
  retrieval/        baselines (vector-abstract, vector-fulltext)
  eval/             gold schema, deterministic metrics, LLM judge, calibration, runner
scripts/            numbered pipeline entrypoints (01_… → 06_…) plus these tools:
  run_pipeline.py     sequence 01–06 with preflight checks and per-stage accounting
  ask.py              ask the corpus a question (retrieve + synthesise)
  inspect_pdf.py      dry-run one PDF: sections, chunks, and extraction routing
  report_state.py     regenerate docs/pipeline-state.md from what is on disk
docs/pipeline-state.md  a generated snapshot of the numbers below
eval/gold/          gold query set + extraction/reproducibility ground truth (jsonl)
eval/runs/          per-run outputs (answers, traces, scores)
tests/              unit tests for the deterministic core
```

## Quickstart

```bash
# 1. Environment (uv recommended; falls back to pip)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,vector]"

# 2. Secrets
cp .env.example .env       # then fill OPENAI_API_KEY, S2_API_KEY

# 3. Optional services (only needed for real PDF parsing)
#    GROBID as a docker service — see Makefile `grobid` target.

# 4. Run the deterministic tests (no API keys required)
make test

# 5. Fetch a small corpus and run the pipeline (needs S2_API_KEY)
python scripts/01_fetch_corpus.py --query "variational quantum eigensolver" --limit 50
python scripts/02_parse_pdfs.py
python scripts/03_chunk.py
python scripts/04_extract.py            # needs OPENAI_API_KEY
python scripts/05_build_stores.py

# 6. Score the vector-fulltext baseline against the gold set (needs OPENAI_API_KEY)
python scripts/06_run_eval.py --system vector_fulltext
```

Or drive the whole thing, with every prerequisite checked before any work happens:

```bash
python scripts/run_pipeline.py --query "variational quantum eigensolver" --limit 200
python scripts/run_pipeline.py --from 02 --to 04          # resume mid-pipeline
python scripts/run_pipeline.py --dry-run                  # plan + preflight only
```

Then ask it something, or look at what the parser sees:

```bash
python scripts/ask.py "what mitigates barren plateaus?"
python scripts/ask.py "..." --retrieval-only              # no LLM call, no cost
python scripts/inspect_pdf.py data/raw/pdfs/<paper_id>.pdf
python scripts/report_state.py --format html              # printable state report
```

## Finding: relevance search cannot build a citation-connected corpus; co-citation can

Building a quantum-computing corpus from 12 Semantic Scholar relevance queries produced a
citation graph too sparse to traverse. Adding 77 co-cited papers fixed it. Both halves are
measured, and the second is the useful part.

### The problem

| 240 papers, 12 relevance queries | |
|---|---|
| reference entries | 8,443 (~35 / paper) |
| pointing at a paper **inside** the corpus | **309 (3.66%)** |
| in-corpus out-degree | **1.29** citations / paper |
| papers with ≥1 outgoing in-corpus citation | 134 / 240 (56%) |
| 2-hop citation neighbourhood | ~2.9 papers |

**96% of references leave the corpus.** Papers cite foundational mathematics, physics, and
machine-learning work that no topically-sampled corpus will ever contain. Relevance search
returns papers *about* a topic, not papers that *cite each other*.

Scaling does not fix this. Going from 24 → 240 papers (10×) moved density only
1.77% → 3.66% (2×), because the share of out-of-field references is roughly constant. A
2-hop neighbourhood reaching ~3 papers cannot answer a question needing 10–20 chunks, and
for the 44% of papers with no outgoing in-corpus citation it returns nothing at all.

### The fix

Co-citation expansion: fetch the papers the corpus *already cites*. Those citations exist
in the reference lists already, dangling because the target is absent — so each paper
added arrives pre-connected. `scripts/01_fetch_corpus.py --expand-citations`.

| | 240 papers | **+77 co-cited (317)** |
|---|---|---|
| in-corpus references | 3.66% | **16.68%** |
| out-degree | 1.29 | **7.64** |
| papers with ≥1 citation | 56% | **75%** |
| 2-hop reach | ~2.9 papers | **~66 papers** |
| `cites` edges in the graph | 309 | **2,333** |

**77 papers bought 2,024 edges — 26 each. The preceding 216 relevance-search papers bought
292 — 1.4 each.** An ~18× difference in edges per paper ingested, because hubs are
selected *because* the corpus already points at them. (The most-cited absent paper was
referenced by 77 of the 240.) Returns peak around a co-citation threshold of 6–10 and then
dilute, so this is not a lever to pull indefinitely.

Note the ordering, which matters: hubs are only identifiable *because* relevance search
built a corpus with 8,443 references to mine first. The method is **relevance search to
establish a field, then co-citation to connect it** — not one technique beating the other.

### What this is not

It is **not** evidence that typed edges outperform untyped ones. At 240 papers that
comparison was confounded — 3,813 typed edges against 309 citation edges means a typed win
could be explained entirely by edge count. After expansion the arms are within the same
order of magnitude (5,738 typed vs 2,333 citation edges), which is what makes the
Iteration-2 ablation an experiment rather than a foregone conclusion.

## Models

Extraction (one-time batch, the bulk-call stage): `gpt-5.4-nano`. Judge / synthesis:
`gpt-5.4-mini`. Embeddings: `allenai/specter2_base` locally via sentence-transformers —
no API cost per chunk. Query-time local inference (Iteration 2 onward):
Qwen2.5-14B-Instruct via vLLM. All configurable in `configs/settings.yaml`.

Provider is inferred from the model id, so pointing the system at a different vendor is a
one-line change — swap `gpt-5.4-mini` for `claude-opus-4-8` and `rpsg.llm.get_chat_client`
routes to the Anthropic adapter instead. Set `models.provider` to override the inference.

Caveat worth knowing: judge and synthesis currently share a family, so self-preference
bias in the judge is unmitigated. That's fine while no comparative claim rests on the
judge scores — revisit before reporting an ablation number.

## License

MIT — see [LICENSE](LICENSE).