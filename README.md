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

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Tests](https://img.shields.io/badge/tests-19%2F19%20passing-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Status — Iteration 1 (complete):** the ingestion → chunk → extract → store pipeline plus
the eval-first harness. The deterministic core (schema, chunking, metrics, calibration,
store interfaces) is verified by 19/19 unit tests. Wire in `ANTHROPIC_API_KEY` /
`S2_API_KEY` to run the LLM stages. Iterations 2–3 add the typed graph, the planner–critic
loop, and the four extensions.

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
5. **Portability by interface.** Graph / vector / LLM sit behind interfaces
   (`rpsg.stores.base`) so Phase 2 (Neo4j AuraDB + Qdrant + Claude API) is a config swap,
   not a rewrite.

## Layout

```
configs/            YAML config (settings, baselines, judge)
data/               raw → interim → processed  (git-ignored; cookiecutter-ds convention)
src/rpsg/
  config.py         pydantic-settings; single source of runtime config
  ingestion/        Semantic Scholar / ArXiv fetch, PDF→sections, section-aware chunking
  extraction/       frozen tiered schema + prompts + API-based extractor
  stores/           GraphStore / VectorStore interfaces + Kuzu / local adapters
  retrieval/        baselines (vector-abstract, vector-fulltext)
  eval/             gold schema, deterministic metrics, LLM judge, calibration, runner
scripts/            numbered pipeline entrypoints (01_… → 06_…)
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
cp .env.example .env       # then fill ANTHROPIC_API_KEY, S2_API_KEY

# 3. Optional services (only needed for real PDF parsing)
#    GROBID as a docker service — see Makefile `grobid` target.

# 4. Run the deterministic tests (no API keys required)
make test

# 5. Fetch a small corpus and run the pipeline (needs S2_API_KEY)
python scripts/01_fetch_corpus.py --query "variational quantum eigensolver" --limit 50
python scripts/02_parse_pdfs.py
python scripts/03_chunk.py
python scripts/04_extract.py            # needs ANTHROPIC_API_KEY
python scripts/05_build_stores.py

# 6. Score the vector-fulltext baseline against the gold set (needs ANTHROPIC_API_KEY)
python scripts/06_run_eval.py --system vector_fulltext
```

## Models

Extraction (one-time batch): `claude-haiku-4-5`. Judge / synthesis: `claude-opus-4-8`.
Query-time local inference (Iteration 2 onward): Qwen2.5-14B-Instruct via vLLM. All configurable in
`configs/settings.yaml`.

## Status

**Iteration 1 — complete.** Deterministic core (schema, chunking, metrics, calibration,
store interfaces) is implemented and verified (`make test` → 19/19). External-service
modules (S2 / ArXiv / GROBID / extractor / judge) are implemented against real signatures;
wire in your keys/services to run them.

**Next (Iteration 2):** entity resolution for Method/Problem nodes, the typed-graph
retrieval system, and the citation-graph ablation — so the by-query-type comparison is
ready the moment extraction lands.

## License

MIT — see [LICENSE](LICENSE).