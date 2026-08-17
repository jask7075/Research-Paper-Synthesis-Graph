# Research-Paper-Synthesis-Graph (RPSG)

An agentic, **typed Graph-RAG** system for synthesizing relational answers across public
literature (ArXiv, Semantic Scholar), built to answer questions like *"what methods were
tried on problem X, which were limited by Y, and what's still open?"*

The original bet was that a *typed* knowledge graph (Paper / Method / Problem / Dataset /
Claim / Limitation, plus a reproducibility layer) would retrieve such answers better than a
citation network or vector-only RAG. **Three iterations of measurement do not support that
bet, and this repository reports the result rather than the intention.** The typed graph
matches free `cites` edges from metadata to within 0.016, and closing the specific edge gap
that was diagnosed as its weakness changed nothing. What *did* work is agentic decomposition
— asking a multi-part question in parts — which beats every static arm on relational queries
by +0.250 (p=0.012) at 1.2× the cost, while being worse everywhere else.

The graph's one durable contribution is as a **planner**: `addresses` neighbourhoods reliably
suggest what to ask next, even where traversal is a poor way to gather evidence.

> **Iteration 3 is complete.** The pipeline, the evaluation harness, the typed-graph and
> citation-graph arms, and the agentic planner–critic loop are all built and scored. Full
> write-ups: [Iteration 1](docs/iteration-1-report.md),
> [Iteration 2](docs/iteration-2-report.md), [Iteration 3](docs/iteration-3-report.md).

[![CI](https://github.com/jask7075/Research-Paper-Synthesis-Graph/actions/workflows/ci.yml/badge.svg)](https://github.com/jask7075/Research-Paper-Synthesis-Graph/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Status

**Iteration 3 result: decomposition beats static retrieval on relational queries, and only
there.** 34-query gold set, paired per query, three repeats, `must_cite_recall`:

| | Δ vs `vector_fulltext` | W / L / T | p |
|---|---|---|---|
| all 34 queries | +0.059 | 12 / 9 / 13 | 0.403 |
| **relational (n=14)** | **+0.250** | 8 / 1 / 5 | **0.012** |
| non-relational (n=20) | −0.075 | 4 / 8 / 8 | 0.340 |

The overall figure is a null. The agentic arm is measurably **worse** on non-relational
queries, which is what makes the relational result credible rather than an evidence-volume
effect. Cost is **1.2×**, not the order of magnitude anticipated.

Arm standings (mean of 3 repeats, spread across repeats):

| arm | `must_cite_recall` | spread | judge `coverage` |
|---|---|---|---|
| `agentic` | **0.492** | 0.461 – 0.539 | **3.76** |
| `typed_graph_chunks` | 0.456 | 0.422 – 0.480 | 3.32 |
| `agentic_no_critique` | 0.449 | 0.436 – 0.466 | 3.71 |
| `citation_graph` | 0.444 | 0.431 – 0.456 | 3.47 |
| `vector_fulltext` | 0.433 | 0.412 – 0.446 | 3.56 |

**The mechanism is not the one that was predicted.** Iteration 2 attributed the relational
weakness to missing `undercuts` edges. Three independent measurements refute that account
while the effect itself replicates — see [§4 of the Iteration 3
report](docs/iteration-3-report.md). Decomposition helps relational queries; *why* is open.

Built and verified by running it:

| | |
|---|---|
| corpus | 353 papers (Semantic Scholar metadata), 271 with full text and extracted |
| sections / chunks | 11,020 chunks, SPECTER embeddings (faiss) |
| typed graph | 27,777 nodes, 14,978 edges (Kuzu) |
| edge layers | `addresses` 3,707 · `cites` 2,446 · `undercuts` 119 · `refutes` 21 |
| gold query set | 34 queries — 14 relational, 9 refutation, 6 lookup, 5 open-directions; 10 active as the development set |
| extraction cost | 271 papers, 5,322 calls, $6.72 (`gpt-5.4-nano`, temperature 0, measured) |
| eval cost per query | $0.024 static, $0.028 agentic |
| tests | 214 passing; ruff + mypy clean in CI |

### Judge calibration — one criterion of five is usable

Measured on 34 hand-graded answers with the judge pinned at temperature 0:

| criterion | QWK | trusted |
|---|---|---|
| `coverage` | **+0.76** | **yes** |
| `attribution` | +0.45 | no |
| `refutation_handling` | +0.44 (n=9) | no |
| `synthesis` | +0.63 | on the 34 only — see below |
| `hedging_accuracy` | +0.25 | no |

Three findings behind that table, each recorded because each invalidated something previously
reported:

- **The judge had been sampled at the provider default (1.0) for the project's entire life.**
  Judging the same 34 answers three times with an identical rubric gave per-criterion κ
  spreads up to 0.25, against a 0.26 gap to the trust threshold. Every calibration figure
  before Iteration 3 was one draw presented as a measurement.
- **`attribution` sits at the human ceiling.** The grader agrees with *themselves* at +0.29
  on a blind re-grade; the judge agrees with them at +0.30. There was never a gap for a rubric
  to close, which is why three rubric rewrites moved offset and ranking but never
  agreement-on-level.
- **The two gold sets disagree about which criteria pass.** `synthesis` reads +0.63 on the 34
  and +0.38 on the active 10; `attribution` +0.45 and +0.79. Only `coverage` is indifferent, so
  it is the only judged criterion reported across arms.

### Known limitations, measured

- **The typed graph is a weak retriever.** It matches free `cites` edges from metadata to
  within 0.016 (Iteration 2 §5.2), and quadrupling the `undercuts` layer — 33 → 119 edges, and
  traversal from zero occurrences to six — left `typed_graph_chunks` at 0.383 → 0.383,
  unchanged. Edge coverage was necessary, not sufficient.
- **Cross-paper contradiction detection does not work.** Two prompts, both at 32.5% edge
  precision on a human audit; the revision discarded ~65% of the real edges for no gain.
- **This does not evaluate graph-based global summarisation.** All 34 gold queries carry a
  `must_cite` list, so this measures graph-based *retrieval on citation-grounded queries*.
  Hierarchical community detection, community summaries and map-reduce global search are not
  implemented, and this gold set could not measure them. The negative results above bear on
  the graph as a retriever, not as a summarisation scaffold.
- **Single grader, permanently.** This is a one-person project, so inter-annotator agreement
  is unmeasured and will remain so. Test–retest is reported in its place and is the weaker
  substitute.
- **The repro layer recovers little.** `code_url` 2 of 5 and `dataset_access` 2 of 6 on the
  papers that state one, after a routing fix that took both from zero.

### Carried forward

- **Local inference (§3.3)** — plumbing landed, run deferred: `Qwen2.5-14B-Instruct-AWQ` needs
  ~8.5 GB in 4-bit and CUDA-only kernels, against an 8 GB M2. Config change plus one run when
  hardware is available.
- **A mechanically-splitting arm** would separate "splitting a question helps" from "planning
  helps" — the open mechanism question above.
- **Why the self-critique is worth +0.091** on relational when it adds a *required* paper on
  only 4–7 of 34 queries.
- **Global-sensemaking queries and metrics**, prerequisite for evaluating anything
  GraphRAG-shaped.

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

## Looking at the graph

The graph lives in a single Kuzu file, `data/processed/rpsg.kuzu`. It is derived data —
fully reconstructible from `papers.jsonl` + `extractions.jsonl` — so it is deleted and
rebuilt from empty on every `05_build_stores.py` run rather than merged into.

```bash
python scripts/show_graph.py --stats                      # node/edge counts, hubs
python scripts/show_graph.py "what mitigates barren plateaus?"   # what a query reaches
python scripts/show_graph.py "..." --mermaid              # diagram that renders in markdown
python scripts/show_graph.py --node method:qaoa           # start from a known node
python scripts/show_graph.py --cypher "MATCH (e:Entity) WHERE e.type = 'Hardware' \
  RETURN e.name LIMIT 10"
```

Drawing all 21k nodes produces a hairball. The view worth having is the neighbourhood a
*query* reaches, because that is exactly what `TypedGraphSystem` walks and what its answer
is built from. Seeding, hop count and the node cap are taken from that class rather than
re-chosen, so the picture is of the system under test and not of a different one. Each node
prints with the hop it was reached at and the edge type that got there — which is what
makes it a diagnostic: a traversal landing in the right region and still scoring badly is a
synthesis problem, one that wanders is a seeding problem, and the two need opposite fixes.

**Kuzu allows one process at a time, readers included.** If a build or an eval run is going
in another terminal, opening the database fails. The store translates that error, because
the raw message (`IO exception: Could not set lock on file`) names the symptom rather than
the cause, and the instinctive response to an IO error on a database file is to delete it.
Wait instead; the lock clears on exit. To see what holds it: `lsof data/processed/rpsg.kuzu`.

### Kuzu Explorer (optional)

A browser UI for clicking around and running Cypher. It needs Docker, an image tag matching
the storage format your database was written with, and it takes the same exclusive lock —
so `--cypher` above is usually the faster route.

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/data/processed:/database" \
  -e KUZU_FILE=rpsg.kuzu \
  -e MODE=READ_ONLY \
  kuzudb/explorer:0.11.3          # pin to your kuzu version, not `latest`
```

Then open `localhost:8000`. Mount the *parent directory*: the database is a single file, not
a directory. `READ_ONLY` is deliberate — nothing should be writing to the graph except
`05_build_stores.py`.

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