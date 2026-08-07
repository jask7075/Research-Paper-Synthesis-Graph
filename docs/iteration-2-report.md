# RPSG — Iteration 2 Report

**Status: core experiment complete.** Branch `iteration-2`, 8 commits, 2026-08-06.
**Thesis under test:** typed-graph retrieval beats vector and citation-graph baselines,
broken out by query type.

Iteration 1 and its numbers are in [iteration-1-report.md](iteration-1-report.md). Running
decisions are in [iteration-2-notes.md](iteration-2-notes.md); this is the standalone
account.

---

## 1. Headline

**Traversal is a better paper selector than similarity search. Node evidence is not a
usable synthesis input.** Both halves are measured, and separating them took two runs of
the same system.

| | value |
|---|---|
| Best arm on citation metrics | `typed_graph_chunks` — `must_cite_recall` **0.483** |
| Vector baseline it beats | `vector_fulltext` — 0.367 (+32%) |
| Same traversal with node quotes | `typed_graph` — 0.183 |
| Calibrated criteria | coverage **ties**, synthesis **loses** |
| Corpus | 353 papers, 270 extracted, 10,993 chunks |
| Graph | 26,879 nodes, 11,975 edges |
| Tests | 168 passing; ruff + mypy clean |

The result is *not* a clean win. It is a win on citation grounding and a tie-or-loss on the
two judge criteria that survived calibration, at n=10 queries.

---

## 2. The two typed-graph experiments

Both use the **same graph, same seeds, same 2-hop traversal, same synthesis prompt**. The
only difference is what text reaches the synthesizer. Running them as a pair is what turned
an unexplained failure into a diagnosis.

### 2A. `typed_graph` — node quotes as evidence

Embed the query, match against **node names**, walk typed edges two hops, and hand the
synthesizer each reached node's `evidence` quote:

```
[P3] Method: Layerwise VQE training via addresses
     "layerwise training reduces the variance of the gradient at shallow depth"
```

The intent was to make the evidence unit an *extracted assertion* rather than a passage —
the graph's native currency. One-sentence quotes, ~9,200 characters per query.

### 2B. `typed_graph_chunks` — the graph as a router

Identical traversal. But instead of quotes, take the **set of papers** the traversal
reached and pull chunks from those papers, ranked by query similarity:

```
[P3] (method) Training the ansatz layerwise rather than end-to-end changes the
     gradient variance profile. We observe ... [~1,800 chars of connected prose]
```

The graph selects *which papers*; chunks supply *the text*. This puts the graph arm on the
same evidence unit as the vector arm, so the comparison becomes traversal-vs-similarity as
a **paper selector**, with formatting held constant.

Routing is done at **paper** granularity rather than node→chunk. Mapping a node's quote
back to its source chunk by substring succeeds for only **57%** of nodes — chunking
normalises whitespace and some quotes straddle chunk boundaries — and building the
experiment on a 57%-reliable mapping would have introduced a second failure mode into a
test designed to isolate one.

---

## 3. Results — four arms, same gold set

| metric | abstract | fulltext | **tg_quotes** | **tg_routed** |
|---|---|---|---|---|
| `must_cite_recall` | 0.367 | 0.367 | 0.183 | **0.483** |
| `citation_precision` | 0.233 | 0.210 | 0.175 | **0.445** |
| `key_claim_source_recall` | 0.417 | 0.317 | 0.183 | **0.533** |
| `refutation_surfaced` | 0.000 | 0.333 | 0.333 | **0.667** |
| judge `coverage` *(calibrated)* | 1.80 | **2.90** | 2.00 | **2.90** |
| judge `synthesis` *(calibrated)* | 2.20 | **3.10** | 2.30 | 2.70 |
| judge `attribution` *(untrusted)* | 2.70 | 2.80 | 2.50 | 3.10 |
| answer chars | 1,746 | 2,382 | 1,714 | 2,043 |
| papers cited | 3.2 | 3.4 | 3.0 | 3.2 |
| cost / 10 queries | $0.08 | $0.50 | **$0.07** | $0.30 |

### The improvement, isolated

Changing only the evidence unit, with traversal held fixed:

| metric | quotes | chunks | change |
|---|---|---|---|
| `must_cite_recall` | 0.183 | 0.483 | **×2.6** |
| `citation_precision` | 0.175 | 0.445 | **×2.5** |
| `key_claim_source_recall` | 0.183 | 0.533 | **×2.9** |
| `refutation_surfaced` | 0.333 | 0.667 | **×2.0** |
| judge `coverage` | 2.00 | 2.90 | +0.90 |
| judge `synthesis` | 2.30 | 2.70 | +0.40 |

Every metric roughly tripled. The graph did not get better at finding papers — it selected
the *same* papers in both runs. The synthesizer simply could not build a citable answer
from disconnected one-sentence fragments.

### Conversion, which is what the first run actually measured

| arm | retrieval ceiling | cited | conversion |
|---|---|---|---|
| `vector_fulltext` | 0.611 | 0.367 | 60% |
| `typed_graph` (quotes) | 0.556 | 0.183 | **33%** |
| `typed_graph_chunks` | 0.556 | 0.483 | **87%** |

Traversal reached slightly fewer required papers than vector search and converted far more
of them once the evidence was usable.

### Where the graph does not win

Judge `coverage` **ties** full-text at 2.90 and `synthesis` **loses**, 2.70 against 3.10.
These are the two criteria that passed calibration at n=20, so they carry the most weight.
The graph arm grounds claims better and covers the same ground; the judge finds full-text's
answers better integrated.

`refutation_surfaced` 0.667 is the largest relative margin in the table and the one typed
edges were predicted to win — but it rests on **3 queries**.

---

## 4. Retrieval configuration, measured not chosen

### `top_k` (vector arm)

| top_k | ceiling | evidence chars |
|---|---|---|
| 20 | 0.389 | 35,857 |
| **60** | **0.611** | 111,394 |
| 150 | 0.778 | 278,212 |
| 400 | 0.778 | 740,918 |

Locked at 60. Scoring it moved `must_cite_recall` 0.217 → 0.367 and judge `coverage`
2.40 → 2.90 — **from changing one integer**. Establishing that before building the graph is
why the graph cannot claim credit for it.

Precision rose alongside recall, contrary to prediction: the model cited more papers
(2.6 → 3.4) *and* more accurately, so the added chunks were relevant rather than noise.

### Hops (graph arm)

| hops | recall | evidence chars |
|---|---|---|
| 1 | 0.389 | 4,573 |
| **2** | **0.556** | 9,184 |
| 3 | 0.556 | 13,326 |

The second hop is the entire gain. Neither more seeds (12→24) nor a larger node cap
(60→300) moved recall at all.

---

## 5. Challenges since Iteration 1

Each is symptom → cause → fix → what it cost.

### 5.1 The graph was never reset between rebuilds

Stage 05 opened the existing Kuzu database and `MERGE`d into it, so every rebuild
**accumulated**: nodes from an earlier extraction survived even when the current one no
longer produced them. The graph on disk had been a union of extraction generations rather
than a picture of one.

It surfaced as a crash, not as wrong data — merging a fresh 25k-node extraction onto a
64 MB database exhausted Kuzu's buffer pool and stage 05 died after 602s with the vector
index already rebuilt. **Fix:** clear the database and its `.wal`/`.shadow` sidecars first.
The graph is derived data, reconstructible from `papers.jsonl` + `extractions.jsonl`.

### 5.2 `Hardware` was unreachable from every section a paper states it in

`NodeType.HARDWARE` was requested in one of eleven section types — `appendix`, 2.9% of
chunks — and `_REPRO_HINT` fired only there. The corpus produced **12 Hardware nodes across
8 of 270 papers** while the large majority state a device or qubit count.

Structurally identical to the `Limitation` gap in Iteration 1: a node type the routing
never asks for is absent with no error anywhere, because extractor, schema and prompt are
each correct in isolation.

**Fix:** `Hardware` added to `method`, `results` and `availability`; the hint gated on the
routing table rather than a section name so the two cannot drift apart. **Effect: 12 → 242
nodes, 8 → 98 papers.** Cost a $6.32 re-extraction.

**And it was still incomplete.** The `repro_gold` audit found a paper scoring **0 of 3** on
Google / Sycamore / 23 qubits with all three stated plainly — in the **abstract**, which the
fix had not covered. Now routed there too, queued for the next re-extraction. The lesson
generalises: routing must follow where papers *actually* state things, not where a reader
expects the detail to live.

### 5.3 Four metrics credited the system for questions never asked

`refutation_surfaced` returned `1.0` when a query had no `known_refutations`. Seven of ten
gold queries qualify, so the reported aggregate was **0.700** while all three queries that
*did* encode a contradiction scored **0.00**.

The same defect sat in `must_cite_recall`, `citation_precision` and
`key_claim_source_recall`. Fixing one and leaving three would have deferred the next
inflated aggregate to the next gold set.

**Fix:** `None` when the *gold* has nothing to measure; a score when the *answer* is
deficient. Reports now print per-metric `n` ("3 of 10") — without it, a mean over three
queries is indistinguishable from a mean over ten, which is how 0.700 reached the
Iteration 1 report.

### 5.4 `attribution` could not be calibrated, for an instrument reason

`traces.jsonl` recorded `len(evidence)` while the full text went to the judge two lines
later. The judge scores attribution with the retrieved context in its prompt; a human
grading from the answer alone judges whether claims *look* sourced. Calibration then
measured the asymmetry: **κ=+0.02, ρ=+0.04, p=0.92** — no relationship at all.

**Fix:** persist the evidence. Not yet re-graded, so the criterion remains untrusted.

### 5.5 Growing calibration n reversed a verdict

Scoring `vector_abstract` doubled graded pairs from 10 to 20. Three of five verdicts moved:

| criterion | n=10 | n=20 |
|---|---|---|
| `coverage` | +0.68 | **+0.74** trusted |
| `synthesis` | +0.53 | **+0.69** now trusted |
| `attribution` | +0.02 | +0.41 still failing |
| `hedging_accuracy` | +0.67 | **−0.19 LOST** |

At n=10, `hedging_accuracy` would have been published as trusted. Negative kappa is
systematic disagreement, not noise: abstract-only answers hedge heavily, which the human
scored as calibrated honesty (4.8 mean) and the judge read as under-confidence (3.7).

### 5.6 The abstract-only floor is not a floor

`vector_abstract` **ties** full-text on `must_cite_recall` (0.367) and beats it on the other
two citation metrics, while scoring 1.80 against 2.90 on coverage. One abstract per paper
means citing an abstract *is* citing the paper, so citation metrics reward paper
identification rather than having read anything.

**Consequence:** `must_cite_recall` cannot headline a system comparison. `coverage` and
`synthesis` are the criteria that both separate systems and survive calibration.

### 5.7 I shipped an over-merge in entity resolution

`normalize()` deleted every parenthetical, on the theory that they are acronym glosses.
Applied to the corpus it merged

    "AlphaQubit 2 (RT) complexity"  with  "AlphaQubit 2 (full) complexity"
    "Eq. (F11) ... subgraph (c)"    with  "Eq. (F10) ... subgraph (b)"

where the parenthesis carried the only distinguishing content — exactly the failure the
module exists to prevent. It shipped because the sample I inspected happened to contain one
legitimate merge (an OCR ligature, `spoofing`/`spooﬁng`) beside two wrong ones.

**Fix:** flatten parentheses like any other punctuation. **Merges dropped 369 (1.50%) → 50
(0.20%): about 86% of what the first version merged was wrong.**

### 5.8 A `libomp` segfault, already documented, hit again

Building a faiss index after loading torch segfaults — three `libomp.dylib` copies in the
environment. Iteration 1 diagnosed this and fixed it in `FaissVectorStore`; the new
candidate-pair search reintroduced it. **Fix:** pin faiss to one thread immediately before
searching, as the vector store already did.

### 5.9 Serial adjudication would have taken an hour and lost everything

3,487 pairs at ~1s each, with the cache written only at the end — the first run timed out
and discarded every verdict already paid for. **Fix:** thread pool plus checkpointing every
200 verdicts, the same shape stage 04 already used.

---

## 6. Assumptions the plan made, and what measurement said

| assumption | outcome |
|---|---|
| ER merges ~1.6% of entity nodes | deterministic **0.20%**, hybrid **3.69%** |
| Duplicates are a naming-verbosity problem | **no** — median name 4.0 → 3.0 words moved merges 8 → 7 |
| Semantic merging = threshold over embedding similarity | **no viable threshold exists** |
| 2.6 gates 2.9 — typed retrieval over duplicates is unmeasurable | **false** — duplicates were not the limiting factor; evidence formatting was |
| Claim-shaped seeds crowd out concept nodes | **backwards** — removing `Claim` dropped recall 0.556 → 0.333 |
| 5 required papers are undiscoverable | **4 pairs across 3 papers**; only one fails universally |

### Why an embedding threshold cannot work here

Nearest-neighbour cosine over 6,193 distinct `Method` nodes never falls below **0.70** — no
negative class exists. And similarity does not rank true pairs above false ones:

    0.993  "Gradient-free classical optimization for QAOA parameters"
        vs "Gradient-based classical optimization for QAOA parameters"   <- opposite
    0.976  "Quantum Multi-value Decision Diagram (QMDD)"
        vs "Quantum Multi-valued Decision Diagram (QMDD)"                <- same

**The hybrid that works:** embeddings reduce ~19M pairs to 3,487 candidates; a model reads
each pair. **909 ids merged (3.69%)** for **$0.30**, 18× the deterministic yield. The model
rejected **67%** of what embeddings proposed, including pairs at 0.999 cosine differing by
an algorithm number, a qubit count, or first- versus second-order.

---

## 7. What Iteration 2 does not claim

- **n=10 queries.** A 0.483-vs-0.367 gap on ten items is consistent with the thesis, not
  established by it. `refutation_surfaced` rests on **3**.
- **No per-query-type breakdown is reported.** Cells would hold 1–4 queries.
- **No citation-graph ablation arm** (2.11). The three-arm comparison is
  abstract / full-text / typed-graph; untyped-edge retrieval was not built.
- **Nothing about `attribution`, `hedging_accuracy` or `refutation_handling`** — those judge
  scores exist but did not clear calibration.
- **The queued re-extraction has not run.** `Hardware`-in-abstract routing and the canonical
  naming prompt are committed but unrealised; every extraction figure here predates them.
- **`repro_gold` is 25 of 140 fields (18%).** Its 76% accuracy is 25% on fields a paper
  actually states — 17 of 19 correct answers are correct silences.

---

## 8. Carried into Iteration 3

1. **Re-grade `attribution`** against persisted evidence — 20 judgements, decides whether a
   third criterion becomes trustworthy.
2. **Re-extraction** (~$6.30) to realise the queued prompt changes, then the 2.16 precision
   audit and the remaining `repro_gold` fields.
3. **Grow the gold set.** n=10 cannot separate systems, and this iteration's headline
   depends on exactly that separation.
4. **Citation-graph ablation** (2.11), the missing third arm.
5. **Cross-paper contradiction** (2.17) — `refutes`/`undercuts` remain 8 and 48.
6. **`Chunk.id` uniqueness** (2.18).