# Iteration 2 — working notes

Running record of what has been measured and decided. Iteration 1 is closed and its
numbers are in [iteration-1-report.md](iteration-1-report.md); this file only covers work
after that.

---

## 2.1 Persist retrieved evidence in traces — **done**

`traces.jsonl` recorded `len(evidence)` while the full text went to the judge two lines
later. That asymmetry is why `attribution` calibrated at κ=+0.02 (ρ=+0.04, p=0.92): the
judge grades attribution with the retrieved context in its prompt, a human grading from
the answer alone is judging whether claims *look* sourced. Evidence is now persisted, so
the criterion is re-gradable against the same material the judge saw.

Cost: traces grow from ~700 bytes to ~400 KB per 10-query run. `eval/runs/*` is gitignored.

---

## 2.2 Raise `top_k` — **done, locked at 60**

Of the 18 required (query, paper) pairs, 7 sat at ranks 40–124: in the index, ranked
sensibly by the embedder, excluded by an arbitrary cutoff. Retrieval-only sweep:

| top_k | ceiling | evidence chars/query |
|---|---|---|
| 20 | 0.389 | 35,857 |
| 60 | 0.611 | 111,394 |
| 150 | 0.778 | 278,212 |
| 400 | 0.778 | 740,918 |

Plateaus at 150; k=400 costs 20.7× the context of k=20 and finds nothing more.

Scored k=60 against the k=20 baseline on the same gold set:

| metric | k=20 | k=60 |
|---|---|---|
| `must_cite_recall` | 0.217 | **0.367** |
| `citation_precision` | 0.150 | **0.210** |
| `key_claim_source_recall` | 0.167 | **0.317** |
| judge `coverage` (calibrated) | 2.40 | **2.90** |

Precision rose alongside recall, contrary to the prediction that more evidence would give
the model more papers to cite wrongly. It cited more (2.6 → 3.4 papers) and more
accurately, so the added chunks were relevant rather than noise. Cost $0.19 → $0.50.

60 rather than 150 because the synthesizer converts only ~60% of the retrieval ceiling
into citations (0.611 available, 0.367 cited); depth past this buys progressively less.

**This is now the number the typed graph has to beat**, and it came from changing one
integer. Establishing it before building the graph is the whole point — otherwise the
graph inherits credit for a config change.

### Open: the conversion gap

40% of required papers the synthesizer *was shown* never reach an answer. Not a retrieval
problem, and no `top_k` fixes it. Two untested hypotheses, both checkable from the k=60
traces now that evidence is persisted: handle resolution silently dropping malformed
citations (`_resolve_handles` discards any handle that was never issued), and position
effects over a 60-chunk prompt.

---

## 2.3 Corpus / gold mismatch — **diagnosed, no index change**

The plan item read *"5 required papers are undiscoverable"*. That was measured at
`top_k=20` and does not survive a deeper probe.

**It is 4 (query, paper) pairs across 3 papers, and only one paper fails universally.**

| paper | required by | rank at depth 400 |
|---|---|---|
| Negre (D-Wave communities) | 5 queries | 44, 56, 124, miss, miss |
| Newman (modularity matrix) | 2 queries | 9, miss |
| Hamming (error-correcting codes) | 1 query | miss |

The same paper is findable or not *depending on the query*. `rel-t02` and `rel-t03`
account for 3 of the 4 misses; both are "quantum + X" phrasings that pull the 234-paper
quantum mass ahead of their actual targets.

**Dilution confirmed.** Ranking the same queries against an 18-paper topic subset — built
by keyword-matching titles and abstracts, deliberately *not* from the gold set, which
would have been circular:

| query | target | rank in full index | rank in subset |
|---|---|---|---|
| rel-t02 | Negre | 142 | **2** |
| rel-t02 | Newman | 218 | **4** |
| rel-t03 | Negre | 249 | **7** |
| rel-t03 | Hamming | 234 | **4** |

Every one lands in the top 7 once 335 off-topic papers are removed. The embedder was never
wrong about relevance; it was outvoted.

**Decision: do not scope the index.** Filtering the corpus to papers matching the gold set
is fitting retrieval to the test — it would produce a large number that means little,
because a deployed system does not know in advance which 18 papers matter.

The result is more useful read the other way: narrowing the candidate set *before* ranking
recovers every miss, and doing that without being told which papers to look at is exactly
what typed-graph retrieval is for. **The subset ranks above are the target the Iteration 2
retriever has to hit on its own.** This measures the headroom the thesis claims, and it is
real.

Widening the gold set instead is defensible but expensive, and it dodges the finding.

---

## 2.4 `Hardware` routing — **fix landed, re-extraction pending**

`NodeType.HARDWARE` was requested in exactly one of eleven section types — `appendix`,
2.9% of chunks (316 / 10,993) — and `_REPRO_HINT`, which carries the "capture vendor and
`qubit_count` exactly" instruction, fired only there too. Result: **12 `Hardware` nodes
across 8 of 270 papers**, in a corpus where the large majority state a device or a qubit
count.

Structurally identical to the `Limitation` gap in Iteration 1: a node type the routing
never asks for is absent from the graph, with no error anywhere, and the extractor, schema
and prompt each correct in isolation.

**Fix.** `Hardware` added to `method` (experimental setup), `results` (run configuration)
and `availability`; `REQUIRES` edges enabled alongside. `_REPRO_HINT` now fires wherever
`Hardware` is askable rather than on `appendix` alone — gated on the routing table, so the
two cannot drift apart again.

`tests/test_prompts.py` pins this, including `test_every_node_type_is_reachable_from_some_
section`, which fails if any extractable type becomes unreachable. Verified the new tests
fail against the old routing.

**Not yet measured.** The fix changes prompts only; the existing graph was extracted under
the old routing. A re-extraction of 270 papers at the measured ~2.8¢/paper is roughly
$7.60 and about an hour. Until it runs, the `Hardware` counts above still stand and
`repro_gold` stays blocked — gold for a component that is almost entirely missing measures
the bug rather than the component.

**Order matters:** re-extract *before* any extraction-precision audit. The fix changes the
prompt for `method`, `results` and `availability`, so an audit sampled beforehand would be
invalidated by it.
---

## 2.15 `repro_gold` — schema and scorer done, gold partial

**Schema** (`rpsg.eval.repro_gold`) expresses three states, where the old file had one:

    a value          the paper states it     -> the system must find it
    "not_reported"   the paper is silent     -> the system must stay silent
    null             gold not established    -> skipped, not scored

The middle state is what made the previous `repro_gold.jsonl` unscoreable: `null` meant
both "the paper says nothing" and "nobody has checked", so a system inventing a qubit
count could not be told apart from one correctly reporting none. Same defect as the
metric defaults fixed in 2.4, corrected the same way.

Six outcomes rather than one accuracy figure — `correct`, `wrong`, `missed`,
`hallucinated`, `correct_absence`, `skipped`. `missed` and `hallucinated` are opposite
failures and a single number merges them.

Field set is quantum-shaped (`quantum_vendor`, `device_name`, `qubit_count` lead) because
the corpus is: `qubit_count` appears in 151 of 242 `Hardware` nodes against `gpu_type` in
107. The scorer also absorbs an extraction quirk — the model writes the literal string
`"unknown"` into fields it cannot answer, so `"unknown"`, `"n/a"`, `""` and `None` all
read as silence and score `missed` rather than `wrong`.

`scripts/author_repro_gold.py --show` prints a paper's passages and deliberately **not**
its extraction. Confirming the system's own output would drive accuracy toward 1.0 by
construction — the circularity `author_gold.py` avoids by using BM25 rather than the
system's retriever.

**First audit — 20 papers, 25 scoreable fields:**

| outcome | n |
|---|---|
| correct | 2 |
| correct_absence | 17 |
| wrong | 2 |
| missed | 4 |
| hallucinated | **0** |

**76% accuracy, but 17 of the 19 correct answers are correct silences.** On the 8 fields
where a paper actually states something the system got 2 — **25%**. Quote both figures;
the aggregate alone would make a system that mostly says nothing look strong.

Zero hallucinations across 17 fields the papers are silent on is a real result: the
extractor fills unanswerable fields with `"unknown"` rather than with plausible fiction.

**Gold is 25 of 140 fields (18%).** 12 of 20 records are still entirely null.

Two gold-design rules were settled by hand and need applying consistently to the rest:
a survey's cited hardware **does** count (`quantum_vendor: IBM` on a review paper), and a
GPU vendor without a model number **does** fill `device_name` ("NVIDIA and AMD GPU").

## QUEUED: prompt change awaiting the next re-extraction

`Hardware` added to the **`abstract`** routing entry. Not yet reflected in any graph — the
extraction on disk predates it.

The audit found it. `91c10ab4c5` scored **0 of 3** on Google / Sycamore / 23 qubits with
the abstract reading "the Google Sycamore superconducting qubit quantum processor" and
"over 23 qubits". The 2.4 routing fix reached `method`, `results` and `availability` and
took the corpus from 12 `Hardware` nodes to 242 — and still could not see the single
clearest device statement in the sample, because it lives in the abstract.

The lesson generalises: routing has to follow where papers *actually* state things, not
where a reader would expect the detail to live.

**Cost to realise: ~$6.30 and ~70 minutes** for 270 papers. Batch it with any other prompt
change rather than re-extracting for this alone. Until it runs, the `Hardware` counts and
the 25%-on-stated-fields figure both stand as measured under the old routing.

---

## 2.6–2.12 — entity resolution and the typed-graph arm

### What the plan assumed, and what measurement said

| assumption | outcome |
|---|---|
| ER merges ~1.6% of entity nodes | **deterministic: 0.20%**, hybrid: **3.69%** |
| Duplicates are a naming-verbosity problem | **no** — median name 4.0 → 3.0 words changed merges 8 → 7 |
| Semantic merging = a threshold over embedding similarity | **no viable threshold exists** (below) |
| 2.6 gates 2.9: typed retrieval over duplicates is unmeasurable | **loosened, not removed** — 2.9 ran anyway and the duplicates were not the limiting factor |

### 2.8 Why an embedding threshold cannot work here

Nearest-neighbour cosine over 6,193 distinct `Method` nodes (SPECTER):

| band | nodes | | band | nodes |
|---|---|---|---|---|
| 0.95–1.00 | 2,052 | | 0.80–0.85 | 189 |
| 0.90–0.95 | 2,742 | | 0.70–0.80 | 10 |
| 0.85–0.90 | 1,200 | | **below 0.70** | **0** |

Nothing falls below 0.70, so there is no negative class. Worse, similarity does not rank
true pairs above false ones: `"Gradient-free classical optimization for QAOA parameters"`
vs `"Gradient-based ..."` scores **0.993**, while the genuine duplicate
`"Quantum Multi-value Decision Diagram (QMDD)"` vs `"Quantum Multi-valued ..."` scores
0.976. Negation is the classic embedding failure and it dominates here.

### 2.8 Hybrid: embeddings for recall, a model for the decision

Embeddings reduce ~19M possible pairs to 3,487 candidates above 0.95; `gpt-5.4-nano` then
reads each pair. **909 ids merged (3.69%)** — 18× the deterministic yield, for **$0.30**.

The model rejected **67%** of what embeddings proposed, and the rejections are the
argument for the architecture. All of these scored 0.999–1.000 cosine:

    "Batch Relaxing (Algorithm 1)"           != "Batch Relaxing (Algorithm 2)"
    "Quantum hardware configuration (12...)" != "Quantum hardware configuration (6...)"
    "First-order optimization for..."        != "Second-order optimization for..."

Verdicts are cached by name pair, which recovers most of the determinism this approach
otherwise gives up.

### 2.10 Hops, measured

| hops | recall | evidence chars/query |
|---|---|---|
| 1 | 0.389 | 4,573 |
| **2** | **0.556** | 9,184 |
| 3 | 0.556 | 13,326 |

The second hop is the entire gain; the third adds volume only. Neither more seeds (12→24)
nor a larger node cap (60→300) moved recall at all.

**A prediction that failed:** removing `Claim` from the seed types, on the theory that
sentence-shaped nodes crowd out the concepts a query asks to enumerate, dropped recall
**0.556 → 0.333**. Claims are sentences and so are queries — they match far better than
3-word concept names and are the doorway into a neighbourhood.

### 2.12 Three-arm comparison — the typed graph loses

| metric | abstract | fulltext | typed_graph |
|---|---|---|---|
| must_cite_recall | 0.367 | **0.367** | **0.183** |
| citation_precision | **0.233** | 0.210 | 0.175 |
| key_claim_source_recall | **0.417** | 0.317 | 0.183 |
| refutation_surfaced | 0.000 | **0.333** | **0.333** |
| judge coverage *(calibrated)* | 1.80 | **2.90** | 2.00 |
| judge synthesis *(calibrated)* | 2.20 | **3.10** | 2.30 |
| cost per 10-query run | $0.08 | $0.50 | **$0.07** |

On both calibrated criteria the typed graph sits barely above abstract-only and clearly
below full-text.

### The diagnosis: conversion, not traversal

| arm | retrieval ceiling | cited | conversion |
|---|---|---|---|
| vector_fulltext | 0.611 | 0.367 | **60%** |
| typed_graph | 0.556 | 0.183 | **33%** |

Traversal found nearly as many required papers as vector retrieval and cited half as many.
The failure is downstream of the graph. The probable cause is the evidence unit: node
evidence is one-sentence quotes stripped of context, against ~1,800 characters of
connected prose per chunk. Shorter answers (1,714 vs 2,382 chars) and fewer citations
(3.0 vs 3.4) fit that reading.

It does win where typed edges should help — `refutation_surfaced` **0.333**, matching
full-text and beating abstract-only's 0.000 — and it is the cheapest arm by 7×.

**Recorded as a negative result for the thesis as stated**, on this corpus at this
configuration. The next experiment is to hand the synthesizer each node's *source chunk*
rather than its quote, making the graph a retrieval router over the same evidence units
the vector arm uses. That isolates traversal from evidence formatting. Retuning until the
graph wins would not be a result.
