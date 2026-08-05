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
