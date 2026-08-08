# Iteration 2 — technical report

Supersedes the previous version of this file, whose headline claim (`typed_graph_chunks`
beating vector retrieval at 0.483) did not survive the re-extraction in §3.2. The prior
text is recoverable in git at `e77f5f2`.

Plain-language companion: [iteration-2-report-plain.md](iteration-2-report-plain.md).
Running working notes: [iteration-2-notes.md](iteration-2-notes.md).

---

## 1. Result

Typed-graph retrieval over a 95%-precision extracted graph does not improve required-paper
recall over a vector baseline, nor over a citation graph built from free Semantic Scholar
metadata. Both alternative explanations for that outcome — poor extraction, or graph
retrieval being unsuited to this corpus — are closed off by ablation rather than argued
away.

| arm | `must_cite_recall` | `citation_precision` | `key_claim_source_recall` |
|---|---|---|---|
| `vector_fulltext` | **0.456** | 0.221 | **0.362** |
| `typed_graph_chunks` | 0.377 | **0.235** | 0.356 |
| `vector_abstract` | 0.324 | 0.168 | 0.328 |
| `typed_graph` | 0.299 | 0.195 | 0.333 |

*n = 34 gold queries, `top_k = 60`, graph with both entity-resolution tiers applied.*

Secondary findings, each from a dedicated measurement rather than inference:

- **Entity resolution contributes nothing measurable.** 944 model-adjudicated merges,
  ablated directly: two of three deterministic metrics identical (§5.1).
- **The typed edge layer is worth ~nothing over free citation edges**, which match it to
  within 0.016 on every metric (§5.2).
- **Extraction precision is 95.0%**, rising monotonically with confidence (§5.3). The graph
  content is not the problem.
- **No arm surfaces contradictions**: 1 of 9 refutation queries (§4.3).
- **The judge is trustworthy on 3 of 5 criteria** (§6).

---

## 2. Setup

**Corpus.** 354 papers with Tier-A metadata, 271 with parsed full text, 11,020 chunks.
Graph: 23,301 extracted nodes across 8 types and 11,186 typed edges, plus 2,298 Tier-A
nodes and 2,446 `cites` edges.

**Arms.** All share the synthesis prompt and citation-handle scheme, so retrieval is the
only variable under test.

| arm | retrieval | evidence unit |
|---|---|---|
| `vector_fulltext` | chunk similarity, `top_k=60` | chunks |
| `vector_abstract` | chunk similarity over abstracts only | chunks |
| `typed_graph` | seed on node names, 2-hop typed traversal | node quotes |
| `typed_graph_chunks` | same traversal, routed to papers | chunks |
| `citation_graph` | seed on titles, 2-hop `cites` traversal | chunks |
| `citation_graph_seeded` | seed by vector retrieval, then `cites` | chunks |

**Metrics.** Four deterministic (`must_cite_recall`, `citation_precision`,
`key_claim_source_recall`, `refutation_surfaced`) and five judge criteria. All
deterministic metrics return `None` when the gold has nothing to measure; every report
carries a per-metric `n` (§3.1).

**Gold.** 34 queries — 14 relational, 9 refutation, 6 lookup, 5 open-directions. Grounded
via BM25 over raw section text, never the SPECTER index: selecting `must_cite` with the
system's own retriever would drive recall toward 1.0 by construction. The active set was
reverted to the 10 thesis-derived queries after measurement, for iteration speed; the 34
are preserved in `eval/gold/queries.full34.jsonl` and remain the basis of every claim here.

---

## 3. Measurement corrections (Gate 0)

Applied before any graph work, on the principle that a win against a misconfigured
baseline is uninterpretable.

### 3.1 Metric semantics

`refutation_surfaced` returned 1.0 for queries with no `known_refutations`. Seven of ten
gold queries had none, so the reported aggregate was 0.700 while **every query that
encoded a contradiction scored 0.00**. All four metrics now return `None` when the gold has
nothing to measure; a deficient *answer* still scores.

`citation_precision` deliberately retains 1.0 for an uncited answer: an answer citing
nothing has made no false attribution, which is the correct reading of precision. The
deficiency is caught by recall instead.

### 3.2 Retrieval configuration

Of 18 required (query, paper) pairs, 7 sat at ranks 40–124 under `top_k=20`.

| `top_k` | recall ceiling | evidence chars/query |
|---|---|---|
| 20 | 0.389 | 35,857 |
| 60 | 0.611 | 111,394 |
| 150 | 0.778 | 278,212 |

Locked at 60. Run first deliberately: had this recovered most of the out-of-range pairs,
the honest headline would have been *"the baseline was under-retrieving"*, and any typed
win would have been confounded with a one-line configuration change.

### 3.3 Evidence persistence

`traces.jsonl` recorded `evidence_chars` while the full text went to the judge. That
asymmetry is why `attribution` calibrated at κ=+0.02 (ρ=+0.04, p=0.92) in Iteration 1: the
judge graded with retrieved context in its prompt, while a human grader could only judge
whether claims *looked* sourced. Evidence is now persisted; post-fix κ=+0.34 (§6).

### 3.4 Corpus/gold mismatch — diagnosed, no change

4 (query, paper) pairs across 3 papers were unreachable at any `top_k`. All rank top-7
within an 18-paper topic subset, so the cause is dilution rather than a retrieval defect.
The available fix — restricting the index to the topic under test — was rejected as fitting
retrieval to the test. Closed as diagnosed.

*(A first attempt at this diagnosis compared paper-rank against chunk-rank, different
units, and was redone with a non-circular topic subset.)*

---

## 4. Retrieval results

### 4.1 Sample size

The arm ordering changed between n=10 and n=34.

| arm | n=10 | n=34 |
|---|---|---|
| `vector_fulltext` | 0.417 | 0.456 |
| `typed_graph_chunks` | 0.383 | 0.377 |
| `vector_abstract` | 0.400 | 0.324 |
| `typed_graph` | 0.333 | 0.299 |

`vector_abstract` moved from second to third. Every n=10 result in this project should be
read against a ±0.10 noise floor.

### 4.2 By query type (n=34)

| type | n | `vector_fulltext` | `typed_graph_chunks` | `typed_graph` |
|---|---|---|---|---|
| lookup | 6 | 0.500 | **0.667** | 0.500 |
| open-directions | 5 | **0.500** | 0.400 | 0.100 |
| refutation | 9 | **0.556** | 0.444 | 0.333 |
| relational | 14 | **0.357** | 0.202 | 0.262 |

The typed graph is **last on relational queries** — the 14-query majority and precisely the
type the design targets. `gold_schema.py` states the intent explicitly: *"over-sample
relational/refutation: that is where the typed graph earns its complexity."* It leads only
on `lookup`, the type nobody claimed graphs would help with.

### 4.3 Contradiction surfacing

| arm | queries surfacing the contradiction |
|---|---|
| `vector_fulltext` | 1 / 9 |
| `vector_abstract` | 1 / 9 |
| `typed_graph` | 1 / 9 |
| `typed_graph_chunks` | 2 / 9 |

At n=3 (Iteration 1 gold) this read 0.667. The cause is structural: the graph holds 8
`refutes` and 25 `undercuts` edges, because per-paper extraction can only observe
contradictions a paper states about itself. Attempted and rejected in §8.2.

### 4.4 Domain split

| | thesis-domain (10) | out-of-domain (24) |
|---|---|---|
| `vector_fulltext` | 0.367 | **0.493** |
| `typed_graph_chunks` | 0.383 | 0.375 |

Vector retrieval's advantage comes entirely from out-of-domain queries; it falls to parity
where the corpus is dense in the query's topic, while the typed arm is flat and leads
in-domain on `citation_precision` (0.300 vs 0.233). Reported as a pre-specified subgroup,
not as the primary result.

---

## 5. Ablations

### 5.1 Entity resolution

Rebuilt with the semantic tier disabled (29 merged ids vs 973), everything else identical,
`typed_graph_chunks` re-run.

| metric | 973 merges | 29 merges |
|---|---|---|
| `must_cite_recall` | 0.383 | 0.383 |
| `key_claim_source_recall` | 0.433 | 0.433 |
| `citation_precision` | 0.333 | 0.350 |

Two of three identical. 944 merges — 3,859 pairs adjudicated at 27.9% acceptance — produce
no measurable retrieval effect. This also reassigns a −0.100 drop previously attributed to
the merges: it belonged to the re-extraction.

The planning assumption that *"typed retrieval over a graph with duplicate Method nodes is
unmeasurable"* is not supported by this measurement.

### 5.2 Citation-graph ablation

`cites` edges from S2 metadata; seeding shape, hop budget, chunk routing and synthesis
prompt held constant.

| arm | `must_cite_recall` | `citation_precision` | `key_claim_source_recall` |
|---|---|---|---|
| `typed_graph_chunks` | 0.383 | 0.300 | 0.433 |
| `citation_graph` | 0.367 | 0.287 | 0.417 |
| `vector_fulltext` | 0.367 | 0.233 | 0.317 |
| `citation_graph_seeded` | 0.317 | 0.193 | 0.267 |

*(thesis-10 subset, current graph)*

Free citation edges match the extracted typed layer within 0.016 on every metric. The
`_seeded` variant — vector retrieval followed by citation expansion — scores **below**
vector retrieval alone: expansion dilutes a good seed set rather than enriching it.

**Design note on the cap.** `max_nodes` was set to 30, not the typed arm's 150. In the
typed graph a node is an entity and 150 entities route to ~10 papers (measured mean 10.1,
max 28); here a node *is* a paper, so 150 would have meant 55 papers with chunks and 201k
chars of evidence against the typed arm's 61k. Copying the number would have let this arm
win on evidence volume rather than structure. At 30 both sit alongside the vector baseline
(95k and 116k chars against 111k).

### 5.3 Extraction precision audit

60 nodes, 20 per confidence band, a third of each band reserved for reproducibility types.
Each labelled against its own evidence quote — never against a second system's output,
which would measure agreement rather than correctness.

| band | precision |
|---|---|
| 0.65–0.75 | 85.0% (17/20) |
| 0.75–0.85 | 100.0% (20/20) |
| 0.85–1.01 | 100.0% (20/20) |
| **overall** | **95.0% (57/60)** |

By type: `Claim`, `Dataset`, `Hardware`, `Software`, `Problem` all 100%; `Method` 92.3%,
`Limitation` 90.9%, `ReproducibilityArtifact` 75.0% (n=4).

Precision rises with confidence, so the score is informative and the 0.65 gate does work at
the bottom band. Raising it to 0.75 would yield 100% precision at some recall cost.

**Scope.** Sub-gate nodes are absent from `extractions.jsonl`, so this measures what
survived — *"of what we kept, 95% is right"* — never *"we wrongly discarded Y% of good
nodes"*. Answering the second needs a re-extraction with the gate lowered.

Precision only, deliberately: recall would require *"what should this paper have
produced"*, a granularity judgement rather than a fact, and matching gold names against
extracted names would report entity-resolution failure as extraction failure.

---

## 6. Judge calibration

34 hand-graded answers on `vector_fulltext`, joined to judge scores by
`scripts/calibrate_judge.py` — the first time this join has existed in code; Iteration 1's
figures were computed by hand.

| criterion | QWK | ρ | p | n | trusted |
|---|---|---|---|---|---|
| `coverage` | **+0.72** | +0.77 | <0.001 | 34 | yes |
| `synthesis` | **+0.68** | +0.72 | <0.001 | 34 | yes |
| `refutation_handling` | **+0.65** | +0.73 | 0.026 | 9 | yes |
| `hedging_accuracy` | +0.55 | +0.41 | 0.017 | 34 | no |
| `attribution` | +0.34 | +0.39 | 0.021 | 34 | no |

Length bias on `synthesis`: +0.02 per 100 chars, p=0.036 — newly detectable (p=0.733 in
Iteration 1) and worth ~1 point across a 5,000-character answer.

Against Iteration 1: `synthesis` 0.53 → 0.68 and `refutation_handling` unmeasured (n=3) →
0.65 become trustworthy; `hedging_accuracy` 0.67 → 0.55 loses trust, which n=10 always
risked.

**`attribution` fails by range restriction, not leniency.** Both means sit near 3.0.

| score | 1 | 2 | 3 | 4 | 5 | sd |
|---|---|---|---|---|---|---|
| human | 11 | 3 | 10 | 2 | 8 | 1.53 |
| judge | 1 | 10 | 11 | 12 | 0 | 0.87 |

The judge never returns 5 and almost never 1. Disagreements are symmetric — it rates the
worst answers 4 and the best 2. This requires a rubric with anchored examples at both ends,
not rescaling.

**Consequence for reporting:** the `coverage` and `synthesis` columns are usable across
arms; `attribution` and `hedging_accuracy` must be labelled untrusted or dropped.

---

## 7. Reproducibility layer

### 7.1 Schema

The prior schema could express one thing where three are needed. A null meant both *"the
paper is silent"* and *"we have not looked"*, so a system inventing a qubit count could not
be distinguished from one correctly reporting nothing. Now three states (value /
`not_reported` / `None`) and six outcomes (`correct`, `wrong`, `missed`, `hallucinated`,
`correct_absence`, `skipped`).

The field set was made quantum-shaped by measurement: `qubit_count` appears in 151 of 242
`Hardware` nodes, `gpu_type` in 107. The original schema led with GPU fields.

### 7.2 Result

21 papers, 138 scoreable fields (from 25).

```
accuracy 73.2%   —  of which 67.4% is correct silence
```

| field | correct | absence | wrong | missed | hallucinated |
|---|---|---|---|---|---|
| `quantum_vendor` | 2 | 12 | 1 | 1 | 4 |
| `device_name` | 1 | 10 | 4 | 1 | 3 |
| `qubit_count` | 3 | 8 | 1 | 6 | 1 |
| `gpu_type` | 1 | 17 | 0 | 2 | 0 |
| `gpu_count` | 1 | 17 | 0 | 2 | 0 |
| `code_url` | 0 | 15 | 0 | 5 | 0 |
| `dataset_access` | 0 | 14 | 0 | 6 | 0 |

**Recall on stated facts: 17%. Hallucination rate: 6%.** The aggregate is uninformative on
this corpus — an empty system scores 67.4% — so the `correct` / `correct_absence` split is
the number to read, and `summarize()` prints both so the flattering figure cannot travel
alone.

`code_url` and `dataset_access` are 0-for-15 and 0-for-14, missing five literal GitHub URLs.
`ReproducibilityArtifact` fires 33 times corpus-wide but not on those papers: a
prompt-routing gap rather than a grounding failure, and the largest available improvement.

Hallucinations concentrate on simulation papers that name a vendor — `e17e52d7` reports
`quantum_vendor: Google` and `device_name: NVIDIA DGX-A100` for work that *simulates*
Sycamore circuits on GPUs with no quantum hardware involved.

---

## 8. Late items

### 8.1 `Chunk.id` uniqueness — fixed

Ids were `{paper_id}::{section_type}::{start}-{end}`, but `char_start`/`char_end` are
offsets into their own section and restart at 0, while papers routinely carry several
sections typed `other`. 43 ids covered 46 excess chunks, always different text within one
paper. A colliding id means two spans share an identity and an id-keyed store serves
whichever was written last. Fixed by including the section index; requires a re-chunk and
re-index to take effect on stored data.

### 8.2 Cross-paper contradictions — attempted, audited, rejected

**Motivation.** Nine of the 34 gold queries ask about a disagreement, and a correct answer
must surface both sides rather than report one as settled. That requires the graph to
*encode* the disagreement, and it holds 33 such edges across 271 papers. The cause is
structural: extraction reads one paper at a time, so the only conflicts it can observe are
ones a paper states about itself. Noticing that paper A conflicts with paper B requires
holding both at once, which per-paper extraction never does. §4.3 is the consequence —
1 of 9, on every arm including plain vector search.

**The pass.** 16,972 cross-paper claim pairs above cosine 0.90, adjudicated three ways,
$1.61.

```
3,072 edges accepted (18.1%)   refutes 163   undercuts 2,909   neither 13,900
spanning 249 of 271 papers
```

A 93× increase in the contradiction layer.

**The audit.** 60 pairs, 20 per model verdict, labelled without sight of the model's
decision.

```
exact agreement 46.7%  (28/60)

  model        human ->   refutes  undercuts  neither
  refutes                       5          3       12
  undercuts                     0          5       15
  neither                       0          2       18
```

**Edge precision 32.5%** — the fraction of accepted pairs a human also calls a
disagreement, counting `refutes` and `undercuts` as interchangeable, since a mistyped edge
is still a real one. Of 3,072 accepted, **~998 are real and ~2,074 are not.** Separately,
2 of 20 rejected pairs are genuine disagreements the pass discarded.

**Decision: the edges are not applied**, and the acceptance figure is not reported as a
result. §4.3 stands unimproved.

**The failure is a single pattern**, and it is the one the prompt already warns against.
Twelve of the twenty spurious `refutes` are two papers describing their own different
scopes rather than disagreeing:

- *"general deterministic treatment of per-qubit noise"* vs *"the general-noise analysis is
  restricted to single-qubit noise"* — each paper stating what it did
- *"2-qubit Pauli error rate set to double the single-qubit rate"* vs *"single-qubit gates
  are perfect"* — two chosen noise models, not a factual conflict

Several accepted pairs plainly *agree*: both stating that sampling overhead grows
exponentially, both stating that current hardware falls far short. A revised prompt
carrying these as explicit worked negatives is the obvious next attempt, and the verdict
cache means a re-run costs only what changes.

**Design note.** In entity merging a high similarity floor acts as a *precision* filter —
two names for one method are near-identical strings. Here the relation inverts:
contradictory claims are similar *by construction*, since they concern the same subject and
differ only in what they assert. Similarity is therefore purely a recall filter and carries
no information about whether a conflict exists; the model performs the entire
discriminative step. Over-acceptance is thus the predicted failure mode, and it is what
occurred. Everything fails closed to `neither` for the same reason — a false `refutes` edge
would route a refutation query to a disagreement no paper asserts.

**Audit caveats.** The labels are one model's, not a human's: a different model, a
different prompt, and no sight of the original verdict, so this *bounds* the error rate
rather than settling it. Separately, `sample_pairs` originally emitted the three strata in
order, so position alone revealed the model's verdict; it now shuffles, but these labels
were made against the unshuffled sheet and were therefore not blind to the strata
boundaries, though each pair was judged on its merits. A human spot-check of the 60 labels
would resolve both.

---

## 9. Operating envelope — what the system can be asked

Everything above measures components. This section states what the assembled system does
for someone typing a question into `scripts/ask.py`, since that is the only claim a reader
can act on. Each band is the measured `must_cite_recall` for that query type at n=34, best
arm quoted; judge figures are given only for calibrated criteria (§6).

**Scope.** 271 readable papers, almost entirely quantum computing — VQE and quantum
chemistry, QAOA, barren plateaus, error correction and mitigation, classical simulation,
quantum machine learning — plus a cluster on classical community detection. Questions
outside that are still answered, from whatever is nearest, which is worse than declining.

### 9.1 Supported

**Single-fact lookup — 0.500 to 0.667.** One paper, a specific reported value.

> *"What logical error rate did the below-threshold surface code memory report, at what
> distance, and on which processor?"*

The best-performing type, and the only one where `typed_graph_chunks` leads.

**Survey of a subfield — human `coverage` 3.0–3.5 / 5.** `coverage` is calibrated
(κ=+0.72), so this figure is trustworthy.

> *"Which error mitigation techniques exist, and what overhead does each impose?"*

Expect real citations and real omissions. Expect also that roughly a third of claims carry
a citation that does not fully support them: hand-graded `attribution` averaged 2.79/5 with
11 of 34 answers scored 1.

### 9.2 Weak

**Multi-part relational — 0.202 to 0.357.** The system names the parts and connects them
poorly; the output reads as a list rather than a synthesis. This is the majority query type
and the one the architecture targets.

### 9.3 Unsupported

| ask | measured | why |
|---|---|---|
| *"Do papers disagree about X?"* | 1 of 9 surfaced | 33 contradiction edges in the graph (§8.2) |
| *"What hardware / code does paper X provide?"* | 17% of stated facts; `code_url` 0/15 | §7.2 |
| *"Is there any work on X?"* — expecting "no" | not measured | no mechanism to detect corpus absence |

The third is the most dangerous, because it fails silently: the system cannot know the
corpus does not cover something, so it answers from the nearest available material with no
signal that it has done so.

### 9.4 Scope for improvement, ordered by measured headroom

1. **Contradiction encoding.** §8.2 is characterised, not unexplained: the failure is one
   prompt pattern, the fix is worked negative examples, and the verdict cache makes a
   re-run cost only what changes. Largest single gap, and the clearest route.
2. **`ReproducibilityArtifact` routing.** `code_url` and `dataset_access` are 0-for-15 and
   0-for-14 while missing five literal GitHub URLs in body text. A routing gap, not a
   grounding failure, so it should be cheap.
3. **Relational synthesis.** The 0.202 in §4.2 is the thesis's central weakness. Worth
   noting that §5.1 and §5.2 have already ruled out the two obvious explanations —
   duplicate nodes and edge quality — so the next hypothesis has to come from elsewhere.
4. **`attribution` rubric.** The judge's range restriction (§6) means answer quality on
   this axis currently cannot be tracked automatically at all.
5. **Corpus-absence detection.** Unbuilt and unmeasured. A calibrated "the corpus does not
   cover this" would remove the failure mode in §9.3 that a reader is least equipped to
   catch.

---

## 10. Threats to validity

**Sample size.** The primary result is n=34; §4.1 demonstrates the ±0.10 noise floor at
n=10 by an ordering that changed. The ablations in §5.1 and §5.2 were run at n=10 and their
gaps are not individually significant — they are reported as directional, and their weight
comes from the size of the effect being *absent* rather than from a significant difference.

**Single grader.** Calibration rests on one annotator with no inter-annotator agreement
measure, so `coverage`/`synthesis` trustworthiness is trustworthiness *relative to this
grader*.

**Untrusted judge criteria.** Two of five fail calibration and are excluded from every
claim above.

**Corpus composition.** 96 of 138 repro gold fields are `not_reported`, and the corpus is
predominantly theory and simulation, so §7 generalises to similar corpora only.

**A judgement call in §5.2.** The citation arm's `max_nodes` (30 vs 150) was chosen, not
measured. The reasoning is documented and the alternative would have been worse, but the
arm's standing depends on that choice.

**Gold authored partly by the same model family** that performs extraction. Mitigated by
BM25-based grounding, and by never showing the authoring tools the system's own output —
but not eliminated.

**§8.2 was audited by a model, not a human** (§8.2 audit caveats), and its edges are
excluded from the graph and from §1 accordingly.

---

## 11. Engineering record

Defects found and resolved, with the measurement that surfaced each.

| defect | surfaced by | resolution |
|---|---|---|
| Graph accumulated across rebuilds — a union of two extraction generations | Kuzu buffer-pool exhaustion | delete DB + `.wal`/`.shadow` before every build |
| `normalize()` stripped all parentheticals; 86% of merges wrong | re-inspection after shipping on a sample of 2 | flatten punctuation, keep content; 369 → 50 merges |
| 909 adjudicated merges never applied to the graph | reading stage 05 while wiring the audit | `_semantic_merges()` in the build, keyed by name |
| `libomp` triple-copy segfault — an Iteration-1 defect reintroduced | exit 139 during candidate generation | `faiss.omp_set_num_threads(1)` immediately before search |
| Serial adjudication lost all work on timeout | 3,487-pair run | ThreadPoolExecutor + checkpoint every 200, fail closed |
| Kuzu binder error on `ORDER BY r.confidence` with `RETURN DISTINCT` | traversal query | order on the returned alias |
| Node-quote evidence scored 0.183 | splitting recall into reach vs conversion | route to chunks at paper granularity; 0.183 → 0.483 |
| Token-containment matching inert in the repro scorer | `A100` vs `NVIDIA A100-SXM-80GB` scored wrong | relax `correct`/`wrong` only; absence logic unchanged |
| A corpus entry was an OSTI landing page, not a paper | authoring repro gold | left unauthored; journal version ingested under its own id |
| `Chunk.id` collisions | uniqueness check | section index in the id |

The 0.183 → 0.483 entry is the most instructive. The first scored typed-graph run looked
like a total failure until the metric was decomposed: retrieval was reaching 0.556 of
required pairs against the vector arm's 0.611, but converting only 33% into citations
against 60%. The traversal was working; one-sentence quotes stripped of surrounding
argument were too thin to write from. Routing to chunks raised conversion to 87%.

**Predictions that were wrong**, recorded because each was cheap to test and testing changed
the conclusion:

1. Semantic merging would help the typed arm — it does nothing (§5.1)
2. Extraction precision would be 50–60% — it is 95% (§5.3)
3. `Claim`-shaped seeds would crowd out concept nodes — removing them halved recall
4. Near-duplicates would dominate contradiction candidates — 13 of 16,972

---

## 12. Carried forward

- **Revise the §8.2 prompt with worked negatives, re-run, re-audit** — the pass is
  rejected at 32.5% edge precision, not abandoned; §9.4 item 1
- **Re-chunk and re-index** so §8.1 takes effect on stored data
- **`attribution` rubric with anchored 1 and 5 examples** — the range restriction in §6
- **`ReproducibilityArtifact` routing** — the 0-for-15 in §7.2 is a prompt gap, not a
  grounding one
- **Inter-annotator agreement** — the single-grader threat in §10
- **n ≥ 34 as the standing default** for any claimed result; the 10-query set is a
  development set only