# Iteration 3 — technical report

Iteration 2 asked whether a typed knowledge graph retrieves better than vector search. It
answered no, and diagnosed the worst case — relational queries at 0.202 — as missing
`undercuts` edges (§4.5). This iteration made the system *act* instead: decompose a question,
retrieve per part, critique its own evidence, and it asks whether that beats the static arms
on the same gold set.

Supporting documents: [the plan](iteration-3-plan.md), [the frozen
configuration](iteration-3-freeze.md), [§3.5's full results](iteration-3-results.md), [the
maintenance track](iteration-3-maintenance.md).

---

## 1. Result

**Decomposition works, and only where it was designed to.** On the 34-query gold set,
paired per query, three repeats:

| | Δ vs `vector_fulltext` | W / L / T | p |
|---|---|---|---|
| all 34 queries | +0.059 | 12 / 9 / 13 | 0.403 |
| **relational (n=14)** | **+0.250** | 8 / 1 / 5 | **0.012** |
| non-relational (n=20) | −0.075 | 4 / 8 / 8 | 0.340 |

The overall figure is a null. The result lives entirely in the relational subset, and the arm
is measurably **worse** everywhere else — which is what makes it credible. The plan
pre-registered a uniform gain as *suspect: that is more retrieval, not better planning*. This
is the opposite of uniform.

**Cost: 1.2×** ($0.028 vs $0.024 per query). Not the order of magnitude the plan anticipated.

**But the mechanism is not the one §4.5 proposed**, and three independent measurements say so
(§4 below). Decomposition helps relational queries; *why* remains open.

---

## 2. What changed underneath, before anything was measured

Three sampling defects were found and fixed. Each was the same defect in a different layer,
and each had been making a reported number a single draw rather than a measurement.

| layer | was | now | what it cost |
|---|---|---|---|
| judge | provider default (1.0) | 0.0 | κ spreads to 0.25 against a 0.26 gap to the trust bar |
| extraction | provider default (1.0) | 0.0 | 9 of 147 field outcomes differed between identical runs |
| planner | — | 0.0, and **still not deterministic** | 3 draws of one plan gave 3 distinct wordings |

The planner case is the one that could not be fixed by pinning. `gpt-5.4-nano` does not return
identical text at temperature 0, and in an agentic arm **the planner's output is the retrieval
query** — so a reworded sub-question embeds differently, reaches different chunks, and changes
which papers can be cited. A static arm embeds a fixed gold query and absorbs that noise; this
arm amplifies it. Measured: `must_cite_recall` of 0.567 and 0.333 across two runs of
functionally identical code.

The fix was architectural, not a parameter. The arm now always retrieves on the **original
query** first, so the plan moves the margin rather than the whole evidence set:

```
with anchor      0.417  0.417  0.417     spread 0.000
without anchor   0.367  0.383  0.417     spread 0.050
```

Adopted on a stopping rule stated before the change: keep only if it narrows the spread.

**The corpus was rebuilt** (271 papers, extraction at temperature 0) to apply §3.6b's fix.
That moved more than intended: `Claim` nodes +44%, nodes 23,689 → 27,777, edges 11,131 →
14,978. Iteration 2's stored figures do not describe this substrate.

---

## 3. The system under test (§3.1)

Plan sub-questions → retrieve per sub-question → critique the evidence against the plan →
re-retrieve the gaps → synthesise. Behind the existing `System` protocol, so the runner scored
it unchanged.

Three properties carry the comparison's validity:

- **Synthesis is the static arms' code, not a copy.** The arm holds a `VectorRAGSystem` and
  calls its evidence formatting, handle resolution and synthesis directly, so the prompt and
  the `P1`-handle scheme are identical by construction. A test pins the function identity. The
  variable under test is the control flow and nothing else.
- **The retrieval budget binds** — 6 per query, refused rather than logged when exceeded, at
  `top_k=20` rather than 60 so several retrievals stay within the same order of evidence as
  one static retrieval.
- **It fails closed.** A planner that raises or returns an empty plan degrades to one
  retrieval on the original query and sets `planner_failed`, so a broken planner cannot score
  as a working one.

---

## 4. The mechanism is not edge substitution

§4.5 proposed that decomposition works by *substituting for the missing `undercuts` edges* —
the second hop becomes a second query rather than a graph traversal. Three measurements test
that account, and none supports it.

**(a) Adding the edges does not work.** §3.6a rewrote the contradiction prompt with the twelve
labelled spurious `refutes` as worked negatives. Edge precision held at **32.5%** — identical
stratum rates under both prompts — while the revision discarded ~65% of the real edges. Worse
than the pass it replaced.

**(b) Edge coverage was not the binding constraint.** The corpus rebuild took `undercuts` from
33 to 119, and traversal from **zero occurrences across 34 queries** to six. `typed_graph_chunks`
scored **0.383 → 0.383**, unchanged to three decimal places. The gap §4.5 identified was real;
closing it changed nothing.

**(c) The gain does not concentrate where the account predicts.** Splitting the 14 relational
queries on whether their gold `key_claims` name a limitation or cost:

| | names a limit (n=5) | does not (n=9) |
|---|---|---|
| Δ vs baseline | +0.167 | **+0.296** |

The gain is *larger* where no limitation is named. Pre-registered as underpowered at 5/9
before the numbers were seen, so it cannot refute the account — but it does not support it.

**And plan quality does not predict outcome.** §3.4's trajectory eval over three repeats,
n=34 each:

| measure | ρ vs `must_cite_recall` | p |
|---|---|---|
| `decomposition_specificity` | −0.16, −0.08, −0.16 | 0.375, 0.654, 0.369 |
| `retrieval_efficiency` | +0.40, +0.43, +0.44 | 0.019, 0.010, 0.010 |

How distinctive a plan is — whether its sub-questions match its own query's facets better than
any of the other 33 queries' plans — carries **no information** about whether the query is
answered well. Consistently, across three independent runs.

`retrieval_efficiency` survives, with its definitional overlap stated: it counts required
papers *reached* per call while `must_cite_recall` counts required papers *cited*. They share a
numerator term, so the correlation is partly built in — though not wholly, since Iteration 2
measured a reach-to-cite conversion of only 33% for the typed-graph arm.

**What survives as a hypothesis**, untested: the gain comes from issuing several retrievals
against the *parts* of a question, not from the plan being good. Volume alone cannot explain it
— the arm is worse on non-relational queries while spending the same retrievals there. So it
appears to be the act of splitting rather than the craft of the split. An arm that splits
mechanically on conjunctions, with no planner at all, would separate the two.

---

## 5. The self-critique

The required ablation. On relational: **+0.091, 8 wins / 1 loss / 5 ties, p=0.043.** Removing
it drops relational from 0.528 to 0.437, back among the static arms.

**And it is unexplained.** The critique changes the evidence on 31–33 of 34 queries but adds a
*required* paper on only 4–7. A step that rarely reaches new required papers should not be
worth 0.091, and §3.4's measures do not say why it is. Under Bonferroni across the two
pre-registered tests (α=0.025), this result does not survive; the relational-vs-baseline result
does.

---

## 6. Query-time STAGED writes (§3.2)

The agent now persists its decomposition as STAGED nodes with provenance — the query, its
sub-questions, and per sub-question the papers it reached. That last field records which papers
*jointly* bear on one part of a question, which per-paper extraction structurally cannot
observe because it never holds two papers at once.

**The item found that the invariant it relies on was never enforced.** `stores/base.py` has
promised since Iteration 1 that metrics query CURATED only. Neither retrieval arm filtered on
`source_layer`. The promise held solely because nothing had ever written a STAGED node — it was
vacuously true, and the first query-time write would have made it false, letting an agent raise
its own score by writing to the graph it is scored against.

Acceptance, demonstrated with real staged data rather than argued:

```
before staging   filtered 20688   unfiltered 20688
38 nodes staged from 10 real trajectories
after  staging   filtered 20688   unfiltered 20726
```

Promotion is now gated behind an explicit approval, on §8.2's precedent.

---

## 7. Local inference (§3.3) — deferred, with the justification refuted

The stated rationale does not hold. §3.3 assumed a loop issuing 5–10 calls per query and an
affordability problem; the measured arm makes **3 calls at $0.028/query**, and the whole
deliverable cost ~$6. There was no cost problem to solve.

The item is kept because its acceptance criterion is a **portability** claim, which nothing
else in the iteration addresses. The `base_url` plumbing is landed — vLLM serves the OpenAI
API, so "route chat calls to vLLM" *is* "point an OpenAI client elsewhere". What waits is the
run, and the reason is hardware: `Qwen2.5-14B-Instruct-AWQ` needs ~8.5 GB in 4-bit and CUDA-only
kernels, against an 8 GB M2 where Metal caps the usable working set near 5.3 GB. Not a Metal
backend problem — 14B does not fit at all.

---

## 8. Maintenance track (§3.6)

Four carry-forwards, three refuted. Full detail in
[iteration-3-maintenance.md](iteration-3-maintenance.md).

| # | outcome |
|---|---|
| 3.6a contradiction v2 | **refuted** — 32.5% precision under both prompts, 65% of real edges lost |
| 3.6b repro routing | **partly confirmed** — three faults not one; `code_url` 0→2, `dataset_access` 0→2 |
| 3.6c attribution rubric | **refuted** — the original rubric beat both rewrites; the judge was at temperature 1.0 |
| 3.6d second annotator | **closed permanently** — one-person project; run as test–retest |

Two findings from that track reach further than their items:

**§6's certification does not survive a deterministic re-measure.** `refutation_handling` was
reported trusted at +0.65 and reads +0.44 (p=0.185) at temperature 0. Only `coverage` is
certified on both gold sets.

**`attribution` is at the human ceiling.** The grader agrees with *themselves* at +0.29; the
judge agrees with them at +0.30. There was never a gap for a rubric to close, which is why
three rubric versions moved offset and ranking but never agreement-on-level.

---

## 9. Corrections to the Iteration 2 report

**The Iteration 2 report is not amended.** It opens the Iteration 3 plan's chain of citations,
and editing it in place would retroactively change what that plan refers to and erase the
record of what was known when the decisions were made. A report that silently updates itself
cannot be audited. The corrections are recorded here instead.

| Iteration 2 claim | status |
|---|---|
| §6 `refutation_handling` κ=+0.65, **trusted** | **superseded** — +0.44, p=0.185 |
| §6's table generally | every figure is one draw from a temperature-1.0 judge; spreads to 0.25 |
| §6 *"requires a rubric with anchored examples… not rescaling"* | **refuted** — the unanchored original scores best |
| §4.1's n=10 `vector_fulltext` 0.417 | **a single lucky draw** — three runs give 0.367, and the old substrate produced 0.367 too |
| §10 *"no inter-annotator agreement measure"* | **permanent**, and now quantified by test–retest |
| §10 *"two of five criteria fail calibration"* | four of five on a deterministic judge |
| §10 *"§8.2 was audited by a model, not a human"* | **strengthened** — a human audit of 60 fresh pairs reproduces 32.5% |
| §7.2 `code_url` 0-for-15 as *a* routing gap | three faults; fixed and applied, `code_url` now 2 |
| §4.5 *"decomposition substitutes for missing edges"* | the effect replicates; **the mechanism does not** (§4) |

---

## 10. Threats to validity

**Scope: this work does not evaluate graph-based global summarisation.** All 34 gold queries
carry a `must_cite` list — every question has specific papers that constitute its answer, and
the headline metric counts required papers cited. That makes this an evaluation of graph-based
**retrieval on citation-grounded queries**.

It is *not* an evaluation of the GraphRAG family, which uses the graph differently: hierarchical
community detection, pre-generated community summaries, and a map-reduce over those summaries to
answer corpus-wide sensemaking questions. None of that is implemented here, and none of it could
be measured by this gold set — a good global answer might legitimately cite forty papers or none
in particular, and `must_cite_recall` would score it as a failure either way. So the negative
results about the typed graph in §5.2 and §4 above bear on the graph as a *retriever*, and say
nothing about the graph as a *summarisation scaffold*. A reader should not take them as
evidence against that approach.

Worth noting the convergence, though: the one thing the graph did reliably in this project was
supply **planner hints** — `addresses` neighbourhoods indicating what to ask next — while
repeatedly failing to justify its cost as an evidence retriever (§5.2: within 0.016 of free
citation edges). That is independent evidence for the proposition that a graph is better at
orientation than at retrieval, which is roughly GraphRAG's premise.

**Multiple comparisons.** Two tests were pre-registered. Under Bonferroni at α=0.025, the
relational-vs-baseline result (p=0.012) survives and the critique ablation (p=0.043) does not.

**n=14 relational**, 5 of those in the required breakdown group.

**One substrate, one metric, one judge-free comparison.** `must_cite_recall` counts required
papers cited and says nothing about whether an answer reads well. `coverage` is the only judged
criterion certified to accompany it, and **it has not yet been scored on the 3.5 runs** — that
measurement is outstanding.

**Three repeats** bound the run-to-run spread; they cannot establish determinism.

**Single grader, permanently.** §10 of Iteration 2 stands and will not be discharged.

---

## 11. Carried forward

| item | why it is open |
|---|---|
| `coverage` on the §3.5 runs | required by the plan; `rejudge.py` on stored answers, ~$4 |
| §3.3's vLLM run | needs CUDA hardware; config change plus one run |
| A mechanically-splitting arm | would separate "splitting helps" from "planning helps" (§4) |
| Why the critique is worth 0.091 | changes the evidence 31–33 times, adds a required paper 4–7 (§5) |
| Global-sensemaking queries and metrics | prerequisite for evaluating anything GraphRAG-shaped (§10) |
| 60 human contradiction labels | would upgrade §8.2's model-labelled 32.5% |
| `README.md` | still carries Iteration 1's judge table, which certifies `hedging_accuracy` and rejects `synthesis` — both now backwards |