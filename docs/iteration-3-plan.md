# Iteration 3 — plan

Iteration 2 is closed; its result and every number cited here are in
[iteration-2-report.md](iteration-2-report.md).

**What changes.** Iterations 1 and 2 measured *static* retrieval: one query in, one
retrieval, one answer. This iteration makes the system act — decompose a question, plan
retrievals, criticise its own answer, and write back what it learns — and asks whether that
beats the static arms on the same gold set.

---

## Why agentic, from the evidence rather than from fashion

Iteration 2 ended with a specific, measured failure and a mechanism for it (§4.2, §4.5):

> Relational queries score **0.202** against vector search's 0.357 — worst of any type, and
> the type the typed graph was built for. §4.5 found why: those queries are near-uniformly
> *"X, and what limits each"*, `addresses` carries 35.7% of all traversal, and `undercuts`
> — the edge that would reach the second half — was traversed **zero times across 34
> queries**, because 33 such edges exist in a graph of 11,186.

There are two ways to fix that, and they are directly comparable on the same number.

1. **Add the missing edges** so the traversal can reach the second half. Attempted in §8.2
   and rejected at 32.5% edge precision; carried into 3.6 below.
2. **Stop needing them.** An agent decomposes *"which methods, and what limits each"* into a
   retrieval for the methods and then one retrieval per method for its limitations. The
   missing edge stops mattering because the second hop becomes a second *query* rather than
   a graph traversal.

This iteration tests (2). That is the hypothesis, and 3.5 is the test.

**It also reframes what the graph is for.** Iteration 2's honest finding is that the typed
graph is a poor *retriever* — free citation edges match it within 0.016 (§5.2). It says
nothing about whether it is a useful *planner*. `addresses` reliably connects a problem to
the methods that tackle it; that is a decent "what should I look up next" signal even where
it is a weak "here is the evidence" signal. 3.1 is the first thing in this project to use
the graph that way.

---

## Stated before measuring

Iteration 2's conclusion is a defensible negative result. The way to spoil it is to keep
adjusting an agent until 34 queries move. So:

| outcome | reading | what gets written |
|---|---|---|
| agentic beats static on relational, and the gain concentrates in queries whose gold names a limitation | hypothesis supported | decomposition substitutes for missing edges — with 3.4 showing *where* in the plan the gain arises |
| agentic matches static at higher cost | hypothesis refuted | the constraint is not retrieval structure; report the cost multiple honestly |
| agentic beats static uniformly across all query types | **suspect** — that is more retrieval, not better planning | report as an evidence-volume effect until an ablation separates them |

**3.5 is scored on the 34-query set, once.** The 10-query set is a development set. Prompt
and planner tuning happen there and are frozen before 3.5 runs.

**Which metric is read on which set — fixed here, before 3.5 runs.** The maintenance track
established that the two gold sets disagree about which judged criteria pass, so "the gold
set" is no longer a single answer. It is now a per-metric decision:

| what | set | why |
|---|---|---|
| deterministic metrics — the headline | **34** | judge-independent by construction, and §4.1 puts a ±0.10 noise floor on n=10 where the top two arms differ by 0.034. Ten queries cannot separate the arms. |
| judged criteria in 3.5 | **`coverage` only** | the one criterion certified on *both* sets (+0.76 on the 10, +0.76 on the 34, +0.72 on the other 24), so the set mismatch cannot bite |
| judge calibration | **10 primary, 34 alongside** | the 10 are what the thesis reports; the 34 is printed beside every figure because the sets diverge |
| 3.1 development | 10 | unchanged — frozen before 3.5 |

The rule this encodes: **a judge certified on one set is not certified on the other.**
`synthesis` scores +0.38 on the 10 and +0.77 on the 34, `attribution` +0.79 and +0.45, so
certifying on the 10 and scoring on the 34 would be incoherent in either direction.
`coverage` is the only criterion indifferent to the choice, which is precisely why it is the
only judged criterion 3.5 may report.

---

## 3.1 Agentic planner–critic loop

Decomposition, retrieval planning, self-critique. A new arm behind the existing `System`
protocol, so `06_run_eval.py` scores it unchanged — the runner has been system-agnostic
since Iteration 1 and this is what that was for.

**Shape.** Plan sub-questions → retrieve per sub-question → draft → critique against the
plan → re-retrieve for whatever the critique flags → synthesise.

**Where the graph earns its place.** Planning may consult the typed graph for *what to ask
next* rather than for evidence: seed on the question, read the `addresses` neighbourhood,
and turn the reached `Method` nodes into sub-questions. That is the one thing Iteration 2
showed traversal does reliably.

**Design constraints, carried from Iteration 2:**

- Same synthesis prompt and citation-handle scheme as the static arms wherever possible, so
  the difference under test is the *loop* and not the writing (§2).
- Hard cap on retrieval calls per query, recorded per run. An agent that wins by issuing
  twenty retrievals has not beaten `top_k=60`, it has spent more — 3.5 must be able to
  report the cost multiple.
- Fail closed: if the planner returns nothing usable, fall back to a single retrieval and
  mark the trace, rather than erroring or silently degrading.

**Acceptance:** runs end-to-end on the 10-query development set, produces well-formed
traces for 3.4, and its retrieval budget is bounded and logged.

---

## 3.2 Query-time STAGED writes with provenance

`SourceLayer.STAGED` exists in `schema.py`, `promote_staged()` is implemented in
`graph_store.py`, and grep finds STAGED written **nowhere**. The interface was built in
Iteration 1 for exactly this and has never been used.

**What it is for.** When the agent derives something at query time — a decomposition that
worked, a link between two papers it had to infer — that finding is currently discarded
when the answer is returned. STAGED lets it persist with provenance while staying out of
everything that gets measured.

**The invariant that makes it safe**, already in `base.py`: metrics query CURATED only.
STAGED is never auto-merged. An agent cannot improve its own score by writing to the graph,
which would be self-grading of the same kind Iteration 2 refused when it declined to pick
`must_cite` with the system's own retriever.

**The promotion path is a review path, not an automation.** `promote_staged` moves reviewed
nodes into CURATED. Given §8.2 — where an unaudited 3,072-edge proposal would have entered
the graph on file presence alone until the approval flag stopped it — promotion needs the
same treatment: nothing enters CURATED without a labelled audit.

**Acceptance:** the agent writes STAGED nodes with provenance; a CURATED-only eval run
produces identical numbers with and without them present. That equality is the test that
the layer separation actually holds.

---

## 3.3 Local query-time inference (vLLM)

`config.py` carries `local_inference_model: "Qwen/Qwen2.5-14B-Instruct-AWQ"` and nothing
reads it.

**Why it matters here specifically.** An agentic loop multiplies calls per query — that is
the point of 3.1 and also its cost. Iteration 2 spent $0.29 for four arms over 34 queries;
a planner-critic loop issuing 5–10 calls per query changes that arithmetic. Local inference
makes the iteration affordable rather than making it better.

**What must be held constant.** Swapping the model changes answer quality independently of
the loop. So either 3.5 runs both arms on the same model, or the model becomes a second
variable and the comparison is void. Recommended: keep the hosted model for 3.5's scored
run, use vLLM for development iterations and for the retrieval-heavy planning steps where
quality matters least.

**Acceptance:** a `--local` flag routes chat calls to vLLM; a 10-query run completes; the
cost and latency delta is recorded. Explicitly *not* required: matching hosted quality.

---

## 3.4 Trajectory eval — scoring the plan, not the answer

Every metric so far scores the *output*. `must_cite_recall` cannot distinguish an agent
that planned well and synthesised badly from one that did the reverse — and for a
planner-critic loop that distinction is the whole object of study.

`traces.jsonl` is the substrate: it already persists evidence per query (§3.3 of the
Iteration 2 report), and 3.1 extends it with the plan, the sub-questions, the retrievals
each produced, and what the critique changed.

**Candidate measures**, to be fixed before 3.5 runs:

- **Decomposition coverage** — do the sub-questions collectively cover the gold `facets`?
  Deterministic, since facets are already authored for all 34 queries.
- **Retrieval efficiency** — required papers found per retrieval call. This is where an
  agent could beat static retrieval on quality while losing on cost, and the number that
  makes that visible.
- **Critique usefulness** — did the second pass add a required paper the first missed? A
  critique that never changes the answer is an expensive no-op.
- **Plan-outcome coupling** — do trajectory scores predict `must_cite_recall`? If not,
  either the trajectory measures are wrong or the plan does not matter, and both are worth
  knowing.

**Discipline from Iteration 2, hardened by 3.6c/3.6d:** any judge-scored trajectory criterion
is untrusted until calibrated against hand grades. §6 reported two of five failing; on a
deterministic judge it is four of five, and the survivors differ by gold set. A new metric
family starts untrusted and inherits three rules:

- **temperature 0.** Nothing in the project pinned it before 3.6c, and per-criterion κ
  spreads reached 0.25 against a 0.26 gap to the trust bar. A criterion that disagrees with
  itself cannot agree with a grader.
- **certified on the worst sample, not one draw.** `calibrate_judge.py --repeats` enforces
  this; a single draw was enough to certify `refutation_handling` at +0.65 and to put it at
  +0.43 on the next.
- **checked against the grader's own self-agreement.** 3.6d found `attribution` at +0.79 on
  the active 10 while the grader reproduces those labels at +0.19. A judge that agrees with
  one sitting better than the grader agrees with themselves has fitted a sitting. Any new
  criterion needs its human ceiling measured before its κ means anything.

---

## 3.5 Agentic vs static on the same gold set — **the deliverable**

The comparison everything above serves. All arms, 34 queries, unchanged metrics, reported
by query type. Deterministic metrics carry the headline; `coverage` is the only judged
criterion that may appear (see *Stated before measuring*). `attribution`,
`hedging_accuracy` and `refutation_handling` must not be reported across arms — the first two
are at or below the grader's own self-agreement, and the third is uncalibrated at n=9 on the
34 and unmeasurable at n=3 on the 10.

| arm | source |
|---|---|
| `vector_fulltext` | Iteration 2 baseline, and the one to beat at 0.456 |
| `typed_graph_chunks` | 0.377 |
| `citation_graph` | 0.367 — the free-metadata control from §5.2 |
| `agentic` | 3.1 |

**Report alongside the score, not after it:** retrieval calls per query, tokens per query,
and cost per query. Iteration 2's arms are roughly equal-cost; this one will not be, and a
win that costs 10× is a different claim from a win that costs 1.2×.

**Required breakdown.** The 14 relational queries split by whether their gold `key_claims`
name a limitation or cost. The §4.5 hypothesis predicts the gain concentrates there. A
uniform rise is evidence for more-evidence-volume, not for decomposition, and must be
reported as such.

**Required ablation.** Agentic with the critique step disabled. Without it, "the loop
helps" cannot be separated from "planning helps"; §5.1 is the template — an ablation that
found the honest answer was *nothing changed*.

---

## 3.6 Maintenance track — **closed**; see [iteration-3-maintenance.md](iteration-3-maintenance.md)

All four ran. Three of four hypotheses are refuted, and in every item the binding constraint
turned out to be the measuring instrument rather than the thing being measured — judge
temperature, extraction temperature, the `DatasetAccess` enum, the audit labeller, and the
grader's own stability.

| # | outcome | consequence for this iteration |
|---|---|---|
| 3.6a | **refuted.** v2 holds edge precision at 32.5% and discards ~65% of the real edges. §8.2's figure is corroborated by a human audit of 60 fresh pairs | **the second route to §4.5 is closed** — 3.1's decomposition is now the only one, which raises what rides on it |
| 3.6b | **partly confirmed.** Three faults, not one; `code_url` 0→3, `dataset_access` 0→3 | fix committed, corpus re-extraction *not* run — it rebuilds the substrate every arm reads and must be decided alongside 3.5 |
| 3.6c | **refuted.** The original rubric beats both rewrites; the judge had been sampled at temperature 1.0 throughout | only `coverage` is certified for cross-arm use; `refutation_handling` loses its §6 certification |
| 3.6d | **closed permanently.** No second annotator exists; run as test–retest | §10's single-grader threat is a standing limitation, not a pending action |

The original scoping is kept below for the record.

| # | item | why |
|---|---|---|
| 3.6a | **Contradiction pass v2** — worked negative examples in the prompt, re-run, re-audit to ≥70% edge precision | §8.2 failed at 32.5% because twelve of twenty spurious `refutes` are papers describing their own different scopes, which a prose warning did not prevent. This is fix (1) from the top of this document, and if it lands it is a *second* route to the same §4.5 problem — directly comparable with 3.5 |
| 3.6b | **`ReproducibilityArtifact` routing** | `code_url` 0-for-15, `dataset_access` 0-for-14, five of them with a plain GitHub URL in body text. A routing gap, so the §2.4 `Hardware` fix applies again; gold is already authored at 138 fields |
| 3.6c | **`attribution` rubric** with anchored 1 and 5 examples | κ=+0.34 fails by range restriction, not bias: the judge never returns 5, almost never 1. Rescaling cannot fix it. Re-grade against the existing 34 hand grades — no new labelling |
| 3.6d | **Second annotator** on 20 of the 34 | §10's standing threat: every calibrated criterion is calibrated relative to one grader |

---

## Not in this iteration

**Phase 2 portability (Neo4j AuraDB + Qdrant).** `stores/base.py` defines `VectorStore`,
`GraphStore` and `Embedder` as ABCs, so this is a config swap behind existing interfaces —
a deployment concern on its own axis, doable at any iteration and belonging to none.

**More gold queries.** 34 detects the effect 3.5 predicts. Growing the set mid-hypothesis
changes instrument and measurement together.

**Retrieval tuning.** `top_k`, hops, seed count and node cap are measured and locked
(§3.2, §4.5). Touching them confounds 3.5, which is the only reason this iteration exists.

---

## Order

```
3.1  planner-critic loop            the system under test
3.4  trajectory eval                 must exist before 3.5, or the run is unrepeatable
3.3  local inference                 makes 3.1 development affordable; not on the scored path
3.2  STAGED writes                   independent; acceptance is that metrics do not move
3.5  agentic vs static + ablation    the deliverable — runs once, after 3.1 is frozen
3.6  maintenance                     parallel throughout; 3.6a is a second route to §4.5
```

The trap to avoid is running 3.5 early and repeatedly. Iteration 2 established that the
ordering of arms changes between n=10 and n=34 (§4.1); a headline measured more than once
against the same 34 queries stops being a measurement.