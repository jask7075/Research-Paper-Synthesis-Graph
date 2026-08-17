# Iteration 3 — technical report

Consolidates what were five separate documents: the report, `iteration-3-results.md`,
`iteration-3-maintenance.md` and `iteration-3-freeze.md`. All four are recoverable in git;
the freeze record matters most there, because its evidential value is that it was committed
**before** §5 ran (commit `c8f7a50`), and §5.1 reproduces it rather than replacing it.

Pre-registration and rationale: [iteration-3-plan.md](iteration-3-plan.md). That file is
deliberately **not** merged here — a plan folded into the report it constrains stops being a
pre-registration.

---

## 1. Result

**Decomposing a question into parts and retrieving per part beats static retrieval on
relational queries, and only there.** 34 gold queries, paired per query, three repeats:

| subset | Δ vs `vector_fulltext` | W / L / T | p |
|---|---|---|---|
| all 34 queries | +0.059 | 12 / 9 / 13 | 0.403 |
| **relational (n=14)** | **+0.250** | 8 / 1 / 5 | **0.012** |
| non-relational (n=20) | −0.075 | 4 / 8 / 8 | 0.340 |

Arm standings, `must_cite_recall`, mean of three repeats:

| arm | mean | spread | judge `coverage` |
|---|---|---|---|
| **`agentic`** | **0.492** | 0.461 – 0.539 | **3.76** |
| `typed_graph_chunks` | 0.456 | 0.422 – 0.480 | 3.32 |
| `agentic_no_critique` | 0.449 | 0.436 – 0.466 | 3.71 |
| `citation_graph` | 0.444 | 0.431 – 0.456 | 3.47 |
| `vector_fulltext` | 0.433 | 0.412 – 0.446 | 3.56 |

The overall figure is a null. The result lives entirely in the relational subset, and the arm
is measurably **worse** everywhere else — which is what makes it credible rather than an
evidence-volume effect. The plan pre-registered a uniform gain as *suspect*.

Secondary findings, each from a dedicated measurement:

- **Cost is 1.2×**, not the order of magnitude the plan assumed (§5.6).
- **The self-critique earns its place**: +0.091 on relational, p=0.043 (§5.5).
- **The proposed mechanism is refuted** by three independent measurements while the effect
  replicates (§6).
- **Plan quality does not predict outcome** — ρ ≈ −0.13, replicated across three runs (§4.3).
- **Sampling was never disabled anywhere in the project**, in three separate layers (§2.1).
- **A documented safety invariant was never enforced** and had held only vacuously (§7).
- **Three of four maintenance hypotheses are refuted** (§9).

---

## 2. Setup — what changed since Iteration 2

### 2.1 Sampling was never disabled

Nothing in the project had ever set a sampling temperature, so every model call used the
provider default of 1.0. Three layers, three consequences:

| layer | measured cost of not pinning it | now |
|---|---|---|
| judge | κ spreads to 0.25 across three identical draws, against a 0.26 gap to the trust bar | 0.0 |
| extraction | 9 of 147 field outcomes differ between identical runs — the graph could not be rebuilt to itself | 0.0 |
| planner | three draws of one plan give three distinct wordings | 0.0, **still not deterministic** |

The planner case cannot be fixed by pinning. In this arm **the planner's output is the
retrieval query**, so a reworded sub-question embeds differently, reaches different chunks,
and changes which papers can be cited at all. A static arm embeds a fixed gold query and
absorbs that noise; this arm amplifies it. Measured: `must_cite_recall` 0.567 and 0.333 across
two runs of functionally identical code.

### 2.2 The anchor

The fix was architectural. The arm now always retrieves on the **original query** first,
before any generated text is involved, so the plan moves the margin rather than the whole
evidence set. `_merge` keeps it first, so the low `P1` citation handles are stable too.

```
with anchor      0.417  0.417  0.417     spread 0.000
without anchor   0.367  0.383  0.417     spread 0.050
```

Adopted on a stopping rule stated before the change: *keep only if it narrows the spread, not
if it raises the mean.* It did both; it was kept on the stated criterion. `anchor=False`
remains as the ablation.

### 2.3 The corpus rebuild

Run to apply §9.2's routing fix, with extraction pinned first so it happened once. 271 papers,
5,322 calls, $6.72, plus $0.26 to re-adjudicate 4,185 entity-merge pairs — required, because
`merge_verdicts.json` is keyed on node names and stage 05 **silently** skips semantic merging
for names it has no verdict for.

It moved more than the repro layer:

| | Iteration 2 | now |
|---|---|---|
| `Claim` nodes | 9,233 | 13,281 (+44%) |
| nodes / edges | 23,689 / 11,131 | 27,777 / 14,978 |
| `undercuts` | 33 | **119** |
| `refutes` | 8 | **21** |

The repro hint now fires on `other` sections, 58% of all chunks, which accounts for most of
the `Claim` increase. **Iteration 2's stored figures do not describe this substrate**; the old
one is recoverable at `data/processed/iteration2-backup/` (local, gitignored).

---

## 3. The system under test (§3.1)

Plan sub-questions → retrieve per sub-question → critique the evidence against the plan →
re-retrieve the gaps → synthesise. Behind the existing `System` protocol, so `06_run_eval.py`
scored it unchanged — the runner has been system-agnostic since Iteration 1 and this is what
that was for.

Three properties carry the comparison's validity:

- **Synthesis is the static arms' code, not a copy.** The arm *holds* a `VectorRAGSystem` and
  calls its evidence formatting, handle resolution and synthesis directly. A test pins the
  function identity, so a fork fails CI. The variable under test is the control flow alone.
- **The budget binds.** `RetrievalBudget.spend()` raises rather than logging an overrun — a cap
  that is merely counted is not a cap, and §5.6 has to report a cost multiple. `top_k` is 20
  rather than 60, so several retrievals stay within the same order of evidence as one static
  retrieval.
- **It fails closed, and says so.** A planner that raises or returns an empty plan degrades to
  one retrieval on the original query and sets `planner_failed`. The flag matters more than the
  fallback: a silent degradation scores as a working agent, and §4 excludes flagged queries
  from claims about plans.

Graph hints: the planner may read the `addresses` neighbourhood for candidate `Method` names,
CURATED only. That is the one thing Iteration 2 showed traversal does reliably (§4.5:
`addresses` carries 35.7% of traversal). Hints supply names to decompose over, never evidence,
so a wrong hint costs a sub-question rather than a citation.

---

## 4. Trajectory eval (§3.4)

Every metric before this scores the **output**. `must_cite_recall` cannot distinguish an agent
that planned well and synthesised badly from one that did the reverse — and for a
planner–critic loop that distinction is the object of study.

### 4.1 Nothing here asks a model, deliberately

§9.3 and §9.4 established what certifying a judged criterion costs: temperature pinned,
certification on the worst sample, and a human ceiling measured before any agreement figure is
interpretable — which for two of five existing criteria turned out to sit *at or below* the
judge. A judged trajectory criterion would inherit all of it and could not appear in §5 without
it. All four measures are arithmetic over the trace and the gold record.

### 4.2 Two measures withdrawn, and why that is reported rather than hidden

**Decomposition coverage is invalid on this corpus.** It read a perfect 1.000 across all 29
facets. Measuring the null — each facet against every *other* query's plan — shows why:

```
matched (own plan)      mean 0.808
null (stranger's plan)  mean 0.769

threshold 0.60:  matched kept 100.0%   null admitted 100.0%
threshold 0.80:  matched kept  62.1%   null admitted  29.9%
```

No threshold separates them. SPECTER is trained on scientific titles and abstracts, so short
phrases from one sub-field are mutually similar by construction. The 1.000 measured the
corpus's topical narrowness. `absolute_coverage_is_valid` makes this reproducible and prints
**before** the scores — a validity check placed underneath a number has already lost.

**Decomposition specificity replaced it, and is a diagnostic only.** A paired comparison —
does a facet's own plan beat *every* rival plan — cancels the similarity floor and clears
chance (0.24 against ~0.03 with 33 rivals). But the rival pool sets both the score and the
chance rate, and neither is constant by query type: relational reads 0.083 against all nine
dev rivals and 0.167 against the six non-relational ones, while chance moves 0.100 → 0.143.
Valid for comparing **arms on the same queries**; invalid across query types.

So §3.4 ships **two** headline measures rather than four.

### 4.3 Results on the scored runs

Three `agentic` repeats, n=34 each. Both figures replicate:

| measure | ρ vs `must_cite_recall` | p |
|---|---|---|
| `decomposition_specificity` | −0.16, −0.08, −0.16 | 0.375, 0.654, 0.369 |
| `retrieval_efficiency` | +0.40, +0.43, +0.44 | 0.019, 0.010, 0.010 |

```
decomposition specificity   0.238  0.216  0.213
retrieval efficiency        0.227  0.247  0.232      required papers per retrieval call
critique added a REQUIRED paper    4/34   7/34   6/34
critique changed the evidence     31/34  33/34  33/34
```

**The null is the informative one.** How distinctive a plan is carries **no information** about
whether its query is answered well — consistently, across three independent runs. That closes
off the obvious reading of §1: the +0.250 is not the arm writing better plans.

`retrieval_efficiency` survives with its definitional overlap stated: it counts required papers
*reached* per call while `must_cite_recall` counts required papers *cited*, so they share a
numerator term. Not wholly circular — Iteration 2 measured a reach-to-cite conversion of 33%
for the typed-graph arm — but partly.

---

## 5. Agentic vs static — the deliverable (§3.5)

### 5.1 Frozen configuration and pre-registration

Recorded before the run, at commit `c8f7a50`.

| component | value |
|---|---|
| planner / critic | `gpt-5.4-nano`, temperature 0.0 |
| synthesis | `gpt-5.4-mini`, provider default; prompt shared verbatim with the static arms |
| retrieval budget | 6 per query, hard ceiling, enforced |
| `top_k` per retrieval | 20 |
| anchor | on |
| graph hints | on, CURATED only |
| critique | on (`agentic`) / off (`agentic_no_critique`) |
| STAGED writes | **off** — the deliverable must not write to the graph it is scored against |

Stated before any 34-query result was seen:

- **The required relational breakdown is underpowered.** The 14 relational queries split
  **5 / 9** on whether their gold `key_claims` name a limitation or cost. A flat breakdown
  therefore reads as underpowered, and a large apparent gain at n=5 is **not** confirmation.
- **Only `coverage` may be reported** among judged criteria.
- **Three repeats per arm**, because §4.1 of Iteration 2 and §6 of Iteration 2 were each a
  single draw.
- **Cost per query beside every score.**
- A uniform gain across all query types is to be reported as an evidence-volume effect.

Not permitted afterward: changing the planner, critic, budget, `top_k`, models, anchor or
graph hints and re-running. Fixing the *measuring instrument* remains legitimate — §4.2 found
a gold-id mismatch that made one measure structurally zero, and correcting that is not tuning.

### 5.2 The result

Given in §1. Statistics are paired per query with repeats averaged **before** pairing: pooling
three repeats as independent observations would treble the apparent n while the questions stay
the same 34.

### 5.3 By query type

| arm | lookup | open-dir | refutation | **relational** |
|---|---|---|---|---|
| `agentic` | 0.556 | 0.433 | 0.426 | **0.528** |
| `agentic_no_critique` | 0.500 | 0.400 | 0.463 | 0.437 |
| `typed_graph_chunks` | 0.667 | 0.600 | 0.407 | 0.345 |
| `citation_graph` | 0.833 | 0.467 | 0.370 | 0.317 |
| `vector_fulltext` | 0.556 | 0.500 | 0.556 | 0.278 |

`agentic` is worst or near-worst on three of four types and first by a wide margin on the
fourth.

### 5.4 The required breakdown — does not support the mechanism

| | names a limit (n=5) | does not (n=9) |
|---|---|---|
| Δ vs baseline | +0.167 | **+0.296** |

§4.5 of Iteration 2 predicted the gain would concentrate where the gold names a limitation,
because that is where the missing `undercuts` edge would have been needed. It is **larger where
none is named**. Pre-registered as underpowered, so this cannot refute the account — but it
does not support it.

### 5.5 The ablation

Removing the critique costs relational 0.528 → 0.437: **+0.091, 8 / 1 / 5, p=0.043.**

**And it is unexplained.** The critique changes the evidence on 31–33 of 34 queries but adds a
*required* paper on only 4–7 (§4.3). A step that rarely reaches new required papers should not
be worth 0.091, and no measure here says why it is. Under Bonferroni across the two
pre-registered tests (α=0.025), this does not survive; §1's relational result does.

### 5.6 Cost

**1.2×** — $0.028 against $0.024 per query. `vector_fulltext` sends 60 chunks to synthesis
while the agentic arm sends a deduped set plus two `gpt-5.4-nano` calls, and those roughly
cancel. An earlier estimate of 13× in the working notes was wrong: it compared against an
Iteration 2 per-arm average instead of measuring the baseline.

### 5.7 The judged criterion

`coverage` scored on repeat 1 of each arm via `rejudge.py`, v1 rubric at temperature 0. Means
in §1. Paired against `vector_fulltext`: all 34 **+0.21, p=0.143**; relational +0.29 (only 5
non-tied, untestable); non-relational +0.15, p=0.562.

**It corroborates the ranking and not the pattern.** `agentic` is first here too, but nothing
is significant and `coverage` shows a weak *uniform* lead where `must_cite_recall` shows the
arm to be worse on non-relational. Two reasons: the metrics ask different questions — facets
addressed versus required papers cited — and `coverage` has almost no resolution at this n,
with the judge assigning both arms the same integer on **21 of 34 queries**. §1 rests on the
deterministic metric.

One incidental finding: `typed_graph_chunks` is **last on `coverage` while second on
`must_cite_recall`** — reaching required papers while addressing gold facets least well of any
arm.

---

## 6. The mechanism is not edge substitution

§4.5 of Iteration 2 proposed that decomposition works by *substituting for the missing
`undercuts` edges*. Four measurements test that account. None supports it.

**(a) Adding the edges does not work.** §9.1 rewrote the contradiction prompt with twelve
labelled spurious `refutes` as worked negatives. Edge precision held at **32.5%** — identical
stratum rates under both prompts — while the revision discarded ~65% of the real edges.

**(b) Edge coverage was not the binding constraint.** The rebuild took `undercuts` from 33 to
119, and traversal from **zero occurrences across 34 queries** to six. `typed_graph_chunks`
scored **0.383 → 0.383**, unchanged to three decimals. Edge types traversed on the dev 10:

```
addresses 43.6%   provides 33.5%   builds_on 8.9%   evaluated_on 7.6%
uses 3.8%         undercuts 2.5%   refutes 0
```

Only 1 of those 6 `undercuts` traversals falls on a relational query. A side effect worth
noting: `provides` is now a third of all traversal, a consequence of §9.2 routing
`ReproducibilityArtifact` into most sections — spent budget under `max_nodes=150` on queries
that are not about code availability, and a plausible reason the extra evidence did not convert.

**(c) The gain does not concentrate where predicted** (§5.4).

**(d) Plan quality does not predict outcome** (§4.3).

**What survives as a hypothesis**, untested: the gain comes from issuing several retrievals
against the *parts* of a question, not from the plan being good. Volume alone cannot explain
it — the arm is worse on non-relational queries while spending the same retrievals. So it
appears to be the act of splitting rather than the craft of the split. An arm that splits
mechanically on conjunctions, with no planner, would separate the two.

---

## 7. Query-time STAGED writes (§3.2)

`SourceLayer.STAGED` existed from Iteration 1, `promote_staged()` was implemented, and grep
found STAGED written **nowhere**.

The agent now persists its decomposition: the query as a `Problem` node, each sub-question as
another, `addresses` edges between them, and per sub-question the papers it reached. That last
field records which papers *jointly* bear on one part of a question — something per-paper
extraction structurally cannot observe, because it never holds two papers at once. Ids are
content-addressed, so re-running a query does not grow the graph; a failed planner stages
nothing, because a fallback to one retrieval is not a decomposition.

**The item found that the invariant it depends on was never enforced.** `stores/base.py` had
promised since Iteration 1 that metrics query CURATED only. **Neither retrieval arm filtered on
`source_layer`** — and each was missing it in two places, seeding and expansion. The promise
held solely because nothing had ever written STAGED. It was vacuously true, and the first
query-time write would have let an agent raise its own score by writing to the graph it is
scored against — the self-grading Iteration 2 refused when it declined to pick `must_cite` with
the system's own retriever.

Acceptance, with real staged data rather than argued:

```
before staging   filtered 20688   unfiltered 20688
38 nodes staged from 10 real trajectories
after  staging   filtered 20688   unfiltered 20726
```

What the arms read is unchanged *while STAGED is present and visible without the filter*, so
the filter is what separates them rather than the absence of data. **Honest limit:** this
exercised the node filter only. Staged edges connect staged nodes to each other, so nothing
curated points at them and the edge filter — a second lock — was not tested by it.

Promotion now refuses without `approved=True`, on §8.2's precedent: a 3,072-edge proposal that
would have entered the graph on file presence alone, stopped only by a flag, later measured at
32.5% precision.

---

## 8. Local inference (§3.3) — deferred, justification refuted

The stated rationale does not hold. §3.3 assumed a loop issuing 5–10 calls per query and an
affordability problem; the measured arm makes **3 calls at $0.028/query**, and the whole
deliverable cost ~$6.

The item is kept because its acceptance criterion is a **portability** claim, which nothing
else addresses. The `base_url` plumbing is landed — vLLM serves the OpenAI API, so *"route
chat calls to vLLM"* **is** *"point an OpenAI client elsewhere"*, and that code was required
whichever hardware eventually runs it. `get_local_chat_client()` raises rather than falling
back to the hosted provider, because a `--local` run that quietly used the hosted model would
report a cost and latency delta of zero and read as success.

What waits is the run, on hardware: `Qwen2.5-14B-Instruct-AWQ` needs ~8.5 GB in 4-bit and
CUDA-only kernels, against an 8 GB M2 where Metal caps the usable working set near 5.3 GB. Not
a missing-backend problem — MLX and llama.cpp run natively on Apple Silicon and 14B still does
not fit. A test pins the configured model, because substituting a smaller one that fits would
report a different experiment under §3.3's name.

---

## 9. Maintenance track (§3.6)

Four carry-forwards from Iteration 2. Three refuted.

### 9.1 Contradiction pass v2 — refuted

`_SYSTEM_V2` carries the twelve labelled spurious `refutes`, generalised into seven categories:
each paper stating its own scope; different modelling choices; claims that agree; unrelated
systems sharing vocabulary; a paper's own contribution vs another's result; a stated need
beside a stated limitation; and author-contribution boilerplate.

Sixty fresh pairs from the v2 verdicts — 20/20/20, seed 7, **zero overlap** with the sixty
behind §8.2 — labelled by hand, blind. The zero overlap is load-bearing: those twelve pairs are
inside the v2 prompt, so scoring on them would be testing on training data.

| | equal-n | population-weighted | real edges |
|---|---|---|---|
| v1 (§8.2) | 32.5% | 25.8% | ~792 of 3,072 |
| **v2** | **32.5%** | **25.9%** | ~304 of 1,172 |

Precision did not move — the stratum rates are identical. And the 62% rejection rate that
looked promising is a loss: of the 1,900 pairs v2 rejected, **6 of 20 sampled are real
disagreements** against v1's own discard rate of 2 of 20. v1's 3,072 hold ~874 real edges; v2
keeps ~304 and discards ~570 for no gain.

**§8.2's 32.5% is corroborated, and an intermediate doubt was wrong.** A `gpt-5.4-mini`
labeller found only 2 of 15 real disagreements, which read as widening §8.2's error bars. The
human pass settles it the other way: on 60 *different* pairs a human independently finds 19
disagreements and reproduces both stratum rates. §8.2's model labeller was adequate; the
`gpt-5.4-mini` labeller is too conservative to audit anything, and `--validate-labeller` is
what caught that.

Two prompts, one with a prose warning and one with seven worked categories drawn from labelled
failures, land on the same number. §8.2's design note predicted it: contradictory claims are
similar *by construction*, so similarity is purely a recall filter and the model performs the
entire discriminative step. Pairwise claim adjudication at 0.90 cosine yields ~26–32% edge
precision here regardless of prompt. Neither version's edges are applied; §4.3 of Iteration 2
stands unimproved.

### 9.2 `ReproducibilityArtifact` routing — partly confirmed

§7.2 called it *"a prompt-routing gap… so it should be cheap"*. It is three faults.

1. **Routing.** Reachable only from `availability` and `appendix`. Four of the five gold papers
   that state a `code_url` state it in a conclusion, an abstract or the results. Now routed to
   `abstract`, `method`, `results`, `conclusion` and the `other` default — `other` alone carries
   repo URLs for 14 papers.
2. **The hint gate followed `Hardware`.** `_REPRO_HINT` carries the field list and was emitted
   only when `Hardware` was in the section's types, so a section could be asked for the node
   type while being told nothing about which attributes to fill.
3. **The schema could not express the answer.** `DatasetAccess` offered
   `open|licensed|irb|unknown`; two gold records author `on request`, which is none of them.
   **Those two fields were unscoreable by any extraction** — a ceiling of 4 of 6 before the
   extractor was involved. 15 corpus papers use that wording.

| field | before | corpus after |
|---|---|---|
| `code_url` | 0 | **2** |
| `dataset_access` | 0 | **2** |
| total correct | 8 | **11** |
| accuracy | 73.2% | 74.6% |

`ReproducibilityArtifact` nodes 34 → 78. `qubit_count` went 3 → 1, at the low end of the 2/1/2
the scratch runs gave — a small real regression, since the longer hint appears to cost some
`Hardware` attention. And **abstract routing does not work**: `60f69f1c` and `cca36fcf` state
their URLs plainly in the last sentence of the abstract and failed 3 of 3 after being routed.
`60f69f1c` shows the mechanism — it emitted `ReproducibilityArtifact` nodes carrying
`{name, version}`, a `Software` payload, for Qiskit and PySCF, because `abstract` is not routed
to `SOFTWARE` and the model used the nearest slot. `dataset_access: open` is also never
inferred from a public repo URL, which may be a gold question rather than an extraction one.

### 9.3 `attribution` rubric — refuted

§6 diagnosed range restriction and prescribed *"a rubric with anchored examples at both ends,
not rescaling"*. Three versions, same 34 hand grades, deterministic judge:

| rubric | κ | ρ | judge distribution 1→5 | mean |
|---|---|---|---|---|
| human grades | — | — | 11 · 3 · 10 · 2 · 8 | 2.79 |
| **v1** (unchanged) | **+0.45** | +0.55 | 1 · 8 · 12 · 13 · 0 | 3.09 |
| v2 (per-claim anchors) | +0.29 | +0.58 | 5 · 24 · 5 · 0 · 0 | 2.00 |
| v3 (v2, low anchor fixed) | +0.35 | +0.43 | 1 · 13 · 11 · 9 · 0 | 2.82 |

**The unanchored original is the best of the three**, so the default was set back to it with a
test recording why. Across every version and sample the judge returned 5 **zero times** where
the human returned it eight — the specific defect anchoring was meant to cure never budged.

v2's defect, for the record: it sent an answer to 1 for *"the same handle repeated after
sentences that assert several different things"*. But a paragraph drawing several assertions
from one excerpt and marking them with that excerpt's handle is *correct* attribution and is
what nearly every answer here does, so the low anchor fired on everything. The judge graded the
human's cleanest 5 a 2, reasoning that *"the second sentence uses the same handle for a
separate assertion"*. That diagnosis came from the judge's persisted justifications — which v1
discarded, and which `rejudge.py` now writes.

**Consequence for §6.** Same answers, same v1 rubric, judge at temperature 0:

| criterion | §6 reported | at temp 0 | verdict |
|---|---|---|---|
| `coverage` | +0.72 OK | **+0.76** | holds |
| `synthesis` | +0.68 OK | +0.63 | holds on the 34 only |
| `refutation_handling` | +0.65 OK | **+0.44** (p=0.185) | **does not hold** |
| `hedging_accuracy` | +0.55 — | +0.25 | untrusted, as reported |
| `attribution` | +0.34 — | +0.45 | untrusted, as reported |

The five criteria are scored in one call and are not independent: `synthesis` is byte-identical
between v1 and v3, enforced by a test, and still moved +0.63 → +0.55 when only the
`attribution` text changed.

### 9.4 Second annotator — closed permanently, run as test–retest

There is no second annotator and there will not be one: this is a single-person project. §10's
threat becomes a **standing limitation of the work** rather than an open action.

The second pass was therefore the original grader, blind, two days later — test–retest, not
inter-annotator agreement. The sheet withheld the first grades, the judge's scores and the
retrieved evidence (the first grader never saw it, §3.1, so showing it would measure that
asymmetry), and was shuffled across strata. 20 of the 34, stratified proportionally.

| criterion | grader vs **self** | judge vs pass A | judge vs pass B | n |
|---|---|---|---|---|
| `coverage` | **+0.81** | +0.65 | +0.72 | 20 |
| `synthesis` | **+0.77** | +0.76 | +0.69 | 20 |
| `attribution` | **+0.29** | +0.30 | +0.29 | 20 |
| `hedging_accuracy` | **+0.39** | +0.66 | +0.52 | 20 |
| `refutation_handling` | +0.57 | +0.33 | +0.15 | **5** |

**`attribution` is at the human ceiling.** The grader reproduced 5 of 20 grades and 6 moved by
two points or more; the distribution relocated rather than wobbled (mean 3.15 → 4.20, and all
five of pass A's 1s came back 2/3/4/4/5). The answers that moved furthest are exactly the
handle-dense, coarsely-mapped ones — the same strict-vs-lenient axis v1 and v2 encoded. The
grader used one reading in August and the other two days later without noticing. So +0.30 is
not the judge falling short of the human; **it is the human number**, and no rubric can close a
gap that does not exist. This retires §9.3's hypothesis permanently rather than parking it.

The recall threat cuts the right way: two days' distance means partial recall was likely, and
recall *inflates* agreement, so +0.29 surviving that tailwind makes a low reading decisive.

**`hedging_accuracy` fails differently** — range restriction in the *human*, the mirror of what
§6 diagnosed in the judge. Three-quarters of answers get a 4 in both passes, so quadratic κ has
almost no beyond-chance signal to explain even though the passes are close.

**`coverage` and `synthesis` are confirmed in the strong form:** the grader reproduces them
with no grade moving two points, and the judge agrees with both passes alike.

---

## 10. Corrections to the Iteration 2 report

**The Iteration 2 report is not amended.** It opens the Iteration 3 plan's chain of citations,
and editing it in place would retroactively change what that plan refers to and erase the
record of what was known when the decisions were made. A report that silently updates itself
cannot be audited.

| Iteration 2 claim | status |
|---|---|
| §6 `refutation_handling` κ=+0.65, **trusted** | **superseded** — +0.44, p=0.185 (§9.3) |
| §6's table generally | every figure is one draw from a temperature-1.0 judge; spreads to 0.25 |
| §6 *"requires anchored examples… not rescaling"* | **refuted** — the unanchored original scores best |
| §4.1's n=10 `vector_fulltext` 0.417 | **a single lucky draw** — three runs give 0.367, and the old substrate produced 0.367 too |
| §10 *"no inter-annotator agreement measure"* | **permanent**, and now quantified by test–retest |
| §10 *"two of five criteria fail calibration"* | four of five on a deterministic judge |
| §10 *"§8.2 was audited by a model, not a human"* | **strengthened** — a human audit of 60 fresh pairs reproduces 32.5% |
| §7.2 `code_url` 0-for-15 as *a* routing gap | three faults; fixed and applied, now 2 (§9.2) |
| §4.5 *"decomposition substitutes for missing edges"* | the effect replicates; **the mechanism does not** (§6) |

---

## 11. Threats to validity

**Scope: this does not evaluate graph-based global summarisation.** All 34 gold queries carry a
`must_cite` list — every question has specific papers that constitute its answer, and the
headline metric counts required papers cited. This is an evaluation of graph-based **retrieval
on citation-grounded queries**.

It is *not* an evaluation of the GraphRAG family, which uses the graph differently:
hierarchical community detection, pre-generated community summaries, and a map-reduce over
those summaries for corpus-wide sensemaking. None of that is implemented, and this gold set
could not measure it — a good global answer might legitimately cite forty papers or none in
particular, and `must_cite_recall` would score it a failure either way. The negative results
about the typed graph bear on it as a **retriever**, not as a summarisation scaffold, and should
not be read as evidence against that approach.

The convergence is worth noting: the graph's one durable contribution here was **planner hints**
(§3), while it repeatedly failed to justify its cost as an evidence retriever. That is
independent support for the proposition that a graph is better at orientation than at
retrieval — roughly GraphRAG's premise.

**Multiple comparisons.** Two tests were pre-registered. Under Bonferroni at α=0.025, §1's
relational result (p=0.012) survives and §5.5's ablation (p=0.043) does not.

**n=14 relational**, 5 of those in the required breakdown group.

**One judged criterion, with almost no resolution.** `coverage` ties two-thirds of queries
(§5.7).

**Three repeats** bound the run-to-run spread; they cannot establish determinism. Temperature 0
is not bit-determinism.

**Single grader, permanently** (§9.4).

**The measured artifacts are not in the repository.** `data/processed/*` and `eval/runs/*` are
gitignored, so the graph, the extractions and all 20 run directories behind these figures are
local only. The numbers are reproducible from the pipeline but not currently auditable from the
repo alone.

---

## 12. Engineering record — 28 fixes

Grouped by area. The pattern worth naming: **seven produced a plausible number rather than an
error**, and in four cases the thing that caught a later mistake was a fix that looked like
housekeeping when it was made.

### Sampling was never disabled (7)

| # | defect | how it was found |
|---|---|---|
| A1 | The judge ran at the provider default. | Three identical draws over 34 answers gave κ spreads to 0.25 against a 0.26 gap to the bar. |
| A2 | Extraction had the same defect, so the graph was unreproducible. | Two identical runs over 21 papers differed on 9 of 147 field outcomes. Pinned *before* the rebuild. |
| A3 | The planner is nondeterministic and pinning does not fix it. | Three draws, three wordings; the same code scored 0.567 and 0.333. Fixed by the anchor (§2.2). |
| A4 | Criteria were certified from a single draw. | `--repeats` added; certification requires the **worst** sample to clear the bar. |
| A5 | The judge's justifications were requested and discarded. | The only reason v2's defect was diagnosable (§9.3). A score cannot explain itself. |
| A6 | No way to re-score stored answers, so a rubric change could not be isolated from resampled synthesis. | `rejudge.py`; never modifies the source run. |
| A7 | Rubric versions were not retained, leaving no control arm. | v1 kept and selectable. **Outcome: v1 scored best of three**, so the default reverted, with a test recording why. |

### Gold-set handling (3)

| # | defect | consequence |
|---|---|---|
| B1 | Scripts defaulted to the 10-query file even for a 34-query run. | Would have scored a third of the data as the whole. Now selects the smallest covering file, or errors. |
| B2 | Calibration figures depended on which set was used, unstated. | The sets disagree on which criteria pass. `calibrate_judge.py` now always prints the complement. |
| B3 | `06_run_eval.py` hardcoded the active gold file. | `--gold` added; non-default runs tagged in the directory name. |

### Reproducibility layer (3)

C1 routing, C2 the hint gate, C3 the missing `on_request` enum member — see §9.2.

### Contradiction tooling (4)

| # | defect | consequence |
|---|---|---|
| D1 | The verdict cache key recorded the claim pair but not the prompt version. | A v2 run would have returned 16,965 **v1** verdicts as v2's. One command from a false finding. |
| D2 | `--score` read the accepted total from the old edge file regardless of what it audited. | Reported "of 3,072 accepted" for a pass that accepted 1,172. |
| D3 | One hardcoded input and output file. | Auditing a second attempt would have destroyed the record of the first. |
| D4 | §8.2's labeller was never itself validated. | `--validate-labeller` caught that it finds **2 of 15** real disagreements (§9.1). Without it the number would have been reported. |

### The layer invariant (3)

E1 `typed_graph` unfiltered on seeding *and* expansion; E2 the same in `citation_graph`;
E3 `promote_staged` ungated. See §7.

### Trajectory measures (2)

| # | defect |
|---|---|
| F1 | Gold `must_cite` is `paper:<id>`; the trace records bare ids. Retrieval efficiency read **0.000 for every query as a plausible number**. The strongest argument for §4 preceding §5. |
| F2 | Decomposition coverage read 1.000 on all 29 facets and was invalid (§4.2). |

### Plumbing (5)

| # | change |
|---|---|
| G1 | `04_extract.py --papers/--out` — prices a routing change at ~$0.36 instead of $6.72 |
| G2 | `SystemOutput.trace` + runner persistence; static arms leave it empty so their traces stay byte-identical |
| G3 | `runner._write_report` → `write_report`, shared with `rejudge.py` |
| G4 | `base_url` through the LLM layer, and `temperature` through all three clients |
| G5 | `resolve_gold` moved into `gold_schema`, since two scripts need it and `scripts/` is not importable |

### Documentation contradicting the measurements (2)

| # | defect |
|---|---|
| H1 | `README.md` certified `hedging_accuracy` and rejected `synthesis` — both backwards — and reported 10 gold queries, 23,460 nodes, 42 tests, and "Next (Iteration 2): entity resolution". Its opening asserted the typed-graph bet as untested. |
| H2 | The co-citation section explained how it made the typed-vs-citation ablation fair, and never recorded that the ablation was lost. |

### One fix found by writing this record

`compare_arms.py` computed p=0.012 with its statistics inside the script: not importable, not
unit-tested. Every other measure in this iteration keeps pure logic in `src/` with a thin
driver in `scripts/`, which is what let them be tested with fakes and no network. Extracted to
`rpsg.eval.comparison` with 16 tests, including a hand-calculable case — six distinct positive
differences give exact two-sided p = 2/2⁶ = 0.03125, which would catch a switch to a normal
approximation, a lost two-sided correction, or ranking by signed rather than absolute value.

Writing the regression lock also caught a bug in the regression lock: the first version used
differences that reproduced the reported 8/1/5 but were guessed, returned p=0.0352, and
asserted only `p < 0.05` — so it passed while locking nothing. Wilcoxon ranks by the *size* of
each difference. The real per-query scores are now copied from the run data and the assertion
pins p=0.0117.

Tests: **42 → 230**, ruff and mypy clean.

---

## 13. Carried forward

| item | why it is open |
|---|---|
| §3.3's vLLM run | needs CUDA hardware; config change plus one run (§8) |
| A mechanically-splitting arm | would separate "splitting helps" from "planning helps" (§6) |
| Why the critique is worth +0.091 | changes the evidence 31–33 times, adds a required paper 4–7 (§5.5) |
| Global-sensemaking queries and metrics | prerequisite for evaluating anything GraphRAG-shaped (§11) |
| 60 human contradiction labels | would upgrade §8.2's model-labelled figure on the original sixty |
| Committing the scored runs | the evidence behind every figure here is local only (§11) |
| Merging the branch | 28 commits on `iteration-3-maintenance`; `main` is 55 behind |
