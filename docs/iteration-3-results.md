# §3.5 — agentic vs static, the deliverable

Run once on `queries.full34.jsonl`, 5 arms × 3 repeats, deterministic metrics. 3.1 was frozen
first ([iteration-3-freeze.md](iteration-3-freeze.md)) and nothing has been changed since.

## Result

`must_cite_recall`, mean of 3 repeats, spread across repeats:

| arm | mean | spread | cost/query |
|---|---|---|---|
| **`agentic`** | **0.492** | 0.461 – 0.539 | $0.028 |
| `typed_graph_chunks` | 0.456 | 0.422 – 0.480 | — |
| `agentic_no_critique` | 0.449 | 0.436 – 0.466 | — |
| `citation_graph` | 0.444 | 0.431 – 0.456 | — |
| `vector_fulltext` | 0.433 | 0.412 – 0.446 | $0.024 |

**Overall the lead is not significant.** Paired over all 34 queries, `agentic` beats
`vector_fulltext` by +0.059 with 12 wins, 9 losses and 13 ties, p=0.403.

## The lead is real, and it is entirely relational

| subset | Δ vs `vector_fulltext` | W / L / T | p |
|---|---|---|---|
| **relational (n=14)** | **+0.250** | 8 / 1 / 5 | **0.012** |
| non-relational (n=20) | **−0.075** | 4 / 8 / 8 | 0.340 |

By query type:

| arm | lookup | open-dir | refutation | **relational** |
|---|---|---|---|---|
| `agentic` | 0.556 | 0.433 | 0.426 | **0.528** |
| `agentic_no_critique` | 0.500 | 0.400 | 0.463 | 0.437 |
| `typed_graph_chunks` | 0.667 | 0.600 | 0.407 | 0.345 |
| `citation_graph` | 0.833 | 0.467 | 0.370 | 0.317 |
| `vector_fulltext` | 0.556 | 0.500 | 0.556 | 0.278 |

`agentic` is **worst or near-worst on three of four types** and first by a wide margin on the
fourth. That matters for the plan's outcome table, which flagged a uniform gain as *suspect —
that is more retrieval, not better planning*. This is the opposite of uniform: the arm is
measurably **worse** on non-relational queries, so the gain cannot be an evidence-volume
effect.

## The critique earns its place

The required ablation, on relational: **+0.091, 8 wins / 1 loss / 5 ties, p=0.043**. Removing
the critique costs most of the relational advantage (0.528 → 0.437), which puts
`agentic_no_critique` back among the static arms.

## Cost

**1.2×** — $0.028 against $0.024 per query. Not the order-of-magnitude premium anticipated:
`vector_fulltext` sends 60 chunks to synthesis while the agentic arm sends a deduped set plus
two `gpt-5.4-nano` calls, and those roughly cancel. A +0.250 relational gain at 1.2× is a
materially different claim from the same gain at 10×.

## Against the pre-registered outcome table

| plan's row | verdict |
|---|---|
| beats static on relational **and the gain concentrates in queries whose gold names a limitation** → supported | **first half yes, second half no** |
| matches static at higher cost → refuted | no — it beats static on relational, at 1.2× |
| beats static **uniformly** → suspect, evidence-volume | ruled out — worse on non-relational |

**The required breakdown does not support §4.5's mechanism.** Splitting the 14 relational
queries on whether their gold `key_claims` name a limitation or cost:

| arm | names a limit (n=5) | does not (n=9) |
|---|---|---|
| `agentic` | 0.500 | 0.543 |
| `vector_fulltext` | 0.333 | 0.247 |
| Δ | **+0.167** | **+0.296** |

§4.5 predicted the gain would concentrate where the gold names a limitation, because that is
where the missing `undercuts` edge would have been needed. The gain is **larger in the group
that does not**. Pre-registered as underpowered at 5/9, so this cannot refute the mechanism —
but it does not support it either, and it was recorded as underpowered before the numbers
were seen precisely so this reading could not be constructed afterwards.

## What this establishes, and what it does not

**Established:** decomposing a relational question into sub-questions and retrieving per
sub-question improves required-paper recall on relational queries by +0.250 (p=0.012) at 1.2×
the cost, and the self-critique contributes about a third of that (p=0.043). The effect is
specific to the query type it was designed for.

**Not established:** *why*. §4.5's account — that decomposition substitutes for the missing
`undercuts` edges — is not supported by the only breakdown that tests it. Two independent
findings this iteration already weakened that account: 3.6a refuted adding the edges, and the
corpus rebuild took `undercuts` from 33 to 119 and from zero traversals to six while
`typed_graph_chunks` scored 0.383 → 0.383 unchanged. Edge coverage was necessary, not
sufficient, and it may not be the operative variable at all.

A plainer candidate mechanism, untested: a relational question asks two things, and one
retrieval on the whole question retrieves for neither half well. That would predict a gain on
any multi-part question regardless of whether the second part concerns a limitation — which
is what the 5/9 split shows.

## §3.4 on the scored runs — the plan's quality does not predict the outcome

Trajectory eval over the three `agentic` repeats, n=34 each. Both figures replicate:

| measure | vs `must_cite_recall` | p | replicates |
|---|---|---|---|
| `decomposition_specificity` | ρ = −0.16, −0.08, −0.16 | 0.375, 0.654, 0.369 | **no relationship, 3/3** |
| `retrieval_efficiency` | ρ = +0.40, +0.43, +0.44 | 0.019, 0.010, 0.010 | **significant, 3/3** |

```
decomposition specificity   0.238  0.216  0.213      (chance is ~0.03 with 33 rivals)
retrieval efficiency        0.227  0.247  0.232      required papers per retrieval call
critique added a REQUIRED paper   4/34   7/34   6/34
critique changed the evidence    31/34  33/34  33/34
```

**The null is the informative one.** How distinctive a plan is — whether its sub-questions
match its own query's facets better than any of the other 33 queries' plans do — carries **no
information** about whether the query gets answered well. Consistently slightly negative,
never significant, across three independent runs.

That closes off the obvious reading of §3.5's result. The +0.250 relational gain is **not**
explained by the agentic arm writing better plans, because plan quality does not predict
outcome at all.

**The surviving correlate is `retrieval_efficiency`**, and it must be read with its
definitional overlap stated: it counts required papers *reached* per retrieval call, while
`must_cite_recall` counts required papers *cited*. They share a numerator term. The
correlation is therefore partly built in — though not wholly, since reaching a paper does not
imply citing it, and Iteration 2 measured exactly that gap (typed-graph retrieval reached
0.556 of required pairs and converted 33% into citations).

**What the two results together suggest**, as a hypothesis and not a conclusion: the gain
comes from *issuing several retrievals against the parts of a question* rather than from the
plan being good. Volume alone cannot be the explanation — §3.5 shows the arm is **worse** on
non-relational queries despite spending the same retrievals there. So it is the act of
splitting a multi-part question, not the craft of the split, that appears to matter. That is
testable and untested: an arm that splits mechanically on conjunctions, with no planner at
all, would separate the two.

**The critique's contribution stays puzzling.** It changes the evidence on 31–33 of 34
queries but adds a *required* paper on only 4–7, and yet removing it costs 0.091 on relational
(p=0.043). A step that rarely reaches new required papers should not be worth that much, and
this measurement does not explain why it is.

## The judged criterion: `coverage`

Scored on repeat 1 of each arm with `rejudge.py` — v1 rubric, judge at temperature 0, the
only configuration and the only criterion certified on both gold sets.

| arm | all 34 | relational | non-relational |
|---|---|---|---|
| **`agentic`** | **3.76** | **3.64** | **3.85** |
| `agentic_no_critique` | 3.71 | 3.57 | 3.80 |
| `vector_fulltext` | 3.56 | 3.36 | 3.70 |
| `citation_graph` | 3.47 | 3.36 | 3.55 |
| `typed_graph_chunks` | 3.32 | 3.07 | 3.50 |

Paired against `vector_fulltext`:

| subset | Δ | W / L / T | p |
|---|---|---|---|
| all 34 | +0.21 | 9 / 4 / 21 | 0.143 |
| relational | +0.29 | 4 / 1 / 9 | — (5 non-tied, too few) |
| non-relational | +0.15 | 5 / 3 / 12 | 0.562 |

**It corroborates the ranking and not the pattern, and that is worth stating plainly.**
`agentic` is first on `coverage` too, so the two metrics agree on which arm is best. But
nothing is significant, and `coverage` shows a weak *uniform* lead (+0.15 on non-relational)
where `must_cite_recall` showed the arm to be **worse** there (−0.075). The relational
concentration that carries §3.5's result does not reproduce in the judged criterion.

Two reasons, neither flattering to the judged measure:

- **The metrics ask different questions.** `coverage` asks whether the answer addresses the
  gold facets; `must_cite_recall` asks whether it cites the required papers. An answer can
  discuss the right things while grounding them on the wrong papers, and the agentic arm's
  extra retrievals plausibly help the first more evenly than the second.
- **`coverage` has almost no resolution here.** It is an integer 1–5, and the judge assigns
  *the same score to both arms on 21 of 34 queries* — 9 of 14 on relational. A measure that
  ties two-thirds of the time cannot localise an effect to a subset of 14.

So the deterministic result stands on its own evidence and `coverage` neither strengthens nor
undermines it. `typed_graph_chunks` is worth one note: **last on `coverage` (3.32) while
second on `must_cite_recall` (0.456)** — it reaches required papers and addresses gold facets
least well of any arm, which is the reach-without-conversion pattern Iteration 2 measured
from the other direction.

## Threats

- **Multiple comparisons.** Two comparisons were pre-registered: against the baseline and the
  ablation. Under Bonferroni at two tests (α=0.025), p=0.012 survives and p=0.043 does not.
  The relational-vs-baseline result is the one to lean on.
- **n=14 relational**, 5 of them in the breakdown group.
- **One substrate, one judge-free metric.** `must_cite_recall` counts required papers cited;
  it says nothing about whether the answer reads well. `coverage` is the only judged criterion
  certified to accompany it.
- **Three repeats** bound the run-to-run spread but cannot establish determinism.
