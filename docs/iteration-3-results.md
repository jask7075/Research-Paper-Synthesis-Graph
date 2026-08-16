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

## Threats

- **Multiple comparisons.** Two comparisons were pre-registered: against the baseline and the
  ablation. Under Bonferroni at two tests (α=0.025), p=0.012 survives and p=0.043 does not.
  The relational-vs-baseline result is the one to lean on.
- **n=14 relational**, 5 of them in the breakdown group.
- **One substrate, one judge-free metric.** `must_cite_recall` counts required papers cited;
  it says nothing about whether the answer reads well. `coverage` is the only judged criterion
  certified to accompany it.
- **Three repeats** bound the run-to-run spread but cannot establish determinism.
