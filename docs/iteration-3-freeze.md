# 3.1 frozen — the configuration 3.5 measures

Recorded before 3.5 runs, so the system under test is a fact in the repository rather than
whatever `HEAD` happened to be. Commit: see `git log` for the commit that adds this file.

## Frozen

| component | value |
|---|---|
| planner model | `gpt-5.4-nano`, temperature 0.0 |
| critic model | `gpt-5.4-nano`, temperature 0.0 |
| synthesis model | `gpt-5.4-mini`, provider default temperature |
| synthesis prompt | `baselines._SYNTH_SYSTEM`, shared verbatim with the static arms |
| retrieval budget | 6 per query, hard ceiling, enforced |
| `top_k` per retrieval | 20 |
| anchor | on — one deterministic retrieval on the original query |
| graph hints | on — `addresses` neighbourhood, CURATED only |
| critique | on (`agentic`) / off (`agentic_no_critique`, the required ablation) |
| STAGED writes | **off** — the deliverable must not write to the graph it is scored against |

## Substrate

Rebuilt 2026-08-14. 271 papers, extraction temperature 0.0. 27,777 nodes, 14,978 edges;
`undercuts` 119, `refutes` 21. Iteration 2's substrate is at
`data/processed/iteration2-backup/` and its numbers do **not** describe this graph.

## Pre-registered, before any 34-query result is seen

- **The required relational breakdown is underpowered.** The 14 relational queries split
  **5 / 9** on whether their gold `key_claims` name a limitation or cost. §4.5 predicts the
  gain concentrates in the 5. At that n only an enormous effect is detectable, so a flat
  breakdown reads as *underpowered*, and a large apparent gain does **not** count as
  confirmation either.
- **Only `coverage` may be reported among judged criteria.** It is the one certified on both
  gold sets. `attribution` and `hedging_accuracy` sit at or below the grader's own
  self-agreement; `refutation_handling` is uncalibrated at n=9 and unmeasurable at n=3.
- **Three repeats per arm**, deterministic metrics. Every figure carries a spread, because
  §4.1's own n=10 number turned out to be a single lucky draw and §6's table was one draw
  per criterion.
- **Cost per query is reported beside every score.** The agentic arm costs roughly 13× the
  static arms. A win at 13× is a different claim from a win at 1.2×.
- **The outcome table in the plan stands as written.** A uniform gain across all query types
  is to be reported as an evidence-volume effect, not as decomposition working.

## What is not permitted after this point

Changing the planner, the critic, the budget, `top_k`, the models, the anchor or the graph
hints and re-running 3.5. The 34 is spent once. If the result disappoints, the honest next
step is a new gold set or a different mechanism — not another look at this one.

Fixing the *measuring instrument* remains legitimate: 3.4 found a gold-id mismatch that made
retrieval efficiency structurally zero, and correcting that is not tuning the system.
