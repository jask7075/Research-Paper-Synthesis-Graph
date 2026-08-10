# Iteration 3 — maintenance track (§3.6)

Carry-forwards from Iteration 2, run alongside the agentic work. Each item is closed here
with its measurement, whether or not the measurement is the one the plan hoped for.

| # | item | status |
|---|---|---|
| 3.6a | contradiction pass v2 | not started |
| 3.6b | `ReproducibilityArtifact` routing | scoped, not started — see below |
| 3.6c | `attribution` rubric | **closed: hypothesis refuted, instrument fixed** |
| 3.6d | second annotator | **closed as test–retest: two criteria confirmed, two retired** |

**The two closed items answer each other.** 3.6c could not make the judge agree with the
grader on `attribution` past +0.45. 3.6d measured why: the grader agrees with *themselves*
at **+0.29**, and the judge already sits at +0.30. The judge is at the human ceiling, and no
rubric can pass it.

---

## 3.6c `attribution` rubric — refuted, and a larger defect found underneath

**What the plan asked for.** §6 put `attribution` at κ=+0.34 and diagnosed range
restriction rather than bias: the judge never returned 5 and almost never 1, against a
bimodal human distribution. The prescription was *"a rubric with anchored examples at both
ends, not rescaling"*, re-graded against the existing 34 hand grades.

**Result: the prescription is wrong.** Three rubric versions, measured on the same 34 hand
grades with a deterministic judge:

| rubric | attribution κ | ρ | judge distribution 1→5 | mean |
|---|---|---|---|---|
| human grades | — | — | 11 · 3 · 10 · 2 · 8 | 2.79 |
| **v1** (Iteration 2, unchanged) | **+0.45** | +0.55 | 1 · 8 · 12 · 13 · 0 | 3.09 |
| v2 (verifiability, per-claim anchors) | +0.29 | +0.58 | 5 · 24 · 5 · 0 · 0 | 2.00 |
| v3 (v2 with the low anchor corrected) | +0.35 | +0.43 | 1 · 13 · 11 · 9 · 0 | 2.82 |

The original rubric is the best of the three. None clears the 0.6 bar, so `attribution`
stays untrusted — the same verdict as §6, now established rather than assumed.

**The top of the scale never opened.** Across every version and every sample, the judge
returned 5 **zero times** in 34 answers, where the human returned it eight times. v3
recentred the mean onto the human's (2.82 vs 2.79) and v2 improved the *ranking*
substantially (ρ +0.39 → +0.58, p<0.001), so the rubric text demonstrably moves what the
judge does. It moves offset and order; it does not move agreement-on-level. A fourth
rewrite is not the next thing to try.

**What v2 got wrong, for the record.** v2 sent an answer to 1 for *"the same handle
repeated after sentences that assert several different things"*. But a paragraph drawing
several related assertions from one excerpt and marking them with that excerpt's handle is
correct attribution, and it is what nearly every answer in this corpus does — so the low
anchor fired on everything and nothing could score above 3. The judge graded the human's
cleanest 5 (`qec-c01`, a faithful two-sentence single-source answer) a 2, reasoning that
*"the second sentence uses the same handle for a separate assertion"*. That diagnosis came
out of the judge's persisted justifications, which is why `rejudge.py` now writes
`judgements.jsonl` — v1 discarded them, and a score alone cannot explain itself.

### The defect underneath: the judge was sampled at temperature 1.0

Nothing in the project set a temperature. Every LLM call — extraction, synthesis, and every
κ in §6 — ran at the provider default of 1.0.

Judging the same 34 answers **three times with a byte-identical rubric**:

| criterion | κ per sample | spread |
|---|---|---|
| `coverage` | +0.75 +0.72 +0.71 | 0.04 |
| `attribution` | +0.15 +0.35 +0.39 | **0.25** |
| `hedging_accuracy` | +0.37 +0.24 +0.32 | 0.13 |
| `refutation_handling` | +0.62 +0.47 +0.40 | **0.22** |
| `synthesis` | +0.57 +0.49 +0.63 | 0.14 |

The distance from `attribution`'s +0.34 to the 0.6 bar is 0.26. The measurement noise was
0.25. **§6's table is one draw per criterion presented as a measurement**, and three of the
five spreads are wide enough to move a criterion across the trust threshold on their own.

`models.judge_temperature` now pins the judge at 0.0. Extraction and synthesis are
deliberately left at the provider default: changing their sampling would make Iteration 2's
stored runs non-comparable for no measured benefit. With the judge pinned, spreads fall to
0.02–0.12.

### Consequence: §6 certified three criteria; one survives

Same 34 answers, same v1 rubric, judge at temperature 0:

| criterion | §6 reported | v1 @ temp 0 | verdict |
|---|---|---|---|
| `coverage` | +0.72 OK | **+0.76** | holds |
| `synthesis` | +0.68 OK | **+0.63** | holds, but carries a significant length bias (+0.02/100 chars, p=0.028) and falls to +0.55 under v3 |
| `refutation_handling` | +0.65 OK | **+0.44** | **does not hold** — n=9, p=0.185 |
| `hedging_accuracy` | +0.55 — | +0.25 | untrusted, as reported |
| `attribution` | +0.34 — | +0.45 | untrusted, as reported |

`refutation_handling` was certified in §6 and does not survive a deterministic re-measure at
n=9. §6's closing line — *"the `coverage` and `synthesis` columns are usable across arms"* —
holds for `coverage`; `synthesis` holds on v1 alone and should be reported with its length
bias attached.

**A methodological note that matters for 3.4.** The five criteria are scored in one call, so
they are not independent. `synthesis` is byte-identical between v1 and v3 (a test enforces
this) and still moved +0.63 → +0.55 when only the `attribution` text changed. Any future
rubric edit perturbs every criterion, not just the one edited — which is why `rejudge.py`
keeps v1 as a control arm rather than deleting it.

### What this changes for Iteration 3

- **3.5 must not report `attribution`, `hedging_accuracy` or `refutation_handling` across
  arms.** Only `coverage` is certified; `synthesis` is usable with its length bias stated.
- **3.4's trajectory criteria inherit the rule.** The plan already says a new judged metric
  family starts untrusted. It should also start at temperature 0 and be certified on its
  *worst* sample, not a single draw — `calibrate_judge.py` now enforces that.
- The 0.6 threshold itself is unchanged and was not tuned.

### Reproducing

```bash
python scripts/rejudge.py 20260807T193557Z_vector_fulltext --prompt-version v1 --repeats 3
python scripts/calibrate_judge.py <the run it prints>
```

`rejudge.py` re-scores stored answers only — no retrieval, no synthesis, and the source run
is never modified, so Iteration 2's numbers stay reproducible from the directory that
produced them. Runs behind this section — local artifacts, since `eval/runs/*` is
gitignored, so the table below is the record of what produced each number:

| run | what |
|---|---|
| `20260809T193039Z_vector_fulltext_judge-v1` | v1 re-run at temp 1.0 — the drift control |
| `20260809T193336Z_vector_fulltext_judge-v2` | v2 at temp 1.0 |
| `20260809T194745Z_vector_fulltext_judge-v3` | v3 at temp 1.0, 3 samples — the stability measurement |
| `20260809T_v1_temp0` | v1 at temp 0 — the headline control |
| `20260809T_v3_temp0` | v3 at temp 0, 2 samples |

Cost: $6.75, 272 judge calls, `gpt-5.4-mini`.

### Threats to this result

- **One judge model.** Everything above is `gpt-5.4-mini`. That the *rubric* cannot open the
  top of the scale is established for this model; whether a stronger judge would return 5 is
  untested, and is the cheapest remaining hypothesis.
- **One grader — now measured, and it is the answer.** 3.6d re-graded 20 of these answers
  blind and found the grader agrees with *themselves* on `attribution` at **+0.29**, against
  the judge's +0.30. The judge is not falling short of the human standard; there is no
  single human standard to fall short of. The two readings the grader alternated between are
  the same strict-vs-lenient axis v1 and v2 encoded, which is why rewriting the rubric moved
  offset and ranking but never agreement-on-level. **This retires the rubric hypothesis
  permanently** rather than leaving it parked behind the judge-model threat above — a
  stronger judge cannot agree with a target that moves.
- **Temperature 0 is not bit-determinism.** Two samples still disagreed on 4 of 34
  attribution scores. It is a large reduction, not an elimination.
- **The 34 have now been looked at three times.** A fourth rubric measured on them is
  fitting to the test set. This is a further reason the next attempt should not be a rubric.

---

## 3.6d Second annotator — run as test–retest; two criteria confirmed, two retired

**What was actually run, and what it can therefore claim.** This is a one-person project, so
the second pass was graded by the original annotator, blind, two days later. That is
**test–retest**, not inter-annotator agreement. §10's single-grader threat is *not*
discharged and must stay in the report. What this does measure is whether the grading
standard is stable enough for any judge to be calibrated against it — and that turns out to
be the question that mattered.

The sheet withheld the first grades, the judge's scores and the retrieved evidence, and was
shuffled across strata. `--annotator retest` is recorded in `eval/gold/annotator_b.jsonl`
and echoed by the scorer, so this cannot later be read as a second annotator.

### Result

| criterion | grader vs **self** | judge vs pass A | judge vs pass B | n | verdict |
|---|---|---|---|---|---|
| `coverage` | **+0.81** | +0.65 | +0.72 | 20 | **trustworthy** — stable standard, judge tracks both passes alike |
| `synthesis` | **+0.77** | +0.76 | +0.69 | 20 | **trustworthy** — same |
| `attribution` | **+0.29** | +0.30 | +0.29 | 20 | **retire** — the judge is already *at* the human ceiling |
| `hedging_accuracy` | **+0.39** | +0.66 | +0.52 | 20 | **retire or re-specify** — below the bar, and the judge agrees with pass A better than the grader agrees with themselves |
| `refutation_handling` | +0.57 | +0.33 | +0.15 | **5** | undecidable at this n |

`eval/gold/annotator_b.score.txt` holds the full output.

### `attribution` is not a judge problem, and 3.6c is closed for good

The grader reproduced only **5 of 20** attribution grades, and **6 of 20 moved by two points
or more**. The distribution did not wobble — it relocated:

```
pass A (Aug 7)   1: 5  2: 0  3: 8  4: 1  5: 6     mean 3.15
pass B (Aug 9)   1: 0  2: 1  3: 1  4:11  5: 7     mean 4.20
```

Every one of pass A's five 1s came back 2, 3, 4, 4, 5. And the answers that moved furthest
— `rel-t03` 1→4, `rel-t04` 1→4, `look-001` 1→5 — are precisely the handle-dense,
coarsely-mapped answers, the ones where several sources are bundled behind one compound
sentence.

That is the *same axis* 3.6c spent three rubric versions oscillating on. `attribution` as
specified admits two defensible readings:

- **strict** — can a reader verify each individual claim against one individual source?
  Bundled handles fail. (Pass A's reading, and what v2 encoded.)
- **lenient** — is the mapping present and correct, whoever it covers? Bundling is fine.
  (Pass B's reading, and roughly what v1 encoded.)

The grader used one in August and the other two days later, on the same answers, without
being aware of switching. So the criterion has no single target, and the judge's +0.30 is
not a shortfall against the human — it *is* the human number. `κ = +0.29` self, `+0.30`
judge: the instrument is as consistent with the grader as the grader is with themselves.

**Consequence.** No rubric wording can close a gap that does not exist. `attribution` cannot
be repaired by prompting; it must either be split into two separately-defined criteria
(claim-level traceability vs. source correctness), each with its own anchors, or dropped.
Until then it is reported as untrusted, and 3.6c is closed permanently rather than parked.

The recall threat cuts the right way here. Two days' distance means partial recall was
likely, and recall inflates agreement. `+0.29` survived that tailwind, which makes a low
reading decisive in a way a high one would not have been.

### `hedging_accuracy` fails for a different reason: the grader barely uses the scale

Self-agreement is +0.39, but 10 of 20 grades are identical and the mean moved +0.15 — the
disagreement is small in magnitude. The problem is variance:

```
pass A   1: 1  2: 0  3: 1  4:15  5: 3
pass B   1: 0  2: 0  3: 3  4:12  5: 5
```

Three-quarters of answers get a 4. Quadratic-weighted κ measures agreement *beyond chance*,
and where a grader concentrates on one value there is almost no signal for it to explain, so
κ collapses even though the two passes are close. This is range restriction in the human
rather than in the judge — the mirror image of what §6 diagnosed. A criterion that assigns
the same score to three-quarters of the corpus is not discriminating anything, and rescaling
cannot fix that either.

### What survives, and why that is the useful half

`coverage` and `synthesis` are confirmed, and confirmed in the strong form: the grader
reproduces them (+0.81, +0.77, no grade moving by two points on either), *and* the judge
agrees with both passes at similar levels. §6 certified them from one draw against one pass;
they now hold across two human passes and a deterministic judge. Those are the two criteria
3.5 may report.

### Limits

- **§10 stands.** One grader, twice, is not two graders. `--annotator second` remains
  unexercised and the threat remains in the report.
- **`refutation_handling` is undecidable at n=5.** Proportional sampling put 5 of the 9
  contradiction-bearing queries in the sheet. Its +0.57 self-agreement sits alongside
  ρ=+0.91, so the grader ranks those five consistently while the level moves — suggestive,
  not usable. Settling it needs the full 9, and really needs more refutation queries than
  the gold set contains.
- **Test–retest is an upper bound on stability**, not an estimate, because of recall.

---
## 3.6b `ReproducibilityArtifact` routing — scoped, not started

§7.2 attributes `code_url` 0-for-15 and `dataset_access` 0-for-14 to a prompt-routing gap.
Tracing all five gold papers that state a `code_url` to the section stating it:

| paper | section | `REPRO_ARTIFACT` routed there? |
|---|---|---|
| `1db5090a` | conclusion | no |
| `60f69f1c` | abstract | no |
| `cca36fcf` | abstract + conclusion | no |
| `30a35856` | results | no |
| `91c10ab4` | **availability** | **yes** |

Four of five are the §2.4 `Hardware` pattern exactly, and the fix is the same shape: the
type is only reachable from `availability` and `appendix`. Corpus-wide, repo/archive URLs
appear in `other` (14 papers), `conclusion` (6), `abstract` (5), `method` (2) and `results`
(2) — and `other` falls through to `_DEFAULT_TYPES`, which does not list the type at all.

`91c10ab4` is **not** a routing gap and the report's framing hides it: the type *is* routed
to `availability`, that paper's availability chunk does contain the URL, and no
`ReproducibilityArtifact` node was produced. Its text reads `https: //github.com/...` — a
space injected by PDF extraction — inside what is otherwise a reference list. So there are
two failures here, and only one of them is routing.

Also worth noting for whoever picks this up: `30a35856`'s URL is on `atomgit.com`, not
GitHub. A fix that keys on `github.com` would miss it.