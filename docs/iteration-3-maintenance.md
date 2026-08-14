# Iteration 3 — maintenance track (§3.6)

Carry-forwards from Iteration 2, run alongside the agentic work. Each item is closed here
with its measurement, whether or not the measurement is the one the plan hoped for.

| # | item | status |
|---|---|---|
| 3.6a | contradiction pass v2 | **closed: v2 fails the bar and is worse than v1; §8.2 corroborated** |
| 3.6b | `ReproducibilityArtifact` routing | **closed: 3 faults found, 2 fixed; not applied to the corpus** |
| 3.6c | `attribution` rubric | **closed: hypothesis refuted on both gold sets, instrument fixed** |
| 3.6d | second annotator | **closed. Inter-annotator agreement is out of scope permanently — see below** |

**Calibration is reported on the active 10** (`queries.jsonl`), with the 34 shown beside every
figure — the two sets disagree about which criteria pass, so neither travels alone. See
*Reporting set* below. Only `coverage` is certified for cross-arm use on either set.

**The two closed items answer each other.** 3.6c could not make the judge agree with the
grader on `attribution` past +0.45. 3.6d measured why: the grader agrees with *themselves*
at **+0.29**, and the judge already sits at +0.30. The judge is at the human ceiling, and no
rubric can pass it.

---

## Two decisions taken on this track

**The Iteration 2 report is not amended.** §6 continues to certify `refutation_handling` at
+0.65 and §10 continues to describe the single-grader threat as an open one. Both are
superseded by findings here, and neither is edited.

The reason is that the Iteration 3 plan opens *"Iteration 2 is closed; its result and every
number cited here are in iteration-2-report.md"* and then cites those numbers to justify what
Iteration 3 does. Editing them in place would retroactively change what the plan refers to
and erase the record of what was known when the decisions were made — a report that silently
updates itself cannot be audited. The corrections are carried into the **Iteration 3 report**
instead, where they belong: as findings of this iteration, with the superseded figures shown
next to them.

What the Iteration 3 report must carry:

| Iteration 2 claim | status | established by |
|---|---|---|
| §6 `refutation_handling` κ=+0.65, **trusted** | **superseded** — +0.44, p=0.185 on a deterministic judge | 3.6c |
| §6's table generally | every figure is one draw from a temperature-1.0 judge; spreads to 0.25 | 3.6c |
| §6 *"requires a rubric with anchored examples… not rescaling"* | **refuted** — the unanchored original beats both rewrites | 3.6c |
| §10 *"no inter-annotator agreement measure"* | **permanent**, and now quantified by test–retest | 3.6d |
| §10 *"two of five fail calibration"* | four of five on a deterministic judge; survivors differ by gold set | 3.6c/3.6d |
| §10 *"§8.2 was audited by a model, not a human"* | **strengthened** — a human audit of 60 fresh pairs reproduces 32.5% | 3.6a |
| §7.2 `code_url` 0-for-15 as a routing gap | three faults, not one; fixed and measured, **not applied** | 3.6b |

**The corpus re-extraction waits for 3.5.** See *Not in this iteration* in the plan. The
short version: extraction is sampled at the provider default and two identical-prompt runs
differ on 9 of 147 field outcomes, so any re-extraction moves the substrate whether or not a
prompt changed. There is no re-extracting to pick up one fix — it happens once, after 3.5,
with every arm re-run together.

---

## Reporting set: the active 10, with the 34 beside it

**Calibration is reported on `queries.jsonl` — the 10 thesis queries — because those are the
queries the thesis reports.** The 34 are shown alongside every figure rather than dropped,
for a reason that is specific to this corpus rather than general caution: the two sets do not
merely differ in power, **they disagree about which criteria pass, in opposite directions.**

Same 34 stored answers, one v1 rubric, judge at temperature 0:

| criterion | the active 10 | the other 24 | all 34 |
|---|---|---|---|
| `coverage` | +0.76 | +0.72 | +0.76 |
| `attribution` | **+0.79** | **+0.30** | +0.45 |
| `hedging_accuracy` | +0.26 | +0.16 | +0.25 |
| `refutation_handling` | +0.50 (n=3) | +0.43 | +0.44 |
| `synthesis` | **+0.38** | **+0.77** | +0.63 |

The 10 are an unusually easy subset for `attribution` and an unusually hard one for
`synthesis`. Only `coverage` is indifferent to the choice. So a single κ with no denominator
beside it would let either subset be reported as the truth, and `calibrate_judge.py` now
prints both by construction — `--gold` selects the calibration set, and the complement is
always shown.

**Stated plainly, because the order of events matters:** the divergence above was measured
*before* the active 10 were adopted as the reporting set, and adopting them raises
`attribution` from +0.45 to +0.79. That is a defensible choice — the judge should be
trustworthy on the queries actually reported — but it is a *choice*, not a finding, and it
must not be read as evidence that `attribution` works.

### What is certified on the active 10

| criterion | judge κ | grader vs self | verdict |
|---|---|---|---|
| `coverage` | **+0.76** | +0.66 | **certified** |
| `attribution` | +0.79 | **+0.19** | **not certified** — see below |
| `synthesis` | +0.38 | +0.60 | untrusted on this set |
| `hedging_accuracy` | +0.26 | +0.38 | untrusted |
| `refutation_handling` | +0.50 | +0.67 | **unmeasurable — n=3** |

**`attribution` is the trap, and the retest is what springs it.** Its +0.79 rests on labels
the grader reproduces at **+0.19** on those same queries (n=7), while the judge tracks the
first pass at +0.83. A judge agreeing with one sitting far better than the grader agrees with
themselves has not learned the criterion; it has fitted one sitting. This is exactly
`ceiling()`'s *"judge tracks one grader markedly better"* case, and it means the certification
would be an artefact. 3.6c's verdict therefore stands on the 10 as well as on the 34 —
reached by a different route, and, if anything, more sharply.

**`refutation_handling` is the real cost of the smaller set.** Only 3 of the 10 carry a known
contradiction, so n=3 — which is precisely what Iteration 1 recorded as *unmeasured*. On the
active set this criterion cannot be calibrated at all, and no amount of judge work changes
that; it needs more refutation queries in the gold set.

**Net: `coverage` alone is certified for cross-arm reporting** — the same bottom line as the
34, arrived at differently. On the 34, `synthesis` survives and `attribution` fails on judge
disagreement. On the 10, `synthesis` fails and `attribution` fails on label instability. The
agreement between two different routes is the strongest thing here.


---

## 3.6c `attribution` rubric — refuted, and a larger defect found underneath

> Figures in this section are on the 34 hand-graded answers, which is the set the item was
> run against. The verdict is unchanged on the active 10 — see *Reporting set* above, where
> `attribution`'s apparent +0.79 is shown to rest on labels the grader reproduces at +0.19.

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

> Figures below are over the 20 retested answers. Restricted to the 7 that are in the active
> 10: `coverage` +0.66, `synthesis` +0.60, `hedging_accuracy` +0.38, `attribution` **+0.19**,
> `refutation_handling` n=2. The direction is identical and `attribution` is worse, which is
> what blocks its certification on the active set.

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

### §10 is permanent, not pending

There is no second annotator and there will not be one: this is a single-person project.
That converts §10's threat from an open action into a **standing limitation of the work**,
and it should be written that way rather than deferred to a future iteration that cannot
happen. Every calibrated criterion in this project is calibrated relative to one reader, and
no amount of further engineering changes that.

What that costs, precisely: `coverage` and `synthesis` are certified as *"the judge tracks
this grader"*, not as *"the judge grades well"*. If the grader's standard on those two is
idiosyncratic, the judge has learned an idiosyncrasy and nothing here would reveal it.

What partly offsets it, and why the item was still worth running:

- **Test–retest bounds the label noise**, which is the part of §10 that *is* reachable
  alone. A criterion whose own author cannot reproduce it cannot be grader-independent
  either, so a low retest κ is decisive evidence against a criterion regardless of how many
  annotators exist. That is exactly what happened to `attribution` and `hedging_accuracy`,
  and neither verdict needs a second reader to stand.
- **The surviving two were confirmed the strong way** — stable across two human passes *and*
  tracked by a deterministic judge across both. That is more evidence than §6 had, even if
  it is not the evidence §10 asks for.
- **`--annotator second` stays in the tool.** It costs nothing to keep and makes the missing
  measurement explicit rather than invisible; if a reader is ever available, the sample and
  the scoring are already built and the sheet is already drawn.

The honest phrasing for the report is that inter-annotator agreement is **unmeasured and
will remain so**, with test–retest reported in its place and labelled as the weaker
substitute it is.

### Other limits
- **`refutation_handling` is undecidable at n=5.** Proportional sampling put 5 of the 9
  contradiction-bearing queries in the sheet. Its +0.57 self-agreement sits alongside
  ρ=+0.91, so the grader ranks those five consistently while the level moves — suggestive,
  not usable. Settling it needs the full 9, and really needs more refutation queries than
  the gold set contains.
- **Test–retest is an upper bound on stability**, not an estimate, because of recall.

---
## 3.6a Contradiction pass v2 — prompt rewritten, bar unmeasurable

**What the plan asked for.** Worked negative examples in the prompt, re-run, re-audit to
≥70% edge precision. §8.2 had characterised the failure as one pattern: twelve of twenty
spurious `refutes` are two papers describing their own different scopes, which v1's prose
warning did not prevent.

### The prompt

`_SYSTEM_V2` carries the twelve labelled spurious `refutes`, generalised into seven
categories: each paper stating its own scope; different modelling choices; claims that
actually agree; unrelated systems sharing vocabulary; a paper's own contribution vs another's
result; a stated need beside a stated limitation; and author-contribution boilerplate, which
is not a scientific assertion at all. `SYSTEM_PROMPTS` keeps v1 as a named control.

**A bug found on the way in.** §8.2 said *"the verdict cache means a re-run costs only what
changes"*. That was wrong, and dangerously so: the cache key was `a_name␟b_name` with no
record of which prompt produced the verdict, so a v2 run against the existing cache would
have returned 16,965 **v1** verdicts and reported them as v2's. The key is now
version-scoped.

### What was measured

Re-adjudicating all 3,072 pairs v1 accepted, under v2 ($0.60):

```
v2 retains 1,172 of 3,072  (38.2%) — it rejects 1,900

  v1 -> v2        refutes  undercuts    neither
  refutes              44         58         61
  undercuts            29      1,041      1,839
```

§8.2's audit put edge precision at 32.5%, i.e. ~998 of the 3,072 are real. v2 retained
**1,172**. That is close to the number of edges the audit says exist, which is what the fix
was supposed to do.

**It is consistent with the fix working and it does not demonstrate it.** v2 could have
rejected 1,900 pairs and retained 1,172 without the retained set being the *right* 1,172.
Retention count is not precision.

### Why the bar cannot be evaluated: the audit instrument fails on the positive class

Scoring v2 needs a labeller. §8.2's sixty labels came from a model — its own caveat says so
— but the labelling was done by hand, was never re-runnable, and was never itself checked.
`--label-with` makes it a scripted, blind pass, and `--validate-labeller` scores it against
those existing sixty labels before anything is believed. That check is new, and it fails:

```
model labeller (gpt-5.4-mini) vs the existing 60 labels     exact 76.7%

  existing -> model   refutes  undercuts  neither
  refutes                   1          1        3     ← finds 2 of 5
  undercuts                 1          0        9     ← finds 1 of 10
  neither                   0          0       45     ← perfect
```

The 76.7% is an artefact of the class balance: 45 of the 60 labels are `neither`, and the
labeller answers `neither` to almost everything. **On the fifteen pairs the audit calls a
real disagreement, it agrees on two.** A labeller like that reports low edge precision
whatever the truth, so it cannot test a ≥70% claim. No stronger model is reachable on this
project key (`gpt-5.4` and `gpt-4.1` both 403; only `-mini` and `-nano` resolve), and no
Anthropic key is configured, so a different-family labeller is not available either.

**This also widens the error bars on §8.2 itself.** Its 32.5% came from a model labeller
too. Two model labellers now disagree substantially on the positive class — the one thing
edge precision depends on — so 32.5% should be read as one labeller's figure, not as the
pass's precision.

### Result: v2 fails the bar, and is strictly worse than v1

Sixty fresh pairs from the v2 verdicts — 20/20/20, seed 7, zero overlap with the sixty
behind §8.2 — labelled by hand, blind. The zero overlap matters: those twelve spurious
`refutes` are inside the v2 prompt, so scoring on the pairs they came from would be testing
on training data.

| | equal-n | population-weighted | real edges |
|---|---|---|---|
| v1 (§8.2) | 32.5% | 25.8% | ~792 of 3,072 |
| **v2** | **32.5%** | **25.9%** | ~304 of 1,172 |

**Edge precision did not move.** Not by a little — the stratum rates are identical, 8/20 of
sampled `refutes` and 5/20 of sampled `undercuts` are real under both prompts. Against a
≥70% bar, v2 is refuted.

And the retention that looked promising was a loss. Of the 1,900 pairs v2 rejected from v1's
accepted set, **6 of 20 sampled are real disagreements** — v1's own discard rate was 2 of 20.

```
real edges in v1's 3,072   ~874
v2 keeps                   ~304      (35% of them)
v2 discards                ~570      at no gain in precision
```

So v2 rejects 62% of v1's edges roughly at random with respect to correctness, and slightly
worse than random: it preferentially drops real ones. The worked negatives §8.2 prescribed —
and which the plan called "a second route to §4.5" — do not work.

### §8.2's 32.5% is corroborated, and my doubt about it was wrong

Earlier in this item the model labeller (`gpt-5.4-mini`) found 2 of 15 real disagreements
that §8.2's labels called real, and I read that as widening the error bars on 32.5%. The
human pass settles it the other way: on 60 *different* pairs a human independently finds 19
disagreements (§8.2's labeller found 15 of 60) and reproduces both stratum rates exactly.
**§8.2's model labeller was adequate and its 32.5% stands** — the outlier was the
`gpt-5.4-mini` labeller, which is too conservative to audit anything. `--validate-labeller`
is what caught it, and remains the reason not to trust a model labeller unchecked.

### What this means beyond v2

§8.2's design note predicted this. Contradictory claims are similar *by construction*, so
similarity is purely a recall filter and **the model performs the entire discriminative
step**. v2 tests whether that step improves when the model is shown exactly what its errors
look like. It does not. Two prompts, one with a prose warning and one with seven worked
categories drawn from labelled failures, land on the same 32.5%.

The conclusion is about the approach, not the wording: pairwise claim adjudication at 0.90
cosine yields ~26–32% edge precision on this corpus regardless of prompt, and prompt
engineering is not the lever. Anything further needs a different mechanism — adjudicating
with both papers' surrounding context rather than two sentences, or a model that can do the
discriminative step at all.

### Status

Neither v1's nor v2's edges are applied (`approved: false`). §4.3 stands unimproved, and the
**second route to §4.5 named in the Iteration 3 plan is closed** — which leaves 3.1's
decomposition as the only remaining route, and raises what rides on it.

A reporting bug was fixed on the way out: `--score` read the accepted total from
`contradictions.json` regardless of which cache it audited, so the v2 audit reported "of
3,072 accepted" for a pass that accepted 1,172.

Cost: $0.63 — 3,072 adjudications plus 60 labeller-validation calls. The labels were free.

---
## 3.6b `ReproducibilityArtifact` routing — measured; two of three causes fixed

§7.2 attributed `code_url` 0-for-15 and `dataset_access` 0-for-14 to a prompt-routing gap.
It is three separate faults, and naming them apart is what made the fix measurable.

### Fault 1 — routing (the one §7.2 named)

Tracing all five gold papers that state a `code_url` to the section that states it:

| paper | section | `REPRO_ARTIFACT` routed there before? |
|---|---|---|
| `1db5090a` | conclusion | no |
| `60f69f1c` | abstract | no |
| `cca36fcf` | abstract + conclusion | no |
| `30a35856` | results | no |
| `91c10ab4` | **availability** | **yes** |

Four of five are the §2.4 `Hardware` pattern exactly. Corpus-wide, repo/archive URLs appear
in `other` (14 papers), `conclusion` (6), `abstract` (5), `method` (2), `results` (2) — and
`other`, 58% of all chunks, fell through to `_DEFAULT_TYPES`, which did not list the type.
`REPRO_ARTIFACT` is now routed to all of those.

### Fault 2 — the hint gate followed `Hardware`

`_REPRO_HINT` carries the `code_url` / `dataset_access` field list, and it was emitted only
when `Hardware` was in the section's type list. So `conclusion` and the `other` default would
have been asked for `ReproducibilityArtifact` **without being told which attributes to
fill**. The instruction was one branch away from the node type that needed it.

### Fault 3 — the schema could not express what 15 papers say

`DatasetAccess` offered `open|licensed|irb|unknown`. Two gold records authored `on request`,
which is none of those: there is no artifact to fetch and no licence to accept, only a person
to email. **Those two fields were unscoreable by any extraction**, so `dataset_access` had a
ceiling of 4 of 6 before the extractor was even involved. 15 papers in the corpus use that
language. `DatasetAccess.ON_REQUEST` now exists; `normalize()` already made the gold's
`on request` match it, so no gold was re-authored.

### Result

21 gold papers re-extracted into a scratch file — `04_extract.py --papers ... --out ...`, so
the corpus extraction is untouched while the answer is unknown. Run three times, because
**extraction temperature is not pinned either** and a single run cannot separate a fix from
resampling. Noise floor: **9 of 147 field outcomes differ between two identical-prompt runs.**

| field | before | after ×3 |
|---|---|---|
| `code_url` | **0** | 2, 3, 3 |
| `dataset_access` | **0** | 3, 3, 2 |
| `quantum_vendor` | 2 | 2, 2, 3 |
| `device_name` | 1 | 2, 2, 2 |
| `qubit_count` | 3 | 2, 1, 2 |
| `gpu_type` | 1 | 2, 1, 1 |
| `gpu_count` | 1 | 1, 1, 1 |
| **total correct** | **8** | **14, 13, 14** |
| accuracy | 73.2% | 77.5% |

Recall on stated facts roughly doubles, 8/37 → 14/37. Hallucinations are unchanged at 8, 8, 7.

**What is reliable, per field, across all three runs:**

| paper | field | section | outcome | which fault |
|---|---|---|---|---|
| `1db5090a` | `code_url` | conclusion | fixed 3/3 | routing |
| `30a35856` | `code_url` | results | fixed 3/3 | routing |
| `31824833` | `dataset_access` | other | fixed 3/3 | routing + enum |
| `10db0441` | `dataset_access` | availability | fixed 3/3 | **enum only** — was never a routing gap |
| `91c10ab4` | `code_url` | availability | 2/3 | neither; see below |
| `30a35856` | `dataset_access` | results | 2/3 | routing |
| `60f69f1c` | `code_url` | abstract | **0/3** | routing did not help |
| `cca36fcf` | `code_url`, `dataset_access` | abstract | **0/3** | routing did not help |
| `1db5090a` | `dataset_access` | conclusion | **0/3** | `open` not inferred |
| `91c10ab4` | `dataset_access` | availability | **0/3** | `open` not inferred |

### Two things the fix did not do, stated plainly

**Abstract routing does not work.** `60f69f1c` and `cca36fcf` state their URLs in the last
sentence of the abstract, in plain language (*"Our implementation can be found at: <url>"*,
*"Reproducibility: source code and computational results are available at <url>"*). Both are
now routed and both failed 3 times out of 3. `60f69f1c` shows the mechanism: it emitted
`ReproducibilityArtifact` nodes carrying **`{name, version}`** — a `Software` payload — for
Qiskit, PySCF and an IBM backend, and never touched the URL. `abstract` is not routed to
`SOFTWARE`, so the model used the nearest available slot and spent its budget there. A hint
forbidding that (*"a named library the paper merely used is `Software`, not an artifact"*) did
not measurably help at n=21; it is kept because a `ReproducibilityArtifact` carrying
`{name, version}` is wrong regardless, but it is **not demonstrated** and should not be
reported as a gain.

**`dataset_access: open` is not inferred from a public repo URL.** Three papers state a
GitHub URL and score `code_url` correct while `dataset_access` stays missed. The model does
not read *"code is available at <public url>"* as implying open access, and arguably should
not — code availability and data availability are different claims. This may be a gold
authoring question rather than an extraction one, and it is the largest remaining block.

### Applied to the corpus — and the substrate moved further than the fix

The deferral was reversed and the rebuild run. Extraction temperature was pinned to 0 first,
so this is the last rebuild needed rather than the first of two.

```
extraction    271 papers, 5,322 calls, $6.72
merge cache   4,185 pairs re-adjudicated, $0.26
stores        vector index rebuilt from unchanged chunks; graph rebuilt from new extractions
```

The merge-cache step is not optional and fails silently: `merge_verdicts.json` is keyed on
node names, new extraction produces new names, and stage 05 skips the semantic-merge tier for
any name it has no verdict for — no error, just degraded entity resolution.

**3.6b holds at corpus scale**, matching the 21-paper measurement:

| field | before | scratch runs | corpus |
|---|---|---|---|
| `code_url` | 0 | 2, 3, 3 | **2** |
| `dataset_access` | 0 | 3, 3, 2 | **2** |
| total correct | 8 | 14, 13, 14 | **11** |
| accuracy | 73.2% | 77.5% | **74.6%** |

`ReproducibilityArtifact` nodes went 34 → 78. `qubit_count` went 3 → 1, at the low end of the
2/1/2 the scratch runs gave, so it reads as a small real regression rather than noise: the
longer repro hint appears to cost some `Hardware` attention.

### The part that matters for 3.5

The rebuild changed far more than the repro layer. `Claim` nodes went **9,233 → 13,281
(+44%)**, nodes 23,689 → 27,777, edges 11,131 → 14,978 — the repro hint now fires on `other`
sections, 58% of all chunks, and sampling moved from the provider default to temperature 0.

And the edge type §4.5 was written about:

| edge | Iteration 2 | now |
|---|---|---|
| `undercuts` | 33 | **119** |
| `refutes` | 8 | **21** |

§4.5 diagnosed the relational weakness as `undercuts` being traversed **zero times across 34
queries**, because only 33 existed in a graph of 11,186. There are now 119 from per-paper
extraction alone — without the contradiction pass 3.6a refuted. **That may partially address
the weakness independently of the agentic loop**, which makes it a finding and a confound at
once: `typed_graph_chunks` 0.377 and `citation_graph` 0.367 describe a corpus that no longer
exists and cannot be inherited into 3.5.

### What the quadrupled `undercuts` layer actually bought: nothing measurable

Re-measured on the new substrate, same 10 dev queries, same code:

| arm | old substrate | new substrate |
|---|---|---|
| `typed_graph_chunks` | 0.383 | **0.383** |
| `citation_graph` | 0.367 | 0.400 |
| `vector_fulltext` | 0.417 | 0.367 (×3 runs, all identical) |

`typed_graph_chunks` is **unchanged to three decimal places** despite the graph gaining 44%
more `Claim` nodes and 4× more `undercuts` edges.

And the traversal does now reach them. Counting edge types traversed across the 10 queries:

```
addresses     103   43.6%
provides       79   33.5%
builds_on      21    8.9%
evaluated_on   18    7.6%
uses            9    3.8%
undercuts       6    2.5%      <- was ZERO across 34 queries on the old substrate
refutes         0
```

So `undercuts` went from unreachable to reachable, and the score did not move. **Edge
coverage was necessary but not sufficient** — §4.5 identified a real gap and closing it
turns out not to be what was binding. Only 1 of those 6 traversals is on a relational query,
which is the class the whole diagnosis was about.

**A second effect worth watching.** `provides` is now 33.5% of all traversal, up from
negligible — a direct consequence of 3.6b routing `ReproducibilityArtifact` and `PROVIDES`
into most sections. On queries that are not about code availability that is spent traversal
budget under `max_nodes=150`, and it is a plausible reason the added evidence did not convert
into recall. It is a cost of the repro fix that the repro score does not show.

**On `vector_fulltext` 0.417 → 0.367.** Its retrieval is provably unchanged by the rebuild —
the vector index is built from `chunks.jsonl`, which re-extraction does not touch. Three runs
on the new substrate all return exactly 0.367, and the *old* substrate produced 0.367 too
(`20260805T003845Z`) before the 0.417 that §4.1 cites. The reading is that 0.367 is the
arm's value on these 10 and §4.1's 0.417 was a single lucky draw — which is the same defect
as §6's judge table, in the retrieval numbers.

**Stale as a result**, flagged rather than deleted: `contradictions.json` and both
`contradiction_audit*.jsonl` reference node ids and claim texts that no longer exist. 3.6a's
conclusion is unaffected — it was about the prompt, not the substrate — but those specific
3,072 edges are gone. Iteration 2's substrate is recoverable from
`data/processed/iteration2-backup/` (gitignored, local).