# Iteration 2 — plain-language report

Written without technical shorthand. Each phase says what it was trying to do, what
went wrong, and what came out of it. Numbers are as measured; nothing is rounded in our
favour.

The technical version with full tables is in
[iteration-2-notes.md](iteration-2-notes.md).

---

## The question the whole iteration exists to answer

Can we answer research questions better by building a **map of what papers say and how
those things connect**, instead of just searching for passages that look similar to the
question?

Ordinary search works well when the answer sits in one passage. It should work badly when
it doesn't. Ask *"which methods tackle this problem, and what's wrong with each"* and no
single passage holds both halves — you need one naming the methods and several describing
their weaknesses.

A map should handle that. Start at the problem, step to the methods, step again to their
limitations, and assemble an answer no single passage contains.

Everything below is instrumentation for that one comparison.

---

## Words used throughout

- **Item** — something pulled out of a paper: a method, a problem, a finding, a
  limitation, a piece of hardware. Each carries the sentence from the paper supporting it.
- **Link** — a connection between two items that a paper actually stated.
- **Passage** — a few paragraphs of a paper kept together as one piece of text.
- **Hit rate** — of the papers we decided in advance a good answer must use, what fraction
  did the system find and cite. Runs 0 to 1.
- **The four arms** — four different ways of answering the same questions, so they can be
  compared: ordinary passage search, search over abstracts only, map-walking, and
  map-walking that fetches passages.

---

## Phase 1 — Make the comparison fair before building anything

**What it does.** Fixes the measuring equipment before measuring anything with it.

The reasoning: if our new method wins against a badly set-up baseline, the win means
nothing. So before touching the map, we asked what could make any result meaningless.

Four things could, and all four were real.

**The search was told to look at too little.** It returned the top 20 passages. Seven of
the 18 papers we knew the answers needed were sitting at positions 40 to 124 — found by
the search, ranked sensibly, and thrown away by an arbitrary cutoff. Raised to 60.

> This one mattered most. If a one-line setting change recovered most of those papers, the
> honest headline is *"our baseline was misconfigured"* — and the map would have been
> credited for it.

**We couldn't check the grading.** The system recorded only *how many characters* of
evidence it had used, not the evidence itself. So when a human graded whether an answer's
claims were properly sourced, they were judging from memory while the automatic grader
judged from the actual text. Two different questions. That is exactly why the automatic
grader's agreement with the human on that criterion was zero in Iteration 1 — not low,
zero.

**One score was inflated by questions it didn't apply to.** We had a measure of "did the
answer surface the disagreement between these two papers". When a question involved no
disagreement, it scored a perfect 1.0 — for having nothing to do. Seven of ten questions
were like that, so the average read 0.700 while *every question that actually had a
disagreement scored 0.00.*

> **Fix, applied everywhere afterwards:** a measure now returns "not applicable" rather
> than a flattering default. The rule is: not-applicable when the *question* has nothing
> to measure, a low score when the *answer* is deficient. Those are different things and
> collapsing them hides failure.

**Some required papers were unreachable.** Four cases. We checked why: all of them rank in
the top seven when the search is restricted to their own topic. So it's dilution — they're
findable, just outranked by 270 other papers.

> **Challenge:** the tempting fix was to restrict the search to the relevant topic. We
> refused. Restricting the search to the topic being tested is fitting the test to the
> answer. Recorded as diagnosed and left alone.

---

## Phase 2 — Getting better information out of the papers

**What it does.** The map is only as good as what we extract from each paper, so this
phase improves the extraction.

**Hardware was almost entirely missing.** 163 of 249 papers state what machine they ran
on; we had extracted 6. The cause was mundane — "hardware" simply wasn't in the list of
things we asked for when reading the methods, results and availability sections. Added it:
6 → 242 items, covering 98 papers.

**Names were descriptions, not names.** The extractor was producing things like
*"application of AQCtensor to the family of Hamiltonians defined in equation (1)"* as the
*name* of an item. That is a sentence about what a paper did, not a name you could ever
look up. Rewrote the instruction: a name is an index entry, not a description.

Re-read all 270 papers with the fixes. Cost $5.80.

| | before | after |
|---|---|---|
| typical name length | 5 words | 3 words |
| names longer than six words | 33% | 11% |

> **Challenge — the map was secretly two maps stacked on top of each other.** Found by
> accident when the database ran out of memory. We had been *adding* to the map on each
> rebuild rather than replacing it, so items from an old extraction were still sitting
> there next to their replacements. There was no way to tell which run any item came from.
>
> **Fix:** delete the map completely before every rebuild. It can always be rebuilt from
> the papers, so starting empty is the honest default.
>
> This was a correctness problem before it was a size problem, and every measurement taken
> before it was found was suspect.

---

## Phase 3 — Merging duplicates

**What it does.** The same method gets written five different ways across five papers
— "QAOA", "Quantum Approximate Optimization Algorithm", "the QAOA algorithm". If those
stay separate, the map is fragmented and walking it reaches less than it should.

**First attempt: rules.** Tidy up the text, expand abbreviations where the meaning is
unambiguous. The ambiguity check matters — "VQE" has 36 different expansions in this
corpus, and merging carelessly put *Adam* together with *Gradient Descent*.

> **Challenge — I shipped a version where most merges were wrong.** The text-tidying
> stripped out everything in brackets, which merged *"AlphaQubit 2 (RT) complexity"* with
> *"(full)"*. I had looked at a sample of two, found one good merge, and shipped it.
>
> **Fix:** keep bracket contents, only flatten the punctuation. Merges dropped from 369 to
> 50 — meaning **86% of what I shipped had been wrong.**
>
> The failure wasn't the code. It was checking two examples.

**Second attempt: meaning-based matching.**

> **Challenge — comparing meaning cannot make this decision.** We measured how similar
> every method name is to its nearest neighbour. Nothing scores below 0.70, so there is no
> "clearly different" group to set a cutoff against. Worse, *"Gradient-free optimization"*
> and *"Gradient-based optimization"* — opposites — score 0.993, while a genuine spelling
> variant of the same thing scores 0.976. A cutoff would merge the opposites and miss the
> real pair.
>
> **Fix:** use similarity only to *propose* candidate pairs, and have a language model
> make the actual yes/no decision on each one. It rejects about 72% of what similarity
> proposes.

> **Challenge — a crash we had already documented and I caused again.** The maths library
> and the machine-learning library each ship their own copy of a parallel-processing
> component, and running them together crashes the process. This was written up in
> Iteration 1. I wrote the same bug in a new file. Documentation didn't prevent it.

> **Challenge — an overnight run lost all its work.** It saved results only at the very
> end, and timed out before reaching it. **Fix:** save every 200 decisions.

> **Challenge — after all that, the merges were never used.** The map-building step was
> still using only the rule-based merges. All 909 model-approved merges had no effect on
> anything we measured, and the entire four-way comparison had been run on an
> unmerged map. Nothing checked that the expensive result reached the thing consuming it.

Final: 3,859 pairs considered, 973 items merged.

---

## Phase 4 — Answering by walking the map

**What it does.** This is the actual idea under test. Find a starting point in the map from
the question, walk outwards along the links, write an answer from what you reach.

**How far to walk.** We tested one, two and three steps.

| steps | hit rate |
|---|---|
| 1 | 0.389 |
| 2 | 0.556 |
| 3 | 0.556 |

All the gain is in the second step; the third finds nothing new. Fixed at two.

**What makes a good starting point — I predicted this wrong.** The map holds short labels
("QAOA") and whole sentences stating what a paper found. I assumed the sentences were
noise that would crowd out the labels. Removing them dropped the hit rate from 0.556 to
0.333.

Obvious backwards: **a question is a sentence.** Matching a sentence against another
sentence works far better than against a three-word label, because there is more to match
on. The sentences are the doorway into the right part of the map.

> **Challenge — a database error that named the wrong line.** We asked for each item once,
> sorted by confidence. The database refused and the error pointed at the sorting. The real
> cause was the de-duplication: asking for each item *once* makes the database stop
> tracking the link it arrived by, so the thing being sorted on no longer exists by the
> time it sorts. **Fix:** return the confidence as part of the result and sort on that copy.

> ### Challenge — the experiment failed, and the score was misleading
>
> First real run: hit rate **0.183**, against **0.367** for ordinary search. Half as good.
> On the face of it, the idea had failed.
>
> Before accepting that, we split the score. Citing a paper needs two things to go right:
> **reaching** it during the walk, and **using** it in the answer.
>
> | | map-walking | ordinary search |
> |---|---|---|
> | reached the right papers | 0.556 | 0.611 |
> | turned that into a citation | 33% | 60% |
>
> That changes the diagnosis completely. **The walking was working** — it reached nearly as
> many of the right papers as ordinary search. The failure was entirely in the second step.
>
> **Why.** The map stores one supporting sentence per item. Enough to prove the item is
> real, not enough to *write* from — a lone sentence with its argument stripped away gives
> nothing to quote and no context to explain it. The answer-writing step skated over them.
>
> **Fix.** Use the map as a *chooser*, not a source of words. Walk it to decide which
> papers matter — which it does well — then fetch proper multi-paragraph passages from
> those papers and write from those.
>
> | | before | after |
> |---|---|---|
> | hit rate | 0.183 | 0.483 |
> | reaching → citing | 33% | 87% |
>
> The idea was never the problem. Handing over one-sentence fragments was.
>
> **A detail that shaped the fix.** The natural approach is to take each stored sentence
> and pull up the passage it came from. It only works for 57% of items — the stored
> sentence has had its spacing tidied and some straddle a boundary between passages. So we
> match at the level of the **paper**, which is recorded exactly and never fails. Slightly
> less precise, completely reliable.

---

## Phase 5 — Checking we extract hardware and code details correctly

**What it does.** A separate strand: can the system reliably pull out the practical details
that make research reproducible — what machine, how many qubits, where the code is?

> **Challenge — the answer key couldn't express "the paper doesn't say".** A blank meant
> both *"this paper is silent about it"* and *"we haven't checked yet"*. So a system that
> invented a qubit count was indistinguishable from one that correctly reported nothing.
>
> **Fix:** three states — a value, "not reported", and "not yet checked". Six possible
> outcomes, and crucially it now distinguishes **missing something** from **making
> something up**. Those are opposite failures needing opposite fixes; one number hides
> which you have.

Read the papers and filled in 106 fields, taking the answer key from 25 usable entries to
131.

> **Challenge — one "paper" wasn't a paper.** While reading, we found one entry whose
> stored file was a library *search results page* — four fragments of website furniture and
> a citation-export box. Left deliberately blank: marking it "not reported" would claim the
> paper is silent when in fact we never obtained the paper, and scoring it would blame the
> extractor for a download failure. You later found the real paper, which we added properly
> under its own record rather than pasting its text into the broken entry — that would have
> left the title and references describing one paper and the text another.

**Result.** 73.2% correct — but 67.4% of that is **correctly saying nothing** about papers
that genuinely report nothing. Strip that out and the system finds **17%** of the facts
papers actually state. It invents something 6% of the time. Code links and data
availability are found **zero times out of fifteen**, including five papers with a plain
GitHub address in the text.

---

## Phase 6 — Making sure the scores mean something

**What it does.** Two problems: we had only 10 test questions, and we were partly relying
on a language model to grade answers without knowing if it grades like a human.

**Ten questions is too few.** A swing of 0.1 is one question changing its answer. We
expanded to 34, drawn from parts of the collection the original ten never touched.

> **Finding:** the ranking of the four methods *changed* between 10 questions and 34. That
> is the small-sample problem demonstrating itself rather than being argued about.

> **Finding the small set had hidden:** the measure of "did the answer surface the
> disagreement" fell from 0.667 to 0.111. With nine disagreement questions instead of
> three, only **one in nine** gets surfaced — by any of the methods. At three questions
> this had looked like a 67% success rate.

**Is the automatic grader trustworthy?** You graded all 34 answers by hand and we compared.

| what it grades | agreement | verdict |
|---|---|---|
| did it cover the question | 0.72 | trustworthy |
| did it connect sources | 0.68 | trustworthy |
| did it surface disagreement | 0.65 | trustworthy |
| is confidence appropriate | 0.55 | not trustworthy |
| are claims properly sourced | 0.34 | not trustworthy |

> **Finding — the sourcing grade fails in a specific and fixable way.** It is not too
> generous; both averages sit near 3 out of 5. It **refuses to use the ends of the scale.**
> You used 1 through 5. It never gives 5, almost never 1, and packs everything into 2–4.
> So it rates your worst answers 4 and your best answers 2. That needs worked examples of
> what a 1 and a 5 look like, not a numerical adjustment.

---

## Phase 7 — Testing whether the expensive parts were worth it

**What it does.** Two experiments that each remove something and re-measure, to find out
what that thing was contributing.

**Were the duplicate merges worth it?** Rebuilt the map with the model-approved merges
switched off, changed nothing else, re-ran. Two of the three scores came back
**identical**; the third moved by 0.017 — noise at this sample size.

> 944 merges, real money, considerable engineering. Contribution: nothing measurable.
> This also reassigned blame for an earlier drop we had wrongly attributed to the merges —
> it belonged to the re-extraction.

**Is the extraction any good?** You labelled 60 items, twenty from each confidence band,
judging each against its own supporting sentence.

```
precision 95%
low-confidence band     85%
middle band            100%
high-confidence band   100%
```

> This is the single most important number in the iteration, and I had predicted 50–60%.
> Wrong by 35 points.
>
> It matters because it **closes off the obvious objection.** The map-based method does not
> underperform because the map is full of rubbish. The map is 95% correct. Whatever is
> wrong is wrong with the approach, not the contents.

**Would any map have done as well?** We built a fourth method that walks a completely
different set of links: plain "paper A cites paper B" connections, which come free with
the catalogue data and need no extraction at all.

| | hit rate |
|---|---|
| map-walking with fetched passages | 0.383 |
| **plain citation links** | **0.367** |
| ordinary passage search | 0.367 |

> Free citation links match our extracted map to within 0.016 on every measure. The
> expensive extracted links buy no advantage over connections that arrive with the data.
>
> And a variant that starts from ordinary search and *then* follows citations scores
> **worse** than ordinary search alone (0.317 vs 0.367). Following citations dilutes a
> good starting set rather than enriching it.

> **Challenge caught before it corrupted the result.** The two map methods have a limit on
> how much they collect. In the extracted map that limit counts *items*, and 150 items
> works out to about 10 papers. In the citation map a node *is* a paper, so copying "150"
> would have meant 150 papers — three times the material, and it would have won on sheer
> volume rather than on structure. Set to 30, matching the other method's actual reach.

---

## Phase 8 — The last two fixes

**Two different passages could share an identity.** Passage labels were built from the
paper, the section type, and the character positions within that section. But positions
restart at zero in every section, and papers routinely have several sections labelled
"other" — so two different passages of the same paper could get identical labels. 43 cases.
Small, but a shared label means a store can serve the wrong text. Fixed by including the
section's position.

**The map barely records disagreements.** Because we read one paper at a time, the only
disagreements we can see are the ones a paper states about itself. That gave 8 direct
contradictions and 25 partial ones across 271 papers — and the measured consequence is the
one-in-nine result from Phase 6. **A map cannot route you to a disagreement it doesn't
contain.**

So we compare claims *between* papers: propose pairs by similarity, have a model rule on
each as contradicting, weakening, or neither.

> **The design point that took measuring to see.** When merging duplicates, high similarity
> means "probably the same thing" — a precision filter. Here it is the reverse.
> Contradictory claims are *highly* similar by nature, because they discuss the same
> subject and differ only in what they assert. So similarity says nothing about whether a
> contradiction exists; it only narrows down what to look at. The model does all the actual
> judging.

> **Safety rule:** anything that goes wrong — a failed request, an unreadable answer —
> results in "no disagreement". A wrongly-claimed contradiction is worse than a missed one,
> because it would send a question to a disagreement no paper actually made.

*(Running at time of writing: 16,972 pairs, roughly $2.)*

---

## What the iteration concluded

**Ordinary passage search wins.** Across 34 questions: 0.456 for ordinary search, 0.377 for
map-walking with fetched passages, 0.324 for abstract-only search, 0.299 for map-walking
alone.

**It loses worst exactly where it should win.** On the multi-part questions the map was
built for, map-walking scores 0.202 against ordinary search's 0.357 — last place on its own
home ground.

**The extracted links are worth nothing over free ones.** Citation links match them within
0.016.

**And this is not an excuse-able failure.** The extraction is 95% accurate, so it isn't
that the map is full of errors. Both obvious objections — *"your extraction was bad"* and
*"no map would have helped"* — are closed off by measurement.

**One genuinely interesting pattern.** Split the questions by topic: ordinary search scores
0.493 on questions outside the thesis's own subject area and only 0.367 inside it, while
map-walking is flat at 0.375 and 0.383. Ordinary search's whole advantage comes from
questions outside the dense part of the collection. Inside it, the map is competitive and
actually leads on *how precisely* it cites.

That is a claim about **when** a map earns its cost, and it is a stronger and more
defensible finding than "our method won" would have been.