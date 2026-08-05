# RPSG — Iteration 1 Report

**Status: closed.** Commit `afac292`, 2026-08-04.
**Exit criterion:** a vector-over-full-text baseline scored end to end by a calibrated LLM
judge. **Met**, with two of five judge criteria clearing the agreement threshold.

*Replaces the 2026-07-29 revision, which described a one-paper corpus. That version is in
git history if the earlier state is needed.*

---

## 1. Headline

| | |
|---|---|
| Baseline | `vector_fulltext`, run `eval/runs/20260804T221802Z_vector_fulltext` |
| Headline number | judge **`coverage` 2.4 / 5** (human 2.1 / 5), QWK **+0.68** |
| Deterministic companion | `must_cite_recall` **0.217** |
| Trusted criteria | `coverage`, `hedging_accuracy` |
| Untrusted criteria | `attribution`, `synthesis`, `refutation_handling` |

The baseline is weak, for a diagnosed reason (§5). That is the finding, not an apology for
it — a weak, explained baseline is what leaves the typed graph something to earn in
Iteration 2.

---

## 2. Corpus and pipeline

| stage | count |
|---|---|
| `papers.jsonl` (S2 metadata) | 353 |
| PDFs downloaded | 270 |
| section files → sections | 270 → 5,767 |
| chunks | 10,993 (10,725 full-text, 268 abstract) |
| papers extracted | 270 |
| vector index | 10,993 chunks, SPECTER (faiss) |
| typed graph (Kuzu) | 23,460 nodes, 10,660 edges |

**Extracted node types** (pre-store, before id de-duplication):

`Claim` 9,578 · `Method` 6,311 · `Limitation` 4,071 · `Problem` 3,540 · `Dataset` 456 ·
`Software` 385 · `ReproducibilityArtifact` 40 · `Hardware` **12**

**Graph edge types:** `addresses` 2,922 · `authored_by` 2,424 · `cites` 2,333 ·
`builds_on` 1,828 · `evaluated_on` 422 · `uses` 376 · `published_in` 246 · `provides` 63 ·
`undercuts` 24 · `requires` 18 · `refutes` **4**

**Cost**, measured from `rpsg.llm.usage` tables printed at end of run:

| run | model | calls | in / out tokens | cost |
|---|---|---|---|---|
| extraction, 18 papers | `gpt-5.4-nano` | 460 | 683,742 / 324,526 | $0.54 |
| extraction, 3 papers | `gpt-5.4-nano` | 47 | 44,112 / 24,813 | $0.04 |
| eval, 10 queries | `gpt-5.4-mini` | 20 | 207,883 / 6,864 | $0.19 |

≈ **2.8¢/paper** extraction, **$0.19** per 10-query eval. `USAGE` does not persist, so the
whole-corpus extraction total is not recoverable; the per-paper rate is measured over the
21 papers extracted during this session, not all 270.

---

## 3. Gold query set

10 queries — 4 relational, 3 refutation, 2 lookup, 1 open-directions (9/10 "hard" types,
which the schema test enforces). Every `must_cite` id resolves to an indexed paper with
chunks: **10/10 retrievable**.

Written from the reference list of the author's MSc thesis, so `key_claims` are grounded in
papers already read closely rather than paraphrased from abstracts. Two claims were
verified directly against source PDFs during authoring; one of those checks failed and was
corrected (§4.4).

---

## 4. Challenges faced and how they were resolved

Each entry is symptom → root cause → fix → measured effect. Figures state the corpus size
they were measured at, because the corpus grew throughout (24 → 240 → 317 → 353 papers)
and several numbers only mean something against a size.

Two patterns recur often enough to name:

- **Idempotency masked partial failure.** Stages skip work when the output exists, so a
  crash mid-run left a partial artifact that every later run treated as complete. Three
  separate bugs hid behind this.
- **Measurement reversed the plan twice** (§4.3.3, §4.6.2). In both cases the reasoning was
  sound and the data said the opposite. Recorded as findings, not as tidy successes.

### 4.1 Getting text out of PDFs

**GROBID cannot run on Apple Silicon.** The container starts, accepts connections, returns
HTTP 200 — and zero parsed sections. GROBID ships amd64-only; under emulation it cannot
spawn its `pdfalto` subprocess, so it degrades to empty TEI rather than failing loudly.
Fixed with a typography-based fallback in `rpsg.ingestion.pdf_parser`, detecting headings
from span font size and the bold flag. **270 of 270 downloaded PDFs parsed without GROBID.**

Two bugs inside that fix, both of which produced *plausible-looking* output: filtering
whitespace-only spans before joining text yielded `'Barrenplateausin…'` (this PDF emits
inter-word spaces as their own spans); and requiring `size >= body_size` for a heading
rejected every 9pt heading in a 10pt document. Fixed by using all spans for text but only
inked spans for size decisions, and by accepting a fully-bold short line as a heading
regardless of size.

**A zero-byte PDF killed stage 02 at paper 29 of 181** — and re-running reported success,
because 29 section files existed and the stage skips papers that already have output. Fixed
with a per-PDF `try/except` plus `_is_usable_pdf` validation at download time.

**GROBID returned an HTML error page and it was cached as valid.** The response was HTTP
206 carrying HTML; `raise_for_status()` passes on 206, the XML parser found no TEI, and the
empty result was cached. Fixed with a content-type check, and a zero-section parse now
raises so the caller falls back rather than caching emptiness.

### 4.2 Making the typed graph non-empty

**Zero `Limitation` nodes across every early run.** One of the node types the thesis
depends on had a count of exactly zero. Extraction is section-routed — `section_type`
selects which node types the model is asked for — and `Limitation` was requested only for
`limitations` and `discussion` sections. Most papers have neither heading and state caveats
in the conclusion, which fell through to the defaults. The model was never asked. Fixed by
adding a `conclusion` routing entry and widening `_DEFAULT_TYPES`.

**Effect: 0 → 3,664 `Limitation` nodes across 240 of 249 papers**; papers with no route to
a `Limitation` fell from 35% to 6%. The highest-value single fix in Iteration 1, and
invisible from the code — extractor, schema and prompts were each correct in isolation.
Only the *composition* of routing table and real heading distribution was wrong.

**69% of sections classified `other`**, from keyword matching against papers with headings
like "The role of measurements". Fixed with positional inference when the keyword pass
fails, fragment-section merging, and two new types: `availability` and `acknowledgments`
(dropped before chunking). 69% → ~54%.

**`Hardware` extraction is effectively broken — still open.** See §6; structurally the same
bug as `Limitation`, which is the lesson: a routing table that silently suppresses a node
type produces a clean-looking graph with a whole tier missing, and no error anywhere.

### 4.3 Making the corpus citation-connected

**Relevance search cannot build a traversable citation graph.** At 240 papers, in-corpus
citation density was 3.66% and mean out-degree 1.29; 44% of papers had no outgoing
in-corpus citation at all. 96% of references leave the corpus — papers cite foundational
maths, physics and ML work no topically-sampled corpus contains. Relevance search returns
papers *about* a topic, not papers that *cite each other*. Scaling does not fix it: 24 →
240 papers (10×) moved density 1.77% → 3.66% (2×).

Fixed by co-citation expansion (`--expand-citations`): fetch the papers the corpus already
cites, which arrive pre-connected because the citations exist in reference lists already,
dangling.

| 240 papers | +77 co-cited |
|---|---|
| in-corpus references 3.66% | **16.68%** |
| out-degree 1.29 | **7.64** |
| 2-hop reach ~2.9 papers | **~66 papers** |
| `cites` edges 309 | **2,333** |

**77 papers bought 2,024 edges — 26 each; the preceding 216 relevance-search papers bought
292 — 1.4 each.** An ~18× difference, because hubs are selected *because* the corpus
already points at them. Ordering matters and is easy to state backwards: hubs are only
identifiable because relevance search first built a corpus with 8,443 references to mine.

**What this is not:** evidence that typed edges beat untyped ones. At 240 papers that
comparison was confounded — 3,813 typed edges against 309 citation edges means a typed win
could be explained by edge count alone. After expansion the arms are within one order of
magnitude, which is what makes the Iteration-2 ablation an experiment rather than a
foregone conclusion.

### 4.4 Batch-job robustness

**One bad attribute value destroyed a 2.5-minute extraction run.** The model returned a
list where `attrs` expected a scalar, and `_normalize` sat *outside* the per-section
`try/except`, so one malformed value killed every result. Fixed by JSON-encoding non-scalar
values and moving `_normalize` inside the guard.

**`references: null` would have crashed every Semantic Scholar fetch.** Pydantic's
`default_factory` fires only when a key is *absent*; S2 returns `"references": null`
explicitly for some papers, so the default never applied and a `None` reached list code.
Found while adding a test, not in production — it would have failed on the first paper with
no references, at any corpus size, and the code looked correct.

### 4.5 Performance and platform

**Extraction took 29 minutes at 0.2% CPU.** The CPU figure is the whole diagnosis: the
stage was waiting on HTTP, not computing, and sections within a paper are independent.
Fixed with a `ThreadPoolExecutor` over sections. **28m55s → 5m07s on 20 papers (5.7×).**

**Segfault in `search()`, and three wrong hypotheses.** `SIGSEGV` in `ask.py`, reproducibly,
only at `search()` — loading the index, encoding, and building a fresh `IndexFlatIP` were
all fine, which is why stage 05 never hit it. Cause: three separate `libomp.dylib` copies
(faiss, torch, sklearn); a process that has loaded torch and then runs a threaded faiss
search dies. Three fixes that did **not** work, recorded because each looked right:
reordering imports (no effect); `faiss.omp_set_num_threads(1)` at import time (*caused*
`OMP Error #15`, running before both runtimes exist); `KMP_DUPLICATE_LIB_OK=TRUE` (no
effect). Fixed by pinning faiss to one thread immediately *before* searching.

### 4.6 Retrieval quality

**The model mangled 40-character citation ids.** Answers cited ids that did not exist — one
with an internal fragment duplicated, another truncated. This mattered more than it looks:
the runner harvests citations by regex, so a mangled id counts as a *distinct cited paper*
and silently corrupts both `must_cite_recall` and `citation_precision`. A measurement bug,
not a cosmetic one. Fixed by showing papers to the synthesizer as `[P1]`, `[P2]` handles
and rewriting them afterwards; a handle that was never issued is dropped, so a hallucinated
citation cannot enter the cited set.

**Short chunks crowded out substance.** 26% of retrieved hits were under 300 characters
against 4.0% of the corpus — ~6× over-representation; median retrieved length 404 chars in
a corpus whose median chunk is 1,881. Short text embeds nearer the corpus centroid, so it
scores moderately against *every* query and wins top-k whenever no longer chunk is a strong
match.

*The first plan was wrong.* Merging trailing chunks into their predecessor was the
principled-sounding fix; measurement killed it, because **73% of short chunks are the sole
chunk of a genuinely short section**. Fixed instead with length-aware score damping —
multiply similarity by `min(1, chars / 800)` before ranking, using the 5× over-fetch that
already existed so no re-embedding was needed.

*A regression the sweep caught:* naive damping dropped `availability` sections from 5 hits
to **0** across six probe queries — including the one asking literally *"where is the code
for these experiments available"*. Length cannot separate a complete 269-char availability
statement from a truncated 162-char stub; section type can. Fixed with
`_DAMPING_EXEMPT_SECTIONS`.

| | before | after |
|---|---|---|
| median retrieved length | 404 ch | **1,208 ch** |
| context per query | 5,289 ch | **10,481 ch** |
| availability hits (6 queries) | 5 | **10** |

**Required papers rank below `top_k` — open.** See §5.3.

### 4.7 Measurement integrity

**An answer citing nothing scored `must_cite_recall = 1.00`.** When the model cited
nothing, `VectorRAGSystem.answer` fell back to reporting *every retrieved paper* as cited,
on the reasoning that `citation_precision` needed a denominator. It does not — it returns
1.0 for an uncited answer by design. So a total attribution failure scored **perfect
citation recall** whenever retrieval had found the required papers, and the well-formedness
check could not see it either, because the ids were backfilled upstream before the check
ran.

| an answer citing nothing | before | after |
|---|---|---|
| `must_cite_recall` | **1.00** | **0.00** |
| `no_citations` violation | silent | fires |

Found by writing Level-1 well-formedness assertions — the "too dumb to be worth writing"
checks — which took about twenty minutes. This is the strongest argument in the document
for cheap assertions: the bug was in the headline metric of the exit criterion, and no
amount of reading the metric code would have revealed it, because the metric was correct.
Its input was not.

**Claims that measurement corrected**, kept deliberately since a retrospective that records
only other people's bugs is not one:

| claimed | measurement said |
|---|---|
| `Hardware` is sparse because papers do not report it | a large majority state a qubit count; routing never asks (§6) |
| merging trailing chunks is the principled fix | 73% of short chunks are sole-chunk sections |
| ~35 references per paper | 46 — the hub papers brought their own bibliographies |
| gold files were empty (`wc -l` = 0) | non-empty; the files lacked a trailing newline |
| `t = ⌊(d−1)/2⌋` is in Hamming 1950 | it is not; Hamming gives a lookup table (§4.8) |

### 4.8 Closing the gold set and the judge

**The gold set scored structural zeros.** `queries.jsonl` held three queries whose
`must_cite` entries were `paper:PLACEHOLDER_A` strings, so `must_cite_recall` and
`citation_precision` were 0.000 *by construction* — comparing against ids that could never
be retrieved. Resolved by writing 10 queries with real S2 ids. Worth naming because it
recurs invisibly: a gold set can look valid, load cleanly and pass its schema test while
measuring nothing.

**The thesis references were not in the corpus.** The queries were drafted from a thesis
bibliography, but checking all 50 references against `papers.jsonl` found **zero overlap** —
the corpus was quantum VQE work, the bibliography was community detection and coding
theory. The single apparent hit was a title collision. Resolved by adding `--ids-file` to
`01_fetch_corpus.py` to fetch a named reading list rather than run a search; 37 references
resolved, 35 by id and 2 by title. Books and web pages were excluded up front — no
open-access PDF means metadata that produces zero chunks.

**Two gold-critical papers were paywalled.** 17 of 37 failed PDF download, two of them
`must_cite`: Hamming 1950 (403 from an open-access repository, which then served 755 bytes
of JavaScript shell to a proper user-agent) and Kernighan-Lin 1970 (no `openAccessPdf` at
all). Scripted access to the IEEE copies returned `HTTP 418`. Resolved by downloading both
through institutional access and storing them under their `paperId` filenames, validated
with `_is_usable_pdf` before ingesting.

**A re-grounded claim turned out to be wrong.** Three `rel-t03` claims cited a textbook with
no S2 record and were re-grounded on Hamming 1950. Reading the PDF afterwards showed the
closed form `t = ⌊(d−1)/2⌋` **is not in Hamming's paper** — he gives a plain
distance-to-capability lookup table. Reworded to what he states. The check only happened
because the PDF became available; otherwise the misattribution would have entered scoring.

**A metric could be gamed by an unsourced claim.** `open-t02` carried a `key_claim` with an
empty `papers` list, and `key_claim_source_recall` counts `not kc.papers` as a hit — so it
scored correct regardless of any answer. Record deleted. Same class of defect as the
placeholder ids: a number that looks like a measurement but is a default.

**Stale stores nearly scored a stale index.** `extractions.jsonl` was rewritten while
`vectors.faiss` and `rpsg.kuzu` were hours older. Resolved by gating the eval behind a
freshness check — both stores must be newer than `extractions.jsonl` or the run aborts
rather than producing a plausible wrong number.

**Human grading had to be redone per-criterion.** The first pass assigned one holistic
score per query, but `calibrate_criterion` runs per-criterion, and back-deriving five
scores from one would have fabricated the human side of the comparison. Re-graded all five.
`refutation_handling` was left unscored on the 7 queries with no `known_refutations`, so it
is calibrated on the 3 where it applies rather than on 10 where 7 are meaningless.

---

## 5. Results

### 5.1 Deterministic metrics

| metric | mean |
|---|---|
| `must_cite_recall` | 0.217 |
| `citation_precision` | 0.150 |
| `key_claim_source_recall` | 0.167 |
| `refutation_surfaced` | 0.700 — **misleading, see below** |

`refutation_surfaced` returns `1.0` when a query has no `known_refutations`; 7 of 10
qualify and all 7 scored 1.0. **All three queries that encode a contradiction scored
0.00.** The honest reading is zero of three, and the metric's default should return `None`
rather than credit.

**By query type** (n is small; directional only):

| type | n | must_cite | precision | key_claim | refutation |
|---|---|---|---|---|---|
| relational | 4 | 0.17 | 0.12 | 0.17 | 1.00 |
| refutation | 3 | 0.50 | 0.33 | 0.33 | 0.00 |
| lookup | 2 | 0.00 | 0.00 | 0.00 | 1.00 |
| open-directions | 1 | 0.00 | 0.00 | 0.00 | 1.00 |

### 5.2 Judge calibration

| criterion | n | human | judge | QWK | Spearman ρ | p | trusted |
|---|---|---|---|---|---|---|---|
| `coverage` | 10 | 2.10 | 2.40 | **+0.68** | +0.87 | 0.001 | yes |
| `hedging_accuracy` | 10 | 4.00 | 3.60 | **+0.67** | +0.75 | 0.013 | yes |
| `synthesis` | 10 | 3.20 | 2.50 | +0.53 | +0.88 | 0.001 | no |
| `attribution` | 10 | 1.80 | 2.60 | **+0.02** | +0.04 | 0.922 | no |
| `refutation_handling` | 3 | 3.33 | 3.00 | +0.57 | +1.00 | — | no |

**Length bias:** slope +0.054 per 100 chars, p=0.733 — **not** significant. The judge is
not rewarding verbosity.

The three failures are not the same failure:

- **`attribution` is an instrument defect, not a judge defect.** ρ=+0.04 at p=0.92 is no
  relationship whatsoever. The judge scores attribution with the retrieved context in its
  prompt; `traces.jsonl` records only `evidence_chars`, so the human grader could not see
  the same evidence. The two were answering different questions. Re-running calibration
  cannot fix this — the asymmetry is structural.
- **`synthesis` is a scale offset.** ρ=+0.88 means the judge orders answers the way the
  human does; it simply sits ~0.7 lower. Note the judge is *harsher* here and *more
  generous* on attribution, so it is not uniformly lenient.
- **`refutation_handling` is unmeasured.** n=3 cannot support a kappa. The ρ=+1.00 is what
  perfect rank correlation on three points always looks like. Report as unmeasured, not as
  a near miss.

### 5.3 Why the baseline is weak: retrieval, diagnosed

Of the 9 distinct papers the gold set requires, **5 never appear in any query's top-20** —
including the D-Wave community-detection paper required by 4 of 10 queries.

Two further papers were retrieved but never cited, which is a separate synthesis-side loss.

The cause is **not** indexing and **not** OCR. Targeted probes pull each missing paper to
rank 1:

```
"detecting multiple communities using quantum annealing D-Wave modularity karate club"
    -> Negre, Negre, Community detection in graphs, Negre
"error detecting and error correcting codes parity check Hamming distance"
    -> Hamming x4
```

It is **corpus dilution**. The index holds 270 papers of which ~88% are quantum VQE /
error-correction work no gold query asks about. Under natural query phrasing that mass
outranks the 36 community-detection papers the queries target: `rel-t02` (needs Negre)
returned 16 distinct papers topped by "A review on QAOA"; `rel-t03` (needs Hamming) matched
"qudit quantum machine learning" on the word *encoding*.

**Ranking vs coverage.** Probing to depth 400 splits the 18 required (query, paper) pairs:

| | count |
|---|---|
| retrieved at `top_k=20` | 7 |
| present but ranked below the cutoff | 7 — at ranks 40, 44, 46, 56, 95, 105, 124 |
| absent from the top 400 entirely | 4 |

So 7 of 18 are a *ranking* problem, not a coverage problem: the paper is in the index and
findable, just below the cutoff. Raising `top_k` is therefore the cheap experiment, and
worth running before anything expensive. **The 4 unreachable pairs are the real signal** —
genuinely beyond this embedder for that query — and they are the argument for Iteration-2
typed-graph retrieval rather than a tuning fix.

One instructive case: `look-t02` asks the time complexity of Kernighan-Lin. The answer is
substantively correct and cites a review reporting the figure, while gold requires the
original paper — which ranks at chunk #5 and *was* retrieved. That is a gold-design
question (is citing a secondary source a failure?) rather than a system failure, and the
kind of ambiguity only reading the answers surfaces.

---

## 6. Defect found but not fixed: `Hardware` routing

`NodeType.HARDWARE` is requested in exactly one of eleven section types — `appendix`, which
is **2.9%** of chunks (316 / 10,993). `_REPRO_HINT`, the block instructing the model to
capture `qubit_count` and vendor, also fires only on `appendix`.

Result: **12 `Hardware` nodes across 8 of 270 papers**, in a corpus where the overwhelming
majority of papers discuss qubit counts.

This is a routing bug, not a property of the literature. It is left for Iteration 2
deliberately: the fix changes the prompt for `results` / `method` / `availability`, so any
extraction-precision audit run beforehand would be invalidated by it.

---

## 7. Carried into Iteration 2

Ordered by what blocks what.

1. **Store retrieved evidence in `traces.jsonl`.** Blocking, narrowly: it gates any future
   `attribution` claim, and no amount of re-grading works around it. Small change.
2. **Resolve the corpus / gold mismatch.** Blocking the Iteration 2 *claim*, not the work.
   If the typed graph beats this baseline while 5 required papers are undiscoverable by the
   vector retriever, the win is confounded with a dilution problem a corpus filter would
   also have fixed. Either scope the index or widen the gold set — before running the
   comparison.
3. **Fix `Hardware` routing, then re-extract, then audit precision.** In that order; the
   fix invalidates any earlier audit.
4. **Grow calibration n.** Not blocking — `coverage` is usable today. Becomes blocking when
   claiming a *difference* between two systems, which n=10 will not separate unless the
   effect is large. Cheapest path: score `vector_abstract` against the same gold set, which
   doubles graded pairs without writing new gold.
5. **Make `refutation_surfaced` return `None`** for queries with no `known_refutations`,
   so the aggregate stops being inflated by inapplicable queries.
5. **Raise `top_k` and re-measure.** The cheapest experiment available: 7 of 18 required
   pairs sit at ranks 40–124, so a larger cutoff recovers them without touching the
   architecture. Run it *before* claiming the typed graph fixed anything, or the graph gets
   credit for a one-line config change.
6. **Make `refutation_surfaced` return `None`** for queries with no `known_refutations`,
   so the aggregate stops being inflated by inapplicable queries.
7. **`extraction_gold` / `repro_gold`.** Still unpopulated. Prefer a precision audit over a
   recall metric for extraction — recall has no well-defined denominator, and name-matching
   gold to extracted nodes would report entity-resolution failure as extraction failure.
   `repro_gold` waits on the `Hardware` routing fix; gold for a component that is almost
   entirely missing measures the bug, not the component.
8. **Entity resolution.** Node ids are slugified surface names, so `MLP` and
   `Multilayer Perceptron (MLP)` are separate nodes. Investigated: exact-name matching
   merges only 0.6% of entity nodes, and unambiguous acronym expansion reaches 1.6% at high
   precision — while *unfiltered* acronym matching over-merges badly, putting `Adam` and
   `Gradient Descent` in one node because `VQE` alone appears with 36 different
   parenthetical expansions. Semantic paraphrase is untouched.
9. **`refutes` / `undercuts` remain near-empty** (5 and 25 edges). Structural, not a tuning
   gap: extraction runs per-section within one paper, and a paper rarely refutes itself.
   Cross-paper contradiction needs a second pass over extracted claims, which needs entity
   resolution first — so it is gated on item 8.
10. **`Chunk.id` uniqueness.** Per-section character offsets can collide across two
    same-typed sections in one paper. Known, documented, unfixed.

---

## 8. What Iteration 1 does not claim

- No comparison between systems. One baseline was scored; `vector_abstract` was not run.
- Nothing about extraction *quality* — no extraction gold set exists, so §2's counts are
  volumes, not accuracies.
- Nothing about `attribution`, `synthesis`, or `refutation_handling` performance; those
  judge scores exist but did not clear calibration.
- The per-type table in §5.1 is not a finding. n=1–4 per cell.