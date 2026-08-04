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

## 4. Challenges and how they were resolved

### 4.1 The gold set scored structural zeros

`queries.jsonl` held three queries whose `must_cite` entries were `paper:PLACEHOLDER_A`
style strings. `must_cite_recall` and `citation_precision` were therefore 0.000 *by
construction* — the metric was comparing against ids that could never be retrieved. The
same run also predated any real corpus.

**Resolved** by writing 10 queries with real S2 paper ids. The failure mode is worth
naming because it recurs invisibly: a gold set can look valid, load cleanly, and pass its
schema test while measuring nothing at all.

### 4.2 The thesis references were not in the corpus

The queries were drafted from the thesis bibliography, but a check of all 50 references
against `papers.jsonl` found **zero overlap** — the corpus was quantum VQE / barren-plateau
work, the bibliography was community detection and coding theory. The single apparent hit
was a title collision.

**Resolved** by adding `--ids-file` to `01_fetch_corpus.py`, which fetches a named reading
list (S2 ids, DOIs, or bare titles) rather than running a search. 37 references resolved,
35 by id and 2 by title lookup. Books, Wikipedia entries and software repos were excluded
up front — they have no open-access PDF and would enter the corpus as metadata that
produces zero chunks.

### 4.3 Two gold-critical papers were paywalled

Of the 37 fetched, 17 failed PDF download. Two of those were `must_cite` in the gold set:
Hamming 1950 (403 from the NPS repository) and Kernighan-Lin 1970 (no `openAccessPdf`
listed at all). Retried Hamming against its open-access repository with a proper
user-agent; it returned 200 with 755 bytes of DSpace JavaScript shell, not a PDF. Scripted
access to the IEEE copies returned `HTTP 418` — their bot filter.

**Resolved** by the author downloading both through institutional access and storing them
under their `paperId` filenames, which is how the parser ties a PDF to its metadata.
Validated with the pipeline's own `_is_usable_pdf` before ingesting.

### 4.4 A re-grounded claim turned out to be wrong

Three `rel-t03` claims cited Pless, a textbook with no S2 record. They were re-grounded on
Hamming 1950. Reading the actual PDF afterwards showed the closed form
`t = ⌊(d−1)/2⌋` **is not in Hamming's paper** — he gives Table V, a plain
distance-to-capability lookup. The formula is Pless's later formalization.

**Resolved** by rewording the claim to what Hamming states. Noted here because the check
only happened because the PDF became available; the misattribution would otherwise have
entered scoring undetected. Hamming also never writes "(7,4,3)" or "Hamming(7,4)" — that
notation is a modern back-formation.

### 4.5 A metric could be gamed by an unsourced claim

`open-t02` carried a `key_claim` with an empty `papers` list. `key_claim_source_recall`
counts `not kc.papers` as a hit, so the claim scored correct regardless of any answer.

**Resolved** by deleting the record. Same class of defect as §4.1: a number that looks like
a measurement but is a default.

### 4.6 Stale stores nearly scored a stale index

`extractions.jsonl` was rewritten while `vectors.faiss` and `rpsg.kuzu` were hours older,
so an eval run would have scored against an index missing the new papers.

**Resolved** by gating the eval behind a freshness check — both store files must be newer
than `extractions.jsonl` or the run aborts rather than producing a plausible wrong number.

### 4.7 Human grading had to be redone per-criterion

The first pass of human grades assigned one holistic score per query.
`calibrate_criterion` runs per-criterion, so a single number cannot calibrate a
five-criterion judge, and back-deriving five scores from one would have fabricated the
human side of the comparison.

**Resolved** by re-grading all five criteria per query. `refutation_handling` was left
unscored on the 7 queries with no `known_refutations`, so the criterion is calibrated on
the 3 where it applies rather than on 10 where 7 are meaningless.

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
6. **`extraction_gold` / `repro_gold`.** Still unpopulated. Prefer a precision audit over a
   recall metric for extraction — recall has no well-defined denominator, and name-matching
   gold to extracted nodes would report entity-resolution failure as extraction failure.
7. **Entity resolution.** Node ids remain slugified surface names.

---

## 8. What Iteration 1 does not claim

- No comparison between systems. One baseline was scored; `vector_abstract` was not run.
- Nothing about extraction *quality* — no extraction gold set exists, so §2's counts are
  volumes, not accuracies.
- Nothing about `attribution`, `synthesis`, or `refutation_handling` performance; those
  judge scores exist but did not clear calibration.
- The per-type table in §5.1 is not a finding. n=1–4 per cell.