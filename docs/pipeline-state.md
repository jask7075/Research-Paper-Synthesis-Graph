# RPSG — pipeline state report

Generated 2026-07-31 22:12 UTC by `scripts/report_state.py`. Every figure is recomputed from the artifacts on disk.

## Corpus

> Papers without a downloadable PDF contribute Tier-A metadata and citation edges but no semantic nodes, so the corpus has two classes of paper.

| | |
|---|---|
| metadata records | 317 |
| PDFs downloaded | 249 |
| papers parsed to sections | 249 |
| with abstract | 297 (94%) |
| Tier-A only (no full text) | 68 — metadata + citations, no Method/Problem/Claim |

## Sections

> `section_type` selects which node and edge types extraction asks for (`rpsg.extraction.prompts`). Only `conclusion`, `discussion`, and `limitations` route `Limitation`, so papers with none of them cannot produce one.

| | |
|---|---|
| total sections | 5,229 |
| per paper | median 18, min 1, max 140 |
| type: other | 2,844 (54.4%) |
| type: method | 499 (9.5%) |
| type: introduction | 435 (8.3%) |
| type: results | 409 (7.8%) |
| type: references | 274 (5.2%)  (dropped before chunking + extraction) |
| type: conclusion | 182 (3.5%) |
| type: appendix | 176 (3.4%) |
| type: acknowledgments | 155 (3.0%)  (dropped before chunking + extraction) |
| type: discussion | 116 (2.2%) |
| type: abstract | 52 (1.0%) |
| type: availability | 43 (0.8%) |
| type: limitations | 26 (0.5%) |
| type: related_work | 18 (0.3%) |
| papers with no Limitation route | 16/249 (6%) |
| extraction calls this corpus costs | 4,800 |

## Chunks

> Similarity is damped for chunks under 800 chars at search time (`availability` exempt), because short text embeds near the corpus centroid and over-scores against every query.

| | |
|---|---|
| total chunks | 9,913 |
| by corpus | fulltext=9,664, abstract=249 |
| length: median | 1,881 chars |
| length: min / max | 80 / 7,757 chars |
| under 300 chars | 396 (4.0%) |

## Extraction (Tier B/C)

> Tier A never appears here — Paper/Author/Venue and `cites` come from Semantic Scholar in stage 05, not from the LLM. `refutes`/`undercuts` are near-empty by construction: extraction runs per-section within one paper, so it cannot see cross-paper contradictions.

| | |
|---|---|
| papers extracted | 249 |
| nodes / edges | 22,054 / 5,738 |
| nodes per paper | 89 |
| node: Dataset | 411 in 155/249 papers |
| node: Method | 5,694 in 235/249 papers |
| node: Problem | 3,209 in 244/249 papers |
| node: Claim | 8,679 in 245/249 papers |
| node: Limitation | 3,664 in 240/249 papers |
| node: Hardware | 8 in 6/249 papers |
| node: Software | 349 in 122/249 papers |
| node: ReproducibilityArtifact | 40 in 33/249 papers |
| edge: evaluated_on | 452 |
| edge: addresses | 2,949 |
| edge: builds_on | 1,850 |
| edge: refutes | 4 |
| edge: undercuts | 24 |
| edge: requires | 18 |
| edge: uses | 378 |
| edge: provides | 63 |
| tier B_semantic | 21,657 nodes, 0 edges |
| tier C_relational | 0 nodes, 5,279 edges |
| tier reproducibility | 397 nodes, 459 edges |
| papers yielding 0 nodes | 2 |
| below confidence gate | 0 nodes, 0 edges (expected 0; gates 0.65/0.5) |

## Citation graph (Tier A)

> Relevance search returns papers *about* a topic, not papers that cite each other. Co-citation expansion (`--expand-citations`) is what makes this traversable; see the finding section of the README.

| | |
|---|---|
| papers | 317 |
| reference entries | 14,523 (~46/paper) |
| pointing inside the corpus | 2,422 (16.68%) |
| in-corpus out-degree | 7.64 citations/paper |
| papers with >=1 outgoing cite | 237/317 (75%) |
| 2-hop reach (approx d + d^2) | ~66 papers |
| traversable for an ablation? | yes |

## Stores

| | |
|---|---|
| vector index | 30.5 MB |
| vector metadata | 19.9 MB |
| graph (Kuzu) | 64.8 MB |
| graph contents | 23,460 nodes, 10,660 edges |
|   edge: addresses | 2,922 |
|   edge: authored_by | 2,424 |
|   edge: cites | 2,333 |
|   edge: builds_on | 1,828 |
|   edge: evaluated_on | 422 |
|   edge: uses | 376 |
|   edge: published_in | 246 |
|   edge: provides | 63 |
|   edge: undercuts | 24 |
|   edge: requires | 18 |
|   edge: refutes | 4 |

## Evaluation state

> The Iteration-1 exit criterion is a scored `vector_fulltext` run against a real gold set. Placeholder ids make `must_cite_recall` and `citation_precision` meaningless, and no driver runs `rpsg.eval.calibration`, so judge scores are uncalibrated.

| | |
|---|---|
| gold/queries.jsonl | 3 record(s)  — CONTAINS PLACEHOLDERS, metrics not meaningful |
| gold/extraction_gold.jsonl | 1 record(s)  — CONTAINS PLACEHOLDERS, metrics not meaningful |
| gold/repro_gold.jsonl | 1 record(s)  — CONTAINS PLACEHOLDERS, metrics not meaningful |
| scored runs | 1 |

## Configuration

| | |
|---|---|
| extraction model | gpt-5.4-nano |
| judge / synthesis model | gpt-5.4-mini / gpt-5.4-mini |
| embeddings | sentence-transformers/allenai-specter (dim 768) |
| confidence gates | nodes >=0.65, edges >=0.5 |
| extraction workers | 8 |
| length damping | 800 chars |
| chunk target | 512 tokens, 64 overlap |
| pricing configured | yes |
