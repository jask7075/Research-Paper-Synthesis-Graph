"""Deterministic metrics — no LLM, no network, fully unit-testable.

These are the numbers you can defend without qualification, because they don't depend on a
judge's calibration. Semantic facet *coverage* is intentionally left to the judge; here we
score only what can be checked exactly: which required papers the answer cited, whether
those citations are precise, and whether known contradictions were surfaced.

An `Answer` (below) is what every system-under-test must emit: the answer text plus the set
of paper ids it cited/grounded on (the runner extracts these from the system's own output).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rpsg.eval.gold_schema import GoldRecord


class Answer(BaseModel):
    qid: str
    text: str
    cited_paper_ids: list[str] = Field(default_factory=list)


def must_cite_recall(answer: Answer, gold: GoldRecord) -> float:
    """Fraction of `gold.must_cite` papers the answer actually cited. 1.0 if none required."""
    required = set(gold.must_cite)
    if not required:
        return 1.0
    cited = set(answer.cited_paper_ids)
    return len(required & cited) / len(required)


def citation_precision(answer: Answer, gold: GoldRecord) -> float:
    """Of the papers the answer cited, the fraction that are 'relevant' — where relevant is
    the union of must_cite and all key_claim source papers. A proxy: high recall with low
    precision means the system is citation-spraying. 1.0 if it cited nothing."""
    cited = set(answer.cited_paper_ids)
    if not cited:
        return 1.0
    relevant = set(gold.must_cite)
    for kc in gold.key_claims:
        relevant.update(kc.papers)
    if not relevant:
        return 1.0
    return len(cited & relevant) / len(cited)


def key_claim_source_recall(answer: Answer, gold: GoldRecord) -> float:
    """Fraction of key claims whose source paper(s) appear in the answer's citations.
    A cheap proxy for 'did it ground the important claims' without semantic matching."""
    if not gold.key_claims:
        return 1.0
    cited = set(answer.cited_paper_ids)
    hit = sum(1 for kc in gold.key_claims if not kc.papers or (set(kc.papers) & cited))
    return hit / len(gold.key_claims)


def refutation_surfaced(answer: Answer, gold: GoldRecord) -> float:
    """Fraction of known contradiction pairs where BOTH sides' papers are cited.

    Deterministic proxy for the judge's `refutation_handling` — if the answer never even
    cites both sides, it certainly didn't reconcile them. Judge scores the reconciliation
    quality on top of this. 1.0 if there are no known refutations for this query."""
    if not gold.known_refutations:
        return 1.0
    cited = set(answer.cited_paper_ids)
    surfaced = 0
    for pair in gold.known_refutations:
        a_papers = _papers_in(pair.a)
        b_papers = _papers_in(pair.b)
        if (a_papers & cited) and (b_papers & cited):
            surfaced += 1
    return surfaced / len(gold.known_refutations)


def _papers_in(text: str) -> set[str]:
    """Pull paper-id-like tokens out of a refutation side string (e.g. 'S2:abc claims X')."""
    tokens = set()
    for tok in text.replace(",", " ").split():
        tok = tok.strip(".;()[]")
        if ":" in tok or tok.startswith("paper:"):
            tokens.add(tok)
    return tokens


def deterministic_scores(answer: Answer, gold: GoldRecord) -> dict[str, float]:
    """All deterministic metrics for one (answer, gold) pair."""
    return {
        "must_cite_recall": must_cite_recall(answer, gold),
        "citation_precision": citation_precision(answer, gold),
        "key_claim_source_recall": key_claim_source_recall(answer, gold),
        "refutation_surfaced": refutation_surfaced(answer, gold),
    }
