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


def must_cite_recall(answer: Answer, gold: GoldRecord) -> float | None:
    """Fraction of `gold.must_cite` papers the answer actually cited.

    `None` when the gold names no required papers: there is nothing to measure, and a
    default of 1.0 credits the system for a question that was never asked. See
    `deterministic_scores` for why that distinction is load-bearing.
    """
    required = set(gold.must_cite)
    if not required:
        return None
    cited = set(answer.cited_paper_ids)
    return len(required & cited) / len(required)


def citation_precision(answer: Answer, gold: GoldRecord) -> float | None:
    """Of the papers the answer cited, the fraction that are 'relevant' — where relevant is
    the union of must_cite and all key_claim source papers. A proxy: high recall with low
    precision means the system is citation-spraying.

    1.0 if the answer cited nothing — that is a property of the answer, and the
    `no_citations` well-formedness check reports it separately. `None` if the *gold* names
    no relevant papers, which is a property of the gold set and unmeasurable.
    """
    cited = set(answer.cited_paper_ids)
    if not cited:
        return 1.0
    relevant = set(gold.must_cite)
    for kc in gold.key_claims:
        relevant.update(kc.papers)
    if not relevant:
        return None
    return len(cited & relevant) / len(cited)


def key_claim_source_recall(answer: Answer, gold: GoldRecord) -> float | None:
    """Fraction of key claims whose source paper(s) appear in the answer's citations.
    A cheap proxy for 'did it ground the important claims' without semantic matching.
    `None` when the gold lists no key claims."""
    if not gold.key_claims:
        return None
    cited = set(answer.cited_paper_ids)
    hit = sum(1 for kc in gold.key_claims if not kc.papers or (set(kc.papers) & cited))
    return hit / len(gold.key_claims)


def refutation_surfaced(answer: Answer, gold: GoldRecord) -> float | None:
    """Fraction of known contradiction pairs where BOTH sides' papers are cited.

    Deterministic proxy for the judge's `refutation_handling` — if the answer never even
    cites both sides, it certainly didn't reconcile them. Judge scores the reconciliation
    quality on top of this. `None` when the query encodes no contradiction: 7 of 10 gold
    queries have none, and defaulting them to 1.0 put the reported aggregate at 0.700 while
    every query that actually had a contradiction scored 0.00."""
    if not gold.known_refutations:
        return None
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


def deterministic_scores(answer: Answer, gold: GoldRecord) -> dict[str, float | None]:
    """All deterministic metrics for one (answer, gold) pair.

    A metric is `None` when the gold record gives it nothing to measure — no required
    papers, no key claims, no known contradictions. Scoring those as 1.0 inflates the
    aggregate with questions the system was never asked: on the first calibrated run
    `refutation_surfaced` read 0.700 that way, while all three queries that did encode a
    contradiction scored 0.00. Consumers must skip `None` rather than coerce it.
    """
    return {
        "must_cite_recall": must_cite_recall(answer, gold),
        "citation_precision": citation_precision(answer, gold),
        "key_claim_source_recall": key_claim_source_recall(answer, gold),
        "refutation_surfaced": refutation_surfaced(answer, gold),
    }
