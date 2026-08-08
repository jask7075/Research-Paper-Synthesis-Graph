"""Sampling and scoring for the contradiction audit.

The rules under test are the ones that decide whether the audit measures anything: equal
strata, an order that does not leak the verdict, and an edge-precision figure that counts
the right thing.
"""

from __future__ import annotations

from rpsg.eval.contradiction_audit import (
    VERDICTS,
    edge_precision,
    precision,
    sample_pairs,
    summarize,
)


def row(i: int, verdict: str) -> dict:
    return {
        "pair_id": f"p{i}", "a_text": "A", "b_text": "B",
        "a_evidence": ["qa"], "b_evidence": ["qb"],
        "a_paper": "pa", "b_paper": "pb", "similarity": 0.95,
        "model_verdict": verdict,
    }


def label(model: str, human: str | None) -> dict:
    return {"model_verdict": model, "human": human}


# --- sampling -------------------------------------------------------------------------

def test_each_verdict_gets_equal_weight():
    """`undercuts` outnumbers `refutes` 18:1, so a proportional sample of 60 would hold
    three `refutes` -- the class the eyeball check flagged."""
    pool = [row(i, "undercuts") for i in range(200)]
    pool += [row(1000 + i, "refutes") for i in range(30)]
    pool += [row(2000 + i, "neither") for i in range(500)]
    s = sample_pairs(pool, per_verdict=10)
    counts = {v: sum(1 for x in s if x.model_verdict == v) for v in VERDICTS}
    assert counts == {"refutes": 10, "undercuts": 10, "neither": 10}


def test_the_sample_is_shuffled_across_verdicts():
    """Grouped output leaks the verdict from position: a labeller who notices the sheet
    change character at item 21 has recovered what the sheet withholds."""
    pool = [row(i, v) for v in VERDICTS for i in range(20)]
    order = [x.model_verdict for x in sample_pairs(pool, per_verdict=10)]
    assert order != sorted(order, key=VERDICTS.index)


def test_sampling_is_deterministic_under_a_seed():
    pool = [row(i, v) for v in VERDICTS for i in range(20)]
    assert [x.pair_id for x in sample_pairs(pool, per_verdict=5, seed=3)] == [
        x.pair_id for x in sample_pairs(pool, per_verdict=5, seed=3)
    ]


def test_a_thin_stratum_yields_what_exists():
    pool = [row(1, "refutes"), *[row(10 + i, "neither") for i in range(30)]]
    s = sample_pairs(pool, per_verdict=20)
    assert sum(1 for x in s if x.model_verdict == "refutes") == 1


def test_a_sample_carries_both_claims_and_their_evidence():
    (s,) = sample_pairs([row(1, "refutes")], per_verdict=1)
    assert s.a_text == "A" and s.b_evidence == ["qb"] and s.a_paper == "pa"


# --- scoring --------------------------------------------------------------------------

def test_unjudged_rows_are_excluded_not_counted():
    r = precision([label("refutes", "refutes"), label("refutes", None)])
    assert r["n"] == 1 and r["agreement"] == 1.0 and r["skipped"] == 1


def test_agreement_is_broken_out_per_model_verdict():
    """Disagreement on an accepted verdict is a false edge entering the graph;
    disagreement on `neither` is a real contradiction discarded. Opposite problems."""
    rows = [label("refutes", "neither"), label("neither", "neither")]
    r = precision(rows)
    assert r["by_verdict"]["refutes"]["agreement"] == 0.0
    assert r["by_verdict"]["neither"]["agreement"] == 1.0


def test_edge_precision_treats_the_two_accepted_labels_as_interchangeable():
    """A `refutes` the human calls `undercuts` is still a real disagreement and still a
    useful edge -- the type is wrong, the edge is not spurious."""
    assert edge_precision(precision([label("refutes", "undercuts")])) == 1.0


def test_edge_precision_counts_a_rejected_pair_as_spurious():
    rows = [label("refutes", "refutes"), label("undercuts", "neither")]
    assert edge_precision(precision(rows)) == 0.5


def test_edge_precision_ignores_the_neither_stratum():
    """`neither` rows were never going to become edges, so including them would dilute a
    figure that is meant to apply to the accepted total."""
    rows = [label("refutes", "refutes"), label("neither", "neither")]
    assert edge_precision(precision(rows)) == 1.0


def test_edge_precision_is_none_when_nothing_was_accepted():
    assert edge_precision(precision([label("neither", "neither")])) is None


def test_summary_projects_edge_precision_onto_the_accepted_total():
    out = summarize(precision([label("refutes", "refutes"), label("refutes", "neither")]),
                    accepted_total=3072)
    assert "50.0%" in out and "1,536" in out


def test_summary_reports_discarded_disagreements():
    out = summarize(precision([label("neither", "refutes"), label("neither", "neither")]))
    assert "1/2 sampled `neither` pairs are real disagreements" in out


def test_summary_says_so_when_nothing_is_judged():
    assert "nothing judged" in summarize(precision([]))