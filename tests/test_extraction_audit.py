"""Sampling and scoring for the precision audit.

The sampling rules exist so a 60-node audit cannot accidentally answer a different
question than the one asked: equal weight near the gate, and the reproducibility types
represented rather than swamped by `Claim`.
"""

from __future__ import annotations

from rpsg.eval.extraction_audit import (
    BANDS,
    band_of,
    confidence_is_informative,
    precision,
    sample_nodes,
    summarize,
)


def _n(i: int, conf: float, ntype: str = "Claim") -> dict:
    return {
        "id": f"{ntype.lower()}:{i}",
        "type": ntype,
        "name": f"node {i}",
        "confidence": conf,
        "evidence": ["a quote"],
        "attrs": {"from_paper": "p1"},
    }


# --- banding ------------------------------------------------------------------------

def test_a_node_below_the_gate_has_no_band():
    """Nodes under 0.65 were dropped at extraction; none should exist to sample."""
    assert band_of(0.5) is None


def test_bands_cover_the_kept_range_without_overlap():
    for conf in (0.65, 0.74, 0.75, 0.84, 0.85, 1.0):
        assert band_of(conf) is not None
    assert len({band_of(c) for c in (0.70, 0.80, 0.90)}) == 3


# --- sampling -----------------------------------------------------------------------

def test_each_band_gets_equal_weight():
    """Proportional sampling would put ~2 nodes near the gate, where the decision lives."""
    nodes = [_n(i, 0.70) for i in range(100)]
    nodes += [_n(1000 + i, 0.80) for i in range(100)]
    nodes += [_n(2000 + i, 0.90) for i in range(100)]
    s = sample_nodes(nodes, per_band=10)
    counts = {b: sum(1 for x in s if x.band == b) for b in {x.band for x in s}}
    assert set(counts.values()) == {10}
    assert len(counts) == len(BANDS)


def test_repro_types_are_reserved_a_share():
    """`Hardware` is the least confident type and the newest; `Claim` is 40% of the graph
    and would otherwise fill every slot."""
    nodes = [_n(i, 0.70, "Claim") for i in range(100)]
    nodes += [_n(500 + i, 0.70, "Hardware") for i in range(20)]
    s = sample_nodes(nodes, per_band=9, repro_share=1 / 3)
    assert sum(1 for x in s if x.node_type == "Hardware") == 3


def test_sampling_is_deterministic_under_a_seed():
    nodes = [_n(i, 0.70) for i in range(100)]
    assert [x.node_id for x in sample_nodes(nodes, per_band=5, seed=7)] == [
        x.node_id for x in sample_nodes(nodes, per_band=5, seed=7)
    ]


def test_a_thin_band_yields_what_exists_rather_than_failing():
    s = sample_nodes([_n(1, 0.70), _n(2, 0.90)], per_band=20)
    assert len(s) == 2


def test_a_sample_carries_the_evidence_needed_to_judge_it():
    (s,) = sample_nodes([_n(1, 0.70)], per_band=1)
    assert s.evidence == ["a quote"] and s.paper_id == "p1"


# --- scoring ------------------------------------------------------------------------

def test_unjudged_rows_are_excluded_not_counted():
    rows = [
        {"band": "0.65-0.75", "node_type": "Claim", "correct": True},
        {"band": "0.65-0.75", "node_type": "Claim", "correct": None},
    ]
    r = precision(rows)
    assert r["n"] == 1 and r["precision"] == 1.0 and r["skipped"] == 1


def test_precision_is_none_when_nothing_is_judged():
    assert precision([{"band": "0.65-0.75", "node_type": "Claim", "correct": None}])["n"] == 0


def test_precision_breaks_out_by_band_and_type():
    rows = [
        {"band": "0.65-0.75", "node_type": "Claim", "correct": False},
        {"band": "0.85-1.01", "node_type": "Hardware", "correct": True},
    ]
    r = precision(rows)
    assert r["by_band"]["0.65-0.75"]["precision"] == 0.0
    assert r["by_type"]["Hardware"]["precision"] == 1.0


def test_flat_precision_means_confidence_carries_no_signal():
    rows = [{"band": b, "node_type": "Claim", "correct": True} for b in ("0.65-0.75", "0.85-1.01")]
    rows += [{"band": b, "node_type": "Claim", "correct": False} for b in ("0.65-0.75", "0.85-1.01")]
    assert confidence_is_informative(precision(rows)) is False


def test_rising_precision_means_the_gate_is_doing_work():
    rows = [{"band": "0.65-0.75", "node_type": "Claim", "correct": False} for _ in range(3)]
    rows += [{"band": "0.85-1.01", "node_type": "Claim", "correct": True} for _ in range(3)]
    assert confidence_is_informative(precision(rows)) is True


def test_a_single_band_cannot_settle_the_question():
    rows = [{"band": "0.65-0.75", "node_type": "Claim", "correct": True}]
    assert confidence_is_informative(precision(rows)) is None


def test_summarize_says_so_when_nothing_is_judged():
    assert "nothing judged" in summarize(precision([]))