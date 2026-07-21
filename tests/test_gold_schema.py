"""Gold set round-trips and the sample queries are valid + relational-heavy."""

from __future__ import annotations

from pathlib import Path

from rpsg.eval.gold_schema import GoldRecord, QueryType, load_gold, save_gold

GOLD = Path(__file__).resolve().parents[1] / "eval" / "gold" / "queries.jsonl"


def test_sample_gold_loads_and_validates():
    records = load_gold(str(GOLD))
    assert len(records) >= 3
    assert all(isinstance(r, GoldRecord) for r in records)


def test_gold_oversamples_relational_and_refutation():
    records = load_gold(str(GOLD))
    hard = {QueryType.RELATIONAL, QueryType.REFUTATION, QueryType.OPEN_DIRECTIONS}
    hard_count = sum(1 for r in records if r.query_type in hard)
    # The whole point: the graph earns its complexity on the hard subset. Keep it heavy.
    assert hard_count >= len(records) / 2


def test_gold_round_trip(tmp_path):
    records = load_gold(str(GOLD))
    out = tmp_path / "rt.jsonl"
    save_gold(records, str(out))
    assert load_gold(str(out)) == records
