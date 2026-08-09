"""The gold-answer record.

Deliberately NOT free-text prose. A structured skeleton is (a) writable in ~20-30 min,
(b) scorable both deterministically (facet/must-cite recall, refutation surfacing) and by
the judge, and (c) forces you to name the relational structure a correct answer needs.

Over-sample `relational` / `refutation` query types: that is where the typed graph earns
its complexity, and a natural mix averages the advantage away. Always report by type.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class QueryType(str, Enum):
    LOOKUP = "lookup"            # single-fact / single-source
    RELATIONAL = "relational"   # multi-hop across methods/problems/limitations
    REFUTATION = "refutation"   # contradicting evidence must be surfaced
    OPEN_DIRECTIONS = "open-directions"  # open-because-nobody vs open-because-everyone-failed


class KeyClaim(BaseModel):
    text: str = Field(description="Atomic, checkable claim a correct answer should make.")
    papers: list[str] = Field(default_factory=list, description="Grounding paper ids.")


class RefutationPair(BaseModel):
    a: str = Field(description="One side of a known contradiction (paper + claim).")
    b: str = Field(description="The opposing side.")


class GoldRecord(BaseModel):
    qid: str
    query: str
    query_type: QueryType
    facets: list[str] = Field(
        description="Sub-questions a complete answer must address; enables facet recall."
    )
    must_cite: list[str] = Field(
        default_factory=list, description="Paper ids a correct answer MUST ground on."
    )
    key_claims: list[KeyClaim] = Field(default_factory=list)
    known_refutations: list[RefutationPair] = Field(
        default_factory=list,
        description="Contradictions the answer SHOULD surface (refutation-handling score).",
    )
    # Filled in on a ~20-query calibration subset only (your own 1-5 ratings per criterion).
    grade: dict[str, int] | None = None
    notes: str | None = None


def resolve_gold(
    qids: set[str], gold_dir: Path, explicit: str | None = None
) -> tuple[Path, list[GoldRecord]]:
    """The gold file covering every qid in `qids`, and its records restricted to them.

    Auto-selected rather than defaulting to `queries.jsonl`, because the active gold set is
    the 10-query thesis subset while the hand-graded calibration run covers 34. Defaulting
    would silently work on 10 of 34 and report a figure over a third of the sample.

    Lives here rather than in a script because `rejudge.py` and `annotator_agreement.py`
    both need it, and `scripts/` is not an importable package.
    """
    from pathlib import Path

    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = gold_dir / explicit
        gold = load_gold(str(p))
        missing = qids - {g.qid for g in gold}
        if missing:
            raise ValueError(
                f"{p.name} is missing {len(missing)} of the requested qids: "
                f"{sorted(missing)[:5]}"
            )
        return p, gold

    covering = []
    for p in sorted(gold_dir.glob("queries*.jsonl")):
        try:
            gold = load_gold(str(p))
        except Exception:  # noqa: BLE001 - a malformed draft must not abort the search
            continue
        if qids <= {g.qid for g in gold}:
            covering.append((p, gold))
    if not covering:
        raise ValueError(f"no gold file under {gold_dir} covers all {len(qids)} qids")
    # Smallest covering set: a superset would pull in queries the caller has no answer for.
    p, gold = min(covering, key=lambda pg: len(pg[1]))
    return p, [g for g in gold if g.qid in qids]


def load_gold(path: str) -> list[GoldRecord]:
    import json
    from pathlib import Path

    lines = Path(path).read_text().splitlines()
    return [GoldRecord(**json.loads(line)) for line in lines if line.strip()]


def save_gold(records: list[GoldRecord], path: str) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(r.model_dump_json() for r in records) + "\n")