"""The gold-answer record.

Deliberately NOT free-text prose. A structured skeleton is (a) writable in ~20-30 min,
(b) scorable both deterministically (facet/must-cite recall, refutation surfacing) and by
the judge, and (c) forces you to name the relational structure a correct answer needs.

Over-sample `relational` / `refutation` query types: that is where the typed graph earns
its complexity, and a natural mix averages the advantage away. Always report by type.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


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