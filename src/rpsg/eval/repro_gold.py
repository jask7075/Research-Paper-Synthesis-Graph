"""Ground truth for the reproducibility layer, and the scorer that reads it.

Extension #4's premise is that hardware and software facts are objective and checkable, so
they can be scored by exact match rather than by a judge. That holds, but only if the gold
can express three different things, and the previous schema could express one:

    a value          the paper states it -> the system must find it
    `not_reported`   the paper does not state it -> the system must stay silent
    `None`           gold not yet established -> not scored at all

Collapsing the middle case into `None` is what made the old `repro_gold.jsonl` unscoreable:
a null meant both "the paper is silent" and "we have not looked", so a system that invented
a qubit count could not be distinguished from one that correctly reported nothing. That is
the same defect fixed in `metrics.py` — a default standing in for an unmeasured value — and
it is corrected the same way here.

The field set is quantum-shaped because the corpus is. Measured over 270 extracted papers,
`qubit_count` appears in 151 of 242 `Hardware` nodes and `gpu_type` in 107; the original
schema led with `gpu_type` / `gpu_count`, which fits a minority of this corpus.

One extraction quirk the scorer must absorb: the model writes the literal string
`"unknown"` into a field it cannot answer rather than omitting the key. `"unknown"`,
`"n/a"`, `""` and `None` are all read as "the system reported nothing".
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

#: Written in gold when the paper genuinely does not state a field. Distinct from `None`,
#: which means the gold has not been established and the field is skipped when scoring.
NOT_REPORTED = "not_reported"

#: Values the extractor emits when it has nothing. Treated as silence, not as an answer.
_SILENT = {"unknown", "n/a", "na", "none", "null", "not reported", "not specified", ""}

Outcome = Literal[
    "correct",          # gold has a value, system matched it
    "wrong",            # gold has a value, system reported a different one
    "missed",           # gold has a value, system reported nothing
    "correct_absence",  # gold says not_reported, system reported nothing
    "hallucinated",     # gold says not_reported, system invented a value
    "skipped",          # gold is None -- not established, not scored
]

#: Scored fields. Quantum first: this corpus reports qubit counts far more often than GPUs.
FIELDS = (
    "quantum_vendor",
    "device_name",
    "qubit_count",
    "gpu_type",
    "gpu_count",
    "code_url",
    "dataset_access",
)


class ReproRecord(BaseModel):
    """One paper's reproducibility ground truth.

    Every field is `None` by default so a partially-filled record is honest: unfilled
    fields are skipped rather than silently asserting the paper is silent.
    """

    paper_id: str
    quantum_vendor: str | None = Field(default=None, description="e.g. IBM, Google, IonQ")
    device_name: str | None = Field(default=None, description="e.g. ibmq_mumbai, Sycamore")
    qubit_count: int | str | None = None
    gpu_type: str | None = Field(default=None, description="e.g. A100, RTX 3090")
    gpu_count: int | str | None = None
    code_url: str | None = None
    dataset_access: str | None = Field(default=None, description="open | licensed | irb | none")
    note: str | None = None


def _is_silent(value: Any) -> bool:
    """True when a system value carries no information."""
    if value is None:
        return True
    return str(value).strip().lower() in _SILENT


def normalize(value: Any) -> str:
    """Case-, punctuation- and whitespace-insensitive form used for comparison.

    `ibmq_mumbai`, `IBM Mumbai` and `ibmq-mumbai` are the same device written three ways;
    scoring them as three different answers would measure formatting, not extraction.
    """
    s = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", s).strip()


def score_field(gold: Any, system: Any) -> Outcome:
    """Classify one (gold, system) field pair.

    Distinguishing `missed` from `hallucinated` is the point: they are opposite failures
    and a single accuracy number hides which one a system commits.
    """
    if gold is None:
        return "skipped"
    silent = _is_silent(system)
    if gold == NOT_REPORTED:
        return "correct_absence" if silent else "hallucinated"
    if silent:
        return "missed"
    return "correct" if normalize(gold) == normalize(system) else "wrong"


def score_record(gold: ReproRecord, system: dict[str, Any]) -> dict[str, Outcome]:
    """Outcome per field for one paper."""
    return {f: score_field(getattr(gold, f), system.get(f)) for f in FIELDS}


def accuracy(outcomes: list[Outcome]) -> float | None:
    """Fraction of *scored* fields that were right, counting a correct silence as right.

    `None` when nothing was scored — consistent with `rpsg.eval.metrics`, where a metric
    with nothing to measure returns `None` rather than a flattering default.
    """
    scored = [o for o in outcomes if o != "skipped"]
    if not scored:
        return None
    right = sum(1 for o in scored if o in ("correct", "correct_absence"))
    return right / len(scored)


def load_repro_gold(path: str) -> list[ReproRecord]:
    import json
    from pathlib import Path

    return [
        ReproRecord(**json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]