"""Read the seven repro fields out of a paper's extracted nodes, and score them.

`repro_gold.py` defines the outcomes and compares one (gold, system) pair. What was
missing is the step before that: an extraction is a bag of typed nodes, not a record with
seven fields, so something has to decide what the system *reported* for `qubit_count`
before anything can be scored.

Where each field comes from, measured over the corpus rather than assumed:

    quantum_vendor   Hardware.attrs quantum_vendor (141) or vendor (126)
    device_name      Hardware.name -- there is no device attr; the node's own name is
                     the only place a device is ever written
    qubit_count      Hardware.attrs qubit_count (156), Dataset.attrs qubit_count (5)
    gpu_type         Hardware.attrs gpu_type (89)
    gpu_count        Hardware.attrs gpu_count (85)
    code_url         ReproducibilityArtifact.attrs code_url, Software.attrs url,
                     Dataset.attrs access_url
    dataset_access   ReproducibilityArtifact.attrs dataset_access, Dataset.attrs
                     access_type

**Two deliberate leniencies, and one deliberate strictness.**

A paper usually yields several `Hardware` nodes -- one run reported four ways -- so a
field has several candidate values. A candidate that matches gold is preferred over one
that does not. That measures whether the extraction *contains* the right fact, not whether
some later stage would have picked the right one out of several, because no such stage
exists; charging the extractor for an aggregation choice the pipeline never makes would
attribute the failure to the wrong component.

Matching is token-containment, not equality: gold `A100` matches an extracted
`NVIDIA A100-SXM-80GB`, and gold `IBM` matches `IBM Quantum hardware`. Requiring equality
would score formatting rather than extraction, which is the same reason `normalize()`
already flattens case and punctuation. Containment is token-level, so `4` does not match
`24 qubits`.

The strictness is on absence. When gold says `not_reported`, *any* candidate is a
hallucination -- leniency in matching cannot rescue a value that should not exist. This is
the asymmetry the three-state schema was built for, so it is preserved exactly.

Candidates are never pooled across fields. `91c10ab4` writes `Sycamore` into
`quantum_vendor` when Sycamore is a device, and reading that as a `device_name` hit would
hide a real field confusion behind a correct-looking score.
"""

from __future__ import annotations

from typing import Any

from rpsg.eval.repro_gold import FIELDS, NOT_REPORTED, ReproRecord, _is_silent, normalize
from rpsg.eval.repro_gold import Outcome, score_field

#: (node type, attribute) pairs feeding each field, in preference order. `None` as the
#: attribute means the node's own `name`.
SOURCES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "quantum_vendor": (("Hardware", "quantum_vendor"), ("Hardware", "vendor")),
    "device_name": (("Hardware", None),),
    "qubit_count": (("Hardware", "qubit_count"), ("Dataset", "qubit_count")),
    "gpu_type": (("Hardware", "gpu_type"),),
    "gpu_count": (("Hardware", "gpu_count"),),
    "code_url": (
        ("ReproducibilityArtifact", "code_url"),
        ("Software", "url"),
        ("Dataset", "access_url"),
    ),
    "dataset_access": (
        ("ReproducibilityArtifact", "dataset_access"),
        ("Dataset", "access_type"),
    ),
}


def candidates(nodes: list[dict[str, Any]], field: str) -> list[str]:
    """Every non-silent value the extraction offers for one field, in source order."""
    out: list[str] = []
    for node_type, attr in SOURCES[field]:
        for n in nodes:
            if n.get("type") != node_type:
                continue
            value = n.get("name") if attr is None else (n.get("attrs") or {}).get(attr)
            if not _is_silent(value) and str(value) not in out:
                out.append(str(value))
    return out


def matches(gold: Any, candidate: str) -> bool:
    """Equal after normalization, or one's tokens contained in the other's.

    Containment is checked as a contiguous run of whole tokens, so `A100` matches
    `NVIDIA A100 SXM 80GB` but `4` does not match `24 qubits`.
    """
    g, c = normalize(gold).split(), normalize(candidate).split()
    if not g or not c:
        return False
    short, long_ = (g, c) if len(g) <= len(c) else (c, g)
    return any(long_[i : i + len(short)] == short for i in range(len(long_) - len(short) + 1))


def reported(nodes: list[dict[str, Any]], gold: Any, field: str) -> str | None:
    """The value to score: a candidate matching gold if one exists, else the first.

    Preferring a match is what makes this "does the extraction contain the fact" rather
    than "does it rank the fact first". When gold is `not_reported` no candidate can
    match, so the first is returned and correctly scores as a hallucination.
    """
    cands = candidates(nodes, field)
    if not cands:
        return None
    if gold is not None and gold != NOT_REPORTED:
        for c in cands:
            if matches(gold, c):
                return c
    return cands[0]


def score_paper(gold: ReproRecord, nodes: list[dict[str, Any]]) -> dict[str, Outcome]:
    """Outcome per field, delegating absence to `score_field` and relaxing equality.

    `score_field` compares a single pair and is right to demand equality after
    normalization; it cannot know that `A100` and `NVIDIA A100-SXM-80GB` name one card,
    because that judgement needs the candidate list. So the absence logic --
    skipped / missed / correct_absence / hallucinated -- is taken from `score_field`
    unchanged, and only its `correct` vs `wrong` verdict is revisited here. Without this
    the token-containment rule would be inert: it would choose the right candidate and
    then have it scored wrong for carrying a model number.
    """
    out: dict[str, Outcome] = {}
    for f in FIELDS:
        want = getattr(gold, f)
        outcome = score_field(want, reported(nodes, want, f))
        if outcome == "wrong" and any(matches(want, c) for c in candidates(nodes, f)):
            outcome = "correct"
        out[f] = outcome
    return out


def summarize(per_paper: dict[str, dict[str, Outcome]]) -> str:
    """Outcome counts overall and per field.

    The headline accuracy is reported alongside the share of the gold that is
    `not_reported`, because on this corpus that share is ~73% -- a system that emits
    nothing at all scores that well, so the aggregate alone is close to uninformative and
    the `correct` / `correct_absence` split is what carries the signal.
    """
    from collections import Counter

    every = [o for outcomes in per_paper.values() for o in outcomes.values()]
    scored = [o for o in every if o != "skipped"]
    if not scored:
        return "nothing scored -- every gold field is null"

    counts = Counter(scored)
    right = counts["correct"] + counts["correct_absence"]
    silence = counts["correct_absence"] / len(scored)
    lines = [
        f"{len(per_paper)} papers, {len(scored)} scored fields "
        f"({len(every) - len(scored)} skipped)",
        f"accuracy {right / len(scored):.1%}  "
        f"({counts['correct']} correct + {counts['correct_absence']} correct_absence)",
        f"  of which correct silence: {silence:.1%} -- an empty system would score this much",
        "",
        f"  {'field':16} {'correct':>8} {'absence':>8} {'wrong':>7} {'missed':>7} "
        f"{'hallucinated':>13}",
    ]
    for f in FIELDS:
        c = Counter(o[f] for o in per_paper.values() if o[f] != "skipped")
        lines.append(
            f"  {f:16} {c['correct']:>8} {c['correct_absence']:>8} {c['wrong']:>7} "
            f"{c['missed']:>7} {c['hallucinated']:>13}"
        )
    return "\n".join(lines)