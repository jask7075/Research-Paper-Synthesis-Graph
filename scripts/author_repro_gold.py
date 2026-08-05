"""Authoring aid for repro_gold. No API keys, no network.

    python scripts/author_repro_gold.py --sample 20 > eval/gold/repro_gold.draft.jsonl
    python scripts/author_repro_gold.py --show <paper_id>
    python scripts/author_repro_gold.py --validate

`--show` prints the passages of a paper that mention hardware, so the fields can be filled
from the source. It deliberately does **not** print what the extractor produced for that
paper. Confirming the system's own output would drive per-field accuracy toward 1.0 by
construction and measure nothing — the same circularity `author_gold.py` avoids by using
BM25 rather than the system's retriever to choose `must_cite`.

Sampling is stratified: papers that never mention a device are included on purpose,
because `not_reported` is a scored outcome. A sample drawn only from papers with hardware
would measure recall on easy cases and never catch a hallucinated value.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from rpsg.config import get_settings
from rpsg.eval.repro_gold import FIELDS, NOT_REPORTED, load_repro_gold

#: Passages worth reading when filling a record. Broad on purpose — a missed passage
#: becomes a wrong `not_reported`, which is the most expensive gold error there is.
_SIGNAL = re.compile(
    r"\bqubit|\bqpu\b|quantum (?:computer|device|processor|hardware|backend)|"
    r"\bibm\w*|\bibmq\b|sycamore|rigetti|ionq|quantinuum|\bgpu\b|\bcpu\b|a100|v100|"
    r"rtx|cuda|\bsimulator\b|wall[- ]clock|runtime|github\.com|zenodo|available at",
    re.I,
)


def _chunks_by_paper() -> dict[str, list[dict]]:
    path = get_settings().paths.data_interim / "chunks.jsonl"
    out: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            out.setdefault(c["paper_id"], []).append(c)
    return out


def _passages(chunks: list[dict], limit: int = 12) -> list[tuple[str, str]]:
    """(section_type, sentence) for sentences that mention hardware or availability."""
    hits: list[tuple[str, str]] = []
    for c in chunks:
        for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
            if _SIGNAL.search(sent) and 20 < len(sent) < 400:
                hits.append((c["section_type"], " ".join(sent.split())))
                if len(hits) >= limit:
                    return hits
    return hits


def show(paper_id: str) -> None:
    chunks = _chunks_by_paper().get(paper_id, [])
    if not chunks:
        raise SystemExit(f"no chunks for {paper_id}")
    passages = _passages(chunks)
    print(f"# {paper_id}   ({len(chunks)} chunks, {len(passages)} hardware-ish passages)\n")
    if not passages:
        print("  (nothing matched — likely a genuine `not_reported` across the board)")
    for sect, sent in passages:
        print(f"  [{sect}] {sent}")
    print("\nFill from the passages above, not from the graph. Use:")
    print('  a value          the paper states it')
    print(f'  "{NOT_REPORTED}"   the paper is silent  <- a scored outcome, not a gap')
    print("  null             you have not checked   <- skipped when scoring")
    print("\n" + json.dumps({"paper_id": paper_id, **{f: None for f in FIELDS}}))


def sample(n: int, seed: int) -> None:
    by_paper = _chunks_by_paper()
    rng = random.Random(seed)
    with_hw = sorted(p for p, cs in by_paper.items() if _passages(cs, limit=1))
    without = sorted(p for p in by_paper if p not in set(with_hw))
    # ~3:1. Papers with no hardware mention still matter: they are where a system that
    # invents values gets caught.
    take_hw = min(len(with_hw), round(n * 0.75))
    picked = rng.sample(with_hw, take_hw) + rng.sample(without, min(len(without), n - take_hw))
    for pid in picked:
        print(json.dumps({"paper_id": pid, **{f: None for f in FIELDS}, "note": None}))


def validate(path: Path) -> None:
    records = load_repro_gold(str(path))
    by_paper = _chunks_by_paper()
    problems = 0
    for r in records:
        if r.paper_id not in by_paper:
            print(f"  {r.paper_id}: not in the corpus")
            problems += 1
        filled = [f for f in FIELDS if getattr(r, f) is not None]
        if not filled:
            print(f"  {r.paper_id}: every field null — nothing here will be scored")
            problems += 1
    scored = sum(1 for r in records for f in FIELDS if getattr(r, f) is not None)
    absent = sum(1 for r in records for f in FIELDS if getattr(r, f) == NOT_REPORTED)
    print(f"{len(records)} records, {scored} scoreable fields ({absent} of them `{NOT_REPORTED}`)")
    if absent == 0 and scored:
        print("  warning: no `not_reported` anywhere — hallucination cannot be detected")
    print("ok" if not problems else f"{problems} problem(s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="emit N blank records, stratified")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", help="print a paper's hardware passages")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    gold = get_settings().paths.eval_gold / "repro_gold.jsonl"
    if args.show:
        show(args.show)
    elif args.sample:
        sample(args.sample, args.seed)
    elif args.validate:
        validate(gold)
    else:
        ap.error("one of --sample / --show / --validate")


if __name__ == "__main__":
    main()
