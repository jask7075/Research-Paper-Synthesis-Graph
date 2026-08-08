"""Extraction precision audit: draw a stratified sample, label it, score it.

    python scripts/audit_extraction.py --sample > eval/gold/extraction_audit.jsonl
    python scripts/audit_extraction.py --show                 # readable labelling sheet
    python scripts/audit_extraction.py --score                # precision once labelled

Reads  data/processed/extractions.jsonl
Writes eval/gold/extraction_audit.jsonl   (one row per sampled node, `correct` to fill)

Mark each row `"correct": true` if the node is a faithful extraction of what its evidence
quote says — right type, and a name that names the thing rather than describing what the
paper did with it. Mark `false` if the type is wrong, the name is not an entity, or the
evidence does not support it. Leave `null` for anything you cannot judge; unjudged rows are
excluded from the score rather than counted either way.

No API keys, no network. The sheet shows the node and its own evidence quote, never a
second system's opinion of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpsg.config import get_settings
from rpsg.eval.extraction_audit import precision, sample_nodes, summarize


def _nodes(path: Path) -> list[dict]:
    seen: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            for n in json.loads(line)["nodes"]:
                seen.setdefault(n["id"], n)
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="emit blank rows as JSONL")
    ap.add_argument("--show", action="store_true", help="print the sheet for labelling")
    ap.add_argument("--score", action="store_true", help="precision from the labelled file")
    ap.add_argument("--per-band", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    settings = get_settings()
    out_path = settings.paths.eval_gold / "extraction_audit.jsonl"

    if args.score:
        if not out_path.exists():
            raise SystemExit(f"no audit file at {out_path} — run --sample first")
        rows = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
        print(summarize(precision(rows)))
        return

    samples = sample_nodes(
        _nodes(settings.paths.data_processed / "extractions.jsonl"),
        per_band=args.per_band,
        seed=args.seed,
    )

    if args.show:
        titles = {}
        papers = settings.paths.data_external / "papers.jsonl"
        if papers.exists():
            for line in papers.read_text().splitlines():
                if line.strip():
                    p = json.loads(line)
                    titles[p["paperId"]] = (p.get("title") or "?").replace("\n", " ")
        for i, s in enumerate(samples, 1):
            print(f"\n{'=' * 92}\n[{i}/{len(samples)}]  {s.node_type}  conf={s.confidence:.2f}"
                  f"  band {s.band}")
            print(f"  name:     {s.name}")
            for q in s.evidence[:3]:
                print(f"  evidence: \"{q[:220]}\"")
            print(f"  paper:    {titles.get(s.paper_id or '', s.paper_id or '?')[:66]}")
            print('  -> correct: true if the node faithfully names what the quote states')
        return

    for s in samples:
        print(
            json.dumps(
                {
                    "node_id": s.node_id,
                    "node_type": s.node_type,
                    "name": s.name,
                    "confidence": s.confidence,
                    "band": s.band,
                    "evidence": s.evidence[:2],
                    "paper_id": s.paper_id,
                    "correct": None,
                    "note": None,
                }
            )
        )


if __name__ == "__main__":
    main()