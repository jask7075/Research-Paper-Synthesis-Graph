"""Score the extracted reproducibility layer against repro_gold. No API keys, no network.

    python scripts/score_repro.py
    python scripts/score_repro.py --per-paper        # one line per paper
    python scripts/score_repro.py --errors           # only the fields that went wrong

Reads  eval/gold/repro_gold.jsonl, data/processed/extractions.jsonl

This is the pass that turns the authored gold into a number. Until it runs, every claim
about the repro layer is a volume -- how many `Hardware` nodes exist -- and none of it says
whether any value is right.
"""

from __future__ import annotations

import argparse
import json

from rpsg.config import get_settings
from rpsg.eval.repro_gold import FIELDS, load_repro_gold
from rpsg.eval.repro_scorer import candidates, score_paper, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-paper", action="store_true")
    ap.add_argument("--errors", action="store_true", help="show wrong/missed/hallucinated")
    args = ap.parse_args()

    settings = get_settings()
    gold = load_repro_gold(str(settings.paths.eval_gold / "repro_gold.jsonl"))

    by_paper: dict[str, list[dict]] = {}
    for line in (settings.paths.data_processed / "extractions.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_paper[r["paper_id"]] = r["nodes"]

    results, absent = {}, []
    for record in gold:
        nodes = by_paper.get(record.paper_id)
        if nodes is None:
            absent.append(record.paper_id)
            continue
        results[record.paper_id] = score_paper(record, nodes)

    if absent:
        print(f"{len(absent)} gold papers have no extraction, skipped: "
              f"{', '.join(p[:10] for p in absent)}\n")

    print(summarize(results))

    if args.per_paper or args.errors:
        gold_by_id = {g.paper_id: g for g in gold}
        print()
        for pid, outcomes in results.items():
            bad = {f: o for f, o in outcomes.items()
                   if o in ("wrong", "missed", "hallucinated")}
            if args.errors and not bad:
                continue
            print(f"{pid[:10]}  " + "  ".join(
                f"{f}={o}" for f, o in outcomes.items() if not args.errors or f in bad))
            for f in (bad if args.errors else FIELDS):
                if f in bad:
                    got = candidates(by_paper[pid], f)
                    print(f"     {f:16} gold={getattr(gold_by_id[pid], f)!r:26} "
                          f"extracted={got[:3]}")


if __name__ == "__main__":
    main()