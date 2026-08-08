"""Contradiction audit: draw a stratified sample, label it blind, score it.

    python scripts/audit_contradictions.py --sample > eval/gold/contradiction_audit.jsonl
    python scripts/audit_contradictions.py --show                # labelling sheet
    python scripts/audit_contradictions.py --score               # agreement once labelled

Reads  data/processed/contradiction_verdicts.json, data/processed/extractions.jsonl
Writes eval/gold/contradiction_audit.jsonl  (one row per pair, `human` to fill)

For each pair, decide independently:

  refutes     the two cannot both be true -- a direct empirical or logical conflict
  undercuts   A does not contradict B but weakens its support: a scope limit, a failure
              mode, a caveat narrowing where B holds
  neither     same topic but compatible, or too vague to judge

Default to `neither`. Claims about different systems, different regimes or different
quantities do not conflict merely by sharing vocabulary -- two papers reporting different
numbers for different setups is not a contradiction. That is the specific error the
eyeball check found, so it is the one to watch for.

Leave `null` for anything you cannot call; unjudged rows are excluded rather than counted.

**The sheet does not show what the model decided.** Shown a verdict, a human agrees with
it, and the result would be a review rather than a measurement. The comparison happens at
--score.
"""

from __future__ import annotations

import argparse
import json

from rpsg.config import get_settings
from rpsg.eval.contradiction_audit import precision, sample_pairs, summarize


def _nodes(settings) -> dict[str, dict]:
    out: dict[str, dict] = {}
    path = settings.paths.data_processed / "extractions.jsonl"
    for line in path.read_text().splitlines():
        if line.strip():
            for n in json.loads(line)["nodes"]:
                out.setdefault(n["id"], n)
    return out


def _adjudicated(settings) -> list[dict]:
    """Rebuild the full pair list -- accepted and rejected -- from the verdict cache.

    The cache is keyed by `a_name␟b_name` and holds every pair including `neither`, which
    `contradictions.json` does not: it stores only accepted edges. Sampling the rejected
    class needs the cache.
    """
    cache = json.loads(
        (settings.paths.data_processed / "contradiction_verdicts.json").read_text()
    )
    by_name: dict[str, list[dict]] = {}
    for n in _nodes(settings).values():
        by_name.setdefault(n["name"], []).append(n)

    rows = []
    for key, res in cache.items():
        a_name, _, b_name = key.partition("␟")
        a_list, b_list = by_name.get(a_name), by_name.get(b_name)
        if not a_list or not b_list:
            continue
        a, b = a_list[0], b_list[0]
        rows.append({
            "pair_id": key,
            "a_text": a_name,
            "b_text": b_name,
            "a_evidence": list(a.get("evidence") or [])[:2],
            "b_evidence": list(b.get("evidence") or [])[:2],
            "a_paper": (a.get("attrs") or {}).get("from_paper"),
            "b_paper": (b.get("attrs") or {}).get("from_paper"),
            "similarity": 0.0,
            "model_verdict": res.get("verdict", "neither"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--per-verdict", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    settings = get_settings()
    out_path = settings.paths.eval_gold / "contradiction_audit.jsonl"

    if args.score:
        if not out_path.exists():
            raise SystemExit(f"no audit file at {out_path} -- run --sample first")
        rows = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
        accepted = None
        edges_path = settings.paths.data_processed / "contradictions.json"
        if edges_path.exists():
            accepted = len(json.loads(edges_path.read_text())["edges"])
        print(summarize(precision(rows), accepted_total=accepted))
        return

    samples = sample_pairs(
        _adjudicated(settings), per_verdict=args.per_verdict, seed=args.seed
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
            print(f"\n{'=' * 94}\n[{i}/{len(samples)}]")
            print(f"\n  CLAIM A: {s.a_text}")
            for q in s.a_evidence[:1]:
                print(f"    quote: {q[:260]}")
            print(f"    paper: {titles.get(s.a_paper or '', s.a_paper or '?')[:64]}")
            print(f"\n  CLAIM B: {s.b_text}")
            for q in s.b_evidence[:1]:
                print(f"    quote: {q[:260]}")
            print(f"    paper: {titles.get(s.b_paper or '', s.b_paper or '?')[:64]}")
            print("\n  -> refutes / undercuts / neither ?")
        return

    for s in samples:
        print(json.dumps({
            "pair_id": s.pair_id,
            "a_text": s.a_text,
            "b_text": s.b_text,
            "a_paper": s.a_paper,
            "b_paper": s.b_paper,
            "model_verdict": s.model_verdict,
            "human": None,
            "note": None,
        }))


if __name__ == "__main__":
    main()