"""Record real runs for the public demo.

    python scripts/record_demo.py                      # the curated set, all four arms
    python scripts/record_demo.py --dry-run            # list what would run, spend nothing
    python scripts/record_demo.py --qids rel-001 ref-002
    python scripts/record_demo.py --arms vector_fulltext typed_graph

GitHub Pages serves static files, so the public demo cannot run retrieval or call a model.
It replays what this script recorded. That is only worth doing if the recordings are real:
every run here goes through `AskService` — the same object the live server calls — against
the same index and graph, so what a visitor reads is what the arm actually answered, down
to the retrieved chunks and the token bill.

Questions come from `eval/gold/queries.full34.jsonl` rather than being invented for the
demo, so the public page shows the system on the set the report scores, including the
query types it handles least well. A demo built from questions chosen after seeing the
answers would be an advert, not evidence.

Output is `docs/demo/recordings.json`, and each run is the exact JSON body /api/ask
returns — which is why `render.js` can draw a recording and a live answer with one code
path. Re-running overwrites it; the file is committed, because Pages serves it.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from rpsg.config import PROJECT_ROOT, get_settings
from rpsg.llm.usage import USAGE
from rpsg.logging import get_logger
from rpsg.web.service import AskResult, AskService, ServiceError

log = get_logger(__name__)

GOLD = Path("eval/gold/queries.full34.jsonl")
OUT = Path("docs/demo/recordings.json")

#: Eight questions spanning all four query types. Chosen for coverage, before any of them
#: was run — see the module docstring on why that ordering matters.
QIDS = (
    "rel-001",    # relational: barren-plateau mitigations
    "rel-t01",    # relational: modularity maximization approaches
    "rel-t04",    # relational: QAOA mixer Hamiltonians
    "ref-001",    # refutation: does identity-block init solve it?
    "ref-002",    # refutation: can error mitigation restore trainability?
    "look-001",   # lookup: what is a barren plateau?
    "look-002",   # lookup: what defines the NISQ era?
    "open-001",   # open-directions: what about barren plateaus is unresolved?
)

#: One vector baseline, both graph arms, and the agentic loop. `vector_abstract` is left
#: out: it is the weakest arm and adds a column nobody compares against.
ARMS = ("vector_fulltext", "typed_graph", "citation_graph", "agentic")


def _load_gold(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"no gold query set at {path} (run from the repository root)")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["qid"]: r for r in rows}


def _total_cost() -> float | None:
    """What this process has spent so far, or None if any model used is unpriced."""
    pricing = get_settings().models.pricing or {}
    from rpsg.llm.usage import cost_usd

    total = 0.0
    for model, usage in USAGE.snapshot().items():
        cost = cost_usd(usage, pricing.get(model))
        if cost is None:
            return None
        total += cost
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", nargs="*", default=list(QIDS), help="gold query ids to record")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), help="arms to record each on")
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--with-evidence",
        action="store_true",
        help="record the full excerpt block too; roughly triples the file",
    )
    ap.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    args = ap.parse_args()

    gold = _load_gold(args.gold)
    missing = [q for q in args.qids if q not in gold]
    if missing:
        raise SystemExit(f"not in {args.gold}: {', '.join(missing)}")

    runs = [(q, a) for q in args.qids for a in args.arms]
    print(f"{len(args.qids)} question(s) x {len(args.arms)} arm(s) = {len(runs)} run(s)")
    for qid in args.qids:
        print(f"  {qid:<10} {gold[qid]['query_type']:<16} {gold[qid]['query'][:80]}")
    if args.dry_run:
        print("\n--dry-run: nothing was asked, nothing was spent.")
        return

    service = AskService()
    settings = get_settings()
    questions: list[dict] = []
    failures: list[str] = []

    for qid in args.qids:
        row = gold[qid]
        entry = {
            "qid": qid,
            "query": row["query"],
            "query_type": row["query_type"],
            "must_cite": row.get("must_cite") or [],
            "runs": {},
        }
        for arm in args.arms:
            print(f"\n>>> {qid} on {arm}")
            try:
                result: AskResult = service.ask(
                    row["query"], system=arm, show_evidence=args.with_evidence
                )
            except ServiceError as exc:
                # Recorded as a failure and skipped rather than aborting the batch: one
                # unavailable arm should not throw away the runs already paid for.
                print(f"    refused: {exc}")
                failures.append(f"{qid}/{arm}: {exc}")
                continue
            entry["runs"][arm] = {
                "question": result.question,
                "arm": result.arm,
                "retrieval_only": result.retrieval_only,
                "elapsed_s": round(result.elapsed_s, 2),
                "answer": result.answer,
                "cited_paper_ids": result.cited_paper_ids,
                "evidence": result.evidence,
                "trace": result.trace,
                "hits": [vars(h) for h in result.hits],
                "usage": [vars(u) for u in result.usage],
            }
            cited = len(result.cited_paper_ids)
            print(f"    {result.elapsed_s:.1f}s, {len(result.hits)} hit(s), {cited} paper(s) cited")
        questions.append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "synthesis_model": settings.models.synthesis_model,
        "embedding_model": settings.embeddings.model_name,
        "gold_set": str(args.gold),
        "arms": args.arms,
        "questions": questions,
    }
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=False))

    size_kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out.relative_to(PROJECT_ROOT) if args.out.is_absolute() else args.out}"
          f"  ({size_kb:,.0f} KB)")
    if failures:
        print(f"\n{len(failures)} run(s) refused:")
        for f in failures:
            print(f"  {f}")
    total = _total_cost()
    print(f"\ntotal spend this run: {'unpriced' if total is None else f'${total:,.4f}'}")
    print(USAGE.summary())


if __name__ == "__main__":
    main()
