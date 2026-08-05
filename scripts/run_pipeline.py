"""Run the pipeline end to end, with preflight checks and per-stage accounting.

Not a pipeline stage itself — this sequences stages 01-06.

    python scripts/run_pipeline.py --query "variational quantum eigensolver" --limit 200
    python scripts/run_pipeline.py --from 02 --to 04         # resume without re-fetching
    python scripts/run_pipeline.py --dry-run                 # print the plan, run nothing
    python scripts/run_pipeline.py --from 02 --hash-embed    # offline embedder

Each stage is the same script you would run by hand; this only orders them, checks
their prerequisites up front, and reports what each produced. Stages 02 and 04 are
idempotent per paper, so a re-run resumes rather than repeating work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from rpsg.config import get_settings
from rpsg.logging import get_logger

log = get_logger(__name__)

STAGES = ["01", "02", "03", "04", "05", "06"]
_SCRIPT = {
    "01": "01_fetch_corpus.py",
    "02": "02_parse_pdfs.py",
    "03": "03_chunk.py",
    "04": "04_extract.py",
    "05": "05_build_stores.py",
    "06": "06_run_eval.py",
}
_LABEL = {
    "01": "fetch corpus (S2 metadata + PDFs)",
    "02": "parse PDFs -> sections",
    "03": "chunk sections",
    "04": "extract Tier B/C (LLM batch)",
    "05": "build vector index + graph",
    "06": "score against the gold set",
}


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _importable(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def preflight(stages: list[str], args: argparse.Namespace) -> list[str]:
    """Return blocking problems. Non-blocking concerns are logged as warnings."""
    settings = get_settings()
    problems: list[str] = []

    if "01" in stages:
        if not args.query:
            problems.append("stage 01 needs --query")
        if not settings.s2_api_key:
            problems.append(
                "stage 01 needs S2_API_KEY in .env — the unauthenticated Semantic Scholar "
                "pool returns a sustained 429, so the fetch cannot proceed without one"
            )
    if {"04", "06"} & set(stages) and not settings.openai_api_key:
        problems.append("stages 04/06 need OPENAI_API_KEY in .env")
    if {"05", "06"} & set(stages):
        if not _importable("faiss"):
            problems.append('stages 05/06 need faiss — pip install -e ".[vector]"')
        if not args.hash_embed and not _importable("sentence_transformers"):
            problems.append(
                'stages 05/06 need sentence-transformers — pip install -e ".[vector]", '
                "or pass --hash-embed to use the offline embedder"
            )
    if "02" in stages:
        pdf_dir = settings.paths.data_raw / "pdfs"
        pdfs = list(pdf_dir.glob("*.pdf"))
        if not pdfs:
            problems.append(
                f"stage 02 found no PDFs in {pdf_dir} — run stage 01, or drop PDFs there"
            )
        elif not (settings.paths.data_external / "papers.jsonl").exists():
            log.warning(
                "%d PDFs present but no papers.jsonl: abstracts and the Tier-A graph will "
                "be empty, and the vector_abstract corpus will have no chunks",
                len(pdfs),
            )
    if "06" in stages:
        gold = settings.paths.eval_gold / "queries.jsonl"
        if _count_lines(gold) == 0:
            problems.append(f"stage 06 needs gold queries in {gold}")
        elif "PLACEHOLDER" in gold.read_text():
            log.warning(
                "gold set contains PLACEHOLDER paper ids — stage 06 will run, but its "
                "deterministic metrics are meaningless until those are real ids"
            )
    return problems


def report(settings) -> str:
    """One-line snapshot of what currently exists on disk."""
    ext_path = settings.paths.data_processed / "extractions.jsonl"
    nodes = edges = 0
    if ext_path.exists():
        for line in ext_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                nodes += len(rec.get("nodes", []))
                edges += len(rec.get("edges", []))
    return (
        f"papers={_count_lines(settings.paths.data_external / 'papers.jsonl')} "
        f"parsed={len(list((settings.paths.data_interim / 'sections').glob('*.json')))} "
        f"chunks={_count_lines(settings.paths.data_interim / 'chunks.jsonl')} "
        f"extracted={_count_lines(ext_path)} (nodes={nodes} edges={edges}) "
        f"index={'yes' if settings.paths.vector_index.exists() else 'no'}"
    )


def stage_args(stage: str, args: argparse.Namespace) -> list[str]:
    if stage == "01":
        extra = ["--query", args.query, "--limit", str(args.limit)]
        return [*extra, "--no-pdf"] if args.no_pdf else extra
    if stage == "05":
        extra = []
        if args.hash_embed:
            extra.append("--hash-embed")
        if args.skip_graph:
            extra.append("--skip-graph")
        return extra
    if stage == "06":
        extra = ["--system", args.system]
        if args.hash_embed:
            extra.append("--hash-embed")
        if args.no_judge:
            extra.append("--no-judge")
        return extra
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_stage", choices=STAGES, default="01")
    ap.add_argument("--to", dest="to_stage", choices=STAGES, default="06")
    ap.add_argument("--query", help="stage 01 search query")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--no-pdf", action="store_true", help="stage 01: metadata only")
    ap.add_argument("--hash-embed", action="store_true", help="offline embedder for 05/06")
    ap.add_argument("--skip-graph", action="store_true", help="stage 05: vectors only")
    ap.add_argument("--system", default="vector_fulltext", help="stage 06 system")
    ap.add_argument("--no-judge", action="store_true", help="stage 06: deterministic metrics only")
    ap.add_argument("--dry-run", action="store_true", help="preflight + plan only")
    args = ap.parse_args()

    if STAGES.index(args.from_stage) > STAGES.index(args.to_stage):
        raise SystemExit(f"--from {args.from_stage} is after --to {args.to_stage}")
    stages = STAGES[STAGES.index(args.from_stage) : STAGES.index(args.to_stage) + 1]

    settings = get_settings()
    scripts_dir = Path(__file__).resolve().parent

    print("\nplan")
    for stage in stages:
        print(f"  {stage}  {_LABEL[stage]}")
    print(f"\nstate  {report(settings)}\n")

    problems = preflight(stages, args)
    if problems:
        print("preflight FAILED")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("preflight OK")

    if args.dry_run:
        print("\n--dry-run: nothing executed")
        return

    started = time.monotonic()
    for stage in stages:
        cmd = [sys.executable, str(scripts_dir / _SCRIPT[stage]), *stage_args(stage, args)]
        print(f"\n{'=' * 72}\n{stage}  {_LABEL[stage]}\n{'=' * 72}")
        t0 = time.monotonic()
        result = subprocess.run(cmd, check=False)
        elapsed = time.monotonic() - t0
        if result.returncode != 0:
            print(f"\nstage {stage} FAILED after {elapsed:.0f}s (exit {result.returncode})")
            print(f"state  {report(settings)}")
            print(f"resume with:  --from {stage}")
            raise SystemExit(result.returncode)
        print(f"\nstage {stage} ok in {elapsed:.0f}s   {report(settings)}")

    print(f"\npipeline complete in {time.monotonic() - started:.0f}s")
    print(f"state  {report(settings)}")


if __name__ == "__main__":
    main()