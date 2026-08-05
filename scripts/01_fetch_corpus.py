"""Fetch corpus metadata (Tier A) and download PDFs.

    # ArXiv — no API key, guarantees full text, but no citation edges
    python scripts/01_fetch_corpus.py --source arxiv --limit 60 \
        --query 'all:"variational quantum eigensolver"'

    # Semantic Scholar — needs S2_API_KEY, supplies references -> citation graph
    python scripts/01_fetch_corpus.py --query "variational quantum eigensolver" --limit 50

Run several queries to build one corpus: records are appended and deduped by paper id.

Writes:
    data/external/papers.jsonl   paper metadata records
    data/raw/pdfs/<paper_id>.pdf downloaded PDFs (idempotent)

See `rpsg.ingestion.arxiv_client` for the source trade-off and how to backfill S2
metadata onto an ArXiv-built corpus once a key is available.
"""

from __future__ import annotations

import argparse
import json

from rpsg.config import get_settings
from rpsg.ingestion.arxiv_client import fetch_pdf, search_arxiv
from rpsg.ingestion.semantic_scholar import (
    S2Paper,
    SemanticScholarClient,
    cocitation_hubs,
)
from rpsg.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="search query (not needed with --expand-citations)")
    ap.add_argument(
        "--expand-citations",
        action="store_true",
        help="fetch the papers the existing corpus cites most, instead of searching",
    )
    ap.add_argument(
        "--min-cocitations",
        type=int,
        default=10,
        help="with --expand-citations: how many corpus papers must cite a target",
    )
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--no-pdf", action="store_true", help="metadata only, skip PDF download")
    ap.add_argument(
        "--source",
        choices=["s2", "arxiv"],
        default="s2",
        help="s2 needs S2_API_KEY but gives references; arxiv needs no key",
    )
    ap.add_argument(
        "--category", default="quant-ph", help="arXiv category filter; empty string disables"
    )
    ap.add_argument("--min-year", type=int, help="arXiv: earliest submission year")
    ap.add_argument("--sort", choices=["relevance", "date"], default="relevance")
    args = ap.parse_args()

    settings = get_settings()
    out_path = settings.paths.data_external / "papers.jsonl"

    if args.expand_citations:
        # Co-citation expansion: the citations to these papers already exist in the
        # corpus's reference lists, dangling because the target is absent. See
        # `cocitation_hubs` for why this beats fetching more search results.
        if not settings.s2_api_key:
            raise SystemExit("--expand-citations needs S2_API_KEY (it fetches by paper id).")
        if not out_path.exists():
            raise SystemExit(f"no corpus at {out_path} — run a search first.")
        existing = [
            S2Paper(**json.loads(line)) for line in out_path.read_text().splitlines() if line
        ]
        hub_ids = cocitation_hubs(existing, args.min_cocitations, limit=args.limit)
        if not hub_ids:
            raise SystemExit(f"no papers cited by >={args.min_cocitations} of the corpus.")
        client = SemanticScholarClient(api_key=settings.s2_api_key)
        papers = []
        for pid in hub_ids:
            try:
                papers.append(client.get_paper(pid))
            except Exception as exc:  # noqa: BLE001 - one missing hub must not stop the rest
                log.warning("could not fetch hub %s: %s", pid, exc)
        client.close()
        log.info("fetched %d/%d hub papers", len(papers), len(hub_ids))
    elif args.source == "arxiv":
        if not args.query:
            raise SystemExit("--query is required unless --expand-citations is used.")
        papers = search_arxiv(
            args.query,
            limit=args.limit,
            category=args.category or None,
            min_year=args.min_year,
            sort=args.sort,
        )
    else:
        if not args.query:
            raise SystemExit("--query is required unless --expand-citations is used.")
        if not settings.s2_api_key:
            raise SystemExit(
                "S2_API_KEY is not set. The unauthenticated Semantic Scholar pool returns a "
                "sustained 429 — set the key in .env, or use --source arxiv (no key needed)."
            )
        client = SemanticScholarClient(api_key=settings.s2_api_key)
        papers = client.search(args.query, limit=args.limit)
        client.close()

    settings.paths.data_external.mkdir(parents=True, exist_ok=True)
    out = out_path
    # Append (dedup by paperId) so multiple queries build one corpus.
    seen = set()
    if out.exists():
        seen = {json.loads(line)["paperId"] for line in out.read_text().splitlines() if line}
    with out.open("a") as fh:
        for p in papers:
            if p.paperId not in seen:
                fh.write(p.model_dump_json() + "\n")
                seen.add(p.paperId)
    log.info("corpus now %d papers -> %s", len(seen), out)

    if args.no_pdf:
        return
    pdf_dir = settings.paths.data_raw / "pdfs"
    got = 0
    for p in papers:
        path = fetch_pdf(p.arxiv_id, p.pdf_url, pdf_dir, p.paperId)
        got += path is not None
    log.info("downloaded %d/%d PDFs", got, len(papers))


if __name__ == "__main__":
    main()
