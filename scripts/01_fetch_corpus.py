"""Fetch corpus metadata (Tier A) from Semantic Scholar and download PDFs.

    python scripts/01_fetch_corpus.py --query "variational quantum eigensolver" --limit 50

Writes:
    data/external/papers.jsonl   raw S2 records
    data/raw/pdfs/<paper_id>.pdf downloaded PDFs (idempotent)
"""

from __future__ import annotations

import argparse
import json

from rpsg.config import get_settings
from rpsg.ingestion.arxiv_client import fetch_pdf
from rpsg.ingestion.semantic_scholar import SemanticScholarClient
from rpsg.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="S2 relevance search query")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--no-pdf", action="store_true", help="metadata only, skip PDF download")
    args = ap.parse_args()

    settings = get_settings()
    client = SemanticScholarClient(api_key=settings.s2_api_key)
    papers = client.search(args.query, limit=args.limit)
    client.close()

    settings.paths.data_external.mkdir(parents=True, exist_ok=True)
    out = settings.paths.data_external / "papers.jsonl"
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
