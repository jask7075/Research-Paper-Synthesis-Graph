"""Authoring aid for the gold query set. No API keys, no network.

    python scripts/author_gold.py --validate
    python scripts/author_gold.py --find "barren plateau mitigation ansatz"
    python scripts/author_gold.py --show d699e0958fe1d8a4c1d691765f7e11b823fa606f

`--validate` is the one to run before every commit of the gold set: a typo in a
40-character hash silently scores `must_cite_recall = 0` and reads as a retrieval failure
rather than a data-entry error.

`--find` ranks papers by BM25 over raw section text. That is deliberate: it shares no
machinery with the SPECTER embeddings the system retrieves with. Using the system's own
retriever to choose `must_cite` would guarantee it retrieves those papers and drive
`must_cite_recall` to ~1.0 by construction — measuring nothing. Keyword search finds
papers that *say the words*, leaving open whether the embedder can find them too, which
is the question the metric is supposed to answer.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from rpsg.config import get_settings
from rpsg.eval.gold_schema import GoldRecord, QueryType

#: Papers whose extraction yielded zero nodes — retrievable as text, but contributing
#: nothing to the typed graph, so a poor choice of `must_cite` while that is unfixed.
ZERO_NODE_PAPERS = {
    "29b37457c74a34d07e07ef0c7c3d60bfd5306624",
    "ff744c5fd2628eea16654da81c5b1e5080e51d35",
}

#: Recommended query-type mix as a *fraction* of the set, so the guidance scales with size
#: rather than being pinned to whatever the set happened to be when this was written.
#: Weighted toward relational and refutation per gold_schema: those are where a typed graph
#: should beat vector search, and a natural mix averages the advantage away.
TARGET_MIX = {
    QueryType.RELATIONAL: (0.35, 0.45),
    QueryType.REFUTATION: (0.20, 0.30),
    QueryType.OPEN_DIRECTIONS: (0.15, 0.25),
    QueryType.LOOKUP: (0.10, 0.20),
}


def _mix_bounds(n: int, lo_frac: float, hi_frac: float) -> tuple[int, int]:
    """Absolute count range for a target fraction, never narrower than one query."""
    lo, hi = round(n * lo_frac), round(n * hi_frac)
    return lo, max(hi, lo + 1)

_WORD = re.compile(r"[a-z0-9]+")
# fmt: off
_STOP = frozenset({
    "the", "a", "an", "of", "for", "in", "on", "with", "using", "based", "and", "or", "to",
    "is", "are", "be", "as", "by", "from", "that", "this", "it", "at", "we", "our", "their",
    "its", "can", "may", "which", "such", "been", "has", "have", "not", "but", "they",
    "than", "then", "between",
})
# fmt: on


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 2]


class Corpus:
    """Papers, their metadata, and per-paper term counts for BM25."""

    def __init__(self, settings) -> None:  # noqa: ANN001 - Settings, avoided for import cost
        papers_path = settings.paths.data_external / "papers.jsonl"
        chunks_path = settings.paths.data_interim / "chunks.jsonl"
        if not papers_path.exists():
            sys.exit(f"no corpus manifest at {papers_path} — run stage 01 first")

        self.meta: dict[str, dict] = {}
        for line in papers_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("paperId"):
                    self.meta[rec["paperId"]] = rec

        self.tf: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.length: dict[str, int] = collections.defaultdict(int)
        #: paper -> (section_type, text) for snippet display
        self.sections: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        if chunks_path.exists():
            for line in chunks_path.read_text().splitlines():
                if not line.strip():
                    continue
                c = json.loads(line)
                if c.get("corpus") != "fulltext":
                    continue
                pid, toks = c["paper_id"], _tokens(c["text"])
                self.tf[pid].update(toks)
                self.length[pid] += len(toks)
                self.sections[pid].append((c["section_type"], c["text"]))

        self.with_fulltext = set(self.tf)
        self.df: collections.Counter = collections.Counter()
        for counts in self.tf.values():
            self.df.update(counts.keys())

    def title(self, pid: str) -> str:
        return (self.meta.get(pid, {}).get("title") or "?").strip()

    def year(self, pid: str) -> str:
        return str(self.meta.get(pid, {}).get("year") or "????")

    def search(self, query: str, limit: int = 12, k1: float = 1.5, b: float = 0.75) -> list[tuple]:
        """BM25 over per-paper term counts. Returns (score, paper_id, best_snippet)."""
        n_docs = len(self.tf)
        if not n_docs:
            sys.exit("no full-text chunks found — run stages 02-03 first")
        avg_len = sum(self.length.values()) / n_docs
        qterms = _tokens(query)
        scored: list[tuple[float, str]] = []
        for pid, counts in self.tf.items():
            score = 0.0
            for term in qterms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1 + (n_docs - self.df[term] + 0.5) / (self.df[term] + 0.5))
                norm = freq * (k1 + 1) / (
                    freq + k1 * (1 - b + b * self.length[pid] / avg_len)
                )
                score += idf * norm
            if score > 0:
                scored.append((score, pid))
        scored.sort(reverse=True)
        return [(s, pid, self._snippet(pid, qterms)) for s, pid in scored[:limit]]

    def _snippet(self, pid: str, qterms: list[str]) -> str:
        """The section whose text hits the most distinct query terms."""
        best, best_hits = "", -1
        wanted = set(qterms)
        for stype, text in self.sections.get(pid, []):
            hits = len(wanted & set(_tokens(text)))
            if hits > best_hits:
                best, best_hits = f"({stype}) {' '.join(text.split())}", hits
        return best


def cmd_find(corpus: Corpus, query: str, limit: int) -> None:
    results = corpus.search(query, limit=limit)
    if not results:
        print(f"no paper matched {query!r}")
        return
    print(f'BM25 over section text — {len(results)} hits for "{query}"')
    print("(keyword search, independent of the SPECTER index the system retrieves with)\n")
    for rank, (score, pid, snippet) in enumerate(results, 1):
        flags = []
        if pid in ZERO_NODE_PAPERS:
            flags.append("ZERO-NODE — avoid in must_cite")
        if pid not in corpus.with_fulltext:
            flags.append("NO FULL TEXT")
        mark = f"   [{'; '.join(flags)}]" if flags else ""
        print(f"{rank:>2}. {score:6.1f}  {corpus.title(pid)[:70]}  ({corpus.year(pid)}){mark}")
        print(f'      "paper:{pid}"')
        print(f"      {snippet[:190]}\n")


def cmd_show(corpus: Corpus, pid: str) -> None:
    pid = pid.removeprefix("paper:")
    if pid not in corpus.meta:
        sys.exit(f"{pid} is not in the corpus")
    m = corpus.meta[pid]
    print(f"{m.get('title')}\n{corpus.year(pid)} | {m.get('venue') or 'n/a'}")
    print(f"paper:{pid}\n")
    if m.get("abstract"):
        print("ABSTRACT\n" + " ".join(m["abstract"].split()) + "\n")
    secs = corpus.sections.get(pid, [])
    if not secs:
        print("no full text parsed for this paper")
        return
    by_type: collections.Counter = collections.Counter(s for s, _ in secs)
    listing = ", ".join(f"{k}×{v}" for k, v in by_type.most_common())
    print(f"SECTIONS ({len(secs)} chunks): {listing}")
    for stype in ("limitations", "conclusion", "discussion", "availability"):
        for st, text in secs:
            if st == stype:
                print(f"\n--- {stype} ---\n{' '.join(text.split())[:900]}")
                break


def cmd_validate(corpus: Corpus, path: Path) -> int:
    if not path.exists():
        sys.exit(f"no gold file at {path}")
    errors: list[str] = []
    warnings: list[str] = []
    records: list[GoldRecord] = []
    seen_qids: set[str] = set()

    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = GoldRecord(**json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"line {lineno}: {str(exc).splitlines()[0]}")
            continue
        records.append(rec)
        if rec.qid in seen_qids:
            errors.append(f"line {lineno}: duplicate qid {rec.qid!r}")
        seen_qids.add(rec.qid)
        if not rec.facets:
            errors.append(f"{rec.qid}: no facets — judge `coverage` has nothing to score")
        if "PLACEHOLDER" in line:
            errors.append(f"{rec.qid}: contains PLACEHOLDER — metrics will be meaningless")

        # Deduplicated: a paper legitimately appears in both must_cite and key_claims,
        # and reporting the same typo twice makes the error list harder to act on.
        cited: dict[str, str] = {}
        for ref in rec.must_cite:
            cited.setdefault(ref, "must_cite")
        for kc in rec.key_claims:
            for ref in kc.papers:
                cited.setdefault(ref, "key_claims")
        for ref, where in cited.items():
            if not ref.startswith("paper:"):
                errors.append(f"{rec.qid}: {where} {ref!r} lacks the mandatory 'paper:' prefix")
                continue
            pid = ref.removeprefix("paper:")
            if pid not in corpus.meta:
                errors.append(f"{rec.qid}: {where} paper:{pid} is not in the corpus (typo?)")
            elif pid not in corpus.with_fulltext:
                warnings.append(
                    f"{rec.qid}: {where} paper:{pid} has metadata but no full text — "
                    "vector retrieval can only reach its abstract"
                )
            elif pid in ZERO_NODE_PAPERS:
                warnings.append(f"{rec.qid}: {where} paper:{pid} extracted zero nodes")
        if rec.query_type is QueryType.REFUTATION and not rec.known_refutations:
            warnings.append(
                f"{rec.qid}: refutation query with no known_refutations — "
                "`refutation_surfaced` will vacuously score 1.0"
            )
        if rec.must_cite and len(rec.must_cite) > 4:
            warnings.append(
                f"{rec.qid}: {len(rec.must_cite)} must_cite papers; each is scored as "
                "mandatory, so aspirational entries belong in key_claims"
            )

    print(f"{len(records)} record(s) in {path}\n")
    mix = collections.Counter(r.query_type for r in records)
    print("query-type mix (recommended range in brackets):")
    for qt, (lo_frac, hi_frac) in TARGET_MIX.items():
        lo, hi = _mix_bounds(len(records), lo_frac, hi_frac)
        got = mix.get(qt, 0)
        ok = "ok " if lo <= got <= hi else "->  "
        print(f"  {ok} {qt.value:<16} {got}   [{lo}-{hi}]")

    graded = sum(1 for r in records if r.grade)
    print(f"\nhand-graded for calibration: {graded}/{len(records)}  (~20 needed for kappa)")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print(f"\n{len(errors)} ERROR(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nno errors." + ("" if warnings else " no warnings."))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--find", metavar="QUERY", help="rank in-corpus papers by BM25")
    ap.add_argument("--show", metavar="PAPER_ID", help="dump one paper's metadata and sections")
    ap.add_argument("--validate", action="store_true", help="check the gold file")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--gold", type=Path, help="gold file (default eval/gold/queries.jsonl)")
    args = ap.parse_args()

    if not (args.find or args.show or args.validate):
        ap.error("give one of --find, --show, --validate")

    settings = get_settings()
    corpus = Corpus(settings)
    if args.find:
        cmd_find(corpus, args.find, args.limit)
    if args.show:
        cmd_show(corpus, args.show)
    if args.validate:
        gold = args.gold or settings.paths.eval_gold / "queries.jsonl"
        raise SystemExit(cmd_validate(corpus, gold))


if __name__ == "__main__":
    main()
