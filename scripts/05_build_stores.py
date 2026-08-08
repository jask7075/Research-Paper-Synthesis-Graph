"""Build the vector index and the Tier-A graph.

    python scripts/05_build_stores.py
    python scripts/05_build_stores.py --hash-embed   # offline smoke test, no model download

Reads  data/interim/chunks.jsonl, data/external/papers.jsonl, data/processed/extractions.jsonl
Writes data/processed/vectors.faiss(+meta), data/processed/rpsg.kuzu
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rpsg.config import get_settings
from rpsg.extraction.entity_resolution import apply_map, build_entity_map
from rpsg.extraction.schema import Edge, ExtractionResult, Node
from rpsg.ingestion.semantic_scholar import S2Paper, to_graph
from rpsg.logging import get_logger
from rpsg.stores.base import Chunk
from rpsg.stores.embedder import HashEmbedder, SentenceTransformerEmbedder
from rpsg.stores.graph_store import KuzuGraphStore
from rpsg.stores.vector_store import FaissVectorStore

log = get_logger(__name__)


def build_vectors(hash_embed: bool) -> None:
    settings = get_settings()
    chunks_path = settings.paths.data_interim / "chunks.jsonl"
    chunks = [
        Chunk(**json.loads(line)) for line in chunks_path.read_text().splitlines() if line
    ]
    if hash_embed:
        embedder = HashEmbedder(dim=settings.embeddings.dim)
    else:
        embedder = SentenceTransformerEmbedder(
            settings.embeddings.model_name, settings.embeddings.dim, settings.embeddings.batch_size
        )
    store = FaissVectorStore(str(settings.paths.vector_index), settings.embeddings.dim)
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts)
    store.add(chunks, embeddings)
    store.persist()
    log.info("indexed %d chunks", len(chunks))


def _reset_graph(db_path: Path) -> None:
    """Delete the graph before rebuilding it.

    `upsert_nodes` uses MERGE, so building into an existing database *accumulates*: nodes
    from an earlier extraction survive even when the current one no longer produces them.
    After the Hardware routing fix that made the graph a union of two extraction
    generations rather than a picture of one, which is a correctness problem before it is
    a size one — you cannot tell which run a node came from.

    It also caused the failure that surfaced this: merging a fresh 25k-node extraction on
    top of a 64 MB database exhausted Kuzu's buffer pool ("Unable to allocate memory! The
    buffer pool is full and no memory could be freed!").

    The graph is derived data, fully reconstructible from papers.jsonl + extractions.jsonl,
    so rebuilding from empty is the honest default. The sidecar .wal / .shadow files go too;
    leaving a WAL behind replays writes belonging to the database we just removed.
    """
    for p in (db_path, Path(f"{db_path}.wal"), Path(f"{db_path}.shadow")):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    log.info("cleared existing graph at %s", db_path)


def build_graph() -> None:
    settings = get_settings()
    _reset_graph(Path(settings.paths.kuzu_db))
    store = KuzuGraphStore(str(settings.paths.kuzu_db))
    store.init_schema()

    # Tier A from S2 metadata
    papers_path = settings.paths.data_external / "papers.jsonl"
    if papers_path.exists():
        papers = [
            S2Paper(**json.loads(line)) for line in papers_path.read_text().splitlines() if line
        ]
        nodes, edges = to_graph(papers, max_references=200)
        store.upsert_nodes(nodes)
        store.upsert_edges(edges)

    # Tier B/C + repro from extractions (curated layer)
    ext_path = settings.paths.data_processed / "extractions.jsonl"
    if ext_path.exists():
        results = [
            ExtractionResult(**json.loads(line))
            for line in ext_path.read_text().splitlines()
            if line
        ]
        # Entity resolution is applied here rather than during extraction because the
        # acronym rule needs the whole corpus: whether `VQE` is ambiguous depends on every
        # paper, not the one being extracted. Doing it per paper would make a node's
        # identity depend on ingestion order.
        all_nodes = [n.model_dump() for r in results for n in r.nodes]
        mapping, merges = build_entity_map(all_nodes)
        # Second tier: pairs the deterministic rules cannot see. Embeddings nominate
        # candidates, a model rules on each, and only accepted pairs merge. Skipped
        # unless a verdict cache exists, so a build never silently starts paying for
        # adjudication -- run scripts/merge_entities.py to produce one.
        mapping.update(_semantic_merges(all_nodes, mapping, settings))
        _write_entity_map(settings.paths.data_processed / "entity_map.json", mapping, merges)
        for result in results:
            store.upsert_nodes(_resolve_nodes(result.nodes, mapping))
            store.upsert_edges(_resolve_edges(result.edges, mapping))
        # Cross-paper contradictions, if a pass has been run. Applied after the per-paper
        # edges so both live in one graph, and resolved through the same map -- a merged
        # endpoint must not leave a dangling edge.
        store.upsert_edges(_resolve_edges(_contradiction_edges(settings), mapping))
    log.info("graph built at %s", settings.paths.kuzu_db)


def _semantic_merges(
    nodes: list[dict], deterministic: dict[str, str], settings: object
) -> dict[str, str]:
    """Accepted verdicts from the cache, as an id map.

    Cache-only by design: adjudicating ~3,500 pairs costs money, and a store rebuild
    should be free and repeatable. `scripts/merge_entities.py` populates the cache.

    Nodes the deterministic pass already moved are skipped — resolving through two maps
    would need chaining, and `apply_map` is deliberately single-hop.
    """
    from rpsg.extraction.semantic_merge import Verdict, merge_map

    cache_path = settings.paths.data_processed / "merge_verdicts.json"  # type: ignore[attr-defined]
    if not cache_path.exists():
        log.info("no merge_verdicts.json — semantic tier skipped")
        return {}
    cache = json.loads(cache_path.read_text())
    by_name: dict[str, list[str]] = {}
    for n in nodes:
        by_name.setdefault(n["name"], []).append(n["id"])
    verdicts: list[Verdict] = []
    for key, result in cache.items():
        if not result.get("same"):
            continue
        a_name, _, b_name = key.partition("␟")
        for a_id in by_name.get(a_name, []):
            for b_id in by_name.get(b_name, []):
                if a_id != b_id and a_id not in deterministic and b_id not in deterministic:
                    verdicts.append(Verdict(a_id, b_id, a_name, b_name, 1.0, True, ""))
    extra = merge_map(verdicts)
    log.info("semantic tier: %d further ids merged", len(extra))
    return extra


def _contradiction_edges(settings: object) -> list[Edge]:
    """`refutes` / `undercuts` edges from a cross-paper pass, if one exists.

    File-only by design, like the semantic merge cache: adjudication costs money and a
    store rebuild should be free. `scripts/find_contradictions.py` produces it.

    Per-paper extraction can only see contradictions a paper states about itself, which
    yielded 8 `refutes` and 25 `undercuts` across 271 papers -- and 1 of 9 refutation gold
    queries surfaced its contradiction on every arm. A graph cannot route to a
    disagreement it does not encode.
    """
    path = settings.paths.data_processed / "contradictions.json"  # type: ignore[attr-defined]
    if not path.exists():
        log.info("no contradictions.json -- cross-paper contradiction pass skipped")
        return []
    data = json.loads(path.read_text())
    # An adjudicated set is a *proposal* until it has been audited. The first pass scored
    # 32.5% edge precision -- ~2,000 of 3,072 edges asserted a disagreement no paper made
    # -- and was rejected. Applying on mere file presence would have put those in the graph
    # on the next rebuild, silently contradicting the report. The flag makes the audit
    # decision enforceable in code rather than a note someone has to remember.
    if not data.get("approved"):
        log.warning(
            "contradictions.json is present but not approved (%d edges) -- skipping. "
            "Audit with scripts/audit_contradictions.py, then set approved: true.",
            len(data.get("edges", [])),
        )
        return []
    raw = data.get("edges", [])
    edges = [Edge(**e) for e in raw]
    log.info("cross-paper contradictions: %d edges", len(edges))
    return edges


def _resolve_nodes(nodes: list[Node], mapping: dict[str, str]) -> list[Node]:
    """Point merged nodes at their canonical id.

    The node keeps its own name and evidence; only the id moves. `upsert_nodes` MERGEs on
    id, so the surviving node is whichever the last writer describes — losing the alias
    text is a known cost of resolving at build time rather than storing aliases.
    """
    out = []
    for n in nodes:
        target = apply_map(n.id, mapping)
        out.append(n if target == n.id else n.model_copy(update={"id": target}))
    return out


def _resolve_edges(edges: list[Edge], mapping: dict[str, str]) -> list[Edge]:
    """Repoint both endpoints, and drop edges that become self-loops.

    A merge can turn `A -> B` into `A -> A`: two names for one entity that the extractor
    thought were related. Keeping those would manufacture self-referential edges that no
    paper asserts.
    """
    out = []
    for e in edges:
        src, dst = apply_map(e.src, mapping), apply_map(e.dst, mapping)
        if src == dst:
            continue
        if (src, dst) == (e.src, e.dst):
            out.append(e)
        else:
            out.append(e.model_copy(update={"src": src, "dst": dst}))
    return out


def _write_entity_map(path: Path, mapping: dict[str, str], merges: list) -> None:
    """Persist the map so a merge can be inspected without re-running the build."""
    by_rule: dict[str, int] = {}
    for m in merges:
        by_rule[m.rule] = by_rule.get(m.rule, 0) + 1
    path.write_text(
        json.dumps(
            {
                "merged_ids": len(mapping),
                "by_rule": by_rule,
                "mapping": mapping,
                "merges": [m._asdict() for m in merges],
            },
            indent=2,
        )
    )
    log.info("entity map: %d ids merged %s -> %s", len(mapping), by_rule, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-embed", action="store_true", help="offline hashing embedder")
    ap.add_argument("--skip-graph", action="store_true")
    ap.add_argument("--skip-vectors", action="store_true")
    args = ap.parse_args()
    if not args.skip_vectors:
        build_vectors(args.hash_embed)
    if not args.skip_graph:
        build_graph()


if __name__ == "__main__":
    main()