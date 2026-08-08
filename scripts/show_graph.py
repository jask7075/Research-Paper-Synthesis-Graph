"""See what the typed graph holds around a query. No API keys, no cost.

    python scripts/show_graph.py "what mitigates barren plateaus?"
    python scripts/show_graph.py "..." --mermaid > docs/fig-neighbourhood.md
    python scripts/show_graph.py --node method:qaoa        # start from a known node
    python scripts/show_graph.py --stats                   # whole-graph shape

Rendering all 23,301 nodes produces a hairball that shows nothing. What is worth looking at
is the neighbourhood a *query* reaches, because that is what `TypedGraphSystem` walks and
what its answer is built from -- so this doubles as a diagnostic. A traversal that lands in
the right region and still scores badly is a synthesis problem; one that wanders somewhere
unrelated is a seeding problem, and the two need opposite fixes.

Seeding, hops and the node cap are taken from `TypedGraphSystem` rather than re-chosen, so
what is drawn is what retrieval actually saw. Anything else would be a picture of a
different system.

`--mermaid` emits a diagram that renders in GitHub markdown. Edge labels carry the typed
relation, since "these two are connected" is far less informative than "this method
`addresses` that problem".
"""

from __future__ import annotations

import argparse
from collections import Counter

from rpsg.config import get_settings
from rpsg.retrieval.typed_graph import TypedGraphSystem
from rpsg.stores.embedder import SentenceTransformerEmbedder
from rpsg.stores.graph_store import KuzuGraphStore

#: Mermaid node shapes by type, so the picture is readable without a legend.
SHAPE = {
    "Method": ("[", "]"), "Problem": ("([", "])"), "Claim": (">", "]"),
    "Limitation": ("{{", "}}"), "Dataset": ("[(", ")]"), "Hardware": ("[/", "/]"),
    "Software": ("[[", "]]"), "ReproducibilityArtifact": ("((", "))"), "Paper": ("[", "]"),
}


def _safe(text: str, n: int = 46) -> str:
    """Mermaid chokes on quotes, brackets and newlines inside labels."""
    t = " ".join(str(text).split())[:n]
    return t.replace('"', "'").replace("[", "(").replace("]", ")").replace("|", "/")


def stats(store: KuzuGraphStore) -> None:
    print("nodes")
    for r in store.query("MATCH (e:Entity) RETURN e.type AS t, count(*) AS n ORDER BY n DESC"):
        print(f"  {r['n']:>7}  {r['t']}")
    print("\nedges")
    for r in store.query(
        "MATCH ()-[r:REL]->() RETURN r.type AS t, count(*) AS n ORDER BY n DESC"
    ):
        print(f"  {r['n']:>7}  {r['t']}")
    print("\nmost connected nodes")
    for r in store.query(
        "MATCH (e:Entity)-[r:REL]-() RETURN e.name AS name, e.type AS t, count(r) AS d "
        "ORDER BY d DESC LIMIT 10"
    ):
        print(f"  {r['d']:>5}  {r['t']:12} {(r['name'] or '')[:56]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--node", help="start from this node id instead of a query")
    ap.add_argument("--mermaid", action="store_true", help="emit a renderable diagram")
    ap.add_argument("--stats", action="store_true", help="whole-graph shape, no traversal")
    ap.add_argument("--cypher", help="run one Cypher statement and print the rows")
    ap.add_argument("--max-draw", type=int, default=40, help="cap nodes in the diagram")
    args = ap.parse_args()

    settings = get_settings()
    store = KuzuGraphStore(str(settings.paths.kuzu_db))

    if args.cypher:
        # Ad-hoc querying without standing up Kuzu Explorer, which needs Docker, a tag
        # matching the storage format, and takes the same exclusive lock anyway.
        rows = store.query(args.cypher)
        if not rows:
            print("(no rows)")
            return
        cols = list(rows[0])
        print("  ".join(f"{c:<28}" for c in cols))
        print("  ".join("-" * 28 for _ in cols))
        for r in rows[:200]:
            print("  ".join(f"{str(r.get(c, ''))[:28]:<28}" for c in cols))
        if len(rows) > 200:
            print(f"... {len(rows) - 200} more rows")
        return

    if args.stats or not (args.query or args.node):
        stats(store)
        return

    embedder = SentenceTransformerEmbedder(
        settings.embeddings.model_name, settings.embeddings.dim, settings.embeddings.batch_size
    )
    system = TypedGraphSystem("show", embedder, store)

    if args.node:
        rows = store.query(
            "MATCH (e:Entity) WHERE e.id = $id RETURN e.id AS id, e.name AS name, "
            "e.type AS type, e.attrs AS attrs, e.evidence AS evidence", {"id": args.node}
        )
        if not rows:
            raise SystemExit(f"no node {args.node}")
        seeds = rows
    else:
        seeds = system._seed_nodes(args.query)

    from rpsg.retrieval.typed_graph import GraphHit

    hits = [GraphHit(s, 0, None) for s in seeds]
    hits += system._expand([s["id"] for s in seeds])
    hits = hits[: system._max_nodes]

    papers = {h.paper_id for h in hits if h.paper_id}
    if not args.mermaid:
        print(f"query: {args.query or args.node}")
        print(f"\n{len(hits)} nodes reached across {len(papers)} papers")
        print(f"  by type: {dict(Counter(h.node['type'] for h in hits))}")
        print(f"  by hop:  {dict(sorted(Counter(h.hop for h in hits).items()))}")
        print(f"  via:     {dict(Counter(h.via for h in hits if h.via).most_common(8))}")
        print(f"\n{'hop':>3}  {'via':16} {'type':12} name")
        for h in hits[:60]:
            print(f"{h.hop:>3}  {(h.via or '(seed)'):16} {h.node['type']:12} "
                  f"{(h.node['name'] or '')[:58]}")
        if len(hits) > 60:
            print(f"     ... {len(hits) - 60} more")
        return

    drawn = hits[: args.max_draw]
    ids = {h.node["id"] for h in drawn}
    alias = {nid: f"n{i}" for i, nid in enumerate(ids)}
    print("```mermaid")
    print("graph LR")
    for h in drawn:
        lo, hi = SHAPE.get(h.node["type"], ("[", "]"))
        mark = ":::seed" if h.hop == 0 else ""
        print(f'  {alias[h.node["id"]]}{lo}"{_safe(h.node["name"])}"{hi}{mark}')
    seen = set()
    for row in store.query(
        "MATCH (a:Entity)-[r:REL]->(b:Entity) WHERE a.id IN $ids AND b.id IN $ids "
        "RETURN DISTINCT a.id AS src, b.id AS dst, r.type AS type", {"ids": list(ids)}
    ):
        key = (row["src"], row["dst"], row["type"])
        if key in seen:
            continue
        seen.add(key)
        print(f'  {alias[row["src"]]} -->|{row["type"]}| {alias[row["dst"]]}')
    print("  classDef seed stroke-width:3px")
    print("```")
    print(f"\n<!-- {len(drawn)} of {len(hits)} nodes, {len(seen)} edges, "
          f"{len(papers)} papers; seeds have a thick border -->")


if __name__ == "__main__":
    main()