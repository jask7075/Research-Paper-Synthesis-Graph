"""Kuzu-backed GraphStore (Phase 1). Cypher-compatible, so it ports to Neo4j AuraDB.

Modeling choice: a single generic `Entity` node table keyed by `id`, discriminated by a
`type` property, plus one `REL` edge table discriminated by `type`. This keeps the schema
DDL tiny and lets new node/edge *types* be added without a migration — the type discipline
lives in `rpsg.extraction.schema`, not in a sprawling set of Kuzu tables. Swap to per-label
tables in Phase 2 if Neo4j query planning benefits.
"""

from __future__ import annotations

import json

from rpsg.extraction.schema import Edge, Node, SourceLayer
from rpsg.logging import get_logger
from rpsg.stores.base import GraphStore

log = get_logger(__name__)


class KuzuGraphStore(GraphStore):
    def __init__(self, db_path: str) -> None:
        import kuzu  # imported lazily so tests that don't touch the graph don't need it

        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)

    def init_schema(self) -> None:
        # Idempotent DDL. `attrs` is stored as a JSON string for schema-free type attrs.
        self._conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Entity(
                id STRING,
                type STRING,
                name STRING,
                aliases STRING,
                attrs STRING,
                source_layer STRING,
                confidence DOUBLE,
                evidence STRING,
                PRIMARY KEY (id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS REL(
                FROM Entity TO Entity,
                type STRING,
                attrs STRING,
                source_layer STRING,
                confidence DOUBLE,
                evidence STRING
            )
            """
        )
        log.info("Kuzu schema initialized.")

    def upsert_nodes(self, nodes: list[Node]) -> None:
        for n in nodes:
            self._conn.execute(
                """
                MERGE (e:Entity {id: $id})
                SET e.type = $type, e.name = $name, e.aliases = $aliases,
                    e.attrs = $attrs, e.source_layer = $source_layer,
                    e.confidence = $confidence, e.evidence = $evidence
                """,
                {
                    "id": n.id,
                    "type": n.type.value,
                    "name": n.name,
                    "aliases": json.dumps(n.aliases),
                    "attrs": json.dumps(n.attrs),
                    "source_layer": n.source_layer.value,
                    "confidence": n.confidence,
                    "evidence": json.dumps(n.evidence),
                },
            )

    def upsert_edges(self, edges: list[Edge]) -> None:
        for e in edges:
            self._conn.execute(
                """
                MATCH (a:Entity {id: $src}), (b:Entity {id: $dst})
                MERGE (a)-[r:REL {type: $type}]->(b)
                SET r.attrs = $attrs, r.source_layer = $source_layer,
                    r.confidence = $confidence, r.evidence = $evidence
                """,
                {
                    "src": e.src,
                    "dst": e.dst,
                    "type": e.type.value,
                    "attrs": json.dumps(e.attrs),
                    "source_layer": e.source_layer.value,
                    "confidence": e.confidence,
                    "evidence": json.dumps(e.evidence),
                },
            )

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        result = self._conn.execute(cypher, params or {})
        rows: list[dict] = []
        cols = result.get_column_names()
        while result.has_next():
            rows.append(dict(zip(cols, result.get_next(), strict=False)))
        return rows

    def promote_staged(self, node_ids: list[str] | None = None) -> int:
        # Explicit promotion only; no auto-merge (README design principle 3).
        where = "e.source_layer = $staged"
        params: dict = {"staged": SourceLayer.STAGED.value, "curated": SourceLayer.CURATED.value}
        if node_ids is not None:
            where += " AND e.id IN $ids"
            params["ids"] = node_ids
        rows = self.query(
            f"MATCH (e:Entity) WHERE {where} SET e.source_layer = $curated RETURN count(e) AS n",
            params,
        )
        return int(rows[0]["n"]) if rows else 0