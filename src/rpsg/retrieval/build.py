"""One place that turns an arm name into a configured `System`.

Both entrypoints build arms: `06_run_eval.py` scores them and `ask.py` asks them
interactively. When each constructed its own, they drifted — `ask.py` was written when
`VectorRAGSystem` was the only arm and still said in its docstring that "everything the
eval harness scores goes through the same VectorRAGSystem", which stopped being true the
moment the typed, citation and agentic arms landed. A user-facing tool that silently
reaches a different system than the one the report measures is worse than one that cannot
reach it at all.

This lives in the package rather than in `scripts/` because `scripts/` is not importable
(the same reason `resolve_gold` moved into `gold_schema`).

Defaults are deliberately *not* re-chosen here. `hops`, `max_nodes`, `seeds` and the
agentic `top_k` come from the modules that measured them; passing `top_k` through to
anything but the vector arm would silently override a swept value with a CLI default.
"""

from __future__ import annotations

from typing import Any

from rpsg.config import get_settings
from rpsg.logging import get_logger
from rpsg.retrieval.agentic import AgenticSystem
from rpsg.retrieval.baselines import System, VectorRAGSystem
from rpsg.retrieval.citation_graph import CitationGraphSystem
from rpsg.retrieval.typed_graph import TypedGraphSystem
from rpsg.stores.base import Embedder
from rpsg.stores.graph_store import KuzuGraphStore
from rpsg.stores.vector_store import FaissVectorStore

log = get_logger(__name__)

#: Arm name -> which chunk corpus the vector arms read.
CORPUS = {"vector_abstract": "abstract", "vector_fulltext": "fulltext"}

#: Every arm either entrypoint can build. `_no_critique` is the required ablation from
#: §3.5: without it, "the loop helps" cannot be separated from "planning helps".
SYSTEMS = (
    *CORPUS,
    "typed_graph",
    "typed_graph_chunks",
    "citation_graph",
    "citation_graph_seeded",
    "agentic",
    "agentic_no_critique",
)

#: Arms that retrieve chunks up front, so a caller can show what was retrieved before
#: paying for synthesis. The graph arms traverse and the agentic arm plans first, so
#: neither has a single pre-synthesis chunk list to display.
RETRIEVES_CHUNKS_UP_FRONT = frozenset(CORPUS)


def _vector_store(settings: Any) -> FaissVectorStore:
    store = FaissVectorStore(str(settings.paths.vector_index), settings.embeddings.dim)
    store.load()
    return store


def build_system(
    name: str,
    *,
    embedder: Embedder,
    top_k: int | None = None,
    max_retrievals: int | None = None,
    graph_hints: bool = True,
    anchor: bool = True,
    stage_writes: bool = False,
) -> System:
    """Construct the named arm.

    `top_k` reaches the vector arms only — see the module docstring. The agentic flags are
    ignored by the other arms rather than rejected, so a caller can pass a uniform set.
    """
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name!r}; choose from {', '.join(SYSTEMS)}")
    settings = get_settings()

    if name.startswith("agentic"):
        # Same vector index and corpus as `vector_fulltext`, so the comparison isolates the
        # loop rather than the retrieval substrate.
        kwargs: dict[str, Any] = {}
        if max_retrievals is not None:
            kwargs["max_retrievals"] = max_retrievals
        return AgenticSystem(
            name=name,
            embedder=embedder,
            store=_vector_store(settings),
            corpus="fulltext",
            graph_store=(
                None if not graph_hints else KuzuGraphStore(str(settings.paths.kuzu_db))
            ),
            critique=not name.endswith("_no_critique"),
            stage_writes=stage_writes,
            anchor=anchor,
            **kwargs,
        )

    if name.startswith("citation_graph"):
        # The ablation: `cites` edges from S2 metadata instead of extracted typed edges,
        # everything else held constant. `_seeded` starts from vector retrieval rather than
        # title similarity, so the pair separates "citations are weak" from "title seeding
        # is weak".
        return CitationGraphSystem(
            name=name,
            embedder=embedder,
            store=KuzuGraphStore(str(settings.paths.kuzu_db)),
            vector_store=_vector_store(settings),
            seed_from="chunks" if name.endswith("_seeded") else "title",
        )

    if name.startswith("typed_graph"):
        # Reads the graph, not the vector index. `hops` and `max_nodes` take the module
        # defaults, both set from the retrieval sweep rather than chosen. The router variant
        # lets traversal pick the papers while chunks supply the evidence, so both arms are
        # compared on the same evidence unit.
        return TypedGraphSystem(
            name=name,
            embedder=embedder,
            store=KuzuGraphStore(str(settings.paths.kuzu_db)),
            vector_store=_vector_store(settings) if name == "typed_graph_chunks" else None,
        )

    # `top_k` omitted rather than defaulted here, so the class's own value stays the single
    # definition of the vector arms' retrieval breadth.
    if top_k is None:
        return VectorRAGSystem(
            name=name,
            embedder=embedder,
            store=_vector_store(settings),
            corpus=CORPUS[name],
        )
    return VectorRAGSystem(
        name=name,
        embedder=embedder,
        store=_vector_store(settings),
        corpus=CORPUS[name],
        top_k=top_k,
    )
