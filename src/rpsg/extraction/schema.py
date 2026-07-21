"""The frozen, tiered knowledge-graph schema.

Design rule (freeze before ingestion): every node/edge type must have (a) a written
extraction prompt, (b) a slot in the eval, and (c) a query that consumes it. A type with
no consumer is bloat — cut it, don't carry it.

Tiers, by extraction reliability:
    A  Metadata      Paper, Author, Venue, Dataset      cheap, high-precision (mostly APIs)
    B  Semantic      Method, Problem, Claim, Limitation LLM-extracted, medium precision
    C  Relational    the edges (evaluated_on … refutes) LLM-inferred, lowest precision

Reproducibility layer (extension #4, the Iteration-2 core): Hardware, Software,
ReproducibilityArtifact — medium difficulty but *objectively evaluable*.

Every node/edge carries provenance so the curated vs. staged distinction is enforceable:
    - source_layer: CURATED (offline, reviewed) vs STAGED (query-time; never auto-merged)
    - confidence:   extractor confidence in [0, 1]; the synthesizer hedges on low values
    - evidence:     the chunk id(s) the extraction was grounded on
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Tier(str, Enum):
    A_METADATA = "A_metadata"
    B_SEMANTIC = "B_semantic"
    C_RELATIONAL = "C_relational"
    REPRO = "reproducibility"


class SourceLayer(str, Enum):
    CURATED = "curated"   # offline, reviewed — metrics run against this
    STAGED = "staged"     # written by the agent at query time; provenance-tagged, not merged


class NodeType(str, Enum):
    # Tier A — metadata
    PAPER = "Paper"
    AUTHOR = "Author"
    VENUE = "Venue"
    DATASET = "Dataset"
    # Tier B — semantic
    METHOD = "Method"
    PROBLEM = "Problem"
    CLAIM = "Claim"
    LIMITATION = "Limitation"
    # Reproducibility layer
    HARDWARE = "Hardware"
    SOFTWARE = "Software"
    REPRO_ARTIFACT = "ReproducibilityArtifact"


class EdgeType(str, Enum):
    # Tier A (free, from S2 references / metadata)
    AUTHORED_BY = "authored_by"       # Paper -> Author
    PUBLISHED_IN = "published_in"     # Paper -> Venue
    CITES = "cites"                   # Paper -> Paper  (this IS the citation-graph baseline)
    # Tier C (LLM-inferred; ordered easiest -> hardest)
    EVALUATED_ON = "evaluated_on"     # Method -> Dataset      (structured tables; easiest of C)
    ADDRESSES = "addresses"           # Method -> Problem
    BUILDS_ON = "builds_on"           # Method -> Method
    REFUTES = "refutes"               # Claim  -> Claim        (very hard, lowest precision)
    UNDERCUTS = "undercuts"           # Claim  -> Claim        (scrutinize vs REFUTES; may merge)
    # Reproducibility layer
    REQUIRES = "requires"             # Paper  -> Hardware
    USES = "uses"                     # Paper  -> Software (carries `version`)
    PROVIDES = "provides"             # Paper  -> ReproducibilityArtifact


#: Which tier each type belongs to — used by reporting to break precision down by tier.
NODE_TIER: dict[NodeType, Tier] = {
    NodeType.PAPER: Tier.A_METADATA,
    NodeType.AUTHOR: Tier.A_METADATA,
    NodeType.VENUE: Tier.A_METADATA,
    NodeType.DATASET: Tier.A_METADATA,
    NodeType.METHOD: Tier.B_SEMANTIC,
    NodeType.PROBLEM: Tier.B_SEMANTIC,
    NodeType.CLAIM: Tier.B_SEMANTIC,
    NodeType.LIMITATION: Tier.B_SEMANTIC,
    NodeType.HARDWARE: Tier.REPRO,
    NodeType.SOFTWARE: Tier.REPRO,
    NodeType.REPRO_ARTIFACT: Tier.REPRO,
}

EDGE_TIER: dict[EdgeType, Tier] = {
    EdgeType.AUTHORED_BY: Tier.A_METADATA,
    EdgeType.PUBLISHED_IN: Tier.A_METADATA,
    EdgeType.CITES: Tier.A_METADATA,
    EdgeType.EVALUATED_ON: Tier.C_RELATIONAL,
    EdgeType.ADDRESSES: Tier.C_RELATIONAL,
    EdgeType.BUILDS_ON: Tier.C_RELATIONAL,
    EdgeType.REFUTES: Tier.C_RELATIONAL,
    EdgeType.UNDERCUTS: Tier.C_RELATIONAL,
    EdgeType.REQUIRES: Tier.REPRO,
    EdgeType.USES: Tier.REPRO,
    EdgeType.PROVIDES: Tier.REPRO,
}


class DatasetAccess(str, Enum):
    OPEN = "open"
    LICENSED = "licensed"
    IRB = "irb"
    UNKNOWN = "unknown"


class Node(BaseModel):
    """A graph node. `key` is the canonical id used for entity resolution / dedup."""

    id: str = Field(description="Stable unique id, e.g. 'method:proximal-policy-optimization'.")
    type: NodeType
    name: str = Field(description="Surface form / canonical label.")
    aliases: list[str] = Field(default_factory=list, description="Known surface variants.")
    attrs: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        description="Type-specific fields, e.g. Hardware{vendor, qubit_count, gpu_count}.",
    )
    # Provenance
    source_layer: SourceLayer = SourceLayer.CURATED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list, description="Grounding chunk ids.")


class Edge(BaseModel):
    src: str = Field(description="Source node id.")
    dst: str = Field(description="Destination node id.")
    type: EdgeType
    attrs: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    # Provenance
    source_layer: SourceLayer = SourceLayer.CURATED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """What the extractor returns for a single paper (or a single chunk)."""

    paper_id: str
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    def by_tier(self) -> dict[Tier, tuple[int, int]]:
        """(#nodes, #edges) per tier — feeds per-tier precision reporting."""
        counts: dict[Tier, list[int]] = {t: [0, 0] for t in Tier}
        for n in self.nodes:
            counts[NODE_TIER[n.type]][0] += 1
        for e in self.edges:
            counts[EDGE_TIER[e.type]][1] += 1
        return {t: (c[0], c[1]) for t, c in counts.items()}


# JSON Schema handed to the extractor (Anthropic structured outputs / `output_config.format`).
# Kept minimal & flat: the model returns nodes+edges; ids are normalized downstream.
EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": [t.value for t in NodeType]},
                    "name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "attrs": {"type": "object"},
                    "confidence": {"type": "number"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["type", "name", "confidence", "evidence_quote"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": [t.value for t in EdgeType]},
                    "src_name": {"type": "string"},
                    "dst_name": {"type": "string"},
                    "attrs": {"type": "object"},
                    "confidence": {"type": "number"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["type", "src_name", "dst_name", "confidence", "evidence_quote"],
            },
        },
    },
    "required": ["nodes", "edges"],
}