"""API-based corpus extractor (the one-time offline batch).

Why API, not a local 8B: extraction reliability IS the system's quality ceiling, and the
base-corpus build is a one-time batch. A small hosted model extracts materially better
than a local 8-14B for a few dollars over a 200-paper corpus — cheap insurance on the
number everything downstream inherits. The local model is reserved for the
Iteration-2-onward query-time graph-growth loop.

This module extracts per (paper, section) with schema-constrained structured outputs
(provider-neutral, via `rpsg.llm`), then
normalizes surface names into stable node ids and stitches edges by name. Entity
resolution (canonicalizing "PPO" == "Proximal Policy Optimization" across papers) is a
separate Iteration-2 step; here we id nodes deterministically from their normalized name so
the same string collapses to one node.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from rpsg.config import get_settings
from rpsg.extraction.prompts import SYSTEM_PROMPT, build_user_prompt
from rpsg.extraction.schema import (
    EXTRACTION_JSON_SCHEMA,
    Edge,
    EdgeType,
    ExtractionResult,
    Node,
    NodeType,
    SourceLayer,
)
from rpsg.ingestion.chunking import DROP_SECTION_TYPES, Section
from rpsg.llm import get_chat_client
from rpsg.logging import get_logger

log = get_logger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


def _node_id(node_type: NodeType, name: str) -> str:
    slug = _SLUG.sub("-", name.strip().lower()).strip("-")[:80]
    return f"{node_type.value.lower()}:{slug}"


def _coerce_attr(value: object) -> str | int | float | bool | None:
    """Flatten one attr value into the scalar space `Node.attrs` declares.

    Models answer with nested values in practice — an observed `attrs` was
    `{"extensions": ["excited states", "error mitigation"]}`. That is real
    information, so it is JSON-encoded rather than dropped, and `attrs` stays
    flat and scalar-valued as the graph store and per-tier reporting assume.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def _parse_attrs(raw: object) -> dict:
    """`attrs` arrives as a JSON string — see the note in `rpsg.extraction.schema`.

    A dict is also accepted so extraction files written before that change still
    load, and a malformed string degrades to `{}` rather than killing the paper.
    """
    parsed: object = raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): _coerce_attr(v) for k, v in parsed.items()}


def _coerce_confidence(raw: object) -> float:
    """Clamp the model's confidence into the [0, 1] that `Node`/`Edge` enforce."""
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    return min(max(value, 0.0), 1.0)


class Extractor:
    """Wraps a `rpsg.llm.ChatClient` with the frozen extraction schema.

    `source_layer` defaults to CURATED because this is the offline, reviewed batch. The
    query-time loop constructs its own Extractor with `source_layer=STAGED`.
    """

    def __init__(
        self,
        model: str | None = None,
        source_layer: SourceLayer = SourceLayer.CURATED,
        max_tokens: int = 4096,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.models.extraction_model
        self._client = get_chat_client(self._model)
        self._source_layer = source_layer
        self._max_tokens = max_tokens
        self._min_node_confidence = settings.extraction.min_node_confidence
        self._min_edge_confidence = settings.extraction.min_edge_confidence
        self._max_workers = settings.extraction.max_workers
        # Pinned. Re-extracting the same 21 papers twice with an identical prompt gave 9 of
        # 147 differing field outcomes, so an unpinned corpus cannot be rebuilt to itself --
        # the same defect 3.6c found in the judge, on the layer the entire graph is built
        # from. Temperature 0 is not bit-determinism, but it removes the sampling component.
        self._temperature = settings.models.extraction_temperature

    def _call(self, user_prompt: str) -> dict:
        # Structured outputs guarantee a schema-valid object.
        return self._client.json(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=EXTRACTION_JSON_SCHEMA,
            schema_name="extraction",
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    def extract_section(self, paper_id: str, section: Section) -> ExtractionResult:
        prompt = build_user_prompt(paper_id, section.title, section.section_type, section.text)
        # `_normalize` is inside the guard on purpose: a schema-valid payload can
        # still fail model validation downstream (an out-of-range confidence, a
        # shape `attrs` can't hold), and that must degrade one section rather
        # than abort a batch that is expensive to restart.
        try:
            raw = self._call(prompt)
            return self._normalize(paper_id, raw)
        except Exception as exc:  # noqa: BLE001 - one bad section must not kill the batch
            log.warning("Extraction failed for %s/%s: %s", paper_id, section.title, exc)
            return ExtractionResult(paper_id=paper_id)

    def extract_paper(self, paper_id: str, sections: list[Section]) -> ExtractionResult:
        """Extract across all sections and merge into one result per paper.

        Sections the chunker discards (`DROP_SECTION_TYPES` — i.e. bibliographies)
        are skipped here too. Without this the batch spent a call per reference
        list and mined nodes out of citation strings.

        Sections are extracted concurrently: each is an independent request with no
        shared state or ordering dependency, and the stage is purely network-bound
        (a sequential 29-minute run measured 0.2% CPU). `executor.map` preserves
        section order so a given corpus produces the same merge order regardless of
        which request finishes first. Per-section failures are already swallowed by
        `extract_section`, so one bad section cannot fail the pool.
        """
        targets = [s for s in sections if s.section_type not in DROP_SECTION_TYPES]
        merged = ExtractionResult(paper_id=paper_id)

        if not targets:
            return merged

        workers = max(1, min(self._max_workers, len(targets)))
        if workers == 1:
            parts = [self.extract_section(paper_id, s) for s in targets]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts = list(pool.map(lambda s: self.extract_section(paper_id, s), targets))

        for part in parts:
            merged.nodes.extend(part.nodes)
            merged.edges.extend(part.edges)

        # Dedup nodes by id (a method can appear in several sections).
        by_id: dict[str, Node] = {}
        for n in merged.nodes:
            existing = by_id.get(n.id)
            if existing is None or n.confidence > existing.confidence:
                by_id[n.id] = n
        merged.nodes = list(by_id.values())
        return merged

    def _normalize(self, paper_id: str, raw: dict) -> ExtractionResult:
        nodes: list[Node] = []
        name_to_id: dict[str, str] = {}
        dropped_nodes = 0
        dropped_edges = 0
        for item in raw.get("nodes", []):
            try:
                ntype = NodeType(item["type"])
            except (KeyError, ValueError):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            confidence = _coerce_confidence(item.get("confidence", 0.5))
            if confidence < self._min_node_confidence:
                dropped_nodes += 1
                continue
            nid = _node_id(ntype, name)
            name_to_id[name.lower()] = nid
            nodes.append(
                Node(
                    id=nid,
                    type=ntype,
                    name=name,
                    aliases=item.get("aliases", []) or [],
                    attrs=_parse_attrs(item.get("attrs")),
                    source_layer=self._source_layer,
                    confidence=confidence,
                    evidence=[item.get("evidence_quote", "")][:1],
                )
            )

        edges: list[Edge] = []
        for item in raw.get("edges", []):
            try:
                etype = EdgeType(item["type"])
            except (KeyError, ValueError):
                continue
            src_name = str(item.get("src_name", "")).strip().lower()
            dst_name = str(item.get("dst_name", "")).strip().lower()
            confidence = _coerce_confidence(item.get("confidence", 0.5))
            if confidence < self._min_edge_confidence:
                dropped_edges += 1
                continue
            src = name_to_id.get(src_name)
            dst = name_to_id.get(dst_name)
            if not src or not dst or src == dst:
                # Endpoints must resolve to nodes we kept — so an edge onto a node the
                # confidence gate dropped is dropped here too, transitively.
                dropped_edges += 1
                continue
            edges.append(
                Edge(
                    src=src,
                    dst=dst,
                    type=etype,
                    attrs=_parse_attrs(item.get("attrs")),
                    source_layer=self._source_layer,
                    confidence=confidence,
                    evidence=[item.get("evidence_quote", "")][:1],
                )
            )

        if dropped_nodes or dropped_edges:
            log.debug(
                "%s: dropped %d nodes (<%.2f) / %d edges (<%.2f)",
                paper_id,
                dropped_nodes,
                self._min_node_confidence,
                dropped_edges,
                self._min_edge_confidence,
            )
        # Attach paper provenance: every extracted node is grounded in this paper.
        for n in nodes:
            n.attrs.setdefault("from_paper", paper_id)
        return ExtractionResult(paper_id=paper_id, nodes=nodes, edges=edges)
