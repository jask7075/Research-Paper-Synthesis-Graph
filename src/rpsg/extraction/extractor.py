"""API-based corpus extractor (the one-time offline batch).

Why API, not a local 8B: extraction reliability IS the system's quality ceiling, and the
base-corpus build is a one-time batch. A frontier-small API model (`claude-haiku-4-5`)
extracts materially better than a local 8-14B for ~$20-45 over 2,000 papers — cheap
insurance on the number everything downstream inherits. The local model is reserved for
the Iteration-2-onward query-time graph-growth loop.

This module extracts per (paper, section) with Anthropic structured outputs, then
normalizes surface names into stable node ids and stitches edges by name. Entity
resolution (canonicalizing "PPO" == "Proximal Policy Optimization" across papers) is a
separate Iteration-2 step; here we id nodes deterministically from their normalized name so
the same string collapses to one node.
"""

from __future__ import annotations

import json
import re

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
from rpsg.ingestion.chunking import Section
from rpsg.logging import get_logger

log = get_logger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


def _node_id(node_type: NodeType, name: str) -> str:
    slug = _SLUG.sub("-", name.strip().lower()).strip("-")[:80]
    return f"{node_type.value.lower()}:{slug}"


class Extractor:
    """Wraps the Anthropic Messages API with structured outputs.

    `source_layer` defaults to CURATED because this is the offline, reviewed batch. The
    query-time loop constructs its own Extractor with `source_layer=STAGED`.
    """

    def __init__(
        self,
        model: str | None = None,
        source_layer: SourceLayer = SourceLayer.CURATED,
        max_tokens: int = 4096,
    ) -> None:
        import anthropic

        settings = get_settings()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.models.extraction_model
        self._source_layer = source_layer
        self._max_tokens = max_tokens

    def _call(self, user_prompt: str) -> dict:
        # Structured outputs guarantee schema-valid JSON in the first text block.
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": EXTRACTION_JSON_SCHEMA}
            },
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        return json.loads(text)

    def extract_section(self, paper_id: str, section: Section) -> ExtractionResult:
        prompt = build_user_prompt(paper_id, section.title, section.section_type, section.text)
        try:
            raw = self._call(prompt)
        except Exception as exc:  # noqa: BLE001 - one bad section must not kill the batch
            log.warning("Extraction failed for %s/%s: %s", paper_id, section.title, exc)
            return ExtractionResult(paper_id=paper_id)
        return self._normalize(paper_id, raw)

    def extract_paper(self, paper_id: str, sections: list[Section]) -> ExtractionResult:
        """Extract across all sections and merge into one result per paper."""
        merged = ExtractionResult(paper_id=paper_id)
        for section in sections:
            part = self.extract_section(paper_id, section)
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
        for item in raw.get("nodes", []):
            try:
                ntype = NodeType(item["type"])
            except (KeyError, ValueError):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            nid = _node_id(ntype, name)
            name_to_id[name.lower()] = nid
            nodes.append(
                Node(
                    id=nid,
                    type=ntype,
                    name=name,
                    aliases=item.get("aliases", []) or [],
                    attrs=item.get("attrs", {}) or {},
                    source_layer=self._source_layer,
                    confidence=float(item.get("confidence", 0.5)),
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
            src = name_to_id.get(src_name)
            dst = name_to_id.get(dst_name)
            if not src or not dst or src == dst:
                continue  # edge endpoints must resolve to nodes we actually extracted
            edges.append(
                Edge(
                    src=src,
                    dst=dst,
                    type=etype,
                    attrs=item.get("attrs", {}) or {},
                    source_layer=self._source_layer,
                    confidence=float(item.get("confidence", 0.5)),
                    evidence=[item.get("evidence_quote", "")][:1],
                )
            )

        # Attach paper provenance: every extracted node is grounded in this paper.
        for n in nodes:
            n.attrs.setdefault("from_paper", paper_id)
        return ExtractionResult(paper_id=paper_id, nodes=nodes, edges=edges)
