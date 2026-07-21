"""Extraction prompts, routed by section type.

Routing rationale (see chunking.py): different node/edge types live in different sections.
Prompting the model to look for `Limitation` in a Results table wastes tokens and lowers
precision. Each section type gets a focused instruction listing only the types plausibly
found there. Tier-A metadata is never extracted here — it comes from Semantic Scholar.
"""

from __future__ import annotations

from rpsg.extraction.schema import EdgeType, NodeType

SYSTEM_PROMPT = """\
You are a precise scientific information-extraction system. You read one section of a
research paper and return typed graph nodes and edges as JSON matching the provided schema.

Rules:
- Extract ONLY what the text states or directly implies. Do not use outside knowledge.
- Every node and edge MUST include a short verbatim `evidence_quote` from the text.
- `confidence` in [0,1] reflects how explicitly the text supports the item. A guessed
  `refutes`/`undercuts` edge should score low, not be omitted — downstream hedges on it.
- Prefer canonical names ("Proximal Policy Optimization" not "our method"); put surface
  variants in `aliases`.
- Do NOT extract Paper/Author/Venue nodes — those come from metadata, not from you.
- Return {"nodes": [], "edges": []} if the section contains nothing extractable.
"""

# Which types each section is asked to produce.
_SECTION_TYPES: dict[str, tuple[list[NodeType], list[EdgeType]]] = {
    "abstract": (
        [NodeType.METHOD, NodeType.PROBLEM, NodeType.CLAIM],
        [EdgeType.ADDRESSES],
    ),
    "introduction": (
        [NodeType.PROBLEM, NodeType.METHOD, NodeType.CLAIM],
        [EdgeType.ADDRESSES, EdgeType.BUILDS_ON],
    ),
    "related_work": (
        [NodeType.METHOD, NodeType.CLAIM],
        [EdgeType.BUILDS_ON, EdgeType.REFUTES, EdgeType.UNDERCUTS],
    ),
    "method": (
        [NodeType.METHOD, NodeType.SOFTWARE],
        [EdgeType.BUILDS_ON, EdgeType.ADDRESSES, EdgeType.USES],
    ),
    "results": (
        [NodeType.DATASET, NodeType.METHOD, NodeType.CLAIM],
        [EdgeType.EVALUATED_ON],
    ),
    "discussion": (
        [NodeType.CLAIM, NodeType.LIMITATION],
        [EdgeType.REFUTES, EdgeType.UNDERCUTS],
    ),
    "limitations": (
        [NodeType.LIMITATION],
        [],
    ),
    "appendix": (  # where reproducibility facts hide (extension #4)
        [NodeType.HARDWARE, NodeType.SOFTWARE, NodeType.REPRO_ARTIFACT, NodeType.DATASET],
        [EdgeType.REQUIRES, EdgeType.USES, EdgeType.PROVIDES, EdgeType.EVALUATED_ON],
    ),
}

# Default for "other"/unclassified sections: the common semantic types, no rare edges.
_DEFAULT_TYPES = (
    [NodeType.METHOD, NodeType.PROBLEM, NodeType.CLAIM],
    [EdgeType.ADDRESSES, EdgeType.BUILDS_ON],
)

_REPRO_HINT = """\
Reproducibility fields to capture in `attrs` when present:
  Hardware: {vendor, gpu_type, gpu_count, quantum_vendor, qubit_count, wall_clock_hours}
  Software: {name, version}
  ReproducibilityArtifact: {code_url, dataset_access(one of open|licensed|irb|unknown)}
Quantum hardware matters: capture vendor + qubit_count exactly (e.g. "IBM", 127).
"""


def build_user_prompt(paper_id: str, section_title: str, section_type: str, text: str) -> str:
    node_types, edge_types = _SECTION_TYPES.get(section_type, _DEFAULT_TYPES)
    allowed_nodes = ", ".join(t.value for t in node_types)
    allowed_edges = ", ".join(t.value for t in edge_types) or "(none expected)"
    hint = _REPRO_HINT if section_type == "appendix" else ""
    return (
        f"PAPER: {paper_id}\n"
        f"SECTION: {section_title}  (type: {section_type})\n"
        f"Extract these node types: {allowed_nodes}\n"
        f"Extract these edge types: {allowed_edges}\n"
        f"{hint}\n"
        f"---\n{text}\n---\n"
        "Return JSON with `nodes` and `edges` only."
    )
