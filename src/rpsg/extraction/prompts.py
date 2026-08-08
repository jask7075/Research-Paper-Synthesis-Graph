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
- `name` is an INDEX ENTRY, not a description. For Method, Problem, Dataset, Software and
  Hardware give the shortest noun phrase that identifies the thing: "Variational Quantum
  Eigensolver", never "estimating ground-state energy using a variational eigensolver".
  Two papers describing the same method must produce the SAME `name`, so drop what this
  paper did with it, drop qualifiers, drop leading verbs. Put the paper's own phrasing in
  `aliases`.
- Claim and Limitation are propositions rather than entities: state each as one complete
  sentence. The index-entry rule does not apply to them.
- Do NOT extract Paper/Author/Venue nodes — those come from metadata, not from you.
- Return {"nodes": [], "edges": []} if the section contains nothing extractable.
"""

# Which types each section is asked to produce.
_SECTION_TYPES: dict[str, tuple[list[NodeType], list[EdgeType]]] = {
    # `Hardware` here because the abstract is where a paper names its headline device:
    # "the Google Sycamore superconducting qubit quantum processor ... over 23 qubits" is
    # an abstract sentence, and the repro_gold audit scored that paper 0/3 with the device
    # stated that plainly. Adding Hardware to method/results/availability took the corpus
    # from 12 nodes to 242, and still missed it — the routing has to follow where papers
    # actually state things, not where a reader would expect the detail to live.
    "abstract": (
        [NodeType.METHOD, NodeType.PROBLEM, NodeType.CLAIM, NodeType.HARDWARE],
        [EdgeType.ADDRESSES, EdgeType.REQUIRES],
    ),
    "introduction": (
        [NodeType.PROBLEM, NodeType.METHOD, NodeType.CLAIM],
        [EdgeType.ADDRESSES, EdgeType.BUILDS_ON],
    ),
    "related_work": (
        [NodeType.METHOD, NodeType.CLAIM],
        [EdgeType.BUILDS_ON, EdgeType.REFUTES, EdgeType.UNDERCUTS],
    ),
    # `Hardware` was reachable only from `appendix` (2.9% of chunks), so the corpus
    # produced 12 Hardware nodes across 8 of 270 papers while the large majority of
    # those papers state a qubit count or a named device. Same failure as `Limitation`
    # before the `conclusion` entry existed: a node type the routing never asks for is
    # absent from the graph with no error anywhere. Experimental setup is stated in
    # `method`, and run configuration in `results`, so both must be able to see it.
    "method": (
        [NodeType.METHOD, NodeType.SOFTWARE, NodeType.HARDWARE],
        [EdgeType.BUILDS_ON, EdgeType.ADDRESSES, EdgeType.USES, EdgeType.REQUIRES],
    ),
    "results": (
        [NodeType.DATASET, NodeType.METHOD, NodeType.CLAIM, NodeType.HARDWARE],
        [EdgeType.EVALUATED_ON, EdgeType.REQUIRES],
    ),
    "discussion": (
        [NodeType.CLAIM, NodeType.LIMITATION],
        [EdgeType.REFUTES, EdgeType.UNDERCUTS],
    ),
    "limitations": (
        [NodeType.LIMITATION],
        [],
    ),
    # Most papers have no `Limitations` or `Discussion` heading at all and state
    # their caveats and open problems in the conclusion instead. Without this
    # entry `conclusion` fell through to the default types and `Limitation` was
    # unreachable for such papers — i.e. the relational core of the thesis
    # ("which methods were limited by Y") had no data to draw on.
    "conclusion": (
        [NodeType.CLAIM, NodeType.LIMITATION, NodeType.PROBLEM],
        [EdgeType.ADDRESSES],
    ),
    # "Data/Code availability" statements: short, and almost pure reproducibility
    # payload (repo URLs, dataset access terms). Asking for Method/Problem/Claim here —
    # which the default did — wastes the one place the repro layer is stated plainly.
    "availability": (
        [NodeType.REPRO_ARTIFACT, NodeType.SOFTWARE, NodeType.DATASET, NodeType.HARDWARE],
        [EdgeType.PROVIDES, EdgeType.USES, EdgeType.EVALUATED_ON, EdgeType.REQUIRES],
    ),
    "appendix": (  # where reproducibility facts hide (extension #4)
        [NodeType.HARDWARE, NodeType.SOFTWARE, NodeType.REPRO_ARTIFACT, NodeType.DATASET],
        [EdgeType.REQUIRES, EdgeType.USES, EdgeType.PROVIDES, EdgeType.EVALUATED_ON],
    ),
}

# Default for "other"/unclassified sections: the common semantic types, no rare edges.
_DEFAULT_TYPES = (
    [NodeType.METHOD, NodeType.PROBLEM, NodeType.CLAIM, NodeType.LIMITATION],
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
    hint = _REPRO_HINT if NodeType.HARDWARE in node_types else ""
    return (
        f"PAPER: {paper_id}\n"
        f"SECTION: {section_title}  (type: {section_type})\n"
        f"Extract these node types: {allowed_nodes}\n"
        f"Extract these edge types: {allowed_edges}\n"
        f"{hint}\n"
        f"---\n{text}\n---\n"
        "Return JSON with `nodes` and `edges` only."
    )
