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
        [NodeType.METHOD, NodeType.PROBLEM, NodeType.CLAIM, NodeType.HARDWARE,
         NodeType.REPRO_ARTIFACT],
        [EdgeType.ADDRESSES, EdgeType.REQUIRES, EdgeType.PROVIDES],
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
        [NodeType.METHOD, NodeType.SOFTWARE, NodeType.HARDWARE, NodeType.REPRO_ARTIFACT],
        [EdgeType.BUILDS_ON, EdgeType.ADDRESSES, EdgeType.USES, EdgeType.REQUIRES,
         EdgeType.PROVIDES],
    ),
    "results": (
        [NodeType.DATASET, NodeType.METHOD, NodeType.CLAIM, NodeType.HARDWARE,
         NodeType.REPRO_ARTIFACT],
        [EdgeType.EVALUATED_ON, EdgeType.REQUIRES, EdgeType.PROVIDES],
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
        [NodeType.CLAIM, NodeType.LIMITATION, NodeType.PROBLEM, NodeType.REPRO_ARTIFACT],
        [EdgeType.ADDRESSES, EdgeType.PROVIDES],
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
# `ReproducibilityArtifact` is here because `other` is 58% of all chunks and carries a
# repo or archive URL for 14 papers -- more than any single typed section. GROBID leaves a
# section untyped whenever the heading is unusual, and "Code and data availability" under a
# heading it does not recognise lands here rather than in `availability`.
_DEFAULT_TYPES = (
    [NodeType.METHOD, NodeType.PROBLEM, NodeType.CLAIM, NodeType.LIMITATION,
     NodeType.REPRO_ARTIFACT],
    [EdgeType.ADDRESSES, EdgeType.BUILDS_ON, EdgeType.PROVIDES],
)

_REPRO_HINT = """\
Reproducibility fields to capture in `attrs` when present:
  Hardware: {vendor, gpu_type, gpu_count, quantum_vendor, qubit_count, wall_clock_hours}
  Software: {name, version}
  ReproducibilityArtifact: {code_url, dataset_access(one of open|licensed|irb|on_request|unknown)}
Quantum hardware matters: capture vendor + qubit_count exactly (e.g. "IBM", 127).
A ReproducibilityArtifact is an AVAILABILITY STATEMENT and nothing else. Emit one only when
the text says where code or data can be obtained, and it MUST carry `code_url` or
`dataset_access`. A named library, package or tool the paper merely used is `Software`, not
an artifact -- do not emit a ReproducibilityArtifact for it, and never put {name, version}
on one.

Availability statements are often one sentence in a conclusion, an abstract or an untyped
section rather than under a heading: "our implementation can be found at <url>", "source
code is available at <url>", "data are available from the authors upon reasonable request".
Capture the URL verbatim, including when the PDF has split it across a line break
("https: //github.com/..." -> "https://github.com/..."), and including hosts other than
GitHub. A reference list in the same chunk does not make the statement a reference.
Use dataset_access=on_request for "upon (reasonable) request", open for a public repository
or archive, licensed where terms are stated, irb for ethics-restricted access.
"""


def build_user_prompt(paper_id: str, section_title: str, section_type: str, text: str) -> str:
    node_types, edge_types = _SECTION_TYPES.get(section_type, _DEFAULT_TYPES)
    allowed_nodes = ", ".join(t.value for t in node_types)
    allowed_edges = ", ".join(t.value for t in edge_types) or "(none expected)"
    # Gated on either type, not on `Hardware` alone. The hint carries the
    # `code_url`/`dataset_access` field list, so a section routed to
    # `ReproducibilityArtifact` but not to `Hardware` -- `conclusion`, and the `other`
    # default -- would otherwise be asked for the node type without being told which
    # attributes to fill, which is how `code_url` came back 0-for-15 with the field list
    # sitting one branch away.
    hint = (
        _REPRO_HINT
        if NodeType.HARDWARE in node_types or NodeType.REPRO_ARTIFACT in node_types
        else ""
    )
    return (
        f"PAPER: {paper_id}\n"
        f"SECTION: {section_title}  (type: {section_type})\n"
        f"Extract these node types: {allowed_nodes}\n"
        f"Extract these edge types: {allowed_edges}\n"
        f"{hint}\n"
        f"---\n{text}\n---\n"
        "Return JSON with `nodes` and `edges` only."
    )
