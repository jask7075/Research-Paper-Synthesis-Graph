"""Cross-paper contradiction detection: find `refutes` / `undercuts` edges between papers.

Extraction reads one paper at a time, so the only contradictions it can see are the ones a
paper states about itself -- "unlike [12], we find...". That yields 8 `refutes` and 25
`undercuts` across 271 papers, and the consequence is measurable: **1 of 9 refutation gold
queries surfaces its contradiction**, across every retrieval arm. A graph cannot route to a
disagreement it does not encode.

This pass compares claims *between* papers. Embeddings nominate candidates and a model
rules on each, the same two-tier shape as `semantic_merge` -- and for the same reason, that
similarity alone cannot make the call.

The critical difference from entity merging is which direction similarity points. Two names
for one method are near-identical strings, so a high floor is a *precision* filter there.
Here the opposite holds: contradictory claims are highly similar by construction, because
they discuss the same subject and differ only in what they assert about it. "Gradient-free
optimizers avoid barren plateaus" and "gradient-free optimizers do not solve barren
plateaus" are near-duplicates to an embedder. Similarity is therefore the right *recall*
filter and carries no information at all about whether a contradiction exists -- the model
does the entire discriminative job.

Three outcomes rather than two, because the schema already distinguishes them:

    refutes     A and B cannot both be true; a direct empirical or logical conflict
    undercuts   A weakens B's support without contradicting it -- a scope limit, a
                failure mode, a caveat that narrows where B holds
    neither     same topic, compatible claims. The overwhelming majority.

Fails closed to `neither`: a malformed response or an API error must not manufacture a
disagreement that no paper asserts, since a false `refutes` edge is worse than a missing
one -- it would route a refutation query to a contradiction that does not exist.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

from rpsg.llm import get_chat_client
from rpsg.logging import get_logger

log = get_logger(__name__)

#: Proposition-shaped nodes only. A `Method` or `Dataset` cannot contradict anything --
#: it is a thing, not an assertion -- and including them would waste adjudication on pairs
#: with no possible verdict.
CLAIM_TYPES = frozenset({"Claim", "Limitation"})

#: Written every N adjudications. Serial adjudication of ~3,500 pairs once timed out and
#: lost the entire run because the cache was written only at the end.
_CHECKPOINT = 100

PROMPT_VERSION = "v2"

_SYSTEM_V1 = """You judge whether two claims from DIFFERENT research papers conflict.

Answer with JSON only: {"verdict": "refutes"|"undercuts"|"neither", "reason": "<12 words"}

  refutes    the two cannot both be true -- a direct empirical or logical conflict
  undercuts  A does not contradict B but weakens its support: a scope limit, a failure
             mode, a caveat narrowing where B holds
  neither    same topic but compatible, or too vague to judge

Default to "neither". Claims about different systems, different regimes, or different
quantities do NOT conflict merely by sharing vocabulary. Two papers reporting different
numbers for different setups is not a contradiction."""

#: v2 replaces v1's one-line warning with the worked negatives §8.2 prescribed.
#:
#: v1 accepted 3,072 of 16,972 pairs at 32.5% edge precision. The audit found the failure is
#: a single pattern -- and it is the pattern v1's last paragraph already warns against in
#: prose. Twelve of twenty spurious `refutes` are two papers describing their own different
#: scopes. A prose warning did not prevent it; the fix is to show the model what the error
#: looks like, since over-acceptance is the predicted failure mode when similarity carries no
#: information about whether a conflict exists (see the module docstring).
#:
#: The seven categories below are the labelled spurious `refutes` from
#: `eval/gold/contradiction_audit.jsonl`, generalised. Because those twelve pairs are now IN
#: the prompt, v2 CANNOT be scored on the sixty pairs they came from -- that is training data.
#: A fresh stratified sample is required, which is what `--label-with` exists to label.
_SYSTEM_V2 = """You judge whether two claims from DIFFERENT research papers conflict.

Answer with JSON only: {"verdict": "refutes"|"undercuts"|"neither", "reason": "<12 words"}

  refutes    the two cannot both be true -- a direct empirical or logical conflict
  undercuts  A does not contradict B but weakens its support: a scope limit, a failure
             mode, a caveat narrowing where B holds
  neither    same topic but compatible, or too vague to judge

Default to "neither", and prefer it whenever you are reasoning about what the claims imply
rather than what they assert. A conflict has to be on the page.

These are all "neither". Each was previously mislabelled as a conflict:

1. EACH PAPER STATING ITS OWN SCOPE. "General deterministic treatment of per-qubit noise"
   vs "the general-noise analysis is restricted to single-qubit noise". Two papers saying
   what they themselves did. Neither asserts the other is wrong.
2. DIFFERENT MODELLING CHOICES. "2-qubit Pauli error rate set to double the single-qubit
   rate" vs "single-qubit gates are perfect in the chosen noise model". Two chosen setups,
   not a factual disagreement.
3. CLAIMS THAT ACTUALLY AGREE. "Experiments use synthetic, low-dimensional datasets" vs
   "only classical 2-feature data evaluated" -- the same limitation twice. Also "RQCs run in
   m full cycles followed by measurement of all qubits" vs "requires measuring all qubits at
   the final step": a restatement, not a conflict.
4. UNRELATED SYSTEMS SHARING VOCABULARY. "Error rates increase with number of latent bits"
   (a VAE) vs "growing the lattice does not increase component error rates" (a surface
   code). Both mention error rates and size; they are about different things.
5. A PAPER'S OWN CONTRIBUTION VS SOMEONE ELSE'S RESULT. "Insights on the quantum advantage
   of QAOA" vs "refutes the Sycamore quantum advantage claim". One describes what a paper
   offers; the other is a different paper's finding about a different system.
6. ONE CLAIM NAMING A GAP THE OTHER FILLS. "Need readout error mitigation beyond gate
   error" vs "readout crosstalk is ignored". A stated need and a stated limitation are
   compatible.
7. METADATA AND BOILERPLATE. "All authors contributed to the writing of the manuscript" vs
   "SA performed the study and wrote the initial draft". Never a scientific conflict.
   Return "neither" for anything that is not a scientific assertion.

A real `refutes` needs both papers to assert something about the SAME system under the SAME
conditions, such that one being right makes the other wrong. A real `undercuts` needs B's
support to be genuinely weakened by A -- a measured failure of B's method, a demonstrated
scope limit on B's claim -- not merely a caveat sitting next to it."""

SYSTEM_PROMPTS = {"v1": _SYSTEM_V1, "v2": _SYSTEM_V2}


class Contradiction(NamedTuple):
    a_id: str
    b_id: str
    a_text: str
    b_text: str
    a_paper: str | None
    b_paper: str | None
    similarity: float
    verdict: str          # refutes | undercuts | neither
    reason: str

    @property
    def is_edge(self) -> bool:
        return self.verdict in ("refutes", "undercuts")


def _paper_of(node: dict[str, Any]) -> str | None:
    attrs = node.get("attrs") or {}
    if isinstance(attrs, str):
        with contextlib.suppress(json.JSONDecodeError):
            attrs = json.loads(attrs)
    return attrs.get("from_paper") if isinstance(attrs, dict) else None


def is_comparable(node: dict[str, Any]) -> bool:
    """Can this node take part in a contradiction at all?

    Proposition-shaped and actually named. A `Method` or `Dataset` is a thing rather than
    an assertion, so nothing it could be paired with has a possible verdict.
    """
    return node.get("type") in CLAIM_TYPES and bool((node.get("name") or "").strip())


def can_conflict(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Could these two claims be in tension — before reading what they say?

    Different papers, both known. Same-paper pairs are excluded because a paper's internal
    contradictions are what per-paper extraction already emits, so re-deriving them would
    double-count and spend adjudication re-finding known edges. A node with no
    `from_paper` is skipped outright rather than assumed cross-paper, since that assumption
    is exactly what the exclusion exists to prevent.

    Separated from `candidate_pairs` so the rule is testable without faiss, which lives in
    the optional `vector` extras and is absent from a `[dev]` install.
    """
    pa, pb = _paper_of(a), _paper_of(b)
    return a["id"] != b["id"] and pa is not None and pb is not None and pa != pb


def candidate_pairs(
    nodes: list[dict[str, Any]],
    embed: Any,
    *,
    floor: float = 0.90,
    neighbours: int = 5,
) -> list[tuple[str, str, float]]:
    """(id_a, id_b, similarity) for claim pairs from *different* papers above `floor`.

    Eligibility is `is_comparable` and `can_conflict`; this adds only the nearest-neighbour
    search that finds pairs worth asking about.
    """
    claims = [n for n in nodes if is_comparable(n)]
    if len(claims) < 2:
        return []

    # Imported here rather than at the top of the function, and not at module level, because
    # faiss and numpy live in the optional `vector` extras. CI installs `[dev]` only, so an
    # import above the early return makes every caller need faiss -- including the cases
    # that provably never search, which is what broke the test suite on a clean install.
    # `semantic_merge.candidate_pairs` places it the same way for the same reason.
    import faiss
    import numpy as np

    vecs = np.asarray(embed.encode([c["name"] for c in claims]), dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    # Pin faiss to one thread immediately before searching, as `semantic_merge` and
    # `FaissVectorStore` both do. Three libomp copies live in this environment (faiss,
    # torch, sklearn) and a process that has loaded torch -- which `embed.encode` above
    # just did -- segfaults on a threaded faiss search.
    with contextlib.suppress(Exception):
        faiss.omp_set_num_threads(1)

    k = min(neighbours + 1, len(claims))
    sims, idxs = index.search(vecs, k)
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, float]] = []
    for i in range(len(claims)):
        for j_pos in range(1, k):
            j, sim = int(idxs[i][j_pos]), float(sims[i][j_pos])
            if sim < floor or not can_conflict(claims[i], claims[j]):
                continue
            a, b = sorted((claims[i]["id"], claims[j]["id"]))
            if (a, b) not in seen:
                seen.add((a, b))
                out.append((a, b, sim))
    log.info("%d cross-paper claim pairs above %.2f (of %d claims)", len(out), floor, len(claims))
    return sorted(out, key=lambda p: -p[2])


def adjudicate(
    pairs: Iterable[tuple[str, str, float]],
    nodes: dict[str, dict[str, Any]],
    *,
    model: str,
    cache_path: Path | None = None,
    workers: int = 8,
    prompt_version: str = PROMPT_VERSION,
) -> list[Contradiction]:
    """Ask the model to rule on each pair, caching by claim-text pair.

    Cached by *text* rather than node id so the cache survives a re-extraction that
    renumbers nodes -- the same choice `semantic_merge` makes, and the reason its cache
    outlived one full re-extraction.

    The cache key includes `prompt_version`, because it must not. A prompt change makes every
    stored verdict stale, and v1's key did not record which prompt produced it -- so a v2 run
    against the v1 cache would return 16,965 v1 verdicts and report them as v2's. §8.2's
    "the verdict cache means a re-run costs only what changes" was wrong for exactly this
    reason: a prompt edit changes everything.
    """
    if prompt_version not in SYSTEM_PROMPTS:
        raise ValueError(
            f"unknown prompt version {prompt_version!r}; have {sorted(SYSTEM_PROMPTS)}"
        )
    system = SYSTEM_PROMPTS[prompt_version]
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())
    client = get_chat_client(model)
    pairs = list(pairs)

    def key_of(a_id: str, b_id: str) -> str:
        return f"{nodes[a_id]['name']}␟{nodes[b_id]['name']}"

    def ask(a_id: str, b_id: str) -> dict:
        key = key_of(a_id, b_id)
        if key in cache:
            return cache[key]
        a, b = nodes[a_id], nodes[b_id]
        user = (
            f"CLAIM A (paper {_paper_of(a)}): {a['name']}\n"
            f"  evidence: {' '.join(a.get('evidence') or [])[:400]}\n\n"
            f"CLAIM B (paper {_paper_of(b)}): {b['name']}\n"
            f"  evidence: {' '.join(b.get('evidence') or [])[:400]}"
        )
        try:
            raw = client.text(system=system, user=user, max_tokens=200)
            parsed = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            verdict = parsed.get("verdict", "neither")
            if verdict not in ("refutes", "undercuts", "neither"):
                verdict = "neither"
            result = {"verdict": verdict, "reason": str(parsed.get("reason", ""))[:120]}
        except Exception as exc:  # noqa: BLE001 - a failure must not invent a conflict
            log.warning("adjudication failed, defaulting to neither: %s", exc)
            result = {"verdict": "neither", "reason": "adjudication failed"}
        cache[key] = result
        return result

    out: list[Contradiction] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (res, (a_id, b_id, sim)) in enumerate(
            zip(pool.map(lambda p: ask(p[0], p[1]), pairs), pairs, strict=True), 1
        ):
            a, b = nodes[a_id], nodes[b_id]
            out.append(
                Contradiction(a_id, b_id, a["name"], b["name"], _paper_of(a), _paper_of(b),
                              sim, res["verdict"], res.get("reason", ""))
            )
            if cache_path and n % _CHECKPOINT == 0:
                cache_path.write_text(json.dumps(cache, indent=2))
                log.info("  %d/%d adjudicated", n, len(pairs))
    if cache_path:
        cache_path.write_text(json.dumps(cache, indent=2))
    return out


def to_edges(found: Iterable[Contradiction]) -> list[dict[str, Any]]:
    """Accepted verdicts as graph edges, ready for `upsert_edges`.

    Confidence carries the embedding similarity rather than 1.0. The model made a binary
    call and has no calibrated probability to offer, but the pair's similarity is a real
    measured quantity and it is what `typed_graph` orders by when `max_nodes` truncates.
    """
    return [
        {
            "src": c.a_id,
            "dst": c.b_id,
            "type": c.verdict,
            "confidence": round(c.similarity, 3),
            "evidence": [c.reason] if c.reason else [],
        }
        for c in found
        if c.is_edge
    ]


def summarize(found: list[Contradiction]) -> str:
    from collections import Counter

    if not found:
        return "no pairs adjudicated"
    c = Counter(f.verdict for f in found)
    edges = c["refutes"] + c["undercuts"]
    lines = [
        f"{len(found)} pairs adjudicated -> {edges} edges "
        f"({edges / len(found):.1%} accepted)",
        f"  refutes    {c['refutes']}",
        f"  undercuts  {c['undercuts']}",
        f"  neither    {c['neither']}",
    ]
    papers = {p for f in found if f.is_edge for p in (f.a_paper, f.b_paper) if p}
    lines.append(f"  spanning {len(papers)} papers")
    return "\n".join(lines)