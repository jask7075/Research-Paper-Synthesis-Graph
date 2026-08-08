"""Semantic entity merging: embeddings for recall, a model for the decision.

Deterministic string resolution (`entity_resolution`) merges 0.2% of ids on this corpus,
because the extractor produces each paper's own phrasing and independent papers rarely
converge on an identical string. The obvious next step — cluster the name embeddings and
merge above a threshold — does not work here, and the measurement is worth recording:

    nearest-neighbour cosine over 6,193 distinct Method nodes (SPECTER)
        0.95-1.00   2,052      0.85-0.90   1,200      0.70-0.80    10
        0.90-0.95   2,742      0.80-0.85     189      below 0.70    0

Nothing falls below 0.70, so there is no negative class to threshold against. Worse,
similarity does not rank true pairs above false ones:

    0.993  "Gradient-free classical optimization for QAOA parameters"
        vs "Gradient-based classical optimization for QAOA parameters"   <- opposite
    0.976  "Quantum Multi-value Decision Diagram (QMDD)"
        vs "Quantum Multi-valued Decision Diagram (QMDD)"                <- same

An antonym pair outscores a genuine duplicate. Negation is the classic embedding failure
and it is the dominant false-positive mode here, so no cutoff admits QMDD while excluding
gradient-free/gradient-based.

Hence the split. Embeddings do what they are good at — reducing ~19M possible pairs to a
few thousand candidates, where only recall matters. A model then reads both strings and
decides, which is trivial for the pair above and impossible for cosine.

Two deliberate restrictions:

- **Entity types only.** `Claim` and `Limitation` are propositions, not entities; merging
  two claims asserts they are the same statement, which is a different and stronger claim
  than "these name the same method". Out of scope here.
- **Same type only**, as in `entity_resolution` — a bad match should not also be a type
  error.

Verdicts are cached by name pair, so re-running is free and the same corpus yields the
same graph unless the cache is cleared. That recovers most of the determinism this
approach otherwise gives up, but not all of it: a cold cache on a new machine may decide
a borderline pair differently.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from rpsg.llm import get_chat_client
from rpsg.logging import get_logger

log = get_logger(__name__)

#: Merging propositions is a different problem with a different risk profile.
ENTITY_TYPES = frozenset({"Method", "Problem", "Dataset", "Software", "Hardware"})

_SYSTEM = """\
You decide whether two names taken from scientific papers refer to the SAME entity.

Same means a reader would index them under one heading: spelling or formatting variants,
an acronym beside its expansion, or one name adding a harmless qualifier.

Different means anything a researcher would need to tell apart. Be strict about negation
and polarity — "gradient-free" and "gradient-based" are DIFFERENT, as are "with noise" and
"without noise". Two methods that address the same problem are still different methods.

When uncertain, answer different: a wrong merge silently destroys a distinction and cannot
be detected downstream, while a missed merge only leaves a duplicate.
"""

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "same": {"type": "boolean"},
        "reason": {"type": "string", "description": "One short clause."},
    },
    "required": ["same", "reason"],
}


class Verdict(NamedTuple):
    a_id: str
    b_id: str
    a_name: str
    b_name: str
    similarity: float
    same: bool
    reason: str


def candidate_pairs(
    nodes: list[dict],
    embed: object,
    *,
    floor: float = 0.95,
    neighbours: int = 5,
) -> list[tuple[str, str, float]]:
    """(id_a, id_b, similarity) for same-type entity nodes above `floor`.

    `neighbours > 1` because a duplicated method often has several near variants; taking
    only the single nearest would cap recall at one merge per node.
    """
    import numpy as np

    out: list[tuple[str, str, float]] = []
    by_type: dict[str, list[dict]] = {}
    for n in nodes:
        if n["type"] in ENTITY_TYPES:
            by_type.setdefault(n["type"], []).append(n)

    for ntype, group in by_type.items():
        if len(group) < 2:
            continue
        import faiss

        names = [g["name"] for g in group]
        vecs = np.asarray(embed.encode(names), dtype="float32")  # type: ignore[attr-defined]
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        # Pin faiss to one thread immediately before searching, as `FaissVectorStore` does.
        # Three libomp copies live in this environment (faiss, torch, sklearn) and a
        # process that has loaded torch — which `embed.encode` above just did — segfaults
        # on a threaded faiss search. Setting it at import time instead raises OMP Error
        # #15, because it runs before both runtimes exist.
        with contextlib.suppress(Exception):  # a missing symbol must not break the search
            faiss.omp_set_num_threads(1)
        k = min(neighbours + 1, len(group))
        sims, idxs = index.search(vecs, k)
        seen: set[tuple[str, str]] = set()
        for i in range(len(group)):
            for j_pos in range(1, k):
                j, sim = int(idxs[i][j_pos]), float(sims[i][j_pos])
                if sim < floor:
                    continue
                a, b = sorted((group[i]["id"], group[j]["id"]))
                if a != b and (a, b) not in seen:
                    seen.add((a, b))
                    out.append((a, b, sim))
        log.info("%s: %d candidate pairs above %.2f", ntype, len(seen), floor)
    return sorted(out, key=lambda p: -p[2])


def adjudicate(
    pairs: Iterable[tuple[str, str, float]],
    names: dict[str, str],
    *,
    model: str,
    cache_path: Path | None = None,
    workers: int = 8,
) -> list[Verdict]:
    """Ask the model to rule on each pair. Errors resolve to `different`.

    Failing closed matters: a transport error must not become a merge.
    """
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())
    client = get_chat_client(model)
    pairs = list(pairs)

    def key_of(a_id: str, b_id: str) -> str:
        return "␟".join(sorted((names[a_id], names[b_id])))

    def ask(a_id: str, b_id: str) -> tuple[str, dict]:
        key = key_of(a_id, b_id)
        try:
            return key, client.json(
                system=_SYSTEM,
                user=f"A: {names[a_id]}\nB: {names[b_id]}\n\nSame entity?",
                schema=_SCHEMA,
                schema_name="entity_match",
                max_tokens=200,
            )
        except Exception as exc:  # noqa: BLE001 - one bad pair must not stop the batch
            log.warning("adjudication failed for %r / %r: %s", names[a_id], names[b_id], exc)
            return key, {"same": False, "reason": f"error: {exc}"}

    # Concurrent and incrementally checkpointed, for the same reason stage 04 is: the work
    # is one small independent HTTP call per pair, and a serial run over ~3,500 pairs takes
    # about an hour. Writing the cache only at the end also meant a timeout threw away
    # every verdict already paid for.
    todo = [(a, b) for a, b, _ in pairs if key_of(a, b) not in cache]
    if todo:
        log.info("adjudicating %d pairs (%d served from cache)", len(todo), len(pairs) - len(todo))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(ask, a, b) for a, b in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                key, result = fut.result()
                cache[key] = result
                if cache_path and i % 200 == 0:
                    cache_path.write_text(json.dumps(cache, indent=2))
                    log.info("  %d/%d adjudicated", i, len(todo))
    if cache_path:
        cache_path.write_text(json.dumps(cache, indent=2))

    verdicts: list[Verdict] = []
    for a_id, b_id, sim in pairs:
        r = cache[key_of(a_id, b_id)]
        verdicts.append(
            Verdict(
                a_id, b_id, names[a_id], names[b_id], sim, bool(r["same"]), str(r.get("reason", ""))
            )
        )
    return verdicts


def merge_map(verdicts: Iterable[Verdict]) -> dict[str, str]:
    """old_id -> canonical_id from the accepted verdicts.

    Union-find, because merges compose: if A~B and B~C are both accepted, all three must
    land on one id. Without it the result would depend on verdict order. The canonical id
    is the lexicographically smallest in each component, matching `entity_resolution`.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for v in verdicts:
        if not v.same:
            continue
        ra, rb = find(v.a_id), find(v.b_id)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    return {node: find(node) for node in parent if find(node) != node}