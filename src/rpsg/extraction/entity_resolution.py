"""Merge extracted nodes that name the same entity.

Node ids are slugified surface names (`_node_id` in `extractor.py`), so `MLP` and
`Multilayer Perceptron (MLP)` are two nodes. This module decides which ids collapse and
emits the mapping; applying it to the graph is a separate step, so a merge can be audited
before it is trusted.

Two rules, in order of how much they can hurt:

1. **Normalization.** Case and punctuation are noise; parenthetical *content* is not.
   Merging on the normalized form only joins nodes whose visible text already agrees.
   An earlier version deleted parentheticals and merged "AlphaQubit 2 (RT) complexity"
   with "AlphaQubit 2 (full) complexity" -- see `normalize` for why that is the shape of
   mistake this module exists to avoid.

2. **Unambiguous acronym expansion.** `Variational Quantum Eigensolver (VQE)` licenses
   folding a bare `VQE` node into it — but *only* when the corpus expands that acronym one
   way. Of 433 acronyms seen, 103 have more than one expansion; `PQC` alone appears with
   11. Merging on those joins unrelated entities, and a prior unfiltered attempt put `Adam`
   and `Gradient Descent` in a single node. The ambiguity filter is the load-bearing part
   of this rule, not the expansion.

Both rules are type-scoped: a `Method` never merges with a `Problem` however alike the
strings. Cross-type merging has no upside here and turns one bad match into a type error.

What this module cannot do: the extractor emits *phrases*, not entity names — `PQC`
expands to both "computing kl expressibility of parameterized quantum circuits" and
"estimating the expressibility of parameterized quantum circuits", the same method
described twice. No string rule joins those, which is why the measured ceiling is a few
percent. Semantic merging is a separate, riskier tier and is deliberately not here.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from typing import NamedTuple

#: `Full Name (ACRO)` / `Full Name (ACROs)` at the end of a name.
_ACRONYM = re.compile(r"\(([A-Z][A-Za-z0-9\-]{1,9})s?\)\s*$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class Merge(NamedTuple):
    """One id folded into another, with the reason, so the map can be reviewed."""

    from_id: str
    to_id: str
    rule: str  # "normalization" | "acronym"
    detail: str


def normalize(name: str) -> str:
    """Comparison form: lowercased, punctuation flattened, parenthetical content KEPT.

    Deleting parentheticals looks harmless and is not. An earlier version dropped them
    all, on the theory that they are acronym glosses, and merged these:

        "AlphaQubit 2 (RT) complexity"  ==  "AlphaQubit 2 (full) complexity"
        "Eq. (F11) ... subgraph (c)"    ==  "Eq. (F10) ... subgraph (b)"

    In both cases the parenthesis carried the only distinguishing content, so the rule
    asserted two different claims were one. Parentheses are now flattened to spaces like
    any other punctuation, which keeps `(RT)` and `(full)` apart. A trailing acronym gloss
    still links to its expansion, but through the acronym rule, where the ambiguity filter
    can see it — not silently here.
    """
    s = unicodedata.normalize("NFKD", name or "").lower()
    return _NON_ALNUM.sub(" ", s).strip()


def acronym_of(name: str) -> str | None:
    """The trailing acronym gloss, if the name carries one."""
    m = _ACRONYM.search(name or "")
    return m.group(1).upper() if m else None


def _expansions(nodes: Iterable[dict]) -> dict[str, set[str]]:
    """acronym -> the normalized expansions the corpus gives it.

    Corpus-wide rather than per-paper: an acronym ambiguous anywhere is ambiguous
    everywhere, and merging on it in the one paper that happens to be consistent would
    produce a graph whose correctness depends on which papers were ingested.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for n in nodes:
        acro = acronym_of(n["name"])
        if acro:
            # The expansion is the name minus its trailing gloss: "Variational Quantum
            # Eigensolver (VQE)" expands VQE to "variational quantum eigensolver". Keeping
            # the gloss in would make the expansion unmatchable against any other node.
            expansion = normalize(_ACRONYM.sub("", n["name"]))
            if expansion:
                out[acro].add(expansion)
    return dict(out)


def build_entity_map(nodes: Iterable[dict]) -> tuple[dict[str, str], list[Merge]]:
    """Return (old_id -> canonical_id) and the merges that produced it.

    Ids absent from the mapping are unchanged; the caller need not special-case them.
    """
    nodes = list(nodes)
    expansions = _expansions(nodes)
    unambiguous = {a: next(iter(e)) for a, e in expansions.items() if len(e) == 1}

    # canonical id per (type, normalized name): the lexicographically smallest id, so the
    # map is stable across runs regardless of extraction order.
    canonical: dict[tuple[str, str], str] = {}
    degloss: dict[tuple[str, str], str] = {}
    for n in sorted(nodes, key=lambda x: x["id"]):
        key = (n["type"], normalize(n["name"]))
        if key[1]:
            canonical.setdefault(key, n["id"])
        # A second index keyed on the gloss-stripped form, so a bare acronym can find
        # "Variational Quantum Eigensolver (VQE)" without the gloss blocking the match.
        dg = (n["type"], normalize(_ACRONYM.sub("", n["name"])))
        if dg[1]:
            degloss.setdefault(dg, n["id"])

    mapping: dict[str, str] = {}
    merges: list[Merge] = []
    seen: set[str] = set()
    for n in sorted(nodes, key=lambda x: x["id"]):
        norm = normalize(n["name"])
        if not norm:
            continue
        target = canonical.get((n["type"], norm))
        if target and target != n["id"]:
            mapping[n["id"]] = target
            if n["id"] not in seen:
                seen.add(n["id"])
                merges.append(Merge(n["id"], target, "normalization", norm))
            continue
        # A bare acronym node folds into its expansion, when there is exactly one.
        bare = norm.upper()
        if bare in unambiguous:
            expanded = unambiguous[bare]
            target = degloss.get((n["type"], expanded))
            if target and target != n["id"] and n["id"] not in seen:
                mapping[n["id"]] = target
                seen.add(n["id"])
                merges.append(Merge(n["id"], target, "acronym", f"{bare} -> {expanded}"))
    return mapping, merges


def apply_map(node_id: str, mapping: dict[str, str]) -> str:
    """Resolve an id through the map. Single hop: `build_entity_map` always points at a
    canonical id, so chains cannot form."""
    return mapping.get(node_id, node_id)


def ambiguous_acronyms(nodes: Iterable[dict]) -> dict[str, set[str]]:
    """Acronyms with more than one expansion — the ones deliberately not merged."""
    return {a: e for a, e in _expansions(nodes).items() if len(e) > 1}