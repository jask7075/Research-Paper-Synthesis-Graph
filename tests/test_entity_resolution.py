"""Entity resolution, where over-merging is the failure that matters.

Under-merging leaves duplicate nodes, which is visible and recoverable. Over-merging
silently asserts two different things are one, and nothing downstream can detect it — an
earlier unfiltered acronym pass put `Adam` and `Gradient Descent` in a single node. Most
of these tests are therefore about what must *not* merge.
"""

from __future__ import annotations

from rpsg.extraction.entity_resolution import (
    acronym_of,
    ambiguous_acronyms,
    apply_map,
    build_entity_map,
    normalize,
)


def _node(nid: str, name: str, ntype: str = "Method") -> dict:
    return {"id": nid, "name": name, "type": ntype}


# --- normalization -----------------------------------------------------------------

def test_case_and_punctuation_are_noise():
    assert normalize("Quantum-Circuit Learning") == normalize("quantum circuit learning")


def test_parenthetical_content_is_kept_because_it_can_be_the_only_difference():
    """Deleting parentheticals merged "AlphaQubit 2 (RT) complexity" with
    "AlphaQubit 2 (full) complexity" -- two different claims. The gloss-to-expansion link
    is the acronym rule's job, where the ambiguity filter can see it."""
    assert (
        normalize("Variational Quantum Eigensolver (VQE)")
        == "variational quantum eigensolver vqe"
    )
    assert normalize("AlphaQubit 2 (RT) complexity") != normalize("AlphaQubit 2 (full) complexity")


def test_distinguishing_parentheticals_are_not_merged_away():
    nodes = [
        _node("claim:aq-rt", "AlphaQubit 2 (RT) complexity", "Claim"),
        _node("claim:aq-full", "AlphaQubit 2 (full) complexity", "Claim"),
    ]
    mapping, _ = build_entity_map(nodes)
    assert mapping == {}, "these are different claims"


def test_the_merge_log_records_each_id_once():
    """A node id recurs across papers; logging per occurrence inflated the count to 620
    against a 369-entry map."""
    nodes = [_node("m:a", "Thing"), _node("m:b", "thing"), _node("m:b", "thing")]
    _, merges = build_entity_map(nodes)
    assert len({m.from_id for m in merges}) == len(merges)


def test_names_differing_only_in_formatting_merge():
    nodes = [_node("method:vqe-solver", "VQE Solver"), _node("method:vqe--solver", "vqe  solver")]
    mapping, merges = build_entity_map(nodes)
    assert len(merges) == 1
    # Which id wins is the lexicographically smallest, not the first seen; assert they
    # agree rather than pinning a winner, since the tie-break is an implementation detail.
    assert apply_map("method:vqe-solver", mapping) == apply_map("method:vqe--solver", mapping)


def test_the_canonical_id_is_stable_regardless_of_input_order():
    a, b = _node("method:aaa", "Same Thing"), _node("method:bbb", "same thing")
    forward, _ = build_entity_map([a, b])
    reverse, _ = build_entity_map([b, a])
    assert forward == reverse


# --- the ambiguity filter, which is the point --------------------------------------

def test_an_acronym_with_one_expansion_folds_into_it():
    nodes = [
        _node(
            "method:variational-quantum-eigensolver-vqe",
            "Variational Quantum Eigensolver (VQE)",
        ),
        _node("method:vqe", "VQE"),
    ]
    mapping, merges = build_entity_map(nodes)
    assert apply_map("method:vqe", mapping).endswith("eigensolver-vqe")
    assert merges[0].rule == "acronym"


def test_an_acronym_with_two_expansions_merges_nothing():
    """`PQC` appears with 11 expansions in this corpus. Folding a bare `PQC` into any one
    of them is a coin flip, and a wrong merge is unrecoverable."""
    nodes = [
        _node("method:parameterized-quantum-circuit-pqc", "Parameterized Quantum Circuit (PQC)"),
        _node("method:post-quantum-cryptography-pqc", "Post Quantum Cryptography (PQC)"),
        _node("method:pqc", "PQC"),
    ]
    mapping, _ = build_entity_map(nodes)
    assert apply_map("method:pqc", mapping) == "method:pqc", "ambiguous acronym must not merge"


def test_ambiguous_acronyms_are_reportable():
    nodes = [
        _node("m:a", "Parameterized Quantum Circuit (PQC)"),
        _node("m:b", "Post Quantum Cryptography (PQC)"),
        _node("m:c", "Variational Quantum Eigensolver (VQE)"),
    ]
    amb = ambiguous_acronyms(nodes)
    assert set(amb) == {"PQC"}, "VQE has one expansion and is not ambiguous"
    assert len(amb["PQC"]) == 2


def test_ambiguity_is_judged_corpus_wide_not_per_paper():
    """An acronym used consistently in one paper but inconsistently elsewhere is still
    ambiguous. Otherwise the graph's correctness depends on which papers were ingested."""
    nodes = [
        _node("m:a", "Quantum Natural Gradient (QNG)"),
        _node("m:b", "Quasi Newton Gradient (QNG)"),
        _node("m:c", "QNG"),
    ]
    mapping, _ = build_entity_map(nodes)
    assert "m:c" not in mapping


# --- type scoping ------------------------------------------------------------------

def test_identical_names_of_different_types_do_not_merge():
    """`barren plateaus` is both a Problem and, elsewhere, a Claim. Merging them would
    turn a bad match into a type error."""
    nodes = [
        _node("problem:barren-plateaus", "Barren Plateaus", "Problem"),
        _node("claim:barren-plateaus", "barren plateaus", "Claim"),
    ]
    mapping, merges = build_entity_map(nodes)
    assert mapping == {} and merges == []


# --- safety ------------------------------------------------------------------------

def test_unrelated_names_are_left_alone():
    nodes = [_node("method:adam", "Adam"), _node("method:gradient-descent", "Gradient Descent")]
    mapping, _ = build_entity_map(nodes)
    assert mapping == {}, "the exact pair a previous unfiltered pass wrongly merged"


def test_an_unmapped_id_passes_through():
    assert apply_map("method:untouched", {}) == "method:untouched"


def test_merges_are_never_chained():
    """Every target is canonical, so resolving once is enough — a chain would make the
    result depend on iteration order."""
    nodes = [_node("m:a", "Thing"), _node("m:b", "thing"), _node("m:c", "THING")]
    mapping, _ = build_entity_map(nodes)
    for src, dst in mapping.items():
        assert dst not in mapping, f"{src} -> {dst} -> {mapping.get(dst)} is a chain"


def test_empty_names_are_skipped_rather_than_collapsed():
    """Normalizing punctuation-only names yields "", which would merge them all together."""
    nodes = [_node("m:a", "---"), _node("m:b", "???")]
    mapping, _ = build_entity_map(nodes)
    assert mapping == {}


def test_acronym_detection_ignores_lowercase_parentheticals():
    assert acronym_of("Some Method (see appendix)") is None
    assert acronym_of("Quantum Circuit Learning (QCL)") == "QCL"