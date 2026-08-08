"""Section-aware chunking: the invariants that keep evidence attached to its section."""

from __future__ import annotations

from rpsg.ingestion.chunking import (
    Section,
    approx_tokens,
    chunk_paper,
    chunk_sections,
    classify_section,
)


def test_classify_section_routes_headings():
    assert classify_section("4. Experiments and Results") == "results"
    assert classify_section("Limitations and Threats to Validity") == "limitations"
    assert classify_section("Appendix B: Hardware Details") == "appendix"
    assert classify_section("Something Idiosyncratic") == "other"


def test_chunks_never_cross_section_boundaries():
    sections = [
        Section(title="Method", text="We propose A. " * 60, section_type="method"),
        Section(title="Results", text="A beats B on X. " * 60, section_type="results"),
    ]
    chunks = chunk_sections("p1", sections, target_tokens=50, overlap_tokens=8)
    # Every chunk belongs to exactly one section type.
    assert {c.section_type for c in chunks} <= {"method", "results"}
    for c in chunks:
        assert c.paper_id == "p1"
        # char offsets index into that section's own text
        assert 0 <= c.char_start < c.char_end


def test_references_dropped_appendix_kept():
    sections = [
        Section(title="References", text="[1] Foo. [2] Bar.", section_type="references"),
        Section(title="Appendix A", text="We used 8 A100 GPUs and a 127-qubit IBM device. " * 5,
                section_type="appendix"),
    ]
    chunks = chunk_sections("p2", sections, target_tokens=50, overlap_tokens=8)
    types = {c.section_type for c in chunks}
    assert "references" not in types
    assert "appendix" in types  # reproducibility facts must survive


def test_chunk_paper_emits_both_corpora():
    sections = [Section(title="Method", text="Body text. " * 40, section_type="method")]
    # A realistic abstract length. The previous fixture was 34 characters, which no real
    # Semantic Scholar abstract is (they run to ~1000+), and which now falls below the
    # minimum chunk length.
    abstract = (
        "We study the trainability of parameterized quantum circuits and show that "
        "gradient variance decays exponentially with qubit count for sufficiently "
        "random ansatze, then evaluate two mitigation strategies. "
    )
    chunks = chunk_paper("p3", abstract=abstract, sections=sections)
    corpora = {c.corpus for c in chunks}
    assert "abstract" in corpora
    assert "fulltext" in corpora


def test_degenerate_chunks_are_dropped():
    """Single-character chunks embed to near-centroid vectors and pollute top-k.

    Observed on a 10,117-chunk index: 156 chunks under 20 characters (values like "A",
    "1", ","), and a real query returned six of them, leaving the synthesizer 201 tokens
    of section labels to answer from.
    """
    sections = [
        Section(title="Body", text="A", section_type="other"),
        Section(title="Real", text="Meaningful sentence about quantum circuits. " * 6,
                section_type="method"),
    ]
    chunks = chunk_sections("p4", sections, target_tokens=200, overlap_tokens=16)
    assert all(len(c.text) >= 80 for c in chunks), [c.text for c in chunks]
    assert any(c.section_type == "method" for c in chunks)  # real content survives


def test_approx_tokens_monotonic():
    assert approx_tokens("one two three") < approx_tokens("one two three four five six")

def test_chunk_ids_are_unique_across_same_typed_sections():
    """`char_start`/`char_end` restart at 0 in every section, and a parsed paper routinely
    has several sections typed `other`. Without the section index in the id, two different
    spans of one paper share an identity -- measured at 43 collisions on an 11,020-chunk
    index -- and a store keyed by id serves whichever was written last."""
    from rpsg.ingestion.chunking import Section, chunk_sections

    body = "Alpha beta gamma delta. " * 8
    other = "Zeta eta theta iota. " * 8
    chunks = chunk_sections(
        "p1",
        [
            Section(title="A", section_type="other", text=body),
            Section(title="B", section_type="other", text=other),
        ],
    )
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), f"collision: {ids}"
    assert len({c.text for c in chunks}) == len(chunks)
