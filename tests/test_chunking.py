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
    chunks = chunk_paper("p3", abstract="A short abstract about the method.", sections=sections)
    corpora = {c.corpus for c in chunks}
    assert "abstract" in corpora
    assert "fulltext" in corpora


def test_approx_tokens_monotonic():
    assert approx_tokens("one two three") < approx_tokens("one two three four five six")