"""Length-aware score damping in the vector store.

Damping corrects a measured embedding artefact: short chunks embed near the corpus
centroid, so they over-score against every query. These tests pin the shape of the
correction and the exemption that keeps terse-by-nature sections retrievable.
"""

from __future__ import annotations

from rpsg.stores.base import Chunk
from rpsg.stores.vector_store import _DAMPING_EXEMPT_SECTIONS, _length_damping


def _chunk(text: str, section_type: str = "other") -> Chunk:
    return Chunk(
        id=f"p::{section_type}::0-{len(text)}",
        paper_id="p",
        text=text,
        section_title="T",
        section_type=section_type,
        char_start=0,
        char_end=len(text),
    )


def test_damping_is_linear_below_reference_and_flat_above():
    assert _length_damping(400, 800) == 0.5
    assert _length_damping(800, 800) == 1.0
    assert _length_damping(4000, 800) == 1.0  # never rewards length beyond the reference


def test_damping_penalises_shorter_chunks_more():
    assert _length_damping(162, 800) < _length_damping(269, 800) < _length_damping(800, 800)


def test_damping_disabled_by_zero_reference():
    assert _length_damping(10, 0) == 1.0


def test_availability_is_exempt_from_damping():
    """A 269-char availability statement is complete, not truncated.

    Without this exemption, damping removed availability sections from results
    entirely — including for a query asking where code is available — which makes the
    reproducibility layer unreachable by vector search.
    """
    assert "availability" in _DAMPING_EXEMPT_SECTIONS


def test_a_short_stub_loses_to_a_longer_chunk_at_similar_similarity():
    """The property the whole change exists for."""
    stub, body = _chunk("x" * 162), _chunk("x" * 1600)
    stub_score = 0.86 * _length_damping(len(stub.text), 800)
    body_score = 0.83 * _length_damping(len(body.text), 800)
    assert body_score > stub_score
