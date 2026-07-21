"""Corpus ingestion: metadata fetch (Tier A), PDF -> sections, section-aware chunking."""

from rpsg.ingestion.chunking import Section, chunk_paper, chunk_sections

__all__ = ["Section", "chunk_sections", "chunk_paper"]
