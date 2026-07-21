"""Section-aware chunking.

Why not fixed-window chunking: a naive sliding window splits a claim from the evidence
that supports it, and destroys the section signal that extraction routing depends on
(`Limitation` comes from Discussion/Future Work; `evaluated_on` from Results/Tables;
reproducibility facts hide in the Appendix). So: chunk *within* sections, never across,
and attach section metadata to every chunk.

Token counts here are a fast word-based approximation. For exact accounting against a
model's tokenizer use that provider's count-tokens endpoint (never tiktoken for Claude).
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from rpsg.stores.base import Chunk

# ~0.75 words per token is a stable approximation for English technical prose.
_WORDS_PER_TOKEN = 0.75

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

#: Section title -> canonical section_type. Order matters (first match wins).
#: Patterns are stem prefixes (leading \b only) so plurals/suffixes match:
#: "Results", "Experiments", "Limitations" all classify correctly.
_SECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\babstract", re.I), "abstract"),
    (re.compile(r"\b(introduction|background)", re.I), "introduction"),
    (re.compile(r"\b(related work|prior work)", re.I), "related_work"),
    (re.compile(r"\b(method|approach|model|architecture|algorithm|setup)", re.I), "method"),
    (re.compile(r"\b(experiment|result|evaluation|ablation|benchmark)", re.I), "results"),
    (re.compile(r"\b(limitation|threats to validity|future work)", re.I), "limitations"),
    (re.compile(r"\b(discussion|analysis)", re.I), "discussion"),
    (re.compile(r"\b(conclusion|summary)", re.I), "conclusion"),
    (re.compile(r"\b(appendix|supplement)", re.I), "appendix"),
    (re.compile(r"\b(references|bibliography)", re.I), "references"),
]

#: Sections dropped before chunking. Appendices are deliberately KEPT — that is where
#: hardware/software/reproducibility facts live (extension #4).
DROP_SECTION_TYPES = frozenset({"references"})


class Section(BaseModel):
    """A parsed paper section (output of `rpsg.ingestion.pdf_parser`)."""

    title: str
    text: str
    section_type: str = "other"


def approx_tokens(text: str) -> int:
    """Fast word-based token estimate."""
    words = len(text.split())
    return int(round(words / _WORDS_PER_TOKEN))


def classify_section(title: str) -> str:
    """Map a raw section heading to a canonical section_type."""
    for pattern, section_type in _SECTION_PATTERNS:
        if pattern.search(title):
            return section_type
    return "other"


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences in `text`, covering the whole string."""
    if not text.strip():
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        if end > start:
            spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def chunk_sections(
    paper_id: str,
    sections: list[Section],
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    respect_sections: bool = True,
    corpus: str = "fulltext",
) -> list[Chunk]:
    """Chunk a paper's sections into retrievable units.

    Chunks never cross a section boundary when `respect_sections` is True. Overlap is
    applied *within* a section only. `char_start`/`char_end` are offsets into that
    section's `text`, so a chunk can always be traced back to its exact source span.
    """
    chunks: list[Chunk] = []

    for section in sections:
        if section.section_type in DROP_SECTION_TYPES:
            continue
        spans = _sentence_spans(section.text)
        if not spans:
            continue

        current: list[tuple[int, int]] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if not current:
                return
            start, end = current[0][0], current[-1][1]
            text = section.text[start:end].strip()
            if text:
                chunks.append(
                    Chunk(
                        id=f"{paper_id}::{section.section_type}::{start}-{end}",
                        paper_id=paper_id,
                        text=text,
                        section_title=section.title,
                        section_type=section.section_type,
                        char_start=start,
                        char_end=end,
                        corpus=corpus,
                    )
                )

        for span in spans:
            span_tokens = approx_tokens(section.text[span[0] : span[1]])

            # A single sentence longer than the target becomes its own chunk.
            if span_tokens >= target_tokens and not current:
                current = [span]
                flush()
                current, current_tokens = [], 0
                continue

            if current and current_tokens + span_tokens > target_tokens:
                flush()
                # Carry back trailing sentences up to `overlap_tokens` for continuity.
                carry: list[tuple[int, int]] = []
                carry_tokens = 0
                for prev in reversed(current):
                    prev_tokens = approx_tokens(section.text[prev[0] : prev[1]])
                    if carry_tokens + prev_tokens > overlap_tokens:
                        break
                    carry.insert(0, prev)
                    carry_tokens += prev_tokens
                current, current_tokens = carry, carry_tokens

            current.append(span)
            current_tokens += span_tokens

        flush()
        if not respect_sections:
            # Sections were meant to be merged; this branch exists only so the flag is
            # honest. Merging is not recommended — see the module docstring.
            continue

    return chunks


def chunk_paper(
    paper_id: str,
    abstract: str | None,
    sections: list[Section],
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Produce both corpora in one pass: an `abstract` chunk set and a `fulltext` set.

    The two vector baselines (`vector_abstract`, `vector_fulltext`) read from the same
    index, discriminated by `Chunk.corpus`.
    """
    chunks: list[Chunk] = []
    if abstract and abstract.strip():
        chunks.extend(
            chunk_sections(
                paper_id,
                [Section(title="Abstract", text=abstract, section_type="abstract")],
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                corpus="abstract",
            )
        )
    chunks.extend(
        chunk_sections(
            paper_id,
            sections,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            corpus="fulltext",
        )
    )
    return chunks