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
    (
        # `simulation`/`numerical` included because theory and physics papers title
        # their evaluation section "Numerical simulations" rather than "Results",
        # which otherwise falls through to `other` and loses Dataset/evaluated_on
        # routing (see rpsg.extraction.prompts).
        re.compile(
            r"\b(experiment|result|evaluation|ablation|benchmark|simulation|numerical)", re.I
        ),
        "results",
    ),
    (re.compile(r"\b(limitation|threats to validity|future work)", re.I), "limitations"),
    (re.compile(r"\b(discussion|analysis)", re.I), "discussion"),
    (re.compile(r"\b(conclusion|summary|outlook)", re.I), "conclusion"),
    # Data/code availability statements carry the reproducibility payload (repo URLs,
    # dataset access terms). Measured on 20 quant-ph papers these were 12 sections all
    # typed `other`, so they were being asked for Method/Problem/Claim — the wrong
    # question of exactly the right text. Must precede the `appendix` rule.
    (
        re.compile(
            r"\b(data|code|software)\s+availability|\bavailability\s+of\s+(data|code)", re.I
        ),
        "availability",
    ),
    (re.compile(r"\backnowledg", re.I), "acknowledgments"),
    (re.compile(r"\b(appendix|supplement)", re.I), "appendix"),
    (re.compile(r"\b(references|bibliography)", re.I), "references"),
]

#: Sections dropped before chunking AND before extraction. Appendices are deliberately
#: KEPT — that is where hardware/software/reproducibility facts live (extension #4).
#: Acknowledgments are funding boilerplate with nothing extractable, and were costing
#: one API call each (8 across 20 papers).
DROP_SECTION_TYPES = frozenset({"references", "acknowledgments"})

#: Types that already provide a route to `Limitation` in `rpsg.extraction.prompts`.
_LIMITATION_ROUTES = frozenset({"conclusion", "discussion", "limitations"})
#: Trailing matter that is not the paper's conclusion.
_TAIL_TYPES = frozenset({"references", "appendix", "acknowledgments", "availability"})
#: A positionally-inferred conclusion must have real content, not be a stray fragment.
_MIN_CONCLUSION_CHARS = 400


class Section(BaseModel):
    """A parsed paper section (output of `rpsg.ingestion.pdf_parser`)."""

    title: str
    text: str
    section_type: str = "other"


#: A section shorter than this is treated as a split artefact, not a real section.
_FRAGMENT_CHARS = 200
#: Chunks shorter than this are dropped rather than embedded — see `_emit_chunk`.
_MIN_CHUNK_CHARS = 80


def merge_fragment_sections(sections: list[Section]) -> list[Section]:
    """Fold untyped fragments into the preceding section.

    Typography-based heading detection false-positives on wrapped lines — observed
    "titles" include `'bility of the BP'` and `'tum Circuits, Mitigation of Barren
    Plateau'`, which are mid-word fragments of running text. Each one splits a real
    section into several tiny ones, and extraction costs one API call per section:
    measured on 179 papers, 4,401 sections with a median of 20 but a maximum of 151,
    where that single paper was ~3.7% of the whole batch.

    Two guards on what may merge:
      - Only `other` fragments. A short but TYPED section (a 150-char "Limitations")
        carries routing information that folding it away would destroy.
      - Never into a dropped section (references/acknowledgments), since that section's
        text is discarded and the fragment's would go with it.

    Runs BEFORE `refine_section_types`, because that function reasons about first/last
    position and merging changes what is first and last.
    """
    merged: list[Section] = []
    for section in sections:
        is_fragment = (
            section.section_type == "other" and len(section.text.strip()) < _FRAGMENT_CHARS
        )
        if is_fragment and merged and merged[-1].section_type not in DROP_SECTION_TYPES:
            previous = merged[-1]
            previous.text = f"{previous.text}\n{section.text}".strip()
            continue
        merged.append(section)
    return merged


def refine_section_types(sections: list[Section]) -> list[Section]:
    """Second pass: infer a type from document position where the title could not.

    Title keywords cannot type most headings. Measured across 20 quant-ph papers, 69%
    of sections were `other`, and inspecting them showed the majority are genuine
    domain subsections ("The role of measurements", "Variational Dicke state ansatz")
    that no keyword scheme will ever match. Position, though, is informative — and it
    matters for one specific reason: `conclusion`/`discussion`/`limitations` are the
    only types that route `Limitation`, and 7 of those 20 papers had none of the three,
    making the relational core of the thesis unreachable for 35% of the corpus.

    Two rules, both conservative:
      1. A leading `other` section is front matter -> `introduction`.
      2. If nothing routes `Limitation`, the last substantial body section becomes
         `conclusion`. Trailing matter (references, appendix, acknowledgments,
         availability) is skipped, and a short fragment is not eligible.

    Mutates nothing: returns the same list with `section_type` updated in place on the
    Section models, which are local to one parse.
    """
    if not sections:
        return sections

    if sections[0].section_type == "other":
        sections[0].section_type = "introduction"

    if not any(s.section_type in _LIMITATION_ROUTES for s in sections):
        for section in reversed(sections):
            if section.section_type in _TAIL_TYPES:
                continue
            if section.section_type == "other" and len(section.text) >= _MIN_CONCLUSION_CHARS:
                section.section_type = "conclusion"
            # Stop at the first non-tail section either way: if it was already typed
            # (say `results`), the paper simply has no conclusion to find.
            break

    return sections


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


def _emit_chunk(
    chunks: list[Chunk],
    paper_id: str,
    section: Section,
    accumulated: list[tuple[int, int]],
    corpus: str,
    section_index: int = 0,
) -> None:
    """Append one chunk covering the accumulated sentence spans (no-op if empty).

    Extracted to a module-level helper (rather than a nested closure) so it takes its
    inputs as explicit parameters instead of closing over the loop's mutable state.
    """
    if not accumulated:
        return
    start, end = accumulated[0][0], accumulated[-1][1]
    text = section.text[start:end].strip()
    # A chunk this short carries no retrievable content, and it is not merely useless:
    # embedding a 1-3 character string ("A", "1", ",") yields a near-centroid vector that
    # scores moderately against every query, so on a query with no strong match these
    # float into the top-k and crowd out real text. Observed on a 10,117-chunk index:
    # 156 chunks (1.5%) under 20 chars, and a real query returned six of them, leaving
    # the synthesizer with 201 tokens of section labels to answer from.
    if len(text) < _MIN_CHUNK_CHARS:
        return
    chunks.append(
        Chunk(
            # `section_index` is load-bearing, not decoration. `char_start`/`char_end`
            # are offsets into *this section's* text and restart at 0 in every section,
            # while a parsed paper routinely has several sections typed `other` -- so
            # paper + type + offsets is not unique. Measured on an 11,020-chunk index:
            # 43 ids each covered two different chunks of the same paper, always with
            # different text. A colliding id means two spans share an identity, so a
            # store keyed by id returns whichever was written last and retrieval can
            # silently serve the wrong passage.
            id=f"{paper_id}::{section_index}::{section.section_type}::{start}-{end}",
            paper_id=paper_id,
            text=text,
            section_title=section.title,
            section_type=section.section_type,
            char_start=start,
            char_end=end,
            corpus=corpus,
        )
    )


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

    for section_index, section in enumerate(sections):
        if section.section_type in DROP_SECTION_TYPES:
            continue
        spans = _sentence_spans(section.text)
        if not spans:
            continue

        current: list[tuple[int, int]] = []
        current_tokens = 0

        for span in spans:
            span_tokens = approx_tokens(section.text[span[0] : span[1]])

            # A single sentence longer than the target becomes its own chunk.
            if span_tokens >= target_tokens and not current:
                _emit_chunk(chunks, paper_id, section, [span], corpus, section_index)
                current, current_tokens = [], 0
                continue

            if current and current_tokens + span_tokens > target_tokens:
                _emit_chunk(chunks, paper_id, section, current, corpus, section_index)
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

        _emit_chunk(chunks, paper_id, section, current, corpus, section_index)
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