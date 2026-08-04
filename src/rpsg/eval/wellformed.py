"""Level-1 checks: is this answer obviously broken?

Not "is it good" — that is what the deterministic metrics and the judge are for. These
catch the failures that make a *score* meaningless: an empty answer, an answer that cites
nothing, a citation to a paper that is not in the corpus. They need no gold set, no model,
and no network, so they run in CI on every commit and cost nothing.

The reason to have them separate from the metrics: a metric answers "how good is this",
which presumes the output is an answer at all. `must_cite_recall = 0.0` reads as "the
system missed the key papers" whether the system produced a careful wrong answer or an
empty string. Only one of those is a quality problem; the other is a bug, and it should
be loud rather than averaged into a mean.

Violations are returned, not raised. A 15-query eval run must not abort on query 3 —
the run records what was malformed and finishes, so one bad answer costs one data point
instead of the whole run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from rpsg.eval.metrics import Answer

#: An unresolved citation handle. `VectorRAGSystem._resolve_handles` should have rewritten
#: every `[P1]` to `[paper:<id>]`; one surviving in the final text means resolution failed
#: and the answer is carrying a citation the metrics cannot read.
_DANGLING_HANDLE = re.compile(r"\[P\d+(?:\s*,\s*P\d+)*\]")

#: Answers shorter than this are stubs or refusals rather than attempts. Deliberately low:
#: the check is for "the system gave up", not "the answer is thin".
MIN_ANSWER_CHARS = 80


class Violation(BaseModel):
    """One failed check. `code` is stable and machine-groupable; `detail` is for a human."""

    qid: str
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.qid}: {self.code} — {self.detail}"


def check_answer(
    answer: Answer,
    *,
    corpus_ids: set[str] | None = None,
    min_chars: int = MIN_ANSWER_CHARS,
) -> list[Violation]:
    """Return every well-formedness violation for one answer. Empty list == well-formed.

    `corpus_ids` holds bare Semantic Scholar ids (no `paper:` prefix). Pass None when the
    corpus manifest is unavailable and the hallucinated-citation check will be skipped
    rather than raising — a missing manifest should not stop the other checks running.
    """
    violations: list[Violation] = []

    def fail(code: str, detail: str) -> None:
        violations.append(Violation(qid=answer.qid, code=code, detail=detail))

    text = answer.text.strip()
    if not text:
        fail("empty_text", "answer text is empty or whitespace")
    elif len(text) < min_chars:
        fail("text_too_short", f"{len(text)} chars, under the {min_chars} minimum: {text[:60]!r}")

    if not answer.cited_paper_ids:
        # Attribution failure, not a style preference: an uncited synthesis over a
        # 317-paper corpus cannot be checked by a reader, which is the entire point of
        # the system.
        fail("no_citations", "answer cites no papers")

    dangling = _DANGLING_HANDLE.findall(answer.text)
    if dangling:
        fail(
            "dangling_handle",
            f"unresolved synthesis handle(s) {sorted(set(dangling))} survived into the answer",
        )

    if corpus_ids is not None:
        unknown = sorted(
            {p for p in answer.cited_paper_ids if p.removeprefix("paper:") not in corpus_ids}
        )
        if unknown:
            # The failure that most damages credibility in a research tool: a citation
            # that looks authoritative and points at nothing.
            fail("unknown_paper", f"cited paper(s) not in the corpus: {unknown}")

    return violations


def check_answers(
    answers: list[Answer],
    *,
    corpus_ids: set[str] | None = None,
    min_chars: int = MIN_ANSWER_CHARS,
) -> list[Violation]:
    """Every violation across a run, in answer order."""
    return [
        v
        for a in answers
        for v in check_answer(a, corpus_ids=corpus_ids, min_chars=min_chars)
    ]


def load_corpus_paper_ids(papers_jsonl: Path) -> set[str]:
    """Bare paper ids from the stage-01 corpus manifest. Empty set if it is absent."""
    if not papers_jsonl.exists():
        return set()
    ids: set[str] = set()
    for line in papers_jsonl.read_text().splitlines():
        if line.strip():
            pid = json.loads(line).get("paperId")
            if pid:
                ids.add(pid)
    return ids


def summarize(violations: list[Violation]) -> str:
    """Markdown summary, grouped by code. Says so explicitly when there are none."""
    if not violations:
        return "No well-formedness violations.\n"
    by_code: dict[str, list[Violation]] = {}
    for v in violations:
        by_code.setdefault(v.code, []).append(v)
    lines = [f"**{len(violations)} well-formedness violation(s).**\n"]
    for code, group in sorted(by_code.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- `{code}` × {len(group)}: {', '.join(v.qid for v in group)}")
        lines.append(f"  - e.g. {group[0].detail}")
    return "\n".join(lines) + "\n"
