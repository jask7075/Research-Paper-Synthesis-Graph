"""LLM-as-judge. Five criteria, 1-5, structured output, versioned prompt.

Trust nothing this emits until `rpsg.eval.calibration` shows quadratic-weighted kappa vs.
your own ratings above threshold. Use a judge model from a different family/tier than any
model you also *generate* answers with, or self-preference inflates the scores.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from rpsg.config import get_settings
from rpsg.eval.gold_schema import GoldRecord
from rpsg.eval.metrics import Answer
from rpsg.logging import get_logger

log = get_logger(__name__)

PROMPT_VERSION = "v1"

CRITERIA = ["coverage", "attribution", "hedging_accuracy", "refutation_handling", "synthesis"]

JUDGE_SYSTEM = "You are a strict grader of research-synthesis answers. Output JSON only."

JUDGE_TEMPLATE = """\
Grade the answer against the gold specification. Score each criterion 1-5 (5 is rare).

QUERY: {query}
GOLD FACETS (all must be addressed): {facets}
GOLD KEY CLAIMS + SOURCES: {key_claims}
KNOWN CONTRADICTIONS IN THE EVIDENCE: {known_refutations}

ANSWER UNDER TEST:
{answer}

RETRIEVED CONTEXT THE SYSTEM USED (for attribution checking):
{evidence}

Score 1-5 with a one-sentence justification citing a specific answer span:
- coverage: addresses every gold facet? 1=one facet, 3=most, 5=all substantively.
- attribution: each claim tied to a specific paper/edge in the retrieved context?
    1=unsourced, 3=mostly sourced, 5=every claim traceable to provided context.
- hedging_accuracy: stated confidence matches evidence strength?
    1=confident where thin OR hedged where strong, 5=calibrated.
- refutation_handling: known contradictions surfaced (not suppressed)?
    1=ignored/one-sided, 3=mentioned not reconciled, 5=both sides reconciled.
- synthesis: combines multiple sources into a relational picture, or just quotes one?
    1=single-source paraphrase, 5=genuine multi-source synthesis.
"""

_JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        c: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "why": {"type": "string"},
            },
            "required": ["score", "why"],
        }
        for c in CRITERIA
    },
    "required": CRITERIA,
}


class JudgeScore(BaseModel):
    qid: str
    scores: dict[str, int] = Field(description="criterion -> 1..5")
    justifications: dict[str, str] = Field(default_factory=dict)
    prompt_version: str = PROMPT_VERSION


class Judge:
    def __init__(self, model: str | None = None, temperature: float | None = None) -> None:
        import anthropic

        settings = get_settings()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.models.judge_model
        self._temperature = (
            settings.eval.judge_temperature if temperature is None else temperature
        )

    def score(self, answer: Answer, gold: GoldRecord, evidence: str = "") -> JudgeScore:
        prompt = JUDGE_TEMPLATE.format(
            query=gold.query,
            facets=json.dumps(gold.facets),
            key_claims=json.dumps([kc.model_dump() for kc in gold.key_claims]),
            known_refutations=json.dumps([r.model_dump() for r in gold.known_refutations]),
            answer=answer.text,
            evidence=evidence or "(not provided)",
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        raw = json.loads(text)
        return JudgeScore(
            qid=answer.qid,
            scores={c: int(raw[c]["score"]) for c in CRITERIA},
            justifications={c: raw[c].get("why", "") for c in CRITERIA},
        )
