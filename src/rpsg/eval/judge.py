"""LLM-as-judge. Five criteria, 1-5, structured output, versioned prompt.

Trust nothing this emits until `rpsg.eval.calibration` shows quadratic-weighted kappa vs.
your own ratings above threshold. Use a judge model from a different family/tier than any
model you also *generate* answers with, or self-preference inflates the scores.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from rpsg.config import get_settings
from rpsg.eval.gold_schema import GoldRecord
from rpsg.eval.metrics import Answer
from rpsg.llm import get_chat_client
from rpsg.logging import get_logger

log = get_logger(__name__)

#: Sentinel distinguishing "caller said nothing" from "caller explicitly said None".
_UNSET: Any = object()

#: v1, not the newest version. 3.6c set out to lift `attribution` over the 0.6 bar with an
#: anchored rubric and failed: measured against the same 34 hand grades on a deterministic
#: judge, v1 scores +0.45, v3 +0.35 and v2 +0.29 — the original rubric is the best of the
#: three and none of them clears the bar. v2 and v3 stay in the file as recorded negative
#: results and as the control arms `rejudge.py` needs, not as candidates.
DEFAULT_PROMPT_VERSION = "v1"

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
{criteria}
"""

#: Iteration 2's rubric, kept verbatim. It is not dead code: `rejudge.py --prompt-version v1`
#: re-scores the same stored answers with it to measure how much a criterion moves from
#: resampling alone, which is the only thing that makes a v2 gain readable as a gain.
_CRITERIA_V1 = """\
- coverage: addresses every gold facet? 1=one facet, 3=most, 5=all substantively.
- attribution: each claim tied to a specific paper/edge in the retrieved context?
    1=unsourced, 3=mostly sourced, 5=every claim traceable to provided context.
- hedging_accuracy: stated confidence matches evidence strength?
    1=confident where thin OR hedged where strong, 5=calibrated.
- refutation_handling: known contradictions surfaced (not suppressed)?
    1=ignored/one-sided, 3=mentioned not reconciled, 5=both sides reconciled.
- synthesis: combines multiple sources into a relational picture, or just quotes one?
    1=single-source paraphrase, 3=lists several sources without integrating them,
    5=genuine multi-source synthesis.
"""

#: v2 changes the `attribution` entry only; the other four are byte-identical to v1 so that
#: any movement in them is resampling noise rather than a rubric effect.
#:
#: Why the rewrite, from §6 of the Iteration 2 report: kappa=+0.34 failed by range
#: restriction, not bias. Over 34 answers the judge returned 5 zero times and 1 once, while
#: the human distribution was bimodal (11 ones, 8 fives). Reading the graded answers shows
#: why the two never meet -- they are scoring different things:
#:
#:   * v1 asks whether claims are *tied to* the context, which is a presence check. An
#:     answer that appends three paper handles to every compound sentence passes it; the
#:     judge scored two such answers 4 where the human scored 1.
#:   * the human is scoring whether a reader can *verify one specific claim against one
#:     specific source*. Bundled handles fail that, because a bundle does not say which
#:     paper carried which assertion. A repeated single handle after sentences making
#:     several distinct claims fails it for the same reason -- the handle is decoration.
#:   * conversely the human gave 5 to single-source answers that tie each sentence narrowly
#:     to what that one excerpt states, and say plainly where it states nothing. v1 offers
#:     the judge no way to reach 5 there, and `synthesis` already penalises single-sourcing,
#:     so attribution must not penalise it a second time.
#:
#: The anchors below are written to reproduce those patterns, not lifted from the 34 graded
#: answers. Quoting the answers being re-graded would tune the rubric on its own test set.
_CRITERIA_V2 = """\
- coverage: addresses every gold facet? 1=one facet, 3=most, 5=all substantively.
- attribution: can a reader check each individual claim against one named source?
    Judge verifiability, NOT the presence or number of [paper:...] handles. A handle
    that does not identify which source carries which assertion earns no credit, and
    citing one source well beats citing six vaguely.
    1 = handles are decorative: the same handle repeated after sentences that assert
        several different things, or a bundle of 2+ handles after a compound sentence
        so that no individual claim maps to an individual source. Also 1 if any claim
        is attributed to a source the retrieved context shows does not support it.
        An answer can be dense with handles and still score 1.
    3 = the main claims map to specific sources, but some sentences carry bundled or
        blanket handles, or a claim or two floats unsourced.
    5 = every assertion is traceable to a specific source, phrased so the reader knows
        what that source actually says, and the answer states plainly where the context
        supports nothing. A faithful single-source answer belongs here if it meets that
        bar; do not cap it for using one source.
- hedging_accuracy: stated confidence matches evidence strength?
    1=confident where thin OR hedged where strong, 5=calibrated.
- refutation_handling: known contradictions surfaced (not suppressed)?
    1=ignored/one-sided, 3=mentioned not reconciled, 5=both sides reconciled.
- synthesis: combines multiple sources into a relational picture, or just quotes one?
    1=single-source paraphrase, 3=lists several sources without integrating them,
    5=genuine multi-source synthesis.
"""

#: v3 fixes a defect in v2's low anchor, found in the judge's own persisted justifications.
#:
#: v2 made things worse where it was meant to help: the judge returned 4 or 5 zero times
#: across 34 answers (distribution 5/24/5/0/0, mean 2.00 against a human mean of 2.79), and
#: kappa fell to +0.29. Range restriction, the diagnosed failure, tightened rather than
#: opened.
#:
#: The cause is a rule written wrong, not a judge that misread it. v2 sent an answer to 1
#: for "the same handle repeated after sentences that assert several different things" --
#: but a paragraph drawing three related assertions from one excerpt and marking them with
#: that excerpt's handle is *correct* attribution, and it is what nearly every answer in
#: this corpus does. So the low anchor fired on everything. The judge scored the human's
#: cleanest 5 -- a faithful two-sentence single-source answer -- a 2, reasoning that "the
#: second sentence uses the same handle for a separate assertion".
#:
#: v3 therefore reserves 1 for attribution that is *wrong* rather than merely coarse: a
#: handle the evidence does not support, or a bundle mixing papers so that no assertion
#: maps to any of them. Repetition of one correct handle is explicitly fine. v2's ranking
#: gain is worth keeping -- rho rose +0.39 -> +0.58, p<0.001, so the verifiability framing
#: orders answers better than v1's presence check -- so that framing survives and only the
#: band boundaries move.
#:
#: OUTCOME: refuted. v3 recentred the scale -- mean 2.82 against the human 2.79, where v2
#: sat at 2.00 -- but kappa did not follow, and on a temperature-0 judge v3 reaches +0.35
#: against v1's +0.45. The top of the scale never opened: across every version and every
#: sample the judge returned 5 exactly zero times in 34 answers, while the human returned it
#: eight times. So §6's prescription -- "this requires a rubric with anchored examples at
#: both ends" -- is tested and wrong. Three rubrics moved the mean and the ranking and left
#: agreement-on-level where it started. Whatever caps `attribution` is not the wording of
#: the rubric, and the next attempt should not be a fourth rewrite.
_CRITERIA_V3 = """\
- coverage: addresses every gold facet? 1=one facet, 3=most, 5=all substantively.
- attribution: can a reader tell which source each claim came from, and is that source
    right? Judge correctness of the mapping, NOT the number of [paper:...] handles.
    Repeating one handle across several claims drawn from that same source is CORRECT and
    must not be penalised; citing one source faithfully beats citing six vaguely.
    1 = the mapping misleads: a claim carries a handle the retrieved context does not
        support, or handles from several papers are bundled so that no assertion can be
        traced to any particular one, or most claims carry no handle at all.
    3 = the mapping is broadly right but coarse: the reader can tell roughly where the
        material came from, while some assertions sit under a bundle or float unsourced.
    5 = every assertion is traceable to a source that genuinely supports it, and the answer
        says plainly where the retrieved context supports nothing. A single-source answer
        that does this belongs here; `synthesis` already scores multi-source breadth, so do
        not charge for it twice.
- hedging_accuracy: stated confidence matches evidence strength?
    1=confident where thin OR hedged where strong, 5=calibrated.
- refutation_handling: known contradictions surfaced (not suppressed)?
    1=ignored/one-sided, 3=mentioned not reconciled, 5=both sides reconciled.
- synthesis: combines multiple sources into a relational picture, or just quotes one?
    1=single-source paraphrase, 3=lists several sources without integrating them,
    5=genuine multi-source synthesis.
"""

CRITERIA_BLOCKS = {"v1": _CRITERIA_V1, "v2": _CRITERIA_V2, "v3": _CRITERIA_V3}
PROMPT_VERSIONS = tuple(CRITERIA_BLOCKS)

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
    prompt_version: str = DEFAULT_PROMPT_VERSION


class Judge:
    def __init__(
        self,
        model: str | None = None,
        prompt_version: str | None = None,
        temperature: float | None = _UNSET,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.models.judge_model
        # `_UNSET` rather than `None` as the default, because `None` is itself meaningful
        # here -- it means "use the provider default", the setting every pre-3.6c run used.
        self.temperature = (
            settings.models.judge_temperature if temperature is _UNSET else temperature
        )
        self.prompt_version = prompt_version or DEFAULT_PROMPT_VERSION
        if self.prompt_version not in CRITERIA_BLOCKS:
            raise ValueError(
                f"unknown judge prompt version {self.prompt_version!r}; "
                f"have {sorted(CRITERIA_BLOCKS)}"
            )
        self._client = get_chat_client(self.model)

    def score(self, answer: Answer, gold: GoldRecord, evidence: str = "") -> JudgeScore:
        prompt = JUDGE_TEMPLATE.format(
            query=gold.query,
            facets=json.dumps(gold.facets),
            key_claims=json.dumps([kc.model_dump() for kc in gold.key_claims]),
            known_refutations=json.dumps([r.model_dump() for r in gold.known_refutations]),
            answer=answer.text,
            evidence=evidence or "(not provided)",
            criteria=CRITERIA_BLOCKS[self.prompt_version],
        )
        raw = self._client.json(
            system=JUDGE_SYSTEM,
            user=prompt,
            schema=_JUDGE_SCHEMA,
            schema_name="judge_scores",
            max_tokens=4096,
            temperature=self.temperature,
        )
        return JudgeScore(
            qid=answer.qid,
            scores={c: int(raw[c]["score"]) for c in CRITERIA},
            justifications={c: raw[c].get("why", "") for c in CRITERIA},
            prompt_version=self.prompt_version,
        )
