"""The judge rubric is versioned, and the versions must stay comparable.

§6 of the Iteration 2 report put `attribution` at kappa=+0.34 — untrusted — and diagnosed
range restriction rather than bias: over 34 answers the judge returned 5 zero times and 1
once, against a bimodal human distribution. 3.6c rewrites that one rubric entry.

The comparison only means something if two things hold, and both are pinned here:

  * the other four criteria are byte-identical across versions, so any movement in them is
    resampling noise rather than a rubric effect;
  * v1 survives verbatim, so `rejudge.py --prompt-version v1` measures that noise floor.

Deterministic: no API key, no network. Nothing here asks a model anything.
"""

from __future__ import annotations

import pytest

from rpsg.eval.judge import (
    CRITERIA,
    CRITERIA_BLOCKS,
    DEFAULT_PROMPT_VERSION,
    JUDGE_TEMPLATE,
    PROMPT_VERSIONS,
    Judge,
)

_UNCHANGED = ["coverage", "hedging_accuracy", "refutation_handling", "synthesis"]


def _flat(text: str) -> str:
    """One space between words. The rubric is hand-wrapped at 100 columns, so a phrase the
    model reads as contiguous ("does not support") is split by a newline in the source."""
    return " ".join(text.split())


def _entry(block: str, criterion: str) -> str:
    """The `- criterion: ...` stanza, up to the next top-level `- `."""
    start = block.index(f"- {criterion}:")
    rest = block[start:]
    nxt = [rest.index(f"- {c}:") for c in CRITERIA if f"- {c}:" in rest[1:] and c != criterion]
    return rest[: min(nxt)] if nxt else rest


def test_all_versions_are_available() -> None:
    assert set(PROMPT_VERSIONS) == {"v1", "v2", "v3"}


def test_the_default_is_the_best_measured_version_not_the_newest() -> None:
    """v1 by measurement, not by inertia: on a temperature-0 judge over the same 34 hand
    grades, attribution scores v1 +0.45, v3 +0.35, v2 +0.29. Shipping the newest rubric
    would ship the worse one, and this pins the reason so a later edit has to argue with
    the number rather than assume newer is better."""
    assert DEFAULT_PROMPT_VERSION == "v1"


@pytest.mark.parametrize("version", PROMPT_VERSIONS)
def test_every_criterion_is_described_in_every_version(version: str) -> None:
    block = CRITERIA_BLOCKS[version]
    for c in CRITERIA:
        assert f"- {c}:" in block, f"{version} does not describe {c}"


@pytest.mark.parametrize("criterion", _UNCHANGED)
@pytest.mark.parametrize("version", ["v2", "v3"])
def test_only_attribution_ever_differs_from_v1(version: str, criterion: str) -> None:
    """The control condition. If `coverage` also changed, a cross-version kappa comparison
    could not attribute a shift to the attribution rewrite."""
    assert _entry(CRITERIA_BLOCKS["v1"], criterion) == _entry(CRITERIA_BLOCKS[version], criterion)


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_attribution_did_change(version: str) -> None:
    v1 = _entry(CRITERIA_BLOCKS["v1"], "attribution")
    assert v1 != _entry(CRITERIA_BLOCKS[version], "attribution")


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_anchors_both_ends_of_the_attribution_scale(version: str) -> None:
    """Range restriction was the diagnosis, so both anchors have to be reachable. v1 gave
    the judge no way to award 5 to a faithful single-source answer, which is a grade the
    human used eight times."""
    entry = _entry(CRITERIA_BLOCKS[version], "attribution")
    assert "1 =" in entry and "5 =" in entry
    high = entry[entry.index("5 =") :]
    assert "single-source" in high, "the high anchor must reach faithful single-sourcing"


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_grades_the_mapping_not_the_handle_count(version: str) -> None:
    """The v1 failure: the judge scored 4 where the human scored 1 on answers that bundle
    several handles after one compound sentence. A bundle is presence without traceability."""
    entry = _entry(CRITERIA_BLOCKS[version], "attribution")
    assert "NOT the" in entry and "handles" in entry
    assert "bundle" in entry or "bundled" in entry


def test_v3_does_not_punish_a_correctly_repeated_handle() -> None:
    """v2's defect, and the whole reason v3 exists. v2 sent "the same handle repeated after
    sentences that assert several different things" to 1, which is what a faithful
    single-source paragraph looks like — so nothing could score above 3 and the judge never
    returned 4 or 5 across 34 answers. v3 must say the opposite, in the rubric the model
    reads rather than only in a comment."""
    entry = _flat(_entry(CRITERIA_BLOCKS["v3"], "attribution"))
    assert "is CORRECT and must not be penalised" in entry
    low = entry[entry.index("1 =") : entry.index("3 =")]
    assert "does not support" in low, "the low anchor must key on a wrong mapping, not a coarse one"


def test_template_renders_the_selected_version() -> None:
    filled = JUDGE_TEMPLATE.format(
        query="q", facets="[]", key_claims="[]", known_refutations="[]",
        answer="a", evidence="e", criteria=CRITERIA_BLOCKS["v1"],
    )
    assert "1=unsourced, 3=mostly sourced" in filled
    assert "NOT the presence or number" not in filled


def test_an_unknown_version_is_rejected_before_any_api_call() -> None:
    with pytest.raises(ValueError, match="unknown judge prompt version"):
        Judge(model="gpt-5.4-mini", prompt_version="v9")

def test_the_judge_pins_its_sampling_temperature() -> None:
    """A grader that cannot reproduce its own grade cannot certify anything.

    Nothing in the project set a temperature before 3.6c, so every call ran at the provider
    default of 1.0 — including every kappa in §6. Judging the same 34 answers three times
    with one rubric gave attribution kappas of +0.15/+0.35/+0.39, a spread of 0.25 against a
    0.26 gap to the trust bar. At temperature 0 the same spread is 0.02.
    """
    from rpsg.config import get_settings

    assert get_settings().models.judge_temperature == 0.0


def test_an_explicit_none_temperature_survives_the_default(monkeypatch) -> None:
    """`None` means "use the provider default" and is the setting every pre-3.6c run used,
    so it has to be distinguishable from "the caller passed nothing"— otherwise the old
    behaviour becomes unreproducible."""
    from rpsg.config import get_settings

    # Constructing a `Judge` builds a hosted client, which validates the key eagerly. No
    # request is made; the placeholder only gets the constructor past that check.
    monkeypatch.setattr(get_settings(), "openai_api_key", "placeholder-not-used")
    judge = Judge(model="gpt-5.4-mini", prompt_version="v1", temperature=None)
    assert judge.temperature is None
