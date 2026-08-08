"""The reproducibility scorer, whose whole job is telling apart failures that a plain
accuracy number merges: inventing a value the paper never stated, versus missing one it did.
"""

from __future__ import annotations

import pytest

from rpsg.eval.repro_gold import (
    FIELDS,
    NOT_REPORTED,
    ReproRecord,
    accuracy,
    normalize,
    score_field,
    score_record,
)

# --- the three-state contract ------------------------------------------------------

def test_unestablished_gold_is_skipped_not_scored():
    """`None` means nobody has looked yet. Scoring it either way invents a measurement."""
    assert score_field(None, "IBM") == "skipped"
    assert score_field(None, None) == "skipped"


def test_silence_the_paper_warrants_is_credited():
    assert score_field(NOT_REPORTED, None) == "correct_absence"


def test_inventing_a_value_the_paper_never_stated_is_hallucination():
    assert score_field(NOT_REPORTED, "127") == "hallucinated"


def test_failing_to_find_a_stated_value_is_a_miss():
    assert score_field(127, None) == "missed"


def test_missed_and_hallucinated_are_not_the_same_failure():
    """Opposite errors. A single accuracy number cannot say which a system commits."""
    assert score_field(127, None) != score_field(NOT_REPORTED, 127)


# --- what counts as the system saying nothing --------------------------------------

@pytest.mark.parametrize("silent", ["unknown", "UNKNOWN", " n/a ", "", "none", "not reported"])
def test_extractor_filler_words_count_as_silence(silent: str):
    """The extractor writes "unknown" into fields it cannot answer instead of omitting
    the key — measured on 242 Hardware nodes. Treating that as an answer would score it
    as a wrong value rather than a miss."""
    assert score_field(127, silent) == "missed"
    assert score_field(NOT_REPORTED, silent) == "correct_absence"


# --- normalization -----------------------------------------------------------------

@pytest.mark.parametrize(
    "a,b",
    [("ibmq-mumbai", "ibmq mumbai"), ("ibmq_mumbai", "IBMQ Mumbai"), ("A100", "a100")],
)
def test_the_same_device_written_differently_matches(a: str, b: str):
    assert score_field(a, b) == "correct"


def test_vendor_prefix_variants_are_a_known_limit():
    """`ibmq_mumbai` vs `IBM Mumbai` is the same device, and normalization does not
    catch it: the tokens differ (`ibmq` / `ibm`). Resolving it needs an alias table,
    which is entity resolution rather than string cleanup. Pinned so the limitation is
    visible in the test suite instead of surfacing as an unexplained `wrong`."""
    assert score_field("ibmq_mumbai", "IBM Mumbai") == "wrong"


def test_genuinely_different_values_do_not_match():
    assert score_field("ibmq_mumbai", "ibmq_cairo") == "wrong"
    assert score_field(127, 27) == "wrong"


def test_numbers_match_across_string_and_int():
    """The extractor emits `"gpu_count": "1"` beside `"qubit_count": 27`."""
    assert score_field(1, "1") == "correct"


def test_normalize_collapses_punctuation_and_case():
    assert normalize("IBM-Mumbai") == normalize("ibm mumbai")


# --- record and aggregate ----------------------------------------------------------

def test_score_record_covers_every_field():
    gold = ReproRecord(paper_id="p1", quantum_vendor="IBM", qubit_count=127)
    out = score_record(gold, {"quantum_vendor": "ibm", "qubit_count": "unknown"})
    assert set(out) == set(FIELDS)
    assert out["quantum_vendor"] == "correct"
    assert out["qubit_count"] == "missed"
    assert out["gpu_type"] == "skipped", "unfilled gold must not be scored"


def test_accuracy_counts_a_warranted_silence_as_right():
    assert accuracy(["correct", "correct_absence"]) == 1.0


def test_accuracy_ignores_skipped_fields():
    assert accuracy(["correct", "skipped", "skipped"]) == 1.0


def test_accuracy_is_none_when_nothing_was_scored():
    """Consistent with rpsg.eval.metrics: nothing measured means no number, not 1.0."""
    assert accuracy(["skipped", "skipped"]) is None
    assert accuracy([]) is None


def test_accuracy_penalises_both_failure_directions():
    assert accuracy(["correct", "missed"]) == 0.5
    assert accuracy(["correct", "hallucinated"]) == 0.5


def test_a_record_may_be_partially_filled():
    """Partial gold is the normal state while it is being written; it must stay honest."""
    gold = ReproRecord(paper_id="p1", qubit_count=127)
    out = score_record(gold, {"qubit_count": 127})
    assert accuracy(list(out.values())) == 1.0
    assert sum(1 for o in out.values() if o == "skipped") == len(FIELDS) - 1