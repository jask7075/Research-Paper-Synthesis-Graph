"""Reading repro fields out of extracted nodes, and scoring them.

The rules under test are the ones that decide whether a number describes the extractor or
describes this scorer's own choices: which nodes feed which field, how several candidates
collapse to one verdict, and where leniency stops.
"""

from __future__ import annotations

from rpsg.eval.repro_gold import ReproRecord
from rpsg.eval.repro_scorer import candidates, matches, reported, score_paper, summarize


def hw(name: str, **attrs) -> dict:
    return {"type": "Hardware", "name": name, "attrs": attrs}


# --- where fields come from ----------------------------------------------------------

def test_device_name_reads_the_node_name_because_there_is_no_device_attr():
    assert candidates([hw("Sycamore", vendor="Google")], "device_name") == ["Sycamore"]


def test_vendor_falls_back_from_quantum_vendor_to_vendor():
    nodes = [hw("a", vendor="IBM"), hw("b", quantum_vendor="Google")]
    assert candidates(nodes, "quantum_vendor") == ["Google", "IBM"]


def test_code_url_comes_from_the_artifact_node():
    nodes = [{"type": "ReproducibilityArtifact", "name": "repo",
              "attrs": {"code_url": "https://github.com/x/y"}}]
    assert candidates(nodes, "code_url") == ["https://github.com/x/y"]


def test_the_extractors_placeholder_values_are_not_candidates():
    """The model writes "unknown" or "" rather than omitting a key."""
    nodes = [hw("a", gpu_type="unknown"), hw("b", gpu_type=""), hw("c", gpu_type="A100")]
    assert candidates(nodes, "gpu_type") == ["A100"]


def test_candidates_are_deduped_across_nodes():
    assert candidates([hw("a", vendor="IBM"), hw("b", vendor="IBM")], "quantum_vendor") == ["IBM"]


def test_fields_are_never_pooled_across_sources():
    """91c10ab4 writes `Sycamore` into quantum_vendor. Reading that as a device_name hit
    would hide a real field confusion behind a correct-looking score."""
    nodes = [hw("23-qubit subgraph", quantum_vendor="Sycamore")]
    assert candidates(nodes, "device_name") == ["23-qubit subgraph"]


# --- matching -------------------------------------------------------------------------

def test_a_model_number_still_names_the_same_card():
    assert matches("A100", "NVIDIA A100-SXM-80GB")


def test_a_vendor_still_matches_a_longer_phrase_naming_it():
    assert matches("IBM", "IBM Quantum hardware")


def test_containment_is_token_level_so_digits_do_not_bleed():
    assert not matches(4, "24 qubits")


def test_an_unrelated_name_does_not_match():
    assert not matches("Sycamore", "23-qubit subgraph")


# --- collapsing several candidates ----------------------------------------------------

def test_a_matching_candidate_is_preferred_over_the_first():
    """Several Hardware nodes per paper is the norm -- one run reported four ways. The
    question is whether the extraction contains the fact, not whether it ranks it first;
    no later stage picks between them, so charging the extractor for the ordering would
    blame the wrong component."""
    nodes = [hw("a", gpu_type="V100-32GB"), hw("b", gpu_type="NVIDIA A100-SXM-80GB")]
    assert reported(nodes, "A100", "gpu_type") == "NVIDIA A100-SXM-80GB"


def test_leniency_does_not_extend_to_absence():
    """When gold says not_reported, any candidate is a hallucination -- no amount of
    matching can rescue a value that should not exist."""
    gold = ReproRecord(paper_id="p", quantum_vendor="not_reported")
    assert score_paper(gold, [hw("x", vendor="Google")])["quantum_vendor"] == "hallucinated"


def test_a_matching_candidate_scores_correct_not_wrong():
    """Regression: matching chose the right candidate, then strict equality in
    `score_field` scored it wrong, leaving the whole rule inert."""
    gold = ReproRecord(paper_id="p", gpu_type="A100")
    assert score_paper(gold, [hw("x", gpu_type="NVIDIA A100-SXM-80GB")])["gpu_type"] == "correct"


def test_a_genuinely_different_value_is_still_wrong():
    gold = ReproRecord(paper_id="p", device_name="Sycamore")
    assert score_paper(gold, [hw("Zuchongzhi")])["device_name"] == "wrong"


def test_nothing_extracted_against_a_real_value_is_missed():
    gold = ReproRecord(paper_id="p", qubit_count=66)
    assert score_paper(gold, [])["qubit_count"] == "missed"


def test_nothing_extracted_against_not_reported_is_correct_absence():
    gold = ReproRecord(paper_id="p", gpu_type="not_reported")
    assert score_paper(gold, [])["gpu_type"] == "correct_absence"


def test_an_unauthored_field_is_skipped_not_counted():
    assert score_paper(ReproRecord(paper_id="p"), [hw("x", vendor="IBM")])["quantum_vendor"] == (
        "skipped"
    )


# --- reporting ------------------------------------------------------------------------

def test_summary_separates_correct_silence_from_correct_recall():
    """On this corpus ~70% of gold is not_reported, so an empty system scores ~70%. The
    aggregate alone would read as competence."""
    out = summarize({
        "p1": {"quantum_vendor": "correct_absence", "device_name": "correct_absence",
               "qubit_count": "correct", "gpu_type": "skipped", "gpu_count": "skipped",
               "code_url": "skipped", "dataset_access": "skipped"},
    })
    assert "an empty system would score this much" in out
    assert "1 correct + 2 correct_absence" in out


def test_summary_says_so_when_the_gold_is_entirely_unauthored():
    blank = dict.fromkeys(
        ("quantum_vendor", "device_name", "qubit_count", "gpu_type", "gpu_count",
         "code_url", "dataset_access"), "skipped")
    assert "nothing scored" in summarize({"p1": blank})