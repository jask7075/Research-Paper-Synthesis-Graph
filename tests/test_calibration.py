"""Judge calibration math — kappa, length-bias detection."""

from __future__ import annotations

from rpsg.eval.calibration import calibrate_criterion, length_bias, quadratic_weighted_kappa


def test_perfect_agreement_kappa_one():
    human = [1, 2, 3, 4, 5, 3, 2]
    judge = list(human)
    assert quadratic_weighted_kappa(human, judge) == 1.0


def test_disagreement_lowers_kappa():
    human = [1, 2, 3, 4, 5]
    judge = [5, 4, 3, 2, 1]  # inverted
    assert quadratic_weighted_kappa(human, judge) < 0.0


def test_calibrate_criterion_trust_threshold():
    human = [1, 2, 3, 4, 5, 4, 3, 2]
    judge = [1, 2, 3, 4, 5, 4, 3, 2]
    cal = calibrate_criterion(human, judge, "coverage", min_kappa=0.6)
    assert cal.trusted
    assert cal.quadratic_kappa == 1.0


def test_length_bias_detected_when_score_tracks_length():
    lengths = [100, 200, 300, 400, 500, 600]
    scores = [1, 2, 3, 4, 5, 5]  # score rises with length
    lb = length_bias(lengths, scores, "coverage", alpha=0.05)
    assert lb.biased
    assert lb.slope > 0


def test_length_bias_absent_when_uncorrelated():
    lengths = [100, 200, 300, 400, 500, 600]
    scores = [3, 3, 3, 3, 3, 3]  # constant
    lb = length_bias(lengths, scores, "coverage", alpha=0.05)
    assert not lb.biased