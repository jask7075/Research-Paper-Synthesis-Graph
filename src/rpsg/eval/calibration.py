"""Judge calibration: is the LLM judge trustworthy?

Protocol:
  1. You grade ~20 held-out answers yourself (1-5, all 5 criteria) BEFORE seeing the judge.
  2. Per criterion, compute quadratic-weighted Cohen's kappa and Spearman rho between your
     scores and the judge's. kappa < min threshold => untrusted on that criterion.
  3. Check the three known judge biases:
       - length: regress judge score on answer length; significant positive slope => the
         judge rewards verbosity, not quality.
       - position: (pairwise A/B only) swap order, measure flip rate.
       - self-preference: use a different model family for judge vs. generator; reported
         here as a reminder, not computed.

Pure functions over already-collected ratings — no API calls, fully testable.
"""

from __future__ import annotations

from pydantic import BaseModel
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


class CriterionCalibration(BaseModel):
    criterion: str
    n: int
    quadratic_kappa: float
    spearman_rho: float
    spearman_p: float
    trusted: bool


class LengthBias(BaseModel):
    criterion: str
    slope: float           # judge-score change per 100 answer-characters
    p_value: float
    biased: bool           # slope significantly > 0 at the configured alpha


def quadratic_weighted_kappa(human: list[int], judge: list[int]) -> float:
    """Cohen's kappa with quadratic weights — the standard for ordinal 1-5 agreement."""
    if len(human) < 2:
        return float("nan")
    return float(cohen_kappa_score(human, judge, weights="quadratic", labels=[1, 2, 3, 4, 5]))


def calibrate_criterion(
    human: list[int], judge: list[int], criterion: str, min_kappa: float
) -> CriterionCalibration:
    rho, p = spearmanr(human, judge)
    kappa = quadratic_weighted_kappa(human, judge)
    return CriterionCalibration(
        criterion=criterion,
        n=len(human),
        quadratic_kappa=kappa,
        spearman_rho=float(rho) if rho == rho else 0.0,  # NaN guard
        spearman_p=float(p) if p == p else 1.0,
        trusted=(kappa == kappa and kappa >= min_kappa),
    )


def length_bias(
    answer_lengths: list[int], judge_scores: list[int], criterion: str, alpha: float
) -> LengthBias:
    """OLS of judge score on answer length (per 100 chars). Positive + significant => the
    judge is length-biased and its scores partly measure verbosity."""
    import numpy as np
    from scipy import stats

    x = np.asarray(answer_lengths, dtype=float) / 100.0
    y = np.asarray(judge_scores, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]):
        return LengthBias(criterion=criterion, slope=0.0, p_value=1.0, biased=False)
    result = stats.linregress(x, y)
    return LengthBias(
        criterion=criterion,
        slope=float(result.slope),
        p_value=float(result.pvalue),
        biased=(result.slope > 0 and result.pvalue < alpha),
    )


class CalibrationReport(BaseModel):
    per_criterion: list[CriterionCalibration]
    length_bias: list[LengthBias]

    def untrusted(self) -> list[str]:
        return [c.criterion for c in self.per_criterion if not c.trusted]

    def summary(self) -> str:
        lines = ["Judge calibration:"]
        for c in self.per_criterion:
            flag = "OK " if c.trusted else "!! "
            lines.append(
                f"  {flag}{c.criterion:20s} kappa={c.quadratic_kappa:+.2f} "
                f"rho={c.spearman_rho:+.2f} (p={c.spearman_p:.3f}, n={c.n})"
            )
        for b in self.length_bias:
            if b.biased:
                lines.append(
                    f"  !! length bias on {b.criterion}: +{b.slope:.2f}/100chars "
                    f"(p={b.p_value:.3f})"
                )
        untrusted = self.untrusted()
        lines.append(
            "  -> all criteria trusted"
            if not untrusted
            else f"  -> UNTRUSTED: {', '.join(untrusted)} (fix prompt or drop)"
        )
        return "\n".join(lines)