"""Evaluation harness — built first, calibrated before the real system exists.

Modules:
    gold_schema   the structured gold-answer record (writable in ~20-30 min each)
    metrics       deterministic scores (must-cite recall, citation precision, refutation)
    judge         LLM-as-judge (the 5 criteria) with structured 1-5 output
    calibration   quadratic-weighted kappa / Spearman / length-bias vs. your own ratings
    runner        run a system over the gold set -> answers.jsonl + scores.jsonl
"""
