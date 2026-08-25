"""This project's evaluation metric and a reusable scoring function.

Source: competition rules — score is the unweighted mean of the 12
per-finding ROC AUCs (macro ROC-AUC). Confirmed independently by both
reviewed reference notebooks (pilkwang/rsna-knee-baseline-v1,
prvsiyan/rsna-knee-read-the-report-then-the-knee), which also note the
metric is invariant to any strictly increasing transform of a label's
scores — so calibration and fixed thresholds earn nothing here, and
combining models should use a rank-based blend rather than averaging
raw probabilities (see rank_blend()).

This is a Phase 0 module per the project plan: pin the metric formula
before any feature/model comparison work, since every later gate depends
on this number being right.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_roc_auc(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Unweighted mean of per-finding ROC AUC across all finding columns.

    y_true and y_pred must share the same columns (one per finding) and
    the same row index (one per study). Raises if a column's y_true has
    only one class present, since ROC AUC is undefined in that case —
    that should surface as an error, not a silently skipped column.
    """
    if list(y_true.columns) != list(y_pred.columns):
        raise ValueError("y_true and y_pred must have matching columns.")
    per_finding_auc = {
        col: roc_auc_score(y_true[col], y_pred[col])
        for col in y_true.columns
    }
    return float(np.mean(list(per_finding_auc.values())))


def per_finding_roc_auc(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.Series:
    """Per-finding ROC AUC, for diagnosing which findings drag the score."""
    return pd.Series(
        {col: roc_auc_score(y_true[col], y_pred[col]) for col in y_true.columns}
    )


def worse_of_two(baseline_gold: float, candidate_gold: float,
                  baseline_weak: float, candidate_weak: float,
                  gold_tol: float, weak_tol: float) -> dict:
    """A candidate passes only if it doesn't regress beyond noise on
    either of two differently-biased gauges: the 58 gold-labeled
    studies (hand-picked to all carry a positive finding — not
    representative of true prevalence) and a much larger held-out
    weak-labeled set (inherits the labeler's own systematic errors
    instead). Returns a diagnostic dict, not just a bool, so a failure
    can be attributed to the gauge that actually caught it.

    `gold_tol`/`weak_tol` are required, not defaulted: `gold_tol` should
    start around the measured 0.02-0.05 pooled-OOF-macro seed variance
    (Fase 4/5); `weak_tol` needs its own measurement once a real model
    produces repeated-seed OOF runs against the weak-label gauge — no
    such run exists yet, so no default is invented here.

    Graduated 2026-08-25 from notebooks/00v2_measurement_gate.ipynb
    (A0, Part A.3). Source: Reference A's own worse_of_two design;
    stevenleehans (this competition, on why the two gauges are
    differently biased rather than redundant).
    """
    gold_delta = candidate_gold - baseline_gold
    weak_delta = candidate_weak - baseline_weak
    gold_ok = gold_delta >= -gold_tol
    weak_ok = weak_delta >= -weak_tol
    return {
        "passed": gold_ok and weak_ok,
        "gold_delta": gold_delta,
        "weak_delta": weak_delta,
        "gold_ok": gold_ok,
        "weak_ok": weak_ok,
    }


def per_label_gate(baseline_auc: pd.Series, candidate_auc: pd.Series,
                    tol: float = 0.03, min_concordant: int = 7) -> dict:
    """Diagnose whether a macro-AUC change reflects a broad effect
    across findings or a narrow one riding on a single noisy label.

    stevenleehans's encoder-scaling post (this competition): a real
    effect moved ~10/12 labels; a null dressed as a win moved ~5/12 and
    rode on the noisiest label. Macro alone can't tell these apart.
    `tol=0.03` and `min_concordant=7` (a bare majority of 12) are
    starting heuristics, not independently derived — same hedge the
    source material gives its own ~0.03 estimate.

    Graduated 2026-08-25 from notebooks/00v2_measurement_gate.ipynb
    (A0, Part A.4).
    """
    delta = candidate_auc - baseline_auc
    macro_delta = float(delta.mean())
    moved = delta[delta.abs() >= tol]
    concordant = int((np.sign(moved) == np.sign(macro_delta)).sum()) if macro_delta != 0 else 0
    return {
        "macro_delta": macro_delta,
        "n_labels_moved": int(len(moved)),
        "n_concordant": concordant,
        "broad_effect": concordant >= min_concordant,
        "per_label_delta": delta,
    }


def gate_decision(baseline_gold_auc: pd.Series, candidate_gold_auc: pd.Series,
                   baseline_weak_macro: float, candidate_weak_macro: float,
                   gold_tol: float, weak_tol: float,
                   label_tol: float = 0.03, min_concordant: int = 7) -> dict:
    """Combine worse_of_two (both macro gauges) with per_label_gate (gold
    per-label deltas — the 58-row set is the primary interpretable
    signal; the weak per-label breakdown is noisier and not checked
    here) into one overall pass/fail decision.

    Graduated 2026-08-25 from notebooks/00v2_measurement_gate.ipynb
    (A0, Part A.5).
    """
    baseline_gold_macro = float(baseline_gold_auc.mean())
    candidate_gold_macro = float(candidate_gold_auc.mean())

    macro_check = worse_of_two(baseline_gold_macro, candidate_gold_macro,
                                baseline_weak_macro, candidate_weak_macro,
                                gold_tol, weak_tol)
    label_check = per_label_gate(baseline_gold_auc, candidate_gold_auc,
                                  tol=label_tol, min_concordant=min_concordant)

    return {
        "passed": macro_check["passed"] and label_check["broad_effect"],
        "macro_check": macro_check,
        "label_check": label_check,
    }


def rank_blend(predictions: "list[pd.DataFrame]", weights: "list[float] | None" = None) -> pd.DataFrame:
    """Combine several models' predictions by percentile rank, not raw mean.

    Averaging raw probabilities lets whichever model happens to be most
    confident (widest score spread) dominate the blend. Since macro
    ROC-AUC only cares about rank order, blend on each model's own
    percentile rank per finding instead — both reviewed reference
    notebooks use this for their final ensembles.
    """
    raise NotImplementedError("Fill in once at least two models are trained.")
