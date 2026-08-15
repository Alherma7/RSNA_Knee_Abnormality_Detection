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


def rank_blend(predictions: "list[pd.DataFrame]", weights: "list[float] | None" = None) -> pd.DataFrame:
    """Combine several models' predictions by percentile rank, not raw mean.

    Averaging raw probabilities lets whichever model happens to be most
    confident (widest score spread) dominate the blend. Since macro
    ROC-AUC only cares about rank order, blend on each model's own
    percentile rank per finding instead — both reviewed reference
    notebooks use this for their final ensembles.
    """
    raise NotImplementedError("Fill in once at least two models are trained.")
