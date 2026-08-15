"""Unit tests for src/evaluate.py — the metric formula itself.

These check the metric's plumbing (macro-averaging, column matching),
not model quality — model quality is what the metric is FOR measuring,
per structuring-ml-projects's step 5.
"""

import pandas as pd
import pytest

from src.evaluate import macro_roc_auc, per_finding_roc_auc


def test_macro_roc_auc_perfect_predictions_scores_one():
    y_true = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    y_pred = pd.DataFrame({"a": [0.0, 0.1, 0.9, 1.0], "b": [0.0, 1.0, 0.0, 1.0]})
    assert macro_roc_auc(y_true, y_pred) == pytest.approx(1.0)


def test_macro_roc_auc_random_predictions_scores_near_half():
    y_true = pd.DataFrame({"a": [0, 1, 0, 1]})
    y_pred = pd.DataFrame({"a": [0.5, 0.5, 0.5, 0.5]})
    # ties at 0.5 for every row: sklearn's roc_auc_score returns 0.5 here.
    assert macro_roc_auc(y_true, y_pred) == pytest.approx(0.5)


def test_macro_roc_auc_averages_across_findings_not_rows():
    # finding "a" perfectly separable, finding "b" perfectly wrong (AUC 0).
    y_true = pd.DataFrame({"a": [0, 1], "b": [0, 1]})
    y_pred = pd.DataFrame({"a": [0.1, 0.9], "b": [0.9, 0.1]})
    assert macro_roc_auc(y_true, y_pred) == pytest.approx(0.5)


def test_macro_roc_auc_raises_on_mismatched_columns():
    y_true = pd.DataFrame({"a": [0, 1]})
    y_pred = pd.DataFrame({"b": [0.1, 0.9]})
    with pytest.raises(ValueError):
        macro_roc_auc(y_true, y_pred)


def test_per_finding_roc_auc_returns_one_score_per_column():
    y_true = pd.DataFrame({"a": [0, 1], "b": [0, 1]})
    y_pred = pd.DataFrame({"a": [0.1, 0.9], "b": [0.9, 0.1]})
    result = per_finding_roc_auc(y_true, y_pred)
    assert set(result.index) == {"a", "b"}
    assert result["a"] == pytest.approx(1.0)
    assert result["b"] == pytest.approx(0.0)
