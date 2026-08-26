"""Unit tests for src/evaluate.py — the metric formula and the A0 gate.

macro_roc_auc/per_finding_roc_auc tests check the metric's plumbing
(macro-averaging, column matching), not model quality — model quality
is what the metric is FOR measuring, per structuring-ml-projects's
step 5. worse_of_two/per_label_gate/gate_decision tests check the gate
logic itself on synthetic scenarios; the real numbers they're built
from (51 leaked scanner fingerprints, 4,399/4,407 studies affected) are
validated in notebooks/00v2_measurement_gate.ipynb (A0), not re-proven
here.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import gate_decision, macro_roc_auc, per_finding_roc_auc, per_label_gate, worse_of_two


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


def test_per_finding_roc_auc_returns_nan_for_a_column_with_only_one_class():
    # "b" is all-positive -- a real possibility on a small validation set
    # (e.g. a rare finding among a fold's ~17 gold studies), not a bug.
    # ROC AUC is mathematically undefined for it.
    y_true = pd.DataFrame({"a": [0, 1], "b": [1, 1]})
    y_pred = pd.DataFrame({"a": [0.1, 0.9], "b": [0.6, 0.7]})
    result = per_finding_roc_auc(y_true, y_pred)
    assert result["a"] == pytest.approx(1.0)
    assert np.isnan(result["b"])


def test_macro_roc_auc_excludes_a_single_class_column_instead_of_going_nan():
    # Found 2026-08-26 running A2 on Kaggle: a fold's 17-gold validation
    # set had a finding with only one class present, and the old
    # implementation let that NaN silently propagate through np.mean into
    # the whole macro score every epoch -- undetected because no existing
    # test covered this case.
    y_true = pd.DataFrame({"a": [0, 1], "b": [1, 1]})
    y_pred = pd.DataFrame({"a": [0.1, 0.9], "b": [0.6, 0.7]})
    assert macro_roc_auc(y_true, y_pred) == pytest.approx(1.0)


def test_worse_of_two_passes_when_both_gauges_improve():
    result = worse_of_two(baseline_gold=0.571, candidate_gold=0.590,
                           baseline_weak=0.540, candidate_weak=0.552,
                           gold_tol=0.03, weak_tol=0.01)
    assert result["passed"] is True
    assert result["gold_ok"] is True
    assert result["weak_ok"] is True


def test_worse_of_two_fails_when_one_gauge_regresses_beyond_tolerance():
    # gold improves a lot, but weak regresses well past its tolerance --
    # exactly the scenario worse_of_two exists to catch.
    result = worse_of_two(baseline_gold=0.571, candidate_gold=0.610,
                           baseline_weak=0.540, candidate_weak=0.520,
                           gold_tol=0.03, weak_tol=0.01)
    assert result["passed"] is False
    assert result["gold_ok"] is True
    assert result["weak_ok"] is False


def test_worse_of_two_passes_within_tolerance_even_if_both_dip_slightly():
    result = worse_of_two(baseline_gold=0.571, candidate_gold=0.565,
                           baseline_weak=0.540, candidate_weak=0.538,
                           gold_tol=0.03, weak_tol=0.01)
    assert result["passed"] is True


def test_per_label_gate_flags_broad_effect_when_most_labels_move_together():
    baseline = pd.Series(0.60, index=[f"finding_{i}" for i in range(12)])
    candidate = baseline.copy()
    candidate[:] += np.array([0.05] * 9 + [0.0] * 3)  # 9/12 move concordantly

    result = per_label_gate(baseline, candidate)

    assert result["n_concordant"] == 9
    assert result["broad_effect"] is True


def test_per_label_gate_flags_narrow_effect_when_one_label_drives_the_macro_move():
    baseline = pd.Series(0.60, index=[f"finding_{i}" for i in range(12)])
    candidate = baseline.copy()
    candidate["finding_0"] += 0.40
    candidate[[f"finding_{i}" for i in range(1, 12)]] += 0.005  # ~flat

    result = per_label_gate(baseline, candidate)

    assert result["macro_delta"] > 0  # macro looks like an improvement...
    assert result["n_concordant"] == 1
    assert result["broad_effect"] is False  # ...but it isn't a broad one


def test_gate_decision_requires_both_macro_and_label_checks_to_pass():
    findings = [f"finding_{i}" for i in range(12)]
    baseline_gold = pd.Series(0.60, index=findings)

    broad_candidate = baseline_gold.copy()
    broad_candidate[:] += 0.04  # all 12 move together

    narrow_candidate = baseline_gold.copy()
    narrow_candidate["finding_0"] += 0.48
    narrow_candidate[findings[1:]] += 0.005

    broad = gate_decision(baseline_gold, broad_candidate,
                           baseline_weak_macro=0.540, candidate_weak_macro=0.552,
                           gold_tol=0.03, weak_tol=0.01)
    assert broad["passed"] is True

    narrow = gate_decision(baseline_gold, narrow_candidate,
                            baseline_weak_macro=0.540, candidate_weak_macro=0.552,
                            gold_tol=0.03, weak_tol=0.01)
    assert narrow["passed"] is False  # macro check alone would have passed this
    assert narrow["macro_check"]["passed"] is True
    assert narrow["label_check"]["broad_effect"] is False
