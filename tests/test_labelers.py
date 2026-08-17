"""Unit tests for src/labelers.py.

Covers the plumbing (normalize/unwrap/clause-splitting, cue matching,
abstention), not model quality — model quality is what
notebooks/03_labeler_validation.ipynb's macro-ROC-AUC gate is for
(structuring-ml-projects step 5). Two tests are direct regressions for
real bugs caught during that validation: negation priority (effusion
scored below random until fixed) and the bullet-marker unwrap bug
(caught in notebooks/02_eda_reports.ipynb).
"""

import pandas as pd

from src.labelers import _unwrap, label_report, label_reports


def test_label_report_abstains_on_empty_report():
    assert label_report("", "acl_injury") == 0.5


def test_label_report_abstains_on_missing_report():
    assert label_report(None, "acl_injury") == 0.5


def test_label_report_abstains_when_finding_not_mentioned():
    # Only meniscus is discussed; nothing here should trigger the ACL cues.
    text = "Lateral meniscus tear noted. All other structures intact."
    assert label_report(text, "acl_injury") == 0.5


def test_label_report_detects_positive_assertion():
    text = "There is a complete tear of the ACL with retraction."
    assert label_report(text, "acl_injury") == 1.0


def test_label_report_detects_explicit_negation():
    text = "The ACL is intact. No other abnormality."
    assert label_report(text, "acl_injury") == 0.0


def test_label_report_negation_wins_over_pathology_in_same_clause():
    # Regression test: the entity name IS the pathology cue for effusion,
    # so "no effusion" matches both the anatomy cue and the pathology cue
    # in the same clause. An earlier version let pathology win in that
    # case, which scored effusion at ROC-AUC 0.438 (below random) in
    # notebooks/03_labeler_validation.ipynb section D — every negated
    # mention was voted positive. Negation must win.
    assert label_report("No joint effusion is seen.", "effusion") == 0.0
    assert label_report("Small joint effusion is present.", "effusion") == 1.0


def test_label_report_rejoins_hard_wrapped_sentence_across_finding_words():
    # "anterior cruciate ligament" is split by a mid-sentence line break
    # with no closing punctuation before it — exactly the hard-wrap
    # pattern measured in notebooks/02_eda_reports.ipynb section C. If
    # unwrap() didn't rejoin the lines, neither line alone would contain
    # the full "anterior cruciate ligament" anatomy cue.
    text = "Complete tear of the anterior cruciate\nligament with retraction."
    assert label_report(text, "acl_injury") == 1.0


def test_unwrap_does_not_merge_a_new_bulleted_statement():
    # Regression test for the bug found in notebooks/02_eda_reports.ipynb
    # section C: a bullet marker ">" is neither upper- nor lowercase, so
    # `s[:1].isupper()` was always False for a bulleted line, and every
    # bulleted line after a non-punctuation-ending line got merged into
    # it as if it were a wrapped continuation — even when the bulleted
    # line was a new, independent, capitalized statement. The fix strips
    # the bullet before checking capitalization, so this stays unmerged.
    text = "increase signal intensity, likely due to sprain\n> No definite tear of the meniscus"
    assert _unwrap(text) == text


def test_unwrap_still_merges_a_genuine_wrapped_continuation():
    # Sanity check that the bullet-marker fix didn't disable unwrapping
    # itself: a plain (non-bulleted) lowercase continuation still merges.
    text = "Complete tear of the anterior cruciate\nligament with retraction."
    assert _unwrap(text) == "Complete tear of the anterior cruciate ligament with retraction."


def test_label_reports_returns_one_column_per_finding_indexed_by_study():
    reports = pd.DataFrame({
        "StudyInstanceUID": ["s1", "s2"],
        "Report": [
            "There is a complete tear of the ACL.",
            "The ACL is intact. No other abnormality.",
        ],
    })
    result = label_reports(reports, ["acl_injury", "fracture"])
    assert list(result.columns) == ["acl_injury", "fracture"]
    assert list(result.index) == ["s1", "s2"]
    assert result.loc["s1", "acl_injury"] == 1.0
    assert result.loc["s2", "acl_injury"] == 0.0
    assert result.loc["s1", "fracture"] == 0.5  # not mentioned -> abstain
