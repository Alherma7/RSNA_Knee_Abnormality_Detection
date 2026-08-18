"""Unit tests for src/data.py.

Only load_reports(), load_gold_labels(), and load_training_labels() are
graduated so far (2026-08-18, validated against the real train.csv in
notebooks/04b_gold_weak_groupkfold.ipynb — see that notebook and
RESOURCES.md for the real-data numbers these unit tests don't repeat).
load_dicom_series, build_dicom_cache, and load_series_metadata remain
NotImplementedError until Fase 4's ad-hoc DICOM-loading notebook code is
itself graduated (see README.md Next steps) — no tests for those yet.

These tests use small synthetic CSVs (not the real train.csv) so they
stay hermetic and fast; the real-data numbers (4,407 rows, 58 gold, 54
duplicate-template groups, 0 template leakage across folds) are already
validated in the notebook above and don't need re-proving here.
"""

from src.config import FINDINGS, OFFICIAL_LABEL_COLUMNS
from src.data import load_gold_labels, load_reports, load_training_labels
from src.labelers import report_group_key

LABEL_COLS = list(OFFICIAL_LABEL_COLUMNS.values())


def _write_train_csv(tmp_path, rows):
    """rows: list of dicts with StudyInstanceUID, Report, and optionally
    the 12 official label columns (omit them entirely for a weak row)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df.to_csv(tmp_path / "train.csv", index=False)
    return tmp_path


def _gold_row(study_id, report, value=1.0):
    row = {"StudyInstanceUID": study_id, "Report": report}
    row.update({col: value for col in LABEL_COLS})
    return row


def _weak_row(study_id, report):
    return {"StudyInstanceUID": study_id, "Report": report}


def test_load_reports_indexes_by_study_and_keeps_report_column(tmp_path):
    raw_dir = _write_train_csv(tmp_path, [
        _weak_row("s1", "ACL torn."),
        _weak_row("s2", "Normal knee."),
    ])
    reports = load_reports(raw_dir)
    assert list(reports.index) == ["s1", "s2"]
    assert list(reports["Report"]) == ["ACL torn.", "Normal knee."]


def test_load_gold_labels_only_returns_fully_labeled_rows(tmp_path):
    raw_dir = _write_train_csv(tmp_path, [
        _gold_row("gold1", "Complete tear of the ACL.", value=1.0),
        _gold_row("gold2", "Normal knee.", value=0.0),
        _weak_row("weak1", "Some report."),
    ])
    gold = load_gold_labels(raw_dir)
    assert list(gold.index) == ["gold1", "gold2"]
    assert list(gold.columns) == FINDINGS
    assert (gold.loc["gold1"] == 1.0).all()
    assert (gold.loc["gold2"] == 0.0).all()


def test_load_training_labels_keeps_gold_labels_exact_and_labels_weak_rows(tmp_path):
    raw_dir = _write_train_csv(tmp_path, [
        _gold_row("gold1", "There is a complete tear of the ACL.", value=0.0),
        _weak_row("weak1", "There is a complete tear of the ACL."),
        _weak_row("weak2", "The ACL is intact. No other abnormality."),
    ])
    # gold1's official columns are all 0.0 above (an inconsistent fixture on
    # purpose): load_training_labels must trust the official column, not
    # re-derive it from the report text via the labeler.
    combined = load_training_labels(raw_dir, n_folds=2)

    assert set(combined.index) == {"gold1", "weak1", "weak2"}
    assert combined.loc["gold1", "acl_injury"] == 0.0
    assert bool(combined.loc["gold1", "is_gold"]) is True
    assert bool(combined.loc["weak1", "is_gold"]) is False

    # weak1's report text alone would score acl_injury=1.0 via the labeler,
    # confirming it actually ran the labeler rather than leaving it blank.
    assert combined.loc["weak1", "acl_injury"] == 1.0
    assert combined.loc["weak2", "acl_injury"] == 0.0


def test_load_training_labels_never_splits_a_shared_report_template_across_folds(tmp_path):
    shared_text = "Normal knee, no acute abnormality."
    raw_dir = _write_train_csv(tmp_path, [
        _gold_row("gold1", shared_text, value=0.0),
        _weak_row("weak1", shared_text),
        _weak_row("weak2", "Complete tear of the ACL with retraction."),
        _weak_row("weak3", "Complex tear of the medial meniscus."),
    ])
    combined = load_training_labels(raw_dir, n_folds=2)

    assert report_group_key(shared_text) == report_group_key(shared_text)
    assert combined.loc["gold1", "fold"] == combined.loc["weak1", "fold"]
    assert (combined["fold"] >= 0).all()
