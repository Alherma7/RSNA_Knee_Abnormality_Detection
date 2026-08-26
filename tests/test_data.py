"""Unit tests for src/data.py.

load_reports(), load_gold_labels(), load_training_labels(),
geometric_slice_order(), load_dicom_series(), build_group_ids(), and
build_scanner_fingerprints() are graduated so far. The first three:
2026-08-18, validated against the real train.csv in
notebooks/04b_gold_weak_groupkfold.ipynb — see that notebook and
RESOURCES.md for the real-data numbers these unit tests don't repeat.
geometric_slice_order/load_dicom_series: 2026-08-25, validated against
the 58 real gold studies in notebooks/01v2_slice_ordering.ipynb (A0b) —
41/58 real series were pure reversals under the old bare-SliceLocation
sort; that real-corpus number isn't re-proven here either, only the
sorting function's own logic on known synthetic inputs.
build_group_ids/build_scanner_fingerprints/load_training_labels'
scanner grouping: 2026-08-25, validated against the real 4,407-study
corpus in notebooks/00v2_measurement_gate.ipynb (A0) — report-only
grouping was measured to leak 51 scanner fingerprints across a fold's
train/val boundary, affecting 4,399/4,407 studies; the combined
report+scanner grouping measured 0 leaks of either kind. That real-
corpus number isn't re-proven here either, only the grouping logic's
own correctness on known synthetic inputs.
load_slot_cache_shard(): 2026-08-26 (A3), validated against the real
published cache on Kaggle in
notebooks/04v2_slot_cache_integration.ipynb — shapes, full 4,407/58-gold
coverage, and a visual check all passed against real data; that
real-corpus validation isn't re-proven here, only the function's own
shape-mismatch guard on known synthetic inputs.

build_dicom_cache and load_series_metadata remain NotImplementedError
until Fase 4's ad-hoc DICOM-loading notebook code is itself graduated
(see README.md Next steps) — no tests for those yet.

These tests use small synthetic CSVs/DICOM files (not the real
train.csv or real DICOM data) so they stay hermetic and fast; the
real-data numbers are already validated in the notebooks above and
don't need re-proving here.
"""

import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.config import FINDINGS, OFFICIAL_LABEL_COLUMNS
from src.data import (
    build_group_ids,
    build_scanner_fingerprints,
    geometric_slice_order,
    load_dicom_series,
    load_gold_labels,
    load_published_labels,
    load_reports,
    load_slot_cache_shard,
    load_training_labels,
)
from src.labelers import report_group_key

LABEL_COLS = list(OFFICIAL_LABEL_COLUMNS.values())


def _write_dicom(path, **tags):
    """Write a minimal but real, readable DICOM file with only the given tags."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.save_as(path, enforce_file_format=True)
    return path


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


def _write_published_labels_csv(raw_dir, rows):
    """rows: list of dicts with StudyInstanceUID + the 12 official
    label-column names (OFFICIAL_LABEL_COLUMNS values), continuous
    floats -- matches the real llm_labels_v4_blend.csv layout."""
    published_dir = raw_dir / "_published_labels"
    published_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(published_dir / "llm_labels_v4_blend.csv", index=False)


def test_load_published_labels_maps_columns_and_indexes_by_study(tmp_path):
    raw_dir = _write_train_csv(tmp_path, [
        _weak_row("s1", "report 1"),
        _weak_row("s2", "report 2"),
    ])
    _write_published_labels_csv(raw_dir, [
        {"StudyInstanceUID": "s1", **{c: 0.9 for c in LABEL_COLS}},
        {"StudyInstanceUID": "s2", **{c: 0.1 for c in LABEL_COLS}},
    ])

    published = load_published_labels(raw_dir)

    assert list(published.index) == ["s1", "s2"]
    assert list(published.columns) == FINDINGS
    assert published.loc["s1", "acl_injury"] == 0.9
    assert published.loc["s2", "acl_injury"] == 0.1


def test_load_published_labels_raises_when_a_study_is_missing(tmp_path):
    raw_dir = _write_train_csv(tmp_path, [
        _weak_row("s1", "report 1"),
        _weak_row("s2", "report 2"),
    ])
    _write_published_labels_csv(raw_dir, [
        {"StudyInstanceUID": "s1", **{c: 0.5 for c in LABEL_COLS}},
    ])

    with pytest.raises(ValueError):
        load_published_labels(raw_dir)


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
    no_scanner_data = pd.Series(
        [None, None, None], index=["gold1", "weak1", "weak2"]
    )
    combined = load_training_labels(raw_dir, no_scanner_data, n_folds=2)

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
    no_scanner_data = pd.Series(
        [None, None, None, None], index=["gold1", "weak1", "weak2", "weak3"]
    )
    combined = load_training_labels(raw_dir, no_scanner_data, n_folds=2)

    assert report_group_key(shared_text) == report_group_key(shared_text)
    assert combined.loc["gold1", "fold"] == combined.loc["weak1", "fold"]
    assert (combined["fold"] >= 0).all()


def test_load_training_labels_never_splits_a_shared_scanner_fingerprint_across_folds(tmp_path):
    # weak2 and weak3 share nothing in their report text, but DO share a
    # scanner fingerprint -- without scanner grouping, GroupKFold would be
    # free to split them (this is exactly the 4,399/4,407-study leak
    # measured in notebooks/00v2_measurement_gate.ipynb).
    raw_dir = _write_train_csv(tmp_path, [
        _gold_row("gold1", "Normal knee.", value=0.0),
        _weak_row("weak1", "Mild degenerative change."),
        _weak_row("weak2", "Complete tear of the ACL with retraction."),
        _weak_row("weak3", "Complex tear of the medial meniscus."),
    ])
    shared_fp = ("SiemensCo", "Avanto", "General Hospital", "SN1", "1.5", "MRI1")
    scanner_data = pd.Series(
        {"gold1": None, "weak1": None, "weak2": shared_fp, "weak3": shared_fp}
    )
    combined = load_training_labels(raw_dir, scanner_data, n_folds=2)

    assert combined.loc["weak2", "fold"] == combined.loc["weak3", "fold"]
    assert (combined["fold"] >= 0).all()


def test_build_group_ids_unions_rows_sharing_any_key_transitively(tmp_path):
    # a-b share key1, b-c share key2 -- a and c must land in the same
    # group even though they share neither key directly.
    key1 = pd.Series({"a": "x", "b": "x", "c": None})
    key2 = pd.Series({"a": None, "b": "y", "c": "y"})

    groups = build_group_ids(key1, key2)

    assert groups["a"] == groups["b"] == groups["c"]


def test_build_group_ids_keeps_rows_separate_when_no_key_is_shared(tmp_path):
    key1 = pd.Series({"a": "x", "b": "z"})
    key2 = pd.Series({"a": None, "b": None})

    groups = build_group_ids(key1, key2)

    assert groups["a"] != groups["b"]


def test_build_group_ids_never_unions_on_missing_values(tmp_path):
    # both rows have a missing key1 -- that must NOT union them, since
    # NaN/None is "unknown", not "the same group".
    key1 = pd.Series({"a": None, "b": None})

    groups = build_group_ids(key1)

    assert groups["a"] != groups["b"]


def test_build_scanner_fingerprints_groups_studies_sharing_all_tags(tmp_path):
    raw_dir = tmp_path
    series_csv = pd.DataFrame([
        {"StudyInstanceUID": "study1", "SeriesInstanceUID": "series1"},
        {"StudyInstanceUID": "study2", "SeriesInstanceUID": "series2"},
    ])
    series_csv.to_csv(raw_dir / "train_series.csv", index=False)

    for study, series in [("study1", "series1"), ("study2", "series2")]:
        series_dir = raw_dir / "train_series" / study / series
        series_dir.mkdir(parents=True)
        _write_dicom(series_dir / "a.dcm", Manufacturer="SiemensCo",
                     InstitutionName="General Hospital")

    fingerprints = build_scanner_fingerprints(raw_dir, split="train")

    assert fingerprints["study1"] == fingerprints["study2"]


def test_build_scanner_fingerprints_uses_first_series_per_study(tmp_path):
    raw_dir = tmp_path
    series_csv = pd.DataFrame([
        {"StudyInstanceUID": "study1", "SeriesInstanceUID": "series_a"},
        {"StudyInstanceUID": "study1", "SeriesInstanceUID": "series_b"},
    ])
    series_csv.to_csv(raw_dir / "train_series.csv", index=False)

    series_dir = raw_dir / "train_series" / "study1" / "series_a"
    series_dir.mkdir(parents=True)
    _write_dicom(series_dir / "a.dcm", Manufacturer="SiemensCo")

    fingerprints = build_scanner_fingerprints(raw_dir, split="train")

    assert len(fingerprints) == 1
    assert fingerprints["study1"][0] == "SiemensCo"


# Standard axial-like orientation: row=[1,0,0], col=[0,1,0] -> normal=[0,0,1],
# so the geometric key reduces to the z-component of ImagePositionPatient.
_AXIAL_IOP = [1, 0, 0, 0, 1, 0]


def test_geometric_slice_order_sorts_by_ipp_projected_onto_iop_normal(tmp_path):
    high = _write_dicom(tmp_path / "high.dcm", ImageOrientationPatient=_AXIAL_IOP,
                         ImagePositionPatient=[0, 0, 30.0])
    low = _write_dicom(tmp_path / "low.dcm", ImageOrientationPatient=_AXIAL_IOP,
                        ImagePositionPatient=[0, 0, 10.0])
    mid = _write_dicom(tmp_path / "mid.dcm", ImageOrientationPatient=_AXIAL_IOP,
                        ImagePositionPatient=[0, 0, 20.0])

    ordered = geometric_slice_order([high, low, mid])

    assert ordered == [low, mid, high]


def test_geometric_slice_order_overrides_a_reversed_slice_location(tmp_path):
    # SliceLocation deliberately encodes the opposite direction from the real
    # geometric position -- exactly the 71%-of-58-studies scenario measured
    # in notebooks/01v2_slice_ordering.ipynb. The geometric key must win.
    a = _write_dicom(tmp_path / "a.dcm", ImageOrientationPatient=_AXIAL_IOP,
                      ImagePositionPatient=[0, 0, 10.0], SliceLocation=90.0)
    b = _write_dicom(tmp_path / "b.dcm", ImageOrientationPatient=_AXIAL_IOP,
                      ImagePositionPatient=[0, 0, 20.0], SliceLocation=80.0)

    ordered = geometric_slice_order([a, b])

    assert ordered == [a, b]


def test_geometric_slice_order_falls_back_to_slice_location_when_geometric_tags_missing(tmp_path):
    high = _write_dicom(tmp_path / "high.dcm", SliceLocation=30.0)
    low = _write_dicom(tmp_path / "low.dcm", SliceLocation=10.0)

    ordered = geometric_slice_order([high, low])

    assert ordered == [low, high]


def test_geometric_slice_order_falls_back_to_instance_number_when_slice_location_missing(tmp_path):
    high = _write_dicom(tmp_path / "high.dcm", InstanceNumber=3)
    low = _write_dicom(tmp_path / "low.dcm", InstanceNumber=1)

    ordered = geometric_slice_order([high, low])

    assert ordered == [low, high]


def test_geometric_slice_order_falls_back_to_input_order_when_all_tags_missing(tmp_path):
    first = _write_dicom(tmp_path / "first.dcm")
    second = _write_dicom(tmp_path / "second.dcm")

    ordered = geometric_slice_order([first, second])

    assert ordered == [first, second]


def test_geometric_slice_order_falls_back_when_orientation_is_degenerate(tmp_path):
    # row_dir parallel to col_dir -> zero-norm cross product -> not usable
    # as a geometric key, must fall back to SliceLocation instead.
    degenerate_iop = [1, 0, 0, 1, 0, 0]
    high = _write_dicom(tmp_path / "high.dcm", ImageOrientationPatient=degenerate_iop,
                         ImagePositionPatient=[0, 0, 99.0], SliceLocation=30.0)
    low = _write_dicom(tmp_path / "low.dcm", ImageOrientationPatient=degenerate_iop,
                        ImagePositionPatient=[0, 0, 1.0], SliceLocation=10.0)

    ordered = geometric_slice_order([high, low])

    assert ordered == [low, high]


def test_load_dicom_series_lists_and_orders_files_from_the_series_directory(tmp_path):
    series_dir = tmp_path / "train_series" / "study1" / "series1"
    series_dir.mkdir(parents=True)
    _write_dicom(series_dir / "b.dcm", ImageOrientationPatient=_AXIAL_IOP,
                 ImagePositionPatient=[0, 0, 30.0])
    _write_dicom(series_dir / "a.dcm", ImageOrientationPatient=_AXIAL_IOP,
                 ImagePositionPatient=[0, 0, 10.0])

    ordered = load_dicom_series(tmp_path, "study1", "series1", split="train")

    assert [f.name for f in ordered] == ["a.dcm", "b.dcm"]


def _write_slot_cache_shard(cache_dir, shard, n_studies=2, n_mismatch=0):
    """A minimal but real slot cache shard: small (n, 6, 9, 4, 4) uint8
    array instead of the real (n, 6, 9, 224, 224) — same shape family,
    cheap to write/read. `n_mismatch` deliberately shortens the studies
    CSV to exercise the row-count guard."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = np.zeros((n_studies, 6, 9, 4, 4), dtype=np.uint8)
    mask = np.ones((n_studies, 6), dtype=np.float32)
    np.save(cache_dir / f"{shard}_cache.npy", cache)
    np.save(cache_dir / f"{shard}_mask.npy", mask)

    study_ids = [f"study{i}" for i in range(n_studies - n_mismatch)]
    pd.DataFrame({"StudyInstanceUID": study_ids}).to_csv(
        cache_dir / f"{shard}_studies.csv", index=False
    )


def test_load_slot_cache_shard_returns_aligned_cache_mask_and_study_ids(tmp_path):
    _write_slot_cache_shard(tmp_path, "train.s00of04", n_studies=3)

    cache, mask, study_ids = load_slot_cache_shard(tmp_path, "train.s00of04")

    assert cache.shape == (3, 6, 9, 4, 4)
    assert mask.shape == (3, 6)
    assert list(study_ids) == ["study0", "study1", "study2"]


def test_load_slot_cache_shard_raises_when_row_counts_disagree(tmp_path):
    _write_slot_cache_shard(tmp_path, "train.s00of04", n_studies=3, n_mismatch=1)

    with pytest.raises(ValueError):
        load_slot_cache_shard(tmp_path, "train.s00of04")
