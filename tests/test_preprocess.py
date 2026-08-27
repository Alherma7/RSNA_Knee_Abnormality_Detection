"""Unit tests for src/preprocess.py — the pure, DICOM-free pieces of A4's
live test-time decode. Ported from stevenleehans's cache-building source
(data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb,
cells cell-10/cell-12/cell-14) per
docs/superpowers/specs/2026-08-27-a4-submission-pipeline-design.md, section 7.
"""

import torch

import pandas as pd

from src.preprocess import annotate_series_headers, normalise_laterality


def _series_row(**overrides):
    row = {
        "SeriesInstanceUID": "s1",
        "SeriesDescription": "",
        "SequenceName": "",
        "ScanOptions": "",
        "RepetitionTime": None,
        "EchoTime": None,
        "ScanningSequence": "",
    }
    row.update(overrides)
    return row


def test_annotate_series_headers_avoids_the_sat_gems_false_positive():
    """GE writes SAT_GEMS for *spatial* saturation, not fat suppression —
    ScanOptions must be matched as exact tokens or this reads as fat-sat."""
    df = pd.DataFrame([_series_row(ScanOptions="SAT_GEMS", SeriesDescription="sag pd")])

    annotated = annotate_series_headers(df)

    assert bool(annotated.loc[0, "fatsat"]) is False


def test_annotate_series_headers_matches_fatsat_options_exactly():
    df = pd.DataFrame([_series_row(ScanOptions="FS|OTHER")])

    annotated = annotate_series_headers(df)

    assert bool(annotated.loc[0, "fatsat"]) is True


def test_annotate_series_headers_normalizes_underscore_separators_before_matching():
    """Underscore is a word character, so `\\bwe\\b` never fires inside
    "t2_de3d_we_tra" unless separators are normalised to spaces first."""
    df = pd.DataFrame([_series_row(SequenceName="t2_de3d_we_tra")])

    annotated = annotate_series_headers(df)

    assert bool(annotated.loc[0, "fatsat"]) is True


def test_annotate_series_headers_classifies_t1_t2_pd_weighting():
    df = pd.DataFrame([
        _series_row(SeriesDescription="sag t1"),
        _series_row(SeriesDescription="sag t2"),
        _series_row(SeriesDescription="cor pd"),
        _series_row(SeriesDescription="unlabeled sequence", RepetitionTime=400),
    ])

    annotated = annotate_series_headers(df)

    assert list(annotated["weight"]) == ["T1", "T2", "PD", "T1"]


def test_annotate_series_headers_fluid_flag_matches_pd_and_t2_only():
    df = pd.DataFrame([
        _series_row(SeriesDescription="sag t1"),
        _series_row(SeriesDescription="sag t2"),
        _series_row(SeriesDescription="cor pd"),
    ])

    annotated = annotate_series_headers(df)

    assert list(annotated["fluid"]) == [False, True, True]


def test_annotate_series_headers_tolerates_missing_column_keys():
    """Missing header columns are tolerated — rows with no header information
    should produce sensible defaults: fatsat=False, weight=UNK, fluid=False."""
    df = pd.DataFrame([{"SeriesInstanceUID": "s1"}])

    annotated = annotate_series_headers(df)

    assert bool(annotated.loc[0, "fatsat"]) is False
    assert annotated.loc[0, "weight"] == "UNK"
    assert bool(annotated.loc[0, "fluid"]) is False


def test_normalise_laterality_flips_coronal_horizontally_for_right_knee():
    img = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)

    flipped = normalise_laterality(img, "Coronal", "R")

    assert torch.equal(flipped, torch.flip(img, dims=[-1]))
    assert not torch.equal(flipped, img)


def test_normalise_laterality_flips_axial_horizontally_for_right_knee():
    img = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)

    flipped = normalise_laterality(img, "Axial", "R")

    assert torch.equal(flipped, torch.flip(img, dims=[-1]))


def test_normalise_laterality_reverses_slice_order_for_sagittal_right_knee():
    img = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)

    flipped = normalise_laterality(img, "Sagittal", "R")

    assert torch.equal(flipped, torch.flip(img, dims=[0]))
    assert not torch.equal(flipped, torch.flip(img, dims=[-1]))


def test_normalise_laterality_leaves_left_and_unknown_unchanged():
    img = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)

    for plane in ("Sagittal", "Coronal", "Axial"):
        assert torch.equal(normalise_laterality(img, plane, "L"), img)
        assert torch.equal(normalise_laterality(img, plane, "unknown"), img)
