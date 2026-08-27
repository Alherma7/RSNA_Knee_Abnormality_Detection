"""Unit tests for src/preprocess.py — the pure, DICOM-free pieces of A4's
live test-time decode. Ported from stevenleehans's cache-building source
(data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb,
cells cell-10/cell-12/cell-14) per
docs/superpowers/specs/2026-08-27-a4-submission-pipeline-design.md, section 7.
"""

import pandas as pd

from src.preprocess import annotate_series_headers


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
