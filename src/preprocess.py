"""Live, per-study DICOM decode for A4's submission pipeline.

This competition is a Code Competition: the hidden test set only exists
during the scored rerun, so the A3 pre-built pixel cache (built before
the test set existed) cannot cover it — inference needs its own decode
step. The functions here port stevenleehans's cache-building source
(data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb),
adapted for single-study synchronous use instead of corpus-wide batch
building. See docs/superpowers/specs/2026-08-27-a4-submission-pipeline-design.md.

Only the pure, DICOM-free pieces live here — fully unit-testable without
Kaggle/GPU/real DICOM files. The full per-study decode orchestration
(reading real pixels, physical crop, laterality flip) stays notebook-only
until the spec's section 3 validation gate passes for real on Kaggle;
see notebooks/07v1_a2_submission_inference.ipynb.
"""

import re

import numpy as np
import pandas as pd
import torch

# Source: cell-10/cell-12 of the reference notebook above.
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
    r"water excit|\btirm\b|\bsting\b|\bfatsup\b"
)
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")
FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}

_HEADER_COLS = (
    "SeriesDescription", "SequenceName", "ScanOptions",
    "RepetitionTime", "EchoTime", "ScanningSequence",
)


def annotate_series_headers(series_df: pd.DataFrame) -> pd.DataFrame:
    """Recover fat-suppression and pulse-sequence weighting from headers.

    Two matching traps this must avoid (both silently invert the answer
    if missed): underscore is a word character, so a token test for `we`
    (water excitation) never fires inside `t2_de3d_we_tra` unless
    separators are normalised first; and GE writes `SAT_GEMS` for
    *spatial* saturation, so ScanOptions must be matched as exact tokens
    or non-fat-suppressed series get marked as suppressed.

    Adds `fatsat` (bool), `weight` (T1/T2/PD/GRE/UNK), `fluid` (bool,
    True iff weight is PD or T2) to a copy of `series_df`. Missing
    optional header columns are treated as absent, not required.

    Source: stevenleehans's `annotate()`,
    data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb
    (cell-12) — same source A3's pre-built cache was built with.
    """
    df = series_df.copy()
    for col in _HEADER_COLS:
        if col not in df.columns:
            df[col] = None

    desc = df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna("")
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)

    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs

    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1 = desc.str.contains(_T1_RX)
    t2 = desc.str.contains(_T2_RX)
    pdw = desc.str.contains(_PD_RX)

    df["weight"] = np.where(
        t1 & ~t2 & ~pdw, "T1",
        np.where(
            t2 & ~pdw, "T2",
            np.where(
                pdw, "PD",
                np.where(
                    gre, "GRE",
                    np.where(
                        tr < 800, "T1",
                        np.where(te > 60, "T2", np.where(tr >= 800, "PD", "UNK")),
                    ),
                ),
            ),
        ),
    )
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    return df


def normalise_laterality(img: "torch.Tensor", plane: str, laterality: str) -> "torch.Tensor":
    """Map every knee onto a left-knee convention.

    Four of the twelve findings are medial/lateral pairs, and medial is
    defined against the body midline — which side of the *image* it
    falls on depends on which knee was scanned. Coronal and axial views
    mirror under a horizontal flip; sagittal stacks don't (each slice is
    unchanged by mirroring) — what differs is the direction the stack
    traverses the joint, so the slice order is reversed instead.

    Where laterality is unresolved (`"unknown"`, or already `"L"`) the
    volume is left alone: a wrong flip is worse than no flip.

    `img`: `(group, H, W)`, one anchor's stacked physically-adjacent
    slices.

    Source: stevenleehans's `normalise_laterality()`,
    data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb
    (cell-14) — same source A3's pre-built cache was built with.
    """
    if laterality != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])
