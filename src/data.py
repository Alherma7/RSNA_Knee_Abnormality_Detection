"""Loading raw DICOM series, radiology reports, and gold labels.

Sources:
- MONAI docs (docs.monai.io) for DICOM-aware readers and transforms.
- pydicom for header/metadata inspection ahead of any framework.
- Both reviewed reference notebooks (pilkwang/rsna-knee-baseline-v1,
  prvsiyan/rsna-knee-read-the-report-then-the-knee) for two DICOM
  gotchas specific to this competition's data, noted per-function below.
See RESOURCES.md for the full citations and why each applies here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import GroupKFold

from src import config
from src.labelers import label_reports, report_group_key


def build_dicom_cache(raw_dir: Path, split: str = "train") -> Path:
    """Scan raw DICOM files once and cache per-study header metadata.

    Avoids re-reading hundreds of files per study on every epoch — both
    reference notebooks decode each study's pixels once into a uint8
    cache up front, since I/O (not arithmetic) dominates cost at this
    file count. `split` selects train_series/ or test_series/ (both
    exist with the same <StudyInstanceUID>/<SeriesInstanceUID>/
    <SOPInstanceUID>.dcm layout per the official Dataset Description —
    inference needs the test_series/ tree, since only `Report` is
    train-only, not the images). Returns the path to the cached metadata
    table.
    """
    raise NotImplementedError("Fill in once the DICOM data is available.")


def geometric_slice_order(files: "list[Path]") -> "list[Path]":
    """Sort a series' DICOM files by physical position along the slice normal.

    Primary key: ImagePositionPatient projected onto the normal of
    ImageOrientationPatient (cross(row_dir, col_dir)) — consistent
    across series because both tags share DICOM's LPS patient coordinate
    system. Falls back to SliceLocation, then InstanceNumber, then
    input file order when the geometric tags are missing or degenerate
    (zero-norm normal). Source: stevenleehans, this competition's
    discussion 735304.

    Do NOT sort by bare SliceLocation alone, which the prior version of
    this docstring (and the now-obsolete 06/06b notebooks) recommended.
    SliceLocation's sign is manufacturer/protocol-defined, not
    guaranteed to point the same physical direction across series:
    measured directly against this geometric method on all 58 real gold
    studies (notebooks/01v2_slice_ordering.ipynb, A0b, 2026-08-25) — 41
    of 58 (71%) were pure reversals (same slice set, opposite
    direction), uncorrelated with gantry tilt (mean 9.2 vs 8.2 degrees
    off-axis for mismatched vs. matched studies) or duplicate
    SliceLocation values (0 found in either group).

    Also do NOT sort by filename — the filename is the SOP Instance UID,
    assigned to be unique rather than ordered — replicated on 10 sample
    series (notebooks/01_eda_dicom.ipynb, section D, 2026-08-16 kernel
    run): rho(filename order, InstanceNumber) ranged -0.24 to 0.69,
    confirming filename order is not a reliable proxy for physical
    order.

    Graduated 2026-08-25 from notebooks/01v2_slice_ordering.ipynb
    (Cells 2-3), user-reviewed before promotion per this project's
    notebook-to-src graduation rule.
    """
    records = []
    for idx, f in enumerate(files):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        key = None

        iop = getattr(ds, "ImageOrientationPatient", None)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) == 3:
            row_dir = np.array(iop[0:3], dtype=float)
            col_dir = np.array(iop[3:6], dtype=float)
            normal = np.cross(row_dir, col_dir)
            if np.linalg.norm(normal) > 1e-6:
                key = float(np.dot(np.array(ipp, dtype=float), normal))

        if key is None:
            sl = getattr(ds, "SliceLocation", None)
            if sl is not None:
                key = float(sl)
        if key is None:
            inum = getattr(ds, "InstanceNumber", None)
            if inum is not None:
                key = float(inum)
        if key is None:
            key = float(idx)

        records.append((key, f))

    records.sort(key=lambda r: r[0])
    return [f for _, f in records]


def load_dicom_series(raw_dir: Path, study_id: str, series_id: str,
                       split: str = "train") -> "list[Path]":
    """Return one series' DICOM files, ordered by physical slice position.

    `split` selects train_series/ or test_series/ (see
    build_dicom_cache) — inference needs the test_series/ tree, since
    only `Report` is train-only, not the images. Ordering is
    geometric_slice_order() — see that function's docstring for why bare
    SliceLocation or filename order are not safe to sort by.
    """
    series_dir = raw_dir / f"{split}_series" / study_id / series_id
    files = sorted(series_dir.glob("*.dcm"))
    return geometric_slice_order(files)


def build_group_ids(*group_key_series: pd.Series) -> pd.Series:
    """Union-find over one or more grouping keys sharing the same index.

    Two rows land in the same final group if they share ANY of the given
    keys, directly or transitively through a third row. A naive
    combination (e.g. a tuple of both keys per row) is wrong: if row A
    shares key 1 with B, and B shares key 2 with C, A and C must end up
    in the same group even though A and C share neither key directly —
    only union-find gives the correct connected components.

    A missing value (NaN/None) in a key series never unions anything —
    every row keeps its own component under that key, since two missing
    values are never treated as "the same" group.

    Graduated 2026-08-25 from notebooks/00v2_measurement_gate.ipynb
    (A0, Part B.3), where this exact construction — unioning
    src.labelers.report_group_key() with build_scanner_fingerprints()
    below — fixed a measured leak: 51 scanner fingerprints were split
    across a fold's train/val boundary under report-only grouping,
    affecting 4,399 of 4,407 studies (Kaggle run, real corpus). The
    fixed folds measured 0 report-template leaks and 0 scanner leaks.
    """
    index = group_key_series[0].index
    parent = {i: i for i in index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for keys in group_key_series:
        valid = keys.dropna()
        for _, idx in valid.groupby(valid).groups.items():
            idx = list(idx)
            for other in idx[1:]:
                union(idx[0], other)

    return pd.Series({i: find(i) for i in index}, name="group_id")


# DICOM tags used to fingerprint the scanner/site a study was acquired
# on. Not an exhaustive identifier — just enough to catch the dominant
# leakage source measured in notebooks/00v2_measurement_gate.ipynb.
SCANNER_FINGERPRINT_TAGS = (
    "Manufacturer",
    "ManufacturerModelName",
    "InstitutionName",
    "DeviceSerialNumber",
    "MagneticFieldStrength",
    "StationName",
)


def build_scanner_fingerprints(raw_dir: Path, split: str = "train") -> pd.Series:
    """Return one scanner fingerprint per study, from a single DICOM header.

    Reads exactly one file (the first series listed for that study in
    {split}_series.csv, its first file by name) with
    `stop_before_pixels=True` — header-only, so this is cheap even
    across the full corpus (no pixel decoding). The fingerprint is a
    tuple of SCANNER_FINGERPRINT_TAGS values (stringified; a missing tag
    becomes the string "None", not a Python None, so two studies missing
    the same tag still fingerprint identically rather than each getting
    a unique-by-accident value).

    Needs the full DICOM tree under raw_dir/{split}_series/ — Kaggle
    only, per project convention (see project-rsna-kaggle-notebooks-
    strategy memory). Raises naturally (FileNotFoundError) if that tree
    isn't present; this is deliberately not caught or defaulted, since a
    silent fallback here is exactly the kind of gate-lying-to-itself
    failure mode A0 exists to rule out (see
    notebooks/00v2_measurement_gate.ipynb).

    Graduated 2026-08-25 from notebooks/00v2_measurement_gate.ipynb
    (A0, Part B.1), where this scan found 51 fingerprints shared across
    a fold's train/val boundary under the then-current report-only
    GroupKFold (see build_group_ids for the fix and full numbers).
    """
    series = pd.read_csv(raw_dir / f"{split}_series.csv")
    first_series = series.drop_duplicates("StudyInstanceUID", keep="first")

    fingerprints = {}
    for row in first_series.itertuples(index=False):
        series_dir = raw_dir / f"{split}_series" / row.StudyInstanceUID / row.SeriesInstanceUID
        files = sorted(series_dir.glob("*.dcm"))
        if not files:
            fingerprints[row.StudyInstanceUID] = None
            continue
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        fingerprints[row.StudyInstanceUID] = tuple(
            str(getattr(ds, tag, None)) for tag in SCANNER_FINGERPRINT_TAGS
        )

    result = pd.Series(fingerprints, name="scanner_fingerprint")
    result.index.name = "StudyInstanceUID"
    return result


def load_reports(raw_dir: Path) -> pd.DataFrame:
    """Load the free-text radiology reports, one row per study.

    Per the official Dataset Description, `Report` is a column in
    train.csv itself (there is no separate reports.csv) and is absent
    from test.csv — text exists only at train time, unlike the images
    (see load_dicom_series). Reports appear in several languages with no
    language identifier column (both reviewed reference notebooks put
    the count at up to config.REPORT_LANGUAGE_COUNT) — downstream
    labeling functions must not route by a guessed language.

    Graduated 2026-08-18, validated in
    notebooks/04b_gold_weak_groupkfold.ipynb against the real 4,407-row
    train.csv (`Report` confirmed non-null on every row, per Fase 1).
    """
    train = pd.read_csv(raw_dir / "train.csv")
    return train[["StudyInstanceUID", "Report"]].set_index("StudyInstanceUID")


def load_gold_labels(raw_dir: Path) -> pd.DataFrame:
    """Load the small officially-labeled subset (exactly 58 gold studies).

    Per the official Dataset Description, the twelve label columns
    (config.OFFICIAL_LABEL_COLUMNS) live in train.csv itself, alongside
    `Report` — there is no separate train_labels.csv. Confirmed
    (notebooks/01_eda_dicom.ipynb, section A, 2026-08-16 kernel run,
    all 4,407 rows of train.csv): the null pattern is strictly
    all-or-nothing — 4,349 rows have 0 of the 12 labels populated,
    exactly 58 rows have all 12, zero rows fall in between. Filter on
    e.g. `train[label_cols].notna().all(axis=1)`, then rename columns
    via config.OFFICIAL_LABEL_COLUMNS to this codebase's snake_case
    names. `Report` was non-null on all 4,407 rows in the same check.

    Graduated 2026-08-18, validated in
    notebooks/04b_gold_weak_groupkfold.ipynb: returns exactly 58 rows
    against the real train.csv.
    """
    train = pd.read_csv(raw_dir / "train.csv")
    label_cols = list(config.OFFICIAL_LABEL_COLUMNS.values())
    gold_mask = train[label_cols].notna().all(axis=1)
    gold = train.loc[gold_mask, ["StudyInstanceUID"] + label_cols].set_index("StudyInstanceUID")
    gold.columns = list(config.OFFICIAL_LABEL_COLUMNS.keys())
    return gold


def load_series_metadata(raw_dir: Path, split: str = "train") -> pd.DataFrame:
    """Load {split}_series.csv (plane, Fluid_Sensitive, Fat_Suppression).

    test_series.csv exists with the same schema as train_series.csv per
    the official Dataset Description — inference needs it too, since the
    per-finding attention pooling (src/model.py) keys off plane/sequence
    slots for every study, train or test.

    Fluid_Sensitive and Fat_Suppression: measured 100% agreement across
    all 24,371 rows of the full train_series.csv (notebooks/
    01_eda_dicom.ipynb, section C, 2026-08-16 kernel run) — 0
    disagreements, confirming the reviewed pilkwang baseline notebook's
    finding, not just a sample of it. The official Dataset Description
    hedges ("although often correlated, ... not necessarily equivalent
    for every case"), so re-check this on test_series.csv too before
    hardcoding the assumption at inference time, but on the full
    training data they carry one real axis, not two.

    All 4,407 training studies have all 3 planes present (confirmed,
    same notebook section) — config.MRI_PLANES coverage is not a
    per-study concern here, though series counts per study still vary
    (mean 5.5, median 5, max 14) since each plane can carry multiple
    sequences. Official Anatomical_Plane values are "Sagittal"/
    "Coronal"/"Axial" (capitalized) — lowercase before comparing to
    config.MRI_PLANES.
    """
    raise NotImplementedError("Fill in once {split}_series.csv is available.")


def load_training_labels(raw_dir: Path, scanner_fingerprints: pd.Series,
                          n_folds: int = None) -> pd.DataFrame:
    """Build the Fase 5 training-label table: gold+weak, with CV folds.

    One row per study (all 4,407), indexed by StudyInstanceUID, with one
    column per config.FINDINGS plus `is_gold` and `fold`:
    - The 58 gold studies (load_gold_labels) keep their exact official
      0/1 labels — never passed through the labeler, since they're
      already ground truth.
    - The 4,349 weak studies get graded labels from
      src.labelers.label_reports() (values in {0.0, 0.5, 1.0}, 0.5 =
      abstain) using the labeler's negation-scoping fix (2026-08-18).
    - `fold` comes from a plain sklearn GroupKFold (n_folds defaults to
      config.CV_FOLDS), grouped by build_group_ids() over BOTH
      src.labelers.report_group_key() and `scanner_fingerprints` — not
      by study, and not by report template alone. A report template
      shared across studies (54 groups / 206 studies measured in
      notebooks/02_eda_reports.ipynb section E, including 1 template
      shared between a gold and a weak study) never spans train and
      validation, and neither does a scanner fingerprint: report-only
      grouping was measured (notebooks/00v2_measurement_gate.ipynb, A0)
      to leak 51 scanner fingerprints across a fold's train/val boundary,
      affecting 4,399 of 4,407 studies. `scanner_fingerprints` is
      REQUIRED, not defaulted, on purpose — an implicit fallback to
      report-only grouping here would silently reproduce that measured
      leak, exactly the "gate lying to itself" failure mode A0 exists to
      rule out. Build it with build_scanner_fingerprints() (Kaggle only,
      needs the DICOM tree); pass a same-index all-None Series only for
      tests that deliberately want to isolate report-group-only
      behavior. Group membership doesn't depend on label values, so no
      stratification is attempted here: the class-imbalance concern that
      motivated MultilabelStratifiedKFold for the tiny 58-gold CV in
      Fase 4 is far less pressing at n=4,407, and stratifying on a mix
      of hard 0/1 and graded 0.5-abstention targets isn't well-defined
      anyway.

    Graduated 2026-08-18 from
    notebooks/04b_gold_weak_groupkfold.ipynb (report-only grouping),
    where this construction was validated against the real train.csv:
    4,407 rows out of 4,407 studies, 0 discrepancies between the gold
    rows here and train.csv's own official columns, 0 report-template
    groups split across folds. Scanner grouping added 2026-08-25 from
    notebooks/00v2_measurement_gate.ipynb (A0) — see build_group_ids
    docstring for the real leak numbers this fixes.
    """
    n_folds = n_folds or config.CV_FOLDS
    reports = load_reports(raw_dir)
    gold = load_gold_labels(raw_dir)

    is_gold = reports.index.isin(gold.index)
    weak_reports = reports.loc[~is_gold].reset_index()
    weak_labels = label_reports(weak_reports, config.FINDINGS)

    combined = pd.concat([gold[config.FINDINGS], weak_labels[config.FINDINGS]])
    combined = combined.loc[reports.index]
    combined["is_gold"] = is_gold

    group_keys = reports["Report"].apply(report_group_key)
    group_ids = build_group_ids(group_keys, scanner_fingerprints.reindex(reports.index))
    gkf = GroupKFold(n_splits=n_folds)
    fold = pd.Series(-1, index=reports.index, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(gkf.split(reports, groups=group_ids.to_numpy())):
        fold.iloc[val_idx] = fold_idx
    combined["fold"] = fold

    return combined
