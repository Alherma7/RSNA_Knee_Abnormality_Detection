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

import pandas as pd
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


def load_dicom_series(raw_dir: Path, study_id: str, series_id: str,
                       split: str = "train") -> "list[Path]":
    """Return one series' slices, ordered by physical slice position.

    Do NOT sort by filename. The filename is the SOP Instance UID,
    assigned to be unique rather than ordered — replicated on 10 sample
    series (notebooks/01_eda_dicom.ipynb, section D, 2026-08-16 kernel
    run): rho(filename order, InstanceNumber) ranged -0.24 to 0.69,
    confirming filename order is not a reliable proxy for physical
    order (matches both reference notebooks' rho ~ 0.01 finding).

    Order by SliceLocation, not by indexing ImagePositionPatient
    yourself. InstanceNumber, ImagePositionPatient, and SliceLocation
    all survive the official 86-tag allowlist (confirmed present on a
    sample file; 64 named tags were populated there — the allowlist
    caps what CAN appear, not every tag on every acquisition).
    ImagePositionPatient is a 3D point whose varying axis depends on the
    scan plane (sagittal/coronal/axial vary a different one of x/y/z) —
    indexing a fixed component (e.g. [2], assuming "z") produced a
    constant column (a scipy ConstantInputWarning) on a non-axial series
    during this same EDA run. SliceLocation is the scalar DICOM already
    computes by projecting onto the slice normal, so use that directly
    instead of re-deriving the projection from ImageOrientationPatient.

    rho(InstanceNumber, SliceLocation) was exactly +-1.000 on all 10
    sample series (same EDA run) — both encode the same physical order,
    rank for rank, but the SIGN flips between series (5 of 10 were -1):
    ascending InstanceNumber means ascending SliceLocation on some
    series and descending on others, presumably per acquisition/scanner
    convention. Pick one field (SliceLocation) and sort by it
    consistently; do not mix ordering by InstanceNumber on some series
    and SliceLocation on others expecting the same direction.

    `split` selects train_series/ or test_series/ (see
    build_dicom_cache).
    """
    raise NotImplementedError("Fill in once the DICOM data is available.")


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


def load_training_labels(raw_dir: Path, n_folds: int = None) -> pd.DataFrame:
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
      config.CV_FOLDS), grouped by src.labelers.report_group_key() —
      not by study — so a report template shared across studies (54
      groups / 206 studies measured in notebooks/02_eda_reports.ipynb
      section E, including 1 template shared between a gold and a weak
      study) never spans train and validation. Group membership doesn't
      depend on label values, so no stratification is attempted here:
      the class-imbalance concern that motivated
      MultilabelStratifiedKFold for the tiny 58-gold CV in Fase 4 is far
      less pressing at n=4,407, and stratifying on a mix of hard 0/1 and
      graded 0.5-abstention targets isn't well-defined anyway.

    Graduated 2026-08-18 from
    notebooks/04b_gold_weak_groupkfold.ipynb, where this exact
    construction was validated against the real train.csv: 4,407 rows
    out of 4,407 studies, 0 discrepancies between the gold rows here and
    train.csv's own official columns, 0 report-template groups split
    across folds.
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
    gkf = GroupKFold(n_splits=n_folds)
    fold = pd.Series(-1, index=reports.index, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(gkf.split(reports, groups=group_keys.to_numpy())):
        fold.iloc[val_idx] = fold_idx
    combined["fold"] = fold

    return combined
