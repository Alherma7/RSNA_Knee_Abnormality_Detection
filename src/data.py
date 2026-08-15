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


def build_dicom_cache(raw_dir: Path) -> Path:
    """Scan raw DICOM files once and cache per-study header metadata.

    Avoids re-reading hundreds of files per study on every epoch — both
    reference notebooks decode each study's pixels once into a uint8
    cache up front, since I/O (not arithmetic) dominates cost at this
    file count. Returns the path to the cached metadata table.
    """
    raise NotImplementedError("Fill in once the DICOM data is available.")


def load_dicom_series(study_id: str, plane: str) -> "list[Path]":
    """Return one study's plane series, ordered by physical slice position.

    Do NOT sort by filename. The filename is the SOP Instance UID,
    assigned to be unique rather than ordered — both reference notebooks
    measured filename-order vs. physical-position rank correlation at
    approximately rho = 0.01 on this corpus. Order by the DICOM
    InstanceNumber / ImagePositionPatient metadata instead.
    """
    raise NotImplementedError("Fill in once the DICOM data is available.")


def load_reports(raw_dir: Path) -> pd.DataFrame:
    """Load the free-text radiology reports, one row per study.

    Reports appear in up to 9 languages with no language identifier
    column (see config.REPORT_LANGUAGE_COUNT) — downstream labeling
    functions must not route by a guessed language.
    """
    raise NotImplementedError("Fill in once reports.csv is available.")


def load_gold_labels(raw_dir: Path) -> pd.DataFrame:
    """Load the small officially-labeled subset (the ~58 gold studies)."""
    raise NotImplementedError(
        "Fill in once train_labels.csv is available."
    )


def load_series_metadata(raw_dir: Path) -> pd.DataFrame:
    """Load train_series.csv (plane, Fluid_Sensitive, Fat_Suppression).

    Caution: the reviewed pilkwang baseline notebook found that
    Fluid_Sensitive and Fat_Suppression agree on every training row —
    i.e. they carry one real axis, not two, in this data. Verify before
    treating them as independent features.
    """
    raise NotImplementedError("Fill in once train_series.csv is available.")
