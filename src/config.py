"""Paths, constants, and shared settings for the RSNA Knee project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RANDOM_STATE = 42
CV_FOLDS = 5

# Confirmed from the reviewed reference notebooks (pilkwang/rsna-knee-
# baseline-v1 and prvsiyan/rsna-knee-read-the-report-then-the-knee) —
# verify against the official data dictionary once train_labels.csv is
# downloaded, since these were read from a third party's description of
# the schema, not the raw file itself.
FINDINGS = [
    "acl_injury",
    "mcl_injury",
    "medial_meniscus_tear",
    "lateral_meniscus_tear",
    "oa_medial_compartment",
    "oa_lateral_compartment",
    "oa_patellofemoral_compartment",
    "effusion",
    "synovitis",
    "bakers_cyst",
    "bone_contusion",
    "fracture",
]

MRI_PLANES = ("sagittal", "coronal", "axial")

# Both reference notebooks report reports.csv text in up to 9 languages,
# with no language identifier column — lexicon-based labeling functions
# must test all languages at once rather than routing by a language guess.
REPORT_LANGUAGE_COUNT = 9
