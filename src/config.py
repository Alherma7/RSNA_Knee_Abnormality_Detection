"""Paths, constants, and shared settings for the RSNA Knee project.

Dataset size note: the full competition dataset is ~0.5 TB of DICOM
files, which does not fit comfortably alongside everything else on the
local disk (~515 GB free at project start). Heavy work (Fases 1+, image
loading, training) runs in Kaggle Notebooks against the dataset already
mounted at _KAGGLE_INPUT_DIR; only a small local subset (the gold-labeled
studies, for prototyping/tests) is expected under DATA_RAW_DIR. See
README.md Next steps.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Kaggle Notebooks mount competition data read-only under
# /kaggle/input/competitions/<slug>/ (confirmed 2026-08-16 on a live
# kernel — /kaggle/input/<slug>/, without "competitions/", does not
# exist for this competition) and set KAGGLE_KERNEL_RUN_TYPE; detect
# that to pick up the full dataset there instead of expecting it to
# have been downloaded locally.
_KAGGLE_INPUT_DIR = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
DATA_RAW_DIR = _KAGGLE_INPUT_DIR if ON_KAGGLE else PROJECT_ROOT / "data" / "raw"

RANDOM_STATE = 42

# 4, not 5: measured 2026-08-25 (notebooks/00v2_measurement_gate.ipynb,
# A0) on the real corpus, grouping by the union of report-template and
# scanner-fingerprint groups (see src/data.py::build_group_ids) — gold
# studies per fold min=11/4 folds vs. min=7/5 folds. The pooled 58-gold
# gate is already noise-limited; a fold with only 7 gold studies makes
# that fold's contribution close to meaningless. Same motivation
# mammography 1st place cites for going 5->4.
CV_FOLDS = 4

# Verified 2026-08-16 against the official Kaggle "Dataset Description"
# (pasted by the user from the competition data tab) — content and order
# confirmed to match the third-party reference notebooks. Canonical
# snake_case names used throughout this codebase; see
# OFFICIAL_LABEL_COLUMNS below for the raw train.csv header spelling.
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

# Maps each FINDINGS entry to its exact column header in train.csv, per
# the official Dataset Description. Use to rename columns on load
# (src.data.load_gold_labels) rather than hardcoding the raw headers
# elsewhere.
OFFICIAL_LABEL_COLUMNS = {
    "acl_injury": "ACL",
    "mcl_injury": "MCL",
    "medial_meniscus_tear": "Medial Meniscus",
    "lateral_meniscus_tear": "Lateral Meniscus",
    "oa_medial_compartment": "Medial OA",
    "oa_lateral_compartment": "Lateral OA",
    "oa_patellofemoral_compartment": "PF OA",
    "effusion": "Effusion",
    "synovitis": "Synovitis",
    "bakers_cyst": "Baker's",
    "bone_contusion": "Contusion",
    "fracture": "Fracture",
}

# Official Dataset Description spells these "Sagittal", "Coronal",
# "Axial" (train_series.csv::Anatomical_Plane) — lowercase when
# comparing/joining against this tuple.
MRI_PLANES = ("sagittal", "coronal", "axial")

# Both reference notebooks report reports.csv text in up to 9 languages,
# with no language identifier column — lexicon-based labeling functions
# must test all languages at once rather than routing by a language guess.
REPORT_LANGUAGE_COUNT = 9
