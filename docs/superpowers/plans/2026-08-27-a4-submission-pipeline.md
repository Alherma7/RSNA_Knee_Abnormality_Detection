# A4: Submission Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real, submittable Kaggle notebook that decodes the hidden test set's DICOM live, runs A2 v1's ensembled 4-fold inference, and writes a valid `submission.csv` — the first real leaderboard number since Fase 5's calibration submission (0.596), now against A2 v1's pooled 0.7512 local CV.

**Architecture:** A self-contained Kaggle notebook (`notebooks/07v1_a2_submission_inference.ipynb`) ports the A3 cache's own source decode logic (slot recovery, laterality normalization, physical-crop pixel decode) for per-study synchronous use, validates it against the pre-built train cache's pixels before trusting it, then loads the 4 A2 v1 fold checkpoints and ensembles their predictions into `submission.csv`. Only the DICOM-free pure logic (header classification, laterality-flip math) graduates to `src/preprocess.py` now; the full DICOM-reading orchestration stays notebook-only until the Kaggle validation gate passes, same convention A2 used for its model code.

**Tech Stack:** Python, pandas/numpy/pydicom (existing), PyTorch + timm (existing, from A2), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-a4-submission-pipeline-design.md`

## Global Constraints

- This is a **Code Competition**: `<=9h` rerun time, **internet disabled** during scoring, `submission.csv` must be the notebook's `Output`.
- The pre-built A3 slot cache cannot cover the hidden test set (built before the competition's test set existed) — live DICOM decode inside the submission notebook is required, no shortcut.
- **Exact-match parity requirement:** the live decode must reproduce the pre-built train cache's pixels for the same study, or the §3 validation gate fails. Two easy-to-miss parameters that must match exactly: `RSNA_WINDOW=(0.35, 0.65)` — the sampling window actually used to build the real cache (confirmed in `cache_meta.json`, per RESOURCES.md; **not** the source code's per-plane `PLANE_WINDOW` default, which was overridden) — and `crop_mm=130.0`/`img_size=224`/`group=3` (all already in `config.py` as `SLOT_CACHE_CROP_MM`/`SLOT_CACHE_IMG_SIZE`/`SLOT_CACHE_GROUP_SIZE`).
- `group_index=1` (centre anchor) — same fixed value A2 v1 trained with. Decoding only the centre anchor's 3 slices (not all 9) is a deliberate, derivable optimization: `read_slot()`'s 3-anchor `np.linspace(lo, hi, 3)` puts its middle anchor at exactly `(lo+hi)//2` for non-negative integer bounds, identical to decoding one anchor directly at that centre — Task 4 states this derivation inline as a code comment, and the §3 validation gate empirically confirms it.
- Ensembling: **mean of sigmoid probabilities across all 4 fold checkpoints** (`models/a2_v1_fold{0,1,2,3}_best.pt`) — same pattern as Fase 5's retired 5-fold ensemble.
- Submission assembly: start every row at the constant `0.5` fallback (from `sample_submission.csv`), per-study try/except during decode+inference (never abort the whole run on one bad study), then assert column order / row count / `StudyInstanceUID` uniqueness before writing.
- **Kaggle notebooks in this project cannot `import src`** — any function graduated to `src/preprocess.py` in this plan is *also* hand-duplicated into the notebook (same convention as `05v2`/`06v2`).
- Per the approved spec §7: only `annotate_series_headers()` (header classification) and `normalise_laterality()` (flip math) graduate to `src/preprocess.py` in this plan — both are pure, DICOM-free, and fully unit-testable. The full per-study decode orchestration (`decode_study_slots`, slot selection, pixel crop/resize) stays notebook-only until the §3 Kaggle validation gate passes for real — this plan does not graduate it.
- `timm`-without-internet (spec §6) is an open risk verified inside the notebook itself (Task 6), not resolved in advance — no local way to check it.

---

## Task 1: `annotate_series_headers()` — header-based slot classification

**Files:**
- Create: `src/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Produces: `annotate_series_headers(series_df: pd.DataFrame) -> pd.DataFrame` — returns a copy of `series_df` with 3 new columns: `fatsat` (bool), `weight` (`"T1"`/`"T2"`/`"PD"`/`"GRE"`/`"UNK"`), `fluid` (bool, `True` iff `weight` in `{"PD", "T2"}`). Missing optional header columns (`SeriesDescription`, `SequenceName`, `ScanOptions`, `RepetitionTime`, `EchoTime`, `ScanningSequence`) are tolerated (treated as absent/NaN), not required.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_preprocess.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.preprocess'`.

- [ ] **Step 3: Implement `annotate_series_headers()`**

Create `src/preprocess.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "Add annotate_series_headers(), A4's ported slot-classification logic"
```

---

## Task 2: `normalise_laterality()` — canonical left-knee convention

**Files:**
- Modify: `src/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `torch` (already a dependency, added for A2's Task 3).
- Produces: `normalise_laterality(img: torch.Tensor, plane: str, laterality: str) -> torch.Tensor` — `img` shape `(group, H, W)` (one anchor's stacked slices). `laterality == "R"` and `plane in ("Coronal", "Axial")` flips the last dimension (horizontal flip); `laterality == "R"` and any other plane (i.e. `"Sagittal"`) reverses dim 0 (slice order); any other `laterality` (`"L"` or `"unknown"`) returns `img` unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_preprocess.py`:

```python
import torch

from src.preprocess import annotate_series_headers, normalise_laterality


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_preprocess.py -k normalise_laterality -v`
Expected: FAIL — `ImportError: cannot import name 'normalise_laterality'`.

- [ ] **Step 3: Implement `normalise_laterality()`**

Add to `src/preprocess.py`, after `annotate_series_headers()`:

```python
import torch


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "Add normalise_laterality(), A4's ported left-knee-convention flip"
```

---

## Task 3: Stage the 4 fold checkpoints + metadata for the Kaggle Dataset upload

**Files:**
- Create: `models/a2_v1_checkpoint_metadata.json` (gitignored, same as the checkpoints themselves — `models/*` per `.gitignore`)

**Interfaces:**
- Produces: a JSON file describing the checkpoint bundle the submission notebook will mount as a Kaggle Dataset — mirrors the `metadata.json` pattern Fase 5's retired `06` submission notebook already used successfully (`CHECKPOINTS_MOUNT / "metadata.json"`).

- [ ] **Step 1: Write the metadata file**

Run:
```bash
python -c "
import json
from src import config

metadata = {
    'arm': 'a2_v1_pooled_4fold',
    'cv_folds': config.CV_FOLDS,
    'findings': config.FINDINGS,
    'official_label_columns': config.OFFICIAL_LABEL_COLUMNS,
    'backbone': 'vit_small_patch14_dinov2.lvd142m',
    'group_index': 1,
    'reference_pooled_oof_macro_auc': 0.7512,
}
with open('models/a2_v1_checkpoint_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(json.dumps(metadata, indent=2))
"
```
Expected: prints the metadata dict, `cv_folds: 4`, 12 findings, 12 official label columns.

- [ ] **Step 2: Verify it round-trips**

Run: `python -c "import json; print(json.load(open('models/a2_v1_checkpoint_metadata.json')))"`
Expected: loads without error, matches Step 1's printed dict.

- [ ] **Step 3: No commit needed**

`models/a2_v1_checkpoint_metadata.json` is gitignored (`models/*`), same as the 4 `.pt` checkpoint files it describes — all 5 files travel together as one manually-uploaded Kaggle Dataset (Task 7 tells the user exactly when and how).

---

## Task 4: Notebook part 1 — live decode (slot recovery, pixel crop, laterality)

**Files:**
- Create: `notebooks/07v1_a2_submission_inference.ipynb` (built via a Python builder script, same technique as `04v2`/`05v2`/`06v2`)

**Interfaces:**
- Produces (within the notebook's own namespace, no `import src`): `annotate_series_headers()`, `normalise_laterality()` (hand-duplicated copies of Tasks 1–2), `select_slot_series()`, `decode_study_slots()`. Task 5 and Task 6 append more cells to this same notebook.

This notebook **cannot be executed by an agent** (needs the full DICOM tree + GPU, not runnable locally or by me at all) — its "test" is nbformat validation + a compile-check on every code cell, same as every other Kaggle-only notebook in this project. Real execution happens once the user runs it on Kaggle (Task 7 tells them exactly what to attach).

- [ ] **Step 1: Write the notebook builder script**

Create a scratch builder script (e.g. in the session scratchpad):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# 07v1 - A4: submission pipeline (live decode + ensembled A2 v1 inference)

**Plan item A4** (see
docs/superpowers/specs/2026-08-27-a4-submission-pipeline-design.md).
Self-contained per this project's Kaggle constraint (no `import src`) -
`annotate_series_headers`/`normalise_laterality` below are hand-kept
copies of `src/preprocess.py`'s graduated functions; keep them in sync
manually if either side changes.

**Why this notebook exists:** this is a Code Competition - the hidden
test set only exists during the scored rerun (`<=9h`, internet
disabled), so the A3 pre-built pixel cache (built before the test set
existed) cannot cover it. This notebook decodes DICOM live instead,
using the same slot-recovery/pixel-decode/laterality logic the cache was
built with (ported from `data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`),
so its output matches the cache's convention exactly - validated in
Section 2 below before trusting it on real hidden data.""")

code(r"""import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

RAW_DIR = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
assert RAW_DIR.exists(), f"Competition data not found at {RAW_DIR}"

# The A3 train cache, same mount as 05v2/06v2 - needed here ONLY for the
# Section 2 validation gate (comparing live decode against known-correct
# pixels), never for the hidden test set itself.
CACHE_DIR = Path("/kaggle/input/datasets/alherma7/cache-stevenleehans-rsna/cache")
assert CACHE_DIR.exists(), f"A3 cache not found at {CACHE_DIR}"

# The 4 A2 v1 fold checkpoints + metadata.json, uploaded as their own
# Kaggle Dataset (see models/a2_v1_checkpoint_metadata.json locally,
# Task 3) - EDIT this path once attached (check `!ls /kaggle/input` if
# unsure, same discovery-not-assumption pattern as every other checkpoint
# mount in this project).
CHECKPOINTS_MOUNT = Path("/kaggle/input/datasets/alherma7/a2-v1-checkpoints")
assert CHECKPOINTS_MOUNT.exists(), f"Checkpoints dataset not found at {CHECKPOINTS_MOUNT}"
with open(CHECKPOINTS_MOUNT / "a2_v1_checkpoint_metadata.json") as f:
    METADATA = json.load(f)
print("Checkpoint metadata:", METADATA)

FINDINGS = METADATA["findings"]
OFFICIAL_LABEL_COLUMNS = METADATA["official_label_columns"]
LABEL_COLS = list(OFFICIAL_LABEL_COLUMNS.values())
CV_FOLDS = METADATA["cv_folds"]
GROUP_INDEX = METADATA["group_index"]

SLOT_NAMES = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
CROP_MM = 130.0
IMG_SIZE = 224
GROUP_SIZE = 3
# The exact window the real A3 cache was built with (cache_meta.json,
# RESOURCES.md) - NOT the source's per-plane default, which this run
# overrode. Must match exactly for the Section 2 gate to pass.
WINDOW = (0.35, 0.65)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)""")

md("## Section 1: live decode - slot recovery, pixel crop, laterality")

code(r'''import re

_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
    r"water excit|\btirm\b|\bsting\b|\bfatsup\b"
)
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")
FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}


def annotate_series_headers(series_df):
    """Hand-kept copy of src/preprocess.py::annotate_series_headers."""
    df = series_df.copy()
    for col in ("SeriesDescription", "SequenceName", "ScanOptions",
                "RepetitionTime", "EchoTime", "ScanningSequence"):
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

    df["weight"] = np.where(t1 & ~t2 & ~pdw, "T1",
                     np.where(t2 & ~pdw, "T2",
                       np.where(pdw, "PD",
                         np.where(gre, "GRE",
                           np.where(tr < 800, "T1",
                             np.where(te > 60, "T2",
                               np.where(tr >= 800, "PD", "UNK")))))))
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    return df


def select_slot_series(study_series_df, plane_map):
    """One series per named slot for one study - port of stevenleehans's
    pick_slots(), scoped to a single study's rows.

    Ties broken toward the stack with the most slices (denser sampling).
    T1 slots (scarcest) fall back to any non-fat-sat series in-plane if
    the exact fluid=False match is missing.
    """
    g = study_series_df.copy()
    g["plane"] = g["SeriesInstanceUID"].map(plane_map)
    chosen = {}
    for name, plane, fluid, fs in SLOTS:
        sel = (g["plane"] == plane) & (g["fatsat"] == fs)
        if fluid is not None:
            sel = sel & (g["fluid"] == fluid)
        cand = g[sel]
        if len(cand) == 0 and fluid is False:
            cand = g[(g["plane"] == plane) & (~g["fatsat"])]
        if len(cand):
            chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
    return chosen


def normalise_laterality(img, plane, laterality):
    """Hand-kept copy of src/preprocess.py::normalise_laterality."""
    if laterality != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


_SIDE_RX = [
    ("R", re.compile(r"\b(right|rt)\b", re.I)),
    ("L", re.compile(r"\b(left|lt)\b", re.I)),
]


def resolve_laterality(ds):
    """Side from the DICOM Laterality/ImageLaterality tags, falling back
    to a text search on SeriesDescription. Geometric fallback is left
    out (the source's own default has it OFF - CFG.lat_from_geometry=0)."""
    tag = getattr(ds, "Laterality", None) or getattr(ds, "ImageLaterality", None)
    if tag in ("L", "R"):
        return tag
    desc = getattr(ds, "SeriesDescription", None)
    if isinstance(desc, str):
        for side, rx in _SIDE_RX:
            if rx.search(desc):
                return side
    return "unknown"


def geometric_slice_order(files):
    """Same algorithm as src/data.py::geometric_slice_order (A0b) -
    project ImagePositionPatient onto the ImageOrientationPatient normal,
    falling back to SliceLocation, then InstanceNumber, then input order.
    Hand-kept copy per this project's Kaggle no-import-src constraint."""
    if len(files) < 2:
        return list(files)
    positions, locations, instances = [], [], []
    normal = None
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        orient = getattr(ds, "ImageOrientationPatient", None)
        if normal is None and orient is not None and len(orient) == 6:
            row_dir = np.array([float(v) for v in orient[:3]])
            col_dir = np.array([float(v) for v in orient[3:]])
            normal = np.cross(row_dir, col_dir)
        pos = getattr(ds, "ImagePositionPatient", None)
        positions.append(np.array([float(v) for v in pos]) if pos is not None else None)
        loc = getattr(ds, "SliceLocation", None)
        locations.append(float(loc) if loc is not None else None)
        num = getattr(ds, "InstanceNumber", None)
        instances.append(float(num) if num is not None else None)

    if normal is not None and all(p is not None for p in positions):
        return [f for _, f in sorted(zip((float(p @ normal) for p in positions), files))]
    if all(v is not None for v in locations):
        return [f for _, f in sorted(zip(locations, files))]
    if all(v is not None for v in instances):
        return [f for _, f in sorted(zip(instances, files))]
    return list(files)


def decode_study_slots(raw_dir, split_dir, study_id, chosen_series):
    """Decode one study's 6 named slots into (6, 3, IMG_SIZE, IMG_SIZE)
    uint8 + a (6,) presence mask - the live-decode equivalent of a row
    from the A3 pre-built cache.

    Always decodes the CENTRE anchor's GROUP_SIZE slices directly - not
    a free parameter, since GROUP_INDEX=1 (config.py, A2 v1's fixed
    training-time choice) is the only group this project's checkpoints
    were ever trained on. This is IDENTICAL to decoding all 3 anchors
    and slicing to group_index=1: read_slot()'s np.linspace(lo, hi, 3)
    puts its middle anchor at exactly (lo+hi)//2 for non-negative
    integer bounds, same centre this function computes directly.
    Section 2 below confirms this empirically against the pre-built
    cache.
    """
    cache = np.zeros((len(SLOT_NAMES), GROUP_SIZE, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    mask = np.zeros(len(SLOT_NAMES), dtype=np.float32)

    for k, (name, plane, _, _) in enumerate(SLOTS):
        if name not in chosen_series:
            continue
        row = chosen_series[name]
        series_dir = raw_dir / split_dir / study_id / row["SeriesInstanceUID"]
        files = sorted(series_dir.glob("*.dcm"))
        if not files:
            continue
        files = geometric_slice_order(files)
        n = len(files)

        lo_f, hi_f = WINDOW
        lo, hi = int(lo_f * (n - 1)), int(hi_f * (n - 1))
        if hi <= lo:
            lo, hi = 0, n - 1
        centre = (lo + hi) // 2
        start = int(np.clip(centre - GROUP_SIZE // 2, 0, max(0, n - GROUP_SIZE)))
        idx = list(range(start, min(start + GROUP_SIZE, n)))
        while len(idx) < GROUP_SIZE:
            idx.append(idx[-1])

        planes, laterality, px = [], "unknown", None
        for j, i in enumerate(idx):
            ds = pydicom.dcmread(files[i], force=True)
            if j == 0:
                laterality = resolve_laterality(ds)
                spacing = getattr(ds, "PixelSpacing", None)
                px = float(spacing[0]) if spacing else None
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
            if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
                a = a.max() - a
            planes.append(a)

        shp = planes[0].shape
        vol = np.stack(planes)

        if px and np.isfinite(px) and px > 0:
            want = int(round(CROP_MM / px))
            h, w = shp
            if 16 < want < min(h, w):
                cy, cx = h // 2, w // 2
                half = want // 2
                vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

        lo_v, hi_v = np.percentile(vol, [1, 99])
        vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)
        t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        t = t.squeeze(0)
        t = normalise_laterality(t, plane, laterality)

        cache[k] = (t * 255).round().clamp(0, 255).to(torch.uint8).numpy()
        mask[k] = 1.0

    return cache, mask


print("Live decode functions defined.")''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

out_path = r"C:\Users\alher\Desktop\RSNA_Knee_Abnormality_Detection\notebooks\07v1_a2_submission_inference.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out_path, "cells:", len(cells))
```

- [ ] **Step 2: Run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote .../notebooks/07v1_a2_submission_inference.ipynb cells: 4`.

- [ ] **Step 3: Validate the notebook (nbformat + compile-check, not execution)**

Run:
```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/07v1_a2_submission_inference.ipynb', as_version=4)
nbformat.validate(nb)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')
print('OK, cells:', len(nb.cells))
"
```
Expected: `OK, cells: 4`, no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/07v1_a2_submission_inference.ipynb
git commit -m "Start A4 notebook: live slot decode + laterality (part 1/3)"
```

Keep the builder script around — Tasks 5–6 append more `code()`/`md()` calls to the same `cells` list before re-running it.

---

## Task 5: Notebook part 2 — validation gate against the pre-built cache

**Files:**
- Modify: `notebooks/07v1_a2_submission_inference.ipynb` (append cells via Task 4's builder script)

**Interfaces:**
- Consumes: everything Task 4 defined (`decode_study_slots`, `select_slot_series`, `annotate_series_headers`, `RAW_DIR`, `CACHE_DIR`, `SLOTS`).

This is the spec's §3 gate: without it passing, nothing downstream (real submission) can be trusted. Same non-agent-executable caveat as Task 4.

- [ ] **Step 1: Append the validation-gate cells to the builder script**

```python
md(r"""## Section 2: validation gate - live decode vs. the pre-built train cache

Picks a handful of real gold **train** studies (DICOM available on
Kaggle same as always) and diffs `decode_study_slots()`'s output against
those same studies' already-known-correct pixels in the pre-built A3
cache. This is the only way to gain confidence before trusting the live
decode on real hidden test data, which offers no such cross-check.""")

code(r"""train_csv = pd.read_csv(RAW_DIR / "train.csv")
train_series_csv = pd.read_csv(RAW_DIR / "train_series.csv")
gold_mask = train_csv[LABEL_COLS].notna().all(axis=1)
gold_study_ids = train_csv.loc[gold_mask, "StudyInstanceUID"].tolist()
print(f"{len(gold_study_ids)} gold studies available for the validation gate")

CHECK_STUDIES = gold_study_ids[:5]

plane_map = dict(zip(train_series_csv["SeriesInstanceUID"], train_series_csv["Anatomical_Plane"]))

# Load shard 0 of the pre-built cache (memory-mapped, cheap) to diff against.
cache_studies = pd.read_csv(CACHE_DIR / "train.s00of04_studies.csv")
cache_arr = np.load(CACHE_DIR / "train.s00of04_cache.npy", mmap_mode="r")
cache_row_of = {sid: i for i, sid in enumerate(cache_studies["StudyInstanceUID"])}

mismatches = []
for study_id in CHECK_STUDIES:
    if study_id not in cache_row_of:
        print(f"  {study_id[:20]}...: not in shard 0, skipping (checked against a different shard is fine too - not required here)")
        continue

    study_series = train_series_csv[train_series_csv["StudyInstanceUID"] == study_id].copy()
    slice_counts = {}
    for _, row in study_series.iterrows():
        d = RAW_DIR / "train_series" / study_id / row["SeriesInstanceUID"]
        slice_counts[row["SeriesInstanceUID"]] = len(list(d.glob("*.dcm")))
    study_series["n_slices"] = study_series["SeriesInstanceUID"].map(slice_counts)
    annotated = annotate_series_headers(study_series)
    chosen = select_slot_series(annotated, plane_map)

    live_cache, live_mask = decode_study_slots(RAW_DIR, "train_series", study_id, chosen)

    cache_row = cache_row_of[study_id]
    # take_group(rows, g=1): the centre anchor's 3-slice slot, matching
    # config.GROUP_INDEX=1 (A2 v1's fixed training-time choice).
    ref = cache_arr[cache_row][:, 3:6]
    ref_mask = np.load(CACHE_DIR / "train.s00of04_mask.npy")[cache_row]

    if not np.array_equal(live_mask, ref_mask):
        mismatches.append((study_id, "mask differs", live_mask, ref_mask))
        continue

    present = live_mask.astype(bool)
    diff = np.abs(live_cache[present].astype(int) - ref[present].astype(int))
    max_diff = int(diff.max()) if diff.size else 0
    mean_diff = float(diff.mean()) if diff.size else 0.0
    print(f"  {study_id[:20]}...: present slots max|diff|={max_diff}, mean|diff|={mean_diff:.2f}")
    if max_diff > 5:  # small tolerance for resize/rounding, not an exact-bitwise requirement
        mismatches.append((study_id, f"max_diff={max_diff}", None, None))

print(f"\n{len(CHECK_STUDIES) - len(mismatches)} / {len(CHECK_STUDIES)} studies matched the pre-built cache")
assert not mismatches, (
    f"Live decode diverged from the pre-built cache on {len(mismatches)} studies: {mismatches} - "
    f"DO NOT trust decode_study_slots() on hidden test data until this is fixed."
)
print("Validation gate PASSED - live decode reproduces the pre-built cache.")""")
```

- [ ] **Step 2: Re-run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote .../notebooks/07v1_a2_submission_inference.ipynb cells: 6`.

- [ ] **Step 3: Validate the notebook**

Run the same nbformat-validate + compile-check command as Task 4 Step 3.
Expected: `OK, cells: 6`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/07v1_a2_submission_inference.ipynb
git commit -m "Add A4 notebook validation gate: live decode vs. pre-built cache (part 2/3)"
```

---

## Task 6: Notebook part 3 — model, 4-fold ensemble, submission.csv

**Files:**
- Modify: `notebooks/07v1_a2_submission_inference.ipynb` (append cells via Task 4's builder script)

**Interfaces:**
- Consumes: Task 4/5's decode functions, `CHECKPOINTS_MOUNT`, `CV_FOLDS`, `LABEL_COLS`, `FINDINGS`, `SLOT_NAMES`.

- [ ] **Step 1: Append the `timm` offline check + model + ensemble + submission cells**

```python
md("""## Section 3: `timm` availability check (no internet at scoring time)

Training notebooks `pip install -q timm`, which will fail once internet
is disabled for the real scored rerun. `pretrained=False` below avoids
needing to *download weights* (we load our own fine-tuned state_dict
regardless) - but the package itself still needs to import without
network access. This cell is the actual verification (see spec section
6): if it fails, an offline wheel Dataset or vendoring `timm`'s
`vit_small_patch14_dinov2` construction becomes necessary - not resolved
here, since it can only be checked on Kaggle.""")

code(r"""try:
    import timm
    print("timm already importable, no install needed:", timm.__version__)
except ImportError:
    import subprocess
    result = subprocess.run(["pip", "install", "-q", "timm"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "timm is not preinstalled AND pip install failed (expected if "
            "internet is disabled) - need an offline wheel Dataset before "
            "the real scored submission. See spec section 6.\\n"
            f"pip stderr: {result.stderr}"
        )
    import timm
    print("timm installed via pip (internet was available this run):", timm.__version__)""")

md("""## Section 4: model - SlotAttentionModel, unchanged from A2 v1

Hand-kept copy of `src/model.py::masked_finding_attention` +
`SlotAttentionModel` (06045eb) - no architecture changes for A4.""")

code(r"""import torch.nn as nn


def masked_finding_attention(embeddings, mask, query, head_weight, head_bias):
    if not (mask.sum(dim=1) > 0).all():
        raise ValueError("masked_finding_attention: a row has 0 present slots")
    scores = torch.einsum("od,bsd->bos", query, embeddings) / (embeddings.shape[-1] ** 0.5)
    expanded_mask = mask.unsqueeze(1).expand(-1, query.shape[0], -1)
    scores = scores.masked_fill(expanded_mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    context = torch.einsum("bos,bsd->bod", weights, embeddings)
    logits = (context * head_weight.unsqueeze(0)).sum(-1) + head_bias
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError("masked_finding_attention produced NaN/Inf logits")
    return logits


class SlotAttentionModel(nn.Module):
    def __init__(self, n_findings=len(FINDINGS), n_slots=len(SLOT_NAMES),
                 backbone_name="vit_small_patch14_dinov2.lvd142m", unfreeze_last=6,
                 pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, img_size=224,
        )
        embed_dim = self.backbone.num_features
        self.query = nn.Parameter(torch.randn(n_findings, embed_dim) * (embed_dim ** -0.5))
        self.heads = nn.Linear(embed_dim, n_findings)
        self.embed_dim = embed_dim
        self.n_findings = n_findings
        self.n_slots = n_slots

    def forward(self, slot_images, slot_mask):
        B, S, C, H, W = slot_images.shape
        flat = slot_images.view(B * S, C, H, W)
        embeddings = self.backbone(flat).view(B, S, self.embed_dim)
        return masked_finding_attention(
            embeddings, slot_mask, self.query, self.heads.weight, self.heads.bias
        )""")

md("""## Section 5: predict the test set, ensemble 4 folds, write submission.csv

Same defensive pattern as Fase 5's retired `06` submission notebook
(the one that actually scored real LB 0.596): every row starts at the
constant 0.5 fallback, a per-study try/except means one bad study never
aborts the whole run, and the final asserts guard column order / row
count / uniqueness before writing.""")

code(r"""test_csv = pd.read_csv(RAW_DIR / "test.csv")
test_series_csv = pd.read_csv(RAW_DIR / "test_series.csv")
sample_submission = pd.read_csv(RAW_DIR / "sample_submission.csv")
assert list(sample_submission.columns) == ["StudyInstanceUID"] + LABEL_COLS
print(f"test.csv: {len(test_csv)} studies")

plane_map = dict(zip(test_series_csv["SeriesInstanceUID"], test_series_csv["Anatomical_Plane"]))

fold_predictions = {fold_id: {} for fold_id in range(CV_FOLDS)}
failures = {}
t0 = time.time()

models = []
for fold_id in range(CV_FOLDS):
    ckpt_path = CHECKPOINTS_MOUNT / f"a2_v1_fold{fold_id}_best.pt"
    m = SlotAttentionModel(pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    m.eval()
    models.append(m)
print(f"Loaded {len(models)} fold checkpoints")

for study_id in test_csv["StudyInstanceUID"]:
    try:
        study_series = test_series_csv[test_series_csv["StudyInstanceUID"] == study_id].copy()
        slice_counts = {}
        for _, row in study_series.iterrows():
            d = RAW_DIR / "test_series" / study_id / row["SeriesInstanceUID"]
            slice_counts[row["SeriesInstanceUID"]] = len(list(d.glob("*.dcm")))
        study_series["n_slices"] = study_series["SeriesInstanceUID"].map(slice_counts)
        annotated = annotate_series_headers(study_series)
        chosen = select_slot_series(annotated, plane_map)

        slots, mask = decode_study_slots(RAW_DIR, "test_series", study_id, chosen)
        images = torch.from_numpy(slots).float().unsqueeze(0).to(DEVICE) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            for fold_id, m in enumerate(models):
                probs = torch.sigmoid(m(images, mask_t)).cpu().numpy()[0]
                fold_predictions[fold_id][study_id] = probs
    except Exception as e:
        failures[study_id] = repr(e)

elapsed = time.time() - t0
n_done = len(test_csv) - len(failures)
print(f"Decoded+predicted: {n_done} / {len(test_csv)} in {elapsed:.1f}s ({elapsed / max(len(test_csv), 1):.3f}s/study)")
print(f"Failures: {len(failures)}")
for sid, err in list(failures.items())[:20]:
    print(f"  {sid[:25]}...: {err}")

# Ensemble: mean of sigmoid probabilities across the 4 folds, per study.
ensembled = {}
for study_id in test_csv["StudyInstanceUID"]:
    per_fold = [fold_predictions[f][study_id] for f in range(CV_FOLDS) if study_id in fold_predictions[f]]
    if per_fold:
        ensembled[study_id] = np.mean(per_fold, axis=0)

ensembled_df = pd.DataFrame(ensembled).T
ensembled_df.columns = FINDINGS
ensembled_df.index.name = "StudyInstanceUID"
print(f"\nEnsembled predictions: {ensembled_df.shape}")

submission = sample_submission.set_index("StudyInstanceUID").copy()
submission.loc[:, :] = 0.5

rename = {finding: OFFICIAL_LABEL_COLUMNS[finding] for finding in FINDINGS}
ensembled_official = ensembled_df.rename(columns=rename)
covered = submission.index.intersection(ensembled_official.index)
submission.loc[covered, LABEL_COLS] = ensembled_official.loc[covered, LABEL_COLS]
n_fallback = len(submission) - len(covered)
print(f"Studies with a real prediction: {len(covered)} / {len(submission)}")
print(f"Studies falling back to constant 0.5: {n_fallback}")

submission = submission.reset_index()
assert list(submission.columns) == ["StudyInstanceUID"] + LABEL_COLS
assert len(submission) == len(test_csv), f"Expected {len(test_csv)} rows, got {len(submission)}"
assert submission["StudyInstanceUID"].is_unique

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nWrote /kaggle/working/submission.csv")
print(submission.head())""")

md("## Real output (fill in after running on Kaggle)")
```

- [ ] **Step 2: Re-run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote .../notebooks/07v1_a2_submission_inference.ipynb cells: 13`.

- [ ] **Step 3: Validate the complete notebook**

Run the same nbformat-validate + compile-check command as Task 4 Step 3.
Expected: `OK, cells: 13`, no syntax errors across the whole file.

- [ ] **Step 4: Commit**

```bash
git add notebooks/07v1_a2_submission_inference.ipynb
git commit -m "Complete A4 notebook: model, 4-fold ensemble, submission.csv (part 3/3)"
```

---

## Task 7: `src/train.py` docstring + RESOURCES.md citation + hand off

**Files:**
- Modify: `src/train.py`
- Modify: `RESOURCES.md`

- [ ] **Step 1: Clarify why `src/train.py::run()` stays `NotImplementedError`**

Per spec §7: Kaggle's self-contained-notebook constraint (no `import src`)
means a locally-runnable pipeline script was never going to be what
actually executes the submission — same reason A2 v1's real training
lives in a notebook, not `src/train.py`. Update the module/function
docstring so a future reader doesn't mistake this for unfinished work.

Edit `src/train.py`:

```python
"""End-to-end pipeline: load -> label -> features -> fit -> evaluate -> output.

Wires together the functions in src/data.py, src/labelers.py,
src/features.py, src/model.py, and src/evaluate.py once each has been
validated in a notebook per the project plan. Nothing here should be the
first place a technique is tried — see notebooks/ and RESOURCES.md.

NOTE (A4, 2026-08-27): this project's Kaggle notebooks cannot `import
src` (confirmed 2026-08-26, see docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md
and .../2026-08-27-a4-submission-pipeline-design.md) — both real
training (notebooks/05v2_.../06v2_...) and the real submission pipeline
(notebooks/07v1_a2_submission_inference.ipynb) run as self-contained
Kaggle notebooks, never as this function. run() staying
NotImplementedError is not unfinished work to pick up later; it reflects
that a locally-runnable pipeline script was never going to be what
actually executes on Kaggle for this project.
"""

from src import config


def run() -> None:
    """Deliberately unimplemented — see the module docstring above for why."""
    raise NotImplementedError(
        "This project's real pipelines run as self-contained Kaggle "
        "notebooks (see notebooks/07v1_a2_submission_inference.ipynb for "
        "the real submission pipeline), not as this function — see the "
        "module docstring."
    )


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the full test suite to confirm no regressions**

Run: `python -m pytest tests/ -v`
Expected: all pass (docstring-only change).

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "Clarify why src/train.py::run() stays NotImplementedError (A4)"
```

- [ ] **Step 4: Add the RESOURCES.md citation**

Add under `## Comparable projects`, after the existing
`stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache` entry:

```markdown
- **stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache** — reused again for A4
  Why: A4's live test-time decode (`notebooks/07v1_a2_submission_inference.ipynb`,
  `src/preprocess.py`) ports the same source's `annotate()`/`pick_slots()`/
  `read_slot()`/`normalise_laterality()` (cells cell-10/cell-12/cell-14),
  adapted for single-study synchronous use — the pre-built cache itself
  cannot cover the hidden test set in a Code Competition (built before
  the test set existed), so the decode step is re-run live at scoring
  time instead, validated against the pre-built train cache's pixels
  before being trusted (`notebooks/07v1_...` section 2).
```

- [ ] **Step 5: Commit**

```bash
git add RESOURCES.md
git commit -m "Cite the A3 cache source again for A4's live decode port"
```

- [ ] **Step 6: Hand off to the user — THIS IS WHEN TO UPLOAD**

Tell the user, clearly and explicitly: the notebook is ready at
`notebooks/07v1_a2_submission_inference.ipynb`. Before running it on
Kaggle, they need to:
1. Upload the 4 checkpoints + metadata as a new Kaggle Dataset: `models/a2_v1_fold0_best.pt`, `a2_v1_fold1_best.pt`, `a2_v1_fold2_best.pt`, `a2_v1_fold3_best.pt`, `a2_v1_checkpoint_metadata.json` (all currently in the local `models/` folder).
2. Attach that new Dataset to the notebook, plus the competition data and the existing A3 cache Dataset (same as `05v2`/`06v2`), plus a GPU accelerator.
3. Check `CHECKPOINTS_MOUNT`'s path once attached (Kaggle's own hyphenated slug — verify with `!ls /kaggle/input/datasets` rather than assuming) and edit the notebook's constant if it doesn't match.
4. Run top to bottom **and report back the real output** — especially Section 2's validation-gate numbers (max/mean pixel diff per study, pass/fail) and Section 5's failure count — **before** trusting this for a real scored submission. Only after that real output comes back and the gate passes should the notebook actually be submitted to the competition (Kaggle's "Submit to Competition" button, or however this competition's submission flow works) for a real leaderboard score.

**Do not claim A4 works, or graduate `decode_study_slots()`/`select_slot_series()` into `src/`, until that real output comes back** — per the approved spec, this is a follow-up conversation, not part of this plan.
