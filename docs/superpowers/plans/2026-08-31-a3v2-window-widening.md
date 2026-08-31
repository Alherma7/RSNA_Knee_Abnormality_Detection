# A3 v2: Widened Slice-Sampling Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate (on real Kaggle data) a rebuilt A3 slot-attention pixel cache that widens the per-plane slice-sampling window (via the vendored reference kernel's own already-implemented `PLANE_WINDOW` defaults, currently overridden by a pinned `RSNA_WINDOW=0.35,0.65`), then run a fold-0 pilot retrain of A2 v2's architecture on it to test whether this improves the weak-finding cluster, per the approved spec's gate.

**Architecture:** No model or reshape-code changes — A2 v2's `SlotCacheDataset(expand_groups=True)` / `SlotAttentionModel` / `masked_finding_attention` are shape-only and have zero dependency on which physical slice indices the cache's pixels came from (verified by the spec's Opus review directly against `src/features.py`/`src/dataset.py`/`src/model.py`). The only real change is upstream: the cache-build step selects different slice indices per plane. A second, independent change bundled into the same pilot notebook: checkpoint selection reports both a best-epoch readout (apples-to-apples with the A2 v2 baseline) and an SWA-averaged readout (this project's first adoption of [[feedback-checkpoint-selection-noise]]'s fix).

**Tech Stack:** Python, pandas/numpy, PyTorch + timm (Kaggle GPU only for Task 2; Task 1 is CPU-only), pydicom, scikit-learn (`GroupKFold`, `roc_auc_score`), `nbformat` (notebook build/validate, local).

**Spec:** `docs/superpowers/specs/2026-08-31-a3v2-window-widening-design.md` (approved after one Opus review pass, 3 Critical + 8 Important + 7 Minor findings all addressed in the spec text — user's explicit call to skip a second confirmation round).

## Global Constraints

- **Scope: window only, at the existing 224px/130mm crop.** The 224px→336px resolution fix is a separate, later, conditional decision — not part of this plan (spec §5).
- **No `src/` changes in this plan.** The cache's array shape is unchanged, so `select_group()`/`expand_slot_groups()`/`SlotCacheDataset`/`SlotAttentionModel` need no edits (spec §1).
- **Kaggle notebooks in this project cannot `import src`** — both notebooks are fully self-contained; every function shared with `src/`'s existing version is a hand-kept copy or a verbatim copy from the vendored reference kernel, not an import.
- **This is this project's first-ever execution of `build_pixel_cache.py`** (spec §2.1, first-review finding C3) — the current cache is `stevenleehans`'s own separately-published Kaggle Dataset Output. Task 1's fingerprint-diff step exists specifically to catch any silent difference beyond the intended window change.
- Cache-build env vars: `RSNA_CROP_MM=130`, `RSNA_N_GROUP_MAX=3`, `RSNA_IMG` left unset (defaults to 224) — **`RSNA_WINDOW` deliberately left unset**, the entire intended change (spec §2.1).
- Training hyperparameters unchanged from A2 v2: `AdamW`, `lr_backbone=8e-6`, `lr_head=1e-3`, `weight_decay=0.02`, last 6 transformer blocks unfrozen, `epochs=12`, `OneCycleLR`, `MICRO_BATCH=8`/`ACCUMULATE_STEPS=4` (A2 v2's own measured-safe VRAM split, re-verified by this pilot's own pre-flight, not assumed).
- **Checkpoint selection: report both readouts from one run** (spec §3, first-review findings C1/C2) — best-epoch tracking (gates against the A2 v2 baseline, apples-to-apples) **and** `swa_epochs=3` weight-averaging of the last 3 of 12 epochs (this project's first adoption of SWA; its own gold-AUC readout must be computed manually, since the reference kernel's own SWA branch only ever scores derived/weak labels, never gold).
- **Gate baseline:** A2 v2's real fold-0 result, **0.7956**, an 11-finding macro (`oa_lateral_compartment` undefined that fold). `gold_tol=0.03`. Directional read on `medial_meniscus_tear` (baseline 0.597) / `lateral_meniscus_tear` (baseline 0.652), bar ~0.10. **Pre-agreed decision rule (spec §4, first-review finding I4):** macro passes **and at least one** of the two meniscus findings clears the bar **and** `mcl_injury`'s delta is reported as context → scale to 4 folds. `mcl_injury` itself does not gate (spec §4's corrected reasoning — its A2 v2 fold-0 baseline is 0.800, not the 0.933 A2 v2's own spec mis-quoted).
- Seed: `09v1` (A2 v2's own pilot) was never seeded — pin `SEED=2026` in this new notebook and document the comparison as seeded-vs-unseeded, an accepted limitation (spec §3, M4).
- Notebook filenames: `notebooks/13v1_a3v2_window_cache_build.ipynb` (Task 1), `notebooks/14v1_a2v3_window_only_fold0_pilot.ipynb` (Task 2).

---

## Task 1: Cache rebuild notebook (CPU-only)

**Files:**
- Create: `notebooks/13v1_a3v2_window_cache_build.ipynb` (built via a Python builder script, same technique as every prior notebook in this project)

**Interfaces:**
- Consumes: the vendored reference kernel `data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb` (cells 10, 11, 12, 13, 15 are copied **verbatim**, unedited — this task's cells are a mechanical assembly, not a rewrite. The exact source text for each is reproduced below so the builder script is self-contained and no separate file lookup is needed at implementation time), the competition's `train_series.csv`/`test_series.csv`/DICOM tree (Kaggle-only), the current cache's `cache_meta.json` (from the currently-attached `alherma7/cache-stevenleehans-rsna` Dataset, for the fingerprint diff).
- Produces: a new Kaggle Dataset (train + test shards + `cache_meta.json`), uploaded by the user after this notebook runs — Task 2 consumes its `CACHE_DIR` path (user-provided once the Dataset exists, same "upload it yourself, edit the path" convention as every prior cache/checkpoint in this project).

This notebook **cannot be executed by an agent** (CPU-only is fine, but it needs the full DICOM tree and ~8-11h of wall-clock, Kaggle-only) — its "test" is that the file is valid, parseable JSON (`nbformat.validate`) and every code cell compiles (`compile(cell.source, ..., "exec")`), same as every prior Kaggle-only notebook in this project.

- [ ] **Step 1: Write the notebook builder script**

Create a scratch builder script (session scratchpad, e.g. `build_13v1.py`):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# 13v1 - A3 v2: widened slice-sampling window, cache rebuild

**Plan item:** the "`RSNA_WINDOW` widening" lead. Full design, evidence,
and the gate the fold-0 pilot (`14v1`) is measured against:
`docs/superpowers/specs/2026-08-31-a3v2-window-widening-design.md`
(approved after one Opus review pass).

**This is this project's first-ever execution of the vendored
`build_pixel_cache.py` script** (cells below are copied verbatim from
`data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`,
cells 10/11/12/13/15/25 - unedited except the env-var cell, which
deliberately omits `RSNA_WINDOW`). The current cache is
`stevenleehans`'s own separately-published Kaggle Dataset Output, not
something this project built before - so this notebook's own Step 0
(below) diffs the new build's fingerprint against the *current* cache's
recorded one, field by field, before spending any real decode time.

**CPU-only, deliberate** (same as A3's own original build) - runs during
a GPU-quota lockout if needed. Real cost, per the source's own measured
table: ~64 min CPU-only for the full corpus at the *current* 224px/130mm
config; window width changes which slice indices get read, not how many,
so this run's cost should land close to that same figure (checked in
Step 2 below, not assumed).

**Needs on Kaggle:** the competition data attached (train_series.csv,
test_series.csv, the full DICOM tree), the *current* cache Dataset
attached (for the fingerprint-diff step only - not read for its
pixels), internet OFF is fine (no model download needed, CPU-only).""")

code(r'''# Cell 10 of the reference kernel, VERBATIM. Real top-level imports and
# the T0/log() helper every function below relies on.
"""RSNA knee: study-level 12-label pipeline.

Synthesised from two community notebooks, with the parts that were wrong or missing in
both replaced. What is inherited, and from where:

  from `rsna-knee-baseline-v1`   header-recovered slot scheme, constant-physical-scale
                                 sampling, laterality normalisation, uint8 cache,
                                 per-diagnosis slot attention, report-hash grouping
  from `rsna-knee-eda-to-2-5d`   protocol-only features as a legal test-time signal

What is new here is listed in `README.md`; the five that change results are grouped
K-fold instead of a single holdout, gold studies held out of their own fold so the
annotated reference is honest, anatomy-preserving augmentation, a backbone that refuses
to train from random initialisation silently, and rank fusion of the folds and the
protocol model.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import gc
import hashlib
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F


T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)''')

md("""## Env vars - the intended change (spec section 2.1)

Same crop/group config as the current cache. `RSNA_WINDOW` is
deliberately **not set** - `read_slot()` (below) falls through to its
own `PLANE_WINDOW` per-plane defaults instead of the narrow
`0.35,0.65` pin the current cache was built with. `RSNA_IMG` is also
left unset (defaults to 224) - resolution stays out of scope for this
plan (spec section 5).""")

code(r'''os.environ["RSNA_CROP_MM"] = "130"
os.environ["RSNA_N_GROUP_MAX"] = "3"
# RSNA_WINDOW intentionally NOT set - this is the entire change vs. the
# current cache. Do not add it back.
os.environ.pop("RSNA_WINDOW", None)  # defensive: guard against Kaggle env leakage

for k, v in os.environ.items():
    if k.startswith("RSNA_"):
        print(f"{k} = {v}")
print("RSNA_WINDOW:", os.environ.get("RSNA_WINDOW", "(not set - PLANE_WINDOW defaults apply)"))''')

md("## `find_root()` - cell 12 of the reference kernel, VERBATIM (the `find_dinov2` half is unused here, omitted)")

code(r'''def find_root(explicit=None):
    if explicit is not None:
        p = Path(explicit)
        if (p / "test.csv").is_file():
            return p
        raise FileNotFoundError(f"no test.csv under {p}")
    for c in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/rsna-knee-abnormality-detection"),
              Path("data"), Path(".")]:
        if (c / "test.csv").is_file() and (c / "test_series").is_dir():
            return c
    base = Path("/kaggle/input")
    if base.is_dir():
        # last resort: two-level scan, because the mount is sometimes nested one deeper
        for depth1 in sorted(p for p in base.iterdir() if p.is_dir()):
            for cand in [depth1] + sorted(p for p in depth1.iterdir() if p.is_dir()):
                if (cand / "test.csv").is_file():
                    return cand
    raise FileNotFoundError("competition mount not found")''')

md("## `CFG` / `SLOTS` - cell 11 of the reference kernel, VERBATIM")

code(r'''class CFG:
    seed = 2026               # folds, target construction, text distillation

    train_seed = int(os.environ.get("RSNA_TRAIN_SEED", "2026"))

    img = int(os.environ.get("RSNA_IMG", "224"))

    crop_mm = float(os.environ.get("RSNA_CROP_MM", "160"))

    pad_short_fov = os.environ.get("RSNA_PAD_SHORT_FOV", "0") == "1"

    swa_epochs = int(os.environ.get("RSNA_SWA_EPOCHS", "0"))

    lat_from_geometry = os.environ.get("RSNA_LAT_GEOMETRY", "0") == "1"
    group = 3                 # slices per encoder input, stacked as the three channels
    n_group_max = int(os.environ.get("RSNA_N_GROUP_MAX", "3"))
    cache_budget_gb = 12.0
    ram_fraction = 0.45       # ceiling on the cache as a share of free RAM

    hdr_threads = 16
    pix_threads = 12

    n_folds = 5
    max_folds_to_run = int(os.environ.get("RSNA_FOLDS", "5"))

    epochs = int(os.environ.get("RSNA_EPOCHS", "12"))
    cycle_epochs = int(os.environ.get("RSNA_CYCLE_EPOCHS", "0"))
    batch_studies = 8
    lr_backbone = 8e-6
    lr_head = 1e-3
    weight_decay = 0.02
    unfreeze_last = 6
    eval_batch = 12
    time_budget = float(os.environ.get("RSNA_TIME_BUDGET", 8.0 * 3600))

    w_text = 0.35
    w_protocol = 0.10
    gold_weight = 3.0


SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]

SLOT_SCHEME = os.environ.get("SLOT_SCHEME", "recovered")
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == "public" else SLOTS_RECOVERED
N_SLOT = len(SLOTS)

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
                        r"water excit|\btirm\b|\bsting\b|\bfatsup\b")
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")


def seed_all(seed=CFG.seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


print(f"CFG.img={CFG.img}, CFG.crop_mm={CFG.crop_mm}, CFG.n_group_max={CFG.n_group_max}, "
      f"SLOT_SCHEME={SLOT_SCHEME!r}, N_SLOT={N_SLOT}")
assert CFG.img == 224, f"expected img=224 (unchanged, resolution out of scope), got {CFG.img}"
assert CFG.crop_mm == 130.0, f"expected crop_mm=130 (unchanged), got {CFG.crop_mm}"
assert CFG.n_group_max == 3, f"expected n_group_max=3 (unchanged), got {CFG.n_group_max}"
assert SLOT_SCHEME == "recovered", f"expected the recovered (6-slot) scheme, got {SLOT_SCHEME!r}"''')

md("## Header pass + slot picking - cell 13 of the reference kernel, VERBATIM")

code(r'''HDR_TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
            "RepetitionTime", "EchoTime", "Laterality", "PixelSpacing", "Rows",
            "Columns", "RescaleSlope", "RescaleIntercept",
            "ImageLaterality", "StudyDescription", "BodyPartExamined",
            "PatientPosition", "ImagePositionPatient"]


def probe(item):
    split, study, series, path = item
    row = {"split": split, "StudyInstanceUID": study, "SeriesInstanceUID": series,
           "dir": path}
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]),
                             stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join(str(x) for x in v)
            else:
                row[t] = str(v)
    except Exception as exc:
        row["err"] = str(exc)[:120]
    return row


def walk(root, split):
    base = Path(root) / split
    items = []
    if not base.is_dir():
        return pd.DataFrame()
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=CFG.hdr_threads) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df):
    """Recover fat suppression and pulse-sequence weighting from the header."""
    if df.empty:
        return df
    for t in HDR_TAGS:
        if t not in df.columns:
            df[t] = None

    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
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
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce")
    return df


def pick_slots(series_df, plane_map):
    """One series per slot per study."""
    if series_df.empty:
        return {}
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
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
        out[study] = chosen
    return out''')

md("""## Slice ordering, sampling, and cache build - cell 15 of the reference
kernel, VERBATIM. `PLANE_WINDOW` here (not read yet - `read_slot()`
checks `RSNA_WINDOW` first, unset in this run) is the whole point of
this notebook.""")

code(r'''ORDER_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "SliceLocation",
              "InstanceNumber"]

import threading

ORDER_STATS = {"geometry": 0, "slice_location": 0, "instance_number": 0, "filename": 0}
_ORDER_LOCK = threading.Lock()

PIX_STATS = {"crop_applied": 0, "crop_padded": 0, "crop_skipped_short_fov": 0,
             "crop_no_spacing": 0, "decode_failed": 0, "mono1_inverted": 0,
             "slices_read": 0}
_PIX_LOCK = threading.Lock()


def _bump(key):
    with _ORDER_LOCK:
        ORDER_STATS[key] += 1


def _bump_pix(key, n=1):
    with _PIX_LOCK:
        PIX_STATS[key] += n


def order_slices(directory, files):
    """Return `files` sorted by position along the stack, nearest-first."""
    if len(files) < 2:
        return list(files)

    positions, locations, instances = [], [], []
    normal = None
    for name in files:
        try:
            ds = pydicom.dcmread(os.path.join(directory, name), stop_before_pixels=True,
                                 force=True, specific_tags=ORDER_TAGS)
        except Exception:
            positions.append(None)
            locations.append(None)
            instances.append(None)
            continue
        pos = getattr(ds, "ImagePositionPatient", None)
        orient = getattr(ds, "ImageOrientationPatient", None)
        if normal is None and orient is not None and len(orient) == 6:
            try:
                row_dir = np.array([float(v) for v in orient[:3]])
                col_dir = np.array([float(v) for v in orient[3:]])
                normal = np.cross(row_dir, col_dir)
            except Exception:
                normal = None
        try:
            positions.append(np.array([float(v) for v in pos]) if pos is not None else None)
        except Exception:
            positions.append(None)
        loc = getattr(ds, "SliceLocation", None)
        locations.append(float(loc) if loc is not None else None)
        num = getattr(ds, "InstanceNumber", None)
        instances.append(float(num) if num is not None else None)

    def sorted_by(values, stat):
        _bump(stat)
        return [f for _, f in sorted(zip(values, files), key=lambda pair: pair[0])]

    if normal is not None and all(p is not None for p in positions):
        return sorted_by([float(p @ normal) for p in positions], "geometry")
    if all(loc is not None for loc in locations):
        return sorted_by(locations, "slice_location")
    if all(num is not None for num in instances):
        return sorted_by(instances, "instance_number")
    _bump("filename")
    return list(files)


PLANE_WINDOW = {"Sagittal": (0.10, 0.90), "Axial": (0.10, 0.90),
                "Coronal": (0.15, 0.85)}


def read_slot(rec, n_slice, out_size, plane=None, group=None):
    """`n_slice` slices from one series, physically ordered, at `out_size` pixels."""
    files, d, px = rec["files"], rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None
    group = CFG.group if group is None else group

    files = order_slices(d, files)

    lo_f, hi_f = PLANE_WINDOW.get(plane, (0.10, 0.90))
    if os.environ.get("RSNA_WINDOW"):
        lo_f, hi_f = (float(x) for x in os.environ["RSNA_WINDOW"].split(","))
    lo, hi = int(lo_f * (n - 1)), int(hi_f * (n - 1))
    if hi <= lo:
        lo, hi = 0, n - 1

    n_anchor = max(1, n_slice // group)
    anchors = (np.linspace(lo, hi, n_anchor).astype(int) if n_anchor > 1
               else np.array([(lo + hi) // 2]))

    idx = []
    for centre in anchors:
        start = int(np.clip(centre - group // 2, 0, max(0, n - group)))
        idx.extend(range(start, min(start + group, n)))
    while len(idx) < n_slice:
        idx.append(idx[-1])

    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
            if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
                a = a.max() - a
                _bump_pix("mono1_inverted")
            _bump_pix("slices_read")
        except Exception:
            a = None
            _bump_pix("decode_failed")
        planes.append(a)

    shp = next((p.shape for p in planes if p is not None), None)
    if shp is None:
        return None
    planes = [p if (p is not None and p.shape == shp) else np.zeros(shp, np.float32)
              for p in planes]
    vol = np.stack(planes)

    if px and np.isfinite(px) and px > 0:
        want = int(round(CFG.crop_mm / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            _bump_pix("crop_applied")
        elif want >= min(h, w):
            if CFG.pad_short_fov:
                py, pxd = max(0, want - h), max(0, want - w)
                vol = np.pad(vol, ((0, 0), (py // 2, py - py // 2),
                                   (pxd // 2, pxd - pxd // 2)))
                _bump_pix("crop_padded")
            else:
                _bump_pix("crop_skipped_short_fov")
        else:
            _bump_pix("crop_skipped_short_fov")
    else:
        _bump_pix("crop_no_spacing")

    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)

    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


_SIDE_RX = [
    ("R", re.compile(r"\b(right|rt|r_?knee|knee_?r|dexter|sağ|derech[ao]|rechts?|"
                     r"droite?|δεξ\w*)\b", re.I)),
    ("L", re.compile(r"\b(left|lt|l_?knee|knee_?l|sinister|sol|izquierd[ao]|links?|"
                     r"gauche|αριστερ\w*)\b", re.I)),
]

LAT_STATS = {"tag": 0, "image_laterality": 0, "text": 0, "geometry": 0, "unknown": 0,
             "geom_agree": 0, "geom_disagree": 0}


def _lat_from_text(*fields):
    blob = " ".join(str(f) for f in fields if f and str(f).lower() != "nan")
    blob = re.sub(r"[^\w\s]", " ", blob)
    hits = {side for side, rx in _SIDE_RX if rx.search(blob)}
    return hits.pop() if len(hits) == 1 else None


def _lat_from_geometry(ipp):
    try:
        x = float(str(ipp).split("|")[0])
    except (TypeError, ValueError):
        return None
    if abs(x) < 20.0:
        return None
    return "R" if x < 0 else "L"


def resolve_laterality(g):
    vals = [str(x).strip().upper() for x in g.get("Laterality", pd.Series(dtype=object))
            .dropna()]
    vals = [v[0] for v in vals if v and v[0] in ("L", "R")]

    geom = next((s for s in (_lat_from_geometry(v)
                             for v in g.get("ImagePositionPatient",
                                            pd.Series(dtype=object)).dropna()) if s), None)
    if vals and geom:
        _bump_lat("geom_agree" if geom == vals[0] else "geom_disagree")
    if vals:
        _bump_lat("tag")
        return vals[0]

    ivals = [str(x).strip().upper() for x in
             g.get("ImageLaterality", pd.Series(dtype=object)).dropna()]
    ivals = [v[0] for v in ivals if v and v[0] in ("L", "R")]
    if ivals:
        _bump_lat("image_laterality")
        return ivals[0]

    txt = _lat_from_text(*g.get("SeriesDescription", pd.Series(dtype=object)).tolist(),
                         *g.get("StudyDescription", pd.Series(dtype=object)).tolist(),
                         *g.get("BodyPartExamined", pd.Series(dtype=object)).tolist())
    if txt:
        _bump_lat("text")
        return txt

    if geom and CFG.lat_from_geometry:
        _bump_lat("geometry")
        return geom

    _bump_lat("unknown")
    return None


def _bump_lat(key):
    LAT_STATS[key] += 1


def normalise_laterality(img, plane, lat):
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


def available_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return None


def plan_cache(n_study):
    budget = CFG.cache_budget_gb
    free = available_ram_gb()
    if free is not None:
        safe = CFG.ram_fraction * free
        if safe < budget:
            log(f"cache budget trimmed {budget:.1f} -> {safe:.1f} GB "
                f"({free:.1f} GB free, keeping {1 - CFG.ram_fraction:.0%} headroom)")
            budget = safe
    per_slice = n_study * N_SLOT * CFG.img * CFG.img
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(CFG.n_group_max, afford // CFG.group))
    if groups < CFG.n_group_max:
        log(f"cache budget {budget:.1f} GB allows {groups} group(s) of "
            f"{CFG.group}, not {CFG.n_group_max}")
    return groups


def build_cache(slot_map, plane_map, lat_map, tag, n_group):
    cache_slices = CFG.group * n_group
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, cache_slices, CFG.img, CFG.img), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f"{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB")

    jobs = [(st, k, plane, slot_map[st][name])
            for st in studies
            for k, (name, plane, _, _) in enumerate(SLOTS)
            if name in slot_map[st]]
    log(f"{tag}: decoding {len(jobs)} slot-series")

    ORDER_STATS.update({k: 0 for k in ORDER_STATS})
    PIX_STATS.update({k: 0 for k in PIX_STATS})

    chunk = 512
    done = 0
    with ThreadPoolExecutor(max_workers=CFG.pix_threads) as pool:
        for c0 in range(0, len(jobs), chunk):
            block = jobs[c0:c0 + chunk]
            imgs = pool.map(lambda j: read_slot(j[3], cache_slices, CFG.img, j[2]), block)
            for (st, k, plane, _), img in zip(block, imgs):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(img, plane,
                                                          lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < chunk:
                log(f"  {tag} {done}/{len(jobs)}")
            if time.time() - T0 > CFG.time_budget:
                log(f"  {tag}: time budget reached during decode")
                break
    total = sum(ORDER_STATS.values()) or 1
    log(f"{tag}: slice ordering " + ", ".join(
        f"{k} {v / total:.1%}" for k, v in ORDER_STATS.items() if v))
    if ORDER_STATS["filename"] / total > 0.05:
        warnings.warn(
            f"{ORDER_STATS['filename'] / total:.0%} of series fell back to filename "
            "order, which is SOP UID order and therefore random. Those slots carry no "
            "depth structure.", RuntimeWarning, stacklevel=2)

    n_crop = sum(PIX_STATS[k] for k in
                 ("crop_applied", "crop_padded", "crop_skipped_short_fov",
                  "crop_no_spacing")) or 1
    log(f"{tag}: physical scale " + ", ".join(
        f"{k.replace('crop_', '')} {PIX_STATS[k]} ({PIX_STATS[k] / n_crop:.1%})"
        for k in ("crop_applied", "crop_padded", "crop_skipped_short_fov",
                  "crop_no_spacing") if PIX_STATS[k]))
    log(f"{tag}: slices read {PIX_STATS['slices_read']}, "
        f"decode failures {PIX_STATS['decode_failed']}, "
        f"MONOCHROME1 inverted {PIX_STATS['mono1_inverted']}")
    skipped = PIX_STATS["crop_skipped_short_fov"] / n_crop
    if skipped > 0.05:
        warnings.warn(
            f"{skipped:.0%} of slots had a field of view at or below crop_mm="
            f"{CFG.crop_mm:.0f}mm, so no physical normalisation was applied to them and "
            "their mm/px came from the acquisition. Set RSNA_PAD_SHORT_FOV=1 to pad "
            "instead.", RuntimeWarning, stacklevel=2)
    gc.collect()
    return studies, cache, mask''')

md(r"""## Step 0: fingerprint diff against the CURRENT cache (spec section
2.1, first-review finding C3)

Runs the same header pass + slot-picking `main()` does internally, so
`fp` is a real inspectable dict before any decode happens - not parsed
from stdout. Diffs every field except `window` against the *current*
cache's own `cache_meta.json`. Any unexplained difference is a real
confound (spec's own explicit instruction: "resolve or explicitly
accept and document" before trusting section 4's comparison).

**EDIT** `CURRENT_CACHE_META` below if the currently-attached cache
Dataset mounts somewhere other than the path already used throughout
this project.""")

code(r'''import json

CURRENT_CACHE_META = Path("/kaggle/input/datasets/alherma7/cache-stevenleehans-rsna/cache/cache_meta.json")
print("CURRENT_CACHE_META:", CURRENT_CACHE_META, "exists:", CURRENT_CACHE_META.exists())
assert CURRENT_CACHE_META.exists(), "current cache's cache_meta.json not found - check the Dataset is attached"

root = find_root()
log(f"input root: {root}")
seed_all()

train_series = pd.read_csv(root / "train_series.csv")
test_series = pd.read_csv(root / "test_series.csv")
both = pd.concat([train_series, test_series])
plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

log("header pass: train")
htr = annotate(walk(root, "train_series"))
log("header pass: test")
hte = annotate(walk(root, "test_series"))
for h in (htr, hte):
    if not h.empty:
        h["plane"] = h["SeriesInstanceUID"].map(plane_map)

slots = {"train": pick_slots(htr, plane_map), "test": pick_slots(hte, plane_map)}

n_group = CFG.n_group_max


def fingerprint(n_group):
    return {
        "img": CFG.img,
        "crop_mm": CFG.crop_mm,
        "pad_short_fov": CFG.pad_short_fov,
        "lat_from_geometry": CFG.lat_from_geometry,
        "group": CFG.group,
        "n_group": int(n_group),
        "window": os.environ.get("RSNA_WINDOW", "default"),
        "slots": [s[0] for s in SLOTS],
        "seed": CFG.seed,
    }


fp = fingerprint(n_group)
print("\nnew build fingerprint:", json.dumps(fp, indent=2, sort_keys=True))

current_meta = json.loads(CURRENT_CACHE_META.read_text())
print("\ncurrent cache fingerprint (relevant fields):",
      json.dumps({k: current_meta.get(k, "<absent>") for k in fp}, indent=2, sort_keys=True))

diff = {k: (current_meta.get(k, "<absent>"), v) for k, v in fp.items()
        if k != "window" and current_meta.get(k, "<absent>") != v}
print("\nUNEXPECTED DIFFERENCES (excluding window, which is meant to differ):")
if diff:
    for k, (old, new) in diff.items():
        print(f"  {k}: current={old!r} vs. new={new!r}")
    print("\n*** STOP and investigate before proceeding - per spec section 2.1, an "
          "unexplained field beyond `window` is a real confound, not something to ignore. ***")
else:
    print("  none - every field matches except window, as expected.")
print(f"\nwindow: current={current_meta.get('window', '<absent>')!r} vs. new={fp['window']!r}")''')

md(r"""## Step 1: probe, corrected acceptance criterion (spec section 2.1,
first-review findings I1/I2)

The reference kernel's own markdown records a 40-study probe measuring
**9.0 slot-series/s** (projecting 0.71h) against a real sustained rate
of **6.25/s** (1.03h) - the probe itself runs ~44% optimistic. So the
right check here is **rate-to-rate** at the same `n_study=40`, not a
wall-clock comparison against the "~55 min" docstring figure (itself
corrected elsewhere in the same source to ~64.2 min CPU-only).""")

code(r'''def shard_name(tag, i, n):
    return tag if n == 1 else f"{tag}.s{i:02d}of{n:02d}"


def shard_studies(slot_map, i, n):
    studies = sorted(slot_map)
    bounds = np.linspace(0, len(studies), n + 1).astype(int)
    mine = studies[bounds[i]:bounds[i + 1]]
    return {s: slot_map[s] for s in mine}


def save_split(out, stem, studies, cache, mask):
    np.save(out / f"{stem}_cache.npy", cache)
    np.save(out / f"{stem}_mask.npy", mask)
    pd.DataFrame({"StudyInstanceUID": studies}).to_csv(
        out / f"{stem}_studies.csv", index=False)
    gb = cache.nbytes / 1024 ** 3
    log(f"{stem}: wrote {cache.shape} = {gb:.2f} GiB, "
           f"slot coverage {mask.mean():.1%}")
    return gb


def estimate_decode_time(slot_map, plane_map, lat_map, n_group, n_study, total_slot_series):
    studies = sorted(slot_map)[:n_study]
    sub = {s: slot_map[s] for s in studies}
    n_series = sum(len(v) for v in sub.values())
    t0 = time.time()
    build_cache(sub, plane_map, lat_map, "probe", n_group)
    dt = time.time() - t0
    rate = n_series / max(dt, 1e-6)
    proj = total_slot_series / max(rate, 1e-9)
    log(f"probe: {n_series} slot-series in {dt:.1f}s = {rate:.1f}/s")
    log(f"probe: {total_slot_series} slot-series projects to "
           f"{proj / 3600:.2f} h for the full split")
    return rate, proj


def lat_of(h):
    return {st: resolve_laterality(g)
            for st, g in h.groupby("StudyInstanceUID")} if not h.empty else {}


lats = {"train": lat_of(htr), "test": lat_of(hte)}
total_train_series = sum(len(v) for v in slots["train"].values())

PROBE_RATE, PROBE_PROJECTION_H = estimate_decode_time(
    slots["train"], plane_map, lats["train"], n_group, 40, total_train_series)

print(f"\nmeasured rate: {PROBE_RATE:.1f} slot-series/s vs. documented baseline 9.0/s "
      f"(the documented baseline was itself later found to run ~44% optimistic vs. a "
      f"real sustained 6.25/s - so a rate anywhere in the 6-10/s range is unremarkable; "
      f"a large deviation outside it is worth understanding before the full run)")
print(f"projected full-corpus decode: {PROBE_PROJECTION_H:.2f}h (train split)")''')

md(r"""## Full build: train + test, 4 shards each (spec section 2.2)

**Only run this cell after reviewing Step 0's fingerprint diff (no
unexplained differences) and Step 1's probe rate (no large, unexplained
deviation).** Same `--shards 4` layout as the current cache - shard
count/size doesn't change with window width. `RSNA_TIME_BUDGET`
defaults to 8h in `CFG`; `main()` internally overrides it to 11h for a
real full-corpus run (see the reference kernel's own `main()` body,
reproduced below) so the decode isn't silently truncated.""")

code(r'''def main(splits="test,train", shards=4, shard="all", out_dir="/kaggle/working/cache"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    CFG.time_budget = float(os.environ.get("RSNA_TIME_BUDGET", 11.0 * 3600))

    # slots/lats/plane_map/root already computed in Step 0's cell above -
    # reused here rather than recomputed, since nothing about them changes
    # between Step 0 and the full build.
    fp_now = fingerprint(n_group)
    log("fingerprint: " + json.dumps(fp_now, sort_keys=True))

    want = list(range(shards)) if shard == "all" else [int(x) for x in str(shard).split(",")]

    total_gb = 0.0
    written = {}
    for tag in splits.split(","):
        tag = tag.strip()
        if not tag or not slots.get(tag):
            log(f"{tag}: no slots, skipped")
            continue
        for i in want:
            stem = shard_name(tag, i, shards)
            if (out / f"{stem}_cache.npy").exists():
                log(f"{stem}: present, skipped")
                continue

            sub = shard_studies(slots[tag], i, shards)
            expect = sum(len(v) for v in sub.values())
            if not expect:
                log(f"{stem}: empty shard (no studies in this block)")
                written[stem] = {"studies": 0, "slot_series": 0}
                continue
            st, C, M = build_cache(sub, plane_map, lats[tag], stem, n_group)

            got = int(M.sum())
            if got < expect:
                raise RuntimeError(
                    f"{stem}: decoded {got} of {expect} slot-series "
                    f"({got / expect:.1%}). This shard is TRUNCATED and has NOT been "
                    "written. Re-run with more shards, or raise RSNA_TIME_BUDGET.")
            total_gb += save_split(out, stem, st, C, M)
            written[stem] = {"studies": len(st), "slot_series": got}
            del C, M

    meta_path = out / "cache_meta.json"
    prior = {}
    if meta_path.exists():
        prior = json.loads(meta_path.read_text())
        clash = {k: (prior.get(k), fp_now[k]) for k in fp_now
                 if k in prior and prior[k] != fp_now[k]}
        if clash:
            raise RuntimeError(f"config changed between shards: {clash}. Start a fresh --out.")
        if prior.get("shards", shards) != shards:
            raise RuntimeError(f"shards changed {prior['shards']} -> {shards}.")
        written = {**prior.get("splits", {}), **written}

    fp_now["shards"] = shards
    fp_now["splits"] = written
    fp_now["built_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fp_now["decode_seconds"] = round(time.time() - t0, 1) + prior.get("decode_seconds", 0)
    meta_path.write_text(json.dumps(fp_now, indent=2, sort_keys=True))

    expect_shards = {shard_name(t.strip(), i, shards)
                     for t in splits.split(",") if slots.get(t.strip())
                     for i in range(shards)}
    missing = sorted(expect_shards - set(written))
    log(f"complete: {len(written)}/{len(expect_shards)} shards"
           if not missing else f"INCOMPLETE, still missing: {', '.join(missing)}")
    log(f"done in {(time.time() - t0) / 60:.1f} min, {total_gb:.2f} GiB written")
    if total_gb > 15:
        log(f"!! {total_gb:.1f} GiB may exceed the Kaggle output limit")
    return written


RESULT = main(splits="test,train", shards=4)
print("\nwritten:", json.dumps(RESULT, indent=2))''')

md(r"""## Real output (fill in after running on Kaggle)

Paste back: Step 0's fingerprint diff output (any unexpected
differences?), Step 1's measured rate vs. the 9.0/s documented
baseline, the full build's `complete: N/N shards` line, and the final
`cache_meta.json` contents (especially `window: "default"`).""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

import json as _json
with open("notebooks/13v1_a3v2_window_cache_build.ipynb", "w", encoding="utf-8") as f:
    _json.dump(nb, f, indent=1)
    f.write("\n")
print("wrote notebooks/13v1_a3v2_window_cache_build.ipynb, cells:", len(cells))
```

Run this from the repo root so the relative output path resolves correctly.

- [ ] **Step 2: Run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote notebooks/13v1_a3v2_window_cache_build.ipynb, cells: 19` (10 markdown + 9 code cells — count each `md()`/`code()` call in the builder script above to verify: 1 intro, then 9 (header, cell)-pairs for imports/env-vars/find_root/CFG-SLOTS/header-pass/slice-ordering-and-build/fingerprint-diff/probe/full-build, plus 1 closing real-output placeholder).

- [ ] **Step 3: Validate the notebook**

Run:
```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/13v1_a3v2_window_cache_build.ipynb', as_version=4)
nbformat.validate(nb)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')
print('OK, cells:', len(nb.cells))
"
```
Expected: `OK, cells: 19`, no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/13v1_a3v2_window_cache_build.ipynb
git commit -m "Build A3 v2 cache-rebuild notebook (widened per-plane window, spec section 2)"
```

---

## Task 2: Fold-0 training pilot notebook (Kaggle GPU)

**Files:**
- Create: `notebooks/14v1_a2v3_window_only_fold0_pilot.ipynb` (built via a Python builder script, mirroring `09v1_a2v2_multigroup_baseline.ipynb`'s own structure — see `docs/superpowers/plans/2026-08-28-a2v2-multigroup-slot-attention.md` for that notebook's full original source, reused here as the base with the changes below layered on)

**Interfaces:**
- Consumes: the new cache Dataset from Task 1 (path provided by the user once uploaded — same "upload it yourself, edit the path" convention as every prior checkpoint/cache in this project), `data/raw/_published_labels/llm_labels_v4_blend.csv` (as the user's attached Kaggle Dataset, unchanged from A2 v2).
- Produces: two checkpoints (`/kaggle/working/a2_v3_fold0_best_epoch.pt`, `/kaggle/working/a2_v3_fold0_swa.pt`), a printed gate decision.

Same non-agent-executable caveat as Task 1's notebook — needs Kaggle GPU + the new cache + the published labels. This task's "test" is structural validation only.

- [ ] **Step 1: Write the notebook builder script**

Create a scratch builder script (e.g. `build_14v1.py`). This notebook reuses `09v1`'s own 22-cell structure (`docs/superpowers/plans/2026-08-28-a2v2-multigroup-slot-attention.md`'s Task 1 Step 1 and Task 2 Step 1 code blocks give that structure's exact `md()`/`code()` call sequence) with these changes, building to **23 cells total**:

1. Copy `09v1`'s Task 1 Step 1 cells **1-8** (intro markdown through the cache-dataset sanity-check cell) verbatim, except: `CACHE_DIR` now points at the Task 1 (`13v1`) output Dataset (path supplied by the user once uploaded — leave a clear `# EDIT this path` comment, same convention as every prior cache attachment); the intro markdown cell's text is replaced with the new intro below (cell count unchanged, still 8 cells).
2. **Insert one new cell (9th)** — the cache-identity-guard code block given below — right after cell 8 (the cache-dataset sanity check), before `09v1`'s `timm` install cell.
3. Copy `09v1`'s Task 2 Step 1 cells **in full** (the `timm` install through the final "Real output" markdown placeholder — cells 9-22 of the original, now cells 10-23), with these substitutions: every `a2_v2_fold0` string becomes `a2_v3_fold0`; the "Full fold-0 training run" code cell is **replaced entirely** by the dual-checkpoint training-loop code block below; the "Final report" code cell is **replaced entirely** by the dual-readout gate code block below; the final "Real output" markdown placeholder's text is replaced with the new placeholder text given at the end of this step.

New intro markdown (replaces `09v1`'s own, cell 1):

```python
md(r"""# 14v1 - A2 v3: window-only fold-0 pilot

**Plan item:** tests whether A3 v2's widened per-plane slice-sampling
window (built by `13v1`, no `src/` changes) improves the weak-finding
cluster on A2 v2's own architecture. Full design, evidence, and the
gate this run is measured against:
docs/superpowers/specs/2026-08-31-a3v2-window-widening-design.md
(approved after one Opus review pass).

Self-contained per this project's Kaggle constraint (no `import src`) -
identical to `09v1`'s own labels/folds/model/eval code (A2 v2's winning
architecture, unchanged) except `CACHE_DIR` points at `13v1`'s new
cache and the checkpoint-selection cells below (dual best-epoch/SWA
readout, per the spec).

**Scope: fold 0 only, window width is the only changed variable** vs.
A2 v2 (`09v1_a2v2_multigroup_baseline.ipynb`) - same hyperparameters,
same labels, same fold split, same 224px/130mm crop.""")
```

New cell 9 (cache-identity guard):

```python
code(r'''# Cache-identity guard (spec section 3, first-review finding I5).
import json as _json

_meta_path = CACHE_DIR / "cache_meta.json"
print("cache_meta.json:", _meta_path, "exists:", _meta_path.exists())
assert _meta_path.exists(), "cache_meta.json not found - check CACHE_DIR points at the Task 1 output"
_meta = _json.loads(_meta_path.read_text())
print("cache fingerprint:", _json.dumps(_meta, indent=2, sort_keys=True))

_EXPECTED = {"window": "default", "crop_mm": 130.0, "img": 224, "group": 3, "n_group": 3}
_mismatch = {k: (v, _meta.get(k)) for k, v in _EXPECTED.items() if _meta.get(k) != v}
assert not _mismatch, (
    f"cache_meta.json doesn't match this notebook's expectations: {_mismatch} - "
    "CACHE_DIR may point at the wrong (or a stale) Dataset. This must be the widened-"
    "window cache from 13v1, not the current narrow-window one."
)
_expected_slots = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
assert _meta.get("slots") == _expected_slots, f"slot order mismatch: {_meta.get('slots')}"
print("cache-identity guard passed - window='default' (widened), crop_mm/img/group/n_group/slots all match")''')
```

Insert this as cell 9, immediately after `09v1`'s cache-dataset sanity-check cell (its cell 8) and before its `timm`-install cell.

The fold-assignment cell is copied **unchanged** from `09v1`'s Task 1 (cell 6 of the original 22 — becomes cell 6 here too, before the new cell 9 insertion point) — it already asserts fold 0's val set against the recorded 1,307/17 split (spec section 3's I6 fix is already satisfied by reusing this cell as-is, no edit needed).

**Replaces `09v1`'s "Full fold-0 training run" code cell** (dual checkpoint tracking: best-epoch, apples-to-apples gate, plus SWA — spec section 3, first-review findings C1/C2/I7/M2/M4):

```python
code(r'''SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
# 09v1 (the A2 v2 baseline this pilot is gated against) was never seeded -
# this is a real, accepted limitation, not silently fixed retroactively
# (spec section 3, M4). Documented here, not hidden.

FOLD = 0
train_idx = np.flatnonzero(label_table["fold"].to_numpy() != FOLD)
val_idx = np.flatnonzero(label_table["fold"].to_numpy() == FOLD)
val_labels = label_table.iloc[val_idx].reset_index()
val_is_gold = val_labels["is_gold"].to_numpy()
print(f"fold {FOLD}: {len(train_idx)} train / {len(val_idx)} val ({val_is_gold.sum()} gold in val)")

BATCH_SIZE = 32
MICRO_BATCH = 8
ACCUMULATE_STEPS = BATCH_SIZE // MICRO_BATCH
assert BATCH_SIZE % MICRO_BATCH == 0, "MICRO_BATCH must evenly divide BATCH_SIZE"
print(f"BATCH_SIZE={BATCH_SIZE}, MICRO_BATCH={MICRO_BATCH}, ACCUMULATE_STEPS={ACCUMULATE_STEPS}")

full_ds = SlotCacheDataset(CACHE_DIR, TRAIN_SHARDS, label_table, expand_groups=True)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(full_ds, train_idx.tolist()), batch_size=MICRO_BATCH,
    shuffle=True, num_workers=2, drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(full_ds, val_idx.tolist()), batch_size=MICRO_BATCH, shuffle=False, num_workers=2,
)

model = SlotAttentionModel().to(DEVICE)
backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone")]
head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
opt = torch.optim.AdamW([
    {"params": backbone_params, "lr": 8e-6},
    {"params": head_params, "lr": 1e-3},
], weight_decay=0.02)

EPOCHS = 12
# SWA_EPOCHS=3: averaging the last 3 of 12 epochs. Under this notebook's
# own OneCycleLR (no pct_start passed -> PyTorch default 0.3, LR peaks at
# epoch ~3.6), epochs 10/11/12 sit at ~28.3%/13.3%/3.5% of max_lr -
# monotonically annealing, well past the schedule's peak, no warmup
# contamination (spec section 3, M2). Not a sourced value - the
# reference kernel exposes RSNA_SWA_EPOCHS but states no chosen number
# anywhere in its own file.
SWA_EPOCHS = 3

steps_per_epoch = len(train_loader) // ACCUMULATE_STEPS
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=[8e-6, 1e-3], total_steps=EPOCHS * steps_per_epoch
)

best_gold_auc = -1.0
best_state = None
swa_state, swa_n = None, 0
for epoch in range(EPOCHS):
    model.train()
    t0 = time.time()
    opt.zero_grad()
    for step, (images, mask, labels) in enumerate(train_loader):
        images, mask, labels = images.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        loss = F.binary_cross_entropy_with_logits(model(images, mask), labels) / ACCUMULATE_STEPS
        loss.backward()
        if (step + 1) % ACCUMULATE_STEPS == 0:
            opt.step()
            opt.zero_grad()
            scheduler.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for images, mask, _ in val_loader:
            images, mask = images.to(DEVICE), mask.to(DEVICE)
            probs.append(torch.sigmoid(model(images, mask)).cpu().numpy())
    val_pred = pd.DataFrame(np.concatenate(probs), columns=FINDINGS)
    gold_auc = macro_roc_auc(val_labels.loc[val_is_gold, FINDINGS], val_pred[val_is_gold])
    print(f"epoch {epoch}: {time.time() - t0:.0f}s, val gold macro-AUC={gold_auc:.4f}")

    if gold_auc > best_gold_auc:
        best_gold_auc = gold_auc
        # .clone() is load-bearing (spec section 3, I7): .cpu() is a
        # no-op on a tensor already on CPU, so without .clone() best_state
        # would alias the live model parameters and the NEXT epoch's
        # training would silently corrupt this saved copy.
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print("  new best (epoch-selected)")

    if epoch >= EPOCHS - SWA_EPOCHS:
        # Same .clone() hazard applies here, for the same reason -
        # explicit per spec section 3, I7.
        sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
        if swa_state is None:
            swa_state, swa_n = sd, 1
        else:
            swa_n += 1
            for k in swa_state:
                swa_state[k] += (sd[k] - swa_state[k]) / swa_n
        print(f"  SWA: accumulated epoch {epoch} (swa_n={swa_n})")

assert best_state is not None and swa_state is not None
torch.save(best_state, "/kaggle/working/a2_v3_fold0_best_epoch.pt")
torch.save(swa_state, "/kaggle/working/a2_v3_fold0_swa.pt")
print(f"\nbest-epoch gold macro-AUC (fold {FOLD}, selection-based): {best_gold_auc:.4f}")
print(f"SWA checkpoint: averaged {swa_n} epochs, no selection - gold AUC computed next cell")''')
```

**Replaces `09v1`'s "Final report" code cell** — evaluates BOTH checkpoints on fold 0's gold subset (spec section 3, first-review finding C2 — the reference kernel's own SWA branch only ever scores derived/weak labels for the averaged model, never gold; no upstream mechanism exists to copy, this is built here):

```python
code(r'''def eval_gold(state_dict, label):
    m = SlotAttentionModel().to(DEVICE)
    m.load_state_dict(state_dict)
    m.eval()
    probs = []
    with torch.no_grad():
        for images, mask, _ in val_loader:
            images, mask = images.to(DEVICE), mask.to(DEVICE)
            probs.append(torch.sigmoid(m(images, mask)).cpu().numpy())
    pred = pd.DataFrame(np.concatenate(probs), columns=FINDINGS)
    per_finding = per_finding_roc_auc(val_labels.loc[val_is_gold, FINDINGS], pred[val_is_gold])
    macro_11 = float(per_finding.drop(index="oa_lateral_compartment").mean())
    print(f"\n{label} gold per-finding AUC:\n{per_finding}")
    print(f"{label} gold macro (11-finding): {macro_11:.4f}")
    del m
    return per_finding, macro_11


gold_per_finding_best, candidate_macro_best = eval_gold(best_state, "best-epoch")
gold_per_finding_swa, candidate_macro_swa = eval_gold(swa_state, "SWA")

print(f"\nselection-noise gap (best-epoch minus SWA): {candidate_macro_best - candidate_macro_swa:+.4f} "
      "- this project's first real, on-our-own-data measurement of this gap "
      "(see feedback_checkpoint_selection_noise.md)")

# ---- spec section 4: fold-0 pilot gate, on the BEST-EPOCH number (apples-to-apples) ----
BASELINE_FOLD0_MACRO_11 = 0.7956  # A2 v2 fold 0, 11-finding mean (09v1's real output)
BASELINE_MEDIAL_MENISCUS = 0.597
BASELINE_LATERAL_MENISCUS = 0.652
GOLD_TOL = 0.03

macro_delta = candidate_macro_best - BASELINE_FOLD0_MACRO_11
macro_ok = macro_delta >= -GOLD_TOL
print(f"\ngold macro (best-epoch, 11-finding) vs. A2 v2 fold-0 baseline {BASELINE_FOLD0_MACRO_11}: "
      f"delta={macro_delta:+.4f}, macro_ok={macro_ok} (tol={GOLD_TOL})")

medial_delta = float(gold_per_finding_best["medial_meniscus_tear"]) - BASELINE_MEDIAL_MENISCUS
lateral_delta = float(gold_per_finding_best["lateral_meniscus_tear"]) - BASELINE_LATERAL_MENISCUS
print(f"\nmedial_meniscus_tear: {gold_per_finding_best['medial_meniscus_tear']:.4f} "
      f"(baseline {BASELINE_MEDIAL_MENISCUS}, delta={medial_delta:+.4f})")
print(f"lateral_meniscus_tear: {gold_per_finding_best['lateral_meniscus_tear']:.4f} "
      f"(baseline {BASELINE_LATERAL_MENISCUS}, delta={lateral_delta:+.4f})")

# mcl_injury: reported as context only, does NOT gate (spec section 4,
# first-review finding I3 - its A2v2 fold-0 baseline is 0.800, and this
# specific change is not provably a no-op for it the way A2v2's own
# group-recombination change was).
BASELINE_MCL = 0.800
mcl_delta = float(gold_per_finding_best["mcl_injury"]) - BASELINE_MCL
print(f"\nmcl_injury (context only, not gating): {gold_per_finding_best['mcl_injury']:.4f} "
      f"(baseline {BASELINE_MCL}, delta={mcl_delta:+.4f})")

# Pre-agreed decision rule (spec section 4, first-review finding I4,
# user's explicit choice) - "at least one" meniscus finding, not "both",
# fixed in advance rather than relitigated after seeing an ambiguous
# result the way A2 v2's own pilot had to be.
at_least_one_real = medial_delta >= 0.10 or lateral_delta >= 0.10
print(f"\nat least one meniscus finding moved >=0.10 (pre-agreed bar): {at_least_one_real}")

print()
if not macro_ok:
    print("DECISION: macro regressed beyond tolerance -> STOP, do not scale to 4 folds.")
elif at_least_one_real:
    print("DECISION: macro OK and at least one meniscus finding moved meaningfully positive "
          "-> SCALE to the remaining 3 folds (pre-agreed rule, spec section 4).")
else:
    print("DECISION: macro OK but neither meniscus finding cleared the noise bar -> STOP, "
          "report as inconclusive, NOT disproved.")

print(f"\nSWA gold macro (11-finding, forward-looking figure): {candidate_macro_swa:.4f}")
print("\nfull gold per-finding AUC (best-epoch) for the record:\n", gold_per_finding_best)
print("\nfull gold per-finding AUC (SWA) for the record:\n", gold_per_finding_swa)''')
```

Replace the final "Real output" markdown placeholder cell (`09v1`'s last cell) with:

```python
md(r"""## Real output (fill in after running on Kaggle)

Paste back: fold-assignment assertion result, cache-identity guard
output (window/crop_mm/img/group/n_group/slots all matched?), pre-flight
loss + peak VRAM, whichever `MICRO_BATCH`/`ACCUMULATE_STEPS` split was
actually used and why, per-epoch val gold macro-AUC, both per-finding
tables (best-epoch and SWA), the selection-noise gap between them, and
the printed `DECISION` line. Per the approved spec, this is what the
user reviews before scaling to 4 folds or touching `src/`.""")
```

- [ ] **Step 2: Run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote notebooks/14v1_a2v3_window_only_fold0_pilot.ipynb, cells: 23` (`09v1`'s own 22 cells plus the one new cache-identity-guard cell inserted at position 9 — no other cell count changes, since every other change replaces a cell's content in place rather than adding or removing one).

- [ ] **Step 3: Validate the notebook**

Run the same nbformat-validate + compile-check command as Task 1 Step 3, adjusted for this filename.
Expected: `OK, cells: 23`, no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/14v1_a2v3_window_only_fold0_pilot.ipynb
git commit -m "Build A2 v3 window-only fold-0 pilot notebook (dual best-epoch/SWA gate, spec section 3-4)"
```

- [ ] **Step 5: Hand off to the user**

Tell the user: `13v1` needs to run first (CPU-only, ~64 min real corpus decode expected, Kaggle) and its output uploaded as a new Kaggle Dataset. Then `14v1` needs that Dataset's path edited into `CACHE_DIR`, plus a GPU, the competition data, and the published-labels Dataset attached, and run top to bottom. Ask them to report back: `13v1`'s fingerprint-diff output and probe rate (any unexplained deviations?), and `14v1`'s full real output — especially the two per-finding tables, the selection-noise gap between best-epoch and SWA, and the final `DECISION` line. **Do not claim the window-widening hypothesis works, scale to the remaining 3 folds, or touch `src/`, until that real output comes back** — per the approved spec, scaling and any future graduation are both follow-up conversations, not part of this plan.
