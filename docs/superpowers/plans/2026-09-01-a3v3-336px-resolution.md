# A3 v3: 336px Resolution Rebuild (Wide Window Folded In) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate (on real Kaggle data) a rebuilt A3 slot-attention pixel cache at 336px resolution with A3 v2's previously-parked wide per-plane window folded in as the default, then run a fold-0 pilot retrain of A2 v2's architecture on it to test whether this improves the weak-finding cluster, per the approved spec's gate.

**Architecture:** No reshape-code changes to `select_group()`/`expand_slot_groups()`/`SlotCacheDataset` — they operate on tensor shape read from the cache array itself. The real changes are: (1) upstream, the cache-build step decodes at `img=336` with `RSNA_WINDOW` unset (carrying forward A3 v2's wide `PLANE_WINDOW` default), split across 8 shards and 2 Kaggle sessions instead of 4 shards/1 session, since the naive single-session output (~25 GiB) exceeds both Kaggle's output cap and the vendored script's own 15 GiB guard; (2) the pilot notebook's hand-duplicated model construction, forward-shape check, and dataset assert all move from a hardcoded `224` to `336` — four concrete literal edits, not a `src/` fix (these notebooks never `import src`); (3) the pilot's `MICRO_BATCH`/`ACCUMULATE_STEPS` split is halved/doubled from A2 v2's own values (empirically re-verified by the pre-flight, not assumed) to fit 336px's larger activation footprint; (4) a new non-gating weak-cluster mean-delta readout, reported alongside the unchanged fold-0 gate.

**Tech Stack:** Python, pandas/numpy, PyTorch + timm (Kaggle GPU only for Task 2; Task 1 is CPU-only), pydicom, scikit-learn (`GroupKFold`, `roc_auc_score`), `nbformat` (notebook build/validate, local).

**Spec:** `docs/superpowers/specs/2026-09-01-a3v3-336px-resolution-design.md` (approved after one Opus review pass, 3 Critical + 8 Important + 8 Minor findings all addressed in the spec text).

## Global Constraints

- **Scope: `img=336` + the wide window bundled together, at the existing `crop_mm=130`.** Both changed from the current production cache at once — a deliberate departure from A3 v2's own confound-avoidance stance, justified in spec §1 (not re-argued here).
- **No `src/` changes in this plan.** Both notebooks are fully self-contained hand-duplicated copies (spec §3/§5/§6) — this project's Kaggle kernels cannot `import src`.
- **Cache-build env vars:** `RSNA_CROP_MM=130` (unchanged), `RSNA_N_GROUP_MAX=3` (unchanged), `RSNA_IMG=336` (new — was unset/224 in every prior cache), `RSNA_WINDOW` **not set** (carries A3 v2's wide `PLANE_WINDOW` default forward, unchanged from `13v1`).
- **Cache build: 8 shards (not 4), across (at least) 2 Kaggle sessions**, using the vendored `build_pixel_cache.py`'s own existing `shards=N`/`shard="i,j,..."`/skip-existing-shard/`cache_meta.json`-merge support (spec §2.2, first-review finding C1) — the naive single-session `--shards 4` output (~25 GiB) exceeds both Kaggle's output cap and the script's own `if total_gb > 15` guard.
- **Fingerprint diff: two expected fields, not one.** Diff the new build against the **original narrow-window 224px cache** (`stevenleehans`'s published Dataset) — `window` and `img` are both expected to differ; every other field must match exactly (spec §2.1).
- **Probe-rate acceptance: an explicit band, not a bare reference number.** Accept 5-10/s, stop and investigate below 5/s (spec §2.1, first-review finding I6) — reused from `13v1`'s own established band, not the raw 9.0/s documented baseline (which is itself ~44% optimistic).
- **Pilot notebook: four literal `224`→`336` edits**, all in the hand-duplicated template, none in `src/` — a cache-identity guard's `_EXPECTED` dict, a `timm.create_model(..., img_size=...)` call, the model's own `forward()` shape check, and a dataset pre-flight shape assert (spec §3.1, first-review finding C2).
- **Shard names read from `cache_meta.json`**, not a hardcoded `s00of04`..`s03of04` list — this cache has 8 train shards (spec §2.2's "downstream consequence" note).
- **VRAM: `MICRO_BATCH=4`/`ACCUMULATE_STEPS=8`** (halved/doubled from A2 v2's own `MICRO_BATCH=8`/`ACCUMULATE_STEPS=4`, keeping effective batch size 32) — the spec's stated hypothesis (§3.3), re-verified by the pre-flight smoke test's real VRAM read before the full training run, not assumed.
- Training hyperparameters otherwise unchanged from A2 v2/A2 v3: `AdamW`, `lr_backbone=8e-6`, `lr_head=1e-3`, `weight_decay=0.02`, last 6 transformer blocks unfrozen, `epochs=12`, `OneCycleLR`, `swa_epochs=3`, `SEED=2026` pinned (comparison against A2 v2's own unseeded `09v1` baseline remains an accepted limitation, spec §4).
- **Gate baseline, unchanged from `14v1`:** A2 v2's real fold-0 result, **0.7956**, 11-finding macro (`oa_lateral_compartment` undefined that fold). `gold_tol=0.03`. Directional read on `medial_meniscus_tear` (baseline 0.597) / `lateral_meniscus_tear` (baseline 0.652), bar ~0.10, **at least one** (not both) must clear it to scale. `mcl_injury` (baseline 0.800) reported as context only, does not gate. **This gate design is locked — not open for revision in this plan** (spec §4, confirmed with the user 2026-09-01).
- **New, non-gating: a weak-cluster mean-delta readout** (spec §4, first-review finding I8) — the mean delta across `mcl_injury`/`medial_meniscus_tear`/`lateral_meniscus_tear`/`oa_lateral_compartment` (wherever both baseline and candidate are numerically defined), reported alongside the gate, does **not** change the decision rule.
- Notebook filenames: `notebooks/15v1_a3v3_336px_cache_build.ipynb` (Task 1), `notebooks/16v1_a2v4_336px_fold0_pilot.ipynb` (Task 2).

---

## Task 1: 336px cache-rebuild notebook (CPU-only, 2-session/8-shard)

**Files:**
- Create: `notebooks/15v1_a3v3_336px_cache_build.ipynb` (built via a Python builder script, same technique as `13v1`)

**Interfaces:**
- Consumes: the vendored reference kernel `data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb` (cells 10/11/12/13/15/25, copied verbatim except the env-var cell and the `CFG` assert — same convention `13v1` already established), the competition's `train_series.csv`/`test_series.csv`/DICOM tree (Kaggle-only), the *original* narrow-window 224px cache's `cache_meta.json` (from the currently-attached `alherma7/cache-stevenleehans-rsna` Dataset, for the fingerprint diff), and — session 2 only — session 1's own uploaded partial output.
- Produces: a new Kaggle Dataset (8 train shards + test shards + `cache_meta.json`), uploaded by the user in two stages after each session — Task 2 consumes its final `CACHE_DIR` path.

This notebook **cannot be executed by an agent** (CPU-only is fine, but it needs the full DICOM tree, ~2x~70min sessions, and manual Dataset upload/re-attachment between them, Kaggle-only) — its "test" is that the file is valid, parseable JSON (`nbformat.validate`) and every code cell compiles (`compile(cell.source, ..., "exec")`), same as every prior Kaggle-only notebook in this project.

- [ ] **Step 1: Write the notebook builder script**

Create a scratch builder script (session scratchpad, e.g. `build_15v1.py`):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# 15v1 - A3 v3: 336px resolution rebuild with the wide window folded in

**Plan item:** the "224px->336px resolution rebuild" lead, with A3 v2's
previously-parked wide per-plane window folded in as the default rather
than re-tested in isolation. Full design, evidence, and the gate the
fold-0 pilot (`16v1`) is measured against:
`docs/superpowers/specs/2026-09-01-a3v3-336px-resolution-design.md`
(approved after one Opus review pass).

**Directly modeled on `13v1`'s own structure** (cells copied verbatim
from `data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`,
cells 10/11/12/13/15/25 - unedited except the env-var cell, which sets
`RSNA_IMG=336` and omits `RSNA_WINDOW`, and the `CFG` assert, which now
checks `img==336`). Diffs the new build's fingerprint against the
*original, narrow-window, 224px* cache before spending any real decode
time - two fields (`window`, `img`) are expected to differ, everything
else must match.

**CPU-only, deliberate** (same as A3/A3 v2's own builds) - runs during a
GPU-quota lockout if needed. Real cost, scaled from `13v1`'s own real
build (61.3 min, 11.13 GiB at 224px): ~2.25x storage/compute at 336px
(`(336/224)^2 = 2.25`) -> **~25 GiB total, split across 8 shards (not 4)
and 2 Kaggle sessions** (~70 min CPU, ~12.5 GiB output, each), since the
naive single-session output would exceed both Kaggle's output cap and
this script's own 15 GiB per-invocation guard (spec section 2.2,
first-review finding C1).

**Needs on Kaggle, each session:** the competition data attached
(train_series.csv, test_series.csv, the full DICOM tree), the *original*
cache Dataset attached (for the fingerprint-diff step only - not read
for its pixels), internet OFF is fine (no model download needed,
CPU-only). **Session 2 additionally needs session 1's uploaded partial
output attached** (see the `SESSION`-variable cell below).""")

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

md("""## Env vars - the intended change (spec section 1/2)

Same crop/group config as every prior cache. `RSNA_IMG=336` is the
resolution change (was unset/224 in every prior cache including
`13v1`). `RSNA_WINDOW` is deliberately **not set** - carries forward
A3 v2's own wide `PLANE_WINDOW` per-plane defaults, not re-derived
here.""")

code(r'''os.environ["RSNA_CROP_MM"] = "130"
os.environ["RSNA_N_GROUP_MAX"] = "3"
os.environ["RSNA_IMG"] = "336"
# RSNA_WINDOW intentionally NOT set - carries A3 v2's wide PLANE_WINDOW
# defaults forward. Do not add it back.
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

md("## `CFG` / `SLOTS` - cell 11 of the reference kernel, VERBATIM except the two asserts marked below")

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
# --- both asserts below changed from 13v1: img now expects 336 (the
# resolution change), crop_mm still expects 130 (unchanged) ---
assert CFG.img == 336, f"expected img=336 (the resolution change), got {CFG.img}"
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
kernel, VERBATIM, unchanged. `img` (336) flows into `read_slot()`/
`build_cache()` purely via `CFG.img`, no logic here depends on its
value beyond that.""")

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

md(r"""## Step 0: fingerprint diff against the ORIGINAL narrow-window
224px cache (spec section 2.1) - two fields expected to differ, not one

Runs the same header pass + slot-picking `main()` does internally, so
`fp` is a real inspectable dict before any decode happens - not parsed
from stdout. This build changes **two** fields at once (`window` and
`img`) - `13v1` only ever expected one, so this diff excludes both from
the "unexpected difference" check. Every other field must match
exactly.

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
print("\noriginal cache fingerprint (relevant fields):",
      json.dumps({k: current_meta.get(k, "<absent>") for k in fp}, indent=2, sort_keys=True))

# Excludes BOTH window and img - both are meant to differ from the
# original cache (spec section 2.1). 13v1's own diff excluded window
# only; this is the one real change from that template.
diff = {k: (current_meta.get(k, "<absent>"), v) for k, v in fp.items()
        if k not in ("window", "img") and current_meta.get(k, "<absent>") != v}
print("\nUNEXPECTED DIFFERENCES (excluding window and img, which are meant to differ):")
if diff:
    for k, (old, new) in diff.items():
        print(f"  {k}: original={old!r} vs. new={new!r}")
    print("\n*** STOP and investigate before proceeding - an unexplained field beyond "
          "`window`/`img` is a real confound, not something to ignore. ***")
else:
    print("  none - every field matches except window and img, as expected.")
print(f"\nwindow: original={current_meta.get('window', '<absent>')!r} vs. new={fp['window']!r}")
print(f"img: original={current_meta.get('img', '<absent>')!r} vs. new={fp['img']!r}")''')

md(r"""## Step 1: probe, explicit acceptance band (spec section 2.1,
first-review finding I6)

`13v1`'s own real 224px result (7.0 slot-series/s) sat inside the
established **5-10/s** band (the raw 9.0/s documented baseline is
itself ~44% optimistic vs. a real sustained rate, so it is not the
right comparand on its own). At 336px only the resize step grows -
decode cost (DICOM read + `pixel_array`) is unchanged - so a modest
slowdown from `13v1`'s 7.0/s is expected, not a proportional 2.25x
drop. **Accept 5-10/s. Stop and investigate below 5/s, before running
the full build.**""")

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

print(f"\nmeasured rate: {PROBE_RATE:.1f} slot-series/s - accept 5-10/s as healthy "
      f"(13v1's own real 224px result was 7.0/s, inside this band; the same band is "
      f"reused here, adjusted for an expected modest 336px slowdown since decode cost "
      f"is dominated by DICOM read, not the resize step); stop and investigate below "
      f"5/s before running the full build (spec section 2.1, first-review finding I6).")
print(f"projected full-corpus decode: {PROBE_PROJECTION_H / 3600:.2f}h (train split) "
      "- PROBE_PROJECTION_H is returned in seconds by estimate_decode_time, not hours; "
      "the /3600 here matches the function's own internal log line above")
assert PROBE_RATE >= 5.0, (
    f"measured rate {PROBE_RATE:.1f}/s is below the 5/s floor - stop and investigate "
    "before spending real decode time on the full corpus"
)''')

md(r"""## Full build: 8 shards, 2 Kaggle sessions (spec section 2.2, first-review finding C1)

**Only run the cells below after reviewing Step 0's fingerprint diff (no
unexplained differences beyond `window`/`img`) and Step 1's probe rate
(>=5/s).** The 336px cache's total output (~25 GiB) exceeds both
Kaggle's output cap and this script's own `if total_gb > 15` guard
(below), so this build uses **8 shards, not 4** - `--shards 8`, each
~3.1 GiB - split across **2 Kaggle sessions of 4 shards each** (~70 min
CPU, ~12.5 GiB output, per session), using `main()`'s own existing
`shard="i,j,..."` subsetting, skip-existing-shard check, and
`cache_meta.json` merge (unedited from `13v1` - already supports this).
`RSNA_TIME_BUDGET` defaults to 8h in `CFG`; `main()` internally
overrides it to 11h for a real full-corpus run.""")

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
    return written''')

code(r'''# EDIT before each Kaggle session: 1 for shards 0-3, 2 for shards 4-7
# (spec section 2.2, first-review finding C1 - 8 shards across 2
# sessions, not 4 shards in 1, because the full 336px output (~25 GiB)
# exceeds both Kaggle's output cap and main()'s own 15 GiB
# per-invocation guard above).
SESSION = 1
SHARD_MAP = {1: "0,1,2,3", 2: "4,5,6,7"}
assert SESSION in SHARD_MAP, f"SESSION must be 1 or 2, got {SESSION}"

if SESSION == 2:
    # Copy session 1's uploaded output (a read-only Kaggle input Dataset)
    # into /kaggle/working/cache so main()'s own skip-existing-shard check
    # and cache_meta.json merge (with its config-clash guard) can see it -
    # EDIT this path once session 1's output is uploaded and attached.
    import shutil
    SESSION1_OUTPUT = Path(
        "/kaggle/input/datasets/alherma7/a3v3-336px-cache-session1/cache"
    )
    assert SESSION1_OUTPUT.exists(), (
        "SESSION1_OUTPUT not found - upload session 1's /kaggle/working/cache "
        "as a Kaggle Dataset and attach it before running session 2"
    )
    dest = Path("/kaggle/working/cache")
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in SESSION1_OUTPUT.iterdir():
        shutil.copy2(f, dest / f.name)
        copied += 1
    print(f"session 2: copied {copied} files from {SESSION1_OUTPUT} into {dest}")
else:
    print("session 1: building shards 0-3 fresh, nothing to copy in")''')

code(r'''RESULT = main(splits="test,train", shards=8, shard=SHARD_MAP[SESSION])
print("\nwritten:", json.dumps(RESULT, indent=2))
print(f"\nSession {SESSION} complete. " +
      ("Upload /kaggle/working/cache as a new Kaggle Dataset, then start a "
       "fresh session with SESSION=2 and attach it at SESSION1_OUTPUT above."
       if SESSION == 1 else
       "Both sessions complete - upload the final /kaggle/working/cache as "
       "the definitive A3 v3 cache Dataset for 16v1 to consume."))''')

md(r"""## Real output (fill in after running both Kaggle sessions)

Paste back, for each session: Step 0's fingerprint diff output (any
unexpected differences beyond `window`/`img`?), Step 1's measured rate
vs. the 5-10/s band, each session's `complete: N/N shards` line
(session 1 covers 4 of 8 expected train shards + any test shards run
alongside it; session 2 covers the remaining 4 plus the merged total),
and the final merged `cache_meta.json` contents (`img: 336`,
`window: "default"`, 8 train shards recorded under `splits`).""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

import json as _json
with open("notebooks/15v1_a3v3_336px_cache_build.ipynb", "w", encoding="utf-8") as f:
    _json.dump(nb, f, indent=1)
    f.write("\n")
print("wrote notebooks/15v1_a3v3_336px_cache_build.ipynb, cells:", len(cells))
```

Run this from the repo root so the relative output path resolves correctly.

- [ ] **Step 2: Run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote notebooks/15v1_a3v3_336px_cache_build.ipynb, cells: 21` (10 markdown + 11 code cells — one more code cell than `13v1`'s 9, since the "full build" section is now 3 code cells (`def main`, the `SESSION`/copy-in cell, the `RESULT = main(...)` call) instead of 1).

- [ ] **Step 3: Validate the notebook**

Run:
```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/15v1_a3v3_336px_cache_build.ipynb', as_version=4)
nbformat.validate(nb)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')
print('OK, cells:', len(nb.cells))
"
```
Expected: `OK, cells: 21`, no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/15v1_a3v3_336px_cache_build.ipynb
git commit -m "Build A3 v3 336px cache-rebuild notebook (wide window folded in, 8-shard/2-session split, spec section 2)"
```

---

## Task 2: 336px fold-0 pilot notebook (Kaggle GPU)

**Files:**
- Create: `notebooks/16v1_a2v4_336px_fold0_pilot.ipynb` (built via a Python builder script, directly modeled on `14v1_a2v3_window_only_fold0_pilot.ipynb`'s own 23-cell structure — full original source reproduced below with the changes layered in, not referenced externally, since every cell needs at least a chance of touching `img_size`)

**Interfaces:**
- Consumes: the new cache Dataset from Task 1 (path provided by the user once both sessions' output is uploaded — same "upload it yourself, edit the path" convention as every prior cache/checkpoint), `data/raw/_published_labels/llm_labels_v4_blend.csv` (the user's attached Kaggle Dataset, unchanged from A2 v2/A2 v3).
- Produces: two checkpoints (`/kaggle/working/a2_v4_fold0_best_epoch.pt`, `/kaggle/working/a2_v4_fold0_swa.pt`), a printed gate decision plus the new non-gating weak-cluster mean-delta readout.

Same non-agent-executable caveat as Task 1's notebook — needs Kaggle GPU + the new cache + the published labels. This task's "test" is structural validation only.

- [ ] **Step 1: Write the notebook builder script**

Create a scratch builder script (e.g. `build_16v1.py`). This notebook keeps `14v1`'s own 23-cell structure exactly (no cells added or removed) — only cell *contents* change, at the sites listed below.

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# --- Cell 0 (md): intro - REWRITTEN from 14v1 ---
md(r"""# 16v1 - A2 v4: 336px + wide-window fold-0 pilot

**Plan item:** tests whether A3 v3's 336px resolution rebuild (built by
`15v1`, with A3 v2's wide per-plane window folded in as the default, no
`src/` changes) improves the weak-finding cluster on A2 v2's own
architecture. Full design, evidence, and the gate this run is measured
against:
`docs/superpowers/specs/2026-09-01-a3v3-336px-resolution-design.md`
(approved after one Opus review pass).

Self-contained per this project's Kaggle constraint (no `import src`) -
identical to `09v1`/`14v1`'s own labels/folds/model/eval code (A2 v2's
winning architecture, unchanged) except `CACHE_DIR` points at `15v1`'s
new 336px cache, four literal `224`->`336` edits (cache-identity guard,
backbone construction, forward-shape check, dataset assert - spec
section 3.1, none of them a `src/` fix), shard names read from the new
cache's own `cache_meta.json` (8 shards, not 4), `MICRO_BATCH`/
`ACCUMULATE_STEPS` halved/doubled for the larger activation footprint
(re-verified empirically below, not assumed), and a new non-gating
weak-cluster mean-delta readout.

**Scope: fold 0 only. Resolution + window are the only changed
variables** vs. A2 v2 (`09v1_a2v2_multigroup_baseline.ipynb`) - same
hyperparameters otherwise, same labels, same fold split, same
`crop_mm=130`.""")

# --- Cell 1 (code): imports/setup - MODIFIED from 14v1 (added `import
# json`, CACHE_DIR points at the new cache, TRAIN_SHARDS read from
# cache_meta.json instead of a hardcoded 4-shard list) ---
code(r'''import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

_KAGGLE_RAW = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
ON_KAGGLE = _KAGGLE_RAW.exists()
if not ON_KAGGLE:
    raise RuntimeError(
        "This notebook needs the full DICOM tree + GPU + the A3 v3 cache "
        "attached - run on Kaggle, not locally."
    )
RAW_DIR = _KAGGLE_RAW
# EDIT this path once 15v1's cache Dataset (both sessions merged) is
# uploaded and attached - this is NOT the current (224px, narrow-window)
# cache or the parked A3 v2 (224px, wide-window) cache, it's the new
# 336px+wide-window one from 15v1_a3v3_336px_cache_build.ipynb.
CACHE_DIR = Path("/kaggle/input/datasets/alherma7/cache-a3v3-336px/cache")

# llm_labels_v4_blend.csv is NOT part of the official competition mount -
# it's the user's own separately-attached Kaggle Dataset, unchanged from
# A2 v2/A2 v3.
PUBLISHED_LABELS_PATH = Path("/kaggle/input/llm-labels-v4-blend/llm_labels_v4_blend.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
print("CACHE_DIR:", CACHE_DIR, "exists:", CACHE_DIR.exists())
print("PUBLISHED_LABELS_PATH:", PUBLISHED_LABELS_PATH, "exists:", PUBLISHED_LABELS_PATH.exists())

FINDINGS = [
    "acl_injury", "mcl_injury", "medial_meniscus_tear", "lateral_meniscus_tear",
    "oa_medial_compartment", "oa_lateral_compartment", "oa_patellofemoral_compartment",
    "effusion", "synovitis", "bakers_cyst", "bone_contusion", "fracture",
]
OFFICIAL_LABEL_COLUMNS = {
    "acl_injury": "ACL", "mcl_injury": "MCL",
    "medial_meniscus_tear": "Medial Meniscus", "lateral_meniscus_tear": "Lateral Meniscus",
    "oa_medial_compartment": "Medial OA", "oa_lateral_compartment": "Lateral OA",
    "oa_patellofemoral_compartment": "PF OA", "effusion": "Effusion",
    "synovitis": "Synovitis", "bakers_cyst": "Baker's",
    "bone_contusion": "Contusion", "fracture": "Fracture",
}
SLOT_NAMES = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
SLOT_CACHE_GROUP_SIZE = 3
SLOT_CACHE_N_GROUPS = 3
N_SLOTS = len(SLOT_NAMES) * SLOT_CACHE_N_GROUPS  # 18 pseudo-slots -- A2 v2's own architecture, unchanged
CV_FOLDS = 4

# Shard names read from the cache's own cache_meta.json rather than a
# hardcoded 4-shard convention (spec section 2.2's "downstream
# consequence" note) - this cache has 8 train shards, not 4.
_cache_meta_path = CACHE_DIR / "cache_meta.json"
if _cache_meta_path.exists():
    _cache_meta = json.loads(_cache_meta_path.read_text())
    TRAIN_SHARDS = sorted(k for k in _cache_meta.get("splits", {}) if k.startswith("train."))
else:
    TRAIN_SHARDS = []
print("TRAIN_SHARDS:", TRAIN_SHARDS)''')

# --- Cells 2-3 (md/code): Labels - UNCHANGED from 14v1 ---
md("## Labels: gold official values + A1a' published set for weak studies")

code(r'''def load_published_labels(path):
    published = pd.read_csv(path)
    label_cols = list(OFFICIAL_LABEL_COLUMNS.values())
    published = published.set_index("StudyInstanceUID")[label_cols]
    published.columns = list(OFFICIAL_LABEL_COLUMNS.keys())
    return published


def load_gold_labels(raw_dir):
    train = pd.read_csv(raw_dir / "train.csv")
    label_cols = list(OFFICIAL_LABEL_COLUMNS.values())
    gold_mask = train[label_cols].notna().all(axis=1)
    gold = train.loc[gold_mask, ["StudyInstanceUID"] + label_cols].set_index("StudyInstanceUID")
    gold.columns = list(OFFICIAL_LABEL_COLUMNS.keys())
    return gold


train_csv = pd.read_csv(RAW_DIR / "train.csv")
reports = train_csv.set_index("StudyInstanceUID")[["Report"]]
gold = load_gold_labels(RAW_DIR)
published = load_published_labels(PUBLISHED_LABELS_PATH)

missing = set(train_csv["StudyInstanceUID"]) - set(published.index)
print("train.csv studies missing from published labels:", len(missing))
assert len(missing) == 0

is_gold = reports.index.isin(gold.index)
label_table = published.reindex(reports.index)[FINDINGS].copy()
label_table.loc[gold.index, FINDINGS] = gold[FINDINGS]
label_table["is_gold"] = is_gold
print(label_table.shape, "gold rows:", label_table["is_gold"].sum())''')

# --- Cells 4-5 (md/code): Folds - UNCHANGED from 14v1 ---
md("""## Folds: report-template + scanner-fingerprint grouping (A0)

Identical logic to `05v2`/`06v2`/`09v1`/`14v1`'s fold-assignment cell -
`GroupKFold` has no shuffling/randomness, so recomputing from the same
inputs reproduces the exact same split. **Asserted, not just assumed**
- fold assignment is independent of the cache rebuild (it comes from
report/scanner grouping, not the pixel decode), but this notebook is a
fresh execution, so the assertion below is re-run rather than trusted
by proximity to `09v1`/`14v1`'s own passing results.""")

code(r'''def report_group_key(report_text):
    if not isinstance(report_text, str):
        normalized = ""
    else:
        t = unicodedata.normalize("NFKD", report_text.lower())
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        normalized = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


SCANNER_FINGERPRINT_TAGS = (
    "Manufacturer", "ManufacturerModelName", "InstitutionName",
    "DeviceSerialNumber", "MagneticFieldStrength", "StationName",
)


def build_scanner_fingerprints(raw_dir, split="train"):
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


def build_group_ids(*group_key_series):
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


t0 = time.time()
scanner_fp = build_scanner_fingerprints(RAW_DIR, split="train")
print(f"scanner fingerprints: {time.time() - t0:.1f}s for {len(scanner_fp)} studies")

group_keys = reports["Report"].apply(report_group_key)
group_ids = build_group_ids(group_keys, scanner_fp.reindex(reports.index))

gkf = GroupKFold(n_splits=CV_FOLDS)
fold = pd.Series(-1, index=reports.index, dtype=int)
for fold_idx, (_, val_idx) in enumerate(gkf.split(reports, groups=group_ids.to_numpy())):
    fold.iloc[val_idx] = fold_idx
label_table["fold"] = fold
print(label_table["fold"].value_counts().sort_index())

fold0_val = label_table[label_table["fold"] == 0]
print(f"\nfold 0 val: {len(fold0_val)} studies, {int(fold0_val['is_gold'].sum())} gold")
assert len(fold0_val) == 1307, (
    f"fold 0 val set size {len(fold0_val)} != recorded 1307 - fold split diverged, "
    "stop and investigate before trusting any comparison against 0.7956"
)
assert int(fold0_val["is_gold"].sum()) == 17, (
    f"fold 0 gold count {int(fold0_val['is_gold'].sum())} != recorded 17"
)
print("fold 0 matches the recorded split exactly - OK, safe to compare against 0.7956")''')

# --- Cells 6-7 (md/code): Reshape - UNCHANGED from 14v1 ---
md("""## Reshape: `select_group()` (A3, unchanged) + `expand_slot_groups()` (A2 v2, unchanged)

Both unchanged from `09v1`/`10v1`/`14v1` - this reshape logic operates
on tensor shape read from the cache array itself, no dependency on
resolution or which physical slices the pixels came from.""")

code(r'''def select_group(cache_slot_stack, group_index):
    if isinstance(group_index, int):
        group_index = [group_index]
    size = SLOT_CACHE_GROUP_SIZE
    groups = [cache_slot_stack[..., g * size:(g + 1) * size, :, :] for g in group_index]
    return np.concatenate(groups, axis=-3)


def expand_slot_groups(cache_slot_stack, slot_mask):
    '''cache_slot_stack: (n_slots, 9, H, W) - one study, all slots.
    slot_mask: (n_slots,). Returns (images, mask):
    images (n_slots * SLOT_CACHE_N_GROUPS, SLOT_CACHE_GROUP_SIZE, H, W),
    slot-major order (pseudo-slot index = s * SLOT_CACHE_N_GROUPS + g).
    mask (n_slots * SLOT_CACHE_N_GROUPS,), each real slot's bit repeated
    SLOT_CACHE_N_GROUPS times.'''
    n_slots = cache_slot_stack.shape[0]
    h, w = cache_slot_stack.shape[-2:]
    groups = [select_group(cache_slot_stack, g) for g in range(SLOT_CACHE_N_GROUPS)]
    stacked = np.stack(groups, axis=1)  # (n_slots, n_groups, 3, H, W)
    images = stacked.reshape(n_slots * SLOT_CACHE_N_GROUPS, SLOT_CACHE_GROUP_SIZE, h, w)
    mask = np.repeat(slot_mask, SLOT_CACHE_N_GROUPS)
    return images, mask


# Self-validate: each channel's pixels are set to that channel's global
# index, so the expected output is exact and checkable by hand.
_demo_stack = np.zeros((6, 9, 4, 4), dtype=np.uint8)
for _c in range(9):
    _demo_stack[:, _c] = _c
_demo_mask = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
_images, _mask = expand_slot_groups(_demo_stack, _demo_mask)
assert _images.shape == (18, 3, 4, 4)
assert _mask.shape == (18,)
for _s in range(6):
    for _g in range(3):
        assert np.array_equal(_images[_s * 3 + _g], select_group(_demo_stack[_s], _g))
assert np.array_equal(_mask, np.repeat(_demo_mask, 3))
print("expand_slot_groups matches select_group per (slot, group) pair and replicates the mask - OK")''')

# --- Cells 8-9 (md/code): Cache dataset - MODIFIED (shape assert 224->336, comment updated) ---
md("""## Cache dataset

Opens all `len(TRAIN_SHARDS)` (8) train shards as memmaps (never
materialises the full cache in RAM). Logic unchanged from `09v1`/
`10v1`/`14v1` - only the shard count differs (read from
`cache_meta.json`, see cell 1).""")

code(r'''class SlotCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, shards, labels_df, group_index=1, expand_groups=False, study_ids=None):
        if expand_groups and group_index != 1:
            raise ValueError(
                "expand_groups=True ignores group_index; pass group_index=1 (default) or omit "
                f"it, not group_index={group_index!r}"
            )
        self.group_index = group_index
        self.expand_groups = expand_groups
        caches, masks, all_study_ids, shard_of, local_idx = [], [], [], [], []
        for shard in shards:
            cache = np.load(cache_dir / f"{shard}_cache.npy", mmap_mode="r")
            mask = np.load(cache_dir / f"{shard}_mask.npy")
            studies = pd.read_csv(cache_dir / f"{shard}_studies.csv")
            caches.append(cache)
            masks.append(mask)
            all_study_ids.append(studies["StudyInstanceUID"].to_numpy())
            shard_of.append(np.full(len(studies), len(caches) - 1))
            local_idx.append(np.arange(len(studies)))

        self.caches = caches
        mask_all = np.concatenate(masks, axis=0).astype(np.float32)
        study_ids_all = np.concatenate(all_study_ids)
        shard_of_all = np.concatenate(shard_of)
        local_idx_all = np.concatenate(local_idx)

        if study_ids is not None:
            keep = np.isin(study_ids_all, np.asarray(list(study_ids)))
            mask_all, study_ids_all = mask_all[keep], study_ids_all[keep]
            shard_of_all, local_idx_all = shard_of_all[keep], local_idx_all[keep]

        self.mask = mask_all
        self.study_ids = study_ids_all
        self.shard_of = shard_of_all
        self.local_idx = local_idx_all

        aligned = labels_df.reindex(self.study_ids)[FINDINGS]
        if aligned.isna().any().any():
            missing = self.study_ids[aligned.isna().any(axis=1).to_numpy()]
            raise ValueError(f"{len(missing)} cache studies missing labels, e.g. {missing[:5]}")
        self.labels = aligned.to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, i):
        shard_idx, row = self.shard_of[i], self.local_idx[i]
        full = self.caches[shard_idx][row]  # (6, 9, 336, 336) uint8
        if self.expand_groups:
            selected, slot_mask_row = expand_slot_groups(full, self.mask[i])
        else:
            g = self.group_index
            selected = full[:, g * SLOT_CACHE_GROUP_SIZE:(g + 1) * SLOT_CACHE_GROUP_SIZE]
            slot_mask_row = self.mask[i]
        images = torch.from_numpy(np.ascontiguousarray(selected)).float() / 255.0
        mask = torch.from_numpy(slot_mask_row)
        label = torch.from_numpy(self.labels[i])
        return images, mask, label


sanity_ds = SlotCacheDataset(CACHE_DIR, TRAIN_SHARDS[:1], label_table, expand_groups=True)
images, mask, label = sanity_ds[0]
print("images:", images.shape, images.dtype, "mask:", mask.shape, "label:", label.shape)
# --- changed from 14v1: 224 -> 336 (spec section 3.1, first-review finding C2) ---
assert images.shape == (N_SLOTS, 3, 336, 336)
assert mask.shape == (N_SLOTS,)
assert not torch.isnan(images).any()
print("SlotCacheDataset(expand_groups=True) sanity check OK")

try:
    SlotCacheDataset(CACHE_DIR, TRAIN_SHARDS[:1], label_table, expand_groups=True, group_index=0)
    raise AssertionError("expand_groups=True with a non-default group_index should have raised")
except ValueError:
    print("expand_groups=True with group_index=0 correctly raises - OK")''')

# --- Cell 10 (code): cache-identity guard - MODIFIED (img 224->336) ---
code(r'''# Cache-identity guard (spec section 3). Checked against 15v1's own
# fingerprint, not 13v1's or the original cache's.
import json as _json

_meta_path = CACHE_DIR / "cache_meta.json"
print("cache_meta.json:", _meta_path, "exists:", _meta_path.exists())
assert _meta_path.exists(), "cache_meta.json not found - check CACHE_DIR points at the 15v1 output"
_meta = _json.loads(_meta_path.read_text())
print("cache fingerprint:", _json.dumps(_meta, indent=2, sort_keys=True))

# --- changed from 14v1: img 224 -> 336 (spec section 3.1, first-review
# finding C2 - this guard would otherwise hard-abort on the new cache) ---
_EXPECTED = {"window": "default", "crop_mm": 130.0, "img": 336, "group": 3, "n_group": 3}
_mismatch = {k: (v, _meta.get(k)) for k, v in _EXPECTED.items() if _meta.get(k) != v}
assert not _mismatch, (
    f"cache_meta.json doesn't match this notebook's expectations: {_mismatch} - "
    "CACHE_DIR may point at the wrong (or a stale) Dataset. This must be the "
    "336px+wide-window cache from 15v1, not the current 224px narrow-window one "
    "or the parked A3 v2 224px wide-window one."
)
_expected_slots = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
assert _meta.get("slots") == _expected_slots, f"slot order mismatch: {_meta.get('slots')}"
assert _meta.get("shards") == 8, f"expected an 8-shard cache, got shards={_meta.get('shards')}"
print("cache-identity guard passed - window='default' (wide), img=336, crop_mm/group/n_group/slots all match, 8 shards")''')

# --- Cell 11 (code): timm install - UNCHANGED from 14v1 ---
code(r'''import subprocess
subprocess.run(["pip", "install", "-q", "timm"], check=True)
import timm
print("timm:", timm.__version__)''')

# --- Cells 12-13 (md/code): Model - MODIFIED (img_size 224->336, forward-shape check 224->336) ---
md("""## Model: DINOv2-small backbone + masked_finding_attention

Unchanged from A2 v2/A2 v3 (`masked_finding_attention()`/
`SlotAttentionModel` are already generic over slot count and the
backbone's own `img_size` is already a `timm.create_model` argument -
zero reshape-code changes needed) except `img_size=336` (was 224 in
every prior notebook, spec section 3.1) and `n_slots` defaults to
`N_SLOTS` (18), same as `09v1`/`14v1`.""")

code(r'''def masked_finding_attention(embeddings, mask, query, head_weight, head_bias):
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
    def __init__(self, n_findings=len(FINDINGS), n_slots=N_SLOTS,
                 backbone_name="vit_small_patch14_dinov2.lvd142m", unfreeze_last=6):
        super().__init__()
        # --- changed from 14v1: img_size 224 -> 336 (spec section 3.1,
        # first-review finding C2). 336 is evenly divisible by DINOv2's
        # patch size 14 (24x24 patch grid), and 24x24 is a SMALLER
        # reduction from DINOv2's native 37x37 (518/14) grid than the
        # 16x16 (224px) already validated in production - the resample
        # this introduces is, if anything, less lossy than today's, not
        # more (spec section 3.2). Still the first time this project
        # runs DINOv2 at any resolution other than 224 - confirmed
        # working empirically by the pre-flight smoke test below, not
        # assumed safe by this argument alone. ---
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, img_size=336,
        )
        embed_dim = self.backbone.num_features
        for p in self.backbone.parameters():
            p.requires_grad = False
        for block in self.backbone.blocks[-unfreeze_last:]:
            for p in block.parameters():
                p.requires_grad = True

        self.query = nn.Parameter(torch.randn(n_findings, embed_dim) * (embed_dim ** -0.5))
        self.heads = nn.Linear(embed_dim, n_findings)
        self.embed_dim = embed_dim
        self.n_findings = n_findings
        self.n_slots = n_slots

    def forward(self, slot_images, slot_mask):
        B, S, C, H, W = slot_images.shape
        # --- changed from 14v1: (224, 224) -> (336, 336) ---
        if (S, C, H, W) != (self.n_slots, 3, 336, 336):
            raise ValueError(
                f"expected slot_images (*, {self.n_slots}, 3, 336, 336), got {tuple(slot_images.shape)}"
            )
        if tuple(slot_mask.shape) != (B, S):
            raise ValueError(f"expected slot_mask ({B}, {S}), got {tuple(slot_mask.shape)}")

        flat = slot_images.view(B * S, C, H, W)
        embeddings = self.backbone(flat).view(B, S, self.embed_dim)
        return masked_finding_attention(
            embeddings, slot_mask, self.query, self.heads.weight, self.heads.bias
        )


print("SlotAttentionModel defined (n_slots=18, img_size=336) - instantiating to confirm "
      "it loads real DINOv2 weights at the new resolution...")
_smoke_model = SlotAttentionModel()
n_trainable = sum(p.numel() for p in _smoke_model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in _smoke_model.parameters())
print(f"embed_dim={_smoke_model.embed_dim}, n_slots={_smoke_model.n_slots}, "
      f"trainable params={n_trainable:,} / {n_total:,}")
del _smoke_model''')

# --- Cells 14-15 (md/code): Evaluation helpers - UNCHANGED from 14v1 ---
md("""## Evaluation helpers

Unchanged hand-kept copies of `src/evaluate.py::macro_roc_auc`/`per_finding_roc_auc`.""")

code(r'''def per_finding_roc_auc(y_true, y_pred):
    scores = {}
    for c in y_true.columns:
        if y_true[c].nunique() < 2:
            scores[c] = float("nan")
        else:
            scores[c] = roc_auc_score(y_true[c], y_pred[c])
    return pd.Series(scores)


def macro_roc_auc(y_true, y_pred):
    per_finding = per_finding_roc_auc(y_true, y_pred)
    undefined = per_finding[per_finding.isna()]
    if len(undefined) > 0:
        print(f"  (macro_roc_auc: {len(undefined)} finding(s) undefined this fold "
              f"- {list(undefined.index)}, excluded from the mean, not treated as 0)")
    return float(per_finding.mean())''')

# --- Cells 16-17 (md/code): Pre-flight - MODIFIED (MICRO_BATCH candidate 8->4) ---
md("""## Pre-flight: overfit 8 real studies, check VRAM/host-RAM headroom

Standard wiring sanity check (same practice as A2 v1/A2 v2/A2 v3) plus
the VRAM/host-RAM read. At 336px, tokens per image rise 256->576 (a
2.25x input, and ViT attention cost is superlinear in token count, so
memory rises by more than 2.25x) - `09v1`'s own measured 6.73 GB peak
at `MICRO_BATCH=8`/224px projects close to ~14.7 GB at the same
micro-batch/336px, over a T4's 85% (~13.3 GB) threshold. **Expected
landing point: `MICRO_BATCH=4`/`ACCUMULATE_STEPS=8`** (spec section
3.3) - this cell uses `batch_size=4` for exactly that reason; confirm
or correct it against the real peak-VRAM number printed below before
trusting the full training run's own `MICRO_BATCH`.""")

code(r'''model = SlotAttentionModel().to(DEVICE)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                         lr=1e-3, weight_decay=0.02)

tiny_studies = label_table.index[:8]
tiny_ds = SlotCacheDataset(CACHE_DIR, TRAIN_SHARDS, label_table, expand_groups=True, study_ids=tiny_studies)
# --- changed from 14v1: batch_size 8 -> 4, matching this notebook's
# MICRO_BATCH hypothesis (spec section 3.3) ---
tiny_loader = torch.utils.data.DataLoader(tiny_ds, batch_size=4, shuffle=False)
images, mask, labels = next(iter(tiny_loader))
images, mask, labels = images.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)

if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
bytes_per_batch = images.element_size() * images.nelement()
print(f"images tensor: {tuple(images.shape)}, {bytes_per_batch / 1e6:.1f} MB for this "
      f"batch of {images.shape[0]} studies at 18 pseudo-slots/336px - host-RAM sanity check.")

eps = 1e-7
loss_floor = -(labels * torch.log(labels.clamp(eps, 1)) +
               (1 - labels) * torch.log((1 - labels).clamp(eps, 1))).mean().item()
print(f"loss floor for this batch (soft-label entropy): {loss_floor:.4f}")

print("pre-flight: overfitting 8 real studies...")
model.train()
for step in range(500):
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(model(images, mask), labels)
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"  step {step}: loss={loss.item():.4f}")
print(f"final pre-flight loss: {loss.item():.4f} (floor: {loss_floor:.4f})")
assert loss.item() < loss_floor + 0.02, (
    f"model failed to reach the soft-label loss floor ({loss_floor:.4f}) on 8 real "
    f"studies - stop and debug before the real run"
)
if DEVICE.type == "cuda":
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak VRAM during pre-flight (batch=4): {peak_gb:.2f} GB - use this to confirm "
          "whether MICRO_BATCH=4/ACCUMULATE_STEPS=8 (this notebook's stated hypothesis, "
          "spec section 3.3) fits an 85% (~13.3 GB on a T4) threshold, or needs further "
          "shrinking.")
print("pre-flight OK")
del model, opt''')

# --- Cell 18 (md): Full fold-0 training run heading - MODIFIED (mentions MICRO_BATCH=4) ---
md("""## Full fold-0 training run - dual checkpoint tracking (spec section 3)

Hyperparameters unchanged from A2 v2/A2 v3 except `MICRO_BATCH=4`/
`ACCUMULATE_STEPS=8` (spec section 3.3, halved/doubled from A2 v2's own
`MICRO_BATCH=8`/`ACCUMULATE_STEPS=4` for 336px's larger activation
footprint - effective `BATCH_SIZE=32` unchanged). Tracks **both** a
best-epoch checkpoint (matching every prior notebook's
`if gold_auc > best_gold_auc` selection - this is what gates against
the A2 v2 baseline, apples-to-apples) **and** an SWA-averaged checkpoint
(carried forward from `14v1`'s own first adoption of
[[feedback-checkpoint-selection-noise]]'s fix, averaging the last 3 of
12 epochs' weights instead of selecting one). Both mechanisms coexist in
the same training loop, unchanged from `14v1`.""")

# --- Cell 19 (code): training loop - MODIFIED (MICRO_BATCH 8->4) ---
code(r'''SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
# 09v1 (the A2 v2 baseline this pilot is gated against) was never
# seeded - this remains a real, accepted limitation, not silently fixed
# retroactively (spec section 4, restated from A3 v2's own M4).

FOLD = 0
train_idx = np.flatnonzero(label_table["fold"].to_numpy() != FOLD)
val_idx = np.flatnonzero(label_table["fold"].to_numpy() == FOLD)
val_labels = label_table.iloc[val_idx].reset_index()
val_is_gold = val_labels["is_gold"].to_numpy()
print(f"fold {FOLD}: {len(train_idx)} train / {len(val_idx)} val ({val_is_gold.sum()} gold in val)")

BATCH_SIZE = 32
# --- changed from 14v1: MICRO_BATCH 8 -> 4 (ACCUMULATE_STEPS auto-
# recomputed to 8, keeping BATCH_SIZE=32 unchanged) - spec section 3.3's
# stated hypothesis for 336px's larger activation footprint, already
# confirmed (or corrected) by the pre-flight cell above before this
# cell runs. ---
MICRO_BATCH = 4
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
# SWA_EPOCHS=3: averaging the last 3 of 12 epochs. Same OneCycleLR
# schedule-position reasoning as 14v1 (epochs 10/11/12 sit well past the
# schedule's peak, no warmup contamination) - unchanged.
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
        # .clone() is load-bearing: .cpu() is a no-op on a tensor already
        # on CPU, so without .clone() best_state would alias the live
        # model parameters and the NEXT epoch's training would silently
        # corrupt this saved copy.
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print("  new best (epoch-selected)")

    if epoch >= EPOCHS - SWA_EPOCHS:
        # Same .clone() hazard applies here, for the same reason.
        sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
        if swa_state is None:
            swa_state, swa_n = sd, 1
        else:
            swa_n += 1
            for k in swa_state:
                swa_state[k] += (sd[k] - swa_state[k]) / swa_n
        print(f"  SWA: accumulated epoch {epoch} (swa_n={swa_n})")

assert best_state is not None and swa_state is not None
torch.save(best_state, "/kaggle/working/a2_v4_fold0_best_epoch.pt")
torch.save(swa_state, "/kaggle/working/a2_v4_fold0_swa.pt")
print(f"\nbest-epoch gold macro-AUC (fold {FOLD}, selection-based): {best_gold_auc:.4f}")
print(f"SWA checkpoint: averaged {swa_n} epochs, no selection - gold AUC computed next cell")''')

# --- Cell 20 (md): Final report heading - MODIFIED (mentions the new I8 readout) ---
md("""## Final report: dual readout (best-epoch + SWA) + weak-cluster mean delta, spec section 4 gate

Evaluates **both** checkpoints on fold 0's gold subset, same as `14v1`.
The gate itself (spec section 4) runs on the **best-epoch** number,
apples-to-apples with how A2 v2's own 0.7956 baseline was produced.
`mcl_injury` is reported as context only, does not gate. **New this
notebook:** a non-gating mean delta across the 4 weak-cluster findings
(spec section 4, first-review finding I8) - roughly 2x more sensitive
than any single finding, reported alongside the gate, does not change
the decision rule. The decision rule is pre-agreed and unchanged from
`14v1` (spec section 4): macro passes and **at least one** of the two
meniscus findings clears the noise bar -> scale.""")

# --- Cell 21 (code): gate/report - MODIFIED (checkpoint filenames, new I8 block added) ---
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
      "- see feedback_checkpoint_selection_noise.md")

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

# mcl_injury: reported as context only, does NOT gate - unchanged reasoning from 14v1.
BASELINE_MCL = 0.800
mcl_delta = float(gold_per_finding_best["mcl_injury"]) - BASELINE_MCL
print(f"\nmcl_injury (context only, not gating): {gold_per_finding_best['mcl_injury']:.4f} "
      f"(baseline {BASELINE_MCL}, delta={mcl_delta:+.4f})")

# --- NEW (spec section 4, first-review finding I8): non-gating
# weak-cluster mean delta. oa_lateral_compartment's own A2v2 fold-0
# baseline is undefined (single class among the 17 gold), so it is
# excluded from the mean whenever either side is NaN, not silently
# treated as 0. Does NOT change the DECISION rule below. ---
WEAK_CLUSTER_BASELINES = {
    "mcl_injury": 0.800,
    "medial_meniscus_tear": 0.597,
    "lateral_meniscus_tear": 0.652,
    "oa_lateral_compartment": float("nan"),
}
_cluster_deltas = {}
for _f, _baseline in WEAK_CLUSTER_BASELINES.items():
    _cand = float(gold_per_finding_best[_f])
    if not (np.isnan(_baseline) or np.isnan(_cand)):
        _cluster_deltas[_f] = _cand - _baseline
if _cluster_deltas:
    print(f"\nnon-gating weak-cluster mean delta (spec section 4, I8) over "
          f"{list(_cluster_deltas)}: {np.mean(list(_cluster_deltas.values())):+.4f} "
          f"(individual deltas: {_cluster_deltas}) - reported for context only, does NOT "
          f"change the DECISION rule below")
else:
    print("\nweak-cluster mean delta: no findings had both a defined baseline and a "
          "defined candidate score this fold - nothing to report")

# Pre-agreed decision rule (spec section 4) - "at least one" meniscus
# finding, not "both", fixed in advance, unchanged from 14v1.
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

# --- Cell 22 (md): Real output placeholder - REWRITTEN ---
md("""## Real output (fill in after running on Kaggle)

Paste back: the fold-assignment assertion output, the cache-identity
guard output (confirming `img=336`, `shards=8`), the pre-flight
loss/peak-VRAM read (confirming or correcting `MICRO_BATCH=4`/
`ACCUMULATE_STEPS=8`), the per-epoch trajectory, the final dual-
checkpoint report (both `medial_meniscus_tear`/`lateral_meniscus_tear`
deltas), the new non-gating weak-cluster mean delta, and the resulting
`DECISION` line.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

import json as _json
with open("notebooks/16v1_a2v4_336px_fold0_pilot.ipynb", "w", encoding="utf-8") as f:
    _json.dump(nb, f, indent=1)
    f.write("\n")
print("wrote notebooks/16v1_a2v4_336px_fold0_pilot.ipynb, cells:", len(cells))
```

Run this from the repo root so the relative output path resolves correctly.

- [ ] **Step 2: Run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote notebooks/16v1_a2v4_336px_fold0_pilot.ipynb, cells: 23` (11 markdown + 12 code cells, same structure as `14v1` — count each `md()`/`code()` call above to verify).

- [ ] **Step 3: Validate the notebook**

Run:
```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/16v1_a2v4_336px_fold0_pilot.ipynb', as_version=4)
nbformat.validate(nb)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')
print('OK, cells:', len(nb.cells))
"
```
Expected: `OK, cells: 23`, no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/16v1_a2v4_336px_fold0_pilot.ipynb
git commit -m "Build A2 v4 336px fold-0 pilot notebook (wide window folded in, spec section 3-4)"
```

---

## Task 3: Update README.md and confirm full test suite unaffected

**Files:**
- Modify: `README.md` (Next steps checklist)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing consumed by later tasks — this is documentation only.

- [ ] **Step 1: Add the two new notebooks to `README.md`'s Next steps / History section**

Add an entry noting `15v1`/`16v1` are built and awaiting the user's Kaggle run (two Kaggle sessions for `15v1`, one GPU session for `16v1`), mirroring the phrasing already used for `13v1`/`14v1`'s own "built, not yet run" entries elsewhere in the file. Read the file's current "Next steps" section first (`Read README.md`) to match its existing style exactly before editing — no fixed line numbers are given here since prior tasks in this same session may have already changed the file.

- [ ] **Step 2: Run the full test suite to confirm it's unaffected**

Run: `python -m pytest`
Expected: same pass count as before this plan started (90 passed, 1 skipped, or whatever the current count is — this plan makes no `src/` changes, so the count must not change; if it does, stop and investigate before committing).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Note A3 v3/A2 v4 notebooks built, awaiting Kaggle run"
```

---

## Handoff (not a task — for the user, after all tasks above are complete)

On Kaggle: (1) run `15v1` with `SESSION=1`, review Step 0's fingerprint diff and Step 1's probe rate before letting the full decode run, upload `/kaggle/working/cache` as a Dataset; (2) start a fresh session, attach that Dataset, run `15v1` again with `SESSION=2` and `SESSION1_OUTPUT` pointed at it, upload the final merged `/kaggle/working/cache` as the definitive A3 v3 cache Dataset; (3) edit `CACHE_DIR` in `16v1`, run it with a GPU + the competition data + the published-labels Dataset + the new cache attached, and report back the pre-flight VRAM read, the per-epoch trajectory, the full dual-checkpoint report, the weak-cluster mean delta, and the `DECISION` line.
