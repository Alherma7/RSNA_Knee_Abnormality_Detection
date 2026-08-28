# A2 v2: Multi-Group Slot Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate (on real Kaggle data) a self-contained fold-0 pilot notebook that tests whether using all 3 A3 anchor groups (18 pseudo-slots) instead of centre-only (`group_index=1`, 6 slots) improves A2 v1's weakest-finding cluster, per the approved spec's fold-0 gate.

**Architecture:** Same DINOv2-small backbone + per-finding masked-softmax attention as A2 v1 (`masked_finding_attention`/`SlotAttentionModel`, unchanged, already generic over slot count). The only real change is upstream of the model: each study's slot stack is reshaped from `(6 slots, 1 group, 3ch)` to `(6 slots x 3 groups, 3ch)` = 18 pseudo-slots, each of the 3 groups of a slot fed to the backbone as an independent input, letting the model's existing per-finding attention learn which group matters per finding instead of that being pre-decided.

**Tech Stack:** Python, pandas/numpy, PyTorch + timm (Kaggle GPU only), pydicom, scikit-learn (`GroupKFold`, `roc_auc_score`), `nbformat` (notebook build/validate, local).

**Spec:** `docs/superpowers/specs/2026-08-28-a2v2-multigroup-slot-attention-design.md` (approved after two Opus review passes)

## Global Constraints

- **Scope: fold 0 only.** This plan does not train folds 1-3 — that only happens in a follow-up conversation if this run's spec-section-5.1 gate says to scale, and needs its own follow-up plan for the section-5.2 pooled gate.
- **No `src/` changes in this plan.** Per the approved spec section 6: `expand_slot_groups()`/`SlotCacheDataset`'s `expand_groups` flag stay notebook-local until the user reviews this notebook's real Kaggle output — same notebook-before-src discipline as A2 v1 and A3.
- **Kaggle notebooks in this project cannot `import src`** (confirmed 2026-08-26, reconfirmed for every notebook since) — this notebook is fully self-contained; every function shared with `src/`'s eventual future version is a hand-kept copy, not an import.
- Hyperparameters unchanged from A2 v1 except pseudo-slot count: `AdamW`, `lr_backbone=8e-6`, `lr_head=1e-3`, `weight_decay=0.02`, last 6 transformer blocks unfrozen, `epochs=12`, `OneCycleLR`, effective `batch_studies=32`. If VRAM forces a smaller real batch, use gradient accumulation (`MICRO_BATCH` x `ACCUMULATE_STEPS = BATCH_SIZE`) to hold the effective batch and the `OneCycleLR` step count constant — **the scheduler must step once per accumulation cycle** (`total_steps = EPOCHS * (len(train_loader) // ACCUMULATE_STEPS)`), not once per micro-batch, or the schedule silently changes anyway (spec section 4).
- Pseudo-slot images: `18` total (`6` slots x `3` groups), slot-major order — pseudo-slot index `s*3 + g` for slot `s` (0-5), group `g` (0-2). Each real slot's mask bit is replicated 3x, one per group.
- Fold reproducibility: **regenerate fold assignments (same code as `05v2`/`06v2`) and assert the result matches `05v2`'s recorded fold-0 split** (1,307 val studies, 17 gold) before trusting any comparison against 0.7689 — `06v2`'s own precedent shows regeneration is deterministic and safe as long as it's asserted, not silently trusted (spec section 6, first-review finding C7).
- Gate (spec section 5.1, this plan's only gate — section 5.2's pooled gate is out of scope here): gold macro (11-finding, `oa_lateral_compartment` excluded — it has no fold-0 baseline) regression check against `0.7689` with `gold_tol=0.03`; directional read (not a hard statistical gate) on `medial_meniscus_tear` (baseline `0.458`) and `lateral_meniscus_tear` (baseline `0.652`) only, `mcl_injury`/`oa_lateral_compartment` excluded from this fold-0-level read per the spec's stated reasons (noise, and 22% blank-`COR_T1` cap for `mcl_injury`; no baseline for `oa_lateral_compartment`).
- Checkpoint filename: `/kaggle/working/a2_v2_fold0_best.pt` (distinct from A2 v1's `a2_v1_fold0_best.pt`, same directory convention).
- Notebook filename: `notebooks/09v1_a2v2_multigroup_baseline.ipynb`.

---

## Task 1: Kaggle notebook, part 1 — labels, folds, reshape + cache dataset

**Files:**
- Create: `notebooks/09v1_a2v2_multigroup_baseline.ipynb` (built via a Python builder script, same technique as `notebooks/05v2_slot_attention_baseline.ipynb`/`08v1_meniscus_mcl_slot_group_check.ipynb`)

**Interfaces:**
- Consumes: `data/raw/train.csv`, `data/raw/train_series.csv`, the full DICOM tree under `data/raw/train_series/` (all Kaggle-only), `data/raw/_published_labels/llm_labels_v4_blend.csv` (as the user's attached Kaggle Dataset), the A3 cache (`CACHE_DIR`, Kaggle-only).
- Produces (within the notebook's own namespace — nothing importable, per the Global Constraints): `label_table` (DataFrame, matches A2 v1's shape/columns, has `fold`/`is_gold` columns), `select_group()`, `expand_slot_groups()`, `SlotCacheDataset` (with the new `expand_groups` parameter), `N_SLOTS` (`18`), `TRAIN_SHARDS`, `FINDINGS`, `SLOT_NAMES`, `CACHE_DIR`, `RAW_DIR`, `DEVICE`. Task 2 appends cells to this same notebook and uses these names directly.

This notebook **cannot be executed by an agent** (needs the full DICOM tree, a GPU, and the user's own attached A3 cache Dataset) — its "test" is that the file is valid, parseable JSON (`nbformat.validate`) and every code cell compiles (`compile(cell.source, ..., "exec")`), same as every prior Kaggle-only notebook in this project. Real execution and reporting happens in a follow-up conversation once the user runs it.

- [ ] **Step 1: Write the notebook builder script (part 1 of 2 — this task's cells)**

Create a scratch builder script (session scratchpad, e.g. `build_09v1.py`):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# 09v1 - A2 v2: multi-group slot attention (test hypothesis 2)

**Plan item:** follow-up to A2 v1, testing whether using all 3 A3 anchor
groups (18 pseudo-slots) instead of centre-only (`group_index=1`, 6
slots) improves the weak-finding cluster
(`mcl_injury`/`medial_meniscus_tear`/`lateral_meniscus_tear`/
`oa_lateral_compartment`). Full design, evidence, and the gate this run
is measured against:
docs/superpowers/specs/2026-08-28-a2v2-multigroup-slot-attention-design.md
(approved after two Opus review passes).

Self-contained per this project's Kaggle constraint (no `import src`) -
`select_group()`/`expand_slot_groups()`/`SlotCacheDataset` below are
hand-kept notebook-local versions; per the approved spec section 6, none
of this graduates to `src/` until the user reviews this notebook's real
output.

**Scope: fold 0 only**, `group_index`/pseudo-slot design is the only
changed variable vs. A2 v1
(`notebooks/05v2_slot_attention_baseline.ipynb`) - same
hyperparameters, same labels, same fold split.""")

code(r"""import hashlib
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
        "This notebook needs the full DICOM tree + GPU + the A3 cache "
        "attached - run on Kaggle, not locally."
    )
RAW_DIR = _KAGGLE_RAW
CACHE_DIR = Path("/kaggle/input/datasets/alherma7/cache-stevenleehans-rsna/cache")

# llm_labels_v4_blend.csv is NOT part of the official competition mount -
# it's the user's own separately-attached Kaggle Dataset. EDIT this path
# to match wherever it actually lands under /kaggle/input/ once attached
# (same convention as CACHE_DIR - check `!ls /kaggle/input` if unsure).
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
N_SLOTS = len(SLOT_NAMES) * SLOT_CACHE_N_GROUPS  # 18 pseudo-slots -- A2 v2's own change
CV_FOLDS = 4
TRAIN_SHARDS = [f"train.s{i:02d}of04" for i in range(4)]""")

md("## Labels: gold official values + A1a' published set for weak studies")

code(r"""def load_published_labels(path):
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
print(label_table.shape, "gold rows:", label_table["is_gold"].sum())""")

md(r"""## Folds: report-template + scanner-fingerprint grouping (A0)

Identical logic to `05v2`/`06v2`'s fold-assignment cell - `GroupKFold`
has no shuffling/randomness, so recomputing from the same inputs
reproduces the exact same split. **Asserted, not just assumed**, per
the approved spec section 6 (first-review finding: fold reproducibility
was previously unstated) - if this assertion ever fails, stop and
investigate before trusting any comparison against `05v2`'s 0.7689.""")

code(r"""def report_group_key(report_text):
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
    f"fold 0 val set size {len(fold0_val)} != 05v2's recorded 1307 - fold split diverged, "
    "stop and investigate before trusting any comparison against 0.7689"
)
assert int(fold0_val["is_gold"].sum()) == 17, (
    f"fold 0 gold count {int(fold0_val['is_gold'].sum())} != 05v2's recorded 17"
)
print("fold 0 matches 05v2's recorded split exactly - OK, safe to compare against 0.7689")""")

md(r"""## Reshape: `select_group()` (A3, unchanged) + `expand_slot_groups()` (A2 v2, new)

`select_group()` is an unchanged hand-kept copy of the graduated
`src/features.py` version. `expand_slot_groups()` is this experiment's
core data-layer change - per the approved spec section 2.1, reshapes
`(6 slots, 9 slices)` into `(18 pseudo-slots, 3 channels)`, slot-major
order, each real slot's mask bit replicated x3. Self-validated below
against a synthetic array before trusting it on real cache data - same
practice as `04v2`'s `select_group` demo cell.""")

code(r"""def select_group(cache_slot_stack, group_index):
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
    SLOT_CACHE_N_GROUPS times. Per approved spec section 2.1.'''
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
print("expand_slot_groups matches select_group per (slot, group) pair and replicates the mask - OK")""")

md(r"""## Cache dataset

Opens all 4 train shards as memmaps (never materialises the ~12GB cache
in RAM). New `expand_groups` parameter (default `False`, matching A2
v1's exact behaviour): when `True`, `__getitem__` calls
`expand_slot_groups()` instead of `select_group(group_index)`, and a
non-default `group_index` passed alongside `expand_groups=True` raises
(spec section 2.2 - avoids a silently-ignored argument).""")

code(r"""class SlotCacheDataset(torch.utils.data.Dataset):
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
        full = self.caches[shard_idx][row]  # (6, 9, 224, 224) uint8
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
assert images.shape == (N_SLOTS, 3, 224, 224)
assert mask.shape == (N_SLOTS,)
assert not torch.isnan(images).any()
print("SlotCacheDataset(expand_groups=True) sanity check OK")

try:
    SlotCacheDataset(CACHE_DIR, TRAIN_SHARDS[:1], label_table, expand_groups=True, group_index=0)
    raise AssertionError("expand_groups=True with a non-default group_index should have raised")
except ValueError:
    print("expand_groups=True with group_index=0 correctly raises - OK")""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

import json
with open("notebooks/09v1_a2v2_multigroup_baseline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
print("wrote notebooks/09v1_a2v2_multigroup_baseline.ipynb, cells:", len(cells))
```

Run this from the repo root so the relative output path resolves correctly.

- [ ] **Step 2: Run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote notebooks/09v1_a2v2_multigroup_baseline.ipynb, cells: 10`.

- [ ] **Step 3: Validate the notebook**

Run:
```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/09v1_a2v2_multigroup_baseline.ipynb', as_version=4)
nbformat.validate(nb)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')
print('OK, cells:', len(nb.cells))
"
```
Expected: `OK, cells: 10`, no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/09v1_a2v2_multigroup_baseline.ipynb
git commit -m "Start A2 v2 notebook: labels, real fold assignment, reshape + cache dataset (part 1/2)"
```

Keep the builder script around (do not discard it) — Task 2 appends more `code()`/`md()` calls to the same `cells` list before re-running it, so the notebook grows in place rather than being rewritten from scratch.

---

## Task 2: Kaggle notebook, part 2 — model, pre-flight test, training, gate report

**Files:**
- Modify: `notebooks/09v1_a2v2_multigroup_baseline.ipynb` (append cells via the same builder script from Task 1)

**Interfaces:**
- Consumes: everything Task 1's cells define (`label_table`, `select_group`, `expand_slot_groups`, `SlotCacheDataset`, `N_SLOTS`, `TRAIN_SHARDS`, `FINDINGS`, `CACHE_DIR`, `RAW_DIR`, `DEVICE`).

Same non-agent-executable caveat as Task 1 — this task's deliverable is a fully-built, syntax-valid, self-contained notebook ready to hand to the user.

- [ ] **Step 1: Append the model, pre-flight, training, and gate cells to the builder script**

Add these `md()`/`code()` calls **before** the `nb["cells"] = cells` line at the end of the same builder script from Task 1:

```python
code(r"""import subprocess
subprocess.run(["pip", "install", "-q", "timm"], check=True)
import timm
print("timm:", timm.__version__)""")

md(r"""## Model: DINOv2-small backbone + masked_finding_attention

Unchanged from A2 v1 (`masked_finding_attention()`/`SlotAttentionModel`
are already generic over slot count, per the approved spec section 3 -
zero `src/model.py` code changes needed, verified by both Opus review
passes) except `n_slots` now defaults to `N_SLOTS` (18) instead of 6.""")

code(r"""def masked_finding_attention(embeddings, mask, query, head_weight, head_bias):
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
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, img_size=224,
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
        if (S, C, H, W) != (self.n_slots, 3, 224, 224):
            raise ValueError(
                f"expected slot_images (*, {self.n_slots}, 3, 224, 224), got {tuple(slot_images.shape)}"
            )
        if tuple(slot_mask.shape) != (B, S):
            raise ValueError(f"expected slot_mask ({B}, {S}), got {tuple(slot_mask.shape)}")

        flat = slot_images.view(B * S, C, H, W)
        embeddings = self.backbone(flat).view(B, S, self.embed_dim)
        return masked_finding_attention(
            embeddings, slot_mask, self.query, self.heads.weight, self.heads.bias
        )


print("SlotAttentionModel defined (n_slots=18) - instantiating to confirm it loads real DINOv2 weights...")
_smoke_model = SlotAttentionModel()
n_trainable = sum(p.numel() for p in _smoke_model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in _smoke_model.parameters())
print(f"embed_dim={_smoke_model.embed_dim}, n_slots={_smoke_model.n_slots}, "
      f"trainable params={n_trainable:,} / {n_total:,}")
del _smoke_model""")

md(r"""## Evaluation helpers

Unchanged hand-kept copies of `src/evaluate.py::macro_roc_auc`/`per_finding_roc_auc`.""")

code(r"""def per_finding_roc_auc(y_true, y_pred):
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
    return float(per_finding.mean())""")

md(r"""## Pre-flight: overfit 8 real studies, check VRAM/host-RAM headroom

Standard wiring sanity check (same practice as A2 v1) plus the VRAM/
host-RAM read spec section 4 asks for, since 18 pseudo-slots costs ~3x
the memory of A2 v1's 6 slots per study.""")

code(r"""model = SlotAttentionModel().to(DEVICE)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                         lr=1e-3, weight_decay=0.02)

tiny_studies = label_table.index[:8]
tiny_ds = SlotCacheDataset(CACHE_DIR, TRAIN_SHARDS, label_table, expand_groups=True, study_ids=tiny_studies)
tiny_loader = torch.utils.data.DataLoader(tiny_ds, batch_size=8, shuffle=False)
images, mask, labels = next(iter(tiny_loader))
images, mask, labels = images.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)

if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
bytes_per_batch = images.element_size() * images.nelement()
print(f"images tensor: {tuple(images.shape)}, {bytes_per_batch / 1e6:.1f} MB for this "
      f"batch of {images.shape[0]} studies at 18 pseudo-slots (~3x A2 v1's 6-slot size) "
      "- host-RAM sanity check per spec section 4.")

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
    print(f"peak VRAM during pre-flight (batch=8): {peak_gb:.2f} GB - use this to judge "
          "whether BATCH_SIZE=32 in the next cell will fit as-is, or whether "
          "MICRO_BATCH/ACCUMULATE_STEPS need adjusting (gradient-accumulation fallback, "
          "spec section 4).")
print("pre-flight OK")
del model, opt""")

md(r"""## Full fold-0 training run

Hyperparameters unchanged from A2 v1 (sourced from
`data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`)
except gradient accumulation, added only if the pre-flight cell above
shows `BATCH_SIZE=32` won't fit at 18 pseudo-slots - **edit
`MICRO_BATCH` down (keeping `BATCH_SIZE` at 32) if so**, per spec
section 4. The scheduler is built on *optimizer* steps
(`len(train_loader) // ACCUMULATE_STEPS`), not micro-batches, so the
`OneCycleLR` schedule stays identical to A2 v1's regardless of the
micro-batch split.""")

code(r"""FOLD = 0
train_idx = np.flatnonzero(label_table["fold"].to_numpy() != FOLD)
val_idx = np.flatnonzero(label_table["fold"].to_numpy() == FOLD)
val_labels = label_table.iloc[val_idx].reset_index()
val_is_gold = val_labels["is_gold"].to_numpy()
print(f"fold {FOLD}: {len(train_idx)} train / {len(val_idx)} val ({val_is_gold.sum()} gold in val)")

# EDIT these two together based on the pre-flight cell's reported peak
# VRAM. BATCH_SIZE is the *effective* batch (kept at A2 v1's 32 to
# isolate group_index/pseudo-slot count as the only changed variable).
# MICRO_BATCH is what actually gets fed to the model per step. Try
# MICRO_BATCH=BATCH_SIZE (ACCUMULATE_STEPS=1) first; if that OOMs,
# halve MICRO_BATCH and double ACCUMULATE_STEPS until it fits.
# MICRO_BATCH must evenly divide BATCH_SIZE.
BATCH_SIZE = 32
MICRO_BATCH = 32
ACCUMULATE_STEPS = BATCH_SIZE // MICRO_BATCH
assert BATCH_SIZE % MICRO_BATCH == 0, "MICRO_BATCH must evenly divide BATCH_SIZE"
print(f"BATCH_SIZE={BATCH_SIZE}, MICRO_BATCH={MICRO_BATCH}, ACCUMULATE_STEPS={ACCUMULATE_STEPS}")

full_ds = SlotCacheDataset(CACHE_DIR, TRAIN_SHARDS, label_table, expand_groups=True)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(full_ds, train_idx.tolist()), batch_size=MICRO_BATCH,
    shuffle=True, num_workers=2, drop_last=True,
    # drop_last=True avoids a partial accumulation cycle at each epoch's
    # end, which would otherwise need special-casing below.
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
# total_steps counts *optimizer* steps, not micro-batches - with
# ACCUMULATE_STEPS>1 an optimizer step only happens every
# ACCUMULATE_STEPS micro-batches, so the scheduler must be built (and
# stepped below) on that basis, or the OneCycle LR schedule silently
# runs faster than A2 v1's (spec section 4).
steps_per_epoch = len(train_loader) // ACCUMULATE_STEPS
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=[8e-6, 1e-3], total_steps=EPOCHS * steps_per_epoch
)

best_gold_auc = -1.0
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
        torch.save(model.state_dict(), "/kaggle/working/a2_v2_fold0_best.pt")
        print("  new best, checkpoint saved")

print(f"\nbest gold macro-AUC (fold {FOLD}): {best_gold_auc:.4f}")""")

md(r"""## Final report: reload best checkpoint, compute the spec section 5.1 gate

Reports per-finding gold AUC, the macro-regression check, and the
directional read on `medial_meniscus_tear`/`lateral_meniscus_tear` -
exactly spec section 5.1, computed for real. `mcl_injury` and
`oa_lateral_compartment` are deliberately excluded from this fold-0-
level read (no fold-0 baseline for the latter; noise + the 22%
blank-`COR_T1` cap for the former) - both come back into the gate at
the pooled 4-fold stage (spec section 5.2), out of scope for this
notebook.""")

code(r"""model.load_state_dict(torch.load("/kaggle/working/a2_v2_fold0_best.pt"))
model.eval()
probs = []
with torch.no_grad():
    for images, mask, _ in val_loader:
        images, mask = images.to(DEVICE), mask.to(DEVICE)
        probs.append(torch.sigmoid(model(images, mask)).cpu().numpy())
val_pred = pd.DataFrame(np.concatenate(probs), columns=FINDINGS)

gold_per_finding = per_finding_roc_auc(val_labels.loc[val_is_gold, FINDINGS], val_pred[val_is_gold])
print("gold per-finding AUC:\n", gold_per_finding)

# ---- spec section 5.1: fold-0 pilot gate ----
BASELINE_FOLD0_MACRO_11 = 0.7689  # A2 v1 fold 0, 11-finding mean (05v2's real output)
BASELINE_MEDIAL_MENISCUS = 0.458
BASELINE_LATERAL_MENISCUS = 0.652
GOLD_TOL = 0.03

candidate_macro_11 = float(gold_per_finding.drop(index="oa_lateral_compartment").mean())
macro_delta = candidate_macro_11 - BASELINE_FOLD0_MACRO_11
macro_ok = macro_delta >= -GOLD_TOL
print(f"\ngold macro (11-finding, oa_lateral_compartment excluded per spec 5.1): {candidate_macro_11:.4f}")
print(f"vs. A2 v1 fold-0 baseline {BASELINE_FOLD0_MACRO_11}: delta={macro_delta:+.4f}, "
      f"macro_ok={macro_ok} (tol={GOLD_TOL})")

medial_delta = float(gold_per_finding["medial_meniscus_tear"]) - BASELINE_MEDIAL_MENISCUS
lateral_delta = float(gold_per_finding["lateral_meniscus_tear"]) - BASELINE_LATERAL_MENISCUS
print(f"\nmedial_meniscus_tear: {gold_per_finding['medial_meniscus_tear']:.4f} "
      f"(baseline {BASELINE_MEDIAL_MENISCUS}, delta={medial_delta:+.4f})")
print(f"lateral_meniscus_tear: {gold_per_finding['lateral_meniscus_tear']:.4f} "
      f"(baseline {BASELINE_LATERAL_MENISCUS}, delta={lateral_delta:+.4f})")

# ">= 0.10" is deliberately below either finding's raw fold-0
# Hanley-McNeil SE (0.144/0.141) - a delta between two correlated
# models trained on the same fold has a smaller SE than either model's
# own point estimate, and a false positive here only costs 3 more folds
# where spec section 5.2's real statistical gate lives, not a wrong
# graduation decision (spec section 5.1).
both_positive_and_real = medial_delta >= 0.10 and lateral_delta >= 0.10
print(f"\nboth findings move positive by >=0.10 (looks real, not single-fold noise): {both_positive_and_real}")

print()
if not macro_ok:
    print("DECISION: macro regressed beyond tolerance -> STOP, do not scale to 4 folds.")
elif both_positive_and_real:
    print("DECISION: macro OK and both findings moved meaningfully positive -> SCALE to the remaining 3 folds.")
else:
    print("DECISION: macro OK but the directional read is inconclusive (noise-sized or "
          "non-positive move) -> STOP, report as inconclusive, NOT disproved. Scaling "
          "further is a judgement call for the user, not automatic (spec section 5.1).")

print("\nfull gold per-finding AUC for the record:\n", gold_per_finding)
print("\nhistorical A2 v1 pooled reference (informal context only): 0.7512 macro, "
      "mcl_injury 0.6145, oa_lateral_compartment 0.6325, medial_meniscus_tear 0.6635, "
      "lateral_meniscus_tear 0.6584")""")

md(r"""## Real output (fill in after running on Kaggle)

Paste back: fold sizes/assertion result, pre-flight loss + peak VRAM,
whichever `MICRO_BATCH`/`ACCUMULATE_STEPS` split was actually used and
why, per-epoch val gold macro-AUC, the full gold per-finding table, and
the printed gate decision (macro_ok, both deltas, the final
DECISION line). Per the approved spec section 6, this is what the user
reviews before anything here graduates to `src/`.""")
```

- [ ] **Step 2: Re-run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote notebooks/09v1_a2v2_multigroup_baseline.ipynb, cells: 22`.

- [ ] **Step 3: Validate the complete notebook**

Run the same nbformat-validate + compile-check command as Task 1 Step 3.
Expected: `OK, cells: 22`, no syntax errors across the whole file.

- [ ] **Step 4: Commit**

```bash
git add notebooks/09v1_a2v2_multigroup_baseline.ipynb
git commit -m "Complete A2 v2 notebook: model, pre-flight, training, fold-0 gate report (part 2/2)"
```

- [ ] **Step 5: Hand off to the user**

Tell the user: the notebook is ready at
`notebooks/09v1_a2v2_multigroup_baseline.ipynb`. It needs, on Kaggle: the
competition data attached, a GPU accelerator enabled, the A3 cache
Dataset attached at
`/kaggle/input/datasets/alherma7/cache-stevenleehans-rsna/cache`, and
the published-labels Dataset attached (same as A2 v1 — check/update
`PUBLISHED_LABELS_PATH` to match wherever it mounts). Ask them to run it
top to bottom and report back the real printed output — especially the
pre-flight's peak-VRAM line (to confirm whether `MICRO_BATCH`/
`ACCUMULATE_STEPS` needed adjusting) and the final gate report's
`DECISION` line. **Do not claim A2 v2 works, scale to the remaining 3
folds, or graduate anything into `src/features.py`/`src/dataset.py`,
until that real output comes back** — per the approved spec, scaling and
graduation are both follow-up conversations, not part of this plan.
