# A2 v1: Slot-Attention Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the never-wired A1a′ label swap, then build and validate (on real Kaggle data) the first real trained model of this project: a DINOv2-small backbone with per-finding masked-attention pooling over the 6 A3 slots.

**Architecture:** Per study, each present slot's centre-anchor 3-channel image goes through a shared, fine-tuned DINOv2-small backbone; each of the 12 findings has its own learned query that attends (masked softmax) over the present slot embeddings, then its own small linear head. Trained on 1 fold only for this first pass.

**Tech Stack:** Python, pandas/numpy (existing), PyTorch + timm (new — CPU-only locally for hermetic tests, real GPU only on Kaggle), pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md`

## Global Constraints

- Architecture: DINOv2-small (`timm` id `vit_small_patch14_dinov2.lvd142m`) + per-finding masked-softmax attention over the 6 named slots (`config.SLOT_NAMES`) — no mean pooling, no compartment-aware attention in v1.
- `group_index=1` (centre anchor) is fixed for v1 — not a free parameter yet.
- Scope: train fold 0 only (of the 4 from A0), not all 4 folds.
- Explicitly out of scope for v1 (Tier B): `pos_weight`, `is_gold` upweighting, EMA, augmentation, ensembling, compartment-aware attention.
- Starting hyperparameters (cite in `RESOURCES.md` when this graduates): `AdamW`, `lr_backbone=8e-6`, `lr_head=1e-3`, `weight_decay=0.02`, last 6 transformer blocks unfrozen, `batch_studies=8`, `epochs=12`, OneCycle LR.
- **Kaggle notebooks in this project cannot `import src`** (confirmed with the user 2026-08-26) — any notebook that runs on Kaggle must be fully self-contained; code shared between `src/` and a notebook is duplicated by hand, not imported.
- Per the approved spec §6: `src/model.py`'s full architecture (`build_backbone`, `build_multiplane_model`) and `src/train.py::run()` stay `NotImplementedError` until the Kaggle notebook (Task 5) produces real, user-reviewed output — **this plan does not graduate them**. Only `src/data.py`'s label fix (Tasks 1–2) and the backbone-independent attention math (Task 3) graduate now, since both are fully verifiable without Kaggle.
- The `weak_macro` gauge needs a confident-subset filter (published-label score `≤0.15` or `≥0.85`, else dropped) since `macro_roc_auc` needs binary ground truth and the published labels are continuous.

---

## Task 1: `load_published_labels()`

**Files:**
- Modify: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `load_published_labels(raw_dir: Path) -> pd.DataFrame` — indexed by `StudyInstanceUID` (same row order as `raw_dir/train.csv`), columns = `config.FINDINGS`, continuous float values. Raises `ValueError` if any `train.csv` study is missing from the published CSV.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data.py` (near the other `load_*` tests, needs `load_published_labels` added to the existing `from src.data import (...)` block):

```python
def _write_published_labels_csv(raw_dir, rows):
    """rows: list of dicts with StudyInstanceUID + the 12 official
    label-column names (OFFICIAL_LABEL_COLUMNS values), continuous
    floats — matches the real llm_labels_v4_blend.csv layout."""
    published_dir = raw_dir / "_published_labels"
    published_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(published_dir / "llm_labels_v4_blend.csv", index=False)


def test_load_published_labels_maps_columns_and_indexes_by_study(tmp_path):
    raw_dir = _write_train_csv(tmp_path, [
        _weak_row("s1", "report 1"),
        _weak_row("s2", "report 2"),
    ])
    _write_published_labels_csv(raw_dir, [
        {"StudyInstanceUID": "s1", **{c: 0.9 for c in LABEL_COLS}},
        {"StudyInstanceUID": "s2", **{c: 0.1 for c in LABEL_COLS}},
    ])

    published = load_published_labels(raw_dir)

    assert list(published.index) == ["s1", "s2"]
    assert list(published.columns) == FINDINGS
    assert published.loc["s1", "acl_injury"] == 0.9
    assert published.loc["s2", "acl_injury"] == 0.1


def test_load_published_labels_raises_when_a_study_is_missing(tmp_path):
    raw_dir = _write_train_csv(tmp_path, [
        _weak_row("s1", "report 1"),
        _weak_row("s2", "report 2"),
    ])
    _write_published_labels_csv(raw_dir, [
        {"StudyInstanceUID": "s1", **{c: 0.5 for c in LABEL_COLS}},
    ])

    with pytest.raises(ValueError):
        load_published_labels(raw_dir)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data.py -k load_published_labels -v`
Expected: FAIL with `ImportError`/`AttributeError` — `load_published_labels` doesn't exist yet. (Also update the `from src.data import (...)` block at the top of `tests/test_data.py` to include `load_published_labels` before running — otherwise the whole file fails to collect.)

- [ ] **Step 3: Implement `load_published_labels()`**

Add to `src/data.py`, near `load_gold_labels`:

```python
def load_published_labels(raw_dir: Path) -> pd.DataFrame:
    """Load the A1a'-adopted published LLM label set (llm_labels_v4_blend.csv).

    Continuous [0, 1] scores per finding — NOT gold-standard 0/1, this is
    an LLM's graded read of the report text, scored at 0.8927 macro-AUC
    vs. our 58 real gold studies (notebooks/03v2_published_label_validation.ipynb,
    A1a'), decisively beating src.labelers.label_reports()'s 0.686. Covers
    all 4,407 studies (gold + weak) per that same validation. Row order
    matches raw_dir/train.csv's own StudyInstanceUID order.

    Raises ValueError if any train.csv study is missing from this file —
    a coverage guard, not a silent NaN, same reasoning as
    build_scanner_fingerprints' explicit-raise-over-silent-default
    choice.

    Graduated 2026-08-26 as part of A2's data-flow fix (see
    docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md,
    section 2.1) — A1a' validated this label set but never wired it into
    load_training_labels(), which kept calling the old regex labeler.
    """
    published = pd.read_csv(raw_dir / "_published_labels" / "llm_labels_v4_blend.csv")
    label_cols = list(config.OFFICIAL_LABEL_COLUMNS.values())
    published = published.set_index("StudyInstanceUID")[label_cols]
    published.columns = list(config.OFFICIAL_LABEL_COLUMNS.keys())

    train = pd.read_csv(raw_dir / "train.csv")
    missing = set(train["StudyInstanceUID"]) - set(published.index)
    if missing:
        raise ValueError(
            f"{len(missing)} train.csv studies missing from the published "
            f"label set, e.g. {sorted(missing)[:5]}"
        )

    return published.reindex(train["StudyInstanceUID"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data.py -k load_published_labels -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Real-data check (no Kaggle needed — both files are already local)**

Run:
```bash
python -c "
from pathlib import Path
from src.data import load_published_labels
labels = load_published_labels(Path('data/raw'))
print(labels.shape)
print(labels.head(3))
print('any NaN:', labels.isna().any().any())
"
```
Expected: shape `(4407, 12)`, no NaN, values in `[0, 1]`. Report this real output before committing — this is the "show real output" step for a change that's fully verifiable locally, per this project's notebook-validation convention (no new notebook needed here since `notebooks/03v2_published_label_validation.ipynb` already validated this exact file against gold).

- [ ] **Step 6: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "Add load_published_labels(), the A1a'-adopted label source A2 needs"
```

---

## Task 2: Wire `load_published_labels()` into `load_training_labels()`

**Files:**
- Modify: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `load_published_labels(raw_dir)` from Task 1.
- Produces: `load_training_labels(raw_dir, scanner_fingerprints, n_folds=None) -> pd.DataFrame` — same signature/columns as before (`config.FINDINGS` + `is_gold` + `fold`), but weak rows now come from `load_published_labels()` instead of `src.labelers.label_reports()`.

- [ ] **Step 1: Update the existing failing tests**

In `tests/test_data.py`, the current `test_load_training_labels_keeps_gold_labels_exact_and_labels_weak_rows` asserts weak rows get labeler-derived values — that's the exact behavior being removed. Replace it, and add published-label fixtures to the two fold-grouping tests that call `load_training_labels` (they'll otherwise hit Task 1's new coverage-guard `ValueError`, since their `raw_dir` won't have a `_published_labels/` file):

```python
def test_load_training_labels_uses_published_labels_for_weak_rows_and_official_values_for_gold(tmp_path):
    raw_dir = _write_train_csv(tmp_path, [
        _gold_row("gold1", "There is a complete tear of the ACL.", value=0.0),
        _weak_row("weak1", "Some report."),
        _weak_row("weak2", "Another report."),
    ])
    _write_published_labels_csv(raw_dir, [
        {"StudyInstanceUID": "gold1", **{c: 1.0 for c in LABEL_COLS}},  # must be ignored for gold1
        {"StudyInstanceUID": "weak1", **{c: 0.7 for c in LABEL_COLS}},
        {"StudyInstanceUID": "weak2", **{c: 0.2 for c in LABEL_COLS}},
    ])
    no_scanner_data = pd.Series([None, None, None], index=["gold1", "weak1", "weak2"])

    combined = load_training_labels(raw_dir, no_scanner_data, n_folds=2)

    assert combined.loc["gold1", "acl_injury"] == 0.0
    assert combined.loc["weak1", "acl_injury"] == 0.7
    assert combined.loc["weak2", "acl_injury"] == 0.2
    assert bool(combined.loc["gold1", "is_gold"]) is True
    assert bool(combined.loc["weak1", "is_gold"]) is False
```

Delete `test_load_training_labels_keeps_gold_labels_exact_and_labels_weak_rows` (superseded by the test above).

In `test_load_training_labels_never_splits_a_shared_report_template_across_folds`, insert this line right after the `_write_train_csv(...)` call (before `no_scanner_data = ...`) — the exact label values don't matter here, only that every study is covered, since this test checks fold grouping, not label values:

```python
    _write_published_labels_csv(raw_dir, [
        {"StudyInstanceUID": sid, **{c: 0.5 for c in LABEL_COLS}}
        for sid in ["gold1", "weak1", "weak2", "weak3"]
    ])
```

In `test_load_training_labels_never_splits_a_shared_scanner_fingerprint_across_folds`, insert the same line (same 4 study IDs — `gold1`, `weak1`, `weak2`, `weak3` — are used in that test too) right after its own `_write_train_csv(...)` call.

- [ ] **Step 2: Run tests to verify the new/updated ones fail**

Run: `python -m pytest tests/test_data.py -k load_training_labels -v`
Expected: FAIL — `load_training_labels` still calls the old labeler, so `weak1`'s `acl_injury` will be `1.0` (labeler-derived from "Some report." — actually check: labeler abstains on a report with no finding-relevant text, giving 0.5, not 0.7) rather than the expected `0.7` from the published fixture.

- [ ] **Step 3: Implement the change**

In `src/data.py`, replace `load_training_labels`'s body:

```python
def load_training_labels(raw_dir: Path, scanner_fingerprints: pd.Series,
                          n_folds: int = None) -> pd.DataFrame:
    """Build the training-label table: gold+weak, with CV folds.

    [... keep the existing docstring's fold-grouping explanation
    unchanged, but replace the weak-label paragraph with:]

    - The 4,349 weak studies get continuous [0, 1] scores from
      load_published_labels() (the A1a'-adopted set, 0.8927 macro-AUC
      vs. gold) — NOT src.labelers.label_reports() anymore (that scored
      0.686, decisively worse; A1a' decided this swap but it was never
      wired in until this change, 2026-08-26).

    [... keep the rest of the docstring as-is]
    """
    n_folds = n_folds or config.CV_FOLDS
    reports = load_reports(raw_dir)
    gold = load_gold_labels(raw_dir)
    published = load_published_labels(raw_dir)

    is_gold = reports.index.isin(gold.index)
    combined = published.reindex(reports.index)[config.FINDINGS].copy()
    combined.loc[gold.index, config.FINDINGS] = gold[config.FINDINGS]
    combined["is_gold"] = is_gold

    group_keys = reports["Report"].apply(report_group_key)
    group_ids = build_group_ids(group_keys, scanner_fingerprints.reindex(reports.index))
    gkf = GroupKFold(n_splits=n_folds)
    fold = pd.Series(-1, index=reports.index, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(gkf.split(reports, groups=group_ids.to_numpy())):
        fold.iloc[val_idx] = fold_idx
    combined["fold"] = fold

    return combined
```

Remove the now-unused `from src.labelers import label_reports, report_group_key` import's `label_reports` half — change to `from src.labelers import report_group_key` (still needed).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data.py -v`
Expected: all pass (no regressions in the untouched tests).

- [ ] **Step 5: Real-data check**

Run:
```bash
python -c "
from pathlib import Path
import pandas as pd
from src.data import load_training_labels
no_scanner = pd.Series(dtype=object)  # empty -> reindex gives all-None, isolates report-only grouping for this local check
labels = load_training_labels(Path('data/raw'), no_scanner, n_folds=4)
print(labels.shape)
print('gold rows:', labels['is_gold'].sum())
print(labels.loc[labels['is_gold'], 'acl_injury'].describe())
print(labels.loc[~labels['is_gold'], 'acl_injury'].describe())
"
```
Expected: `(4407, 14)` shape (12 findings + `is_gold` + `fold`), 58 gold rows, gold `acl_injury` values are exactly 0/1 (official), weak `acl_injury` values are continuous (not the old 0.0/0.5/1.0 three-point scale). Report this real output before committing.

- [ ] **Step 6: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "Wire the A1a' published label set into load_training_labels()"
```

---

## Task 3: `masked_finding_attention()` — the pooling math, backbone-independent

**Files:**
- Create: `src/model.py` (modify — add alongside the existing stubs)
- Test: `tests/test_model.py` (new file)
- Modify: `requirements.txt` (add `torch`, CPU-only is fine locally — no GPU/timm download needed for this task)

**Interfaces:**
- Produces: `masked_finding_attention(embeddings: torch.Tensor, mask: torch.Tensor, query: torch.Tensor, head_weight: torch.Tensor, head_bias: torch.Tensor) -> torch.Tensor` — shapes `embeddings (B,S,D)`, `mask (B,S)`, `query (O,D)`, `head_weight (O,D)`, `head_bias (O,)` → returns `(B,O)` logits. Raises `ValueError` if any row of `mask` is all-zero.

This is the one piece of A2's model deliberately graduated to `src/` now (not deferred to post-Kaggle like the rest of `model.py`) — it's pure tensor math, fully verifiable without a real backbone, real weights, or real data, matching how `build_group_ids`' union-find logic graduated on synthetic tests alone.

- [ ] **Step 1: Add `torch` to requirements.txt and install it**

Edit `requirements.txt`, changing:
```
# Add once real image-model training starts (A2/A3), not before:
#   torch, torchvision, monai, timm, snorkel
```
to:
```
# CPU-only here - real GPU training happens on Kaggle, this is just for
# hermetic unit tests of tensor-shape/masking logic (A2).
torch

# Add once the rest of A2 graduates (post-Kaggle validation):
#   torchvision, monai, timm, snorkel
```

Run: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
Expected: installs successfully. Verify with `python -c "import torch; print(torch.__version__)"`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_model.py`:

```python
"""Unit tests for src/model.py.

masked_finding_attention() is graduated (2026-08-26, A2) — pure tensor
math, independent of the real DINOv2 backbone, verified here on small
hand-controlled embeddings. build_backbone()/build_multiplane_model()
stay NotImplementedError until the Kaggle validation notebook
(notebooks/05v2_slot_attention_baseline.ipynb) produces real, reviewed
output — see docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md
section 6. No tests for those yet.
"""

import pytest
import torch

from src.model import masked_finding_attention


def test_masked_finding_attention_ignores_masked_slot_values():
    torch.manual_seed(0)
    embeddings = torch.randn(1, 3, 4)  # (B=1, S=3, D=4)
    query = torch.randn(2, 4)          # (O=2, D=4)
    head_weight = torch.randn(2, 4)
    head_bias = torch.randn(2)
    mask = torch.tensor([[1.0, 1.0, 0.0]])  # slot 2 absent

    baseline = masked_finding_attention(embeddings, mask, query, head_weight, head_bias)

    perturbed = embeddings.clone()
    perturbed[0, 2] = torch.randn(4) * 1000  # wildly different values in the masked slot
    perturbed_out = masked_finding_attention(perturbed, mask, query, head_weight, head_bias)

    assert torch.allclose(baseline, perturbed_out, atol=1e-5)


def test_masked_finding_attention_uses_present_slots_normally():
    torch.manual_seed(1)
    embeddings = torch.randn(2, 2, 4)  # (B=2, S=2, D=4)
    query = torch.randn(1, 4)          # (O=1, D=4)
    head_weight = torch.randn(1, 4)
    head_bias = torch.randn(1)
    mask = torch.ones(2, 2)

    out = masked_finding_attention(embeddings, mask, query, head_weight, head_bias)

    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_masked_finding_attention_raises_on_a_fully_masked_row():
    embeddings = torch.randn(1, 2, 4)
    query = torch.randn(1, 4)
    head_weight = torch.randn(1, 4)
    head_bias = torch.randn(1)
    mask = torch.zeros(1, 2)

    with pytest.raises(ValueError):
        masked_finding_attention(embeddings, mask, query, head_weight, head_bias)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'masked_finding_attention'`.

- [ ] **Step 4: Implement `masked_finding_attention()`**

Add to `src/model.py`, after the module docstring and before `build_backbone`:

```python
import torch


def masked_finding_attention(embeddings: "torch.Tensor", mask: "torch.Tensor",
                              query: "torch.Tensor", head_weight: "torch.Tensor",
                              head_bias: "torch.Tensor") -> "torch.Tensor":
    """Per-finding masked-softmax attention over slot embeddings, then a
    per-finding linear head — A2's pooling mechanism, independent of
    whatever backbone produced `embeddings`.

    embeddings: (B, S, D) - S slot embeddings per study.
    mask: (B, S) - 1.0 where a slot is present, 0.0 where absent. A
    masked-out slot gets attention weight 0 for every finding, verified
    in tests/test_model.py by perturbing a masked slot's values and
    confirming the output doesn't change.
    query: (O, D) - one learned attention query per finding.
    head_weight: (O, D), head_bias: (O,) - one linear D->1 head per
    finding, applied via the O rows of an nn.Linear(D, O)'s weight
    matrix (row o IS finding o's D->1 head) - avoids an O-way ModuleList
    of separate nn.Linear(D, 1) layers for the same result.

    Raises ValueError if any row of `mask` is all-zero: a masked softmax
    over an all -inf row is NaN by construction. Real corpus data has a
    measured minimum of 3 present slots per study (checked against
    train_series.csv, see project memory), so this should never fire in
    practice - it exists as an explicit failure mode instead of a silent
    NaN if that assumption is ever wrong.

    Source: docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md,
    section 3 (A2 v1 design, user-approved 2026-08-26).
    """
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_model.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/model.py tests/test_model.py requirements.txt
git commit -m "Graduate masked_finding_attention(), A2's backbone-independent pooling math"
```

---

## Task 4: Kaggle notebook, part 1 — labels, folds, cache dataset

**Files:**
- Create: `notebooks/05v2_slot_attention_baseline.ipynb` (built via a Python builder script, same technique as `notebooks/04v2_slot_cache_integration.ipynb`)

**Interfaces:**
- Consumes: `data/raw/train.csv`, `data/raw/train_series.csv`, the full DICOM tree under `data/raw/train_series/` (all Kaggle-only), `data/raw/_published_labels/llm_labels_v4_blend.csv`, the A3 cache (`CACHE_DIR`, Kaggle-only, the user's own attached copy).
- Produces (within the notebook's own namespace — nothing importable, per the Global Constraints): `label_table` (a DataFrame matching `load_training_labels()`'s shape/columns), `SlotCacheDataset` (a `torch.utils.data.Dataset` class), `TRAIN_SHARDS`, `FINDINGS`, `SLOT_NAMES`, `CACHE_DIR`, `RAW_DIR`, `DEVICE`. Task 5 continues appending cells to this same notebook and uses these names directly.

This notebook **cannot be executed by an agent** (needs the full DICOM tree, a GPU, and the user's own attached A3 cache dataset) — its "test" is that the file is valid, importable JSON (`nbformat.validate`) and every code cell compiles (`compile(cell.source, ..., "exec")`), exactly as done for `04v2`. Real execution and reporting happens in a follow-up conversation once the user runs it on Kaggle.

- [ ] **Step 1: Write the notebook builder script (part 1 of 2 — this task's cells)**

Create a scratch builder script (any path outside the repo, e.g. the session scratchpad) with this structure — this exact pattern was already used successfully for `04v2`:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# 05v2 - A2: slot-attention model, v1

**Plan item A2** (see [[project-rsna-phase-status]] /
docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md).
Self-contained per this project's Kaggle constraint (no `import src` —
confirmed 2026-08-26) - every function below is a hand-kept copy of the
matching `src/` function where one exists (`src/data.py::load_published_labels`,
`src/data.py::load_training_labels`, `src/model.py::masked_finding_attention`)
- keep them in sync manually if either side changes.

**v1 scope:** DINOv2-small + per-finding masked attention over the 6 A3
slots, centre anchor only (`group_index=1`), fold 0 only, no Tier B
items (no `pos_weight`, no `is_gold` upweighting, no EMA, no
augmentation, no compartment-aware attention).""")

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

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
GROUP_INDEX = 1  # centre anchor, per the approved A2 spec section 1
CV_FOLDS = 4
TRAIN_SHARDS = [f"train.s{i:02d}of04" for i in range(4)]""")

md("## Labels: gold official values + A1a' published set for weak studies")

code(r"""def load_published_labels(raw_dir):
    published = pd.read_csv(raw_dir / "_published_labels" / "llm_labels_v4_blend.csv")
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
published = load_published_labels(RAW_DIR)

missing = set(train_csv["StudyInstanceUID"]) - set(published.index)
print("train.csv studies missing from published labels:", len(missing))
assert len(missing) == 0

is_gold = reports.index.isin(gold.index)
label_table = published.reindex(reports.index)[FINDINGS].copy()
label_table.loc[gold.index, FINDINGS] = gold[FINDINGS]
label_table["is_gold"] = is_gold
print(label_table.shape, "gold rows:", label_table["is_gold"].sum())""")

md("""## Folds: report-template + scanner-fingerprint grouping (A0)

Real DICOM header scan across all 4,407 studies - header-only
(`stop_before_pixels=True`), cheap even at this count. Saves
`fold_assignments.csv` as a Kaggle output so a future notebook doesn't
need to redo this scan.""")

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

label_table.to_csv("/kaggle/working/fold_assignments.csv")
print("saved fold_assignments.csv")""")

md("""## Cache dataset

Opens all 4 train shards as memmaps (never materialises the ~12GB cache
in RAM). `group_index=1` selects the centre anchor's 3 slices, mirroring
`src/features.py::select_group`'s indexing exactly.""")

code(r"""class SlotCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, shards, labels_df, group_index=GROUP_INDEX):
        self.group_index = group_index
        caches, masks, study_ids, shard_of, local_idx = [], [], [], [], []
        for shard in shards:
            cache = np.load(cache_dir / f"{shard}_cache.npy", mmap_mode="r")
            mask = np.load(cache_dir / f"{shard}_mask.npy")
            studies = pd.read_csv(cache_dir / f"{shard}_studies.csv")
            caches.append(cache)
            masks.append(mask)
            study_ids.append(studies["StudyInstanceUID"].to_numpy())
            shard_of.append(np.full(len(studies), len(caches) - 1))
            local_idx.append(np.arange(len(studies)))

        self.caches = caches
        self.mask = np.concatenate(masks, axis=0).astype(np.float32)
        self.study_ids = np.concatenate(study_ids)
        self.shard_of = np.concatenate(shard_of)
        self.local_idx = np.concatenate(local_idx)

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
        g = self.group_index
        selected = full[:, g * SLOT_CACHE_GROUP_SIZE:(g + 1) * SLOT_CACHE_GROUP_SIZE]
        images = torch.from_numpy(np.ascontiguousarray(selected)).float() / 255.0
        mask = torch.from_numpy(self.mask[i])
        label = torch.from_numpy(self.labels[i])
        return images, mask, label


sanity_ds = SlotCacheDataset(CACHE_DIR, TRAIN_SHARDS[:1], label_table)
images, mask, label = sanity_ds[0]
print("images:", images.shape, images.dtype, "mask:", mask.shape, "label:", label.shape)
assert images.shape == (6, 3, 224, 224)
assert not torch.isnan(images).any()
print("SlotCacheDataset sanity check OK")""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

out_path = r"C:\Users\alher\Desktop\RSNA_Knee_Abnormality_Detection\notebooks\05v2_slot_attention_baseline.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out_path, "cells:", len(cells))
```

- [ ] **Step 2: Run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote .../notebooks/05v2_slot_attention_baseline.ipynb cells: 5`.

- [ ] **Step 3: Validate the notebook (nbformat + compile-check, not execution)**

Run:
```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/05v2_slot_attention_baseline.ipynb', as_version=4)
nbformat.validate(nb)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')
print('OK, cells:', len(nb.cells))
"
```
Expected: `OK, cells: 5`, no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/05v2_slot_attention_baseline.ipynb
git commit -m "Start A2 notebook: labels, real fold assignment, cache dataset (part 1/2)"
```

Keep the builder script around (do not discard it) — Task 5 appends more `code()`/`md()` calls to the same `cells` list before re-running it, so the notebook grows in place rather than being rewritten from scratch.

---

## Task 5: Kaggle notebook, part 2 — model, pre-flight test, training, report

**Files:**
- Modify: `notebooks/05v2_slot_attention_baseline.ipynb` (append cells via the same builder script from Task 4)
- Modify: `RESOURCES.md` (cite the DINOv2/timm source and the hyperparameter source)

**Interfaces:**
- Consumes: everything Task 4's cells define (`label_table`, `SlotCacheDataset`, `TRAIN_SHARDS`, `FINDINGS`, `CACHE_DIR`, `RAW_DIR`, `DEVICE`, `GROUP_INDEX`).

Same non-agent-executable caveat as Task 4 — this task's deliverable is a fully-built, syntax-valid, self-contained notebook ready to hand to the user.

- [ ] **Step 1: Add `timm` install + model cells to the builder script**

Append to the same builder script (before the `nb["cells"] = cells` line), starting with a pip-install cell (Kaggle images don't always ship `timm`):

```python
code(r"""import subprocess
subprocess.run(["pip", "install", "-q", "timm"], check=True)
import timm
print("timm:", timm.__version__)""")

md("""## Model: DINOv2-small backbone + masked_finding_attention

`masked_finding_attention` below is a hand-kept copy of
`src/model.py::masked_finding_attention` (Task 3, already unit-tested on
synthetic embeddings) — same function, duplicated here since this
notebook can't `import src`.""")

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
    def __init__(self, n_findings=len(FINDINGS), n_slots=len(SLOT_NAMES),
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


print("SlotAttentionModel defined - instantiating to confirm it loads real DINOv2 weights...")
_smoke_model = SlotAttentionModel()
n_trainable = sum(p.numel() for p in _smoke_model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in _smoke_model.parameters())
print(f"embed_dim={_smoke_model.embed_dim}, trainable params={n_trainable:,} / {n_total:,}")
del _smoke_model""")

md("""## Evaluation helpers

Hand-kept copies of `src/evaluate.py::macro_roc_auc`/`per_finding_roc_auc`
(unchanged), plus a new `confident_subset()` for the weak gauge (spec
section 5) - published labels are continuous, `roc_auc_score` needs
binary ground truth, so filter to confident rows (`<=0.15` or `>=0.85`)
before scoring that gauge.""")

code(r"""def macro_roc_auc(y_true, y_pred):
    return float(np.mean([roc_auc_score(y_true[c], y_pred[c]) for c in y_true.columns]))


def per_finding_roc_auc(y_true, y_pred):
    return pd.Series({c: roc_auc_score(y_true[c], y_pred[c]) for c in y_true.columns})""")

md("""## Pre-flight: overfit 8 real studies

Standard wiring sanity check before spending real training time - if the
model can't drive the loss near zero on 8 memorised examples, something
upstream (label alignment, a frozen parameter that should be trainable,
a sign error) is broken, and it's far cheaper to find that here than
after a multi-hour run.""")

code(r"""model = SlotAttentionModel().to(DEVICE)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                         lr=1e-3, weight_decay=0.02)

tiny_studies = label_table.index[:8]
tiny_ds = SlotCacheDataset(CACHE_DIR, TRAIN_SHARDS, label_table.loc[tiny_studies])
tiny_loader = torch.utils.data.DataLoader(tiny_ds, batch_size=8, shuffle=False)
images, mask, labels = next(iter(tiny_loader))
images, mask, labels = images.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)

print("pre-flight: overfitting 8 real studies...")
model.train()
for step in range(100):
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(model(images, mask), labels)
    loss.backward()
    opt.step()
    if step % 20 == 0:
        print(f"  step {step}: loss={loss.item():.4f}")
print(f"final pre-flight loss: {loss.item():.4f}")
assert loss.item() < 0.05, "model failed to overfit 8 real studies - stop and debug before the real run"
print("pre-flight OK")
del model, opt""")

md("""## Full single-fold (fold 0) training run

Hyperparameters sourced from `data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`
(same competition, same backbone family) - see RESOURCES.md.""")

code(r"""FOLD = 0
train_idx = np.flatnonzero(label_table["fold"].to_numpy() != FOLD)
val_idx = np.flatnonzero(label_table["fold"].to_numpy() == FOLD)
val_labels = label_table.iloc[val_idx].reset_index()
val_is_gold = val_labels["is_gold"].to_numpy()
print(f"fold {FOLD}: {len(train_idx)} train / {len(val_idx)} val ({val_is_gold.sum()} gold in val)")

full_ds = SlotCacheDataset(CACHE_DIR, TRAIN_SHARDS, label_table)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(full_ds, train_idx.tolist()), batch_size=8, shuffle=True, num_workers=2,
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(full_ds, val_idx.tolist()), batch_size=8, shuffle=False, num_workers=2,
)

model = SlotAttentionModel().to(DEVICE)
backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone")]
head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
opt = torch.optim.AdamW([
    {"params": backbone_params, "lr": 8e-6},
    {"params": head_params, "lr": 1e-3},
], weight_decay=0.02)

EPOCHS = 12
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=[8e-6, 1e-3], total_steps=EPOCHS * len(train_loader)
)

best_gold_auc = -1.0
for epoch in range(EPOCHS):
    model.train()
    t0 = time.time()
    for images, mask, labels in train_loader:
        images, mask, labels = images.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(images, mask), labels)
        loss.backward()
        opt.step()
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
        torch.save(model.state_dict(), "/kaggle/working/a2_v1_fold0_best.pt")
        print("  new best, checkpoint saved")

print(f"\nbest gold macro-AUC (fold {FOLD}): {best_gold_auc:.4f}")""")

md("## Final report: reload the best checkpoint, score both gauges")

code(r"""model.load_state_dict(torch.load("/kaggle/working/a2_v1_fold0_best.pt"))
model.eval()
probs = []
with torch.no_grad():
    for images, mask, _ in val_loader:
        images, mask = images.to(DEVICE), mask.to(DEVICE)
        probs.append(torch.sigmoid(model(images, mask)).cpu().numpy())
val_pred = pd.DataFrame(np.concatenate(probs), columns=FINDINGS)

gold_per_finding = per_finding_roc_auc(val_labels.loc[val_is_gold, FINDINGS], val_pred[val_is_gold])
print("gold per-finding AUC:\n", gold_per_finding)
print("gold macro AUC:", float(gold_per_finding.mean()))

weak_mask = ~val_is_gold
weak_true = val_labels.loc[weak_mask, FINDINGS].reset_index(drop=True)
weak_pred = val_pred[weak_mask].reset_index(drop=True)
confident = (weak_true <= 0.15) | (weak_true >= 0.85)
keep = confident.all(axis=1)
n_confident = int(keep.sum())
print(f"\nweak gauge: {n_confident} / {len(weak_true)} weak val studies confident enough")
if n_confident > 0:
    weak_true_bin = (weak_true[keep] >= 0.5).astype(float).reset_index(drop=True)
    weak_pred_conf = weak_pred[keep].reset_index(drop=True)
    print("weak gauge macro AUC (confident subset):", macro_roc_auc(weak_true_bin, weak_pred_conf))

print("\nhistorical real-leaderboard reference (Fase 5, informal sanity check only): 0.596")""")

md("""## Real output (fill in after running on Kaggle)

_Placeholder - paste the real printed output here once this notebook
has actually been run on Kaggle with the cache Dataset attached and GPU
enabled, same convention as `00v2`/`03v2`/`04v2`. This is where
`per_label_gate()` (already in `src/evaluate.py`, no change needed)
gets applied to *future* A2 candidates against this run's number - this
run has nothing to gate against yet, it establishes the baseline._""")
```

- [ ] **Step 2: Re-run the builder script**

Run: `python <builder_script_path>`
Expected: prints `wrote .../notebooks/05v2_slot_attention_baseline.ipynb cells: 12`.

- [ ] **Step 3: Validate the complete notebook**

Run the same nbformat-validate + compile-check command as Task 4 Step 3.
Expected: `OK, cells: 12`, no syntax errors across the whole file.

- [ ] **Step 4: Add the RESOURCES.md citation**

Add under `## Comparable projects`, after the existing
`stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache` entry:

```markdown
- **timm's `vit_small_patch14_dinov2.lvd142m`** (Hugging Face model card,
  huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
  Why: A2's backbone (`notebooks/05v2_slot_attention_baseline.ipynb`) -
  confirmed 2026-08-26 this is the exact tagged identifier timm needs
  (the bare name without `.lvd142m` does not resolve); native pretrained
  resolution is 518x518, interpolated down to our cache's 224x224 via
  `img_size=224`.
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/05v2_slot_attention_baseline.ipynb RESOURCES.md
git commit -m "Complete A2 notebook: model, pre-flight test, single-fold training, report (part 2/2)"
```

- [ ] **Step 6: Hand off to the user**

Tell the user: the notebook is ready at `notebooks/05v2_slot_attention_baseline.ipynb`. It needs, on Kaggle: the competition data attached, a GPU accelerator enabled, and the A3 cache Dataset attached at `/kaggle/input/datasets/alherma7/cache-stevenleehans-rsna/cache`. Ask them to run it top to bottom and report back the real printed output (fold sizes, pre-flight loss, per-epoch gold AUC, final gold/weak macro AUC). **Do not claim A2 works, or graduate anything further into `src/model.py`/`src/train.py`, until that real output comes back** — per the approved spec, this is a follow-up conversation, not part of this plan.
