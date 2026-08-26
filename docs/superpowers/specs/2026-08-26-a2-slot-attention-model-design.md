# A2 — slot-attention model, v1 design

**Status:** approved by user 2026-08-26, ready for implementation plan.
**Plan item:** A2 (see README.md / [the strategy artifact](https://claude.ai/code/artifact/1fec15df-75b4-43dc-8b6e-88f0c8c8147a)).
**Depends on:** A0 (fixed CV folds), A0b (slice ordering), A1a′ (published
labels), A3 (slot cache) — all done.

## 1. Overview

A2 is this project's first real trained model. Per study, each of the 6
named slots (`config.SLOT_NAMES`, from the A3 cache) that is present
passes through a shared, fine-tuned DINOv2-small backbone to produce one
embedding. Each of the 12 findings has its own learned attention query
that attends (masked softmax) over the present slots' embeddings — no
flat mean pooling, no compartment-aware attention yet. Output: 12 logits
per study.

**Explicitly out of scope for v1** (Tier B, per the 2026-08-25
reorientation plan): `pos_weight`, `is_gold` upweighting, EMA,
augmentation, ensembling, compartment-aware attention. Each is a
separate future change with its own gate against this v1's number.

**Scope decisions made during brainstorming (2026-08-26, user-approved):**
- Build only the 6-slot-attention architecture first. The competing
  "random bag of slices" design (Tom Aindow, 0.915 single-model,
  discussion 735304) is not built now — it stays a candidate if v1
  underperforms, not a parallel A/B.
- Compartment-aware attention (medial/lateral/PF-aware queries) is
  deferred. Confirmed with the user this requires retraining (it changes
  the attention head's parameters, and the backbone is fine-tuned
  jointly) — deferring is about isolating the baseline for easier
  debugging, not about saving compute, since a second training run costs
  roughly the same either way.
- DINOv2-small over EfficientNet-B0: multiple independent sources for
  this exact competition converge on DINOv2-family backbones (including
  the source of the A3 cache itself), and encoder *size* was already
  measured as a null lever (small→base, +0.0011 vs. a 0.0020 noise
  floor) — so small is the cheap, already-corroborated choice.
- Fixing `load_training_labels()` to use the A1a′-adopted published
  label set (instead of the still-wired-in regex labeler) is part of
  A2's scope, not a separate prerequisite — A2 cannot produce a
  meaningful number without it.

## 2. Data flow

### 2.1 Label source fix

New function `src/data.py::load_published_labels(raw_dir)`: reads
`data/raw/_published_labels/llm_labels_v4_blend.csv`, renames columns via
`config.OFFICIAL_LABEL_COLUMNS` (already confirmed to match exactly, no
ambiguous mapping — see `notebooks/03v2_published_label_validation.ipynb`,
A1a′). Covers all 4,407 studies (gold + weak) with continuous [0, 1]
scores. Raises if any `StudyInstanceUID` from `train.csv` is missing from
this file — a coverage guard, not a silent NaN.

`src/data.py::load_training_labels()` changes: instead of running
`src.labelers.label_reports()` over the weak (non-gold) reports, it starts
from `load_published_labels()` for all 4,407 rows, then overwrites the 58
gold rows with `load_gold_labels()`'s exact official 0/1 values. Fold
assignment logic (`build_group_ids` over report-template + scanner
fingerprint, `GroupKFold`) is unchanged — that part was already correct
per A0.

`src.labelers.label_reports()` stops being used in this pipeline (A1a′
already decided A1a/A1b are unnecessary) but stays in the codebase,
unused, not deleted. `src.labelers.report_group_key()` is still used —
that's for fold grouping, unrelated to label values.

### 2.2 Cache → model input

A `torch.utils.data.Dataset` (exact module location — `src/data.py` or a
new module — is an implementation-plan decision, not fixed here) that:
- On init: opens all 4 train shards' `cache.npy` as memory-maps (via
  `load_slot_cache_shard`, lazy — never materializes the ~12GB train
  cache in RAM), concatenates `mask`/`study_ids` (small, eager), builds a
  global-row → (shard, local-row) index.
- On `__getitem__(i)`: resolves shard/local index, reads that one
  study's `(6, 9, 224, 224)` slice from the mmap (touches only that
  study's bytes), applies `features.select_group(..., group_index=1)`
  (centre anchor, 3 channels — v1's fixed default; which group(s) to use
  is deliberately open per `select_group`'s own docstring, revisited only
  if v1 underperforms) to get `(6, 3, 224, 224)`, looks up the matching
  row from `load_training_labels()` by `StudyInstanceUID`, and returns
  `(slot_images, slot_mask, labels)`.

Whether a missing slot's pixel data is genuinely zero-filled in the real
cache needs confirming against real data in the validation notebook (see
§6) — not assumed here.

## 3. Model architecture (`src/model.py`)

- `build_backbone("vit_small_patch14_dinov2.lvd142m", pretrained=True)`
  via `timm` (exact tagged identifier confirmed 2026-08-26 — the bare
  name without the `.lvd142m` weights tag does not resolve), fine-tunable
  (not frozen) — matches the existing module docstring's stated
  principle and the sourced starting hyperparameters in §4. This
  checkpoint's native pretrained resolution is 518×518, not our cache's
  224×224 — `timm.create_model(..., img_size=224)` is expected to
  interpolate the position embeddings for ViT models, but this needs
  confirming for real in the validation notebook (§6), not assumed.
- `build_multiplane_model`: for a batch of `B` studies, flattens
  `(B, 6, 3, 224, 224)` → `(B*6, 3, 224, 224)`, runs the backbone once →
  `(B*6, D)` embeddings, reshapes to `(B, 6, D)` (`D` = the backbone's
  native embedding dim, e.g. 384 for DINOv2-small).
- Per-finding attention: a learned query matrix `Q` of shape `(12, D)`.
  For each finding, dot-product attention scores against the 6 slot
  embeddings, masked (`slot_mask`, absent slots set to `-inf` before
  softmax — never receive weight), softmax, weighted sum → one context
  vector `(B, 12, D)`.
- Classification head: one small linear layer **per finding** (`D → 1`,
  not shared across findings — each finding already has its own query,
  so its own final linear layer is a small, consistent addition) →
  `(B, 12)` logits.
- Loss: `BCEWithLogitsLoss` per finding against the continuous published-
  label targets (soft-label BCE is well-defined; no binarization needed
  for training).

### 3.1 Guards (user-requested, 2026-08-26)

- Shape assertions at the top of the forward pass: cache tensor
  `(B, 6, 3, 224, 224)`, mask `(B, 6)` — fail loudly and specifically on
  a mismatch, not downstream as a cryptic broadcast error.
- Assert every study has **at least 1 present slot** before the masked
  softmax — an all-`-inf` row produces NaN. Real corpus data (checked
  against `train_series.csv` locally) shows a minimum of 3 slots present
  per study, so this should never fire in practice; it exists as an
  explicit failure mode rather than a silent NaN if that assumption is
  ever wrong.
- Assert no NaN/Inf appears in the forward pass's output — catches
  numerical instability during fine-tuning early, not after a multi-hour
  run.

## 4. Training loop and hyperparameters (`src/train.py`)

**Scope: 1 fold only for this first real run**, not all 4 — same
practice as the A3 cache's own source notebook ("prove it works on one
fold first, then raise it"), justified here by both real Kaggle quota
cost (12h session limit, 30h/week) and this being the very first real
training run of a brand-new architecture.

**Starting hyperparameters**, sourced from
`data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`
(same competition, same backbone family — a reasoned starting point, not
re-derived blindly, cite in `RESOURCES.md`):
- Optimizer: `AdamW`, `weight_decay=0.02`.
- `lr_backbone=8e-6` (gentle fine-tune), `lr_head=1e-3`.
- Last 6 transformer blocks unfrozen (rest of the backbone stays frozen).
- `batch_studies=8` (one training example = one study's 6 slots).
- `epochs=12`, OneCycle LR schedule.

**Checkpointing:** best epoch by gold-only AUC on the validation fold
(the most trustworthy signal available) saved to `/kaggle/working/` —
enables a future warm-start (e.g. for a compartment-aware variant) without
retraining the backbone fully from its pretrained starting point.

## 5. Evaluation and gating (`src/evaluate.py` — reused, not changed)

**The `weak_macro` gauge needs adapting** to continuous published-label
targets: `macro_roc_auc` requires binary ground truth. For *measuring*
this gauge only (not for training targets), filter published-label rows
to a confident subset — score `≤ 0.15` (confident negative) or
`≥ 0.85` (confident positive), treated as pseudo-binary 0/1, dropping the
ambiguous middle. Same spirit as the old regex labeler's `abstain=0.5`
exclusion, applied as a measurement-time filter, not a training-time
transform. How many rows survive this filter on the real data is to be
confirmed in the validation notebook, not assumed.

**What this run is measured against:** there is no directly comparable
prior "candidate vs. baseline" — this is the first model trained under
the new pipeline (A0's fixed folds + A1a′'s labels + A3's cache). This
run:
1. Reports real gold-only and filtered-weak macro AUC.
2. Runs `per_label_gate()` as a diagnostic (broad effect across the 12
   findings, or riding on one/two) — informative, not a pass/fail gate
   yet.
3. Is informally compared against the historical real leaderboard number
   (0.596, Fase 5) as a sanity check, not a formal target.
4. **Becomes the baseline** that `gate_decision()` checks future A2
   changes (Tier B items, compartment-aware attention, scaling to 4
   folds, etc.) against.

## 6. Testing strategy

**Unit tests** (hermetic, synthetic fixtures, no Kaggle/GPU needed):
- `load_published_labels()`: correct column mapping on a synthetic CSV;
  raises on missing coverage.
- `load_training_labels()` (updated): gold rows keep exact official
  values; weak rows come from the published-label fixture, not the regex
  labeler.
- Attention-masking correctness: with small hand-controlled embeddings,
  changing a masked-out slot's values must not change the model's output
  at all — proves the mask actually excludes it, not just that the
  forward pass doesn't crash.
- The three forward-pass guards from §3.1, each triggered deliberately
  on a bad synthetic input.

**Kaggle validation notebook** (`notebooks/05v2_slot_attention_baseline.ipynb`,
needs GPU + the A3 cache attached, not runnable locally):
1. Load real published labels + real cache; confirm whether an absent
   slot's cached pixels are genuinely zero-filled.
2. Build the `Dataset`; sanity-check a few real batches (shapes, no
   NaN).
3. Overfit-a-tiny-batch smoke test (~8 real studies, many steps, loss
   should approach zero) — catches wiring bugs (label misalignment,
   frozen backbone not receiving gradients, a sign error in the loss)
   cheaply, before the expensive full run.
4. Full single-fold (fold 0) training run with §4's hyperparameters.
5. Report real gold/weak-gauge AUC, `per_label_gate()` output, and save
   the checkpoint.

Only after the user reviews this notebook's real output does the code
graduate to `src/train.py::run()` and `src/model.py`'s functions lose
their `NotImplementedError` bodies for real, per this project's
notebook-before-src convention.

## 7. Open items carried forward, not blocking v1

- Which `select_group` group_index(es) to actually use — v1 fixes group
  1 (centre anchor); comparing against all 3 groups is a later, cheap
  (no re-preprocessing needed) experiment.
- Compartment-aware attention (§1) — deferred, own future design/gate.
- Scaling from 1 fold to the full 4-fold CV — a follow-up step once v1's
  single-fold number looks sane, not part of this spec's scope.
- The competing random-bag-of-slices architecture (Tom Aindow's design)
  — stays a candidate only, not built now.
