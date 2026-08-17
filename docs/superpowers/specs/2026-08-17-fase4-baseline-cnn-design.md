# Fase 4 — Baseline 2.5D CNN (un plano, un backbone)

Status: executed on Kaggle, cell-by-cell (2026-08-17) — 3-fold macro
ROC-AUC 0.574 (std 0.052), beats the 0.5 baseline. Pending user review
of `notebooks/04_baseline_cnn.ipynb` before anything graduates to `src/`.
Date: 2026-08-17
Related: README.md (plan Fase 4), src/model.py, src/features.py, src/data.py,
notebooks/00_export_local_gold_subset.ipynb, notebooks/01_eda_dicom.ipynb,
notebooks/04_baseline_cnn.ipynb

## 12. Final result (2026-08-17)

Executed cell-by-cell on Kaggle against the real gold DICOM data (not
simulated) — see notebooks/04_baseline_cnn.ipynb for the full,
real-output record. Several design points changed from the sections
above during execution, based on real evidence found along the way
(each is called out inline in its section, not just here):

- **Gap redefined in mm, not slices** (Section 4.2) — inter-slice
  spacing varies 13.75x across the 58 selected series (0.4-5.5mm),
  discovered while investigating a legitimate 320-slice outlier series.
- **Added `center_crop_or_pad` to a fixed physical FOV** (Section 4.3,
  `CROP_MM=130.0`, pilkwang's value) — not in the original design;
  needed because `normalize_physical_scale` alone still leaves a
  different pixel shape per study.
- **Augmentation deferred, then tried, then not adopted** (Section 6) —
  the original draft assumed augmentation upfront; per the user's
  explicit call, it was only introduced after the first un-augmented
  run showed real overfitting (train macro AUC 0.698 vs. val 0.492 on
  fold 0). Once tried, it did not beat `weight_decay=1e-2` alone at any
  strength/epoch budget tested (see the regularization sweep table in
  the notebook) — logged as a negative result, not discarded
  permanently.
- **Winning configuration:** EfficientNet-B0 (sagittal only),
  `GAP_MM=4.0`, `TARGET_MM_PER_PIXEL=0.35`, `CROP_MM=130.0`,
  differential LR (backbone `1e-5`, head `1e-3`), dropout `0.5`,
  `weight_decay=1e-2`, early stopping (patience 8) on validation macro
  ROC-AUC, no augmentation.
- **Result:** fold-level macro AUC 0.595 / 0.515 / 0.612 — mean 0.574,
  std 0.052, all three individually above the 0.5 baseline.
- **Honest caveat:** every fold's train macro AUC climbs to 0.95-0.99
  (fold 2 reaches 0.999) — the model is close to memorizing its
  ~38-40-study training set in every fold. That it still generalizes to
  a mean of 0.574 is a real but fragile signal, not evidence of a
  robust model — expected at this sample size, and worth keeping in
  mind rather than overselling the win.
- **Not yet done:** graduating any of this into `src/data.py`,
  `src/features.py`, `src/model.py` — per this project's established
  rule, that happens only after the user reviews the actual notebook.

## 1. Objective

Train and validate a first CNN baseline that predicts the 12 findings
from knee MRI images, using only the 58 gold-labeled studies (no weak
labels yet — that's Fase 5). This is the first time `src/data.py`,
`src/features.py`, and `src/model.py` — all still `NotImplementedError`
stubs — get real implementations. The goal is a first honest macro
ROC-AUC number, not the final architecture: `src/model.py`'s existing
docstring already specifies a more sophisticated multi-plane,
per-finding-attention-pooled model, and that design is intentionally
deferred to Fase 6 (ensembling), once Fase 5 has added the weak-labeled
studies and there's enough data to support that model's capacity without
overfitting.

## 2. Context that shaped this design

- Only 58/4,407 studies have gold labels (README "Hecho clave"). Training
  a CNN on 58 examples is a small-data problem first, an architecture
  problem second — every design choice below is chosen to minimize
  overfitting risk over that constraint.
- `src/model.py`'s docstring (written before this phase) already commits
  to two decisions for the *eventual* model: per-finding attention
  pooling over plane/sequence slots (not mean pooling), and fine-tuning
  the encoder (not freezing it). Both are sound engineering choices, but
  multi-slot attention pooling is a much higher-capacity model than 58
  studies can train stably — hence deferring it, not disputing it.
- `src/features.py`'s docstring already commits to 2.5D triplet input
  (`[center-gap, center, center+gap]` slices stacked as 3 channels) and
  two required preprocessing steps: `normalize_physical_scale` (resize by
  physical mm, not fixed pixel count — PixelSpacing varies 5.14x across
  studies, measured in Fase 1) and `normalize_laterality` (flip so
  medial/lateral is consistent, using the DICOM `Laterality` tag,
  confirmed present in Fase 1).
- Measured directly against the real gold subset (`data/raw/train.csv` +
  `data/raw/train_series.csv`, exported via
  `notebooks/00_export_local_gold_subset.ipynb`), 2026-08-17: all 58 gold
  studies have at least one sagittal series (mean 2.34, max 5); 56/58
  have at least one `Fluid_Sensitive=True` sagittal series (48/58
  unambiguously — exactly one; 8/58 have more than one and need a
  tiebreak; 2/58 have none and need a fallback).
- MCL has only 9 positives out of 58 gold rows (the rarest finding) —
  this bounds how many CV folds are usable before per-fold validation
  sets risk having 0-1 positives, which breaks `macro_roc_auc` (raises on
  a single-class column) or makes the AUC estimate meaningless.

## 3. Decisions locked in during brainstorming

| Decision | Choice | Why |
|---|---|---|
| Training environment | Kaggle Notebook | GPU needed for CNN training in reasonable time; full dataset already mounted there; matches the project's established Kaggle-for-heavy-work strategy. |
| Model scope | Single plane, single backbone | 58 studies can't stably train the multi-slot attention model from `src/model.py`'s docstring; that's deferred to Fase 6. |
| CV folds | 3, **stratified** (not `config.py`'s global `CV_FOLDS=5`) | 5-fold leaves ~1-2 MCL positives per validation fold — too unstable/risks a degenerate fold. 3-fold gives ~19 studies and ~3 MCL positives per fold *if stratified*; a plain random `KFold` doesn't guarantee that (self-review finding, see Section 7). This is a Fase-4-specific override, documented here rather than changing the global default. |
| Plane | Sagittal | Primary clinical read plane for knee MRI; best for ACL/PCL/menisci; present in all 58 gold studies. |
| Backbone | EfficientNet-B0 (via `timm`, ImageNet-pretrained) | Few parameters, fast fine-tuning, lower overfitting risk than a ViT on 58 studies — one of the two options `src/model.py`'s docstring already names. Source for the architecture itself: Tan & Le, ICML 2019 (see Section 5), not the library books. |

## 4. Data pipeline

### 4.1 Series selection (one sagittal series per study)

Per gold study, pick exactly one sagittal series:
1. Prefer `Fluid_Sensitive == True` sagittal series (most informative for
   effusion, bone marrow edema/contusion, cartilage signal).
2. If more than one qualifies (8/58 studies), or none qualify and we fall
   back to all sagittal series (2/58 studies), break ties by **most
   slices** (a proxy for the most complete acquisition). If still tied
   (identical slice count), break by lexicographically smallest
   `SeriesInstanceUID` — arbitrary but deterministic, so the same study
   always resolves to the same series across reruns.

This is a deliberate simplification: a study's other ~1.3 average
sagittal series (and its coronal/axial series entirely) are not used in
Fase 4. That information is picked up again in Fase 6's multi-plane
ensemble, where there will also be more training data (post-Fase-5) to
support a higher-capacity model.

### 4.2 2.5D triplet construction

For the selected series: order slices by `SliceLocation` (not filename —
Fase 1), pick a center slice index (`sample_slice_indices`, uniform
sampling — for the Fase 4 baseline, one center triplet per series is
enough; revisit multi-triplet sampling only if validation shows the
model needs more spatial context), and stack
`[center-gap, center, center+gap]` as a 3-channel image
(`build_25d_triplet`).

**Revised 2026-08-17, during cell-by-cell notebook construction on
Kaggle:** `gap` is defined in **millimetres, not raw slice indices**.
Measured directly against the 58 gold studies' selected sagittal series
(`notebooks/04_baseline_cnn.ipynb`, cells checkpointed on Kaggle):
inter-slice spacing ranges from 0.4mm to 5.5mm (13.75x ratio) across the
58 selected series — a fixed index gap (e.g. `gap=2`) would span ~0.8mm
of real tissue in a 0.4mm-spacing series but ~11mm in a 5.5mm-spacing
one, making the triplet's physical meaning inconsistent across studies
in exactly the way `normalize_physical_scale` already prevents for the
in-plane (row/column) axes. Per study, `gap_slices =
max(1, round(gap_mm / spacing_mm))`, where `spacing_mm` comes from the
same fallback chain as Fase 1: `SpacingBetweenSlices` (52/58 selected
series had it), else `SliceThickness` (6/58), else the median of
consecutive `SliceLocation` deltas (not needed for any of the 58 — kept
as a safety net). The target `gap_mm` value itself is still a
hyperparameter to sweep empirically in the notebook against a small
range (e.g. 2-6mm) — no source commits to one value, so this stays
logged as an experiment, not assumed.

### 4.3 Preprocessing

- `normalize_physical_scale`: resize using PixelSpacing (mm/pixel), not
  a fixed pixel target, per `src/features.py`'s existing docstring and
  Fase 1's measured 5.14x spacing ratio.
- `normalize_laterality`: flip using the DICOM `Laterality` tag so
  medial/lateral falls on a consistent image side across all studies.
- **Added 2026-08-17, during cell-by-cell notebook construction:**
  `center_crop_or_pad(pixel_array, crop_px)` — after
  `normalize_physical_scale`, every study still has a *different* pixel
  shape (its original `Rows`/`Columns` scaled by a different factor), so
  batching needs one more step: crop to a fixed physical field of view,
  not a resize (a second resize would undo the physical-scale
  normalization just applied). `CROP_MM = 130.0`, source: pilkwang/
  rsna-knee-baseline-v1 (already cited in RESOURCES.md) — measured
  against their corpus to cover 99.6% of series' field of view while
  still containing the joint. Pads with zeros (black) if a study's
  normalized image is smaller than the crop in either dimension, rather
  than erroring. `crop_px = round(CROP_MM / target_mm_per_pixel)` —
  371px at the `TARGET_MM_PER_PIXEL = 0.35` starting point (Section 6
  lists this and `gap_mm` together as the hyperparameters to sweep).

## 5. Model

Single EfficientNet-B0 — source: Tan & Le, "EfficientNet: Rethinking
Model Scaling for CNNs" (ICML 2019), not the library books (verified
2026-08-17: neither Dive into Deep Learning nor Hands-On Machine
Learning teaches EfficientNet; D2L only name-drops it once as a Neural
Architecture Search example, citing this same paper — see
RESOURCES.md). ImageNet-pretrained, fully fine-tuned with **differential
learning rates**: a small LR on the pretrained backbone layers, a larger
LR on the new head — this is D2L §14.2's actual recommended mechanism
(verified against the book text), not "fine-tune everything at one
rate" as an earlier draft of this spec said. Shared trunk, one linear
head producing 12 logits (one per finding) — multi-task with a shared
representation, appropriate for 58 studies (separate per-finding models
would have even less data each). No attention pooling in this phase
(see Section 3).

## 6. Training regimen

- **Loss:** `BCEWithLogitsLoss` per finding, with `pos_weight` set per
  PyTorch's own documented formula — `pos_weight = n_negative / n_positive`
  for that finding's train-fold class balance (verified against
  docs.pytorch.org 2026-08-17; the docs' own worked example is a
  multi-label binary classification scenario, matching this project's
  12-finding setup exactly). Mitigates the 15.5%-60.3% prevalence range
  measured in Fase 3.
- **Augmentation — deferred until the baseline shows it's needed
  (revised 2026-08-17):** an earlier draft of this spec baked
  augmentation into the first training run. Self-review during
  cell-by-cell notebook construction: augmentation is a response to
  observed overfitting, not a default to assume upfront — with 58
  studies the first run's train/val gap (or lack of one) is itself the
  evidence for whether it's needed at all, so the first training loop
  runs **without any augmentation**, and augmentation only gets added as
  a follow-up experiment (with its own before/after macro-ROC-AUC
  comparison) if that first result shows overfitting. **The trap to
  remember when it is added:** `normalize_laterality` already
  canonicalizes which side of the image is medial vs. lateral, which 5
  of the 12 findings depend on. D2L §14.1 states plainly that "flipping
  the image left and right usually does not change the category of the
  object" — i.e. the book's own default assumption for when horizontal
  flip is safe is exactly the assumption that breaks here. A random
  horizontal flip augmentation would silently undo `normalize_laterality`
  and corrupt those 5 labels for the flipped copy — whenever augmentation
  is introduced, it must exclude horizontal flip (small rotations,
  intensity/contrast jitter, and small translations are fine, per D2L
  §14.1's own catalogue of standard, orientation-safe augmentations),
  with a code comment pointing at this reasoning so a future reader
  doesn't "fix" it by adding the flip back in.
- **Regularization:** dropout in the head (D2L §5.6; the book's own
  worked example uses 0.5 as an illustrative value for an MLP — a
  reasonable starting point here, not a value this problem has
  validated) + weight decay, used **together with early stopping on
  validation macro ROC-AUC**, not weight decay alone: D2L §5.5.4 notes
  that in deep networks, typical L2 weight-decay strength is often
  insufficient by itself to prevent interpolating the training data, and
  "the benefits if interpreted as regularization might only make sense
  in combination with the early stopping criterion" — given 58 studies,
  this project can't afford to skip that combination and rely on weight
  decay alone.

## 7. Validation

3-fold CV (Section 3) over the 58 gold studies, split with
**`MultilabelStratifiedKFold`** (`iterative-stratification` package,
implementing Sechidis, Tsoumakas & Vlahavas, "On the Stratification of
Multi-Label Data", ECML PKDD 2011 — verified 2026-08-17), not a plain
`KFold`. **Self-review finding:** an earlier draft of this spec justified
3 folds by "~3 MCL positives per fold", but that's only true in
expectation under a stratified split — a plain random `KFold` over 58
studies has no guarantee against a fold landing with 0-1 MCL positives
by chance, which is exactly the degenerate-fold risk Section 2 already
flags. Stratifying on all 12 finding columns at once (not just MCL) also
better balances the other 11 findings across folds. Per fold: train,
predict on the held-out fold, compute
`per_finding_roc_auc`/`macro_roc_auc` (`src/evaluate.py`, already
implemented and tested). Report the across-fold mean and spread (not
just a single number), given how small each fold's validation set is.
Gate: same as Fase 3 — must beat a constant-0.5 baseline, and the result
(win or not) gets logged in `README.md`/`RESOURCES.md` either way, per
the project's negative-result logging convention.

## 8. Testing strategy

Unit-testable (deterministic, `tests/test_features.py` and
`tests/test_data.py`, known inputs -> known outputs):
- `sample_slice_indices` (edge cases: series shorter than needed, gap
  larger than series length) — operates in slice-index units.
- `mm_to_slice_gap(gap_mm, spacing_mm) -> int` (new, added 2026-08-17):
  converts the physical `gap_mm` into the per-series integer gap
  `sample_slice_indices`/`build_25d_triplet` consume — `max(1,
  round(gap_mm / spacing_mm))`. Edge case: `spacing_mm` much larger than
  `gap_mm` still returns 1, never 0 (a triplet needs distinct
  neighbours).
- `build_25d_triplet` (correct channel stacking and ordering)
- `normalize_physical_scale` (a known pixel array + spacing -> expected
  output shape/scale)
- `normalize_laterality` (flips when `is_right_knee` requires it, no-op
  otherwise)
- The sagittal + `Fluid_Sensitive` series-selection rule (Section 4.1),
  as a pure function over `train_series.csv`-shaped input: exactly-one
  match, tie (multiple `Fluid_Sensitive`), and fallback (none) cases,
  including the `SeriesInstanceUID` tiebreak.
- The parameter-group split for differential learning rates (Section
  5): given a constructed model, the backbone's parameters land in the
  low-LR group and only the new head's parameters land in the high-LR
  group — self-review addition, this is deterministic plumbing (not
  training quality) and easy to get silently wrong (e.g. an extra layer
  added to the head later that isn't included in the high-LR group).
- Model output shape: given a batch of input triplets, the model
  produces exactly 12 logits per study — self-review addition, catches
  a wrong `num_classes`/head configuration before it reaches training.

Not unit-tested: model training/quality itself — that's what the
notebook's macro-ROC-AUC gate (Section 7) is for, per
structuring-ml-projects's existing rule (metric answers "did it improve",
tests answer "does the plumbing do what its docstring says").

## 9. Environment and dependencies

- `notebooks/04_baseline_cnn.ipynb` runs on Kaggle, self-contained (no
  `from src import ...`, same convention as notebooks 00-03, since
  Kaggle doesn't have this repo mounted — only the notebook itself).
- `requirements.txt` gets `torch`, `torchvision`, `timm`, and
  `iterative-stratification` added under the already-present "Fase 4 en
  adelante" comment (`monai`/`snorkel` stay commented out until an
  actual need arises — YAGNI).

## 10. Explicitly out of scope for Fase 4

- Multi-plane / multi-sequence attention pooling (Fase 6).
- Weak-labeled (non-gold) studies (Fase 5).
- `report_group_key`-based grouping (not relevant — Fase 4 doesn't touch
  report text at all).
- Ensembling / rank-blending multiple models (Fase 6,
  `src/evaluate.py::rank_blend`).
- Inference-time packaging / no-internet constraint (Fase 7).

## 11. Open risks (carried forward, not blocking)

- 58 studies (even split 3 ways) is a very small CV validation set; some
  per-finding AUCs will likely still be noisy point estimates, similar to
  Fase 3's labeler validation. This is expected and will be reported
  honestly, not smoothed over.
- The `gap_mm`, `TARGET_MM_PER_PIXEL`, and the exact regularization
  *strengths* (dropout rate, weight decay coefficient) are still not
  sourced from a specific citation, even after the 2026-08-17 library
  verification pass — the *techniques* themselves are now sourced (D2L
  §5.5-5.6 for dropout/weight decay/early stopping, PyTorch docs for
  `pos_weight`), but not the specific numeric values for this dataset.
  `CROP_MM=130.0` is the one exception with a real source (pilkwang's
  measured corpus, Section 4.3). The rest are hyperparameters to sweep
  in the notebook against the macro-ROC-AUC gate, logged as experiments
  per structuring-ml-projects's workflow (random search over a small
  range if more than one hyperparameter is tuned at once, not grid
  search).
- The 3-fold-vs-5-fold reasoning (Section 3) is this project's own
  statistical judgment about MCL's 9 positives, not something the
  library verification pass found stated in D2L's cross-validation
  section (§3.6.3 confirms K-fold generally suits scarce data, but
  doesn't discuss per-class-rarity fold-count tradeoffs) — flagged here
  so it isn't mistaken for a cited recommendation.
