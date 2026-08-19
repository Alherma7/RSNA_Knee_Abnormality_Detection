# Fase 5 (training) — gold+weak training test

Status: design, not yet executed.
Date: 2026-08-19
Related: README.md (Fase 5 infra, already graduated), src/data.py::load_training_labels,
src/labelers.py::label_reports/report_group_key, docs/superpowers/specs/2026-08-17-fase4-baseline-cnn-design.md,
notebooks/04_baseline_cnn.ipynb, notebooks/04b_gold_weak_groupkfold.ipynb

## 1. Objective

The Fase 4 corrected baseline (single sagittal plane, EfficientNet-B0,
58 gold studies only) reached macro ROC-AUC 0.532 (std 0.021, 3-fold),
with severe overfitting (train AUC 0.95-0.98) diagnosed as a training-set
*size* problem, not an architecture problem (see README's 2026-08-18 MIL
entry: tripling views per study via multi-plane pooling gave no clear
win — 0.530 vs 0.532 — while train AUC stayed just as high). Fase 5's
label/split infrastructure (`load_training_labels()`,
`report_group_key()`) is already graduated and tested, but nothing has
been trained on the 4,349 weak-labeled studies yet — this design covers
that missing piece: does adding them break the 0.53 ceiling?

Per this project's own graduation gate (`structuring-ml-projects`, step
6), the Fase 4 image pipeline (`normalize_laterality`,
`normalize_physical_scale`, `build_25d_triplet`, etc.) has only been
validated against the gold-only baseline — not against a run that
actually uses weak data — so it stays in the notebook until *this*
experiment shows a win, rather than graduating on the strength of the
gold-only result alone.

## 2. Decisions locked in during brainstorming (2026-08-19)

| Decision | Choice | Why |
|---|---|---|
| Weak label 0.5 (labeler abstention) | Soft target in `BCEWithLogitsLoss` | Simpler than per-column masking; lets the model express uncertainty directly rather than the loss silently dropping ~90%-silence-rate findings like `oa_lateral_compartment` from most weak rows. |
| Validation set per fold | Gold rows only (`fold==k & is_gold`) | Weak graded labels come from a labeler with only 0.686 AUC against gold (Fase 3) — validating against them would measure agreement with a noisy labeler, not truth. Keeps this run's macro-AUC directly comparable to Fase 4's 0.532 (also gold-only). |
| Plane(s) | Single sagittal plane only (same as Fase 4) | The 2026-08-18 MIL experiment already showed multi-plane pooling doesn't help at this data scale and confirmed the bottleneck is study *count*. Changing planes AND adding data at once would confound which change caused any AUC delta. |
| Scale | All 4,349 weak studies (no pilot subset) | User's explicit call — go straight to the full set rather than a 500-1000 pilot. |
| Preprocessing/training split | Two notebooks: preprocess to a Kaggle Dataset, then a separate training notebook mounts it | Kaggle sessions have a GPU time budget; decoding+normalizing 4,349 DICOM series is expensive enough that it should happen once, not on every training retry (hyperparameter change, bug fix). |
| Validation aggregation | Pooled out-of-fold (OOF): concatenate each fold's gold-only held-out predictions into one 58-row table, compute **one** `macro_roc_auc` over it | 5-fold `GroupKFold` (not stratified) leaves gold split 16/11/14/8/9 (Fase 5 infra); the smallest folds risk a single-class column for a rare finding (MCL, 9 positives total), which raises in `macro_roc_auc` by design. Pooling avoids that failure mode and gives one number, still comparable in magnitude to Fase 4's 0.532. |

## 3. Data pipeline

### 3.1 Series selection

Identical rule to Fase 4 Section 4.1, applied to all 4,407 studies (not
just the 58 gold): prefer `Fluid_Sensitive=True` sagittal series, tie-break
by most slices, then by lexicographically smallest `SeriesInstanceUID`.
This is a pure function of `train_series.csv` — no gold-specific
assumption — but has only been exercised against the 58 gold studies so
far; Notebook A's first job is confirming it resolves cleanly (exactly
one series) across the full 4,407, and logging how often each tie-break
branch fires at this larger scale.

### 3.2 Preprocessing (Notebook A — "05a_weak_dicom_preprocess.ipynb")

Reuses the corrected Fase 4 pipeline unchanged: order slices by
`SliceLocation`, apply the laterality list-order fix (sagittal:
reverse slice order by `Laterality`, not pixel flip — see Fase 4 audit
finding (1)), `normalize_physical_scale` (target 0.35mm/pixel),
`center_crop_or_pad` (`CROP_MM=130.0`), one center 2.5D triplet per
study (`GAP_MM=4.0`, same physical-mm gap logic as Fase 4 Section 4.2),
intensity normalization via `apply_voi_lut` with the percentile
fallback (Fase 4 audit fix (3)).

For each of the 4,407 studies: decode, preprocess, save as one `.npy`
file (float32, shape `(3, 371, 371)`) to `/kaggle/working/triplets/`.
Estimated size: ~1.65MB/study x 4,407 ≈ 7.3GB — fits comfortably in a
Kaggle Dataset. On completion, `Save Version` and publish as a private
Kaggle Dataset (e.g. `rsna-knee-weak-sagittal-triplets`).

Log any per-study preprocessing failures (missing sagittal series,
decode error) with the `StudyInstanceUID` rather than letting one bad
study kill the whole run — report a failure count/list at the end
rather than crashing partway through 4,407 studies.

### 3.3 Training (Notebook B — "05b_gold_weak_training.ipynb")

Mounts the Notebook-A Dataset as input. Loads `load_training_labels()`
for the label/fold table (already graduated in `src/data.py`). For each
of the 5 folds: train on all rows (gold+weak) not in that fold, predict
on that fold's **gold-only** rows, collect those predictions. After all
5 folds, concatenate the 5 held-out gold prediction sets (58 rows total)
and compute one `macro_roc_auc`/`per_finding_roc_auc` (`src/evaluate.py`,
already implemented/tested, reused as-is — no change needed for OOF
pooling, since it just receives whatever rows are passed in).

## 4. Model and training regimen

Unchanged from the Fase 4 winning configuration — reusing the same
config is deliberate (Section 2): EfficientNet-B0 (`timm`,
ImageNet-pretrained), differential LR (backbone 1e-5, head 1e-3),
dropout 0.5, weight_decay 1e-2, fixed epoch budget (8, no
checkpoint-selection-by-val — the Fase 4 audit's metric-leakage fix),
`torch.manual_seed(42)`, no augmentation (Fase 4's augmentation
experiment was a confirmed negative result, not revisited here since
this experiment already isolates a different variable).

**Loss, extended for soft targets:** `BCEWithLogitsLoss`, `pos_weight`
per finding computed as `n_positive/n_negative` where
`n_positive = target.sum()` and `n_negative = (1 - target).sum()` over
that finding's training-fold rows — a direct generalization of Fase 4's
hard-label formula to continuous {0.0, 0.5, 1.0} targets, not a new
technique.

**Batch composition:** gold and weak rows are mixed within the same
training fold with no reweighting between them (58 gold vs. ~3,479 weak
per fold, roughly 60:1) — noted as an open risk (Section 7), not a
decision to solve upfront; if the weak-label noise (labeler AUC 0.686)
turns out to swamp the gold signal, a follow-up experiment could
upweight gold rows or downweight low-confidence weak rows, but that's
premature before seeing whether the unweighted version helps or hurts
at all.

## 5. Validation

5-fold `GroupKFold` over all 4,407 studies, grouped by
`report_group_key()` — already computed by `load_training_labels()`, no
change needed. Per fold: train on non-fold rows (gold+weak), predict on
the fold's gold-only rows. Pool all 5 folds' held-out gold predictions
(58 rows) and compute one `macro_roc_auc` (Section 2 rationale).

**Gate:** must beat Fase 4's 0.532 (same gold studies, same metric,
different training set) to be considered a win — logged either way
(win or negative result) per `structuring-ml-projects`'s convention. If
it doesn't win, this stays in the notebook; nothing graduates to
`src/features.py`/`src/data.py`'s DICOM-loading stubs on the strength of
this result alone.

## 6. Testing strategy

No new `src/` functions are introduced by this experiment — it reuses
already-graduated/tested code (`load_training_labels`,
`report_group_key`, `macro_roc_auc`, `per_finding_roc_auc`) plus
notebook-only preprocessing/training code, per this project's
notebook-before-src rule. If this experiment wins and the image pipeline
graduates afterward, that graduation (and its tests) is a separate,
follow-up step — same as Fase 4 left graduation pending user review.

## 7. Explicitly out of scope

- Re-sweeping Fase 4's hyperparameters (`GAP_MM`, `TARGET_MM_PER_PIXEL`,
  `CROP_MM`, dropout, weight_decay) — reused as-is to isolate the
  more-data variable (Section 2).
- Multi-plane pooling (already tested 2026-08-18, no clear win; would
  also confound this experiment's variable).
- Gold/weak reweighting or confidence-based weak-row filtering — noted
  as a follow-up if the unweighted mix underperforms (Section 4).
- Graduating any pipeline code to `src/` — contingent on this
  experiment's result (Section 5 gate).

## 8. Open risks (carried forward, not blocking)

- ~60:1 weak:gold ratio per fold with no reweighting (Section 4) — if
  weak-label noise dominates, the effect could go either direction and
  won't be diagnosable from the macro-AUC number alone; per-finding AUC
  breakdown (`per_finding_roc_auc`) should be inspected alongside the
  macro number to see whether specific findings (e.g. the weaker-labeled
  `oa_lateral_compartment`, 0.553 labeler AUC per Fase 3) drag the result
  down disproportionately.
- Preprocessing 4,407 studies' DICOM (vs. 58 in Fase 4) is untested at
  this scale — series-selection edge cases (ties, fallbacks) and DICOM
  format edge cases (transfer syntax, corrupt files) that never appeared
  in the 58-study sample could appear here; Section 3.2's per-study
  failure logging is the mitigation, not a guarantee nothing needs a
  follow-up fix mid-run.
- Kaggle GPU session time budget for training on ~4,400 studies x 8
  epochs x 5 folds is unmeasured — Notebook B may need to be split
  further (e.g. one fold per session) if a single session's time limit
  is insufficient; not designed here since the actual per-epoch cost at
  this scale is unknown until Notebook A's preprocessing is done and a
  first training epoch is timed.
